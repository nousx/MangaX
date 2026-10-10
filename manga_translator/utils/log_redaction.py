"""Keep credentials out of log files.

Log files are long-lived and get shared when asking for help, so anything
written to them must not carry an API key, a password or a token.
"""
import re
from urllib.parse import urlsplit, urlunsplit

REDACTED = "[redacted]"

# Shapes of credentials that commonly show up in error responses.
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|authorization)\b(\s*[\"']?\s*[:=]\s*[\"']?)[^\s\"',;&}]{4,}"),
)
# Known secrets shorter than this are not searched for: they would match ordinary text.
_MIN_KNOWN_SECRET_LENGTH = 6


def safe_url_for_log(url) -> str:
    """Return the URL with its credentials and query string removed.

    Base URLs are set by the user and may embed a password
    (https://user:pass@host) or a key in the query string. Scheme, host,
    port and path are enough to tell which endpoint failed.
    """
    text = str(url or "")
    try:
        parts = urlsplit(text)
    except ValueError:
        return REDACTED
    if not parts.scheme or not parts.netloc:
        # Not a URL we can take apart safely.
        return REDACTED if text else ""
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port else host
    suffix = "?" + REDACTED if parts.query else ""
    return urlunsplit((parts.scheme, netloc, parts.path, "", "")) + suffix


def redact_secrets(text, known_secrets=()) -> str:
    """Return the text with known secrets and credential-shaped values replaced."""
    result = str(text or "")
    for secret in known_secrets or ():
        if isinstance(secret, str) and len(secret) >= _MIN_KNOWN_SECRET_LENGTH:
            result = result.replace(secret, REDACTED)
    result = _SECRET_PATTERNS[0].sub(lambda match: f"{match.group(1)} {REDACTED}", result)
    result = _SECRET_PATTERNS[1].sub(REDACTED, result)
    result = _SECRET_PATTERNS[2].sub(REDACTED, result)
    return _SECRET_PATTERNS[3].sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", result)
