"""Keep credentials out of log files and error messages.

Log files are long-lived and get shared when asking for help, and error
messages are shown on screen and copied into bug reports. Nothing written to
either may carry an API key, a password or a token.
"""
import re
from urllib.parse import urlsplit, urlunsplit

REDACTED = "[redacted]"

# Path segments an API endpoint is normally made of. A base URL is set by the
# user and some gateways carry an account ID or a token in the path, so a
# segment is only kept when it is one of these words.
_SAFE_PATH_SEGMENTS = frozenset({
    "api", "openai", "v1", "v1beta", "v1alpha", "v2", "v3", "chat", "completions", "responses",
    "models", "embeddings", "images", "generations", "edits", "audio", "files", "messages",
})
# What may follow a model name: Google-style "model:generateContent".
_MODEL_METHODS = frozenset({"generatecontent", "streamgeneratecontent", "counttokens", "embedcontent"})
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_KEY_SHAPED_RE = re.compile(r"^(sk-|AIza|ghp_|gho_|xox[abp]-|eyJ)")

_URL_IN_TEXT_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]{1,15}://[^\s\"'<>)\]}]+")
# Shapes of credentials that commonly show up in error responses.
_SCHEME_TOKEN_RE = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}")
_KEY_PREFIX_RES = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"\b(?:ghp|gho)_[A-Za-z0-9]{20,}"),
    re.compile(r"\bxox[abp]-[A-Za-z0-9-]{10,}"),
    # JSON Web Tokens: three base64url parts.
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
)
_NAMED_VALUE_RE = re.compile(
    r"(?i)\b(x-goog-api-key|x-api-key|api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|"
    r"password|passwd|authorization|cookie)\b(\s*[\"']?\s*[:=]\s*[\"']?)[^\s\"',;&}]{4,}"
)
# Known secrets shorter than this are not searched for: they would match ordinary text.
_MIN_KNOWN_SECRET_LENGTH = 6


def _safe_path(path: str) -> str:
    """Keep the endpoint words of a URL path and replace every other segment."""
    kept = []
    previous = ""
    for segment in path.split("/"):
        lowered = segment.lower()
        name, _, method = segment.partition(":")
        is_model = (
            previous == "models"
            and _MODEL_NAME_RE.match(name)
            and not _KEY_SHAPED_RE.match(name)
            and (not method or method.lower() in _MODEL_METHODS)
        )
        if segment == "" or lowered in _SAFE_PATH_SEGMENTS or is_model:
            kept.append(segment)
        else:
            kept.append(REDACTED)
        previous = lowered
    return "/".join(kept)


def safe_url_for_log(url) -> str:
    """Return the URL with everything that could be a credential removed.

    Dropped: the user name and password, the query string, the fragment, and
    every path segment that is not a known endpoint word. Scheme, host and
    port stay, which is enough to tell which service failed.
    """
    text = str(url or "")
    if not text:
        return ""
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return REDACTED
    if not parts.scheme or not parts.hostname:
        # Not a URL that can be taken apart safely.
        return REDACTED
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    netloc = f"{host}:{port}" if port else host
    suffix = "?" + REDACTED if parts.query or parts.fragment else ""
    return urlunsplit((parts.scheme, netloc, _safe_path(parts.path), "", "")) + suffix


def redact_secrets(text, known_secrets=()) -> str:
    """Return the text with known secrets, URLs and credential-shaped values made safe."""
    result = str(text or "")
    for secret in known_secrets or ():
        if isinstance(secret, str) and len(secret) >= _MIN_KNOWN_SECRET_LENGTH:
            result = result.replace(secret, REDACTED)
    result = _URL_IN_TEXT_RE.sub(lambda match: safe_url_for_log(match.group(0)), result)
    result = _SCHEME_TOKEN_RE.sub(lambda match: f"{match.group(1)} {REDACTED}", result)
    for pattern in _KEY_PREFIX_RES:
        result = pattern.sub(REDACTED, result)
    return _NAMED_VALUE_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", result)
