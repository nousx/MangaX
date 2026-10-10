"""Keep credentials out of log files and error messages.

Log files are long-lived and get shared when asking for help, and error
messages are shown on screen and copied into bug reports. Nothing written to
either may carry an API key, a password or a token.

Two layers use this module. Call sites that know what they are logging use
safe_url_for_log() and redact_secrets() directly. install_log_redaction() adds
a process-wide safety net that runs redact_secrets() over every log record, so
a call site that was missed, or one added later, is still covered.
"""
import logging
import re
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

REDACTED = "[redacted]"

# Path segments an API endpoint is normally made of. A base URL is set by the
# user and some gateways carry an account ID or a token in the path, so a
# segment is only kept when it is one of these words. A model name cannot be
# told apart from a token, so it is replaced as well.
SAFE_PATH_SEGMENTS = frozenset({
    "api", "openai", "v1", "v1beta", "v1alpha", "v2", "v3", "chat", "completions", "responses",
    "models", "embeddings", "images", "generations", "edits", "audio", "files", "messages",
})
# Google-style method suffix, as in "<model>:generateContent".
MODEL_METHODS = frozenset({"generatecontent", "streamgeneratecontent", "counttokens", "embedcontent"})

# A value is treated as a credential when the name in front of it contains one
# of these words. Matching inside the name covers compound names such as
# "client_secret", "x-goog-api-key" or "refresh_token" without listing each.
SECRET_NAME_WORDS = (
    "authorization", "credential", "passphrase", "signature", "password", "passwd", "session",
    "cookie", "secret", "token", "apikey", "auth", "key", "pwd", "sig",
)
# Prefixes that identify a credential wherever it appears.
SECRET_PREFIX_PATTERNS = {
    "sk-": r"\bsk-[A-Za-z0-9_-]{8,}",
    "AIza": r"\bAIza[0-9A-Za-z_-]{20,}",
    "ghp_": r"\bghp_[A-Za-z0-9]{20,}",
    "gho_": r"\bgho_[A-Za-z0-9]{20,}",
    "xoxb-": r"\bxox[abp]-[A-Za-z0-9-]{10,}",
    # JSON Web Token: three base64url parts.
    "eyJ": r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
}

_NAME = r"[A-Za-z0-9_.-]*(?:" + "|".join(SECRET_NAME_WORDS) + r")[A-Za-z0-9_.-]*"
# name: value, name=value, "name": "value". Any value of four characters or more.
_NAMED_VALUE_RE = re.compile(r"(?i)(?<![A-Za-z0-9])(" + _NAME + r")(\s*[\"']?\s*[:=]\s*[\"']?)[^\s\"',;&}]{4,}")
# name value. Only when the value looks like a credential, so that ordinary
# prose such as "token expired" is left alone.
_NAMED_SPACED_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(" + _NAME + r")(\s+)(?=[^\s]*[0-9])(?=[^\s]*[A-Za-z])[A-Za-z0-9._~+/=-]{12,}"
)
# A Cookie header carries several name=value pairs; drop the whole value.
_COOKIE_HEADER_RE = re.compile(r"(?i)\b((?:set-)?cookie)(\s*[\"']?\s*[:=]\s*[\"']?)[^\r\n\"']+")
_SCHEME_TOKEN_RE = re.compile(r"(?i)\b(bearer|basic|digest)\s+[A-Za-z0-9._~+/=-]{8,}")
_PREFIX_RES = tuple(re.compile(pattern) for pattern in SECRET_PREFIX_PATTERNS.values())
_URL_IN_TEXT_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]{1,15}://[^\s\"'<>)\]}]+")
# Last resort for a credential with no name and no known prefix: a long run
# that mixes upper case, lower case and digits. Hex digests and ordinary
# words do not match.
_RANDOM_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*[A-Z])(?=[A-Za-z0-9_-]*[0-9])"
    r"[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_-])"
)
_HOST_LABEL_TOKEN_RE = re.compile(r"^(?=.*[a-z])(?=.*[0-9])[a-z0-9-]{20,}$", re.IGNORECASE)
# Cheap test for whether a log message needs the full treatment.
_TRIGGER_RE = re.compile(
    r"(?i)://|bearer|basic |digest |" + "|".join(SECRET_NAME_WORDS) + "|"
    + "|".join(re.escape(prefix) for prefix in SECRET_PREFIX_PATTERNS) + r"|[A-Za-z0-9_-]{24,}"
)
# Known secrets shorter than this are not searched for: they would match ordinary text.
_MIN_KNOWN_SECRET_LENGTH = 6


def _safe_path(path: str) -> str:
    """Keep the endpoint words of a URL path and replace every other segment."""
    kept = []
    for segment in path.split("/"):
        name, separator, method = segment.partition(":")
        if segment == "" or segment.lower() in SAFE_PATH_SEGMENTS:
            kept.append(segment)
        elif separator and method.lower() in MODEL_METHODS:
            kept.append(f"{REDACTED}:{method}")
        else:
            kept.append(REDACTED)
    return "/".join(kept)


def _safe_host(hostname: str) -> str:
    """Replace host labels that look like a token used as a subdomain."""
    if ":" in hostname:
        return f"[{hostname}]"
    return ".".join(REDACTED if _HOST_LABEL_TOKEN_RE.match(label) else label for label in hostname.split("."))


def safe_url_for_log(url) -> str:
    """Return the URL with everything that could be a credential removed.

    Dropped: the user name and password, the query string, the fragment,
    every path segment that is not a known endpoint word, and host labels
    that look like tokens. Scheme, host and port stay, which is enough to
    tell which service failed.
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
    netloc = _safe_host(parts.hostname)
    if port:
        netloc = f"{netloc}:{port}"
    suffix = "?" + REDACTED if parts.query or parts.fragment else ""
    return urlunsplit((parts.scheme, netloc, _safe_path(parts.path), "", "")) + suffix


def _known_secret_forms(secret: str):
    """The secret as written, and as it appears inside a URL."""
    return {secret, quote(secret, safe=""), quote_plus(secret)}


def redact_secrets(text, known_secrets=()) -> str:
    """Return the text with known secrets, URLs and credential-shaped values made safe."""
    result = str(text or "")
    for secret in known_secrets or ():
        if isinstance(secret, str) and len(secret) >= _MIN_KNOWN_SECRET_LENGTH:
            for form in _known_secret_forms(secret):
                result = result.replace(form, REDACTED)
    result = _URL_IN_TEXT_RE.sub(lambda match: safe_url_for_log(match.group(0)), result)
    result = _COOKIE_HEADER_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", result)
    result = _SCHEME_TOKEN_RE.sub(lambda match: f"{match.group(1)} {REDACTED}", result)
    for pattern in _PREFIX_RES:
        result = pattern.sub(REDACTED, result)
    result = _NAMED_VALUE_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", result)
    result = _NAMED_SPACED_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", result)
    return _RANDOM_TOKEN_RE.sub(REDACTED, result)


def _redacting_factory(previous_factory):
    def factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        try:
            message = record.getMessage()
            if _TRIGGER_RE.search(message):
                cleaned = redact_secrets(message)
                if cleaned != message:
                    record.msg = cleaned
                    record.args = None
        except Exception:
            # Logging must never fail because a message could not be formatted here;
            # the handler reports formatting problems in its own way.
            pass
        return record

    factory._manga_redaction_installed = True
    return factory


def install_log_redaction() -> None:
    """Redact every log record created in this process. Safe to call more than once."""
    current = logging.getLogRecordFactory()
    if getattr(current, "_manga_redaction_installed", False):
        return
    logging.setLogRecordFactory(_redacting_factory(current))
