"""Keep credentials out of log files and error messages.

Log files are long-lived and get shared when asking for help, and error
messages are shown on screen and copied into bug reports. Nothing written to
either may carry an API key, a password or a token.

Two layers use this module. Call sites that know what they are logging use
safe_url_for_log() and redact_secrets() directly. install_log_redaction() adds
a process-wide safety net that runs redact_secrets() over every log record,
including its traceback, so a call site that was missed, or one added later,
is still covered.

The text being redacted is often controlled by a remote server, so every
step here runs in time proportional to the length of the text, and no more
than MAX_REDACTED_CHARS are processed at all.
"""
import logging
import re
import traceback
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

REDACTED = "[redacted]"
# Longer text is cut: what follows is not logged at all, which is safe and
# keeps the cost of redaction bounded.
MAX_REDACTED_CHARS = 20000
WITHHELD = "[log message withheld: it could not be checked for credentials]"

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
# of these words, which covers compound names such as "client_secret" or
# "refresh_token" without listing each.
SECRET_NAME_WORDS = (
    "authorization", "credential", "passphrase", "signature", "password", "passwd", "session",
    "cookie", "secret", "token", "apikey", "auth", "key", "pwd", "sig",
)
# Short words also occur inside ordinary words. They count when they are a
# whole part of the name ("api_key", "sig"), or when a part ends with "key" or
# "pwd" or starts with "auth" or "key" ("accessKey", "privatekey", "authkey",
# "keyid"), unless that part is one of the ordinary words below.
_WHOLE_PART_WORDS = frozenset(word for word in SECRET_NAME_WORDS if len(word) <= 4)
_ORDINARY_PARTS = frozenset({
    "monkey", "hotkey", "hotkeys", "turkey", "donkey", "hockey", "jockey", "whiskey",
    "author", "authors", "authored", "keyboard", "keyboards", "keyword", "keywords",
    "keyframe", "keyframes", "keynote",
})
_SUBSTRING_WORDS = tuple(word for word in SECRET_NAME_WORDS if len(word) > 4)
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

# Every pattern below consumes each character at most once per attempt and is
# either anchored by the caller or starts on a literal, so matching is linear.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")
_PART_SPLIT_RE = re.compile(r"[_.-]")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# After a name: separator then value (name: value, name=value, "name": "value").
_SEPARATED_VALUE_RE = re.compile(
    r"(\s{0,64}[\"']?\s{0,64}[:=]\s{0,64})"
    # A quoted value is taken whole, spaces included; a bare one up to the next delimiter.
    r"(?:\"(?P<double>[^\"\r\n]{1,512})|'(?P<single>[^'\r\n]{1,512})|(?P<bare>[^\s\"',;&}]{1,}))"
)
# After a name: a space then a value. Only redacted when it looks like a
# credential, so that prose such as "token expired" is left alone.
_SPACED_VALUE_RE = re.compile(r"(\s{1,64})(?P<spaced>[A-Za-z0-9._~+/=-]{12,})")
# A Cookie header carries several name=value pairs; drop the whole value.
_COOKIE_HEADER_RE = re.compile(r"(?i)\b((?:set-)?cookie)(\s{0,64}[\"']?\s{0,64}[:=]\s{0,64}[\"']?)[^\r\n]+")
_SCHEME_TOKEN_RE = re.compile(r"(?i)\b(bearer|basic|digest)\s{1,64}[A-Za-z0-9._~+/=-]{8,}")
_PREFIX_RES = tuple(re.compile(pattern) for pattern in SECRET_PREFIX_PATTERNS.values())
_URL_IN_TEXT_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]{1,15}://\S+")
# Punctuation that usually closes a sentence or a bracket around a URL.
_URL_TRAILING_PUNCTUATION = ")]}>\"'.,;"
_HOST_LABEL_TOKEN_RE = re.compile(r"^[a-z0-9-]{20,}$", re.IGNORECASE)
# An unnamed credential: a run this long that mixes upper case, lower case and digits.
_RANDOM_TOKEN_MIN_LENGTH = 24
# Known secrets shorter than this are not searched for: they would match ordinary text.
_MIN_KNOWN_SECRET_LENGTH = 6


def _safe_path(path: str) -> str:
    """Keep the endpoint words of a URL path and replace every other segment."""
    kept = []
    for segment in path.split("/"):
        _name, separator, method = segment.partition(":")
        if segment == "" or segment.lower() in SAFE_PATH_SEGMENTS:
            kept.append(segment)
        elif separator and method.lower() in MODEL_METHODS:
            kept.append(f"{REDACTED}:{method}")
        else:
            kept.append(REDACTED)
    return "/".join(kept)


def _looks_like_host_token(label: str) -> bool:
    return bool(
        _HOST_LABEL_TOKEN_RE.match(label)
        and any(char.isdigit() for char in label)
        and any(char.isalpha() for char in label)
    )


def _safe_host(hostname: str) -> str:
    """Replace host labels that look like a token used as a subdomain."""
    if ":" in hostname:
        return f"[{hostname}]"
    return ".".join(REDACTED if _looks_like_host_token(label) else label for label in hostname.split("."))


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
    if len(text) > MAX_REDACTED_CHARS:
        return REDACTED
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


def _is_secret_name(token: str) -> bool:
    if any(word in token.lower() for word in _SUBSTRING_WORDS):
        return True
    for part in _PART_SPLIT_RE.split(_CAMEL_BOUNDARY_RE.sub("_", token).lower()):
        if part in _WHOLE_PART_WORDS:
            return True
        if part in _ORDINARY_PARTS:
            continue
        if part.endswith(("key", "keys", "pwd")) or part.startswith(("auth", "key")):
            return True
    return False


def _looks_random(token: str) -> bool:
    for part in token.split("."):
        if (
            len(part) >= _RANDOM_TOKEN_MIN_LENGTH
            and any(char.islower() for char in part)
            and any(char.isupper() for char in part)
            and any(char.isdigit() for char in part)
        ):
            return True
    return False


def _redact_named_and_random(text: str) -> str:
    """One pass over the name-like tokens: redact values after secret names, and random-looking tokens."""
    pieces = []
    position = 0
    for match in _TOKEN_RE.finditer(text):
        if match.start() < position:
            # Already consumed as the value of a name before it.
            continue
        token = match.group(0)
        if _is_secret_name(token):
            value = _SEPARATED_VALUE_RE.match(text, match.end())
            if value is None:
                value = _SPACED_VALUE_RE.match(text, match.end())
                if value is not None:
                    candidate = value.group("spaced")
                    if not (any(c.isdigit() for c in candidate) and any(c.isalpha() for c in candidate)):
                        value = None
            if value is not None:
                # The named group that matched: a quoted, bare or spaced value.
                group = value.lastgroup
                pieces.append(text[position:value.start(group)])
                pieces.append(REDACTED)
                position = value.end(group)
                continue
        if _looks_random(token):
            pieces.append(text[position:match.start()])
            pieces.append(REDACTED)
            position = match.end()
    pieces.append(text[position:])
    return "".join(pieces)


def _redact_url_match(match) -> str:
    raw = match.group(0)
    core = raw.rstrip(_URL_TRAILING_PUNCTUATION)
    return safe_url_for_log(core) + raw[len(core):]


def _known_secret_forms(secret: str):
    """The secret as written, and as it appears inside a URL."""
    return {secret, quote(secret, safe=""), quote_plus(secret)}


def redact_secrets(text, known_secrets=()) -> str:
    """Return the text with known secrets, URLs and credential-shaped values made safe."""
    result = str(text or "")
    dropped = len(result) - MAX_REDACTED_CHARS
    if dropped > 0:
        result = result[:MAX_REDACTED_CHARS]
    for secret in known_secrets or ():
        if isinstance(secret, str) and len(secret) >= _MIN_KNOWN_SECRET_LENGTH:
            for form in _known_secret_forms(secret):
                result = result.replace(form, REDACTED)
    result = _URL_IN_TEXT_RE.sub(_redact_url_match, result)
    result = _COOKIE_HEADER_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", result)
    result = _SCHEME_TOKEN_RE.sub(lambda match: f"{match.group(1)} {REDACTED}", result)
    for pattern in _PREFIX_RES:
        result = pattern.sub(REDACTED, result)
    result = _redact_named_and_random(result)
    if dropped > 0:
        # The cut may have split a credential, so the last stretch goes too.
        result = result[:-64] + f" [{dropped + 64} more characters not logged]"
    return result


def _redact_record(record: logging.LogRecord) -> None:
    # Every record is checked. A cheaper test deciding which records to check
    # would be a second set of rules that can disagree with the real one.
    message = record.getMessage()
    cleaned = redact_secrets(message)
    if cleaned != message:
        record.msg = cleaned
        record.args = None
    # The traceback and the stack are formatted by the handler, after this
    # point. Format them here so the handler writes the redacted text.
    if record.exc_info and not record.exc_text:
        exc_info = record.exc_info
        if isinstance(exc_info, tuple) and exc_info[0] is not None:
            record.exc_text = redact_secrets("".join(traceback.format_exception(*exc_info)).rstrip("\n"))
    elif record.exc_text:
        record.exc_text = redact_secrets(record.exc_text)
    if record.stack_info:
        record.stack_info = redact_secrets(record.stack_info)


def _redacting_factory(previous_factory):
    def factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        try:
            record.getMessage()
        except Exception:
            # A malformed log call: the handler reports it, to stderr and not to the log file.
            return record
        try:
            _redact_record(record)
        except Exception:
            # Fail closed: a record that could not be checked is not written as it is.
            record.msg = WITHHELD
            record.args = None
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return record

    factory._manga_redaction_installed = True
    return factory


def install_log_redaction() -> None:
    """Redact every log record created in this process. Safe to call more than once."""
    current = logging.getLogRecordFactory()
    if getattr(current, "_manga_redaction_installed", False):
        return
    logging.setLogRecordFactory(_redacting_factory(current))
