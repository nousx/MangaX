import re
import unicodedata

_TRAILING_REMOVABLE_PUNCT_CHARS = ".。．｡︒﹒,，︐﹐‚„"
_TRAILING_CLOSERS = "\"'”’)]}）］】〉》」』"


def _split_text_tail(text: str) -> tuple[str, str]:
    value = str(text or "")
    end = len(value)

    while end > 0 and value[end - 1].isspace():
        end -= 1

    whitespace_suffix = value[end:]
    core = value[:end]
    close_start = len(core)

    while close_start > 0 and core[close_start - 1] in _TRAILING_CLOSERS:
        close_start -= 1

    return core[:close_start], core[close_start:] + whitespace_suffix


def has_terminal_punctuation(text: str) -> bool:
    body, _suffix = _split_text_tail(text)
    if not body:
        return False

    last_char = body[-1]
    if last_char in _TRAILING_REMOVABLE_PUNCT_CHARS:
        return True

    return unicodedata.category(last_char).startswith("P")


def remove_trailing_period_if_needed(source_text: str, translation_text: str, enabled: bool) -> str:
    translation = str(translation_text or "")
    if not enabled or not translation or has_terminal_punctuation(source_text):
        return translation

    body, suffix = _split_text_tail(translation)
    if not body or body[-1] not in _TRAILING_REMOVABLE_PUNCT_CHARS:
        return translation

    if len(body) >= 2 and body[-2] in _TRAILING_REMOVABLE_PUNCT_CHARS:
        return translation

    return body[:-1] + suffix


_EXCLAMATION_MARKS = "!\uFF01\u203C"
_QUESTION_MARKS = "?\uFF1F"
_MIXED_MARKS = "\u2048\u2049"
_MARK_RUN = re.compile(f"[{re.escape(_EXCLAMATION_MARKS + _QUESTION_MARKS + _MIXED_MARKS)}]+")
# Words that already make a Thai sentence a question, so a question mark adds nothing.
_THAI_QUESTION_WORDS = (
    "\u0e2b\u0e23\u0e37\u0e2d\u0e40\u0e1b\u0e25\u0e48\u0e32", "\u0e23\u0e36\u0e40\u0e1b\u0e25\u0e48\u0e32",
    "\u0e40\u0e21\u0e37\u0e48\u0e2d\u0e44\u0e2b\u0e23\u0e48", "\u0e40\u0e21\u0e37\u0e48\u0e2d\u0e44\u0e23",
    "\u0e40\u0e17\u0e48\u0e32\u0e44\u0e2b\u0e23\u0e48", "\u0e40\u0e17\u0e48\u0e32\u0e44\u0e23",
    "\u0e2d\u0e22\u0e48\u0e32\u0e07\u0e44\u0e23", "\u0e17\u0e35\u0e48\u0e44\u0e2b\u0e19",
    "\u0e22\u0e31\u0e07\u0e44\u0e07", "\u0e17\u0e33\u0e44\u0e21", "\u0e2d\u0e30\u0e44\u0e23",
    "\u0e40\u0e2b\u0e23\u0e2d", "\u0e2b\u0e23\u0e2d", "\u0e2b\u0e23\u0e37\u0e2d", "\u0e44\u0e2b\u0e21",
    "\u0e21\u0e31\u0e49\u0e22", "\u0e21\u0e31\u0e4a\u0e22", "\u0e43\u0e04\u0e23", "\u0e23\u0e36",
)
# Particles that may follow the question word before the sentence ends.
_THAI_FINAL_PARTICLES = (
    "\u0e04\u0e23\u0e31\u0e1a", "\u0e04\u0e31\u0e1a", "\u0e04\u0e30", "\u0e04\u0e48\u0e30", "\u0e08\u0e4a\u0e30",
    "\u0e08\u0e49\u0e30", "\u0e19\u0e30", "\u0e19\u0e48\u0e30", "\u0e25\u0e48\u0e30", "\u0e40\u0e19\u0e35\u0e48\u0e22",
    "\u0e01\u0e31\u0e19\u0e41\u0e19\u0e48", "\u0e01\u0e31\u0e19", "\u0e27\u0e30", "\u0e27\u0e48\u0e30", "\u0e2d\u0e35\u0e01",
)
_QUESTION_WORD_THEN_MARK = re.compile(
    "(" + "|".join(_THAI_QUESTION_WORDS) + ")"
    + "((?:\\s*(?:" + "|".join(_THAI_FINAL_PARTICLES) + ")){0,2})"
    + "\\s*(\\?!|\\?)"
)


def is_thai_language(code) -> bool:
    normalized = str(code or "").strip().lower().replace("-", "_")
    return normalized in ("th", "tha", "thai") or normalized.startswith("th_")


def _collapse_mark_run(match: "re.Match[str]") -> str:
    run = match.group(0)
    asks = any(mark in _QUESTION_MARKS + _MIXED_MARKS for mark in run)
    exclaims = any(mark in _EXCLAMATION_MARKS + _MIXED_MARKS for mark in run)
    if asks and exclaims:
        return "?!"
    return "?" if asks else "!"


def normalize_thai_punctuation(translation_text, enabled: bool):
    """Tidy sentence-ending marks the way Thai is normally written.

    English sources end most lines with "!" or "?" and translations tend to
    copy them. This keeps one mark per ending and removes a question mark
    that follows a word which already marks the sentence as a question. It
    never removes a lone "!": whether a line is shouted cannot be decided
    from the text alone.
    """
    if not enabled or not isinstance(translation_text, str) or not translation_text:
        return translation_text

    collapsed = _MARK_RUN.sub(_collapse_mark_run, translation_text)
    return _QUESTION_WORD_THEN_MARK.sub(
        lambda match: match.group(1) + match.group(2) + ("!" if match.group(3) == "?!" else ""),
        collapsed,
    )
