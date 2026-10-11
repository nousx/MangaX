"""
Username validation policy.

Usernames end up in HTML, URLs, log lines and file names, so new accounts are
restricted to a conservative allowlist.  Accounts that already exist are never
re-validated: loading, login and administration of legacy usernames keep
working (the web UI renders them with DOM APIs, not HTML interpolation).
"""

import unicodedata

USERNAME_MIN_LENGTH = 2
USERNAME_MAX_LENGTH = 50
USERNAME_ALLOWED_PUNCTUATION = "_-."

USERNAME_RULES_MESSAGE = (
    f"Username must be {USERNAME_MIN_LENGTH}-{USERNAME_MAX_LENGTH} characters, "
    "use only letters, digits, '_', '-' and '.', and start and end with a letter or digit"
)


class InvalidUsernameError(ValueError):
    """Raised when a username does not satisfy the allowlist."""


def _is_letter_or_digit(ch: str) -> bool:
    category = unicodedata.category(ch)
    # L* = letters of any script, Nd = decimal digits.
    return category.startswith("L") or category == "Nd"


def _is_allowed_char(ch: str) -> bool:
    if ch in USERNAME_ALLOWED_PUNCTUATION:
        return True
    if _is_letter_or_digit(ch):
        return True
    # Combining marks (Mn/Mc) are required to spell names in scripts such as
    # Thai or Devanagari.  They are only accepted after a base character.
    return unicodedata.category(ch) in ("Mn", "Mc")


def validate_username(username) -> str:
    """
    Validate a username for a NEW account and return it unchanged.

    Raises:
        InvalidUsernameError: if the username is not acceptable.
    """
    if not isinstance(username, str):
        raise InvalidUsernameError(USERNAME_RULES_MESSAGE)
    if not (USERNAME_MIN_LENGTH <= len(username) <= USERNAME_MAX_LENGTH):
        raise InvalidUsernameError(USERNAME_RULES_MESSAGE)
    # Require the canonical composed form so that visually identical names
    # cannot be registered twice with different code point sequences.
    if unicodedata.normalize("NFC", username) != username:
        raise InvalidUsernameError(USERNAME_RULES_MESSAGE)
    if not all(_is_allowed_char(ch) for ch in username):
        raise InvalidUsernameError(USERNAME_RULES_MESSAGE)
    if not _is_letter_or_digit(username[0]):
        raise InvalidUsernameError(USERNAME_RULES_MESSAGE)
    if username[-1] in USERNAME_ALLOWED_PUNCTUATION:
        raise InvalidUsernameError(USERNAME_RULES_MESSAGE)
    return username


def is_valid_username(username) -> bool:
    """Return True if `username` is acceptable for a new account."""
    try:
        validate_username(username)
    except InvalidUsernameError:
        return False
    return True
