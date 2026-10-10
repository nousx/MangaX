"""Leave a trace for errors that the caller deliberately does not act on.

Many handlers in the pipeline catch an error and carry on, which is right for
optional steps but hides the cause when a result looks wrong. They report here
so that a verbose log shows what was ignored and where.
"""
import logging

_logger = logging.getLogger("manga-translator").getChild("ignored")


def note_ignored_error(error: BaseException, where: str) -> None:
    """Record at debug level that an error was caught and ignored."""
    try:
        _logger.debug("%s ignored %s: %s", where, type(error).__name__, error)
    except Exception:  # noqa: BLE001 - reporting must never raise into the caller
        return
