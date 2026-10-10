"""Keep the computer awake while a long job runs."""

import contextlib
import sys

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _set_execution_state(flags: int) -> bool:
    import ctypes

    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))


@contextlib.contextmanager
def keep_system_awake():
    """Ask Windows not to sleep until the block ends. The display may still turn off.

    The request belongs to the calling thread, so enter and leave the block on
    the thread that runs the job. Yields whether the request was accepted;
    other systems and failures yield False and change nothing.
    """
    active = False
    if sys.platform == 'win32':
        try:
            active = _set_execution_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
        except Exception:
            active = False
    try:
        yield active
    finally:
        if active:
            with contextlib.suppress(Exception):
                _set_execution_state(_ES_CONTINUOUS)
