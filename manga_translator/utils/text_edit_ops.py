"""Collecting and minimally narrowing plain-text edit operations (backend logic; the UI layer only forwards events).

``contentsChange`` of a Qt text box gives ``(position, removed count, inserted count)``, with two quirks:
- when a change reaches the end of the document, the removed and inserted counts include the final paragraph separator (1 too many);
- IME pre-editing and committing report the whole document as one "full replacement".

This module compares each report with a mirror of the text from before the change and narrows it to the smallest real operation
``[pos, removed_len, inserted_text]``; a "fake replacement" during pre-editing, where the text did not change, is empty after narrowing
and is dropped. This is an exact narrowing of recorded operations, not a fuzzy diff.

The coordinate convention is that of the edit box, with real ``\\n`` line breaks; each line break still counts as one character.
"""

from __future__ import annotations

from typing import List, Optional


def minimal_edit_op(
    mirror: str,
    current: str,
    position: int,
    chars_removed: int,
    chars_added: int,
) -> Optional[list]:
    """Narrow one contentsChange to the smallest operation; ``None`` is returned when nothing really changed.

    ``mirror`` is the text before the change and ``current`` the text after it; the unchanged start and end of the reported range
    are trimmed, which restores the real inserted, removed or replaced range. Overstated counts (the final paragraph
    separator) are clamped naturally by slicing out of range.
    """
    pos = max(0, int(position))
    removed_text = mirror[pos : pos + max(0, int(chars_removed))]
    added_text = current[pos : pos + max(0, int(chars_added))]
    while removed_text and added_text and removed_text[0] == added_text[0]:
        removed_text = removed_text[1:]
        added_text = added_text[1:]
        pos += 1
    while removed_text and added_text and removed_text[-1] == added_text[-1]:
        removed_text = removed_text[:-1]
        added_text = added_text[:-1]
    if not removed_text and not added_text:
        return None
    return [pos, len(removed_text), added_text]


class EditOpRecorder:
    """Track the sequence of edit operations of one text box and produce the edit_info used for rich-text sync.

    Only three things are left for the UI layer to do:
    - on every text change, call :meth:`record_change` (a user edit) or
      :meth:`invalidate` (a programmatic write, which voids the operations);
    - after a programmatic refresh, call :meth:`reset` to rebuild the baseline;
    - when emitting the change signal, call :meth:`take_edit_info` to take the operations and advance the baseline.
    """

    def __init__(self) -> None:
        self._ops: List[list] = []
        # Mirror of the text in the box as it is, used to narrow the next report
        self._doc_text = ""
        # The canonical text at the last take/reset, used as pre_text of the next edit_info
        self._baseline = ""

    def reset(self, current_text: str) -> None:
        """Rebuild the baseline from the current text (called after a programmatic refresh)."""
        self._doc_text = current_text
        self._ops = []
        self._baseline = current_text

    def invalidate(self, current_text: str) -> None:
        """A programmatic write: the mirror follows and the accumulated operations are void; the baseline is rebuilt by the reset that follows."""
        self._doc_text = current_text
        self._ops = []

    def record_change(
        self,
        current_text: str,
        position: int,
        chars_removed: int,
        chars_added: int,
    ) -> None:
        """Record one user edit (the contentsChange report + the full text after the change)."""
        mirror = self._doc_text
        self._doc_text = current_text
        op = minimal_edit_op(mirror, current_text, position, chars_removed, chars_added)
        if op is not None:
            self._ops.append(op)

    def take_edit_info(self, current_text: str) -> dict:
        """Take the accumulated operations, return {ops, pre_text, post_text} (in \\n convention) and advance the baseline."""
        post_text = current_text
        edit_info = {
            "ops": self._ops,
            "pre_text": self._baseline,
            "post_text": post_text,
        }
        self._ops = []
        self._baseline = post_text
        return edit_info
