"""
Editor history manager.
Built on Qt's native QUndoStack; wraps command execution, macros and the clipboard in one place.
"""

import copy
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QUndoCommand, QUndoStack


class ClipboardManager:
    """Clipboard manager that handles copy and paste of internal data."""

    def __init__(self):
        self.clipboard_data = None
        self.logger = logging.getLogger(__name__)

    def copy_to_clipboard(self, data: Any):
        """Copy data to the internal clipboard."""
        self.clipboard_data = copy.deepcopy(data)
        self.logger.debug("Data copied to internal clipboard")

    def paste_from_clipboard(self) -> Any:
        """Paste data from the internal clipboard."""
        if self.clipboard_data is not None:
            return copy.deepcopy(self.clipboard_data)
        return None

    def has_data(self) -> bool:
        """Check whether the clipboard holds data."""
        return self.clipboard_data is not None


class EditorStateManager(QObject):
    """
    Editor state manager.

    Responsibilities:
    - running QUndoCommand objects in one place
    - macro commands (a batch operation undone in one step)
    - providing the can_undo/can_redo state signals to the outside
    - providing the internal clipboard
    """

    undo_redo_state_changed = pyqtSignal(bool, bool)  # can_undo, can_redo
    stack_index_changed = pyqtSignal(int)

    def __init__(self, undo_limit: int = 50, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(max(1, int(undo_limit)))
        self.clipboard = ClipboardManager()
        self._macro_depth = 0
        self._revision = 0

        self.undo_stack.canUndoChanged.connect(self._on_undo_redo_changed)
        self.undo_stack.canRedoChanged.connect(self._on_undo_redo_changed)
        self.undo_stack.indexChanged.connect(self._on_index_changed)

    def execute(self, command: Optional[QUndoCommand]):
        """
        Run a command.
        QUndoStack.push() calls command.redo() at once.
        """
        if command is None:
            return
        self.undo_stack.push(command)

    def push_command(self, command: Optional[QUndoCommand]):
        """For the old interface."""
        self.execute(command)

    def begin_macro(self, text: str):
        """Begin a macro command."""
        self.undo_stack.beginMacro(text)
        self._macro_depth += 1

    def end_macro(self):
        """End a macro command."""
        if self._macro_depth <= 0:
            self.logger.warning("end_macro called without active macro")
            return
        self.undo_stack.endMacro()
        self._macro_depth -= 1

    @contextmanager
    def macro(self, text: str) -> Iterator[None]:
        """Macro command in the form of a context manager."""
        self.begin_macro(text)
        try:
            yield
        finally:
            self.end_macro()

    def undo(self):
        """Undo the last operation."""
        if self.undo_stack.canUndo():
            self.undo_stack.undo()

    def redo(self):
        """Redo the last operation that was undone."""
        if self.undo_stack.canRedo():
            self.undo_stack.redo()

    def can_undo(self) -> bool:
        """Check whether undo is possible."""
        return self.undo_stack.canUndo()

    def can_redo(self) -> bool:
        """Check whether redo is possible."""
        return self.undo_stack.canRedo()

    def set_undo_limit(self, limit: int):
        """Set the limit of the undo stack at run time."""
        self.undo_stack.setUndoLimit(max(1, int(limit)))

    def copy_to_clipboard(self, data: Any):
        """Copy data to the internal clipboard."""
        self.clipboard.copy_to_clipboard(data)

    def paste_from_clipboard(self) -> Any:
        """Paste data from the internal clipboard."""
        return self.clipboard.paste_from_clipboard()

    def has_clipboard_data(self) -> bool:
        """Check whether the clipboard holds data."""
        return self.clipboard.has_data()

    def clear(self):
        """Clear the history."""
        while self._macro_depth > 0:
            self.end_macro()
        self.undo_stack.clear()
        self.logger.debug("Cleared undo stack")

    def mark_clean(self):
        """Mark the current state as saved."""
        self.undo_stack.setClean()

    def mark_dirty(self):
        """Clear the saved mark (for example to take back an optimistic mark after a failed export)."""
        self.undo_stack.resetClean()

    def is_clean(self) -> bool:
        """Whether the state is saved."""
        return self.undo_stack.isClean()

    def current_revision(self) -> int:
        return self._revision

    def create_undo_action(self, parent, text: str = "撤销"):
        """Create the undo action (for menus and toolbars)."""
        return self.undo_stack.createUndoAction(parent, text)

    def create_redo_action(self, parent, text: str = "重做"):
        """Create the redo action (for menus and toolbars)."""
        return self.undo_stack.createRedoAction(parent, text)

    def _on_undo_redo_changed(self, _):
        self.undo_redo_state_changed.emit(
            self.undo_stack.canUndo(), self.undo_stack.canRedo()
        )

    def _on_index_changed(self, index: int):
        self._revision += 1
        self.stack_index_changed.emit(index)


# --- Singleton Pattern ---
_history_service_instance: Optional[EditorStateManager] = None


def get_history_service() -> EditorStateManager:
    """Get the singleton of the history service."""
    global _history_service_instance
    if _history_service_instance is None:
        _history_service_instance = EditorStateManager()
    return _history_service_instance
