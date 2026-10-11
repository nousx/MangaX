from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QApplication, QDialog, QFrame, QLabel, QWidget
from qfluentwidgets import CardWidget, FluentStyleSheet
from qframelesswindow import FramelessDialog


def normalize_dialog_parent(parent):
    """Normalise any widget parent to the top-level window it belongs to.

    A Fluent dialog is a real top-level window. Using a native child widget embedded in a stacked page
    (for example a page inside MSFluentWindow) directly as transient parent would make Qt use
    a QWidgetWindow that is not top-level, which breaks positioning and modality. When parent is invalid or None,
    it falls back to the currently active window.
    """
    candidate = parent if isinstance(parent, QWidget) else QApplication.activeWindow()
    if candidate is None:
        return None
    top_level = candidate.window()
    return top_level if top_level is not None else candidate


def _centered_window_position(size: QSize, anchor: QRect, available: QRect) -> QPoint:
    """Return an owner-centered position clamped to the screen work area."""
    x = anchor.center().x() - size.width() // 2
    y = anchor.center().y() - size.height() // 2
    max_x = max(available.right() - size.width() + 1, available.left())
    max_y = max(available.bottom() - size.height() + 1, available.top())
    return QPoint(
        min(max(x, available.left()), max_x),
        min(max(y, available.top()), max_y),
    )


def _resolve_dialog_owner(dialog: QWidget, owner: QWidget | None = None) -> QWidget | None:
    candidate = owner if owner is not None else dialog.parentWidget()
    if candidate is None:
        active = QApplication.activeWindow()
        if active is not dialog:
            candidate = active
    normalized = normalize_dialog_parent(candidate) if candidate is not None else None
    return None if normalized is dialog else normalized


def center_dialog_on_owner(dialog: QWidget, owner: QWidget | None = None) -> None:
    """Center a top-level dialog over its owner and keep it on the same screen."""
    owner = _resolve_dialog_owner(dialog, owner)

    screen = owner.screen() if owner is not None else dialog.screen()
    screen = screen or QApplication.primaryScreen()
    if screen is None:
        return

    available = screen.availableGeometry()
    anchor = owner.frameGeometry() if owner is not None and owner.isVisible() else available
    dialog.move(_centered_window_position(dialog.size(), anchor, available))


class FluentSecondaryDialog(FramelessDialog):
    """Shared Fluent shell for secondary dialogs.

    - the parent is normalised to the top-level window automatically (falling back to activeWindow when parent=None);
    - the TitleBar is hidden by default, but the window can be dragged by holding the empty background or a display-only widget
      (startSystemMove, the proper way to drag a frameless window);
    - before the first show, the minimum and initial sizes are clamped to within 90% of the available screen area;
    - at each show, the dialog is centred on its top-level window and its position is clamped to the work area of the same screen.
    """

    _SCREEN_CLAMP_RATIO = 0.9

    def __init__(self, parent=None):
        super().__init__(normalize_dialog_parent(parent))
        self._screen_clamped = False
        self.titleBar.setVisible(False)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setContentsMargins(0, 0, 0, 0)
        self.apply_fluent_dialog_style()

    def apply_fluent_dialog_style(self):
        FluentStyleSheet.DIALOG.apply(self)

    # ─── Dragging the frameless window ──────────────────────
    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._is_drag_region(event.position().toPoint())
        ):
            window = self.windowHandle()
            if window is not None and window.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def _is_drag_region(self, pos: QPoint) -> bool:
        """Dragging is only allowed from the background or a display-only widget; interactive widgets keep their behaviour.

        The parent chain is walked from the widget that was hit up to the dialog itself, and any interactive widget on the way
        (a button, an input box, a tree view and so on) makes it a non-drag area.
        """
        widget = self.childAt(pos)
        while widget is not None and widget is not self:
            if not self._is_passive_widget(widget):
                return False
            widget = widget.parentWidget()
        return True

    @staticmethod
    def _is_passive_widget(widget: QWidget) -> bool:
        if isinstance(widget, QLabel):
            interactive_flags = (
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.LinksAccessibleByMouse
            )
            return not (widget.textInteractionFlags() & interactive_flags)
        if isinstance(widget, CardWidget):
            return not widget.isClickEnabled()
        # Pure layout containers: the type is matched exactly, so interactive subclasses of QWidget/QFrame are not mistaken for background
        return type(widget) in (QWidget, QFrame)

    # ─── Clamping to the screen size ───────────────────────
    def showEvent(self, event):
        if not self._screen_clamped:
            self._screen_clamped = True
            self._clamp_to_screen()
        super().showEvent(event)
        center_dialog_on_owner(self)

    def _clamp_to_screen(self):
        owner = _resolve_dialog_owner(self)
        screen = owner.screen() if owner is not None else self.screen()
        screen = screen or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        max_w = max(int(available.width() * self._SCREEN_CLAMP_RATIO), 320)
        max_h = max(int(available.height() * self._SCREEN_CLAMP_RATIO), 240)

        minimum = self.minimumSize()
        if minimum.width() > max_w or minimum.height() > max_h:
            self.setMinimumSize(min(minimum.width(), max_w), min(minimum.height(), max_h))
        if self.width() > max_w or self.height() > max_h:
            self.resize(min(self.width(), max_w), min(self.height(), max_h))

        # Position fallback: keeps the whole window from appearing outside the work area (QDialog may still centre on the parent window afterwards,
        # but by then the window size has been clamped, so the centred result is certain to be on screen).
        geo = self.geometry()
        x = min(max(geo.x(), available.left()), max(available.right() - geo.width() + 1, available.left()))
        y = min(max(geo.y(), available.top()), max(available.bottom() - geo.height() + 1, available.top()))
        if x != geo.x() or y != geo.y():
            self.move(x, y)


DialogCode = QDialog.DialogCode
