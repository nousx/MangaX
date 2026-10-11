"""
Wheel event filter

One convention: a slider, spin box or drop-down does not respond to the wheel while it has no keyboard focus,
and the event passes through to the parent scroll area; once it has keyboard focus (click / Tab), the control keeps its default wheel behaviour.
"""

from PyQt6.QtCore import QAbstractAnimation, QEvent, QObject, Qt
from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox, QSlider, QWidget
from qfluentwidgets import ComboBox
from qfluentwidgets.components.widgets.combo_box import ComboBoxMenu

# Control types whose wheel behaviour is taken over (note: QScrollBar is not among them; scroll bars must always respond to the wheel)
_WHEEL_TARGET_TYPES = (QAbstractSpinBox, QComboBox, QSlider, ComboBox)


def _stop_popup_animation(menu: QWidget) -> None:
    """Stop Fluent's popup animation before a menu is closed or destroyed.

    Fluent Widgets keeps the animation object on the menu, while the animation
    targets the menu itself.  Closing a ``WA_DeleteOnClose`` menu during its
    entrance animation can therefore leave Qt trying to update a deleted target.
    """
    manager = getattr(menu, "aniManager", None)
    if manager is None:
        return

    # On Windows a WA_DeleteOnClose popup can be destroyed on mouse press,
    # before the combo box receives the matching mouse release.  The manager
    # then still holds a Python wrapper whose underlying animation is gone,
    # and any use of it raises, including a truth test (`a or b`), which Qt
    # answers by asking the deleted object for its length.
    try:
        animation = getattr(manager, "aniGroup", None)
        if animation is None:
            animation = getattr(manager, "ani", None)
        if animation is None:
            return
        if animation.state() != QAbstractAnimation.State.Stopped:
            animation.stop()
    except RuntimeError:
        pass


class _SafeComboBoxMenu(ComboBoxMenu):
    """Combo menu that does not outlive its target animation."""

    def closeEvent(self, event):
        _stop_popup_animation(self)
        super().closeEvent(event)


class TopLevelComboBox(ComboBox):
    """Fluent combo box whose popup is owned by the real top-level window.

    Most controls live below a ``QStackedWidget``.  Using the control itself as
    a ``QMenu`` parent makes Qt try to activate the stacked page's
    ``QWidgetWindow`` on Windows, producing a warning and occasionally a
    misplaced popup.
    """

    def _popup_parent(self) -> QWidget:
        window = self.window()
        return window if window is not None and window.isWindow() else self

    def _createComboMenu(self):
        return _SafeComboBoxMenu(self._popup_parent())

    def _closeComboMenu(self):
        menu = self.dropMenu
        try:
            if menu is not None:
                _stop_popup_animation(menu)
        finally:
            # Always let the base implementation clear a stale dropMenu
            # reference, otherwise every later click is treated as a close.
            try:
                super()._closeComboMenu()
            except RuntimeError:
                # The popup itself was already destroyed.
                self.dropMenu = None


class NoWheelComboBox(TopLevelComboBox):
    """Drop-down with wheel events disabled"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._editable = False

    def addItem(self, text: str, userData=None):
        super().addItem(text, userData=userData)

    def insertItem(self, index: int, text: str, userData=None):
        super().insertItem(index, text, userData=userData)

    def setEditable(self, editable: bool):
        self._editable = bool(editable)

    def isEditable(self) -> bool:
        return self._editable

    def mouseReleaseEvent(self, event):
        super(ComboBox, self).mouseReleaseEvent(event)
        self.showPopup()

    def showPopup(self):
        self._toggleComboMenu()

    def hidePopup(self):
        self._closeComboMenu()

    def wheelEvent(self, event):
        """Ignore wheel events completely"""
        event.ignore()


class WheelEventFilter(QObject):
    """
    Wheel event filter

    Without focus, the control's own wheel handling is stopped and the event is left unaccepted - Qt's wheel
    propagation passes an unaccepted event on to the parent (the scroll area); with focus nothing is done.
    """

    def eventFilter(self, obj, event):
        if (
            event.type() == QEvent.Type.Wheel
            and isinstance(obj, QWidget)
            and not obj.hasFocus()
        ):
            event.ignore()
            return True
        return super().eventFilter(obj, event)


def _demote_wheel_focus(widget: QWidget):
    """WheelFocus → StrongFocus: the wheel no longer takes the focus, while click / Tab still can."""
    if widget.focusPolicy() == Qt.FocusPolicy.WheelFocus:
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


def install_wheel_filter(widget: QWidget) -> WheelEventFilter:
    """
    Install the wheel event filter on the given widget and all its slider, spin box and drop-down children

    Args:
        widget: the top-level widget the filter is installed on

    Returns:
        The installed filter instance (its parent is widget, and it is destroyed with it)
    """
    wheel_filter = WheelEventFilter(widget)

    targets = {
        child
        for target_type in _WHEEL_TARGET_TYPES
        for child in widget.findChildren(target_type)
    }
    if isinstance(widget, _WHEEL_TARGET_TYPES):
        targets.add(widget)

    for target in targets:
        target.installEventFilter(wheel_filter)
        _demote_wheel_focus(target)

    return wheel_filter
