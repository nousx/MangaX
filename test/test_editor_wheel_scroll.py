import _bootstrap  # noqa: F401, I001

import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from ui.editor.graphics_view_input import GraphicsViewInputMixin


class _View(GraphicsViewInputMixin, QGraphicsView):
    def _update_cursor(self):
        pass

    def _emit_view_state_changed(self):
        pass

    # The editor handlers need the full view state; an exception raised
    # inside a Qt virtual aborts the interpreter.
    def focusOutEvent(self, event):
        QGraphicsView.focusOutEvent(self, event)

    def enterEvent(self, event):
        QGraphicsView.enterEvent(self, event)

    def leaveEvent(self, event):
        QGraphicsView.leaveEvent(self, event)


@pytest.fixture
def view():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene(0, 0, 4000, 4000)
    widget = _View(scene)
    widget.resize(400, 400)
    widget.show()
    app.processEvents()
    widget.verticalScrollBar().setValue(1000)
    widget.horizontalScrollBar().setValue(1000)
    yield widget
    widget.close()


def _wheel(view, delta_y, modifiers=Qt.KeyboardModifier.NoModifier):
    pos = QPointF(200, 200)
    event = QWheelEvent(
        pos,
        QPointF(view.viewport().mapToGlobal(pos.toPoint())),
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    view.wheelEvent(event)


def test_should_scroll_down_without_zooming_when_wheel_has_no_modifier(view):
    _wheel(view, -120)

    assert view.verticalScrollBar().value() > 1000
    assert view.transform().m11() == pytest.approx(1.0)


def test_should_scroll_up_without_zooming_when_wheel_has_no_modifier(view):
    _wheel(view, 120)

    assert view.verticalScrollBar().value() < 1000
    assert view.transform().m11() == pytest.approx(1.0)


def test_should_zoom_in_when_ctrl_wheel_up(view):
    _wheel(view, 120, Qt.KeyboardModifier.ControlModifier)

    assert view.transform().m11() == pytest.approx(1.15)


def test_should_zoom_out_when_ctrl_wheel_down(view):
    _wheel(view, -120, Qt.KeyboardModifier.ControlModifier)

    assert view.transform().m11() == pytest.approx(1 / 1.15)


def test_should_scroll_sideways_without_zooming_when_shift_wheel(view):
    _wheel(view, -120, Qt.KeyboardModifier.ShiftModifier)

    assert view.horizontalScrollBar().value() > 1000
    assert view.verticalScrollBar().value() == 1000
    assert view.transform().m11() == pytest.approx(1.0)
