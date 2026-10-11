"""The font picker popup is a see-through window, so it has to paint its own background."""
import _bootstrap  # noqa: F401, I001

import pytest
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication, QWidget

from utils.thai_font_menu import ThaiFontMenu


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def menu(application):
    entries = [("Sample Sans - Regular", "Sample Sans::Regular", "sample sans regular")]
    popup = ThaiFontMenu(entries, "Search fonts...", "", False)
    popup.setFixedSize(430, 550)
    popup._container.setGeometry(QRect(15, 10, 400, 530))
    popup._content_layout.activate()
    yield popup
    popup.deleteLater()


def rendered(widget: QWidget) -> QImage:
    image = QImage(widget.size(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    widget.render(painter, QPoint(), flags=QWidget.RenderFlag.DrawChildren)
    painter.end()
    return image


def test_popup_should_be_opaque_behind_the_controls_outside_the_list(menu):
    image = rendered(menu)

    for control in (menu.preview, menu.search_edit, menu.apply_button):
        centre = control.mapTo(menu, control.rect().center())
        # Just left of the control: inside the panel, but not painted by the control itself.
        beside = QPoint(control.mapTo(menu, QPoint()).x() - 3, centre.y())
        assert image.pixelColor(beside).alpha() == 255, control


def test_popup_should_stay_see_through_outside_its_panel(menu):
    image = rendered(menu)

    assert image.pixelColor(0, 0).alpha() == 0
