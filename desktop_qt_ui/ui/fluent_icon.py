import os

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QIconEngine, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import isDarkTheme, themeColor
from qfluentwidgets.common.icon import drawSvgIcon, writeSvg

from utils.resource_helper import resource_path

# (SVG path, colour) -> SVG bytes with the fill colour rewritten. writeSvg has to parse and rewrite
# the XML every time, so a small cache is kept here; after a theme switch the colour changes and the keys expire by themselves.
_SVG_BYTES_CACHE: dict[tuple[str, str], bytes] = {}
_SVG_BYTES_CACHE_MAX = 256


def _themed_svg_bytes(icon_path: str, color_name: str) -> bytes:
    key = (icon_path, color_name)
    cached = _SVG_BYTES_CACHE.get(key)
    if cached is None:
        svg = writeSvg(icon_path, fill=color_name)
        cached = svg.encode() if svg else b""
        if len(_SVG_BYTES_CACHE) >= _SVG_BYTES_CACHE_MAX:
            _SVG_BYTES_CACHE.clear()
        _SVG_BYTES_CACHE[key] = cached
    return cached


class _ThemedFluentSvgIconEngine(QIconEngine):
    def __init__(self, icon_path: str, color_token: str):
        super().__init__()
        self.icon_path = icon_path
        self.color_token = color_token

    def paint(self, painter: QPainter, rect, mode, state):
        color = _resolve_icon_color(self.color_token).name()
        svg_bytes = _themed_svg_bytes(self.icon_path, color)
        if not svg_bytes:
            QIcon(self.icon_path).paint(painter, rect, Qt.AlignmentFlag.AlignCenter, mode, state)
            return

        painter.save()
        if mode == QIcon.Mode.Disabled:
            painter.setOpacity(0.5)
        elif mode == QIcon.Mode.Selected:
            painter.setOpacity(0.7)

        drawSvgIcon(svg_bytes, painter, rect)
        painter.restore()

    def clone(self):
        return _ThemedFluentSvgIconEngine(self.icon_path, self.color_token)

    def pixmap(self, size, mode, state):
        app = QApplication.instance()
        scale = app.devicePixelRatio() if app is not None else 1.0
        return self.scaledPixmap(size, mode, state, scale)

    def scaledPixmap(self, size, mode, state, scale):
        """按目标 DPR 渲染大图并 setDevicePixelRatio，高 DPI 下不发糊。"""
        scale = max(1.0, float(scale))
        device_size = QSize(
            max(1, round(size.width() * scale)),
            max(1, round(size.height() * scale)),
        )
        image = QImage(device_size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        image.setDevicePixelRatio(scale)
        pixmap = QPixmap.fromImage(image)

        painter = QPainter(pixmap)
        # Drawing coordinates are in device-independent pixels; QPainter scales by the DPR of the pixmap
        self.paint(painter, QRect(0, 0, size.width(), size.height()), mode, state)
        painter.end()
        return pixmap


def themed_fluent_svg_icon(filename: str, color_token: str = "text_primary") -> QIcon:
    icon_path = resource_path(os.path.join("desktop_qt_ui", "ui", "icons", filename))
    return QIcon(_ThemedFluentSvgIconEngine(icon_path, color_token))


def _resolve_icon_color(color_role: str) -> QColor:
    role = (color_role or "").lower()
    if role in {"accent", "primary", "theme", "theme_color", "btn_primary_bg"}:
        return QColor(themeColor())
    if role in {"muted", "secondary", "text_secondary", "text_muted"}:
        return QColor(255, 255, 255, 179) if isDarkTheme() else QColor(0, 0, 0, 153)
    if role in {"disabled", "text_disabled"}:
        return QColor(255, 255, 255, 92) if isDarkTheme() else QColor(0, 0, 0, 92)
    return QColor(255, 255, 255) if isDarkTheme() else QColor(31, 31, 31)
