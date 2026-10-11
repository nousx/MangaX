"""
Canvas layer of paste overlays: display items, selection decoration, interaction and drag-and-drop import.

Coordinate conventions: scene unit == source image pixel (the base image item is not scaled); ``center_x/y`` and
``width/height`` of an overlay are values at the resolution of the source image and are used directly for placing it in the scene.
z order: overlays start at 50 (above the base image at 2 and the inpaint preview, below the region text boxes at 100).

The selection decoration (dashed frame + four corner handles + rotation handle) is drawn in one go by a single custom-painted child item,
in the style of the text box (RegionTextItem). Every data change goes through EditorController (undoable).
"""

from __future__ import annotations

import math
import os

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QCursor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsSceneMouseEvent,
)

from manga_translator.image_formats import SUPPORTED_IMAGE_EXTENSIONS

from editor.paste_overlay_state import (
    png_base64_to_rgba_overlay,
    rgba_overlay_to_png_base64,
)

from .graphics_items import (
    _editor_pen,
    _fluent_accent,
    _fluent_surface,
    _shadow_color,
)

_PASTE_BASE_Z = 50
_PASTE_MAX_DIMENSION = 2048
_IMAGE_SUFFIXES = set(SUPPORTED_IMAGE_EXTENSIONS)
_ROTATE_OFFSET_PX = 40.0


def _rgba_to_qimage(rgba):
    height, width = rgba.shape[:2]
    image = QImage(
        rgba.data,
        width,
        height,
        rgba.strides[0],
        QImage.Format.Format_RGBA8888,
    )
    return image.copy()


def _view_lod_of(item) -> float:
    """Scale of the scene view (screen pixels / scene unit)."""
    try:
        scene = item.scene()
        if scene is not None and scene.views():
            return max(abs(scene.views()[0].transform().m11()), 0.01)
    except (RuntimeError, AttributeError):
        pass
    return 1.0


class _PasteOverlaySelectionItem(QGraphicsItem):
    """Selection decoration of a paste overlay: dashed frame + the same corner and rotation handles as a text box (painted in one go, no flicker)."""

    def __init__(self, overlay_item: "PasteOverlayItem"):
        super().__init__(overlay_item)
        self._overlay_item = overlay_item
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setZValue(15)

    def shape(self) -> QPainterPath:
        # Decoration only: takes no part in hit testing; all mouse events land on the overlay itself (whose shape already covers the extended area)
        return QPainterPath()

    def _geometry(self) -> tuple[float, float, float]:
        lod = _view_lod_of(self)
        pixmap = self._overlay_item.pixmap()
        width = float(pixmap.width())
        height = float(pixmap.height())
        if width <= 0 or height <= 0:
            lod, width, height = 1.0, 1.0, 1.0
        return lod, width, height

    def boundingRect(self) -> QRectF:
        _, width, height = self._geometry()
        lod = _view_lod_of(self)
        pad = 10.0 / lod
        top_pad = _ROTATE_OFFSET_PX / lod + 12.0 / lod + pad
        return QRectF(-pad, -top_pad, width + pad * 2.0, height + top_pad + pad)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802
        lod, width, height = self._geometry()
        pw = 1.15 / lod
        frame = QRectF(0.0, 0.0, width, height)
        accent = _fluent_accent(235)
        surface = _fluent_surface(246)

        # Dashed outer frame (double stroke: shadow + accent colour), matching the selected state of text boxes
        path = QPainterPath()
        path.addRect(frame)
        painter.setBrush(QBrush(_fluent_accent(16)))
        painter.setPen(_editor_pen(_shadow_color(135), 4.0 / lod))
        painter.drawPath(path)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.setPen(
            _editor_pen(accent, 1.6 / lod, Qt.PenStyle.DashLine)
        )
        painter.drawPath(path)

        # Scale handles at the four corners (the same rounded squares as the white box of text boxes)
        corner_hs = 13.0 / lod
        corners = (
            QPointF(0.0, 0.0),
            QPointF(width, 0.0),
            QPointF(0.0, height),
            QPointF(width, height),
        )
        radius = min(3.5 / lod, corner_hs / 2.0)
        for corner in corners:
            rect = QRectF(
                corner.x() - corner_hs / 2.0,
                corner.y() - corner_hs / 2.0,
                corner_hs,
                corner_hs,
            )
            painter.setBrush(QBrush(surface))
            painter.setPen(_editor_pen(_shadow_color(120), pw * 2.3))
            painter.drawRoundedRect(rect, radius, radius)
            painter.setPen(_editor_pen(_fluent_accent(238), pw * 1.15))
            painter.drawRoundedRect(
                rect.adjusted(pw * 0.45, pw * 0.45, -pw * 0.45, -pw * 0.45),
                radius,
                radius,
            )

        # Rotation handle at the top (as for text boxes: connecting rod + ring + dot)
        rot_hs = 14.0 / lod
        rotate_center = QPointF(width / 2.0, -_ROTATE_OFFSET_PX / lod)
        painter.setPen(_editor_pen(_shadow_color(125), pw * 3.0))
        painter.drawLine(QPointF(width / 2.0, 0.0), rotate_center)
        painter.setPen(_editor_pen(_fluent_accent(205), pw * 1.45))
        painter.drawLine(QPointF(width / 2.0, 0.0), rotate_center)
        rot_rect = QRectF(
            rotate_center.x() - rot_hs / 2.0,
            rotate_center.y() - rot_hs / 2.0,
            rot_hs,
            rot_hs,
        )
        painter.setBrush(QBrush(_fluent_surface(245)))
        painter.setPen(_editor_pen(_shadow_color(110), pw * 2.8))
        painter.drawEllipse(rot_rect)
        painter.setPen(_editor_pen(_fluent_accent(235), pw * 1.2))
        painter.drawEllipse(
            rot_rect.adjusted(pw * 0.45, pw * 0.45, -pw * 0.45, -pw * 0.45)
        )
        dot = rot_hs * 0.32
        painter.setBrush(QBrush(_fluent_accent(225)))
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.drawEllipse(
            QRectF(
                rotate_center.x() - dot / 2.0,
                rotate_center.y() - dot / 2.0,
                dot,
                dot,
            )
        )


class PasteOverlayItem(QGraphicsPixmapItem):
    """A single paste overlay: pixmap, geometry and opacity are driven by the overlay dictionary; it has its own selection and drag interaction."""

    def __init__(self, overlay: dict, view=None):
        super().__init__()
        self.overlay = dict(overlay)
        self._paste_view = view
        self._selection_item: _PasteOverlaySelectionItem | None = None
        self._drag_mode: str | None = None
        self._drag_start_scene = QPointF()
        self._drag_start_center = (0.0, 0.0)
        self._drag_start_size = (0.0, 0.0)
        self._drag_start_rotation = 0.0
        self._drag_start_dist = 1.0
        self._drag_start_angle = 0.0
        self._source_pixmap: QPixmap = QPixmap()
        self.setAcceptHoverEvents(True)
        self._rebuild()

    # ------------------------------------------------------------------
    # Base data / view sync
    # ------------------------------------------------------------------

    def overlay_id(self) -> str:
        return str(self.overlay.get("id", ""))

    def set_view(self, view) -> None:
        self._paste_view = view

    def update_overlay(self, overlay: dict) -> None:
        self.overlay = dict(overlay)
        self._rebuild()

    def set_selected(self, selected: bool) -> None:
        if selected:
            if self._selection_item is None:
                self.prepareGeometryChange()
                self._selection_item = _PasteOverlaySelectionItem(self)
        elif self._selection_item is not None:
            self.prepareGeometryChange()
            try:
                scene = self._selection_item.scene()
                if scene:
                    scene.removeItem(self._selection_item)
            except (RuntimeError, AttributeError):
                pass
            self._selection_item = None

    def _view_lod(self) -> float:
        return _view_lod_of(self)

    def boundingRect(self) -> QRectF:
        pixmap = self.pixmap()
        width = float(pixmap.width())
        height = float(pixmap.height())
        if self._selection_item is not None:
            lod = self._view_lod()
            pad = 10.0 / lod
            top_pad = _ROTATE_OFFSET_PX / lod + 12.0 / lod + pad
            return QRectF(-pad, -top_pad, width + pad * 2.0, height + top_pad + pad)
        return QRectF(0.0, 0.0, width, height)

    def _rebuild(self) -> None:
        overlay = self.overlay
        pixmap = QPixmap()
        image_b64 = overlay.get("image", "")
        if image_b64:
            rgba = png_base64_to_rgba_overlay(image_b64)
            if rgba is not None:
                qimage = _rgba_to_qimage(rgba)
                if not qimage.isNull():
                    raw = QPixmap.fromImage(qimage)
                    if not raw.isNull():
                        pixmap = raw
        # Cache the decoded original: scaling always resamples from the original, so detail is not lost by resampling frame after frame while dragging
        self._source_pixmap = pixmap
        self.prepareGeometryChange()
        self.setPixmap(pixmap)
        self._apply_geometry()

    def _apply_geometry(self, *, rasterize: bool = True) -> None:
        overlay = self.overlay
        # The decoded original is only the source for resampling; the final size is the width/height of the overlay
        source = (
            self._source_pixmap
            if not self._source_pixmap.isNull()
            else self.pixmap()
        )
        if source.isNull() or source.width() <= 0 or source.height() <= 0:
            self.setVisible(False)
            return

        target_width = max(
            1, int(round(float(overlay.get("width", source.width()))))
        )
        target_height = max(
            1, int(round(float(overlay.get("height", source.height()))))
        )
        # While scaling by drag (rasterize=False) the pixmap is not resampled; the target/current ratio is folded into
        # the item transform, and the final rasterisation happens only once, on the rebuild after release, to avoid allocating a large pixmap every frame
        display = self.pixmap()
        if rasterize:
            if (
                display.isNull()
                or display.width() != target_width
                or display.height() != target_height
            ):
                display = source.scaled(
                    target_width,
                    target_height,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.prepareGeometryChange()
                self.setPixmap(display)
        else:
            if display.isNull() or display.width() <= 0 or display.height() <= 0:
                display = source.scaled(
                    target_width,
                    target_height,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.prepareGeometryChange()
                self.setPixmap(display)
                rasterize = True

        # The transform anchor has to use the "currently displayed size"
        width = display.width()
        height = display.height()
        if width <= 0 or height <= 0:
            self.setVisible(False)
            return

        center_x = float(overlay.get("center_x", 0.0))
        center_y = float(overlay.get("center_y", 0.0))
        rotation = float(overlay.get("rotation", 0.0))
        flip_h = -1.0 if overlay.get("flip_h") else 1.0
        flip_v = -1.0 if overlay.get("flip_v") else 1.0

        scale_x = 1.0
        scale_y = 1.0
        if not rasterize:
            scale_x = target_width / float(width)
            scale_y = target_height / float(height)

        transform = QTransform()
        transform.translate(center_x, center_y)
        transform.rotate(rotation)
        transform.scale(flip_h * scale_x, flip_v * scale_y)
        transform.translate(-width / 2.0, -height / 2.0)
        self.setTransform(transform)

        try:
            opacity = float(overlay.get("opacity", 1.0))
        except (TypeError, ValueError):
            opacity = 1.0
        self.setOpacity(max(0.0, min(1.0, opacity)))
        self.setZValue(_PASTE_BASE_Z + int(overlay.get("z", 0)))
        self.setVisible(bool(overlay.get("visible", True)))

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        pixmap = self.pixmap()
        if pixmap.isNull() or pixmap.width() <= 0 or pixmap.height() <= 0:
            return path
        width = float(pixmap.width())
        height = float(pixmap.height())
        path.addRect(QRectF(0.0, 0.0, width, height))
        if self._selection_item is not None:
            lod = self._view_lod()
            pad = 8.0 / lod
            path.addRect(
                QRectF(
                    -pad,
                    -_ROTATE_OFFSET_PX / lod - 14.0 / lod - pad,
                    width + pad * 2.0,
                    height + _ROTATE_OFFSET_PX / lod + 14.0 / lod + pad * 2.0,
                )
            )
        return path

    # ------------------------------------------------------------------
    # Mouse interaction: move / corner scale / rotate (commits go through the undo stack)
    # ------------------------------------------------------------------

    def _hover_cursor_for(self, local_pos: QPointF):
        """The Qt cursor for a hit result (as for text boxes: corner = diagonal resize, rotation = four-way move)."""
        if self._selection_item is None:
            return None
        pixmap = self.pixmap()
        width = float(pixmap.width())
        height = float(pixmap.height())
        if width <= 0 or height <= 0:
            return None
        lod = self._view_lod()
        hit = 10.0 / lod
        rotate_center = QPointF(width / 2.0, -_ROTATE_OFFSET_PX / lod)
        if (
            math.hypot(
                local_pos.x() - rotate_center.x(), local_pos.y() - rotate_center.y()
            )
            <= hit + 7.0 / lod
        ):
            return Qt.CursorShape.SizeAllCursor
        corners = (
            QPointF(0.0, 0.0),
            QPointF(width, 0.0),
            QPointF(0.0, height),
            QPointF(width, height),
        )
        for corner in corners:
            if math.hypot(local_pos.x() - corner.x(), local_pos.y() - corner.y()) <= hit:
                top_left = local_pos.x() <= width / 2.0 and local_pos.y() <= height / 2.0
                bottom_right = local_pos.x() > width / 2.0 and local_pos.y() > height / 2.0
                return (
                    Qt.CursorShape.SizeFDiagCursor
                    if top_left or bottom_right
                    else Qt.CursorShape.SizeBDiagCursor
                )
        return None

    def _apply_hover_cursor(self, shape) -> None:
        self.setCursor(QCursor(shape))
        view = self._paste_view
        if view is not None and view.model.get_active_tool() == "select":
            view.viewport().setCursor(QCursor(shape))

    def _clear_hover_cursor(self) -> None:
        self.unsetCursor()
        view = self._paste_view
        if view is not None and view.model.get_active_tool() == "select":
            view.viewport().unsetCursor()

    def hoverMoveEvent(self, event) -> None:  # noqa: N802 - Qt API naming
        try:
            view = self._paste_view
            if (
                self._selection_item is None
                or not self.isVisible()
                or view is None
                or view.model.get_active_tool() != "select"
            ):
                self._clear_hover_cursor()
                super().hoverMoveEvent(event)
                return
            shape = self._hover_cursor_for(QPointF(event.pos()))
            if shape is None:
                self._clear_hover_cursor()
            else:
                self._apply_hover_cursor(shape)
            super().hoverMoveEvent(event)
        except Exception:
            pass

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 - Qt API naming
        self._clear_hover_cursor()
        super().hoverLeaveEvent(event)

    def _drag_enabled(self) -> bool:
        view = self._paste_view
        if view is None or not self.isVisible() or self.pixmap().isNull():
            return False
        model = getattr(view, "model", None)
        return model is None or model.get_active_tool() == "select"

    def _hit_mode(self, local_pos: QPointF) -> str | None:
        """Hit test in local coordinates: rotation handle / four corner scale handles / move inside."""
        pixmap = self.pixmap()
        width = float(pixmap.width())
        height = float(pixmap.height())
        lod = self._view_lod()
        hit = 10.0 / lod
        rotate_center = QPointF(width / 2.0, -_ROTATE_OFFSET_PX / lod)
        if (
            math.hypot(local_pos.x() - rotate_center.x(), local_pos.y() - rotate_center.y())
            <= hit + 7.0 / lod
        ):
            return "rotate"
        for corner in (
            QPointF(0.0, 0.0),
            QPointF(width, 0.0),
            QPointF(0.0, height),
            QPointF(width, height),
        ):
            if math.hypot(local_pos.x() - corner.x(), local_pos.y() - corner.y()) <= hit:
                return "resize"
        return "move"

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._drag_enabled():
            super().mousePressEvent(event)
            return
        view = self._paste_view
        if view is not None and getattr(view, "model", None) is not None:
            view.model.set_selection([])
        # Note: the "selection frame" is not added here right away. Creating it triggers prepareGeometryChange,
        # and changing geometry during the dispatch of the mouse press can make the whole overlay jump for one frame at the press
        # and return on release - a flicker. The selected state is added in mouseRelease instead (rebuilt after the commit).
        local_pos = self.mapFromScene(event.scenePos())
        self._drag_mode = self._hit_mode(local_pos)
        self._drag_start_scene = event.scenePos()
        self._drag_start_center = (
            float(self.overlay.get("center_x", 0.0)),
            float(self.overlay.get("center_y", 0.0)),
        )
        self._drag_start_size = (
            float(self.overlay.get("width", 1.0)),
            float(self.overlay.get("height", 1.0)),
        )
        self._drag_start_rotation = float(self.overlay.get("rotation", 0.0))
        delta = event.scenePos() - QPointF(
            self._drag_start_center[0], self._drag_start_center[1]
        )
        self._drag_start_dist = max(1.0, math.hypot(delta.x(), delta.y()))
        self._drag_start_angle = math.degrees(math.atan2(delta.y(), delta.x()))
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._drag_mode is None or not self._drag_enabled():
            super().mouseMoveEvent(event)
            return
        overlay = self.overlay
        center = QPointF(
            float(overlay.get("center_x", 0.0)),
            float(overlay.get("center_y", 0.0)),
        )
        if self._drag_mode == "move":
            delta = event.scenePos() - self._drag_start_scene
            overlay["center_x"] = self._drag_start_center[0] + delta.x()
            overlay["center_y"] = self._drag_start_center[1] + delta.y()
        elif self._drag_mode == "rotate":
            delta = event.scenePos() - center
            current_angle = math.degrees(math.atan2(delta.y(), delta.x()))
            overlay["rotation"] = self._drag_start_rotation + (
                current_angle - self._drag_start_angle
            )
        elif self._drag_mode == "resize":
            delta = event.scenePos() - center
            distance = math.hypot(delta.x(), delta.y())
            factor = distance / self._drag_start_dist
            # Guard against runaway enlargement: limited to 4 times the decode limit of the source overlay, so one drag cannot request a huge pixmap
            max_side = float(_PASTE_MAX_DIMENSION * 4)
            longest = max(self._drag_start_size) * factor
            if longest > max_side:
                factor *= max_side / longest
            overlay["width"] = max(2.0, self._drag_start_size[0] * factor)
            overlay["height"] = max(2.0, self._drag_start_size[1] * factor)
        # While scaling by drag only the transform changes, without resampling; moving and rotating still refresh from the geometry at once
        self._apply_geometry(rasterize=self._drag_mode != "resize")
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        mode = self._drag_mode
        self._drag_mode = None
        if mode is None or not self._drag_enabled():
            super().mouseReleaseEvent(event)
            return
        overlay_id = self.overlay_id()
        view = self._paste_view
        overlay = self.overlay
        if mode == "move":
            patch = {
                "center_x": float(overlay.get("center_x", 0.0)),
                "center_y": float(overlay.get("center_y", 0.0)),
            }
        elif mode == "rotate":
            patch = {"rotation": float(overlay.get("rotation", 0.0))}
        else:
            patch = {
                "width": float(overlay.get("width", 0.0)),
                "height": float(overlay.get("height", 0.0)),
            }
        # After the commit the model rebuilds all overlay items (the current item will be removed), so self must not be touched afterwards
        if view is not None and getattr(view, "controller", None) is not None:
            view.controller.update_paste_overlay(overlay_id, patch)
            view.select_paste_overlay(overlay_id)
        event.accept()


class GraphicsViewPasteOverlayMixin:
    """Sync of paste overlays on the canvas: listens to the model signals to rebuild the visible items, and supports importing a PNG by drag and drop."""

    def _rebuild_paste_overlay_items(self) -> None:
        items = getattr(self, "_paste_overlay_items", None)
        if items is None:
            return
        existing = {item.overlay_id(): item for item in items}
        items.clear()

        if self._image_item is None:
            self._remove_paste_items(existing.values())
            self._selected_paste_overlay_id = None
            return
        overlays = self.model.get_paste_overlays() if self.model else []
        selected_id = getattr(self, "_selected_paste_overlay_id", None)
        valid_ids = {str(item.get("id", "")) for item in overlays}
        if selected_id not in valid_ids:
            self._selected_paste_overlay_id = None
            selected_id = None
        new_items = []
        for overlay in overlays:
            overlay_id = str(overlay.get("id", ""))
            old_item = existing.pop(overlay_id, None)
            if (
                old_item is not None
                and old_item.overlay.get("image") == overlay.get("image")
            ):
                # The image did not change: reuse the item and only update geometry and properties, without rebuilding or decoding
                old_item.overlay = dict(overlay)
                old_item._apply_geometry()
                item = old_item
            else:
                if old_item is not None:
                    self._remove_paste_items([old_item])
                item = PasteOverlayItem(overlay, self)
                self.scene.addItem(item)
            item.set_selected(item.overlay_id() == selected_id)
            new_items.append(item)
        # Remove overlay items that no longer exist from the scene
        self._remove_paste_items(existing.values())

        # Add to the scene again in list order: Qt decides the stacking of items with equal z by insertion order,
        # and a reused item keeps its old order, so the insertion order is rebuilt here to keep the canvas and
        # the export compositing (compose_paste_overlays: ascending z, list order for equal z) the same
        for item in new_items:
            try:
                if item.scene():
                    item.scene().removeItem(item)
            except (RuntimeError, AttributeError):
                pass
        for item in new_items:
            self.scene.addItem(item)

        self._paste_overlay_items = new_items
        self.scene.update()

    @staticmethod
    def _remove_paste_items(items) -> None:
        for item in items:
            try:
                if item is not None and item.scene():
                    item.scene().removeItem(item)
            except (RuntimeError, AttributeError):
                pass

    def _clear_paste_overlay_items(self) -> None:
        items = getattr(self, "_paste_overlay_items", None)
        if items is None:
            return
        for item in list(items):
            try:
                if item.scene():
                    self.scene.removeItem(item)
            except (RuntimeError, AttributeError):
                pass
        items.clear()
        self._selected_paste_overlay_id = None

    def select_paste_overlay(self, overlay_id: str) -> None:
        """Select a single paste overlay (canvas-level state, outside the region selection system)."""
        self._selected_paste_overlay_id = overlay_id
        if hasattr(self, "model") and self.model:
            self.model.set_selection([])
        for item in self._paste_overlay_items:
            item.set_selected(item.overlay_id() == overlay_id)
        self.scene.update()

    def clear_paste_overlay_selection(self) -> None:
        if not getattr(self, "_selected_paste_overlay_id", None):
            return
        self._selected_paste_overlay_id = None
        for item in self._paste_overlay_items:
            item.set_selected(False)
        self.scene.update()

    def _on_model_selection_changed_for_paste_overlays(
        self, selected_indices: list
    ) -> None:
        """When the selection of text regions in the model is not empty, clear the overlay selection automatically, keeping the two interactions exclusive."""
        if selected_indices:
            self.clear_paste_overlay_selection()

    def clear_paste_overlay_selection_for_press(self, event) -> None:
        """When a click hits something that is not a paste overlay (a text box, empty space and so on), hide the overlay handles."""
        if not getattr(self, "_selected_paste_overlay_id", None):
            return
        try:
            top_item = self.itemAt(event.position().toPoint())
        except AttributeError:
            top_item = self.itemAt(event.pos())
        while top_item is not None:
            if isinstance(top_item, PasteOverlayItem):
                return
            top_item = top_item.parentItem()
        self.clear_paste_overlay_selection()

    def delete_selected_paste_overlay(self) -> bool:
        overlay_id = getattr(self, "_selected_paste_overlay_id", None)
        controller = getattr(self, "controller", None)
        if not overlay_id or controller is None:
            return False
        if controller.remove_paste_overlay(overlay_id):
            self._selected_paste_overlay_id = None
            return True
        return False

    def keyPressEvent(self, event):  # noqa: N802 - Qt API naming
        if event.key() == Qt.Key.Key_Delete and self.delete_selected_paste_overlay():
            event.accept()
            return
        super().keyPressEvent(event)

    def on_paste_overlays_changed(self, overlays=None) -> None:
        self._rebuild_paste_overlay_items()

    # --- Importing PNG by drag and drop ---
    def _dropped_image_paths(self, event):
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return []
        paths = []
        for url in mime.urls():
            if url is None or not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if os.path.splitext(path)[1].lower() in _IMAGE_SUFFIXES:
                paths.append(path)
        return paths

    def dragEnterEvent(self, event):  # noqa: N802 - Qt API naming
        if self._image_item is None or not self._dropped_image_paths(event):
            event.ignore()
            return
        event.acceptProposedAction()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt API naming
        if self._image_item is None or not self._dropped_image_paths(event):
            event.ignore()
            return
        event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802 - Qt API naming
        if self._image_item is None:
            event.ignore()
            return
        controller = getattr(self, "controller", None)
        if controller is None or getattr(self.model, "get_source_image_path", None) is None:
            event.ignore()
            return
        paths = self._dropped_image_paths(event)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()

        scene_pos = self.mapToScene(event.position().toPoint())
        try:
            import numpy as np
            from PIL import Image

            image_path = paths[0]
            rgba = None
            with Image.open(image_path) as source:
                source.load()
                rgba = source.convert("RGBA")
                array = np.array(rgba, dtype=np.uint8, copy=True)
            if rgba is not None and getattr(rgba, "close", None):
                rgba.close()
            if array.ndim != 3 or array.shape[2] != 4:
                event.ignore()
                return
            height, width = array.shape[:2]
            longest = max(width, height)
            if longest > _PASTE_MAX_DIMENSION:
                import cv2

                scale = _PASTE_MAX_DIMENSION / longest
                array = cv2.resize(
                    array,
                    (
                        max(1, int(round(width * scale))),
                        max(1, int(round(height * scale))),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            overlay = {
                "name": os.path.basename(image_path),
                "center_x": float(scene_pos.x()),
                "center_y": float(scene_pos.y()),
                "width": float(array.shape[1]),
                "height": float(array.shape[0]),
                "image": rgba_overlay_to_png_base64(array),
            }
            if controller.add_paste_overlay(overlay):
                current = self.model.get_paste_overlays()
                if current:
                    self.select_paste_overlay(current[-1]["id"])
        except Exception as error:
            self.logger.warning("Failed to import overlay: %s", error)
