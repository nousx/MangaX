"""
Graphics item of a region's text - the Qt Graphics Item layer.

Core design:
- for a RegionTextItem, pos() = the centre of the source region (world) and rotation() = angle
- local coordinate system: the centre of the source region is the origin, and line segments and the white box are in local coordinates
- the white box defines the boundary of the text render, and is a rectangle the user can adjust by hand
- the text pixmap is positioned with render_center (the world coordinates of the white box centre) as anchor

Note on the rotation bug fix:
  In the old implementation, update_text_pixmap positioned the top-left corner of the pixmap with mapFromScene(pos).
  But pos is the top-left corner of an axis-aligned rectangle in world coordinates, and after mapFromScene (which includes the inverse rotation)
  the pixmap centre drifted away from the white box centre in local coordinates.
  Fix: render_center → mapFromScene → local centre → minus half the size,
  so the pixmap is always positioned from the white box centre.
"""

import copy
import logging
import math
import traceback
from typing import List

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsSimpleTextItem,
    QStyle,
)
from qfluentwidgets import isDarkTheme, themeColor

from editor.desktop_ui_geometry import (
    calculate_center_scaled_rect,
    calculate_new_edge_on_drag,
    calculate_new_vertices_on_drag,
    rotate_point,
)
from editor.geometry_commit_pipeline import (
    build_rotate_region_data,
    build_white_frame_region_data,
)
from editor.region_geometry_state import RegionGeometryState

logger = logging.getLogger("manga_translator")

DRAWING_TOOLS = frozenset({"pen", "brush", "eraser", "paint", "paint_erase"})


def _clamp_alpha(alpha: int) -> int:
    return max(0, min(255, int(alpha)))


def _fluent_accent(alpha: int = 255) -> QColor:
    color = QColor(themeColor())
    if not color.isValid():
        color = QColor("#0F6CBD")
    color.setAlpha(_clamp_alpha(alpha))
    return color


def _fluent_surface(alpha: int = 242) -> QColor:
    color = QColor(43, 43, 43) if isDarkTheme() else QColor(255, 255, 255)
    color.setAlpha(_clamp_alpha(alpha))
    return color


def _fluent_neutral(alpha: int = 190) -> QColor:
    color = QColor(255, 255, 255) if isDarkTheme() else QColor(31, 31, 31)
    color.setAlpha(_clamp_alpha(alpha))
    return color


def _shadow_color(alpha: int = 130) -> QColor:
    return QColor(0, 0, 0, _clamp_alpha(alpha))


def _editor_pen(color: QColor, width: float, style=Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(color)
    pen.setWidthF(max(0.1, float(width)))
    pen.setStyle(style)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


# ======================================================================
# Helper classes
# ======================================================================


class TransparentPixmapItem(QGraphicsPixmapItem):
    """Pixmap item that is fully transparent to mouse events and does not block selection of the parent item."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

    def shape(self) -> QPainterPath:
        return QPainterPath()


# ======================================================================
# Main graphics item
# ======================================================================


class RegionTextItem(QGraphicsItemGroup):
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, region_data, region_index, geometry_callback, parent=None):
        super().__init__(parent)
        self.region_data = copy.deepcopy(region_data)
        self.region_index = region_index
        self.geometry_callback = geometry_callback

        self._image_item = None  # Reference to the image item, for coordinate conversion
        self._in_callback = False  # Guards against re-entrant callbacks

        # Single source of geometry state: geo
        self.geo = RegionGeometryState.from_region_data(self.region_data)

        # Qt coordinates: pos = centre of the source region, rotation = angle
        self.rotation_angle = float(self.geo.angle)
        self.visual_center = QPointF(
            float(self.geo.center[0]),
            float(self.geo.center[1]),
        )
        self.setPos(self.visual_center)
        self.setRotation(self.rotation_angle)
        self.setTransformOriginPoint(QPointF(0, 0))

        # Build the polygons in local coordinates
        self.polygons: List[QPolygonF] = []
        self._rebuild_qt_polygons()

        # Child item for the text pixmap
        self.text_item = TransparentPixmapItem(self)
        self.text_item.setZValue(-1)

        # Interaction state
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self._interaction_mode = "none"
        self._is_dragging = False

        # Drag state variables
        self._drag_handle_indices = None
        self._drag_start_pos = QPointF()
        self._drag_start_polygons: List[QPolygonF] = []
        self._batch_drag_peers: list = []  # Snapshot of the other selected items during a batch move
        self._drag_start_rotation = 0.0
        self._drag_raw_rotation = 0.0
        self._drag_start_visual_center = QPointF()
        self._drag_start_white_frame_local = None
        self._drag_start_scene_pos = QPointF()
        self._drag_start_pivot_scene = QPointF()
        self._drag_start_text_item_pos = QPointF()
        self._drag_start_white_handle_world = None
        self._drag_last_angle_rad = 0.0

        # Display
        self._polygons_visible = True
        self._show_white_box = True
        self._shape_path = None

        # Rotation angle label (scene level, created lazily)
        self._angle_label = None
        # Alignment guides (scene level)
        self._guide_lines = []
        self._spacing_labels: list = []
        # Controlled in one place by the persistent switch in the editor's general menu.
        self._snap_enabled = False
        # Snap threshold (pixels)
        self._snap_threshold = 1.0
        self._spacing_snap_threshold = 5.0

        self._setup_pens()

    # ------------------------------------------------------------------
    # Internal construction
    # ------------------------------------------------------------------

    def _rebuild_qt_polygons(self):
        """Rebuild the list of Qt QPolygonF objects from geo.polygons_local."""
        self.polygons = []
        for local_poly_data in self.geo.polygons_local:
            poly = QPolygonF()
            for x, y in local_poly_data:
                poly.append(QPointF(x, y))
            self.polygons.append(poly)

    def _setup_pens(self):
        self.white_pen = QPen(QColor("white"), 3)
        self.white_pen.setStyle(Qt.PenStyle.DashLine)
        self.white_brush = QBrush(Qt.BrushStyle.NoBrush)

    # ------------------------------------------------------------------
    # Coordinate conversion (image <-> scene)
    # ------------------------------------------------------------------

    def set_image_item(self, item):
        self._image_item = item

    def set_snap_enabled(self, enabled: bool):
        """Turn move/rotate snapping on or off; turning it off clears the snapping guide lines at once."""
        self._snap_enabled = bool(enabled)
        if not self._snap_enabled:
            self._clear_guide_lines()

    # ------------------------------------------------------------------
    # Data updates
    # ------------------------------------------------------------------

    def _apply_region_state(self, region_data: dict):
        """Sync the full region_data to the local state of the item."""
        self.region_data = copy.deepcopy(region_data)
        self.geo = RegionGeometryState.from_region_data(self.region_data)

        self.rotation_angle = float(self.geo.angle)
        self.visual_center = QPointF(
            float(self.geo.center[0]),
            float(self.geo.center[1]),
        )

        self.prepareGeometryChange()
        self._shape_path = None

        self.setPos(self.visual_center)
        self.setRotation(self.rotation_angle)
        self.setTransformOriginPoint(QPointF(0, 0))
        self._rebuild_qt_polygons()

    def update_from_data(self, region_data: dict):
        """Update the whole item state from new region_data."""
        try:
            if self._is_dragging or self._in_callback:
                return
            if not self.scene() or not hasattr(self, "region_data"):
                return

            old_rect = self.sceneBoundingRect() if self.scene() else None
            was_selected = self.isSelected()

            # The model data is the single source of truth, so leftover state of an old item cannot pollute it
            self._apply_region_state(region_data)

            if was_selected != self.isSelected():
                self.setSelected(was_selected)

            self.update()
            self._invalidate_scene_rect(old_rect)

        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[RegionTextItem] update_from_data: {e}")
        except Exception as e:
            logger.error(
                f"[RegionTextItem] update_from_data: {e}\n{traceback.format_exc()}"
            )

    # ------------------------------------------------------------------
    # Core fix: positioning the text pixmap
    # ------------------------------------------------------------------

    def update_text_pixmap(
        self, pixmap, pos, rotation=0.0, pivot_point=None, render_center=None
    ):
        """Update the position of the text pixmap.

        When render_center is available (the world coordinates of the white box centre), the pixmap is centred on it as anchor,
        which avoids the centre offset mapFromScene(pos) causes under the rotation transform.
        """
        self.text_item.setPixmap(pixmap)

        if render_center is not None and pixmap and not pixmap.isNull():
            # Positioned with the centre of the white box as the anchor
            local_center = self.mapFromScene(
                QPointF(float(render_center[0]), float(render_center[1]))
            )
            local_pos = QPointF(
                local_center.x() - pixmap.width() / 2.0,
                local_center.y() - pixmap.height() / 2.0,
            )
        else:
            # Fallback: map the top-left corner in world coordinates directly
            local_pos = self.mapFromScene(QPointF(float(pos.x()), float(pos.y())))

        self.text_item.setPos(local_pos)
        self.text_item.setTransformOriginPoint(self.text_item.boundingRect().center())
        self.text_item.setRotation(0)  # The parent item is already rotated

    def set_dst_points(self, dst_points):
        """Set the render dst_points; in automatic mode they are synced to the white box."""
        self.geo.set_render_box(dst_points)
        self.prepareGeometryChange()
        self._shape_path = None
        self.update()

    # ------------------------------------------------------------------
    # ID / serialisation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def set_text_visible(self, visible: bool):
        self.text_item.setVisible(visible)

    def set_box_visible(self, visible: bool):
        if self._polygons_visible != visible:
            self._polygons_visible = visible
            self.update()

    def set_white_box_visible(self, visible: bool):
        self._show_white_box = visible
        self.update()

    # ------------------------------------------------------------------
    # Overrides Qt requires: shape / boundingRect / paint
    # ------------------------------------------------------------------

    def shape(self) -> QPainterPath:
        try:
            if self._shape_path is not None:
                return self._shape_path

            path = QPainterPath()
            if self._show_white_box and self.geo.white_frame_local is not None:
                left, top, right, bottom = self.geo.white_frame_local
                path.addRect(QRectF(left, top, right - left, bottom - top))

                if self.isSelected():
                    metrics = self._white_handle_metrics()
                    r = metrics["hit_radius"]
                    for p in self._white_corner_points() + self._white_edge_points():
                        path.addEllipse(p, r, r)
                    ri = self._rotate_handle_info()
                    rr = ri["hit_radius"]
                    path.addEllipse(ri["rot_pos"], rr, rr)
                    path.moveTo(ri["center"])
                    path.lineTo(ri["rot_pos"])
            else:
                path = self._core_polygon_path()

            self._shape_path = path
            return path
        except Exception as e:
            logger.error(f"[RegionTextItem] shape: {e}\n{traceback.format_exc()}")
            return QPainterPath()

    def boundingRect(self) -> QRectF:
        try:
            # The margin grows with 1/lod: the pen width of the selection outline and handle shadow scales with 1/lod,
            # and at very small zoom a fixed margin would be smaller than the pen overhang, leaving trails when dragging
            margin = 10.0 + 6.0 / self._lod()
            return (
                self.shape().boundingRect().adjusted(-margin, -margin, margin, margin)
            )
        except Exception:
            return QRectF(0, 0, 100, 100)

    def paint(self, painter, option, widget=None):
        try:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            is_sel = bool(option.state & QStyle.StateFlag.State_Selected)

            if (
                self._show_white_box
                and self._polygons_visible
                and self.geo.white_frame_local is not None
            ):
                self._draw_white_box(painter, is_sel)

            if (
                is_sel
                and self._show_white_box
                and self.geo.white_frame_local is not None
                and self._polygons_visible
            ):
                ri = self._rotate_handle_info()
                hs = ri["handle_size"]
                pw = ri["pen_width"]
                half = hs / 2.0
                rot_rect = QRectF(
                    ri["rot_pos"].x() - half,
                    ri["rot_pos"].y() - half,
                    hs,
                    hs,
                )

                painter.setPen(_editor_pen(_shadow_color(125), pw * 3.0))
                painter.drawLine(ri["center"], ri["rot_pos"])
                painter.setPen(_editor_pen(_fluent_accent(205), pw * 1.45))
                painter.drawLine(ri["center"], ri["rot_pos"])
                painter.setBrush(QBrush(_fluent_surface(245)))
                painter.setPen(_editor_pen(_shadow_color(110), pw * 2.8))
                painter.drawEllipse(rot_rect)
                painter.setPen(_editor_pen(_fluent_accent(235), pw * 1.2))
                painter.drawEllipse(
                    rot_rect.adjusted(pw * 0.45, pw * 0.45, -pw * 0.45, -pw * 0.45)
                )
                dot = hs * 0.32
                dot_rect = QRectF(
                    ri["rot_pos"].x() - dot / 2.0,
                    ri["rot_pos"].y() - dot / 2.0,
                    dot,
                    dot,
                )
                painter.setBrush(QBrush(_fluent_accent(225)))
                painter.setPen(QPen(Qt.PenStyle.NoPen))
                painter.drawEllipse(dot_rect)
                self._draw_white_handles(painter)

            painter.restore()
        except Exception as e:
            logger.error(f"[RegionTextItem] paint: {e}\n{traceback.format_exc()}")
            try:
                painter.restore()
            except Exception:
                pass

    def itemChange(self, change, value):
        try:
            if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
                self.prepareGeometryChange()
                self._shape_path = None
            elif change == QGraphicsItem.GraphicsItemChange.ItemSceneChange:
                if value is None:
                    self._remove_angle_label()
                    self._clear_guide_lines()
        except Exception as e:
            logger.error(f"[RegionTextItem] itemChange: {e}")
        return super().itemChange(change, value)

    # ------------------------------------------------------------------
    # Helpers for drawing the white box
    # ------------------------------------------------------------------

    def _white_corner_points(self) -> list:
        wf = self.geo.white_frame_local
        if wf is None:
            return []
        left, top, right, bottom = wf
        return [
            QPointF(left, top),
            QPointF(right, top),
            QPointF(right, bottom),
            QPointF(left, bottom),
        ]

    def _white_edge_points(self) -> list:
        wf = self.geo.white_frame_local
        if wf is None:
            return []
        left, top, right, bottom = wf
        return [
            QPointF((left + right) / 2, top),
            QPointF(right, (top + bottom) / 2),
            QPointF((left + right) / 2, bottom),
            QPointF(left, (top + bottom) / 2),
        ]

    def _draw_white_box(self, painter, is_selected: bool):
        wf = self.geo.white_frame_local
        if wf is None:
            return
        left, top, right, bottom = wf
        poly = QPolygonF(
            [
                QPointF(left, top),
                QPointF(right, top),
                QPointF(right, bottom),
                QPointF(left, bottom),
            ]
        )

        if is_selected:
            lod = self._lod()
            painter.setBrush(QBrush(_fluent_accent(18)))
            painter.setPen(_editor_pen(_shadow_color(135), 5.0 / lod))
            painter.drawPolygon(poly)
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.setPen(_editor_pen(_fluent_accent(235), 2.35 / lod))
            painter.drawPolygon(poly)
        else:
            lod = self._lod()
            painter.setPen(
                _editor_pen(_shadow_color(105), 3.0 / lod, Qt.PenStyle.DashLine)
            )
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawPolygon(poly)
            painter.setPen(
                _editor_pen(_fluent_neutral(180), 1.25 / lod, Qt.PenStyle.DashLine)
            )
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawPolygon(poly)

    def _draw_white_handles(self, painter):
        mtx = self._white_handle_metrics()
        hs = mtx["visual_size"]
        pw = mtx["pen_width"]
        half = hs / 2.0
        radius = min(3.5 / mtx["lod"], half)
        accent = _fluent_accent(238)
        surface = _fluent_surface(246)
        handle_shadow = _shadow_color(120)

        for p in self._white_corner_points():
            rect = QRectF(p.x() - half, p.y() - half, hs, hs)
            painter.setBrush(QBrush(surface))
            painter.setPen(_editor_pen(handle_shadow, pw * 2.3))
            painter.drawRoundedRect(rect, radius, radius)
            painter.setPen(_editor_pen(accent, pw * 1.15))
            painter.drawRoundedRect(
                rect.adjusted(pw * 0.45, pw * 0.45, -pw * 0.45, -pw * 0.45),
                radius,
                radius,
            )

        for p in self._white_edge_points():
            rect = QRectF(p.x() - half, p.y() - half, hs, hs)
            painter.setBrush(QBrush(surface))
            painter.setPen(_editor_pen(handle_shadow, pw * 2.3))
            painter.drawEllipse(rect)
            painter.setBrush(QBrush(_fluent_accent(36)))
            painter.setPen(_editor_pen(accent, pw * 1.15))
            painter.drawEllipse(
                rect.adjusted(pw * 0.45, pw * 0.45, -pw * 0.45, -pw * 0.45)
            )

    # ------------------------------------------------------------------
    # Geometry / handle parameters
    # ------------------------------------------------------------------

    def _lod(self) -> float:
        if self.scene() and self.scene().views():
            return max(abs(self.scene().views()[0].transform().m11()), 0.01)
        return 1.0

    def _rotate_handle_info(self) -> dict:
        lod = self._lod()
        hs = 14.0 / lod
        if self.geo.white_frame_local is not None:
            left, top, right, bottom = self.geo.white_frame_local
            cx = (left + right) / 2
            cy = (top + bottom) / 2
            center = QPointF(cx, cy)
            rot_pos = QPointF(cx, top - 40.0 / lod)
        else:
            center = QPointF(0, 0)
            rot_pos = QPointF(0, -40.0 / lod)
        return {
            "lod": lod,
            "handle_size": hs,
            "hit_radius": (hs / 2.0) + (6.0 / lod),
            "pen_width": 1.15 / lod,
            "center": center,
            "rot_pos": rot_pos,
        }

    def _white_handle_metrics(self) -> dict:
        lod = self._lod()
        vs = 13.0 / lod
        return {
            "lod": lod,
            "visual_size": vs,
            "hit_radius": (vs / 2.0) + (6.0 / lod),
            "pen_width": 1.15 / lod,
        }

    def _rotation_pivot_local(self) -> QPointF:
        """Rotation pivot: the white box centre (falling back to the local origin)."""
        if self.geo.white_frame_local is not None:
            left, top, right, bottom = self.geo.white_frame_local
            return QPointF((left + right) / 2, (top + bottom) / 2)
        return QPointF(0, 0)

    def _core_polygon_path(self) -> QPainterPath:
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.WindingFill)
        seen = set()
        for poly in self.polygons:
            if not poly.isEmpty():
                key = tuple((p.x(), p.y()) for p in poly)
                if key not in seen:
                    path.addPolygon(poly)
                    path.closeSubpath()
                    seen.add(key)
        return path

    def _primary_view(self):
        scene = self.scene()
        if scene is None:
            return None
        views = scene.views()
        return views[0] if views else None

    def _active_view_tool(self):
        view = self._primary_view()
        if view is None:
            return None
        return getattr(view, "_active_tool", None)

    def _is_center_scale_enabled(self) -> bool:
        view = self._primary_view()
        return (
            bool(getattr(view, "_center_scale_enabled", False))
            if view is not None
            else False
        )

    def _rotated_world_point(
        self, point: tuple[float, float], cx: float, cy: float, angle: float
    ):
        x, y = point
        return rotate_point(x, y, angle, cx, cy) if angle != 0 else (x, y)

    def _white_edge_world_points(
        self, left: float, top: float, right: float, bottom: float, cx: float, cy: float
    ) -> list[tuple[float, float]]:
        return [
            ((left + right) / 2.0 + cx, top + cy),
            (right + cx, (top + bottom) / 2.0 + cy),
            ((left + right) / 2.0 + cx, bottom + cy),
            (left + cx, (top + bottom) / 2.0 + cy),
        ]

    def _clear_drag_context(self):
        self._drag_handle_indices = None
        self._drag_start_white_frame_local = None
        self._drag_start_white_handle_world = None
        self._drag_start_text_item_pos = QPointF()
        self._drag_start_pivot_scene = QPointF()
        self._drag_raw_rotation = 0.0
        self._drag_last_angle_rad = 0.0
        self._batch_drag_peers = []

    def _reset_interaction_state(self):
        self._interaction_mode = "none"
        self._is_dragging = False
        self._clear_drag_context()
        self._hide_angle_label()
        self._clear_guide_lines()

    def _capture_white_frame_drag_context(self, *, capture_text_pos: bool = False):
        self._drag_start_white_frame_local = (
            list(self.geo.white_frame_local) if self.geo.white_frame_local else None
        )
        self._drag_start_white_handle_world = None
        if capture_text_pos:
            self._drag_start_text_item_pos = QPointF(self.text_item.pos())

    def _capture_batch_drag_peers(self):
        """Batch move: record the initial state of the other selected items in the scene, to move them along during the drag."""
        self._batch_drag_peers = []
        sc = self.scene()
        if sc is None:
            return
        for item in sc.items():
            if not isinstance(item, RegionTextItem):
                continue
            if item is self or not item.isSelected():
                continue
            wf = item.geo.white_frame_local
            if wf is None:
                continue
            self._batch_drag_peers.append(
                {
                    "item": item,
                    "start_wf_local": list(wf),
                    "start_text_pos": QPointF(item.text_item.pos()),
                    "start_center": list(item.geo.center),
                    "old_rect": item.sceneBoundingRect(),
                }
            )

    def _move_batch_peers(self, scene_dx: float, scene_dy: float):
        """Apply the same scene displacement to the batch peers."""
        for peer in self._batch_drag_peers:
            item = peer["item"]
            if sip.isdeleted(item) or item.scene() is None:
                continue
            angle_rad = np.radians(item.rotation())
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            ldx = scene_dx * cos_a + scene_dy * sin_a
            ldy = -scene_dx * sin_a + scene_dy * cos_a
            wf = peer["start_wf_local"]
            moved = [wf[0] + ldx, wf[1] + ldy, wf[2] + ldx, wf[3] + ldy]
            item.prepareGeometryChange()
            item._shape_path = None
            item.geo.set_custom_white_frame_local(moved)
            item.text_item.setPos(peer["start_text_pos"] + QPointF(ldx, ldy))
            item.update()
            item._invalidate_scene_rect(peer["old_rect"])

    def _commit_batch_peers(self, event):
        """Commit the position changes of the batch peers to the model.

        The commit callback of each peer may trigger a rebuild of the items, so each one is checked for being alive.
        """
        for peer in self._batch_drag_peers:
            item = peer["item"]
            if sip.isdeleted(item) or item.scene() is None:
                continue
            patch = item.geo.to_region_data_patch()
            new_data = build_white_frame_region_data(
                item.region_index,
                item.region_data,
                patch,
                item.geo.white_frame_local,
                old_white_frame_local=peer["start_wf_local"],
                edit_mode="white_move",
            )
            item._emit_region_update(event, new_data)

    # ------------------------------------------------------------------
    # Rotation angle display
    # ------------------------------------------------------------------

    def _ensure_angle_label(self):
        """Make sure the angle label exists (it is created in the scene)."""
        if self._angle_label is not None:
            return
        scene = self.scene()
        if scene is None:
            return
        self._angle_label = QGraphicsSimpleTextItem()
        self._angle_label.setZValue(1000)
        font = QFont("Arial", 12)
        font.setBold(True)
        self._angle_label.setFont(font)
        self._angle_label.setBrush(QBrush(_fluent_accent(235)))
        self._angle_label.setPen(_editor_pen(_shadow_color(120), 0.5))
        scene.addItem(self._angle_label)
        self._angle_label.setVisible(False)

    def _show_angle_label(self, angle_deg: float, scene_pos: QPointF):
        """Show the label with the current rotation angle at the rotation position."""
        self._ensure_angle_label()
        if self._angle_label is None:
            return
        normalized = angle_deg % 360
        if normalized > 180:
            normalized -= 360
        self._angle_label.setText(f"{normalized:.1f}°")
        self._angle_label.setBrush(QBrush(_fluent_accent(235)))
        self._angle_label.setPen(_editor_pen(_shadow_color(120), 0.5))
        lod = self._lod()
        scale = 1.0 / max(lod, 0.1)
        self._angle_label.setScale(scale)
        offset_x = 18.0 / max(lod, 0.1)
        offset_y = -20.0 / max(lod, 0.1)
        self._angle_label.setPos(scene_pos.x() + offset_x, scene_pos.y() + offset_y)
        self._angle_label.setVisible(True)

    def _hide_angle_label(self):
        """Hide the rotation angle label."""
        if self._angle_label is not None:
            self._angle_label.setVisible(False)

    def _remove_angle_label(self):
        """Remove the angle label from the scene."""
        if self._angle_label is not None:
            scene = self.scene()
            if scene is not None:
                try:
                    scene.removeItem(self._angle_label)
                except (RuntimeError, AttributeError):
                    pass
            self._angle_label = None

    # ------------------------------------------------------------------
    # Alignment guides and snapping
    # ------------------------------------------------------------------

    def _get_white_frame_world_points_from_local(self, wf_local) -> dict:
        """Get the alignment reference points in world coordinates for the given local white box coordinates."""
        if wf_local is None:
            return {}
        left, top, right, bottom = wf_local
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        return {
            "center": self.mapToScene(QPointF(cx, cy)),
            "left": self.mapToScene(QPointF(left, cy)),
            "right": self.mapToScene(QPointF(right, cy)),
            "top": self.mapToScene(QPointF(cx, top)),
            "bottom": self.mapToScene(QPointF(cx, bottom)),
        }

    def _get_white_frame_world_points(self) -> dict:
        """Get the alignment reference points of the current white box in world coordinates."""
        return self._get_white_frame_world_points_from_local(self.geo.white_frame_local)

    def _get_other_items_snap_targets(self) -> list:
        """Get the alignment reference points of the other RegionTextItem objects in the scene."""
        targets = []
        scene = self.scene()
        if scene is None:
            return targets
        for item in scene.items():
            if isinstance(item, RegionTextItem) and item is not self:
                pts = item._get_white_frame_world_points()
                if pts:
                    targets.append(pts)
        return targets

    def _calculate_snap_offset(self, my_points: dict, targets: list) -> tuple:
        """Compute the alignment snapping offset between the current text box and the other items in the scene, and the guide line coordinates."""
        threshold = self._snap_threshold  # Keep an absolute distance threshold of 1px that does not change with the zoom
        best_dx = None
        best_dy = None
        best_dx_dist = threshold + 0.001
        best_dy_dist = threshold + 0.001
        guide_x_info = None
        guide_y_info = None

        ref_keys = ["center", "left", "right", "top", "bottom"]
        for target_pts in targets:
            for my_key in ref_keys:
                my_pt = my_points.get(my_key)
                if my_pt is None:
                    continue
                for tgt_key in ref_keys:
                    tgt_pt = target_pts.get(tgt_key)
                    if tgt_pt is None:
                        continue
                    dx = tgt_pt.x() - my_pt.x()
                    if abs(dx) < best_dx_dist:
                        best_dx_dist = abs(dx)
                        best_dx = dx
                        guide_x_info = (tgt_pt.x(), my_pt.y(), tgt_pt.y())
                    dy = tgt_pt.y() - my_pt.y()
                    if abs(dy) < best_dy_dist:
                        best_dy_dist = abs(dy)
                        best_dy = dy
                        guide_y_info = (tgt_pt.y(), my_pt.x(), tgt_pt.x())

        snap_dx = 0.0
        snap_dy = 0.0
        guides = []
        if best_dx is not None and best_dx_dist <= threshold:
            snap_dx = best_dx
            if guide_x_info:
                x, y1, y2 = guide_x_info
                guides.append({"kind": "vertical", "x": x})
        if best_dy is not None and best_dy_dist <= threshold:
            snap_dy = best_dy
            if guide_y_info:
                y, x1, x2 = guide_y_info
                guides.append({"kind": "horizontal", "y": y})
        return snap_dx, snap_dy, guides

    def _visible_scene_rect(self, scene: QGraphicsScene) -> QRectF:
        """Return the visible scene area, united over the current views."""
        visible_rect = QRectF()
        has_visible_rect = False
        for view in scene.views():
            view_rect = view.mapToScene(view.viewport().rect()).boundingRect()
            if not has_visible_rect:
                visible_rect = view_rect
                has_visible_rect = True
            else:
                visible_rect = visible_rect.united(view_rect)
        if not has_visible_rect or visible_rect.isNull():
            visible_rect = scene.sceneRect()
        return visible_rect

    def _build_guide_line(
        self,
        scene: QGraphicsScene,
        visible_rect: QRectF,
        extent: float,
        pen: QPen,
        guide_spec,
    ):
        """Create scene lines from guide line descriptions; both the explicit-direction format and the end-point segment format are accepted."""
        if isinstance(guide_spec, dict):
            kind = guide_spec.get("kind")
            if kind == "vertical":
                x = guide_spec.get("x")
                if x is None:
                    return None
                return scene.addLine(
                    x, visible_rect.top(), x, visible_rect.bottom(), pen
                )
            if kind == "horizontal":
                y = guide_spec.get("y")
                if y is None:
                    return None
                return scene.addLine(
                    visible_rect.left(), y, visible_rect.right(), y, pen
                )
            if kind == "segment":
                start = guide_spec.get("start")
                end = guide_spec.get("end")
                if start is None or end is None:
                    return None
                x1, y1 = start
                x2, y2 = end
            else:
                return None
        else:
            try:
                (x1, y1), (x2, y2) = guide_spec
            except (TypeError, ValueError):
                return None

        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 0.001:
            return None
        if abs(dx) < 0.001:
            return scene.addLine(x1, visible_rect.top(), x1, visible_rect.bottom(), pen)
        if abs(dy) < 0.001:
            return scene.addLine(visible_rect.left(), y1, visible_rect.right(), y1, pen)

        ux, uy = dx / length, dy / length
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return scene.addLine(
            cx - extent * ux, cy - extent * uy, cx + extent * ux, cy + extent * uy, pen
        )

    def _detect_spacing_snap(self, my_points: dict) -> tuple:
        """Smart spacing snap: detects equal-spacing relations with pairs of other items.

        Each pair (pa, pb) is normalised so the one with the smaller far is the "left/top" box, and two
        snap candidates are tried on each of the two axes:
          1. dragged to the far side of the right box: my_near aligns to right_far + gap
          2. dragged to the near side of the left box: my_far  aligns to left_near - gap
        """
        scene = self.scene()
        if scene is None:
            return 0.0, 0.0, []

        others: list[dict] = []
        for item in scene.items():
            if not isinstance(item, RegionTextItem):
                continue
            if item is self or item.isSelected():
                continue
            pts = item._get_white_frame_world_points()
            if pts and all(pts.get(k) for k in ("left", "right", "top", "bottom")):
                others.append(pts)
        if len(others) < 2:
            return 0.0, 0.0, []

        threshold = self._spacing_snap_threshold
        snap = [0.0, 0.0]  # [dx, dy]
        guides: list = []

        # Metadata of each axis: (axis_index, ori, near_key, far_key, perp_key, abcd_keys)
        axes = [
            (0, "h", "left", "right", "y", ("x1", "x2", "x3", "x4")),
            (1, "v", "top", "bottom", "x", ("y1", "y2", "y3", "y4")),
        ]

        for axis_idx, ori, near_k, far_k, perp_k, abcd_keys in axes:
            my_near = my_points.get(near_k)
            my_far = my_points.get(far_k)
            if not (my_near and my_far):
                continue
            comp = (lambda p: p.x()) if axis_idx == 0 else (lambda p: p.y())
            perp = (lambda p: p.y()) if axis_idx == 0 else (lambda p: p.x())

            for i, pa in enumerate(others):
                for pb in others[i + 1 :]:
                    an, af = comp(pa[near_k]), comp(pa[far_k])
                    bn, bf = comp(pb[near_k]), comp(pb[far_k])
                    # Normalise: the box with the smaller far value is the "left/top" one
                    if af < bn:
                        ln_, lf_, rn_, rf_ = an, af, bn, bf
                    elif bf < an:
                        ln_, lf_, rn_, rf_ = bn, bf, an, af
                    else:
                        continue
                    gap = rn_ - lf_
                    if gap < 4:
                        continue
                    candidates = [
                        (rf_ + gap, my_near, (lf_, rn_, rf_, rf_ + gap)),
                        (ln_ - gap, my_far, (ln_ - gap, ln_, lf_, rn_)),
                    ]
                    for tgt, my_edge, abcd in candidates:
                        if (
                            abs(comp(my_edge) - tgt) < threshold
                            and abs(snap[axis_idx]) < 0.01
                        ):
                            snap[axis_idx] = tgt - comp(my_edge)
                            spec = {
                                "kind": "spacing",
                                "ori": ori,
                                "label": f"{gap:.0f}",
                                perp_k: perp(my_edge),
                            }
                            for key, value in zip(abcd_keys, abcd):
                                spec[key] = value
                            guides.append(spec)

        return snap[0], snap[1], guides

    def _show_guide_lines(self, guide_specs: list, is_rotation: bool = False):
        """Draw full-screen dashed alignment/rotation guide lines and spacing rulers in the scene."""
        self._clear_guide_lines()
        scene = self.scene()
        if scene is None or not guide_specs:
            return

        visible_rect = self._visible_scene_rect(scene)
        extent = 2.0 * math.hypot(visible_rect.width(), visible_rect.height())

        if is_rotation:
            pen = QPen(QColor(255, 165, 0, 255), 1.5)
        else:
            pen = QPen(QColor(0, 255, 255, 255), 2.0)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)

        # Pen of the spacing ruler: a solid bright orange line, 2px thick
        sp_pen = QPen(QColor(255, 180, 60, 255), 2)
        sp_pen.setCosmetic(True)

        for guide_spec in guide_specs:
            if isinstance(guide_spec, dict) and guide_spec.get("kind") == "spacing":
                ori = guide_spec.get("ori", "h")
                label = guide_spec.get("label", "")
                tick_half = 24.0  # Half length of a tick

                if ori == "h":
                    # A pair of vertical lines each for the reference spacing (x1,x2) and the target spacing (x3,x4)
                    for xa, xb in [
                        (guide_spec.get("x1"), guide_spec.get("x2")),
                        (guide_spec.get("x3"), guide_spec.get("x4")),
                    ]:
                        if xa is not None and xb is not None:
                            y = guide_spec.get("y", 0)
                            ln = scene.addLine(
                                xa, y - tick_half, xa, y + tick_half, sp_pen
                            )
                            ln.setZValue(9999)
                            self._guide_lines.append(ln)
                            ln = scene.addLine(
                                xb, y - tick_half, xb, y + tick_half, sp_pen
                            )
                            ln.setZValue(9999)
                            self._guide_lines.append(ln)
                            # The label goes above the middle of the two ticks
                            lbl = QGraphicsSimpleTextItem(label)
                            lbl.setPos((xa + xb) / 2.0 - 10, y - tick_half - 22)
                            lbl.setZValue(9999)
                            lbl.setBrush(QBrush(QColor(255, 180, 60)))
                            f = lbl.font()
                            f.setPixelSize(16)
                            f.setBold(True)
                            lbl.setFont(f)
                            scene.addItem(lbl)
                            self._spacing_labels.append(lbl)
                else:
                    for ya, yb in [
                        (guide_spec.get("y1"), guide_spec.get("y2")),
                        (guide_spec.get("y3"), guide_spec.get("y4")),
                    ]:
                        if ya is not None and yb is not None:
                            x = guide_spec.get("x", 0)
                            ln = scene.addLine(
                                x - tick_half, ya, x + tick_half, ya, sp_pen
                            )
                            ln.setZValue(9999)
                            self._guide_lines.append(ln)
                            ln = scene.addLine(
                                x - tick_half, yb, x + tick_half, yb, sp_pen
                            )
                            ln.setZValue(9999)
                            self._guide_lines.append(ln)
                            lbl = QGraphicsSimpleTextItem(label)
                            lbl.setPos(x - tick_half - 34, (ya + yb) / 2.0 - 10)
                            lbl.setZValue(9999)
                            lbl.setBrush(QBrush(QColor(255, 180, 60)))
                            f = lbl.font()
                            f.setPixelSize(16)
                            f.setBold(True)
                            lbl.setFont(f)
                            scene.addItem(lbl)
                            self._spacing_labels.append(lbl)
            else:
                line = self._build_guide_line(
                    scene, visible_rect, extent, pen, guide_spec
                )
                if line is None:
                    continue
                line.setZValue(9999)
                self._guide_lines.append(line)

    def _clear_guide_lines(self):
        """Clear all guide lines and spacing labels."""
        scene = self.scene()
        for line in self._guide_lines:
            try:
                if scene is not None:
                    scene.removeItem(line)
            except (RuntimeError, AttributeError):
                pass
        self._guide_lines.clear()
        for label in self._spacing_labels:
            try:
                if scene is not None:
                    scene.removeItem(label)
            except (RuntimeError, AttributeError):
                pass
        self._spacing_labels.clear()

    # ------------------------------------------------------------------
    # Handle hit testing
    # ------------------------------------------------------------------

    def _get_handle_at(self, pos: QPointF):
        ri = self._rotate_handle_info()
        rot_hit_r = ri["hit_radius"]
        dx = ri["rot_pos"].x() - pos.x()
        dy = ri["rot_pos"].y() - pos.y()
        if dx * dx + dy * dy <= rot_hit_r * rot_hit_r:
            return "rotate", (-1, -1)

        if self._show_white_box and self.geo.white_frame_local is not None:
            result = self._white_handle_at(pos)
            if result[0] is not None:
                return result
            if self._point_in_white_frame(pos):
                return "white_move", (0, 0)

        return None, (-1, -1)

    def _white_handle_at(self, pos: QPointF):
        mtx = self._white_handle_metrics()
        hr_sq = mtx["hit_radius"] ** 2

        for i, p in enumerate(self._white_corner_points()):
            if (p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2 <= hr_sq:
                return "white_corner", i

        for i, p in enumerate(self._white_edge_points()):
            if (p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2 <= hr_sq:
                return "white_edge", i

        return None, (-1, -1)

    def _point_in_white_frame(self, pos: QPointF) -> bool:
        wf = self.geo.white_frame_local
        if wf is None:
            return False
        return wf[0] <= pos.x() <= wf[2] and wf[1] <= pos.y() <= wf[3]

    # ------------------------------------------------------------------
    # Commit entry point
    # ------------------------------------------------------------------

    def _emit_region_update(self, event, new_data: dict):
        cb = self.geometry_callback
        idx = self.region_index
        self.region_data = copy.deepcopy(new_data)
        super().mouseReleaseEvent(event)
        self._in_callback = True
        try:
            cb(idx, new_data)
        finally:
            self._in_callback = False

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def hoverMoveEvent(self, event: QGraphicsSceneMouseEvent):
        try:
            if not self.scene():
                super().hoverMoveEvent(event)
                return
            if self._active_view_tool() in DRAWING_TOOLS:
                self._clear_hover_cursor()
                super().hoverMoveEvent(event)
                return

            if self.isSelected():
                handle, idx = self._get_handle_at(event.pos())
                cursor_map = {
                    "rotate": Qt.CursorShape.SizeAllCursor,
                    "white_move": Qt.CursorShape.SizeAllCursor,
                }
                if handle == "white_corner":
                    cursor_map["white_corner"] = (
                        Qt.CursorShape.SizeFDiagCursor
                        if idx in (0, 2)
                        else Qt.CursorShape.SizeBDiagCursor
                    )
                elif handle == "white_edge":
                    cursor_map["white_edge"] = (
                        Qt.CursorShape.SizeVerCursor
                        if idx in (0, 2)
                        else Qt.CursorShape.SizeHorCursor
                    )
                cursor = cursor_map.get(handle)
                if cursor:
                    self._apply_hover_cursor(cursor)
                else:
                    self._clear_hover_cursor()
            else:
                self._clear_hover_cursor()
            super().hoverMoveEvent(event)
        except Exception as e:
            logger.debug(f"[RegionTextItem] hoverMoveEvent: {e}")

    def hoverLeaveEvent(self, event):
        self._clear_hover_cursor()
        super().hoverLeaveEvent(event)

    def _apply_hover_cursor(self, shape):
        self.setCursor(shape)
        view = self._primary_view()
        if view is not None and self._active_view_tool() == "select":
            view.viewport().setCursor(QCursor(shape))

    def _clear_hover_cursor(self):
        self.unsetCursor()
        view = self._primary_view()
        if view is not None and self._active_view_tool() == "select":
            view.viewport().unsetCursor()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        try:
            if not self.scene():
                super().mousePressEvent(event)
                return

            local_pos = event.pos()

            if event.button() == Qt.MouseButton.LeftButton:
                view = self._primary_view()
                if not view or not hasattr(view, "model"):
                    super().mousePressEvent(event)
                    return

                was_selected = self.isSelected()
                handle, indices = (None, (-1, -1))
                if was_selected:
                    handle, indices = self._get_handle_at(local_pos)
                super().mousePressEvent(event)

                if was_selected and self.isSelected():
                    self._save_drag_start_state(event, local_pos, handle, indices)
                else:
                    self._reset_interaction_state()
                event.accept()
                return
            super().mousePressEvent(event)
        except Exception as e:
            logger.error(
                f"[RegionTextItem] mousePressEvent: {e}\n{traceback.format_exc()}"
            )

    def _save_drag_start_state(self, event, local_pos, handle, indices):
        """Save all state at the start of a drag."""
        self._clear_drag_context()
        self._is_dragging = bool(handle)
        self._drag_start_pos = local_pos
        self._drag_start_scene_pos = event.scenePos()
        self._drag_start_polygons = [QPolygonF(p) for p in self.polygons]
        self._drag_start_rotation = self.rotation()
        self._drag_start_angle = self.rotation_angle
        self._drag_start_center = self._rotation_pivot_local()
        self._drag_start_scene_rect = self.sceneBoundingRect() if self.scene() else None
        self._drag_start_visual_center = QPointF(self.visual_center)

        if handle:
            self._interaction_mode = handle
            self._drag_handle_indices = indices

            if handle in ("white_corner", "white_edge"):
                self._capture_white_frame_drag_context()
                self._drag_start_white_handle_world = (
                    self._white_handle_world_at_start()
                )
            elif handle == "white_move":
                self._capture_white_frame_drag_context(capture_text_pos=True)
                self._capture_batch_drag_peers()
            elif handle == "rotate":
                self.setTransformOriginPoint(QPointF(0, 0))
                self._drag_start_pivot_scene = self.mapToScene(self._drag_start_center)
                vec = event.scenePos() - self._drag_start_pivot_scene
                self._drag_start_angle_rad = np.arctan2(vec.y(), vec.x())
                self._drag_last_angle_rad = self._drag_start_angle_rad
                self._drag_raw_rotation = self.rotation()
            else:
                self._reset_interaction_state()
        else:
            self._reset_interaction_state()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        try:
            if self._interaction_mode == "none":
                super().mouseMoveEvent(event)
                return

            if self._interaction_mode == "rotate":
                self._handle_rotate_drag(event)
            elif self._interaction_mode in ("white_corner", "white_edge"):
                self._handle_white_frame_edit(event)
            elif self._interaction_mode == "white_move":
                self._handle_white_frame_move(event)
            else:
                super().mouseMoveEvent(event)
            event.accept()
        except Exception as e:
            logger.error(
                f"[RegionTextItem] mouseMoveEvent: {e}\n{traceback.format_exc()}"
            )

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            # Releasing the right or middle button during a drag: never commit geometry and never interrupt the drag in progress
            try:
                super().mouseReleaseEvent(event)
            except RuntimeError:
                pass
            return

        # The commit callback may trigger regions_changed -> item rebuild, which destroys this object;
        # after the callback returns, self must be checked for being alive before it is touched, and reset is called only once, at the end.
        mode = self._interaction_mode
        try:
            if mode == "rotate":
                self._commit_rotation(event)
            elif mode in ("white_corner", "white_edge", "white_move"):
                self._commit_white_frame(event, mode)
            else:
                super().mouseReleaseEvent(event)
        except RuntimeError:
            # The item was destroyed in the callback; any further access would raise from a virtual function and abort
            return
        except Exception as e:
            logger.error(
                f"[RegionTextItem] mouseReleaseEvent: {e}\n{traceback.format_exc()}"
            )

        if sip.isdeleted(self):
            return
        try:
            self._reset_interaction_state()
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Rotation drag
    # ------------------------------------------------------------------

    def _handle_rotate_drag(self, event):
        """Run the rotation drag logic, with live angle display and snapping."""
        center_scene = self._drag_start_pivot_scene
        vec = event.scenePos() - center_scene
        new_angle_rad = np.arctan2(vec.y(), vec.x())
        delta_rad = np.arctan2(
            np.sin(new_angle_rad - self._drag_last_angle_rad),
            np.cos(new_angle_rad - self._drag_last_angle_rad),
        )
        delta_deg = np.degrees(delta_rad)
        self._drag_last_angle_rad = new_angle_rad
        self._drag_raw_rotation += delta_deg
        new_rot = self._drag_raw_rotation

        if self._snap_enabled:
            # --- Angle snapping ---
            snap_targets = [
                0.0,
                90.0,
                180.0,
                270.0,
                360.0,
                -90.0,
                -180.0,
                -270.0,
                -360.0,
            ]
            # Get the angles of the other text boxes
            scene = self.scene()
            if scene is not None:
                for item in scene.items():
                    if isinstance(item, RegionTextItem) and item is not self:
                        snap_targets.append(item.rotation() % 360)
                        snap_targets.append((item.rotation() % 360) - 360)

            best_diff = 3.0  # Angle snap threshold: 3 degrees
            snapped_rot = new_rot
            normalized_rot = new_rot % 360
            for target in snap_targets:
                normalized_target = target % 360
                diff = min(
                    abs(normalized_rot - normalized_target),
                    360 - abs(normalized_rot - normalized_target),
                )
                if diff <= best_diff:
                    best_diff = diff
                    # An actual rotation in degrees has to be worked out
                    # Stay as close as possible to the number of turns of new_rot
                    rounds = round((new_rot - target) / 360.0)
                    snapped_rot = target + rounds * 360.0

            new_rot = snapped_rot
        self.setRotation(new_rot)

        # Keep the centre of the white box (a local point) fixed in the scene
        theta = np.radians(new_rot)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        px, py = self._drag_start_center.x(), self._drag_start_center.y()
        self.setPos(
            QPointF(
                center_scene.x() - (px * cos_t - py * sin_t),
                center_scene.y() - (px * sin_t + py * cos_t),
            )
        )
        self.visual_center = QPointF(self.pos())

        # Show the rotation angle
        rot_handle_scene = self.mapToScene(self._rotate_handle_info()["rot_pos"])
        self._show_angle_label(new_rot, rot_handle_scene)

        # ====== Extension lines of the whole box while rotating (to help align with slanted lines in the background) ======
        wf = self.geo.white_frame_local
        if wf is not None:
            left, top, right, bottom = wf
            pts = [
                self.mapToScene(QPointF(left, top)),
                self.mapToScene(QPointF(right, top)),
                self.mapToScene(QPointF(right, bottom)),
                self.mapToScene(QPointF(left, bottom)),
            ]
            rot_guides = [
                ((pts[0].x(), pts[0].y()), (pts[1].x(), pts[1].y())),
                ((pts[1].x(), pts[1].y()), (pts[2].x(), pts[2].y())),
                ((pts[2].x(), pts[2].y()), (pts[3].x(), pts[3].y())),
                ((pts[3].x(), pts[3].y()), (pts[0].x(), pts[0].y())),
            ]
            self._show_guide_lines(rot_guides, is_rotation=True)

    def _commit_rotation(self, event):
        new_angle = self.rotation()
        self.rotation_angle = float(new_angle)

        old_center = self.geo.center
        if not (isinstance(old_center, (list, tuple)) and len(old_center) >= 2):
            old_center = [self.visual_center.x(), self.visual_center.y()]
        delta_x = float(self.pos().x()) - float(old_center[0])
        delta_y = float(self.pos().y()) - float(old_center[1])

        new_lines = []
        for poly in self.region_data.get("lines", []):
            new_poly = []
            for p in poly:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    new_poly.append([float(p[0]) + delta_x, float(p[1]) + delta_y])
            if new_poly:
                new_lines.append(new_poly)

        new_cx, new_cy = float(self.pos().x()), float(self.pos().y())
        self.visual_center = QPointF(new_cx, new_cy)

        # Sync the geo state: update center, lines and angle, and rebuild polygons_local and the white box
        self.geo.center = [new_cx, new_cy]
        self.geo.angle = float(new_angle)
        if new_lines:
            self.geo.lines = new_lines
        self.geo._rebuild_polygons_local()

        new_data = build_rotate_region_data(
            self.region_data,
            new_angle,
            new_center=[new_cx, new_cy],
            new_lines=new_lines or None,
        )
        self._emit_region_update(event, new_data)

    # ------------------------------------------------------------------
    # White box editing
    # ------------------------------------------------------------------

    def _handle_white_frame_edit(self, event: QGraphicsSceneMouseEvent):
        try:
            if (
                self._drag_handle_indices is None
                or self._drag_start_white_frame_local is None
            ):
                return

            left0, top0, right0, bottom0 = self._drag_start_white_frame_local
            cx, cy = self.geo.center
            start_verts = [
                [left0 + cx, top0 + cy],
                [right0 + cx, top0 + cy],
                [right0 + cx, bottom0 + cy],
                [left0 + cx, bottom0 + cy],
            ]

            delta = event.scenePos() - self._drag_start_scene_pos
            if self._drag_start_white_handle_world is not None:
                bx, by = self._drag_start_white_handle_world
                mouse = (bx + delta.x(), by + delta.y())
            else:
                mouse = (event.scenePos().x(), event.scenePos().y())

            if self._is_center_scale_enabled():
                angle = self.rotation()
                local_mouse = (
                    rotate_point(mouse[0], mouse[1], -angle, cx, cy) if angle else mouse
                )
                nl, nt, nr, nb = calculate_center_scaled_rect(
                    self._drag_start_white_frame_local,
                    "corner" if self._interaction_mode == "white_corner" else "edge",
                    self._drag_handle_indices,
                    (local_mouse[0] - cx, local_mouse[1] - cy),
                )
            else:
                if self._interaction_mode == "white_corner":
                    new_verts = calculate_new_vertices_on_drag(
                        start_verts,
                        self._drag_handle_indices,
                        mouse,
                        self.rotation_angle,
                        (cx, cy),
                    )
                elif self._interaction_mode == "white_edge":
                    new_verts = calculate_new_edge_on_drag(
                        start_verts,
                        self._drag_handle_indices,
                        mouse,
                        self.rotation_angle,
                        (cx, cy),
                    )
                else:
                    return

                if not new_verts or len(new_verts) != 4:
                    return

                nl = min(p[0] for p in new_verts) - cx
                nr = max(p[0] for p in new_verts) - cx
                nt = min(p[1] for p in new_verts) - cy
                nb = max(p[1] for p in new_verts) - cy
            min_s = 8.0
            if nr - nl < min_s:
                e = (min_s - (nr - nl)) / 2.0
                nl -= e
                nr += e
            if nb - nt < min_s:
                e = (min_s - (nb - nt)) / 2.0
                nt -= e
                nb += e

            old_rect = self.sceneBoundingRect() if self.scene() else None
            self.prepareGeometryChange()
            self._shape_path = None
            self.geo.set_custom_white_frame_local([nl, nt, nr, nb])
            self.update()
            self._invalidate_scene_rect(old_rect)

        except Exception as e:
            logger.error(
                f"[RegionTextItem] _handle_white_frame_edit: {e}\n{traceback.format_exc()}"
            )

    def _handle_white_frame_move(self, event: QGraphicsSceneMouseEvent):
        """Run the white box move logic, with position alignment snapping and guide lines."""
        try:
            if self._drag_start_white_frame_local is None:
                return

            old_rect = self.sceneBoundingRect() if self.scene() else None
            scene_delta = event.scenePos() - self._drag_start_scene_pos

            # Scene displacement -> local displacement (reverse rotation)
            angle_rad = np.radians(self.rotation())
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            dx = scene_delta.x() * cos_a + scene_delta.y() * sin_a
            dy = -scene_delta.x() * sin_a + scene_delta.y() * cos_a

            left, top, right, bottom = self._drag_start_white_frame_local
            moved = [left + dx, top + dy, right + dx, bottom + dy]

            if self._snap_enabled:
                # --- Snapping: edges and spacing are computed independently from the same position; spacing has priority ---
                my_points = self._get_white_frame_world_points_from_local(moved)
                targets = self._get_other_items_snap_targets()
                guide_specs = []
                edge_dx = edge_dy = 0.0
                if my_points and targets:
                    edge_dx, edge_dy, edge_guides = self._calculate_snap_offset(
                        my_points, targets
                    )
                    guide_specs.extend(edge_guides)
                spacing_dx, spacing_dy, spacing_guides = self._detect_spacing_snap(
                    my_points
                )
                guide_specs.extend(spacing_guides)
                if spacing_dx != 0.0 or spacing_dy != 0.0:
                    sldx = spacing_dx * cos_a + spacing_dy * sin_a
                    sldy = -spacing_dx * sin_a + spacing_dy * cos_a
                    moved = [
                        moved[0] + sldx,
                        moved[1] + sldy,
                        moved[2] + sldx,
                        moved[3] + sldy,
                    ]
                    dx += sldx
                    dy += sldy
                elif edge_dx != 0.0 or edge_dy != 0.0:
                    eldx = edge_dx * cos_a + edge_dy * sin_a
                    eldy = -edge_dx * sin_a + edge_dy * cos_a
                    moved = [
                        moved[0] + eldx,
                        moved[1] + eldy,
                        moved[2] + eldx,
                        moved[3] + eldy,
                    ]
                    dx += eldx
                    dy += eldy
                if guide_specs:
                    self._show_guide_lines(guide_specs)
                else:
                    self._clear_guide_lines()
                # --- End of snapping ---
            else:
                self._clear_guide_lines()

            self.prepareGeometryChange()
            self._shape_path = None
            self.geo.set_custom_white_frame_local(moved)
            # Light preview: only the text layer is translated
            self.text_item.setPos(self._drag_start_text_item_pos + QPointF(dx, dy))
            self.update()
            self._invalidate_scene_rect(old_rect)

            # Move the other selected items as a batch
            self._move_batch_peers(scene_delta.x(), scene_delta.y())

        except Exception as e:
            logger.error(
                f"[RegionTextItem] _handle_white_frame_move: {e}\n{traceback.format_exc()}"
            )

    def _commit_white_frame(self, event, edit_mode=None):
        scene = self.scene()
        if (
            scene
            and hasattr(self, "_drag_start_scene_rect")
            and self._drag_start_scene_rect is not None
        ):
            update_rect = self._drag_start_scene_rect.united(self.sceneBoundingRect())
            scene.invalidate(update_rect, QGraphicsScene.SceneLayer.ItemLayer)
            scene.update(update_rect)

        patch = self.geo.to_region_data_patch()
        new_data = build_white_frame_region_data(
            self.region_index,
            self.region_data,
            patch,
            self.geo.white_frame_local,
            old_white_frame_local=self._drag_start_white_frame_local,
            edit_mode=edit_mode,
        )
        self._emit_region_update(event, new_data)

        # The callback may have destroyed this item: the batch peers are only committed while it is alive
        if sip.isdeleted(self):
            return
        self._commit_batch_peers(event)

    def _white_handle_world_at_start(self):
        if self._drag_start_white_frame_local is None:
            return None
        left, top, right, bottom = self._drag_start_white_frame_local
        cx, cy = self.geo.center
        angle = self.rotation_angle

        if self._interaction_mode == "white_corner":
            idx = self._drag_handle_indices
            corners = [
                (left + cx, top + cy),
                (right + cx, top + cy),
                (right + cx, bottom + cy),
                (left + cx, bottom + cy),
            ]
            if not (0 <= idx < 4):
                return None
            return self._rotated_world_point(corners[idx], cx, cy, angle)

        if self._interaction_mode == "white_edge":
            idx = self._drag_handle_indices
            edge_points = self._white_edge_world_points(
                left, top, right, bottom, cx, cy
            )
            if not (0 <= idx < 4):
                return None
            return self._rotated_world_point(edge_points[idx], cx, cy, angle)

        return None

    # ------------------------------------------------------------------
    # Scene refresh
    # ------------------------------------------------------------------

    def _invalidate_scene_rect(self, old_rect):
        if self.scene() and old_rect is not None:
            new_rect = self.sceneBoundingRect()
            update_rect = old_rect.united(new_rect)
            self.scene().invalidate(update_rect, QGraphicsScene.SceneLayer.ItemLayer)
            self.scene().update(update_rect)

    # ------------------------------------------------------------------
    # WYSIWYG placeholder (the actual rendering is driven by GraphicsView)
    # ------------------------------------------------------------------
