from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPalette, QTransform
from PyQt6.QtWidgets import QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from editor.editor_model import EditorModel
from editor.render_coordinator import RenderCoordinator
from services import get_logger
from ui.theme import is_dark_theme

from .graphics_view_input import GraphicsViewInputMixin
from .graphics_view_layers import GraphicsViewLayersMixin
from .graphics_view_paste_overlays import GraphicsViewPasteOverlayMixin
from .graphics_view_rendering import GraphicsViewRenderingMixin
from .mask_layer import MaskLayer
from .overlay_layer import OverlayLayerManager
from .selection_manager import SelectionManager


def canvas_background_color(theme: str | None = None) -> QColor:
    """Canvas background colour. Dark or light is always decided through ui.theme (gray/forest/sunset/rose are dark themes too)."""
    return QColor("#1A1C20" if is_dark_theme(theme) else "#F7F7F7")


class GraphicsView(
    GraphicsViewLayersMixin,
    GraphicsViewRenderingMixin,
    GraphicsViewInputMixin,
    GraphicsViewPasteOverlayMixin,
    QGraphicsView,
):
    """Edit canvas: the main file only keeps initialisation, signal wiring and shared state."""

    region_geometry_changed = pyqtSignal(int, dict)
    view_state_changed = pyqtSignal(object, object)
    region_drag_started = pyqtSignal()
    region_drag_finished = pyqtSignal()
    blank_canvas_pressed = pyqtSignal()

    MASK_PREVIEW_MAX_PIXELS = 2_000_000
    INPAINT_PREVIEW_MAX_PIXELS = 6_000_000

    def __init__(
        self, model: EditorModel, controller=None, parent=None, editor_view=None
    ):
        super().__init__(parent)
        self.model = model
        self.controller = controller
        # Keep the EditorView reference explicitly: addWidget reparents this view to the canvas container,
        # so reaching EditorView through parent() afterwards no longer works
        self.editor_view = editor_view if editor_view is not None else parent
        self.logger = get_logger(__name__)
        self.render_coordinator = RenderCoordinator()

        self.scene = QGraphicsScene(self)
        # The editor often rebuilds a small set of text boxes as a whole; with the BSP index off, the extra cost of maintaining it on add/remove is avoided.
        self.scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.setScene(self.scene)

        self._image_item: QGraphicsPixmapItem = None
        self._q_image_ref = None
        self._document_identity: tuple[int, str] | None = None
        self._display_source_image_ref = None
        self._preview_item: QGraphicsPixmapItem = None
        self.mask_layer = MaskLayer(self)
        self.overlay_layers = OverlayLayerManager(self)

        self._region_items = []
        self._paste_overlay_items = []
        self._selected_paste_overlay_id = None
        self.setAcceptDrops(True)
        self._font_preview_overrides: dict[int, dict] = {}
        self._snap_enabled = False
        self._center_scale_enabled = False
        self._pending_geometry_edit_kinds: dict[int, str] = {}
        self._immediate_render_update_pending = False

        self._active_tool = "select"
        self._brush_size = 30
        self._brush_color = "#ffffff"
        self._is_drawing = False
        self._current_draw_scene_points: list[QPointF] = []
        self._current_draw_mask_points: list[tuple[int, int]] = []
        self._current_draw_mask_shape: tuple[int, int] | None = None

        # Clone stamp: the sampling point of the right click (image pixel coordinates); the offset is locked at the first stroke after sampling,
        # kept across strokes (aligned cloning), and reset on the next right-click sample
        self._clone_sample_image_point = None
        self._clone_offset = None  # (dx, dy)：src = dest + offset
        self._clone_marker_item = None
        # State while a stroke is in progress
        self._clone_drawing = False
        self._clone_old_overlay = None
        self._clone_working_overlay = None
        self._clone_composite = None
        self._clone_last_dab = None
        self._clone_preview_pixmap = None

        self._potential_drag = False
        self._drag_start_pos = None
        self._drag_threshold = 5
        self._region_drag_candidate = False
        self._region_drag_active = False
        self._hand_scroll_active = False

        self._is_drawing_textbox = False
        self._textbox_start_pos = None
        self._textbox_preview_item = None

        self.render_debounce_timer = QTimer(self)
        self.render_debounce_timer.setSingleShot(True)
        self.render_debounce_timer.setInterval(150)
        self.render_debounce_timer.timeout.connect(self._perform_render_update)

        self._setup_view()
        self._connect_model_signals()
        self.blank_canvas_pressed.connect(self.clear_paste_overlay_selection)

    def set_controller(self, controller) -> None:
        self.controller = controller

    def set_snap_enabled(self, enabled: bool) -> None:
        """Sync the existing text boxes on the canvas, and use the value as the default snapping state of text boxes created later."""
        self._snap_enabled = bool(enabled)
        for item in self._region_items:
            if item is not None:
                item.set_snap_enabled(self._snap_enabled)
        self.scene.update()

    def set_center_scale_enabled(self, enabled: bool) -> None:
        """Set whether dragging an edge or corner of a text box scales symmetrically about the centre."""
        self._center_scale_enabled = bool(enabled)

    def clear_pending_geometry_edits(self) -> None:
        self._clear_pending_geometry_edits()

    def get_live_region_state_patch(self, region_index: int) -> dict | None:
        if not (0 <= region_index < len(self._region_items)):
            return None

        item = self._region_items[region_index]
        geo = getattr(item, "geo", None) if item is not None else None
        if geo is None:
            return None

        patch = geo.to_persisted_state_patch()
        patch["center"] = list(geo.center)
        return patch

    def get_image_scene_rect(self) -> QRectF | None:
        """Return the bounding rectangle of the image item in the scene, for the "canvas" reference mode of alignment."""
        if self._image_item is not None:
            r = self._image_item.sceneBoundingRect()
            if r.isValid() and not r.isNull():
                return QRectF(r)
        return None

    def get_view_scene_rect(self) -> QRectF | None:
        """Return the scene range the current view really uses, for the two-pane view to sync the panning limits."""
        rect = self.scene.sceneRect()
        if rect.isValid() and not rect.isNull():
            return QRectF(rect)
        return self.get_image_scene_rect()

    def _setup_view(self):
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        # CacheBackground is not used: the background is a plain colour, and the cache would only add the allocation and blit of a viewport-sized pixmap.
        # DontAdjustForAntialiasing is not used: it cuts the margin of the update area from 2px to 0,
        # which, combined with thick outlines scaled by 1/lod, leaves trails.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self.apply_theme()
        self.selection_manager = SelectionManager(
            self.model, self.scene, lambda: self._region_items
        )

    def apply_theme(self, theme: str | None = None):
        canvas_color = canvas_background_color(theme)
        self.scene.setBackgroundBrush(canvas_color)
        self.setBackgroundBrush(canvas_color)
        self.setAutoFillBackground(True)
        palette = self.viewport().palette()
        palette.setColor(QPalette.ColorRole.Base, canvas_color)
        palette.setColor(QPalette.ColorRole.Window, canvas_color)
        self.viewport().setPalette(palette)
        self.viewport().setAutoFillBackground(True)
        self.scene.update()
        self.viewport().update()

    def _connect_model_signals(self):
        self.model.display_layers_changed.connect(self.on_display_layers_changed)
        self.model.regions_changed.connect(self.on_regions_changed)
        self.model.raw_mask_changed.connect(
            lambda mask: self.mask_layer.on_mask_data_changed("raw", mask)
        )
        self.model.refined_mask_changed.connect(
            lambda mask: self.mask_layer.on_mask_data_changed("refined", mask)
        )
        self.model.display_mask_type_changed.connect(
            self.mask_layer.on_display_mask_type_changed
        )
        self.model.paint_overlay_changed.connect(
            self.overlay_layers.paint_overlay.set_image
        )
        self.model.stamp_overlay_changed.connect(
            self.overlay_layers.stamp_overlay.set_image
        )
        self.model.paste_overlays_changed.connect(self.on_paste_overlays_changed)
        self.model.selection_changed.connect(
            self._on_model_selection_changed_for_paste_overlays
        )
        self.model.region_display_mode_changed.connect(
            self.on_region_display_mode_changed
        )
        self.model.active_tool_changed.connect(self._on_active_tool_changed)
        self.model.brush_size_changed.connect(self._on_brush_size_changed)
        self.model.brush_color_changed.connect(self._on_brush_color_changed)

    def get_view_state(self):
        if self._image_item is None:
            return None, None
        center_scene = self.mapToScene(self.viewport().rect().center())
        return QTransform(self.transform()), QPointF(center_scene)

    def _emit_view_state_changed(self):
        transform, center_scene = self.get_view_state()
        if transform is None or center_scene is None:
            return
        self.view_state_changed.emit(transform, center_scene)
