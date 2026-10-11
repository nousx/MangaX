from PyQt6.QtCore import QObject, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsRectItem
from qfluentwidgets import themeColor

from services import get_logger


class SelectionManager(QObject):
    """
    One place for the selection logic of the editor:
    - forward sync: Qt scene.selectionChanged → model.set_selection
    - reverse sync: model.selection_changed → Qt items setSelected
    - rubber-band selection: start/update/finish/cancel
    - the _syncing flag: prevents a sync loop, and is managed only inside this class
    """

    def __init__(self, model, scene, get_region_items_fn):
        """
        Args:
            model: the EditorModel instance
            scene: the QGraphicsScene instance
            get_region_items_fn: Callable that returns the current list of region items
        """
        super().__init__(model)
        self._model = model
        self._scene = scene
        self._get_region_items = get_region_items_fn
        self._logger = get_logger(__name__)

        # Sync guard
        self._syncing = False

        # Box selection state
        self._is_box_selecting = False
        self._box_select_start_pos = None
        self._box_select_rect_item: QGraphicsRectItem = None

        # Connect the signals
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)
        self._model.selection_changed.connect(self._sync_qt_from_model)

    def _region_items(self):
        return list(self._get_region_items() or [])

    @staticmethod
    def _is_live_item(item) -> bool:
        try:
            return bool(item and hasattr(item, "scene") and item.scene())
        except (RuntimeError, AttributeError):
            return False

    def _set_item_selected(self, item, selected: bool) -> None:
        if not self._is_live_item(item):
            return
        try:
            if item.isSelected() != selected:
                item.setSelected(selected)
        except (RuntimeError, AttributeError):
            pass

    def _end_box_select(self, *, remove_item: bool = False) -> QRectF | None:
        """Release box-selection state and return its last rectangle."""
        self._is_box_selecting = False
        self._box_select_start_pos = None
        item = self._box_select_rect_item
        if item is None:
            return None
        try:
            rect = QRectF(item.rect())
            if remove_item:
                if item.scene():
                    self._scene.removeItem(item)
                self._box_select_rect_item = None
            else:
                item.setVisible(False)
                item.setRect(0, 0, 0, 0)
            return rect
        except RuntimeError:
            self._box_select_rect_item = None
            return None

    def _selected_region_indices_from_scene(self) -> list[int]:
        from .graphics_items import RegionTextItem

        return sorted(
            item.region_index
            for item in self._region_items()
            if isinstance(item, RegionTextItem)
            and self._is_live_item(item)
            and item.isSelected()
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Box selection API
    # ------------------------------------------------------------------ #

    def start_box_select(self, scene_pos):
        """Begin a rubber-band selection"""
        self._is_box_selecting = True
        self._box_select_start_pos = scene_pos

        # Create or reuse the selection rectangle
        need_create = self._box_select_rect_item is None
        if not need_create:
            try:
                _ = self._box_select_rect_item.scene()
            except RuntimeError:
                need_create = True
                self._box_select_rect_item = None

        if need_create:
            self._box_select_rect_item = self._scene.addRect(0, 0, 0, 0)
            self._box_select_rect_item.setZValue(300)

        # The theme colour is read each time a box selection starts, so a cached old themeColor does not linger after the theme changes
        accent = themeColor().toRgb()
        if not accent.isValid():
            accent = QColor("#0F6CBD")
        fill = QColor(accent)
        accent.setAlpha(190)
        fill.setAlpha(36)
        pen = QPen(accent)
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        self._box_select_rect_item.setPen(pen)
        self._box_select_rect_item.setBrush(QBrush(fill))

        self._box_select_rect_item.setVisible(True)

    def update_box_select(self, scene_pos):
        """Update the rubber-band rectangle"""
        if not self._is_box_selecting or self._box_select_start_pos is None:
            return False
        try:
            rect = QRectF(self._box_select_start_pos, scene_pos).normalized()
            self._box_select_rect_item.setRect(rect)
            return True
        except (AttributeError, RuntimeError):
            self._end_box_select()
            return False

    def finish_box_select(self, ctrl_pressed):
        """Finish the rubber-band selection: compute the intersecting regions and update the selection"""
        select_rect = self._end_box_select()
        if select_rect is None:
            return

        try:
            # Use the item shape for exact hit testing, so rotated or thin regions are not selected by boundingRect alone
            from .graphics_items import RegionTextItem

            region_items = self._region_items()
            hit_items = self._scene.items(
                select_rect, Qt.ItemSelectionMode.IntersectsItemShape
            )
            selected_indices = sorted(
                {
                    int(item.region_index)
                    for item in hit_items
                    if isinstance(item, RegionTextItem) and self._is_live_item(item)
                }
            )

            # Set the selection state of the Qt items as a batch
            self._syncing = True
            try:
                if not ctrl_pressed:
                    for item in region_items:
                        self._set_item_selected(item, False)
                for idx in selected_indices:
                    if 0 <= idx < len(region_items):
                        self._set_item_selected(region_items[idx], True)
            finally:
                self._syncing = False

            # Trigger a sync by hand, once
            self._on_scene_selection_changed()
        except RuntimeError:
            pass

    def cancel_box_select(self):
        """Discard an in-progress box selection without changing selection."""
        self._end_box_select()

    @property
    def is_box_selecting(self):
        return self._is_box_selecting

    # ------------------------------------------------------------------ #
    #  Sync (internal)
    # ------------------------------------------------------------------ #

    def _on_scene_selection_changed(self):
        """Forward sync: Qt scene → model"""
        if self._syncing:
            return

        selected_indices = self._selected_region_indices_from_scene()

        if selected_indices != self._model.get_selection():
            self._model.set_selection(selected_indices)

    def _sync_qt_from_model(self, selected_indices):
        """Reverse sync: model → Qt items"""
        self._syncing = True
        try:
            region_items = self._region_items()

            # Clear the selection of all items
            for item in region_items:
                self._set_item_selected(item, False)

            # Set the newly selected items
            for idx in selected_indices:
                if 0 <= idx < len(region_items):
                    self._set_item_selected(region_items[idx], True)
        except Exception as e:
            self._logger.warning("Selection sync failed: %s", e, exc_info=True)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def suppress_forward_sync(self, suppress):
        """Pause or resume the forward sync during a batch operation"""
        self._syncing = suppress

    def restore_selection_after_rebuild(self):
        """Restore the selection state after the items were rebuilt (synced from the model to Qt)"""
        self._sync_qt_from_model(self._model.get_selection())

    def clear_state(self):
        """Clear all rubber-band state (when the image is switched and similar cases)"""
        self._end_box_select(remove_item=True)

    def on_scene_cleared(self):
        """Reset the rubber-band state after scene.clear() was called."""
        self._end_box_select()
