import logging
from functools import partial
from typing import Any

from PyQt6.QtCore import QPointF, QRect, QSize, Qt, QTimer, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    LineEdit,
    PopUpAniStackedWidget,
    PrimaryPushButton,
    PushButton,
    SegmentedWidget,
    SimpleCardWidget,
    ToolButton,
)
from qfluentwidgets import (
    FluentIcon as FIF,
)

from editor.editor_controller import EditorController
from editor.editor_logic import EditorLogic
from editor.editor_model import EditorModel
from services import get_config_service, get_i18n_manager
from ui.widgets.editor_toolbar import EditorToolbar
from ui.widgets.file_list_view import FileListView
from ui.widgets.hover_hint import set_hover_hint
from ui.widgets.property_panel import PropertyPanel
from ui.widgets.region_list_view import RegionListView
from ui.widgets.rich_text_floating_editor import RichTextFloatingEditor

from .graphics_view import GraphicsView
from .original_compare_view import OriginalCompareView
from .shortcut_manager import EditorShortcutManager


def _rich_editor_preferred_position(
    *,
    region_left: int,
    region_top: int,
    region_right: int,
    region_bottom: int,
    popup_width: int,
    popup_height: int,
    available: QRect,
    margin: int = 8,
    preserve_top: bool = False,
    previous_top: int | None = None,
) -> tuple[int, int, str]:
    """Place the editor below, then right, then left; never above the region."""
    region_left = int(region_left)
    region_top = int(region_top)
    region_right = int(region_right)
    region_bottom = int(region_bottom)
    popup_width = max(1, int(popup_width))
    popup_height = max(1, int(popup_height))
    margin = max(0, int(margin))
    spaces = {
        "below": available.bottom() - region_bottom - margin + 1,
        "right": available.right() - region_right - margin + 1,
        "left": region_left - available.left() - margin,
    }
    required = {
        "below": popup_height,
        "right": popup_width,
        "left": popup_width,
    }
    preferences = ("below", "right", "left")
    placement = next(
        (name for name in preferences if spaces[name] >= required[name]),
        None,
    )
    if placement is None:
        # Never fall back above the region. If no direction fits completely,
        # use the roomier horizontal side; left is allowed only in this case.
        placement = max(
            ("right", "left"),
            key=lambda name: spaces[name] / required[name],
        )

    region_center_x = (region_left + region_right) / 2.0
    region_center_y = (region_top + region_bottom) / 2.0
    if placement == "below":
        x = int(round(region_center_x - popup_width / 2.0))
        y = region_bottom + margin
    else:
        x = (
            region_right + margin
            if placement == "right"
            else region_left - popup_width - margin
        )
        y = (
            int(previous_top)
            if preserve_top and previous_top is not None
            else int(round(region_center_y - popup_height / 2.0))
        )

    min_x = available.left() + margin
    min_y = available.top() + margin
    max_x = max(min_x, available.right() - popup_width - margin + 1)
    max_y = max(min_y, available.bottom() - popup_height - margin + 1)
    return max(min_x, min(x, max_x)), max(min_y, min(y, max_y)), placement


class EditorView(QWidget):
    """
    Main view of the editor, with the file list, the canvas and the property panel.
    """

    LEFT_TRANSLATION_ROUTE = "editor_left_translation"
    LEFT_PROPERTY_ROUTE = "editor_left_property"
    LEFT_ASSISTANT_ROUTE = "chapter_assistant"
    EDITOR_SETTING_DEFAULTS = {
        "editor_snap_enabled": False,
        "editor_center_scale_enabled": False,
        "editor_rich_text_popup_enabled": True,
        "editor_rich_text_popup_pinned": False,
        "editor_auto_save_on_switch": True,
        "editor_auto_export_on_switch": True,
        "editor_suppress_unsaved_warning": False,
        "editor_auto_rich_text_rules": True,
        "editor_delete_and_recover": False,
    }

    def __init__(
        self,
        app_logic: Any,
        model: EditorModel,
        controller: EditorController,
        logic: EditorLogic,
        parent=None,
    ):
        super().__init__(parent)
        self.app_logic = app_logic
        self.model = model
        self.controller = controller
        self.logic = logic
        self.i18n = get_i18n_manager()
        self.config_service = (
            getattr(controller, "config_service", None) or get_config_service()
        )
        editor_settings = self._read_editor_settings()
        self._snap_enabled = editor_settings["editor_snap_enabled"]
        self._center_scale_enabled = editor_settings["editor_center_scale_enabled"]
        self._rich_text_popup_enabled = editor_settings[
            "editor_rich_text_popup_enabled"
        ]
        self._rich_text_popup_pinned = editor_settings["editor_rich_text_popup_pinned"]
        self.toolbar: EditorToolbar | None = None
        self.main_splitter: QSplitter | None = None
        self.left_panel_widget: QWidget | None = None
        self.left_segmented_widget: SegmentedWidget | None = None
        self.left_stack: PopUpAniStackedWidget | None = None
        self.find_input: LineEdit | None = None
        self.replace_input: LineEdit | None = None
        self.replace_all_button: PushButton | None = None
        self.apply_translations_button: PrimaryPushButton | None = None
        self.region_list_view: RegionListView | None = None
        self.property_panel: PropertyPanel | None = None
        self.compare_preview_container: QWidget | None = None
        self.original_compare_view: OriginalCompareView | None = None
        self.edit_canvas_container: QWidget | None = None
        self.graphics_view: GraphicsView | None = None
        self.rich_text_editor: RichTextFloatingEditor | None = None
        self._rich_editor_anchor_region = -1
        self._rich_editor_anchor_placement: str | None = None
        self._rich_editor_restore_on_show = False
        self._selection_from_translation_list = False
        self.add_files_button: PushButton | None = None
        self.add_folder_button: PushButton | None = None
        self.clear_list_button: PushButton | None = None
        self.file_list: FileListView | None = None

        # Give the controller a reference to the view, for updating the UI state
        self.controller.set_view(self)

        # The main layout becomes vertical, to make room for the top bar
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # 1. Top toolbar
        self.toolbar = EditorToolbar(
            self,
            snap_enabled=self._snap_enabled,
            center_scale_enabled=self._center_scale_enabled,
            rich_text_popup_enabled=self._rich_text_popup_enabled,
            rich_text_popup_pinned=self._rich_text_popup_pinned,
            auto_save_on_switch=editor_settings["editor_auto_save_on_switch"],
            auto_export_on_switch=editor_settings["editor_auto_export_on_switch"],
            suppress_unsaved_warning=editor_settings["editor_suppress_unsaved_warning"],
            auto_rich_text_rules=editor_settings["editor_auto_rich_text_rules"],
            delete_and_recover=editor_settings["editor_delete_and_recover"],
        )
        self.toolbar.setFixedHeight(56)
        self.layout.addWidget(self.toolbar)

        # 2. Main content splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.setHandleWidth(6)
        self.main_splitter = main_splitter
        self.layout.addWidget(main_splitter)

        # --- Left panel (tabs) ---
        left_panel = self._create_left_panel()

        # --- Central canvas area (canvas and zoom slider) ---
        center_panel = self._create_center_panel()

        # --- Right panel (file list) ---
        right_panel = self._create_right_panel()

        # --- Putting the layout together ---
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(center_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)  # Let the central canvas stretch
        main_splitter.setStretchFactor(2, 0)

        # --- Connect signals and slots ---
        self._connect_signals()

        # --- Set up the shortcut manager ---
        self.shortcut_manager = EditorShortcutManager(self)

        # --- Apply the editor style (the same as the main page) ---
        self._apply_editor_style()
        self._apply_initial_splitter_sizes()

    def _t(self, key: str, **kwargs) -> str:
        """Translation helper methods"""
        if self.i18n:
            return self.i18n.translate(key, **kwargs)
        return key

    def _read_editor_settings(self, config=None) -> dict[str, bool]:
        """Read every editor flag from one config snapshot."""
        if config is None and self.config_service is not None:
            config = self.config_service.get_config()
        app = (
            config.get("app", {})
            if isinstance(config, dict)
            else getattr(config, "app", None)
        )
        if isinstance(app, dict):
            return {
                key: bool(app.get(key, default))
                for key, default in self.EDITOR_SETTING_DEFAULTS.items()
            }
        return {
            key: bool(getattr(app, key, default))
            for key, default in self.EDITOR_SETTING_DEFAULTS.items()
        }

    def _apply_editor_setting(self, key: str, enabled: bool) -> None:
        """Apply one editor flag to every surface that consumes it."""
        enabled = bool(enabled)
        if key == "editor_snap_enabled":
            self._snap_enabled = enabled
            if self.toolbar is not None:
                self.toolbar.set_snap_enabled(enabled)
            if self.graphics_view is not None:
                self.graphics_view.set_snap_enabled(enabled)
        elif key == "editor_center_scale_enabled":
            self._center_scale_enabled = enabled
            if self.toolbar is not None:
                self.toolbar.set_center_scale_enabled(enabled)
            if self.graphics_view is not None:
                self.graphics_view.set_center_scale_enabled(enabled)
        elif key == "editor_rich_text_popup_enabled":
            changed = enabled != self._rich_text_popup_enabled
            self._rich_text_popup_enabled = enabled
            if self.toolbar is not None:
                self.toolbar.set_rich_text_popup_enabled(enabled)
            editor = self.rich_text_editor
            if editor is None:
                return
            if not enabled:
                if self._rich_text_popup_pinned:
                    self._apply_editor_setting("editor_rich_text_popup_pinned", False)
                self._rich_editor_restore_on_show = False
                self._reset_rich_editor_anchor()
                if changed or editor.isVisible():
                    # clear_region first flushes the content still waiting in the debounce period, then unbinds and hides.
                    editor.clear_region()
            elif changed:
                self._on_selection_changed_for_rich_editor(self.model.get_selection())
        elif key == "editor_rich_text_popup_pinned":
            self._rich_text_popup_pinned = enabled
            if self.toolbar is not None:
                self.toolbar.set_rich_text_popup_pinned(enabled)
            editor = self.rich_text_editor
            if (
                editor is None
                or not self._rich_text_popup_enabled
                or not self.isVisible()
            ):
                return
            selected = self.model.get_selection()
            has_single_selection = bool(selected) and len(selected) == 1
            if enabled:
                if editor.isVisible() or has_single_selection:
                    self._on_selection_changed_for_rich_editor(selected)
            elif self._translation_list_is_active() or not has_single_selection:
                editor.clear_region()
            else:
                editor.reset_manual_position()
                self._position_rich_text_editor(int(selected[0]))
        elif self.toolbar is not None:
            if key == "editor_auto_save_on_switch":
                self.toolbar.set_auto_save_on_switch(enabled)
            elif key == "editor_auto_export_on_switch":
                self.toolbar.set_auto_export_on_switch(enabled)
            elif key == "editor_suppress_unsaved_warning":
                self.toolbar.set_suppress_unsaved_warning(enabled)
            elif key == "editor_auto_rich_text_rules":
                self.toolbar.set_auto_rich_text_rules(enabled)
            elif key == "editor_delete_and_recover":
                self.toolbar.set_delete_and_recover(enabled)

    def _persist_editor_setting(self, key: str, enabled: bool) -> None:
        """Apply and persist a toolbar editor flag through one path."""
        enabled = bool(enabled)
        self._apply_editor_setting(key, enabled)
        if self.config_service is None:
            return
        current = self._read_editor_settings(self.config_service.get_config())
        if current[key] != enabled:
            self.config_service.update_config({"app": {key: enabled}})
        self.config_service.save_config_file()

    @pyqtSlot(dict)
    def _on_config_changed(self, config: dict) -> None:
        for key, enabled in self._read_editor_settings(config).items():
            self._apply_editor_setting(key, enabled)

    def force_save_property_panel_edits(self):
        """Force the text edits in the property panel to be saved"""
        self.property_panel.force_save_text_edits()

    def _handle_copy_from_panel(self):
        """Handle the copy button of the property panel"""
        selected_regions = self.model.get_selection()
        if selected_regions:
            self.controller.copy_regions(selected_regions)

    def _handle_paste_from_panel(self):
        """Handle the paste button of the property panel"""
        selected_regions = self.model.get_selection()
        if selected_regions and len(selected_regions) == 1:
            # With a single selected region, paste the style
            self.controller.paste_region_style(selected_regions[0])
        else:
            # With no selected region, paste a new region
            self.controller.paste_region()

    def _handle_delete_from_panel(self):
        """Handle the delete button of the property panel"""
        selected_regions = self.model.get_selection()
        if selected_regions:
            self.controller.delete_regions(selected_regions)

    def _create_left_panel(self) -> QWidget:
        """Create the tabs on the left, with the region list and the property panel"""
        left_panel = SimpleCardWidget(self)
        # setFixedWidth cannot be used: inside a QSplitter, min==max makes the splitter handle impossible to drag.
        # The initial width is set by _apply_initial_splitter_sizes from sizeHint.
        left_panel.setMinimumWidth(280)
        left_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        self.left_panel_widget = left_panel
        self.left_segmented_widget = SegmentedWidget(left_panel)
        self.left_segmented_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.left_stack = PopUpAniStackedWidget(left_panel)
        self.left_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        left_layout.addWidget(self.left_segmented_widget)
        left_layout.addWidget(self.left_stack, 1)

        # Create the "translation list" tab
        translation_widget = SimpleCardWidget(left_panel)
        translation_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        translation_layout = QVBoxLayout(translation_widget)
        translation_layout.setContentsMargins(8, 8, 8, 8)
        translation_layout.setSpacing(8)

        # --- Find and replace ---
        replace_widget = SimpleCardWidget(translation_widget)
        replace_layout = QVBoxLayout(replace_widget)
        replace_layout.setContentsMargins(8, 8, 8, 8)
        replace_layout.setSpacing(6)
        self.find_input = LineEdit()
        self.find_input.setPlaceholderText(self._t("Find"))
        self.replace_input = LineEdit()
        self.replace_input.setPlaceholderText(self._t("Replace with"))
        self.replace_all_button = PushButton()
        self.replace_all_button.setText(self._t("Replace All"))
        self.replace_all_button.setIcon(FIF.SYNC)
        replace_layout.addWidget(self.find_input)
        replace_layout.addWidget(self.replace_input)
        replace_layout.addWidget(self.replace_all_button)

        self.apply_translations_button = PrimaryPushButton()
        self.apply_translations_button.setText(self._t("Apply All Translation Changes"))
        self.apply_translations_button.setIcon(FIF.ACCEPT)
        self.region_list_view = RegionListView(self.model, self)

        translation_layout.addWidget(replace_widget)
        translation_layout.addWidget(self.apply_translations_button)
        translation_layout.addWidget(self.region_list_view)

        self.property_panel = PropertyPanel(self.model, self.app_logic, self)

        self.left_stack.addWidget(translation_widget)
        self.left_stack.addWidget(self.property_panel)
        # The assistant is optional: its failure must never keep the editor from opening.
        self.chapter_assistant = None
        try:
            from ui.widgets.chapter_assistant import ChapterAssistant
            self.chapter_assistant = ChapterAssistant(self, left_panel)
            self.left_stack.addWidget(self.chapter_assistant)
        except Exception:
            logging.getLogger("manga_translator").exception(
                "Chapter assistant failed to start; the editor continues without it"
            )
            self._chapter_assistant_placeholder = BodyLabel(left_panel)
            self._chapter_assistant_placeholder.setWordWrap(True)
            self._chapter_assistant_placeholder.setAlignment(
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
            )
            self._chapter_assistant_placeholder.setContentsMargins(12, 12, 12, 12)
            self._chapter_assistant_placeholder.setText(
                self._t("The assistant could not start. The rest of the editor works normally; see the log for details.")
            )
            self.left_stack.addWidget(self._chapter_assistant_placeholder)
        self.left_segmented_widget.addItem(
            self.LEFT_TRANSLATION_ROUTE, self._t("Translation List")
        )
        self.left_segmented_widget.addItem(
            self.LEFT_PROPERTY_ROUTE, self._t("Property Editor")
        )
        self.left_segmented_widget.addItem(
            self.LEFT_ASSISTANT_ROUTE, self._t("Assistant")
        )
        self.left_segmented_widget.currentItemChanged.connect(self._set_left_route)

        # Show the "property editing" tab by default
        self._set_left_route(self.LEFT_PROPERTY_ROUTE, emit_changed=False)

        return left_panel

    def _set_left_route(self, route_key: str, emit_changed: bool = True):
        if self.left_stack is None or self.left_segmented_widget is None:
            return
        if route_key == self.LEFT_TRANSLATION_ROUTE:
            index = 0
        elif route_key == self.LEFT_PROPERTY_ROUTE:
            index = 1
        elif route_key == self.LEFT_ASSISTANT_ROUTE:
            index = 2
        else:
            return
        if index >= self.left_stack.count():
            return
        changed = self.left_stack.currentIndex() != index
        self.left_stack.setCurrentIndex(index)
        self.left_segmented_widget.setCurrentItem(route_key)
        if emit_changed and changed:
            self._on_left_route_index_changed(index)

    def _on_left_route_index_changed(self, index: int):
        if index == 0 and self.region_list_view is not None:
            self.region_list_view.flush_pending_regions()
            self._hide_rich_text_editor_for_list_action()

    def _translation_list_is_active(self) -> bool:
        return self.left_stack is not None and self.left_stack.currentIndex() == 0

    def refresh_tab_titles(self):
        """Refresh the tab titles (for a language switch)"""
        if self.left_segmented_widget is None:
            return

        self.left_segmented_widget.setItemText(
            self.LEFT_TRANSLATION_ROUTE, self._t("Translation List")
        )
        self.left_segmented_widget.setItemText(
            self.LEFT_PROPERTY_ROUTE, self._t("Property Editor")
        )
        self.left_segmented_widget.setItemText(
            self.LEFT_ASSISTANT_ROUTE, self._t("Assistant")
        )

    def refresh_ui_texts(self):
        """Refresh all UI texts (for a language switch)"""
        # Refresh the tab titles
        self.refresh_tab_titles()

        # Refresh the find and replace buttons
        if self.find_input is not None:
            self.find_input.setPlaceholderText(self._t("Find"))
        if self.replace_input is not None:
            self.replace_input.setPlaceholderText(self._t("Replace with"))
        if self.replace_all_button is not None:
            self.replace_all_button.setText(self._t("Replace All"))
        if self.apply_translations_button is not None:
            self.apply_translations_button.setText(
                self._t("Apply All Translation Changes")
            )
        if self.region_list_view is not None:
            self.region_list_view.refresh_ui_texts()

        # Refresh the toolbar
        if self.toolbar is not None:
            self.toolbar.refresh_ui_texts()

        # Refresh the property panel
        if self.property_panel is not None:
            self.property_panel.refresh_ui_texts()

        if getattr(self, "chapter_assistant", None) is not None:
            self.chapter_assistant.refresh_ui_texts()
        elif getattr(self, "_chapter_assistant_placeholder", None) is not None:
            self._chapter_assistant_placeholder.setText(
                self._t("The assistant could not start. The rest of the editor works normally; see the log for details.")
            )

        if self.rich_text_editor is not None:
            self.rich_text_editor.refresh_ui_texts()

        # Refresh the buttons of the file list on the right
        if self.add_files_button is not None:
            set_hover_hint(self.add_files_button, self._t("Add Files"))
        if self.add_folder_button is not None:
            set_hover_hint(self.add_folder_button, self._t("Add Folder"))
        if self.clear_list_button is not None:
            set_hover_hint(self.clear_list_button, self._t("Clear List"))

        # The text of the file items needs no rebuilding; on a language switch only the placeholder hint of the empty list is repainted.
        if self.file_list is not None:
            self.file_list.refresh_empty_state_text()

    def _apply_initial_splitter_sizes(self):
        """Use the actual sizeHint of the left pane as the initial width, instead of a hard-coded constant."""
        if self.main_splitter is None or self.left_panel_widget is None:
            return

        self.left_panel_widget.ensurePolished()
        left_width = self.left_panel_widget.maximumWidth()
        if left_width <= 0 or left_width >= 16777215:
            left_width = self.left_panel_widget.sizeHint().width()

        right_width = 260
        if self.file_list is not None:
            self.file_list.setMinimumWidth(0)
        self.main_splitter.setSizes([left_width, 860, right_width])

    def _on_apply_changes_clicked(self):
        """Apply all translations changed in the list"""
        translations = self.region_list_view.get_all_translations()
        self.controller.update_multiple_translations(translations)

    def save_editor_state(self):
        """Save the current editor project data."""
        if self.rich_text_editor is not None:
            self.rich_text_editor.flush_pending_changes()
        return self.controller.save_editor_state()

    def export_image(self):
        """Export the current rendered image, without saving the project data."""
        if self.rich_text_editor is not None:
            self.rich_text_editor.flush_pending_changes()
        return self.controller.export_image()

    def _on_replace_all_clicked(self):
        """Run find and replace over all translations"""
        find_text = self.find_input.text()
        replace_text = self.replace_input.text()

        if not find_text:
            return

        self.region_list_view.find_and_replace_in_all_translations(
            find_text, replace_text
        )

    def _on_align_requested(self, mode: str):
        """Handle a click on an align button."""
        reference = self.toolbar.get_align_reference()
        self.controller.align_regions(mode, reference)

    def _on_distribute_requested(self, mode: str):
        """Handle a click on a distribute button."""
        self.controller.distribute_regions(mode)

    def _on_selection_changed_for_toolbar(self, selected_indices: list):
        """Update the enabled state of the align/distribute buttons from the number of selected items."""
        count = len(selected_indices) if selected_indices else 0
        self.toolbar.update_align_distribute_buttons(count)

    def _reset_rich_editor_anchor(self) -> None:
        self._rich_editor_anchor_region = -1
        self._rich_editor_anchor_placement = None

    def _clear_rich_text_editor(self, *, raise_visible: bool = True) -> None:
        editor = self.rich_text_editor
        if editor is None:
            return
        keep_visible = self._rich_text_popup_pinned and editor.isVisible()
        editor.clear_region(hide=not keep_visible)
        if keep_visible and raise_visible:
            editor.raise_()

    def _sync_rich_text_editor_region(
        self,
        region_index: int,
        *,
        refresh_only: bool = False,
        reset_position: bool = False,
        position_only: bool = False,
    ) -> None:
        """Flush, bind, and display the selected region through one transition."""
        editor = self.rich_text_editor
        if editor is None:
            return
        anchor_changed = region_index != self._rich_editor_anchor_region
        if not refresh_only and (anchor_changed or reset_position):
            self._rich_editor_anchor_region = region_index
            self._rich_editor_anchor_placement = None
            if reset_position or not self._rich_text_popup_pinned:
                editor.reset_manual_position()
        if position_only:
            self._position_rich_text_editor(region_index)
            editor.show()
            editor.raise_()
            return

        # F22: before rebinding, write back the pending content of the previous region's debounce period, then take the data of the new region.
        editor.flush_pending_changes()
        region_data = self.model.get_region_by_index(region_index)
        if not region_data:
            self._clear_rich_text_editor(raise_visible=False)
            return
        if refresh_only:
            editor.refresh_region_if_changed(region_index, region_data)
            return

        was_visible = editor.isVisible()
        editor.set_region(region_index, region_data)
        if reset_position or not self._rich_text_popup_pinned or not was_visible:
            self._position_rich_text_editor(region_index)
        editor.show()
        editor.raise_()
        # F09: selecting no longer calls focus_text() to grab the focus - the focus stays on the canvas,
        # so canvas shortcuts such as Delete/A/D/Q/W/E keep working; clicking the text box focuses it naturally and starts editing.

    def _on_selection_changed_for_rich_editor(self, selected_indices: list):
        editor = self.rich_text_editor
        if editor is None:
            return
        if (
            self._selection_from_translation_list or self._translation_list_is_active()
        ) and not self._rich_text_popup_pinned:
            self._hide_rich_text_editor_for_list_action()
            return
        if not self._rich_text_popup_enabled:
            self._rich_editor_restore_on_show = False
            editor.hide()
            return
        has_single_selection = bool(selected_indices) and len(selected_indices) == 1
        if not self.isVisible():
            # A top-level tool window is not hidden automatically with the
            # stacked editor page. Never show it over another page.
            self._rich_editor_restore_on_show = (
                has_single_selection or self._rich_text_popup_pinned
            )
            editor.hide()
            return
        self._rich_editor_restore_on_show = False
        if not has_single_selection:
            self._reset_rich_editor_anchor()
            self._clear_rich_text_editor()
            return
        self._sync_rich_text_editor_region(int(selected_indices[0]))

    def _hide_rich_text_editor_for_list_action(self) -> None:
        if self._rich_text_popup_pinned:
            return
        self._rich_editor_restore_on_show = False
        editor = self.rich_text_editor
        if editor is None:
            return
        editor.flush_pending_changes()
        editor.hide()

    def _on_region_selected_from_list(self, indices: list) -> None:
        self._selection_from_translation_list = True
        try:
            self._hide_rich_text_editor_for_list_action()
            self.controller.set_selection_from_list(indices)
        finally:
            self._selection_from_translation_list = False

    def _on_region_move_requested_from_list(
        self, source_index: int, target_index: int
    ) -> None:
        self._selection_from_translation_list = True
        try:
            self._hide_rich_text_editor_for_list_action()
            self.controller.move_region_from_list(source_index, target_index)
        finally:
            self._selection_from_translation_list = False

    def _on_regions_changed_for_rich_editor(self, change=None):
        """Refresh a visible bound document without feeding its own write-back around."""
        editor = self.rich_text_editor
        if editor is None or not editor.isVisible() or editor.is_applying_own_change():
            return
        selected = self.model.get_selection()
        if not selected or len(selected) != 1:
            return
        region_index = int(selected[0])
        if getattr(change, "kind", "") == "updated" and region_index not in getattr(
            change, "indices", ()
        ):
            return
        # Pending input wins before model data refreshes the bound document.
        self._sync_rich_text_editor_region(region_index, refresh_only=True)

    def _position_rich_text_editor_for_selection(self, *args):
        preserve_top = len(args) == 1 and isinstance(args[0], bool) and args[0]
        editor = self.rich_text_editor
        if not self._rich_text_popup_enabled:
            if editor is not None:
                editor.hide()
            return
        if self._rich_text_popup_pinned and editor is not None and editor.isVisible():
            return
        selected = self.model.get_selection()
        if selected and len(selected) == 1:
            if editor is not None and editor.is_manually_positioned():
                return
            self._position_rich_text_editor(int(selected[0]), preserve_top=preserve_top)
        elif editor is not None:
            editor.hide()

    def _hide_rich_text_editor_for_region_drag(self):
        if self._rich_text_popup_pinned:
            return
        if self.rich_text_editor is not None:
            self.rich_text_editor.hide()

    def _restore_rich_text_editor_after_region_drag(self):
        if (
            not self._rich_text_popup_enabled
            or self._rich_text_popup_pinned
            or self._translation_list_is_active()
        ):
            return
        # Wait until the geometry commit and a possible item rebuild are done, then restore the floating editor at the new position.
        QTimer.singleShot(0, self._show_rich_text_editor_after_region_drag)

    def _show_rich_text_editor_after_region_drag(self):
        if (
            not self._rich_text_popup_enabled
            or self._rich_text_popup_pinned
            or self._translation_list_is_active()
        ):
            return
        selected = self.model.get_selection()
        if selected and len(selected) == 1:
            # After a drag the text box has moved; choose the docking side again for the new position.
            self._sync_rich_text_editor_region(
                int(selected[0]), reset_position=True, position_only=True
            )

    def _position_rich_text_editor(
        self, region_index: int, *, preserve_top: bool = False
    ):
        if (
            not self._rich_text_popup_enabled
            or self.rich_text_editor is None
            or self.graphics_view is None
        ):
            return
        region_items = getattr(self.graphics_view, "_region_items", [])
        if not (0 <= region_index < len(region_items)):
            return
        item = region_items[region_index]
        if item is None or item.scene() is None:
            return

        rect = item.sceneBoundingRect()
        viewport = self.graphics_view.viewport()

        def desktop_position(scene_position: QPointF):
            return viewport.mapToGlobal(self.graphics_view.mapFromScene(scene_position))

        center = rect.center()
        left_anchor = desktop_position(QPointF(rect.left(), center.y()))
        top_anchor = desktop_position(QPointF(center.x(), rect.top()))
        right_anchor = desktop_position(QPointF(rect.right(), center.y()))
        bottom_anchor = desktop_position(QPointF(center.x(), rect.bottom()))
        previous_top = self.rich_text_editor.y()
        preserve_popup_top = (
            preserve_top
            and self.rich_text_editor.isVisible()
            and region_index == self._rich_editor_anchor_region
            and self._rich_editor_anchor_placement in {"left", "right"}
        )
        # adjustSize() is not called: the size of the popup is managed by its own _refresh_layout_size,
        # and calling adjustSize here would fight it for the size and cause jitter
        popup_w = self.rich_text_editor.width()
        popup_h = self.rich_text_editor.height()
        margin = 8
        # Automatic placement is screen-aware rather than canvas-aware. We
        # occupy any area outside the canvas and can be dragged to another
        # screen without being clipped by the viewport.
        screen = QApplication.screenAt(right_anchor)
        if screen is None:
            screen = self.rich_text_editor.screen() or self.window().screen()
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()

        # Re-evaluate on every geometry change so a left fallback is abandoned
        # as soon as the preferred below or right placement fits again.
        if region_index != self._rich_editor_anchor_region:
            self._rich_editor_anchor_region = region_index
            self._rich_editor_anchor_placement = None
        x, y, self._rich_editor_anchor_placement = _rich_editor_preferred_position(
            region_left=left_anchor.x(),
            region_top=top_anchor.y(),
            region_right=right_anchor.x(),
            region_bottom=bottom_anchor.y(),
            popup_width=popup_w,
            popup_height=popup_h,
            available=available,
            margin=margin,
            preserve_top=preserve_popup_top,
            previous_top=previous_top,
        )
        self.rich_text_editor.move(x, y)

    def hideEvent(self, event):
        editor = self.rich_text_editor
        if editor is not None:
            # Preserve only an actually visible popup. If the user previously
            # closed/hid it, switching pages must not resurrect it.
            self._rich_editor_restore_on_show = self._rich_editor_restore_on_show or (
                self._rich_text_popup_enabled and editor.isVisible()
            )
            editor.hide()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if self._rich_editor_restore_on_show:
            # Wait until the stacked page and graphics viewport have their
            # final geometry before restoring the desktop-positioned window.
            QTimer.singleShot(0, self._restore_rich_text_editor_after_page_show)

    def _restore_rich_text_editor_after_page_show(self):
        if not self.isVisible():
            return
        should_restore = self._rich_editor_restore_on_show
        self._rich_editor_restore_on_show = False
        if not should_restore or not self._rich_text_popup_enabled:
            return
        selected = self.model.get_selection()
        if selected and len(selected) == 1:
            self._on_selection_changed_for_rich_editor(selected)
        elif self._rich_text_popup_pinned and self.rich_text_editor is not None:
            self.rich_text_editor.clear_region(hide=False)
            self.rich_text_editor.show()
            self.rich_text_editor.raise_()

    def _connect_signals(self):
        # Model changes fan out to each view that owns the corresponding UI state.
        for slot in (
            self.region_list_view.update_selection,
            self.property_panel.on_selection_changed,
            self._on_selection_changed_for_toolbar,
            self._on_selection_changed_for_rich_editor,
        ):
            self.model.selection_changed.connect(slot)
        for signal, slot in (
            (self.model.regions_changed, self.region_list_view.on_regions_changed),
            (self.model.regions_changed, self._on_regions_changed_for_rich_editor),
            (
                self.model.brush_size_changed,
                self.property_panel.sync_brush_size_from_model,
            ),
            (
                self.model.brush_color_changed,
                self.property_panel.sync_brush_color_from_model,
            ),
            (
                self.model.active_tool_changed,
                self.property_panel.sync_active_tool_from_model,
            ),
            (self.model.compare_image_changed, self.original_compare_view.update_image),
            (
                self.model.original_image_alpha_changed,
                self.toolbar.set_original_image_alpha_slider,
            ),
        ):
            signal.connect(slot)

        for signal, slot in (
            (self.region_list_view.region_selected, self._on_region_selected_from_list),
            (
                self.region_list_view.region_move_requested,
                self._on_region_move_requested_from_list,
            ),
            (self.apply_translations_button.clicked, self._on_apply_changes_clicked),
            (self.replace_all_button.clicked, self._on_replace_all_clicked),
            (self.add_files_button.clicked, self.logic.open_and_add_files),
            (self.add_folder_button.clicked, self.logic.open_and_add_folder),
            (self.clear_list_button.clicked, self.logic.clear_list),
            (self.file_list.file_remove_requested, self._on_file_remove_requested),
            (self.file_list.file_selected, self.logic.load_image_into_editor),
            (self.file_list.files_dropped, self.logic.add_files_from_paths),
            (self.logic.file_list_loading, self.file_list.set_loading),
            (self.logic.file_snapshot_changed, self.file_list.set_snapshot),
            (self.logic.file_list_error, self.file_list.set_error),
        ):
            signal.connect(slot)

        for signal, slot in (
            (self.toolbar.save_requested, self.save_editor_state),
            (self.toolbar.export_requested, self.export_image),
            (self.toolbar.undo_requested, self.controller.undo),
            (self.toolbar.redo_requested, self.controller.redo),
            (self.toolbar.zoom_in_requested, self.graphics_view.zoom_in),
            (self.toolbar.zoom_out_requested, self.graphics_view.zoom_out),
            (self.toolbar.fit_window_requested, self.graphics_view.fit_to_window),
            (self.toolbar.display_mode_changed, self.controller.set_display_mode),
            (
                self.toolbar.original_image_alpha_changed,
                self.controller.set_original_image_alpha,
            ),
            (self.toolbar.align_requested, self._on_align_requested),
            (self.toolbar.distribute_requested, self._on_distribute_requested),
        ):
            signal.connect(slot)
        for signal, key in (
            (self.toolbar.snap_enabled_changed, "editor_snap_enabled"),
            (self.toolbar.center_scale_enabled_changed, "editor_center_scale_enabled"),
            (self.toolbar.auto_save_on_switch_changed, "editor_auto_save_on_switch"),
            (
                self.toolbar.auto_export_on_switch_changed,
                "editor_auto_export_on_switch",
            ),
            (
                self.toolbar.suppress_unsaved_warning_changed,
                "editor_suppress_unsaved_warning",
            ),
            (
                self.toolbar.rich_text_popup_enabled_changed,
                "editor_rich_text_popup_enabled",
            ),
            (
                self.toolbar.rich_text_popup_pinned_changed,
                "editor_rich_text_popup_pinned",
            ),
            (self.toolbar.auto_rich_text_rules_changed, "editor_auto_rich_text_rules"),
            (self.toolbar.delete_and_recover_changed, "editor_delete_and_recover"),
        ):
            signal.connect(partial(self._persist_editor_setting, key))
        if self.config_service is not None:
            self.config_service.config_changed.connect(self._on_config_changed)

        for signal, slot in (
            (
                self.graphics_view.region_geometry_changed,
                self.controller.update_region_geometry,
            ),
            (
                self.graphics_view.view_state_changed,
                self.original_compare_view.sync_view_state,
            ),
            (
                self.graphics_view.view_state_changed,
                self._position_rich_text_editor_for_selection,
            ),
            (
                self.graphics_view.region_drag_started,
                self._hide_rich_text_editor_for_region_drag,
            ),
            (
                self.graphics_view.region_drag_finished,
                self._restore_rich_text_editor_after_region_drag,
            ),
            (
                self.graphics_view.blank_canvas_pressed,
                self._hide_rich_text_editor_for_region_drag,
            ),
        ):
            signal.connect(slot)

        # Property edits that create undoable document changes remain commands.
        for signal, slot in (
            (
                self.property_panel.translated_text_modified,
                self.controller.update_translated_text,
            ),
            (
                self.property_panel.translation_raw_modified,
                self.controller.update_translation_raw,
            ),
            (
                self.property_panel.original_text_modified,
                self.controller.update_original_text,
            ),
            (self.property_panel.ocr_requested, self.controller.run_ocr_for_selection),
            (
                self.property_panel.translation_requested,
                self.controller.run_translation_for_selection,
            ),
            (self.property_panel.font_size_changed, self.controller.update_font_size),
            (self.property_panel.font_color_changed, self.controller.update_font_color),
            (
                self.property_panel.stroke_color_changed,
                self.controller.update_stroke_color,
            ),
            (
                self.property_panel.stroke_width_changed,
                self.controller.update_stroke_width,
            ),
            (
                self.property_panel.line_spacing_changed,
                self.controller.update_line_spacing,
            ),
            (
                self.property_panel.letter_spacing_changed,
                self.controller.update_letter_spacing,
            ),
            (self.property_panel.angle_changed, self.controller.update_angle),
            (
                self.property_panel.font_family_changed,
                self.controller.update_font_family,
            ),
            (
                self.property_panel.font_family_preview_requested,
                self._on_font_family_preview_requested,
            ),
            (self.property_panel.alignment_changed, self.controller.update_alignment),
            (self.property_panel.direction_changed, self.controller.update_direction),
            (
                self.property_panel.style_patch_requested,
                self.controller.update_region_style_patch,
            ),
            (self.property_panel.copy_region_requested, self._handle_copy_from_panel),
            (self.property_panel.paste_region_requested, self._handle_paste_from_panel),
            (
                self.property_panel.delete_region_requested,
                self._handle_delete_from_panel,
            ),
            (
                self.property_panel.clear_all_masks_requested,
                self.controller.clear_all_masks,
            ),
            (
                self.property_panel.clear_paint_overlay_requested,
                self.controller.clear_paint_overlay,
            ),
            (
                self.property_panel.clear_stamp_overlay_requested,
                self.controller.clear_stamp_overlay,
            ),
            (
                self.property_panel.paint_overlay_visibility_changed,
                self.graphics_view.overlay_layers.set_paint_overlay_visible,
            ),
            (
                self.property_panel.stamp_overlay_visibility_changed,
                self.graphics_view.overlay_layers.set_stamp_overlay_visible,
            ),
        ):
            signal.connect(slot)

        # Simple tool state belongs directly to the editor model.
        self.property_panel.toggle_mask_visibility.connect(
            lambda visible: self.model.set_display_mask_type(
                "refined" if visible else "none"
            )
        )
        self.property_panel.mask_tool_changed.connect(self.model.set_active_tool)
        self.property_panel.brush_size_changed.connect(self.model.set_brush_size)
        self.property_panel.brush_color_changed.connect(self.model.set_brush_color)

        if self.rich_text_editor is not None:
            self.rich_text_editor.rich_text_changed.connect(
                self.controller.update_translation_rich
            )
            self.rich_text_editor.layout_size_changed.connect(
                self._position_rich_text_editor_for_selection
            )
        self.app_logic.render_setting_changed.connect(
            self.controller.handle_global_render_setting_change
        )

    @pyqtSlot(list, str)
    def _on_font_family_preview_requested(self, region_indices: list, family: str):
        if self.graphics_view is None:
            return
        if family:
            self.graphics_view.preview_region_style(
                region_indices, {"font_family": family}
            )
        else:
            self.graphics_view.clear_region_style_preview()

    def _create_center_panel(self) -> QWidget:
        """Create the central canvas area"""
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        self.compare_preview_container = QWidget()
        compare_layout = QVBoxLayout(self.compare_preview_container)
        compare_layout.setContentsMargins(0, 0, 0, 0)
        compare_layout.setSpacing(0)
        self.original_compare_view = OriginalCompareView(parent=self)
        compare_layout.addWidget(self.original_compare_view)
        self.compare_preview_container.hide()

        self.edit_canvas_container = QWidget()
        edit_canvas_layout = QVBoxLayout(self.edit_canvas_container)
        edit_canvas_layout.setContentsMargins(0, 0, 0, 0)
        edit_canvas_layout.setSpacing(0)

        # Canvas (the scroll bars are configured in GraphicsView)
        self.graphics_view = GraphicsView(
            self.model, controller=self.controller, parent=self, editor_view=self
        )
        self.graphics_view.set_snap_enabled(self._snap_enabled)
        self.graphics_view.set_center_scale_enabled(self._center_scale_enabled)
        self.original_compare_view.set_source_view(self.graphics_view)
        edit_canvas_layout.addWidget(self.graphics_view)
        # Keep QObject ownership under the editor page (theme refresh/lifetime),
        # while the Tool window flag makes it a real top-level window that is
        # not clipped by the graphics-view canvas.
        self.rich_text_editor = RichTextFloatingEditor(self)

        center_layout.addWidget(self.compare_preview_container, 1)
        center_layout.addWidget(self.edit_canvas_container, 1)

        return center_widget

    def _create_right_panel(self) -> QWidget:
        """Create the file list panel on the right"""
        right_panel = QWidget()
        right_panel.setMinimumWidth(220)
        right_panel.setMaximumWidth(300)
        right_panel.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # File operation buttons
        file_button_widget = CardWidget()
        file_button_widget.setFixedHeight(64)
        file_buttons_layout = QHBoxLayout(file_button_widget)
        file_buttons_layout.setContentsMargins(12, 10, 12, 10)
        file_buttons_layout.setSpacing(10)
        file_action_size = QSize(48, 40)
        file_action_icon_size = QSize(22, 22)

        self.add_files_button = ToolButton()
        self.add_files_button.setIcon(FIF.ADD)
        self.add_files_button.setFixedSize(file_action_size)
        self.add_files_button.setIconSize(file_action_icon_size)
        set_hover_hint(self.add_files_button, self._t("Add Files"))
        self.add_folder_button = ToolButton()
        self.add_folder_button.setIcon(FIF.FOLDER_ADD)
        self.add_folder_button.setFixedSize(file_action_size)
        self.add_folder_button.setIconSize(file_action_icon_size)
        set_hover_hint(self.add_folder_button, self._t("Add Folder"))
        self.clear_list_button = ToolButton()
        self.clear_list_button.setIcon(FIF.DELETE)
        self.clear_list_button.setFixedSize(file_action_size)
        self.clear_list_button.setIconSize(file_action_icon_size)
        set_hover_hint(self.clear_list_button, self._t("Clear List"))
        file_buttons_layout.addStretch()
        file_buttons_layout.addWidget(self.add_files_button)
        file_buttons_layout.addWidget(self.add_folder_button)
        file_buttons_layout.addWidget(self.clear_list_button)
        file_buttons_layout.addStretch()
        right_layout.addWidget(file_button_widget)

        # File list
        file_list_card = CardWidget()
        file_list_layout = QVBoxLayout(file_list_card)
        file_list_layout.setContentsMargins(8, 8, 8, 8)
        file_list_layout.setSpacing(0)
        self.file_list = FileListView(
            None,
            self,
            data_service=getattr(self.app_logic, "file_list_data_service", None),
        )
        file_list_layout.addWidget(self.file_list)
        right_layout.addWidget(file_list_card, 1)

        return right_panel

    @pyqtSlot(str)
    def _on_file_remove_requested(self, file_path: str):
        """Handle a file removal request: only the editor's own file list is handled"""
        # Remove from the view first (to avoid rebuilding the list)
        self.file_list.remove_file(file_path)

        # Call editor_logic to remove the file (it checks whether the canvas has to be cleared)
        self.logic.remove_file(file_path)

        # The editor has its own file list and does not need to sync with app_logic of the main page

    def _apply_editor_style(self, theme: str | None = None):
        """Refresh the theme of the canvas and the custom colour controls."""
        from ui.widgets.color_picker import ColorPickerWidget

        if self.toolbar is not None:
            self.toolbar.refresh_theme()
        if self.graphics_view is not None:
            self.graphics_view.apply_theme(theme)
        if self.original_compare_view is not None:
            self.original_compare_view.apply_theme(theme)
        if self.file_list is not None:
            self.file_list.refresh_empty_state_text()
        if self.rich_text_editor is not None:
            self.rich_text_editor.refresh_theme()
        for picker in self.findChildren(ColorPickerWidget):
            picker.refresh_theme()

    def set_compare_mode(self, enabled: bool):
        enabled = bool(enabled)
        if self.compare_preview_container is not None:
            self.compare_preview_container.setVisible(enabled)
        if self.original_compare_view is not None:
            self.original_compare_view.set_compare_mode(
                enabled, self.model.get_compare_image()
            )
