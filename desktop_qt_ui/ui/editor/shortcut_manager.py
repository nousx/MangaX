"""
Shortcut management module.
Responsible for setting up and handling all shortcuts of the Qt UI in one place
"""

from functools import partial
from typing import Callable, Optional

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QKeyEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QLineEdit, QTextEdit, QWidget


class ShortcutManager(QObject):
    """
    Shortcut manager.
    Manages all shortcuts of the application in one place
    """

    def __init__(self, parent: QWidget):
        """
        Initialise the shortcut manager

        Args:
            parent: the parent widget
        """
        super().__init__(parent)
        self.parent_widget = parent
        self.shortcuts = {}

    def register_shortcut(
        self,
        name: str,
        key_sequence: QKeySequence.StandardKey,
        callback: Callable,
        context_aware: bool = False,
    ) -> QShortcut:
        """
        Register a shortcut

        Args:
            name: name of the shortcut (used to identify it)
            key_sequence: the key sequence
            callback: the callback function
            context_aware: whether it is context aware (checks the focused widget)

        Returns:
            The QShortcut object that was created
        """
        shortcut = QShortcut(key_sequence, self.parent_widget)

        if context_aware:
            # Wrap the callback, adding a context check
            def context_aware_callback():
                # parent_widget.focusWidget() does not cross windows: while the focus is in the floating editor
                # (a Qt.Tool top-level window) it still returns the old focus inside the main window, which deleted the selected canvas region by mistake.
                focused_widget = QApplication.focusWidget()
                if (
                    focused_widget is not None
                    and focused_widget.window() is not self.parent_widget.window()
                ):
                    # The focus is in another top-level window (such as the floating rich-text editor): editor shortcuts are not handled at all
                    return
                callback(focused_widget)

            shortcut.activated.connect(context_aware_callback)
        else:
            shortcut.activated.connect(callback)

        self.shortcuts[name] = shortcut
        return shortcut

    def get_shortcut(self, name: str) -> Optional[QShortcut]:
        """
        Get a shortcut object

        Args:
            name: name of the shortcut

        Returns:
            The QShortcut object, or None when it does not exist
        """
        return self.shortcuts.get(name)

    @staticmethod
    def is_text_widget(widget) -> bool:
        """
        Check whether a widget is a text editing widget

        Args:
            widget: the widget to check

        Returns:
            Whether it is a text editing widget
        """
        return isinstance(widget, (QTextEdit, QLineEdit))


class EditorShortcutManager(ShortcutManager):
    """
    Editor shortcut manager.
    Manages the shortcuts of the editor view specifically
    """

    def __init__(self, editor_view):
        """
        Initialise the editor shortcut manager

        Args:
            editor_view: the editor view object
        """
        super().__init__(editor_view)
        self.editor_view = editor_view
        self.controller = editor_view.controller
        self._setup_editor_shortcuts()
        self._setup_wheel_shortcuts()

    def _setup_editor_shortcuts(self):
        """Register editor shortcuts from one explicit policy table."""
        panel = self.editor_view.property_panel
        shortcuts = (
            ("undo", QKeySequence.StandardKey.Undo, self._handle_undo, True),
            ("redo", QKeySequence.StandardKey.Redo, self._handle_redo, True),
            ("copy", QKeySequence.StandardKey.Copy, self._handle_copy, True),
            ("paste", QKeySequence.StandardKey.Paste, self._handle_paste, True),
            (
                "select_all",
                QKeySequence.StandardKey.SelectAll,
                self._handle_select_all,
                True,
            ),
            ("delete", QKeySequence.StandardKey.Delete, self._handle_delete, True),
            ("save", QKeySequence.StandardKey.Save, self._handle_save, True),
            ("export", QKeySequence("Ctrl+Q"), self._handle_export, True),
            (
                "toggle_rich_text_popup",
                QKeySequence("Ctrl+Shift+R"),
                self._handle_toggle_rich_text_popup,
                False,
            ),
            (
                "tool_select",
                QKeySequence("Q"),
                partial(
                    self._handle_panel_shortcut,
                    0,
                    Qt.Key.Key_Q,
                    "q",
                    "tool_select",
                    panel.activate_image_edit_tool,
                ),
                True,
            ),
            (
                "tool_brush",
                QKeySequence("W"),
                partial(
                    self._handle_panel_shortcut,
                    1,
                    Qt.Key.Key_W,
                    "w",
                    "tool_brush",
                    panel.activate_image_edit_tool,
                ),
                True,
            ),
            (
                "tool_eraser",
                QKeySequence("E"),
                partial(
                    self._handle_panel_shortcut,
                    2,
                    Qt.Key.Key_E,
                    "e",
                    "tool_eraser",
                    panel.activate_image_edit_tool,
                ),
                True,
            ),
            (
                "image_edit_tab_mask",
                QKeySequence("1"),
                partial(
                    self._handle_panel_shortcut,
                    0,
                    Qt.Key.Key_1,
                    "1",
                    "image_edit_tab_mask",
                    panel.activate_image_edit_tab,
                ),
                True,
            ),
            (
                "image_edit_tab_paint",
                QKeySequence("2"),
                partial(
                    self._handle_panel_shortcut,
                    1,
                    Qt.Key.Key_2,
                    "2",
                    "image_edit_tab_paint",
                    panel.activate_image_edit_tab,
                ),
                True,
            ),
            (
                "image_edit_tab_stamp",
                QKeySequence("3"),
                partial(
                    self._handle_panel_shortcut,
                    2,
                    Qt.Key.Key_3,
                    "3",
                    "image_edit_tab_stamp",
                    panel.activate_image_edit_tab,
                ),
                True,
            ),
            (
                "prev_image",
                QKeySequence("A"),
                partial(
                    self._handle_navigation,
                    Qt.Key.Key_A,
                    "a",
                    "prev_image",
                    self.editor_view.file_list.select_prev_image,
                ),
                True,
            ),
            (
                "next_image",
                QKeySequence("D"),
                partial(
                    self._handle_navigation,
                    Qt.Key.Key_D,
                    "d",
                    "next_image",
                    self.editor_view.file_list.select_next_image,
                ),
                True,
            ),
            (
                "toggle_text_direction",
                QKeySequence("V"),
                self._handle_toggle_text_direction,
                True,
            ),
        )
        for name, key, callback, context_aware in shortcuts:
            self.register_shortcut(name, key, callback, context_aware)

        # This one must also close the Qt.Tool popup when that window has focus.
        self.get_shortcut("toggle_rich_text_popup").setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )

    def _handle_undo(self, focused_widget):
        """Handle the undo shortcut"""
        if self.is_text_widget(focused_widget):
            # When the focus is on a text control, let it handle undo
            focused_widget.undo()
        else:
            # Otherwise call the editor's undo
            self.controller.undo()

    def _handle_redo(self, focused_widget):
        """Handle the redo shortcut"""
        if self.is_text_widget(focused_widget):
            # When the focus is on a text control, let it handle redo
            focused_widget.redo()
        else:
            # Otherwise call the editor's redo
            self.controller.redo()

    def _handle_copy(self, focused_widget):
        """Handle the copy shortcut"""
        if self.is_text_widget(focused_widget):
            # When the focus is on a text control, let it handle copy
            focused_widget.copy()
        else:
            # When a paste overlay is selected on the canvas, copy the overlay first
            graphics_view = getattr(self.editor_view, "graphics_view", None)
            overlay_id = getattr(
                graphics_view, "_selected_paste_overlay_id", None
            )
            if overlay_id:
                self.controller.copy_paste_overlay(overlay_id)
                return
            # Otherwise copy the selected regions
            selected_regions = self.editor_view.model.get_selection()
            if selected_regions:
                self.controller.copy_regions(selected_regions)

    def _handle_paste(self, focused_widget):
        """Handle the paste shortcut"""
        if self.is_text_widget(focused_widget):
            # When the focus is on a text control, let it handle paste
            focused_widget.paste()
        else:
            # Otherwise what paste does depends on whether a region is selected
            selected_regions = self.editor_view.model.get_selection()
            if selected_regions and len(selected_regions) == 1:
                # With a single selected region, paste the style
                self.controller.paste_region_style(selected_regions[0])
            elif selected_regions:
                # Several regions selected: the existing logic applies (paste a new region at the mouse position)
                self._paste_new_region_at_cursor()
            else:
                last_kind = (
                    self.controller.last_clipboard_kind()
                    if hasattr(self.controller, "last_clipboard_kind")
                    else None
                )
                has_overlay = self.controller.paste_overlay_clipboard_available()
                has_region = bool(
                    getattr(self.controller, "history_service", None)
                    and self.controller.history_service.has_clipboard_data()
                )

                if last_kind == "paste_overlay" and has_overlay:
                    self._paste_paste_overlay_at_cursor()
                elif last_kind == "region" and has_region:
                    self._paste_new_region_at_cursor()
                elif has_overlay:
                    self._paste_paste_overlay_at_cursor()
                else:
                    self._paste_new_region_at_cursor()

    def _paste_paste_overlay_at_cursor(self) -> None:
        """No region selected and the overlay clipboard holds something: paste the overlay at the mouse position.

        Overlay geometry is in scene coordinates (source image pixels), so the image-local coordinates of _cursor_image_position cannot be used.
        """
        mouse_scene_pos = self._cursor_scene_position()
        if self.controller.paste_paste_overlay(mouse_scene_pos):
            graphics_view = getattr(self.editor_view, "graphics_view", None)
            if graphics_view is not None:
                overlays = self.editor_view.model.get_paste_overlays()
                if overlays:
                    graphics_view.select_paste_overlay(overlays[-1]["id"])

    def _cursor_image_position(self):
        """Convert the current mouse position to image (scene) coordinates; None when there is no canvas."""
        from PyQt6.QtGui import QCursor

        graphics_view = getattr(self.editor_view, "graphics_view", None)
        if not graphics_view or not graphics_view._image_item:
            return None
        mouse_pos_scene = graphics_view.mapToScene(
            graphics_view.mapFromGlobal(QCursor.pos())
        )
        return graphics_view._image_item.mapFromScene(mouse_pos_scene)

    def _cursor_scene_position(self):
        """Convert the current mouse position to scene coordinates (the overlay coordinate system); None when there is no canvas."""
        from PyQt6.QtGui import QCursor

        graphics_view = getattr(self.editor_view, "graphics_view", None)
        if not graphics_view or not graphics_view._image_item:
            return None
        return graphics_view.mapToScene(graphics_view.mapFromGlobal(QCursor.pos()))

    def _paste_new_region_at_cursor(self):
        """Paste a new region at the mouse position when no region is selected (the original behaviour outside the overlay branch)."""
        mouse_pos_image = self._cursor_image_position()
        if mouse_pos_image is not None:
            self.controller.paste_region(mouse_pos_image)
        else:
            self.controller.paste_region()

    def _handle_select_all(self, focused_widget):
        """Handle the select-all shortcut"""
        if self.is_text_widget(focused_widget):
            focused_widget.selectAll()
        else:
            graphics_view = getattr(self.editor_view, "graphics_view", None)
            if graphics_view is not None:
                graphics_view.clear_paste_overlay_selection()
            regions = self.editor_view.model.get_regions()
            self.editor_view.model.set_selection(list(range(len(regions))))

    def _handle_delete(self, focused_widget):
        """Handle the delete shortcut"""
        if not self.is_text_widget(focused_widget):
            # When a paste overlay is selected on the canvas, delete the overlay first
            graphics_view = getattr(self.editor_view, "graphics_view", None)
            if (
                graphics_view is not None
                and getattr(graphics_view, "_selected_paste_overlay_id", None)
            ):
                graphics_view.delete_selected_paste_overlay()
                return
            # Otherwise delete the selected regions
            selected_regions = self.editor_view.model.get_selection()
            if selected_regions:
                self.controller.delete_regions(selected_regions)
                return

    def _handle_save(self, focused_widget):
        """Handle the save shortcut (Ctrl+S)."""
        self.editor_view.save_editor_state()

    def _handle_export(self, focused_widget):
        """Handle the export shortcut (Ctrl+Q)"""
        # Shares one entry point with the toolbar, so the rich-text body and the ruby are flushed before the model is read.
        self.editor_view.export_image()

    def _handle_toggle_rich_text_popup(self):
        """Toggle the floating rich-text editor (Ctrl+Shift+R)."""
        if not self.editor_view.isVisible():
            return
        toolbar = getattr(self.editor_view, "toolbar", None)
        if toolbar is None:
            return
        toolbar.set_rich_text_popup_enabled(
            not toolbar.is_rich_text_popup_enabled(), emit=True
        )

    def _forward_key_to_widget(self, widget, key_code, text, shortcut_name):
        """Forward a text key while preventing its editor shortcut from recurring."""
        shortcut = self.get_shortcut(shortcut_name)
        if shortcut is None:
            return
        shortcut.setEnabled(False)
        try:
            for event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                QApplication.sendEvent(
                    widget,
                    QKeyEvent(
                        event_type,
                        key_code,
                        Qt.KeyboardModifier.NoModifier,
                        text,
                    ),
                )
        finally:
            shortcut.setEnabled(True)

    def _handle_panel_shortcut(
        self, index, key_code, text, name, activate, focused_widget
    ):
        if self.is_text_widget(focused_widget):
            self._forward_key_to_widget(focused_widget, key_code, text, name)
        else:
            activate(index)

    def _handle_toggle_text_direction(self, focused_widget):
        """Press V to switch the selected text boxes between horizontal and vertical."""
        if self.is_text_widget(focused_widget):
            self._forward_key_to_widget(
                focused_widget, Qt.Key.Key_V, "v", "toggle_text_direction"
            )
            return

        selected_regions = self.editor_view.model.get_selection()
        if not selected_regions:
            return

        regions = self.editor_view.model.get_regions()
        anchor_index = selected_regions[-1]
        if not 0 <= anchor_index < len(regions):
            return

        anchor_region = regions[anchor_index]
        direction = str(anchor_region.get("direction", "")).strip().lower()
        if direction in ("v", "vertical"):
            is_vertical = True
        elif direction in ("h", "horizontal"):
            is_vertical = False
        else:
            white_frame = self.editor_view.property_panel._calculate_white_frame_info(
                anchor_region
            )
            is_vertical = bool(white_frame and white_frame[3] > white_frame[2])

        next_direction = "horizontal" if is_vertical else "vertical"
        self.controller.update_region_style_patch(
            selected_regions, {"direction": next_direction}
        )

    def _handle_navigation(self, key_code, text, name, navigate, focused_widget):
        if self.is_text_widget(focused_widget):
            self._forward_key_to_widget(focused_widget, key_code, text, name)
        else:
            navigate()

    def _setup_wheel_shortcuts(self):
        """Set up the mouse wheel shortcuts (implemented with an event filter)"""
        # Install an event filter on the viewport of graphics_view
        if hasattr(self.editor_view, "graphics_view"):
            # Wheel events reach the viewport first
            self.editor_view.graphics_view.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        """
        Event filter that handles the mouse wheel shortcuts

        Supported shortcuts:
        - Ctrl + wheel: scale the selected text boxes proportionally (both the box size and the font)
        - Shift + wheel: adjust the size of the mask brush
        """
        if event.type() == QEvent.Type.Wheel:
            # Check whether it is the viewport of graphics_view
            if obj == self.editor_view.graphics_view.viewport():
                modifiers = event.modifiers()

                # Shift + wheel: change the brush size (whatever the current tool is)
                if modifiers == Qt.KeyboardModifier.ShiftModifier:
                    current_size = self.editor_view.model.get_brush_size()
                    # Try to get the wheel direction
                    angle_delta = event.angleDelta().y()
                    if angle_delta == 0:
                        angle_delta = event.pixelDelta().y()

                    delta = 1 if angle_delta > 0 else -1
                    new_size = max(5, min(200, current_size + delta))
                    self.editor_view.model.set_brush_size(new_size)
                    return True  # Stop the event from propagating

                # Ctrl + wheel (including combinations such as Ctrl+Shift): change the font size of the selected text boxes;
                # with no text box selected but a paste overlay selected, scale the overlay proportionally.
                # The event is swallowed whether or not something is selected - this gesture means "change font size / scale overlay"
                # and must never fall through to canvas zoom, where the user thinks the font size is changing while the canvas zooms.
                elif modifiers & Qt.KeyboardModifier.ControlModifier:
                    angle_delta = event.angleDelta().y()
                    if angle_delta == 0:
                        angle_delta = event.pixelDelta().y()
                    if angle_delta == 0:
                        return True

                    graphics_view = getattr(
                        self.editor_view, "graphics_view", None
                    )
                    overlay_id = getattr(
                        graphics_view, "_selected_paste_overlay_id", None
                    )
                    if overlay_id:
                        step = 1.05 if angle_delta > 0 else 1.0 / 1.05
                        for overlay in self.editor_view.model.get_paste_overlays():
                            if overlay.get("id") == overlay_id:
                                width = float(overlay.get("width", 1.0))
                                height = float(overlay.get("height", 1.0))
                                self.controller.update_paste_overlay(
                                    overlay_id,
                                    {
                                        "width": round(width * step, 1),
                                        "height": round(height * step, 1),
                                    },
                                )
                                break
                    else:
                        selected_regions = self.editor_view.model.get_selection()
                        if selected_regions:
                            for region_index in selected_regions:
                                region_data = self.editor_view.model.get_region_by_index(
                                    region_index
                                )
                                if region_data:
                                    old_size = region_data.get("font_size", 20)
                                    delta = max(1, int(old_size * 0.05))
                                    new_size = max(
                                        1,
                                        old_size
                                        + (delta if angle_delta > 0 else -delta),
                                    )
                                    self.controller.update_font_size(
                                        region_index, new_size
                                    )
                    return True  # Stop the event from propagating

        # Other events propagate as usual
        return super().eventFilter(obj, event)
