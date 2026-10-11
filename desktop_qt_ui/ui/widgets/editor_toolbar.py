from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QActionGroup
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QProxyStyle,
    QSizePolicy,
    QStyle,
    QStyleFactory,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CardWidget,
    CheckableMenu,
    DropDownPushButton,
    MenuAnimationType,
    MenuIndicatorType,
    PrimaryPushButton,
    PushButton,
    RoundMenu,
    SingleDirectionScrollArea,
    Slider,
    ToolButton,
    VerticalSeparator,
)
from qfluentwidgets import (
    FluentIcon as FIF,
)

from services import get_i18n_manager
from ui.fluent_icon import themed_fluent_svg_icon
from ui.widgets.hover_hint import set_hover_hint


class _LeadingIndicatorMenuStyle(QProxyStyle):
    """Makes a separate column for the selection mark on the left, in the order: mark → icon → text.

    QProxyStyle takes ownership of the base style passed in, so the globally shared
    QApplication.style() instance must never be given to it (destroying the menu view would delete the style of the whole
    application with it). Here QStyleFactory creates a private instance of the same name as the base.
    """

    CONTENT_OFFSET = 24

    def __init__(self):
        app_style = QApplication.style()
        base_style = (
            QStyleFactory.create(app_style.objectName())
            if app_style is not None
            else None
        )
        if base_style is not None:
            super().__init__(base_style)
        else:
            # Constructed without arguments: the proxy uses the application style lazily and does not take ownership of it
            super().__init__()

    def subElementRect(self, element, option, widget=None):
        rect = super().subElementRect(element, option, widget)
        if element == QStyle.SubElement.SE_ItemViewItemDecoration:
            rect.translate(self.CONTENT_OFFSET, 0)
        elif element == QStyle.SubElement.SE_ItemViewItemText:
            rect.setLeft(rect.left() + self.CONTENT_OFFSET)
        return rect


class _ScreenBoundMenuMixin:
    """Shift a centered popup back on-screen when its anchor is near an edge."""

    def _screen_geometry_for(self, pos: QPoint):
        app = QApplication.instance()
        screen = app.screenAt(pos) if app is not None else None
        if screen is None:
            window_handle = self.windowHandle()
            screen = window_handle.screen() if window_handle is not None else None
        if screen is None and app is not None:
            screen = app.primaryScreen()
        return screen.availableGeometry() if screen is not None else None

    def _safe_popup_position(self, pos: QPoint) -> QPoint:
        rect = self._screen_geometry_for(pos)
        if rect is None:
            return pos

        margins = self.layout().contentsMargins()
        # RoundMenu computes x as ``anchor.x() - left_margin`` without a
        # corresponding lower bound.  Clamp the anchor only when that would
        # place the popup outside the left edge; its own right-edge handling is
        # retained for the opposite side.
        minimum_anchor_x = rect.left() + margins.left()
        if pos.x() < minimum_anchor_x:
            return QPoint(minimum_anchor_x, pos.y())
        return pos

    def exec(self, pos, ani=True, aniType=MenuAnimationType.DROP_DOWN):
        super().exec(self._safe_popup_position(pos), ani, aniType)


class _IconCheckableMenu(_ScreenBoundMenuMixin, CheckableMenu):
    """CheckableMenu with a separate selection mark column on the left and a semantic icon column."""

    def __init__(self, title="", parent=None, indicatorType=MenuIndicatorType.CHECK):
        super().__init__(title, parent, indicatorType)
        self._leading_indicator_style = _LeadingIndicatorMenuStyle()
        self._leading_indicator_style.setParent(self.view)
        self.view.setStyle(self._leading_indicator_style)


class _ScreenBoundCheckableMenu(_ScreenBoundMenuMixin, CheckableMenu):
    """Checkable menu variant used by the display-mode popup."""


class _StayOpenCheckableMenu(_IconCheckableMenu):
    """Single-choice menu that does not close when an option is clicked.

    The arrange menu needs several operations in a row after opening once (switching the reference, aligning and distributing repeatedly);
    _onItemClicked of the parent class calls _hideMenu before triggering the action, and the closing is skipped here.
    The menu still closes normally by a click outside or Esc.
    """

    def _onItemClicked(self, item):
        action = item.data(Qt.ItemDataRole.UserRole)
        if action not in self._actions or not action.isEnabled():
            return
        action.trigger()


class EditorToolbar(CardWidget):
    """
    Toolbar at the top of the editor. Only fit-to-window and the opacity slider of the original image stay as permanent controls;
    the other operations go into three single-level drop-down menus (no nesting):
    "Menu" = undo/redo/zoom + general switches; "Display mode" = single choice of the canvas display;
    "Arrange" = single choice of the reference + align/distribute text options (a click does not close it, for repeated use).
    There is no entry for returning to the main page: the sidebar of the main window can switch pages at any time.
    """

    save_requested = pyqtSignal()
    export_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    fit_window_requested = pyqtSignal()
    display_mode_changed = pyqtSignal(str)
    original_image_alpha_changed = pyqtSignal(int)
    align_requested = pyqtSignal(str)
    distribute_requested = pyqtSignal(str)
    snap_enabled_changed = pyqtSignal(bool)
    center_scale_enabled_changed = pyqtSignal(bool)
    rich_text_popup_enabled_changed = pyqtSignal(bool)
    rich_text_popup_pinned_changed = pyqtSignal(bool)
    auto_save_on_switch_changed = pyqtSignal(bool)
    auto_export_on_switch_changed = pyqtSignal(bool)
    suppress_unsaved_warning_changed = pyqtSignal(bool)
    auto_rich_text_rules_changed = pyqtSignal(bool)
    delete_and_recover_changed = pyqtSignal(bool)

    def __init__(
        self,
        parent=None,
        snap_enabled: bool = False,
        rich_text_popup_enabled: bool = True,
        rich_text_popup_pinned: bool = False,
        auto_save_on_switch: bool = True,
        auto_export_on_switch: bool = True,
        suppress_unsaved_warning: bool = False,
        center_scale_enabled: bool = False,
        auto_rich_text_rules: bool = True,
        delete_and_recover: bool = False,
    ):
        super().__init__(parent)
        self.i18n = get_i18n_manager()
        self._themed_icon_buttons: list[tuple[ToolButton, str]] = []
        self.content_widget: QWidget | None = None
        # The menus are rebuilt as a whole on a language switch; all state that has to be restored is kept in fields
        self._display_mode = "full"
        self._align_ref = "selection"
        self._can_undo = False
        self._can_redo = False
        self._export_enabled = True
        self._last_selection_count = 0
        self._snap_enabled = bool(snap_enabled)
        self._center_scale_enabled = bool(center_scale_enabled)
        self._rich_text_popup_enabled = bool(rich_text_popup_enabled)
        self._rich_text_popup_pinned = bool(rich_text_popup_pinned)
        self._auto_save_on_switch = bool(auto_save_on_switch)
        self._auto_export_on_switch = bool(auto_export_on_switch)
        self._suppress_unsaved_warning = bool(suppress_unsaved_warning)
        self._auto_rich_text_rules = bool(auto_rich_text_rules)
        self._delete_and_recover = bool(delete_and_recover)
        self.main_menu: RoundMenu | None = None
        self.display_menu: RoundMenu | None = None
        self.arrange_menu: RoundMenu | None = None
        self._init_ui()
        self._connect_signals()

    def _t(self, key: str, **kwargs) -> str:
        """Translation helper methods"""
        if self.i18n:
            return self.i18n.translate(key, **kwargs)
        return key

    def _init_ui(self):
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(54)

        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(6, 4, 6, 4)
        outer_layout.setSpacing(0)

        self.scroll_area = SingleDirectionScrollArea(self, Qt.Orientation.Horizontal)
        self.scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.scroll_area.setMinimumHeight(44)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setFrameShape(SingleDirectionScrollArea.Shape.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.enableTransparentBackground()
        outer_layout.addWidget(self.scroll_area)

        # Constructed without a parent: setWidget takes ownership, and attaching to the frame first would leave a ghost placeholder
        self.content_widget = QWidget()
        layout = QHBoxLayout(self.content_widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        # --- Drop-down menu group: general / display mode / arrange (each is a single-level menu; functions are not nested) ---
        self.menu_button = DropDownPushButton()
        self.menu_button.setIcon(FIF.MENU)
        self.menu_button.setText(self._t("Menu"))
        layout.addWidget(self.menu_button)

        self.display_mode_button = DropDownPushButton()
        self.display_mode_button.setIcon(FIF.VIEW)
        self.display_mode_button.setText(self._t("Display Mode"))
        layout.addWidget(self.display_mode_button)

        self.arrange_button = DropDownPushButton()
        self.arrange_button.setIcon(FIF.LAYOUT)
        self.arrange_button.setText(self._t("Arrange"))
        layout.addWidget(self.arrange_button)

        self.save_button = PrimaryPushButton()
        self.save_button.setIcon(FIF.SAVE)
        self.save_button.setText(self._t("Save"))
        self.save_button.clicked.connect(self.save_requested)
        layout.addWidget(self.save_button)

        self.export_button = PushButton()
        self.export_button.setIcon(FIF.IMAGE_EXPORT)
        self.export_button.setText(self._t("Export Image"))
        self.export_button.clicked.connect(self.export_requested)
        layout.addWidget(self.export_button)

        self._build_menus()

        layout.addWidget(self._create_separator())

        # --- Always shown: fit to window ---
        self.fit_window_button = ToolButton()
        self.fit_window_button.setIcon(FIF.FIT_PAGE)
        set_hover_hint(self.fit_window_button, self._t("Fit to Window"))
        layout.addWidget(self.fit_window_button)

        layout.addWidget(self._create_separator())

        # --- Always shown: opacity of the original image ---
        self.opacity_label = BodyLabel(self._t("Original Image Opacity:"))
        layout.addWidget(self.opacity_label)
        self.original_image_alpha_slider = Slider(Qt.Orientation.Horizontal)
        self.original_image_alpha_slider.setRange(0, 100)
        self.original_image_alpha_slider.setValue(
            0
        )  # Default to 0 (fully transparent, show inpainted)
        self.original_image_alpha_slider.setMinimumWidth(140)
        layout.addWidget(self.original_image_alpha_slider)

        layout.addStretch()  # Pushes everything to the left
        self.content_widget.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self.scroll_area.setWidget(self.content_widget)
        self.scroll_area.enableTransparentBackground()
        self._sync_content_width()

    # ------------------------------------------------------------------
    # Main menu
    # ------------------------------------------------------------------

    def _build_menus(self):
        """Build the three independent single-level drop-down menus. On a language switch they are rebuilt as a whole, and the state is restored from the fields."""
        old_menus = [self.main_menu, self.display_menu, self.arrange_menu]
        # A popup menu must have a real top-level window as its QWidget parent. With the toolbar
        # (inside the QStackedWidget of FluentWindow) as the parent, Qt on Windows
        # tries to activate the QWidgetWindow of that stacked container and logs a "must be a top
        # level window" warning. When the toolbar is tested on its own, outside a window, it is a top-level control itself,
        # so itself is kept as the fallback parent.
        menu_parent = self.window()
        if menu_parent is None or not menu_parent.isWindow():
            menu_parent = self
        # The themed icon buttons of the old menu are about to be destroyed; clear the registry first to prevent dangling references
        self._themed_icon_buttons.clear()
        menu = _IconCheckableMenu(
            parent=menu_parent, indicatorType=MenuIndicatorType.CHECK
        )

        # --- General menu: undo and redo / zoom / persistent switches ---

        # The real undo/redo shortcuts are registered globally by EditorShortcutManager (aware of focus);
        # here they are only shown as a hint in the text, with no QAction shortcut, to avoid triggering twice.
        self.undo_action = Action(FIF.LEFT_ARROW, self._t("Undo") + " (Ctrl+Z)")
        self.undo_action.setEnabled(self._can_undo)
        self.undo_action.triggered.connect(self.undo_requested)
        menu.addAction(self.undo_action)

        self.redo_action = Action(FIF.RIGHT_ARROW, self._t("Redo") + " (Ctrl+Y)")
        self.redo_action.setEnabled(self._can_redo)
        self.redo_action.triggered.connect(self.redo_requested)
        menu.addAction(self.redo_action)
        menu.addSeparator()

        self.zoom_in_action = Action(FIF.ZOOM_IN, self._t("Zoom In (+)"))
        self.zoom_in_action.triggered.connect(self.zoom_in_requested)
        menu.addAction(self.zoom_in_action)

        self.zoom_out_action = Action(FIF.ZOOM_OUT, self._t("Zoom Out (-)"))
        self.zoom_out_action.triggered.connect(self.zoom_out_requested)
        menu.addAction(self.zoom_out_action)

        menu.addSeparator()
        self.snap_action = Action(
            themed_fluent_svg_icon("ic_fluent_target_arrow_24_regular.svg"),
            self._t("Enable Editor Snapping"),
        )
        self.snap_action.setCheckable(True)
        self.snap_action.setChecked(self._snap_enabled)
        self.snap_action.triggered.connect(self._on_snap_action_triggered)
        menu.addAction(self.snap_action)

        self.center_scale_action = Action(
            FIF.ZOOM_IN,
            self._t("Scale Text Boxes from Center"),
        )
        self.center_scale_action.setCheckable(True)
        self.center_scale_action.setChecked(self._center_scale_enabled)
        self.center_scale_action.triggered.connect(
            self._on_center_scale_action_triggered
        )
        menu.addAction(self.center_scale_action)

        self.rich_text_popup_action = Action(
            themed_fluent_svg_icon("ic_fluent_text_edit_style_24_regular.svg"),
            self._t("Show Rich Text Editor Popup") + " (Ctrl+Shift+R)",
        )
        self.rich_text_popup_action.setCheckable(True)
        self.rich_text_popup_action.setChecked(self._rich_text_popup_enabled)
        self.rich_text_popup_action.triggered.connect(
            self._on_rich_text_popup_action_triggered
        )
        menu.addAction(self.rich_text_popup_action)
        self.rich_text_popup_pinned_action = Action(
            self._t("Pin Rich Text Editor Popup")
        )
        self.rich_text_popup_pinned_action.setCheckable(True)
        self.rich_text_popup_pinned_action.setChecked(self._rich_text_popup_pinned)
        self.rich_text_popup_pinned_action.triggered.connect(
            self._on_rich_text_popup_pinned_action_triggered
        )
        menu.addAction(self.rich_text_popup_pinned_action)

        self.auto_rich_text_rules_action = Action(
            FIF.FONT,
            self._t("Auto Apply Rich Text Rules While Editing"),
        )
        self.auto_rich_text_rules_action.setCheckable(True)
        self.auto_rich_text_rules_action.setChecked(self._auto_rich_text_rules)
        self.auto_rich_text_rules_action.triggered.connect(
            self._on_auto_rich_text_rules_action_triggered
        )
        menu.addAction(self.auto_rich_text_rules_action)

        self.auto_save_action = Action(
            FIF.SAVE,
            self._t("Auto Save on Image Switch"),
        )
        self.auto_save_action.setCheckable(True)
        self.auto_save_action.setChecked(self._auto_save_on_switch)
        self.auto_save_action.triggered.connect(self._on_auto_save_action_triggered)
        menu.addAction(self.auto_save_action)

        self.auto_export_action = Action(
            FIF.IMAGE_EXPORT,
            self._t("Auto Export on Image Switch"),
        )
        self.auto_export_action.setCheckable(True)
        self.auto_export_action.setChecked(self._auto_export_on_switch)
        self.auto_export_action.triggered.connect(self._on_auto_export_action_triggered)
        menu.addAction(self.auto_export_action)

        self.suppress_unsaved_warning_action = Action(
            FIF.INFO,
            self._t("Do Not Warn About Unsaved Changes"),
        )
        self.suppress_unsaved_warning_action.setCheckable(True)
        self.suppress_unsaved_warning_action.setChecked(self._suppress_unsaved_warning)
        self.suppress_unsaved_warning_action.triggered.connect(
            self._on_suppress_unsaved_warning_action_triggered
        )
        menu.addAction(self.suppress_unsaved_warning_action)

        self.delete_and_recover_action = Action(
            FIF.DELETE,
            self._t("Delete and Recover Removed Text"),
        )
        self.delete_and_recover_action.setCheckable(True)
        self.delete_and_recover_action.setChecked(self._delete_and_recover)
        self.delete_and_recover_action.triggered.connect(
            self._on_delete_and_recover_action_triggered
        )
        menu.addAction(self.delete_and_recover_action)

        self.main_menu = menu
        self.menu_button.setMenu(menu)

        # --- Display mode menu: a single choice among five canvas display states ---
        display_menu = _ScreenBoundCheckableMenu(
            parent=menu_parent,
            indicatorType=MenuIndicatorType.RADIO,
        )
        display_group = QActionGroup(display_menu)
        display_group.setExclusive(True)
        self.display_mode_actions: dict[str, Action] = {}
        for mode, text_key in self._display_mode_definitions():
            action = Action(self._t(text_key))
            action.setCheckable(True)
            action.setChecked(mode == self._display_mode)
            action.triggered.connect(
                lambda checked, m=mode: self._on_display_mode_selected(m)
            )
            display_group.addAction(action)
            display_menu.addAction(action)
            self.display_mode_actions[mode] = action

        self.display_menu = display_menu
        self.display_mode_button.setMenu(display_menu)

        # --- Arrange menu: a single reference choice + align and distribute options (text + icon; a click does not close the menu) ---
        arrange_menu = _StayOpenCheckableMenu(
            parent=menu_parent, indicatorType=MenuIndicatorType.RADIO
        )
        ref_group = QActionGroup(arrange_menu)
        ref_group.setExclusive(True)
        self.align_ref_actions: dict[str, Action] = {}
        for reference, icon_file, text_key in (
            (
                "selection",
                "ic_fluent_select_object_24_regular.svg",
                "Reference: Selection",
            ),
            ("canvas", "ic_fluent_image_24_regular.svg", "Reference: Canvas"),
        ):
            action = Action(themed_fluent_svg_icon(icon_file), self._t(text_key))
            action.setCheckable(True)
            action.setChecked(reference == self._align_ref)
            action.triggered.connect(
                lambda checked, r=reference: self._on_align_ref_selected(r)
            )
            ref_group.addAction(action)
            arrange_menu.addAction(action)
            self.align_ref_actions[reference] = action

        arrange_menu.addSeparator()
        self.align_actions: dict[str, Action] = {}
        for mode, icon_file, text_key in (
            ("left", "align_left.svg", "Align Left"),
            (
                "horizontal_center",
                "align_horizontal_center.svg",
                "Align Horizontal Center",
            ),
            ("right", "align_right.svg", "Align Right"),
            ("top", "align_top.svg", "Align Top"),
            ("vertical_center", "align_vertical_center.svg", "Align Vertical Center"),
            ("bottom", "align_bottom.svg", "Align Bottom"),
        ):
            action = Action(themed_fluent_svg_icon(icon_file), self._t(text_key))
            action.setEnabled(False)
            action.triggered.connect(
                lambda checked, m=mode: self.align_requested.emit(m)
            )
            arrange_menu.addAction(action)
            self.align_actions[mode] = action

        arrange_menu.addSeparator()
        self._dist_v_action = Action(
            themed_fluent_svg_icon("distribute_spacing_v.svg"),
            self._t("Distribute Vertical Spacing"),
        )
        self._dist_v_action.setEnabled(False)
        self._dist_v_action.triggered.connect(
            lambda: self.distribute_requested.emit("spacing_v")
        )
        arrange_menu.addAction(self._dist_v_action)

        self._dist_h_action = Action(
            themed_fluent_svg_icon("distribute_spacing_h.svg"),
            self._t("Distribute Horizontal Spacing"),
        )
        self._dist_h_action.setEnabled(False)
        self._dist_h_action.triggered.connect(
            lambda: self.distribute_requested.emit("spacing_h")
        )
        arrange_menu.addAction(self._dist_h_action)

        # After a rebuild (language switch), restore the enabled state from the current number of selected items
        self._apply_align_button_states()

        self.arrange_menu = arrange_menu
        self.arrange_button.setMenu(arrange_menu)

        for old in old_menus:
            if old is not None:
                old.deleteLater()

    def _display_mode_definitions(self):
        return [
            ("full", "Show Text and Boxes"),
            ("text_only", "Show Text Only"),
            ("box_only", "Show Boxes Only"),
            ("none", "Show Nothing"),
            ("compare_original_split", "Compare with Original (Two Panels)"),
        ]

    def _on_display_mode_selected(self, mode: str):
        if mode == self._display_mode:
            return
        self._display_mode = mode
        self.display_mode_changed.emit(mode)

    def _on_snap_action_triggered(self, checked: bool = False):
        self.set_snap_enabled(checked, emit=True)

    def set_snap_enabled(self, enabled: bool, emit: bool = False):
        """Sync the snapping switch of the editor; when the configuration is synced from outside, no signal is sent back by default."""
        enabled = bool(enabled)
        changed = enabled != self._snap_enabled
        self._snap_enabled = enabled

        action = getattr(self, "snap_action", None)
        if action is not None and action.isChecked() != enabled:
            action.blockSignals(True)
            action.setChecked(enabled)
            action.blockSignals(False)

        if emit and changed:
            self.snap_enabled_changed.emit(enabled)

    def is_snap_enabled(self) -> bool:
        return self._snap_enabled

    def _on_center_scale_action_triggered(self, checked: bool = False):
        self.set_center_scale_enabled(checked, emit=True)

    def set_center_scale_enabled(self, enabled: bool, emit: bool = False):
        """Sync the scale-about-centre switch; when the configuration is synced from outside, no signal is sent back by default."""
        enabled = bool(enabled)
        changed = enabled != self._center_scale_enabled
        self._center_scale_enabled = enabled

        action = getattr(self, "center_scale_action", None)
        if action is not None and action.isChecked() != enabled:
            action.blockSignals(True)
            action.setChecked(enabled)
            action.blockSignals(False)

        if emit and changed:
            self.center_scale_enabled_changed.emit(enabled)

    def is_center_scale_enabled(self) -> bool:
        return self._center_scale_enabled

    def _on_rich_text_popup_action_triggered(self, checked: bool = False):
        if not checked:
            self.set_rich_text_popup_pinned(False, emit=True)
        self.set_rich_text_popup_enabled(checked, emit=True)

    def set_rich_text_popup_enabled(self, enabled: bool, emit: bool = False):
        """Sync the floating rich-text editor switch; when the configuration is synced from outside, no signal is sent back by default."""
        enabled = bool(enabled)
        if not enabled and self._rich_text_popup_pinned:
            self.set_rich_text_popup_pinned(False, emit=emit)
        changed = enabled != self._rich_text_popup_enabled
        self._rich_text_popup_enabled = enabled

        action = getattr(self, "rich_text_popup_action", None)
        if action is not None and action.isChecked() != enabled:
            action.blockSignals(True)
            action.setChecked(enabled)
            action.blockSignals(False)

        if emit and changed:
            self.rich_text_popup_enabled_changed.emit(enabled)

    def is_rich_text_popup_enabled(self) -> bool:
        return self._rich_text_popup_enabled

    def _on_rich_text_popup_pinned_action_triggered(self, checked: bool = False):
        if checked and not self._rich_text_popup_enabled:
            self.set_rich_text_popup_enabled(True, emit=True)
        self.set_rich_text_popup_pinned(checked, emit=True)

    def set_rich_text_popup_pinned(self, pinned: bool, emit: bool = False):
        """Sync the pin switch of the floating rich-text window; this state is only kept for the current run."""
        pinned = bool(pinned)
        changed = pinned != self._rich_text_popup_pinned
        self._rich_text_popup_pinned = pinned

        action = getattr(self, "rich_text_popup_pinned_action", None)
        if action is not None and action.isChecked() != pinned:
            action.blockSignals(True)
            action.setChecked(pinned)
            action.blockSignals(False)

        if emit and changed:
            self.rich_text_popup_pinned_changed.emit(pinned)

    def is_rich_text_popup_pinned(self) -> bool:
        return self._rich_text_popup_pinned

    def _on_auto_rich_text_rules_action_triggered(self, checked: bool = False):
        self.set_auto_rich_text_rules(checked, emit=True)

    def set_auto_rich_text_rules(self, enabled: bool, emit: bool = False):
        """Sync the switch for applying the rich-text rules automatically while editing; when the configuration is synced from outside, no signal is sent back by default."""
        enabled = bool(enabled)
        changed = enabled != self._auto_rich_text_rules
        self._auto_rich_text_rules = enabled

        action = getattr(self, "auto_rich_text_rules_action", None)
        if action is not None and action.isChecked() != enabled:
            action.blockSignals(True)
            action.setChecked(enabled)
            action.blockSignals(False)

        if emit and changed:
            self.auto_rich_text_rules_changed.emit(enabled)

    def is_auto_rich_text_rules(self) -> bool:
        return self._auto_rich_text_rules

    def _on_auto_save_action_triggered(self, checked: bool = False):
        self.set_auto_save_on_switch(checked, emit=True)

    def set_auto_save_on_switch(self, enabled: bool, emit: bool = False):
        """Sync the switch for saving automatically when the image is switched."""
        enabled = bool(enabled)
        changed = enabled != self._auto_save_on_switch
        self._auto_save_on_switch = enabled

        action = getattr(self, "auto_save_action", None)
        if action is not None and action.isChecked() != enabled:
            action.blockSignals(True)
            action.setChecked(enabled)
            action.blockSignals(False)

        if emit and changed:
            self.auto_save_on_switch_changed.emit(enabled)

    def is_auto_save_on_switch(self) -> bool:
        return self._auto_save_on_switch

    def _on_auto_export_action_triggered(self, checked: bool = False):
        self.set_auto_export_on_switch(checked, emit=True)

    def set_auto_export_on_switch(self, enabled: bool, emit: bool = False):
        """Sync the switch for exporting automatically when the image is switched."""
        enabled = bool(enabled)
        changed = enabled != self._auto_export_on_switch
        self._auto_export_on_switch = enabled

        action = getattr(self, "auto_export_action", None)
        if action is not None and action.isChecked() != enabled:
            action.blockSignals(True)
            action.setChecked(enabled)
            action.blockSignals(False)

        if emit and changed:
            self.auto_export_on_switch_changed.emit(enabled)

    def is_auto_export_on_switch(self) -> bool:
        return self._auto_export_on_switch

    def _on_suppress_unsaved_warning_action_triggered(self, checked: bool = False):
        self.set_suppress_unsaved_warning(checked, emit=True)

    def set_suppress_unsaved_warning(self, enabled: bool, emit: bool = False):
        """Sync the switch for no longer warning about unsaved edits when the image is switched."""
        enabled = bool(enabled)
        changed = enabled != self._suppress_unsaved_warning
        self._suppress_unsaved_warning = enabled

        action = getattr(self, "suppress_unsaved_warning_action", None)
        if action is not None and action.isChecked() != enabled:
            action.blockSignals(True)
            action.setChecked(enabled)
            action.blockSignals(False)

        if emit and changed:
            self.suppress_unsaved_warning_changed.emit(enabled)

    def is_suppress_unsaved_warning(self) -> bool:
        return self._suppress_unsaved_warning

    def _on_delete_and_recover_action_triggered(self, checked: bool = False):
        self.set_delete_and_recover(checked, emit=True)

    def set_delete_and_recover(self, enabled: bool, emit: bool = False):
        """Sync the stored switch for restoring the original image when a text box is deleted."""
        enabled = bool(enabled)
        changed = enabled != self._delete_and_recover
        self._delete_and_recover = enabled

        action = getattr(self, "delete_and_recover_action", None)
        if action is not None and action.isChecked() != enabled:
            action.blockSignals(True)
            action.setChecked(enabled)
            action.blockSignals(False)

        if emit and changed:
            self.delete_and_recover_changed.emit(enabled)

    def is_delete_and_recover(self) -> bool:
        return self._delete_and_recover

    def _on_align_ref_selected(self, reference: str):
        if reference == self._align_ref:
            return
        self._align_ref = reference
        self._apply_align_button_states()

    def get_align_reference(self) -> str:
        return self._align_ref

    def update_align_distribute_buttons(self, selection_count: int):
        """Update the enabled state of the align/distribute options from the number of selected items and the reference mode."""
        self._last_selection_count = selection_count
        self._apply_align_button_states()

    def _apply_align_button_states(self):
        count = self._last_selection_count
        align_enabled = (count >= 1 and self._align_ref == "canvas") or (count >= 2)
        dist_enabled = count >= 3
        for action in self.align_actions.values():
            action.setEnabled(align_enabled)
        self._dist_v_action.setEnabled(dist_enabled)
        self._dist_h_action.setEnabled(dist_enabled)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _create_separator(self):
        separator = VerticalSeparator()
        separator.setFixedHeight(24)
        return separator

    def _sync_content_width(self):
        """Keep the scroll area's inner widget as wide as its controls need."""
        if self.content_widget is None:
            return

        content_layout = self.content_widget.layout()
        if content_layout is not None:
            content_layout.activate()
            content_width = content_layout.sizeHint().width()
        else:
            content_width = self.content_widget.sizeHint().width()

        self.content_widget.setMinimumWidth(content_width)
        viewport_width = (
            self.scroll_area.viewport().width() if hasattr(self, "scroll_area") else 0
        )
        self.content_widget.resize(
            max(content_width, viewport_width),
            max(
                self.content_widget.sizeHint().height(),
                self.scroll_area.viewport().height(),
            ),
        )

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._sync_content_width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_content_width()

    def _connect_signals(self):
        self.fit_window_button.clicked.connect(self.fit_window_requested)
        self.original_image_alpha_slider.valueChanged.connect(
            self.original_image_alpha_changed
        )

    # --- Public Slots ---
    def update_undo_redo_state(self, can_undo: bool, can_redo: bool):
        self._can_undo = bool(can_undo)
        self._can_redo = bool(can_redo)
        self.undo_action.setEnabled(self._can_undo)
        self.redo_action.setEnabled(self._can_redo)

    def set_original_image_alpha_slider(self, alpha: float):
        """Project the model opacity onto the slider, without sending user input back."""
        slider_value = round(max(0.0, min(1.0, float(alpha))) * 100)
        self.original_image_alpha_slider.blockSignals(True)
        self.original_image_alpha_slider.setValue(slider_value)
        self.original_image_alpha_slider.blockSignals(False)

    def set_export_enabled(self, enabled: bool):
        """Set the enabled state of the save and export buttons."""
        self._export_enabled = bool(enabled)
        self.save_button.setEnabled(self._export_enabled)
        self.export_button.setEnabled(self._export_enabled)

    def refresh_ui_texts(self):
        """Refresh all UI texts (for a language switch). The menus are rebuilt as a whole, and the state is restored from the fields."""
        self.menu_button.setText(self._t("Menu"))
        self.display_mode_button.setText(self._t("Display Mode"))
        self.arrange_button.setText(self._t("Arrange"))
        self.save_button.setText(self._t("Save"))
        self.export_button.setText(self._t("Export Image"))
        set_hover_hint(self.fit_window_button, self._t("Fit to Window"))
        self.opacity_label.setText(self._t("Original Image Opacity:"))
        self._build_menus()
        self._sync_content_width()

    def refresh_theme(self):
        for button, icon_file in self._themed_icon_buttons:
            button.setIcon(themed_fluent_svg_icon(icon_file))
