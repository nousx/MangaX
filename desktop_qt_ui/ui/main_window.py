import logging
import os
import re

from manga_translator.utils.system_proxy import set_system_proxy_enabled
from PyQt6.QtCore import QLibraryInfo, QLocale, Qt, QTimer, QTranslator, QUrl, pyqtSlot
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import FluentWindow, NavigationItemPosition

from app_logic import MainAppLogic
from services import (
    ServiceManager,
    get_config_service,
    get_i18n_manager,
    get_logger,
    get_state_manager,
)
from services.file_list_data_service import FileCatalogSnapshot, FileListDataService
from theme_registry import THEME_OPTIONS
from ui.main_page.view import MainView
from ui.secondary_pages.themed_message_box import show_error_dialog
from utils.app_version import format_app_title, get_app_version


class MainWindow(FluentWindow):
    """
    Main window of the application.
    Hosts all UI components, the side navigation, page switching and so on.
    The sidebar is collapsed to a narrow icon strip by default, and the hamburger button at the top left expands it (following the configuration of AiNiee).
    """

    def __init__(self):
        super().__init__()

        self.logger = get_logger(__name__)
        self.i18n = get_i18n_manager()
        self.app_version = get_app_version()
        self._qt_translator = None
        self._apply_qt_translator(
            self.i18n.get_current_locale() if self.i18n else "en_US"
        )

        self._update_window_title()
        self.resize(1300, 800)  # Set the default window size (20 pixels larger)
        self.setMinimumSize(800, 600)  # Set the minimum window size
        # No maximum size is set, so resizing is unlimited

        # Centre the window
        from PyQt6.QtGui import QScreen

        screen = QScreen.availableGeometry(self.screen())
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

        # Sidebar: collapsed by default (a 48px strip of icons), with tooltips on hover; the hamburger button expands it
        self.navigationInterface.setExpandWidth(200)
        self.navigationInterface.setUpdateIndicatorPosOnCollapseFinished(True)
        self.navigationInterface.setReturnButtonVisible(False)

        # The title bar is made narrower: 48 by default -> 36, and the top margin of the content area is tightened with it
        self.titleBar.setFixedHeight(36)
        self.widgetLayout.setContentsMargins(0, 36, 0, 0)

        # The window icon is set in main.py and does not need setting again here

        # The theme currently applied (for logic decisions)
        self.current_applied_theme = "light"

        self._setup_logic_and_models()
        self._setup_ui()
        self._load_theme()
        self._connect_signals()

        self.app_logic.initialize()

    def _t(self, key: str, **kwargs) -> str:
        """Translation helper methods"""
        if self.i18n:
            return self.i18n.translate(key, **kwargs)
        return key

    def _update_window_title(self):
        """Keep the product name stable across interface languages."""
        self.setWindowTitle(format_app_title("MangaX", self.app_version))

    def _setup_logic_and_models(self):
        """Create all logic and data models"""
        self.config_service = get_config_service()
        self.state_manager = get_state_manager()
        config = self.config_service.get_config()
        set_system_proxy_enabled(bool(config.app.use_system_proxy))

        initial_theme = config.app.theme
        if initial_theme == "system":
            detected_theme = self._detect_system_theme()
            if detected_theme == "dark":
                initial_theme = "dark"
            else:
                initial_theme = config.app.theme_user_preference

        from ui.theme import apply_application_theme

        apply_application_theme(initial_theme, QApplication.instance())
        self.current_applied_theme = initial_theme

        # --- Logic Controllers ---
        self.app_logic = MainAppLogic()
        self.file_list_data_service = FileListDataService(self, max_workers=2)
        self.app_logic.file_list_data_service = self.file_list_data_service
        self._file_catalog_snapshot = FileCatalogSnapshot.empty()
        self._main_catalog_generation = 0
        self._main_catalog_loading = False
        self._pending_editor_open = None
        ServiceManager.register_service("app_logic", self.app_logic)
        self.editor_model = None
        self.editor_controller = None
        self.editor_logic = None
        self.editor_view = None

    def _setup_ui(self):
        """Initialise the UI components"""
        # No menu bar at the top; the menu functions are all in the settings area
        self._create_ui_actions()

        self.main_view = MainView(self.app_logic, self)

        # Give app_logic a reference to main_view, for updating the progress bar
        self.app_logic.main_view = self.main_view

        self.stacked_widget = self.stackedWidget
        self._main_navigation_items = {}
        self._register_main_interfaces()
        self._ensure_editor_initialized()
        self.stacked_widget.currentChanged.connect(self._on_fluent_page_changed)

    def _register_main_interfaces(self):
        pages = [
            (
                "translation",
                self.main_view.translation_interface,
                FIF.HOME,
                self._t("Translation Interface"),
            ),
            (
                "settings",
                self.main_view.settings_page,
                FIF.SETTING,
                self._t("Settings"),
            ),
            ("env", self.main_view.env_page, FIF.CONNECT, self._t("API Management")),
            (
                "prompts",
                self.main_view.prompt_page,
                FIF.DOCUMENT,
                self._t("Prompt Management"),
            ),
            (
                "replacements",
                self.main_view.replacements_page,
                FIF.EDIT,
                self._t("Replacement Rules"),
            ),
            (
                "rich_text_rules",
                self.main_view.rich_text_rules_page,
                FIF.FONT,
                self._t("Rich Text Rules"),
            ),
            (
                "batch_edit",
                self.main_view.batch_edit_page,
                FIF.LIBRARY,
                self._t("Batch Management"),
            ),
            (
                "about",
                self.main_view.about_page,
                FIF.INFO,
                self._t("About Application"),
            ),
        ]
        for key, page, icon, text in pages:
            page.setObjectName(f"main_{key}_page")
            self._main_navigation_items[key] = self.addSubInterface(page, icon, text)

        self._main_pages_by_widget = {page: key for key, page, _icon, _text in pages}
        self.main_view.set_navigation_switcher(self._switch_main_page)
        self.switchTo(self.main_view.translation_interface)

    def _switch_main_page(self, page_key: str):
        page = (
            self.main_view.page_widgets.get(page_key)
            if hasattr(self.main_view, "page_widgets")
            else None
        )
        if page is not None:
            self.switchTo(page)
            self._on_main_page_activated(page_key)

    def _on_fluent_page_changed(self, index: int):
        widget = self.stacked_widget.widget(index)
        page_key = getattr(self, "_main_pages_by_widget", {}).get(widget)
        if page_key:
            self._on_main_page_activated(page_key)

    def _on_main_page_activated(self, page_key: str):
        if page_key == "settings" and not getattr(
            self.main_view, "_settings_ui_ready", False
        ):
            self.main_view.set_parameters(self.config_service.get_config().model_dump())
        elif page_key == "about":
            self.main_view._refresh_about_page_texts()
        elif page_key == "env":
            self.main_view._refresh_env_api_groups()
        elif page_key == "replacements":
            if hasattr(self.main_view, "replacements_editor_panel"):
                self.main_view.replacements_editor_panel.refresh()
        elif page_key == "rich_text_rules":
            if hasattr(self.main_view, "rich_text_rules_editor_panel"):
                self.main_view.rich_text_rules_editor_panel.refresh()
        elif page_key == "batch_edit":
            if hasattr(self.main_view, "batch_edit_panel"):
                self.main_view.batch_edit_panel.set_catalog_snapshot(
                    self._file_catalog_snapshot
                )
                self.main_view.batch_edit_panel.refresh()

    def _ensure_editor_initialized(self):
        if self.editor_view is not None:
            return

        from editor.editor_controller import EditorController
        from editor.editor_logic import EditorLogic
        from editor.editor_model import EditorModel
        from ui.editor.view import EditorView

        self.editor_model = EditorModel()
        self.editor_controller = EditorController(self.editor_model)
        self.editor_logic = EditorLogic(
            self.editor_controller,
            parent=self,
            file_data_service=self.file_list_data_service,
        )
        self.editor_view = EditorView(
            self.app_logic,
            self.editor_model,
            self.editor_controller,
            self.editor_logic,
            self,
        )
        self.editor_view.setObjectName("editor_page")
        self.addSubInterface(
            self.editor_view,
            FIF.EDIT,
            self._t("Editor View"),
            position=NavigationItemPosition.BOTTOM,
        )

        self.app_logic.config_loaded.connect(
            self.editor_view.property_panel.repopulate_options
        )

        self.editor_view._apply_editor_style(self.current_applied_theme)
        self.editor_view.property_panel.repopulate_options()

        if hasattr(self.main_view, "batch_edit_panel"):
            # The editor keeps the regions in memory and does not watch for file changes; after a batch write-back it has to
            # reload, otherwise the automatic save on switching images would overwrite the changes just written with old data.
            self.main_view.batch_edit_panel.set_editor_context(
                self.editor_model.get_source_image_path,
                self.editor_controller.document_service.load_image_and_regions,
            )

    def _create_ui_actions(self):
        """Create the internal action objects (there is no menu bar at the top)"""
        self.add_files_action = QAction(self._t("&Add Files..."), self)
        self.undo_action = QAction(self._t("&Undo"), self)
        self.redo_action = QAction(self._t("&Redo"), self)
        self.main_view_action = QAction(self._t("Main View"), self)
        self.editor_view_action = QAction(self._t("Editor View"), self)
        self.theme_actions = {}
        for theme_key, theme_label in THEME_OPTIONS:
            action = QAction(self._t(theme_label), self)
            self.theme_actions[theme_key] = action
            setattr(self, f"{theme_key}_theme_action", action)

    def _load_theme(self):
        """Initialise the qfluentwidgets theme from the configuration."""
        from services import get_config_service

        config_service = get_config_service()
        config = config_service.get_config()

        # Get the theme setting; Pydantic uses the default 'light' automatically
        theme = config.app.theme
        self._apply_theme(theme)

    def _apply_theme(self, theme: str):
        """Apply the given theme"""

        # System theme logic: 'system' is resolved to the actual theme
        if theme == "system":
            sys_theme = self._detect_system_theme()
            if sys_theme == "dark":
                self._apply_theme("dark")
            else:
                config = self.config_service.get_config()
                # Use the user's preference (all themes other than dark)
                self._apply_theme(config.app.theme_user_preference)
            return

        # Record the theme actually applied
        self.current_applied_theme = theme

        app = QApplication.instance()
        from ui.theme import apply_application_theme

        apply_application_theme(theme, app)

        # Tell the views to refresh their local Fluent theme state (MainView is a pure logic object and needs no update)
        if hasattr(self, "main_view") and self.main_view:
            self.main_view.apply_fluent_theme(theme)
        if hasattr(self, "editor_view") and self.editor_view:
            self.editor_view._apply_editor_style(theme)
            self.editor_view.update()
        if hasattr(self, "stacked_widget") and self.stacked_widget:
            self.stacked_widget.update()
        self.update()
        # Deferred to the next turn of the event loop, where the native title bar theme is applied once, so it is not set twice for one switch
        QTimer.singleShot(
            0,
            lambda active_theme=theme: self._apply_native_title_bar_theme(active_theme),
        )

    def _apply_native_title_bar_theme(self, theme: str):
        """Sync the colour of the native Windows title bar, so a dark content area does not get a light system title bar."""
        from ui.theme import apply_native_title_bar_theme

        apply_native_title_bar_theme(self, theme, logger=self.logger)

    def _detect_system_theme(self) -> str | None:
        """Detect the system theme through Qt; None when the platform gives no appearance information."""
        scheme = QApplication.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return "dark"
        if scheme == Qt.ColorScheme.Light:
            return "light"
        return None

    def _on_system_theme_changed(self, scheme: Qt.ColorScheme):
        """Respond to a system appearance notification; only the interface in follow-system mode is updated."""
        config = self.config_service.get_config()
        if config.app.theme != "system":
            return

        if scheme == Qt.ColorScheme.Dark:
            theme = "dark"
        elif scheme == Qt.ColorScheme.Light:
            theme = config.app.theme_user_preference
        else:
            # When the appearance cannot be recognised, the current theme is kept and the user's preference is not overwritten.
            return

        if theme != self.current_applied_theme:
            self.logger.info(f"System color scheme changed: {scheme.name}; applying {theme}")
            self._apply_theme(theme)

    def _change_theme(self, theme: str):
        """Switch the theme and save it to the configuration"""
        from services import get_config_service

        config_service = get_config_service()
        config = config_service.get_config()

        self._apply_theme(theme)

        # Only a manually chosen non-dark theme is saved, to be restored when the system is in light mode.
        if theme not in ("dark", "system"):
            config.app.theme_user_preference = theme

        # Save to the configuration
        config.app.theme = theme
        config_service.set_config(config)

        # Save to the file
        config_service.save_config_file()

    def _connect_signals(self):
        # --- MainAppLogic Connections ---
        self.app_logic.config_loaded.connect(self.main_view.set_parameters)
        self.app_logic.file_sources_changed.connect(self._request_main_file_snapshot)
        self.app_logic.file_removed.connect(self.main_view.file_list.remove_file)
        self.app_logic.file_removed.connect(self._on_file_removed_update_editor)
        self.app_logic.files_cleared.connect(self.main_view.file_list.clear)
        self.app_logic.files_cleared.connect(self._on_files_cleared_update_editor)
        self.app_logic.output_path_updated.connect(
            self.main_view.update_output_path_display
        )
        self.app_logic.task_completed.connect(
            self.on_task_completed, type=Qt.ConnectionType.QueuedConnection
        )
        self.app_logic.error_dialog_requested.connect(
            self._show_error_dialog, type=Qt.ConnectionType.QueuedConnection
        )
        self.app_logic.warning_dialog_requested.connect(
            self._show_warning_dialog, type=Qt.ConnectionType.QueuedConnection
        )
        self.config_service.write_failed.connect(
            self._show_config_write_failed,
            type=Qt.ConnectionType.QueuedConnection,
        )
        deferred_write_error = self.config_service.take_deferred_write_error()
        if deferred_write_error:
            QTimer.singleShot(
                0,
                lambda error=deferred_write_error: self._show_config_write_failed(
                    error
                ),
            )
        self.file_list_data_service.loading.connect(
            self._on_main_catalog_loading,
            type=Qt.ConnectionType.QueuedConnection,
        )
        self.file_list_data_service.snapshot_ready.connect(
            self._on_main_catalog_ready,
            type=Qt.ConnectionType.QueuedConnection,
        )
        self.file_list_data_service.error.connect(
            self._on_main_catalog_error,
            type=Qt.ConnectionType.QueuedConnection,
        )

        # --- View to Logic Connections ---
        self.main_view.setting_changed.connect(self.app_logic.update_single_config)
        self.main_view.env_var_changed.connect(self.app_logic.save_env_var)
        self.main_view.editor_view_requested.connect(self.switch_to_editor_view)
        self.main_view.theme_change_requested.connect(self._change_theme)
        self.main_view.language_change_requested.connect(
            self._change_language, type=Qt.ConnectionType.QueuedConnection
        )

        # --- View to Coordinator Connections ---
        self.main_view.file_list.file_selected.connect(
            self.on_file_selected_from_main_list
        )
        self.main_view.file_list.files_dropped.connect(
            self.app_logic.add_files
        )  # Support for dropping files
        # self.main_view.enter_editor_button.clicked.connect(self.enter_editor_mode) # Example for a dedicated button

        # --- View Switching Connections ---
        self.main_view_action.triggered.connect(
            lambda: self.switchTo(self.main_view.translation_interface)
        )
        self.editor_view_action.triggered.connect(self.switch_to_editor_view)

        # --- Undo/redo are forwarded to the editor controller, lazily ---
        self.undo_action.triggered.connect(self._handle_undo)
        self.redo_action.triggered.connect(self._handle_redo)

        # --- Theme switch connections ---
        # Qt still uses the old palette when it sends the notification; queue until its update is done, then apply the custom theme.
        QApplication.styleHints().colorSchemeChanged.connect(
            self._on_system_theme_changed,
            type=Qt.ConnectionType.QueuedConnection,
        )
        for theme_key, action in getattr(self, "theme_actions", {}).items():
            action.triggered.connect(
                lambda checked=False, selected_theme=theme_key: self._change_theme(
                    selected_theme
                )
            )

    @pyqtSlot()
    def _request_main_file_snapshot(self):
        self._main_catalog_loading = True
        self.main_view.file_list.set_loading(keep_items=True)
        try:
            self._main_catalog_generation = (
                self.file_list_data_service.request_snapshot(
                    "main",
                    tuple(self.app_logic.source_files),
                    tuple(self.app_logic.excluded_subfolders),
                    tuple(self.app_logic.excluded_files),
                )
            )
        except RuntimeError as exc:
            self._main_catalog_loading = False
            self.main_view.file_list.set_error(str(exc))

    @pyqtSlot(str, int)
    def _on_main_catalog_loading(self, channel: str, generation: int):
        if channel != "main":
            return
        self._main_catalog_loading = True
        self._main_catalog_generation = generation

    @pyqtSlot(str, int, object)
    def _on_main_catalog_ready(self, channel: str, generation: int, snapshot: object):
        if channel != "main" or generation != self._main_catalog_generation:
            return
        self._main_catalog_loading = False
        self._file_catalog_snapshot = snapshot
        self.main_view.file_list.set_snapshot(snapshot)
        if hasattr(self.main_view, "batch_edit_panel"):
            # The scope of batch editing follows the file list of the main page; it is synced whenever the snapshot changes
            self.main_view.batch_edit_panel.set_catalog_snapshot(snapshot)
        for warning in snapshot.warnings:
            self.logger.warning(warning)

        pending = self._pending_editor_open
        self._pending_editor_open = None
        if pending is not None:
            file_to_load, files_to_load = pending
            QTimer.singleShot(
                0,
                lambda: self.enter_editor_mode(
                    file_to_load=file_to_load,
                    files_to_load=files_to_load,
                ),
            )

    @pyqtSlot(str, int, str)
    def _on_main_catalog_error(self, channel: str, generation: int, message: str):
        if channel != "main" or generation != self._main_catalog_generation:
            return
        self._main_catalog_loading = False
        self.main_view.file_list.set_error(message)

    @pyqtSlot(str)
    def on_file_selected_from_main_list(self, file_path: str):
        """
        Coordinator slot. Handles when a file is double-clicked in the main view.
        It tells the editor logic to load the file, then switches the view.
        """
        self.logger.info(
            f"File double-clicked from main list: {file_path}. Switching to editor."
        )
        self.enter_editor_mode(file_to_load=file_path)

    def _on_file_removed_update_editor(self, file_path: str):
        """When a file is removed on the main page, update the editor (if the editor is showing that file)"""
        if not self.editor_view or not self.editor_controller:
            return
        if self.stacked_widget.currentWidget() == self.editor_view:
            # Check whether the image currently loaded was removed
            current_image = self.editor_controller.model.get_source_image_path()

            if current_image:
                import os

                norm_current = os.path.normpath(current_image)
                norm_removed = os.path.normpath(file_path)

                # When the current image is what was removed
                if norm_current == norm_removed:
                    self.editor_controller.document_service.clear_editor_state()
                # When a folder was removed, check whether the current image is inside it
                elif os.path.isdir(file_path):
                    try:
                        # Check whether the current image is inside the removed folder
                        if (
                            os.path.commonpath([norm_current, norm_removed])
                            == norm_removed
                        ):
                            self.editor_controller.document_service.clear_editor_state()
                    except ValueError:
                        # A different drive: skip
                        pass

            # Note: the editor has its own file list and does not need to follow removals on the main page
            # The editor list is only cleared when the main page has no files left at all

    def _on_files_cleared_update_editor(self):
        """When the file list is cleared, clear the editor"""
        if not self.editor_view or not self.editor_logic:
            return
        self.logger.info("Files cleared. Clearing editor.")
        self.editor_logic.clear_list()

    def _change_language(self, locale_code: str):
        """Switch the language"""
        if self.i18n and self.i18n.set_locale(locale_code):
            self._apply_qt_translator(locale_code)
            # Save the language setting to the configuration
            config = self.config_service.get_config()
            config.app.ui_language = locale_code
            self.config_service.set_config(config)
            self.config_service.save_config_file()

            # Refresh the UI text
            self._refresh_ui_texts()
            self.logger.info(f"Language switched to: {locale_code}")

    def _apply_qt_translator(self, locale_code: str):
        """Load the translations of Qt's built-in controls (such as QColorDialog), so they follow the application language."""
        app = QApplication.instance()
        if app is None:
            return

        if self._qt_translator is not None:
            app.removeTranslator(self._qt_translator)
            self._qt_translator.deleteLater()
            self._qt_translator = None

        translator = QTranslator(self)
        qt_translations_dir = QLibraryInfo.path(
            QLibraryInfo.LibraryPath.TranslationsPath
        )

        # locale_code looks like zh_CN / en_US; an exact match is tried first, then a match at language level
        language = QLocale(locale_code).name().split("_", 1)[0]
        candidates = (
            f"qtbase_{locale_code}",
            f"qtbase_{language}",
            f"qt_{locale_code}",
            f"qt_{language}",
        )

        loaded = any(translator.load(name, qt_translations_dir) for name in candidates)
        if loaded:
            app.installTranslator(translator)
            self._qt_translator = translator

    def _refresh_ui_texts(self):
        """Refresh the UI texts"""
        self._update_window_title()
        self._refresh_action_texts()

        # Refresh all text of the main view
        if hasattr(self, "main_view") and self.main_view:
            self.main_view.refresh_ui_texts()
            self._refresh_navigation_texts()

        # Refresh all text of the editor view (when it exists)
        if hasattr(self, "editor_view") and self.editor_view:
            if hasattr(self.editor_view, "refresh_ui_texts"):
                self.editor_view.refresh_ui_texts()

    def _refresh_action_texts(self):
        """Refresh the texts of the internal actions (the action objects are kept even while the menu bar is hidden)"""
        if hasattr(self, "add_files_action"):
            self.add_files_action.setText(self._t("&Add Files..."))
        if hasattr(self, "undo_action"):
            self.undo_action.setText(self._t("&Undo"))
        if hasattr(self, "redo_action"):
            self.redo_action.setText(self._t("&Redo"))
        if hasattr(self, "main_view_action"):
            self.main_view_action.setText(self._t("Main View"))
        if hasattr(self, "editor_view_action"):
            self.editor_view_action.setText(self._t("Editor View"))
        for theme_key, theme_label in THEME_OPTIONS:
            action = getattr(self, "theme_actions", {}).get(theme_key)
            if action is not None:
                action.setText(self._t(theme_label))

    def _handle_undo(self):
        if self.editor_controller:
            self.editor_controller.undo()

    def _handle_redo(self):
        if self.editor_controller:
            self.editor_controller.redo()

    @pyqtSlot(list)
    def on_task_completed(self, saved_files: list):
        """
        Handles the completion of a translation task.
        Asks the user if they want to open the results in the editor.
        """
        try:
            if not saved_files:
                return

            # Translation creates or updates the JSON metadata; refresh the full snapshot first, and the request to open the editor waits automatically.
            self._request_main_file_snapshot()

            if not self._should_prompt_open_results_in_editor():
                return

            # When a frameless qfluentwidgets dialog pops up with a minimised main window as its parent,
            # Windows 11 may stop compositing new window content after it is restored; restoring before the dialog avoids that state.
            if self.isMinimized():
                self.showNormal()

            from PyQt6.QtWidgets import QMessageBox

            reply = show_error_dialog(
                self,
                self._t("Task Completed"),
                "",
                self._t(
                    "Translation completed, {count} files saved.\n\nOpen results in editor?",
                    count=len(saved_files),
                ),
                icon=QMessageBox.Icon.Question,
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                default_button=QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.enter_editor_mode(files_to_load=saved_files)
        except Exception as e:
            self.logger.error(f"Error in on_task_completed: {e}", exc_info=True)
            import traceback

            traceback.print_exc()

    def _should_prompt_open_results_in_editor(self) -> bool:
        """Only prompt for workflows that produce editor-meaningful results."""
        try:
            config = self.config_service.get_config()
            cli = getattr(config, "cli", None)
            if cli is None:
                return True

            if getattr(cli, "replace_translation", False):
                return True

            if getattr(cli, "load_text", False):
                return True

            incompatible_modes = (
                getattr(cli, "translate_json_only", False),
                getattr(cli, "template", False),
                getattr(cli, "generate_and_export", False),
                getattr(cli, "colorize_only", False),
                getattr(cli, "upscale_only", False),
                getattr(cli, "inpaint_only", False),
            )
            return not any(incompatible_modes)
        except Exception as e:
            self.logger.warning(f"Failed to determine whether to show editor prompt; showing it by default: {e}")
            return True

    @pyqtSlot(str)
    def _show_error_dialog(self, error_message: str):
        """Show the translation error box"""
        try:
            log_dir = self._resolve_log_folder_from_message(error_message)
            show_error_dialog(
                self,
                self._t("Translation Error"),
                "",
                error_message,
                extra_button_text=self._t("Open log folder"),
                extra_button_callback=lambda: self._open_log_folder(log_dir),
            )
        except Exception as e:
            self.logger.error(f"_show_error_dialog error: {e}", exc_info=True)

    def _resolve_log_folder_from_message(self, message: str) -> str:
        match = re.search(r"日志文件[：:]\s*(.+)", str(message or ""))
        if match:
            log_path = match.group(1).strip()
            log_path = log_path.splitlines()[0].strip()
            if log_path:
                return os.path.dirname(os.path.normpath(os.path.abspath(log_path)))
        for handler in reversed(logging.getLogger().handlers):
            if isinstance(handler, logging.FileHandler):
                log_path = str(getattr(handler, "baseFilename", "") or "").strip()
                if log_path:
                    return os.path.dirname(os.path.normpath(os.path.abspath(log_path)))
        return os.path.normpath(os.path.abspath(os.path.join(os.getcwd(), "result")))

    def _open_log_folder(self, folder: str):
        target = os.path.normpath(
            os.path.abspath(folder or os.path.join(os.getcwd(), "result"))
        )
        os.makedirs(target, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    @pyqtSlot(str)
    def _show_warning_dialog(self, message: str):
        """Show the task notice box"""
        try:
            show_error_dialog(
                self,
                self._t("Warning"),
                "",
                message,
            )
        except Exception as e:
            self.logger.error(f"_show_warning_dialog error: {e}", exc_info=True)

    @pyqtSlot(str)
    def _show_config_write_failed(self, error: str):
        """Say clearly that writing the configuration failed, so the user does not think the API key was saved."""
        try:
            guidance = self._t(
                "Configuration save failed. Changes were not saved. Check file permissions and antivirus or security software blocking access."
            )
            detail = str(error or "").strip()
            message = f"{guidance}\n\n{detail}" if detail else guidance
            show_error_dialog(self, self._t("Error"), "", message)
        except Exception as exc:
            self.logger.error(f"_show_config_write_failed error: {exc}", exc_info=True)

    def switch_to_editor_view(self):
        """
        Simply switches to the editor view without reloading file lists.
        Used when user manually switches views.
        """
        self._ensure_editor_initialized()
        if self.editor_view and self.editor_view.property_panel:
            self.editor_view.property_panel.repopulate_options()
        self.switchTo(self.editor_view)

    def enter_editor_mode(self, file_to_load: str = None, files_to_load: list = None):
        """
        Switches to the editor view and loads the necessary files.
        file_to_load: path of a single file (used when a file is double-clicked)
        files_to_load: list of saved results (used when entering after a translation finished, to locate the original image to open)
        """
        try:
            self._ensure_editor_initialized()
            if self.editor_view and self.editor_view.property_panel:
                self.editor_view.property_panel.repopulate_options()

            if self._main_catalog_loading:
                self._pending_editor_open = (
                    file_to_load,
                    list(files_to_load) if files_to_load else None,
                )
                self.editor_view.file_list.set_loading()
                self.switchTo(self.editor_view)
                return

            if self.app_logic.source_files and not self._file_catalog_snapshot.sources:
                self._pending_editor_open = (
                    file_to_load,
                    list(files_to_load) if files_to_load else None,
                )
                self.editor_view.file_list.set_loading()
                self._request_main_file_snapshot()
                self.switchTo(self.editor_view)
                return

            editor_snapshot = self._file_catalog_snapshot.images_only()
            self.editor_logic.apply_file_snapshot(
                editor_snapshot,
                excluded_folders=self.app_logic.excluded_subfolders,
                excluded_files=self.app_logic.excluded_files,
            )

            target_path = file_to_load
            if files_to_load:
                target_path = (
                    self.app_logic.resolve_completed_source(files_to_load[0])
                    or files_to_load[0]
                )
            if target_path:
                self.editor_logic.load_image_into_editor(target_path)
            elif editor_snapshot.editor_files:
                self.editor_logic.load_image_into_editor(
                    editor_snapshot.editor_files[0]
                )

            self.switchTo(self.editor_view)
        except Exception as e:
            self.logger.error(f"Error in enter_editor_mode: {e}", exc_info=True)
            import traceback

            traceback.print_exc()

    def _refresh_navigation_texts(self):
        nav_labels = {
            "translation": self._t("Translation Interface"),
            "settings": self._t("Settings"),
            "about": self._t("About Application"),
            "env": self._t("API Management"),
            "prompts": self._t("Prompt Management"),
            "replacements": self._t("Replacement Rules"),
            "rich_text_rules": self._t("Rich Text Rules"),
            "batch_edit": self._t("Batch Management"),
        }
        for key, text in nav_labels.items():
            item = getattr(self, "_main_navigation_items", {}).get(key)
            if item is not None and hasattr(item, "setText"):
                item.setText(text)
                # In the collapsed state the items are recognised by their hover tooltips, which are refreshed on a language switch too
                item.setToolTip(text)

    def closeEvent(self, event):
        """Handle the window close event"""
        if hasattr(self, "main_view") and hasattr(self.main_view, "update_checker"):
            self.main_view.update_checker.stop()
        unfinished_exports = 0
        if self.editor_controller is not None:
            unfinished_exports = (
                self.editor_controller.export_service.unfinished_count()
            )
        if unfinished_exports:
            from PyQt6.QtWidgets import QMessageBox

            reply = show_error_dialog(
                self,
                "后台任务尚未完成",
                "",
                f"还有 {unfinished_exports} 个保存或导出任务正在处理。\n\n等待全部任务完成后退出？",
                icon=QMessageBox.Icon.Question,
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                default_button=QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            self.editor_controller.shutdown()

        # Wait first for the background threads of the API test, model list and so on to finish (with a timeout),
        # to avoid "QThread: Destroyed while thread is still running".
        if hasattr(self, "main_view") and self.main_view:
            self.main_view.shutdown_background_threads(3000)
            if hasattr(self.main_view, "batch_edit_panel"):
                self.main_view.batch_edit_panel.shutdown()
        if self.editor_logic is not None:
            self.editor_logic.shutdown()
        self.file_list_data_service.shutdown()
        self.app_logic.shutdown()
        event.accept()
