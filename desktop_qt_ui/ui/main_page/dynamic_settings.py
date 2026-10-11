import json
import os

from PyQt6.QtCore import QSignalBlocker, Qt, QTimer, pyqtSlot
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    HorizontalSeparator,
    SimpleCardWidget,
    StrongBodyLabel,
    SpinBox,
    themeColor,
)
from qfluentwidgets import LineEdit as FluentLineEdit
from qfluentwidgets import PushButton as QPushButton

from ui.widgets.hover_hint import set_hover_hint
from ui.widgets.toggle_switch import ToggleSwitch
from ui.widgets.wheel_filter import NoWheelComboBox as QComboBox
from ui.widgets.widget_cleanup import clear_layout
from utils.font_list import FontComboBox, set_system_fonts_enabled


class QLineEdit(FluentLineEdit):
    """Fluent LineEdit with the PyQt constructor forms used by existing settings code."""

    def __init__(self, text: str | QWidget | None = "", parent: QWidget | None = None):
        if isinstance(text, QWidget) and parent is None:
            parent = text
            text = ""
        super().__init__(parent)
        if text:
            self.setText(str(text))


API_GROUP_SPECS = {
    "translator_openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_API_BASE"),
    "translator_gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_API_BASE"),
    "ocr_openai": ("OCR_OPENAI_API_KEY", "OCR_OPENAI_MODEL", "OCR_OPENAI_API_BASE"),
    "ocr_gemini": ("OCR_GEMINI_API_KEY", "OCR_GEMINI_MODEL", "OCR_GEMINI_API_BASE"),
    "color_openai": ("COLOR_OPENAI_API_KEY", "COLOR_OPENAI_MODEL", "COLOR_OPENAI_API_BASE"),
    "color_gemini": ("COLOR_GEMINI_API_KEY", "COLOR_GEMINI_MODEL", "COLOR_GEMINI_API_BASE"),
    "render_openai": ("RENDER_OPENAI_API_KEY", "RENDER_OPENAI_MODEL", "RENDER_OPENAI_API_BASE"),
    "render_gemini": ("RENDER_GEMINI_API_KEY", "RENDER_GEMINI_MODEL", "RENDER_GEMINI_API_BASE"),
}

SIMPLE_API_GROUP_SPECS = {
    "translator_sakura": ("SAKURA_API_BASE", "SAKURA_DICT_PATH"),
}

def _normalize_selected_value(value) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip()


def _selected_api_group_keys(config) -> dict[str, list[str]]:
    translator_value = _normalize_selected_value(getattr(config.translator, "translator", ""))
    ocr_value = _normalize_selected_value(getattr(config.ocr, "ocr", ""))
    secondary_ocr_value = _normalize_selected_value(getattr(config.ocr, "secondary_ocr", ""))
    colorizer_value = _normalize_selected_value(getattr(config.colorizer, "colorizer", ""))
    renderer_value = _normalize_selected_value(getattr(config.render, "renderer", ""))

    result = {
        "translation": [],
        "ocr": [],
        "color": [],
        "render": [],
    }

    if translator_value in {"openai", "openai_hq"}:
        result["translation"].append("translator_openai")
    elif translator_value in {"gemini", "gemini_hq"}:
        result["translation"].append("translator_gemini")
    elif translator_value == "sakura":
        result["translation"].append("translator_sakura")

    selected_ocr_values = [ocr_value]
    if bool(getattr(config.ocr, "use_hybrid_ocr", False)):
        selected_ocr_values.append(secondary_ocr_value)
    if "openai_ocr" in selected_ocr_values:
        result["ocr"].append("ocr_openai")
    if "gemini_ocr" in selected_ocr_values:
        result["ocr"].append("ocr_gemini")

    if colorizer_value == "openai_colorizer":
        result["color"].append("color_openai")
    elif colorizer_value == "gemini_colorizer":
        result["color"].append("color_gemini")

    if renderer_value == "openai_renderer":
        result["render"].append("render_openai")
    elif renderer_value == "gemini_renderer":
        result["render"].append("render_gemini")

    return result


def _add_empty_api_hint(self, layout, row: int, translation_key: str) -> int:
    notice = SimpleCardWidget()
    notice.setMinimumHeight(120)
    notice.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    notice_layout = QHBoxLayout(notice)
    notice_layout.setContentsMargins(12, 12, 12, 12)
    notice_layout.setSpacing(0)
    notice_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

    text = BodyLabel(self._t(translation_key))
    text.setAlignment(Qt.AlignmentFlag.AlignCenter)
    text.setWordWrap(True)
    text.setMinimumHeight(56)
    text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    notice_layout.addWidget(text, 1)
    layout.addWidget(notice, row, 0, 1, 3)
    layout.setRowStretch(row, 1)
    return row + 1


def _add_api_section_panel(
    self,
    container_layout,
    section_key: str,
    group_keys: list[str],
    current_env_values: dict,
    empty_hint_key: str,
):
    section_card = SimpleCardWidget()
    section_card.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding if not group_keys else QSizePolicy.Policy.Preferred,
    )
    section_card_layout = QVBoxLayout(section_card)
    section_card_layout.setContentsMargins(16, 14, 16, 16)
    section_card_layout.setSpacing(12)

    self.env_layout = QGridLayout()
    self.env_layout.setColumnStretch(1, 1)
    self.env_layout.setColumnStretch(2, 0)
    self.env_layout.setHorizontalSpacing(12)
    self.env_layout.setVerticalSpacing(10)
    self.env_layout.setContentsMargins(0, 0, 0, 0)

    self._create_api_feature_selector_row(section_key)

    if group_keys:
        for group_key in group_keys:
            if group_key in API_GROUP_SPECS:
                api_key_env, model_env, api_base_env = API_GROUP_SPECS[group_key]
                self._create_api_rotation_widgets(
                    api_key_env=api_key_env,
                    model_env=model_env,
                    api_base_env=api_base_env,
                    current_values=current_env_values,
                )
            elif group_key in SIMPLE_API_GROUP_SPECS:
                self._create_env_widgets(
                    list(SIMPLE_API_GROUP_SPECS[group_key]),
                    current_env_values,
                )
    elif empty_hint_key in ("Codex CLI account hint", "Claude CLI account hint"):
        from ui.widgets.codex_account_panel import CodexAccountPanel

        row = self.env_layout.rowCount()
        provider = "Claude" if empty_hint_key.startswith("Claude") else "Codex"
        path_setting = "claude_cli_path" if provider == "Claude" else "codex_cli_path"
        panel = CodexAccountPanel(
            self._t,
            lambda: getattr(self.controller.config_service.get_config().translator, path_setting),
            provider=provider,
        )
        self.env_layout.addWidget(panel, row, 0, 1, 3)
        self.env_layout.setRowStretch(row, 1)
    else:
        _add_empty_api_hint(self, self.env_layout, self.env_layout.rowCount(), empty_hint_key)

    section_card_layout.addLayout(self.env_layout, 1 if not group_keys else 0)
    container_layout.addWidget(section_card, 1 if not group_keys else 0)


_CACHED_SETTINGS_WIDGET_ATTRS = (
    "translator_combo",
    "upscale_ratio_combo",
)

_FIXED_PROMPT_KEYS = frozenset({
    "ocr.ai_ocr_prompt_path",
    "colorizer.ai_colorizer_prompt_path",
    "render.ai_renderer_prompt_path",
})

_SKIPPED_SETTING_KEYS = frozenset({
    "cli.load_text",
    "cli.translate_json_only",
    "cli.template",
    "cli.generate_and_export",
    "cli.colorize_only",
    "cli.upscale_only",
    "cli.inpaint_only",
    "cli.replace_translation",
    "cli.replace_translation_mode",
    "upscale.realcugan_model",
    "render.gimp_font",
    "translator.high_quality_prompt_path",
    "app.last_open_dir",
    "app.last_output_path",
    "app.favorite_folders",
    "app.folder_dialog_sort",
    "app.current_preset",
})

_OPTIONAL_INPUT_KEYS = frozenset({
    "tile_size",
    "line_spacing",
    "letter_spacing",
    "font_size",
    "ocr_vl_custom_prompt",
    "ai_ocr_custom_prompt",
})

_LEGACY_SETTING_SECTIONS = (
    "translator",
    "cli",
    "detector",
    "inpainter",
    "render",
    "upscale",
    "colorizer",
    "ocr",
    "app",
)


def _drop_cached_settings_widget_refs(view):
    """丢弃随设置页重建而销毁的控件缓存引用，避免重建窗口期悬空访问。"""
    for attr in _CACHED_SETTINGS_WIDGET_ATTRS:
        if hasattr(view, attr):
            delattr(view, attr)
    view._highlighted_rows = []


def _clear_layout_widgets(layout, *, restore_stretch: bool = False):
    """递归隐藏并延迟删除布局内容；可补回设置页末尾 stretch。"""
    clear_layout(layout, restore_stretch=restore_stretch)


def _env_group_structure_signature(active_api_groups: dict, current_env_values: dict) -> str:
    from manga_translator.api_key_rotation import get_rotation_slot_count

    from ui.main_page.env_management import API_ROTATION_UI_MAX_SLOTS

    slot_counts = {}
    for group_keys in active_api_groups.values():
        for group_key in group_keys:
            slot_keys = API_GROUP_SPECS.get(group_key)
            if slot_keys:
                slot_counts[group_key] = get_rotation_slot_count(
                    current_env_values,
                    slot_keys,
                    default=1,
                    maximum=API_ROTATION_UI_MAX_SLOTS,
                )
    return json.dumps(
        {"groups": active_api_groups, "slot_counts": slot_counts},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _sync_env_widget_values(self, current_env_values: dict) -> None:
    from ui.main_page.env_management import _set_env_widget_value

    for key, (_label, widget) in list(self.env_widgets.items()):
        blocker = QSignalBlocker(widget)
        try:
            _set_env_widget_value(widget, current_env_values.get(key, ""))
            if hasattr(widget, "setPlaceholderText"):
                widget.setPlaceholderText(self._get_env_default_placeholder(key))
        finally:
            del blocker


def _refresh_env_api_groups(self, *, force: bool = False):
    if not all(
        hasattr(self, attr)
        for attr in (
            "env_group_container_layout",
            "ocr_container_layout",
            "color_container_layout",
            "render_container_layout",
        )
    ):
        return

    active_api_groups = _selected_api_group_keys(self.controller.config_service.get_config())
    current_env_values = self.controller.config_service.load_env_vars()
    structure_signature = json.dumps({
        "api_groups": _env_group_structure_signature(active_api_groups, current_env_values),
        "translator": _normalize_selected_value(self.controller.config_service.get_config().translator.translator),
    }, sort_keys=True)
    value_signature = json.dumps(
        {
            "env": current_env_values,
            "preset": self.controller.config_service.get_current_preset(),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    if not force and getattr(self, "_env_api_groups_structure_signature", None) == structure_signature:
        self._refresh_api_feature_selectors()
        if getattr(self, "_env_api_groups_signature", None) == value_signature:
            return
        _sync_env_widget_values(self, current_env_values)
        self._env_api_groups_signature = value_signature
        return

    self._env_api_groups_structure_signature = structure_signature
    self._env_api_groups_signature = value_signature
    self.env_widgets.clear()
    self._api_slot_status_widgets = []
    for layout in [
        self.env_group_container_layout,
        self.ocr_container_layout,
        self.color_container_layout,
        self.render_container_layout,
    ]:
        _clear_layout_widgets(layout)

    _add_api_section_panel(
        self,
        self.env_group_container_layout,
        "translation",
        active_api_groups["translation"],
        current_env_values,
        {"codex": "Codex CLI account hint", "claude": "Claude CLI account hint"}.get(
            _normalize_selected_value(self.controller.config_service.get_config().translator.translator),
            "No translation API required",
        ),
    )
    self.env_group_container_layout.addStretch()

    _add_api_section_panel(
        self,
        self.ocr_container_layout,
        "ocr",
        active_api_groups["ocr"],
        current_env_values,
        "No OCR API required",
    )
    self.ocr_container_layout.addStretch()

    _add_api_section_panel(
        self,
        self.color_container_layout,
        "color",
        active_api_groups["color"],
        current_env_values,
        "No colorization API required",
    )
    self.color_container_layout.addStretch()

    _add_api_section_panel(
        self,
        self.render_container_layout,
        "render",
        active_api_groups["render"],
        current_env_values,
        "No render API required",
    )
    self.render_container_layout.addStretch()


def _get_setting_description(view, full_key: str) -> str:
    """通过 i18n 获取设置项描述，key 格式为 desc_{full_key} (. 替换为 _)"""
    desc_key = "desc_" + full_key.replace(".", "_")
    if hasattr(view, '_t'):
        result = view._t(desc_key)
        # When the translation result equals the key itself, there is no translation for it
        if result != desc_key:
            return result
    return ""


def _append_settings_row(parent_layout, row: QWidget):
    insert_at = parent_layout.count()
    if insert_at:
        last_item = parent_layout.itemAt(insert_at - 1)
        if last_item and last_item.spacerItem():
            insert_at -= 1
    parent_layout.insertWidget(insert_at, row)


def _insert_settings_row(parent_layout, index: int, row: QWidget):
    parent_layout.insertWidget(index, row)



def _open_filter_list(self):
    """打开过滤列表编辑器"""
    from manga_translator.utils.text_filter import ensure_filter_list_exists

    filter_path = ensure_filter_list_exists()
    from ui.secondary_pages.filter_list_editor import FilterListEditorDialog

    dialog = FilterListEditorDialog(filter_path, t_func=self._t, parent=self._dialog_parent())
    dialog.exec()


def _open_ai_ocr_prompt_editor(self):
    _open_fixed_prompt_editor(self, "ocr.ai_ocr_prompt_path")


def _open_ai_colorizer_prompt_editor(self):
    _open_fixed_prompt_editor(self, "colorizer.ai_colorizer_prompt_path")


def _open_ai_renderer_prompt_editor(self):
    _open_fixed_prompt_editor(self, "render.ai_renderer_prompt_path")


def _get_fixed_prompt_editor_spec(self, full_key: str):
    if full_key == "ocr.ai_ocr_prompt_path":
        from manga_translator.ocr.prompt_loader import (
            DEFAULT_AI_OCR_PROMPT,
            DEFAULT_AI_OCR_PROMPT_PATH,
            ensure_ai_ocr_prompt_file,
            load_ai_ocr_prompt_file,
            save_ai_ocr_prompt_file,
        )

        return {
            "label": self._t("label_ai_ocr_prompt_path"),
            "description": self._t("desc_ocr_ai_ocr_prompt_path"),
            "section": self._t("label_ai_ocr_prompt_path"),
            "hint": DEFAULT_AI_OCR_PROMPT_PATH,
            "default_prompt": DEFAULT_AI_OCR_PROMPT,
            "ensure_func": ensure_ai_ocr_prompt_file,
            "load_func": load_ai_ocr_prompt_file,
            "save_func": save_ai_ocr_prompt_file,
        }

    if full_key == "colorizer.ai_colorizer_prompt_path":
        from manga_translator.colorization.prompt_loader import (
            DEFAULT_AI_COLORIZER_PROMPT,
            DEFAULT_AI_COLORIZER_PROMPT_PATH,
            ensure_ai_colorizer_prompt_file,
            load_ai_colorizer_prompt_file,
            save_ai_colorizer_prompt_file,
        )

        return {
            "label": self._t("label_ai_colorizer_prompt_path"),
            "description": self._t("desc_colorizer_ai_colorizer_prompt_path"),
            "section": self._t("label_ai_colorizer_prompt_path"),
            "hint": DEFAULT_AI_COLORIZER_PROMPT_PATH,
            "default_prompt": DEFAULT_AI_COLORIZER_PROMPT,
            "ensure_func": ensure_ai_colorizer_prompt_file,
            "load_func": load_ai_colorizer_prompt_file,
            "save_func": save_ai_colorizer_prompt_file,
        }

    if full_key == "render.ai_renderer_prompt_path":
        from manga_translator.rendering.prompt_loader import (
            DEFAULT_AI_RENDERER_PROMPT,
            DEFAULT_AI_RENDERER_PROMPT_PATH,
            ensure_ai_renderer_prompt_file,
            load_ai_renderer_prompt_file,
            save_ai_renderer_prompt_file,
        )

        return {
            "label": self._t("label_ai_renderer_prompt_path"),
            "description": self._t("desc_render_ai_renderer_prompt_path"),
            "section": self._t("label_ai_renderer_prompt_path"),
            "hint": DEFAULT_AI_RENDERER_PROMPT_PATH,
            "default_prompt": DEFAULT_AI_RENDERER_PROMPT,
            "ensure_func": ensure_ai_renderer_prompt_file,
            "load_func": load_ai_renderer_prompt_file,
            "save_func": save_ai_renderer_prompt_file,
        }

    return None


def _open_fixed_prompt_editor(self, full_key: str):
    spec = _get_fixed_prompt_editor_spec(self, full_key)
    if not spec:
        return

    abs_path = spec["ensure_func"](spec["hint"])
    if full_key == "colorizer.ai_colorizer_prompt_path":
        from ui.secondary_pages.ai_colorizer_prompt_editor import (
            AIColorizerPromptEditorDialog,
        )

        dialog = AIColorizerPromptEditorDialog(abs_path, t_func=self._t, parent=self._dialog_parent())
        dialog.exec()
        return

    from ui.secondary_pages.simple_prompt_editor_dialog import SimplePromptEditorDialog
    dialog = SimplePromptEditorDialog(
        abs_path,
        title_text=spec["label"],
        description_text=spec["description"],
        section_text=spec["section"],
        hint_text=spec["hint"],
        default_prompt_text=spec["default_prompt"],
        ensure_prompt_func=spec["ensure_func"],
        load_prompt_func=spec["load_func"],
        save_prompt_func=spec["save_func"],
        t_func=self._t,
        parent=self._dialog_parent(),
    )
    dialog.exec()


def _create_fixed_prompt_editor_row(self, parent_layout, full_key: str):
    spec = _get_fixed_prompt_editor_spec(self, full_key)
    if not spec:
        return False

    label_text = spec["label"]

    edit_button = QPushButton(self._t("Edit"))
    edit_button.setFixedWidth(120)
    if full_key == "ocr.ai_ocr_prompt_path":
        edit_button.clicked.connect(self._open_ai_ocr_prompt_editor)
    elif full_key == "colorizer.ai_colorizer_prompt_path":
        edit_button.clicked.connect(self._open_ai_colorizer_prompt_editor)
    elif full_key == "render.ai_renderer_prompt_path":
        edit_button.clicked.connect(self._open_ai_renderer_prompt_editor)

    row = _ClickableRow(self, full_key, label_text, [edit_button])
    _append_settings_row(parent_layout, row)
    return True


def _iter_rendered_setting_values(self, config: dict):
    if getattr(self, "_settings_tabs_use_reclassify", False):
        seen = set()
        for tab in getattr(self, "settings_tab_layout", []) or []:
            for item in tab.get("items", []):
                if isinstance(item, dict):
                    continue
                full_key = str(item or "").strip()
                if not full_key or full_key in seen:
                    continue
                seen.add(full_key)
                exists, value = _resolve_config_value(config, full_key)
                if exists:
                    yield full_key, full_key.rsplit(".", 1)[-1], value
        return

    for section in _LEGACY_SETTING_SECTIONS:
        values = config.get(section)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            yield f"{section}.{key}", str(key), value
    for key, value in config.items():
        if key not in _LEGACY_SETTING_SECTIONS:
            yield str(key), str(key), value


def _setting_control_kind(full_key: str, key: str, value, options, display_map) -> str | None:
    if full_key in _SKIPPED_SETTING_KEYS:
        return None
    if full_key in _FIXED_PROMPT_KEYS:
        return "prompt-button"
    if full_key == "upscale.upscale_ratio":
        return "combo"
    if full_key == "filter_text_enabled":
        return "toggle-action"
    if full_key == "render.font_family":
        return "font-action"
    if isinstance(value, bool):
        return "toggle-action" if full_key == "use_custom_api_params" else "toggle"
    if isinstance(value, float):
        return "float-input"
    if isinstance(value, int):
        return "int-input"
    if value is None and key in _OPTIONAL_INPUT_KEYS:
        return "optional-input"
    if (isinstance(value, str) or value is None) and (options or display_map):
        return "combo"
    if isinstance(value, str):
        return "text-input"
    return None


def _settings_structure_signature(self, config: dict) -> str | None:
    try:
        rows = []
        for full_key, key, value in _iter_rendered_setting_values(self, config):
            options = self.controller.get_options_for_key(key) or []
            display_map = self.controller.get_display_mapping(key) or {}
            kind = _setting_control_kind(full_key, key, value, options, display_map)
            if kind is not None:
                rows.append((full_key, kind, list(options), dict(display_map)))
        rows.sort(key=lambda row: row[0])
        return json.dumps(
            {
                "rows": rows,
                "tab_layout": getattr(self, "settings_tab_layout", None),
                "reclassify": bool(getattr(self, "_settings_tabs_use_reclassify", False)),
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return None


def _sync_setting_widget_values(self, config: dict) -> bool:
    bindings = getattr(self, "_settings_value_bindings", {})
    try:
        for full_key, (widget, display_map) in list(bindings.items()):
            exists, value = _resolve_config_value(config, full_key)
            if not exists:
                return False

            blocker = QSignalBlocker(widget)
            try:
                if isinstance(widget, ToggleSwitch):
                    widget.setChecked(bool(value))
                elif isinstance(widget, SpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, FontComboBox):
                    widget.setCurrentFamily(str(value or ""))
                elif isinstance(widget, QComboBox):
                    if full_key == "upscale.upscale_ratio":
                        continue
                    target = display_map.get(value, value) if display_map else value
                    if full_key == "translator.high_quality_prompt_path":
                        target = os.path.basename(value) if value else ""
                    if target is None and widget.count():
                        widget.setCurrentIndex(0)
                    else:
                        widget.setCurrentText(str(target or ""))
                elif hasattr(widget, "setText"):
                    widget.setText("" if value is None else str(value))
            finally:
                del blocker

        upscale_binding = bindings.get("upscale.upscale_ratio")
        exists, upscaler = _resolve_config_value(config, "upscale.upscaler")
        if upscale_binding is not None and exists:
            widget = upscale_binding[0]
            blocker = QSignalBlocker(widget)
            try:
                _repopulate_upscale_ratio_options(self, widget, upscaler)
            finally:
                del blocker
        return True
    except (AttributeError, RuntimeError):
        return False


@pyqtSlot(dict)
def set_parameters(self, config: dict):
    """
    Receives a config dictionary and starts the incremental creation of setting widgets.
    """
    render_config = config.get("render", {}) if isinstance(config, dict) else {}
    set_system_fonts_enabled(not bool(render_config.get("disable_system_fonts", False)))
    try:
        config_signature = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        config_signature = None

    structure_signature = _settings_structure_signature(self, config)

    if (
        config_signature is not None
        and getattr(self, "_settings_ui_ready", False)
        and getattr(self, "_settings_rendered_signature", None) == config_signature
    ):
        return

    if (
        structure_signature is not None
        and getattr(self, "_settings_ui_ready", False)
        and getattr(self, "_settings_rendered_structure_signature", None) == structure_signature
        and _sync_setting_widget_values(self, config)
    ):
        self._settings_pending_signature = config_signature
        self._settings_rendered_signature = config_signature
        _refresh_env_api_groups(self)
        self._refresh_api_feature_selectors()
        self._refresh_prompt_manager()
        return

    self._settings_ui_ready = False
    self._settings_pending_signature = config_signature
    self._settings_pending_structure_signature = structure_signature
    self._settings_value_bindings = {}

    # Build generation: increased on every rebuild and checked at each step of the chain; a stale build chain ends itself,
    # so a second config_loaded does not start a second chain concurrently and duplicate the controls.
    self._settings_build_seq = getattr(self, "_settings_build_seq", 0) + 1
    build_seq = self._settings_build_seq

    # Clear existing widgets immediately (the bottom stretch is added back, keeping the spacer convention)
    for panel in self.tab_frames.values():
        _clear_layout_widgets(panel.layout(), restore_stretch=True)
    _drop_cached_settings_widget_refs(self)

    if getattr(self, "_settings_tabs_use_reclassify", False):
        _populate_settings_by_reclassify_layout(self, config)
        self._finalize_settings_ui(build_seq)
        return

    # Store config and sections to process
    self._config_to_process = config
    self._sections_to_process = [
        "translator", "cli", "detector", "inpainter",
        "render", "upscale", "colorizer", "ocr", "app", "global"
    ]

    # Schedule the first chunk of work
    QTimer.singleShot(0, lambda: self._process_next_setting_chunk(build_seq))


def _resolve_config_value(config: dict, full_key: str):
    parts = str(full_key or "").split(".")
    current = config
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _add_settings_divider(self, parent_layout, title: str, is_sub: bool = False):
    row = QWidget()
    row_layout = QHBoxLayout(row)

    if is_sub:
        row_layout.setContentsMargins(16, 8, 8, 4)
        row_layout.setSpacing(8)

        dot_label = CaptionLabel("◆")
        dot_label.setFixedWidth(14)

        title_label = BodyLabel(title)

        row_layout.addWidget(dot_label)
        row_layout.addWidget(title_label)
        row_layout.addWidget(HorizontalSeparator(), 1)
    else:
        row_layout.setContentsMargins(4, 18, 4, 6)
        row_layout.setSpacing(10)

        title_label = StrongBodyLabel(title.upper())
        row_layout.addWidget(title_label)
        row_layout.addWidget(HorizontalSeparator(), 1)

    _append_settings_row(parent_layout, row)


def _create_widget_from_full_key(self, config: dict, full_key: str, parent_layout):
    if full_key in _FIXED_PROMPT_KEYS:
        return _create_fixed_prompt_editor_row(self, parent_layout, full_key)

    exists, value = _resolve_config_value(config, full_key)
    if not exists:
        return False

    if "." in full_key:
        section, key = full_key.split(".", 1)
        return bool(self._create_param_widgets({key: value}, parent_layout, section))
    return bool(self._create_param_widgets({full_key: value}, parent_layout, ""))


def _populate_settings_by_reclassify_layout(self, config: dict):
    rendered_rows = 0
    for tab in getattr(self, "settings_tab_layout", []) or []:
        tab_id = str(tab.get("id", "")).strip()
        panel = self.tab_frames.get(tab_id)
        if not panel:
            continue

        panel_layout = panel.layout()
        if panel_layout is None:
            continue

        has_primary_divider = False
        for item in tab.get("items", []):
            if isinstance(item, dict) and str(item.get("kind", "")).lower() == "divider":
                title_key = str(item.get("title", "")).strip() or "Group"
                title = self._t(title_key)
                is_sub = title_key == "Advanced" and has_primary_divider
                _add_settings_divider(self, panel_layout, title, is_sub=is_sub)
                if not is_sub:
                    has_primary_divider = True
                continue

            full_key = str(item or "").strip()
            if not full_key:
                continue
            if _create_widget_from_full_key(self, config, full_key, panel_layout):
                rendered_rows += 1
    return rendered_rows


def _process_next_setting_chunk(self, build_seq: int):
    """
    Processes one section of the settings UI and schedules the next one.
    build_seq 与当前构建代号不一致时说明本链已过期，直接终止。
    """
    if build_seq != getattr(self, "_settings_build_seq", None):
        return
    if not self._sections_to_process:
        self._finalize_settings_ui(build_seq)
        return

    section = self._sections_to_process.pop(0)
    config = self._config_to_process

    # Use fixed English key names
    panel_map = {
        "translator": self.tab_frames.get("Basic Settings"),
        "cli": self.tab_frames.get("Basic Settings"),
        "detector": self.tab_frames.get("Advanced Settings"),
        "inpainter": self.tab_frames.get("Advanced Settings"),
        "render": self.tab_frames.get("Advanced Settings"),
        "upscale": self.tab_frames.get("Advanced Settings"),
        "colorizer": self.tab_frames.get("Advanced Settings"),
        "ocr": self.tab_frames.get("Options"),
        "app": self.tab_frames.get("Application Settings"),
        "global": self.tab_frames.get("Options"),
    }

    panel = panel_map.get(section)
    if section == "global":
        # Handle the top-level global parameters
        global_params = {k: v for k, v in config.items() if k not in ["translator", "cli", "detector", "inpainter", "render", "upscale", "colorizer", "ocr", "app"]}
        if global_params and panel:
            self._create_param_widgets(global_params, panel.layout(), "")
    elif panel and section in config:
        self._create_param_widgets(config[section], panel.layout(), section)

    # Schedule the next chunk
    QTimer.singleShot(0, lambda: self._process_next_setting_chunk(build_seq))

def _finalize_settings_ui(self, build_seq: int | None = None):
    """
    Called after all incremental updates are done. Sets up dependent UI like .env section.
    """
    if build_seq is not None and build_seq != getattr(self, "_settings_build_seq", None):
        return
    # Add the "unload models after translation" checkbox at the very top of the CLI settings area
    cli_panel = self.tab_frames.get("Basic Settings")
    if cli_panel and not getattr(self, "_settings_tabs_use_reclassify", False):
        cli_layout = cli_panel.layout()
        if cli_layout is not None:
            # Create the toggle switch
            unload_models_checkbox = ToggleSwitch()
            
            # Read the initial state from the configuration
            config = self.config_service.get_config()
            unload_models_checkbox.setCheckedSilently(config.app.unload_models_after_translation)
            
            # Connect the signal
            unload_models_checkbox.checkedChanged.connect(
                lambda checked: self.controller.update_single_config(
                    'app.unload_models_after_translation', 
                    bool(checked)
                )
            )
            
            # Create the label
            label_text = self._t("label_unload_models_after_translation")
            if not label_text or label_text == "label_unload_models_after_translation":
                label_text = "Unload Models After Translation"
            row = _ClickableRow(
                self,
                "app.unload_models_after_translation",
                label_text,
                unload_models_checkbox,
            )
            
            # Insert at the very top (index 0)
            _insert_settings_row(cli_layout, 0, row)
    
    if hasattr(self, "env_tab_widget"):
        # Update tab text matching locale dynamically if needed
        for route_key, title_key in getattr(self, "env_tab_title_keys", {}).items():
            self.env_tab_widget.setItemText(route_key, self._t(title_key))

    # Clear containers
    for layout in [self.env_preset_layout, self.env_group_container_layout, self.ocr_container_layout, self.color_container_layout, self.render_container_layout]:
        _clear_layout_widgets(layout)
                
    # --- Global API preset toolbar ---
    preset_label = BodyLabel(self._t("Preset:"))
    self.preset_combo = QComboBox()
    self.preset_combo.setMinimumWidth(180)
    self.preset_combo.setEditable(False)
    self._refresh_preset_list()

    saved_preset = self.controller.config_service.get_current_preset()
    index = self.preset_combo.findText(saved_preset)
    if index >= 0:
        self.preset_combo.setCurrentIndex(index)

    self.add_preset_button = QPushButton("+")
    self.add_preset_button.setFixedWidth(36)
    set_hover_hint(self.add_preset_button, self._t("Add new preset"))

    self.delete_preset_button = QPushButton(self._t("Delete"))
    set_hover_hint(self.delete_preset_button, self._t("Delete selected preset"))
    self.delete_preset_button.setEnabled(self.preset_combo.currentText() not in ("", "默认"))

    self.env_preset_layout.addWidget(preset_label)
    self.env_preset_layout.addWidget(self.preset_combo)
    self.env_preset_layout.addWidget(self.add_preset_button)
    self.env_preset_layout.addWidget(self.delete_preset_button)
    self.env_preset_layout.addStretch()

    self._current_preset_name = self.preset_combo.currentText() if self.preset_combo.count() > 0 else ""

    self.add_preset_button.clicked.connect(self._on_add_preset_clicked)
    self.delete_preset_button.clicked.connect(self._on_delete_preset_clicked)
    self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
    delete_preset_button = self.delete_preset_button
    self.preset_combo.currentTextChanged.connect(
        lambda preset_name, button=delete_preset_button: button.setEnabled(
            preset_name not in ("", "默认")
        )
    )

    
    _refresh_env_api_groups(self, force=True)
    self._refresh_api_feature_selectors()

    self._refresh_prompt_manager()
    self._settings_rendered_signature = getattr(self, "_settings_pending_signature", None)
    self._settings_rendered_structure_signature = getattr(
        self,
        "_settings_pending_structure_signature",
        None,
    )
    self._settings_ui_ready = True

def _create_dynamic_settings(self):
    """读取配置文件并动态创建所有设置控件"""
    try:
        config = self.config_service.get_config().model_dump() # Get default config
        self.set_parameters(config)
    except Exception as e:
        print(f"Error creating dynamic settings: {e}")


def _on_setting_changed(self, value, full_key, display_map=None):
    """A slot to handle when any setting widget is changed by the user."""
    final_value = value
    # Handle reverse mapping for QComboBox
    if display_map:
        reverse_map = {v: k for k, v in display_map.items()}
        final_value = reverse_map.get(value, value) # Fallback to value itself if not in map
    
    # Special handling: when upscaler changes, update the dynamic upscale_ratio drop-down
    if full_key == "upscale.upscaler":
        self._update_upscale_ratio_options(final_value)

    if full_key == "render.disable_system_fonts":
        set_system_fonts_enabled(not bool(final_value))
    
    self.setting_changed.emit(full_key, final_value)
    if full_key in {
        "translator.translator",
        "ocr.ocr",
        "ocr.secondary_ocr",
        "ocr.use_hybrid_ocr",
        "colorizer.colorizer",
        "render.renderer",
    }:
        QTimer.singleShot(100, lambda: _refresh_env_api_groups(self))

def _on_upscale_ratio_changed(self, text, full_key):
    """处理 upscale_ratio 动态下拉框的变化"""
    config = self.config_service.get_config()
    
    if config.upscale.upscaler == "realcugan":
        # Currently realcugan
        if text == self._t("upscale_ratio_not_use"):
            # Upscaling off
            self.setting_changed.emit("upscale.upscale_ratio", None)
            self.setting_changed.emit("upscale.realcugan_model", None)
        else:
            # text may be a localised display name and has to be converted back to the English value
            display_map = self.controller.get_display_mapping("realcugan_model")
            model_value = text
            
            # With a display_map, do a reverse lookup
            if display_map:
                reverse_map = {v: k for k, v in display_map.items()}
                model_value = reverse_map.get(text, text)
            
            # Extract the ratio from the model name
            scale_str = model_value.split('x')[0] if 'x' in model_value else None
            if scale_str and scale_str.isdigit():
                scale = int(scale_str)
                # Update realcugan_model and upscale_ratio together
                self.setting_changed.emit("upscale.realcugan_model", model_value)
                self.setting_changed.emit("upscale.upscale_ratio", scale)
            else:
                # The ratio cannot be parsed: only the model is updated
                self.setting_changed.emit("upscale.realcugan_model", model_value)
    elif config.upscale.upscaler == "mangajanai":
        # Currently mangajanai: the option is stored in upscale_ratio directly
        if text == self._t("upscale_ratio_not_use"):
            # Upscaling off
            self.setting_changed.emit("upscale.upscale_ratio", None)
        else:
            # Store the option string directly (x2, x4, DAT2 x4)
            self.setting_changed.emit("upscale.upscale_ratio", text)
    else:
        # Currently another upscaling model: text is the ratio
        if text == self._t("upscale_ratio_not_use"):
            self.setting_changed.emit(full_key, None)
        else:
            try:
                ratio = int(text)
                self.setting_changed.emit(full_key, ratio)
            except ValueError:
                self.setting_changed.emit(full_key, None)

def _on_numeric_input_changed(self, text, full_key, value_type):
    """统一处理数值类型输入框的变化（支持 int 和 float）"""
    if not text or not text.strip():
        # Empty = use the default (None)
        self.setting_changed.emit(full_key, None)
    else:
        try:
            value = value_type(text)
            self.setting_changed.emit(full_key, value)
        except ValueError:
            # Invalid input = use the default
            self.setting_changed.emit(full_key, None)

def _update_upscale_ratio_options(self, upscaler):
    """当 upscaler 变化时，更新 upscale_ratio 下拉框的选项"""
    upscale_ratio_widget = getattr(self, "upscale_ratio_combo", None)
    if not upscale_ratio_widget:
        return
    
    # Block the signals (try/finally restores them on the exception path too)
    upscale_ratio_widget.blockSignals(True)
    try:
        _repopulate_upscale_ratio_options(self, upscale_ratio_widget, upscaler)
    finally:
        upscale_ratio_widget.blockSignals(False)


def _repopulate_upscale_ratio_options(self, upscale_ratio_widget, upscaler):
    """清空并按当前 upscaler 重新填充 upscale_ratio 下拉框（调用方负责 blockSignals）。"""
    upscale_ratio_widget.clear()

    if upscaler == "realcugan":
        # Show the list of Real-CUGAN models (with localised names)
        realcugan_models = self.controller.get_options_for_key("realcugan_model")
        display_map = self.controller.get_display_mapping("realcugan_model")
        
        if realcugan_models:
            # With a display_map, use the localised names
            if display_map:
                display_options = [display_map.get(model, model) for model in realcugan_models]
                all_options = [self._t("upscale_ratio_not_use")] + display_options
            else:
                all_options = [self._t("upscale_ratio_not_use")] + realcugan_models
            
            upscale_ratio_widget.addItems(all_options)
        
        # Set the default value
        config = self.config_service.get_config()
        if config.upscale.realcugan_model:
            # With a display_map, show the localised name
            if display_map:
                display_name = display_map.get(config.upscale.realcugan_model, config.upscale.realcugan_model)
                upscale_ratio_widget.setCurrentText(display_name)
            else:
                upscale_ratio_widget.setCurrentText(config.upscale.realcugan_model)
        elif config.upscale.upscale_ratio is None:
            upscale_ratio_widget.setCurrentText(self._t("upscale_ratio_not_use"))
        elif realcugan_models:
            if display_map:
                upscale_ratio_widget.setCurrentText(display_map.get(realcugan_models[0], realcugan_models[0]))
            else:
                upscale_ratio_widget.setCurrentText(realcugan_models[0])
    elif upscaler == "mangajanai":
        # Show the special MangaJaNai options
        mangajanai_options = ["x2", "x4", "DAT2 x4"]
        all_options = [self._t("upscale_ratio_not_use")] + mangajanai_options
        upscale_ratio_widget.addItems(all_options)
        
        # Set the default value - upscale_ratio stores the option string directly
        config = self.config_service.get_config()
        ratio = config.upscale.upscale_ratio
        if ratio is None:
            upscale_ratio_widget.setCurrentText(self._t("upscale_ratio_not_use"))
        elif isinstance(ratio, str) and ratio in mangajanai_options:
            upscale_ratio_widget.setCurrentText(ratio)
        elif ratio == 2:
            upscale_ratio_widget.setCurrentText("x2")
        else:
            upscale_ratio_widget.setCurrentText("x4")
    else:
        # Show the ordinary ratio options
        ratio_options = [self._t("upscale_ratio_not_use"), "2", "3", "4"]
        upscale_ratio_widget.addItems(ratio_options)
        # Set the default value
        config = self.config_service.get_config()
        if config.upscale.upscale_ratio is None:
            upscale_ratio_widget.setCurrentText(self._t("upscale_ratio_not_use"))
        else:
            upscale_ratio_widget.setCurrentText(str(config.upscale.upscale_ratio))

def _create_param_widgets(self, data, parent_layout, prefix=""):
    if not isinstance(data, dict):
        return 0

    added_rows = 0
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key

        # These options are skipped, because a drop-down replaces them or they do not need to be shown in the UI
        # realcugan_model is handled through the dynamic upscale_ratio drop-down
        # gimp_font is deprecated; fonts always use font_family.
        # replace_translation and replace_translation_mode are controlled through the workflow mode drop-down
        # The app paths, favourites, folder sort order and current preset are internal state and are not shown in the UI.
        if full_key in _SKIPPED_SETTING_KEYS:
            continue

        label_text = key
        if full_key == "app.unload_models_after_translation":
            translated = self._t("label_unload_models_after_translation")
            label_text = translated if translated != "label_unload_models_after_translation" else "Unload Models After Translation"
        if self.controller.get_display_mapping('labels') and self.controller.get_display_mapping('labels').get(key):
            label_text = self.controller.get_display_mapping('labels').get(key)
        widget = None

        options = self.controller.get_options_for_key(key)
        display_map = self.controller.get_display_mapping(key)

        if full_key == "filter_text_enabled":
            # Special handling: the filter list switch + the edit filter list button
            checkbox = ToggleSwitch(checked=value)
            checkbox.checkedChanged.connect(lambda checked, k=full_key: self._on_setting_changed(bool(checked), k, None))
            
            open_btn = QPushButton(self._t("btn_open_filter_list"))
            open_btn.clicked.connect(self._open_filter_list)
            widget = [checkbox, open_btn]

        elif full_key == "render.font_family":
            locale_getter = self.i18n.get_current_locale if self.i18n else None
            combo = FontComboBox(locale_getter=locale_getter)
            combo.setMinimumWidth(260)
            if value:
                combo.setCurrentFamily(str(value))
            combo.currentFontChanged.connect(
                lambda _font, k=full_key, c=combo: self._on_setting_changed(c.currentFamily(), k, None)
            )
            button = QPushButton(self._t("Open Directory"))
            button.clicked.connect(self.controller.open_fonts_directory)
            widget = [combo, button]

        elif full_key == "translator.high_quality_prompt_path":
            # Create a custom ComboBox that refreshes the prompt list when it drops down
            class RefreshablePromptComboBox(QComboBox):
                def __init__(self, controller_ref, parent=None):
                    super().__init__(parent)
                    self.controller_ref = controller_ref
                
                def showPopup(self):
                    current_text = self.currentText()
                    self.clear()
                    prompt_files = self.controller_ref.get_hq_prompt_options()
                    if prompt_files:
                        self.addItems(prompt_files)
                    # Restore the value that was selected before
                    if current_text:
                        index = self.findText(current_text)
                        if index >= 0:
                            self.setCurrentIndex(index)
                        else:
                            self.setCurrentText(current_text)
                    super().showPopup()
            
            combo = RefreshablePromptComboBox(self.controller)
            combo.setMinimumWidth(260)
            prompt_files = self.controller.get_hq_prompt_options()
            if prompt_files:
                combo.addItems(prompt_files)
            filename = os.path.basename(value) if value else ""
            combo.setCurrentText(filename)
            combo.currentTextChanged.connect(lambda text, k=full_key: self._on_setting_changed(os.path.join('dict', text).replace('\\', '/') if text else None, k, None))
            button = QPushButton(self._t("Open Directory"))
            button.clicked.connect(self.controller.open_dict_directory)
            widget = [combo, button]

        elif isinstance(value, bool):
            # Special handling: use_custom_api_params needs an "open file" button
            if full_key == "use_custom_api_params":
                checkbox = ToggleSwitch(checked=value)
                checkbox.checkedChanged.connect(lambda checked, k=full_key: self._on_setting_changed(bool(checked), k, None))
                
                open_file_button = QPushButton(self._t("Edit"))
                open_file_button.setFixedWidth(100)
                open_file_button.clicked.connect(self._on_open_custom_api_params_file)
                widget = [checkbox, open_file_button]
            else:
                widget = ToggleSwitch(checked=value)
                widget.checkedChanged.connect(lambda checked, k=full_key: self._on_setting_changed(bool(checked), k, None))

        # Special handling: the dynamic upscale_ratio drop-down (must come before the int/float test)
        elif full_key == "upscale.upscale_ratio":
            widget = QComboBox()
            self.upscale_ratio_combo = widget
            widget.setMinimumWidth(100)  # Set a minimum width, so the options are shown more fully
            
            # The current upscaler value decides which options are shown
            config = self.config_service.get_config()
            current_upscaler = config.upscale.upscaler
            
            if current_upscaler == "realcugan":
                # Show the list of Real-CUGAN models (with localised names)
                realcugan_models = self.controller.get_options_for_key("realcugan_model")
                display_map = self.controller.get_display_mapping("realcugan_model")
                
                if realcugan_models:
                    # With a display_map, use the localised names
                    if display_map:
                        display_options = [display_map.get(model, model) for model in realcugan_models]
                        all_options = [self._t("upscale_ratio_not_use")] + display_options
                    else:
                        all_options = [self._t("upscale_ratio_not_use")] + realcugan_models
                    widget.addItems(all_options)
                
                # Set the current value (from realcugan_model)
                current_model = config.upscale.realcugan_model
                if current_model:
                    # With a display_map, show the localised name
                    if display_map:
                        display_name = display_map.get(current_model, current_model)
                        widget.setCurrentText(display_name)
                    else:
                        widget.setCurrentText(current_model)
                elif value is None:
                    widget.setCurrentText(self._t("upscale_ratio_not_use"))
                elif realcugan_models:
                    widget.setCurrentText(realcugan_models[0])
            elif current_upscaler == "mangajanai":
                # Show the special MangaJaNai options
                mangajanai_options = ["x2", "x4", "DAT2 x4"]
                all_options = [self._t("upscale_ratio_not_use")] + mangajanai_options
                widget.addItems(all_options)
                
                # Set the current value - upscale_ratio stores the option string directly
                if value is None:
                    widget.setCurrentText(self._t("upscale_ratio_not_use"))
                elif isinstance(value, str) and value in mangajanai_options:
                    widget.setCurrentText(value)
                elif value == 2:
                    widget.setCurrentText("x2")
                else:
                    widget.setCurrentText("x4")
            else:
                # Show the ordinary ratio options
                ratio_options = [self._t("upscale_ratio_not_use"), "2", "3", "4"]
                widget.addItems(ratio_options)
                # Set the current value
                if value is None:
                    widget.setCurrentText(self._t("upscale_ratio_not_use"))
                else:
                    widget.setCurrentText(str(value))
            
            widget.currentTextChanged.connect(lambda text, k=full_key: self._on_upscale_ratio_changed(text, k))
        
        elif full_key in {"translator.codex_timeout", "translator.codex_batch_size",
                          "translator.claude_timeout", "translator.claude_batch_size"}:
            widget = SpinBox()
            widget.setRange(10, 3600) if key.endswith("_timeout") else widget.setRange(1, 100)
            widget.setValue(int(value))
            widget.valueChanged.connect(lambda number, k=full_key: self._on_setting_changed(number, k, None))

        elif isinstance(value, (int, float)):
            widget = QLineEdit(str(value))
            widget.editingFinished.connect(lambda k=full_key, w=widget: self._on_numeric_input_changed(w.text(), k, float if isinstance(value, float) else int))

        elif value is None and key in _OPTIONAL_INPUT_KEYS:
            # Handle optional parameters whose value is None (numbers and strings)
            widget = QLineEdit("")
            # Set the placeholder text by parameter name
            if key == 'tile_size':
                widget.setPlaceholderText(self._t("Default: 400"))
                widget.editingFinished.connect(lambda k=full_key, w=widget: self._on_numeric_input_changed(w.text(), k, int))
            elif key == 'line_spacing':
                widget.setPlaceholderText(self._t("Default: 1.0 (multiplier for base spacing)"))
                widget.editingFinished.connect(lambda k=full_key, w=widget: self._on_numeric_input_changed(w.text(), k, float))
            elif key == 'letter_spacing':
                widget.setPlaceholderText(self._t("Default: 1.0 (multiplier for base spacing)"))
                widget.editingFinished.connect(lambda k=full_key, w=widget: self._on_numeric_input_changed(w.text(), k, float))
            elif key == 'font_size':
                widget.setPlaceholderText(self._t("Auto"))
                widget.editingFinished.connect(lambda k=full_key, w=widget: self._on_numeric_input_changed(w.text(), k, int))
            elif key == 'ocr_vl_custom_prompt':
                widget.setMinimumWidth(320)
                widget.setPlaceholderText("OCR: Extract all Arabic text.")
                widget.editingFinished.connect(lambda k=full_key, w=widget: self._on_setting_changed(w.text(), k, None))
            elif key == 'ai_ocr_custom_prompt':
                widget.setMinimumWidth(320)
                widget.setPlaceholderText("Read the text and return only the recognized text.")
                widget.editingFinished.connect(lambda k=full_key, w=widget: self._on_setting_changed(w.text(), k, None))

        elif (isinstance(value, str) or value is None) and (options or display_map):
            widget = QComboBox()
            if key == "translator":
                self.translator_combo = widget
                widget.setMinimumWidth(180)  # Minimum width of the translator drop-down
            elif full_key == "ocr.ocr_vl_language_hint":
                widget.setMinimumWidth(260)  # The full names of OCR languages are long; avoid cutting them off
            else:
                widget.setMinimumWidth(180)
            
            if display_map:
                widget.addItems(list(display_map.values()))
                current_display_name = display_map.get(value) if value is not None else None
                if current_display_name:
                    widget.setCurrentText(current_display_name)
                widget.currentTextChanged.connect(lambda text, k=full_key, dm=display_map: self._on_setting_changed(text, k, dm))
            else:
                widget.addItems(options)
                if value is not None:
                    widget.setCurrentText(value)
                else:
                    # For a None value, the first option is set as the default (usually the "not used" entry)
                    if options:
                        widget.setCurrentText(options[0])
                widget.currentTextChanged.connect(lambda text, k=full_key: self._on_setting_changed(text, k, None))

        elif isinstance(value, str):
            widget = QLineEdit(value)
            if full_key in {"ocr.ocr_vl_custom_prompt", "ocr.ai_ocr_custom_prompt"}:
                widget.setMinimumWidth(320)
                if full_key == "ocr.ocr_vl_custom_prompt":
                    widget.setPlaceholderText("OCR: Extract all Arabic text.")
                else:
                    widget.setPlaceholderText("Read the text and return only the recognized text.")
            widget.editingFinished.connect(lambda k=full_key, w=widget: self._on_setting_changed(w.text(), k, None))
        if widget is not None:
            row = _ClickableRow(self, full_key, label_text, widget)
            value_widget = widget[0] if isinstance(widget, (list, tuple)) else widget
            self._settings_value_bindings[full_key] = (value_widget, dict(display_map or {}))
            _append_settings_row(parent_layout, row)
            added_rows += 1

    return added_rows


class _ClickableRow(SimpleCardWidget):
    """Fluent setting row that keeps the existing description-panel behavior."""

    def __init__(self, view, full_key: str, title: str, widget: QWidget | list[QWidget] | tuple[QWidget, ...]):
        super().__init__()
        self._view = view
        self._full_key = full_key
        self._title = str(title or "").rstrip(":：")
        self._widgets = list(widget) if isinstance(widget, (list, tuple)) else [widget]
        self._selected = False
        self._event_filter_targets: list[QWidget] = []

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        row_layout = QHBoxLayout(self)
        row_layout.setContentsMargins(12, 8, 12, 8)
        row_layout.setSpacing(14)

        self._title_label = BodyLabel(f"{self._title}:")
        self._title_label.setMinimumWidth(120)
        row_layout.addWidget(self._title_label)

        for index, control in enumerate(self._widgets):
            if not isinstance(control, ToggleSwitch):
                control.setSizePolicy(QSizePolicy.Policy.Expanding if index == 0 else QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            row_layout.addWidget(control, 1 if len(self._widgets) == 1 or index == 0 else 0)

        if len(self._widgets) > 1 or any(isinstance(control, ToggleSwitch) for control in self._widgets):
            row_layout.addStretch(1)

        for control in self._widgets:
            self._install_child_event_filter(control)
        self._install_row_event_filter(self._title_label)

    def _install_child_event_filter(self, widget):
        self._install_row_event_filter(widget)
        for child in widget.findChildren(QWidget):
            self._install_row_event_filter(child)

    def _install_row_event_filter(self, widget: QWidget):
        widget.installEventFilter(self)
        self._event_filter_targets.append(widget)

    def _cleanup_event_filters(self):
        targets = list(getattr(self, "_event_filter_targets", []))
        self._event_filter_targets = []
        for widget in targets:
            try:
                widget.removeEventFilter(self)
            except RuntimeError:
                pass

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.FocusIn):
            self._activate()
        return False  # The event is not consumed, so the child controls work normally

    def _activate(self):
        """激活此行：更新描述面板和当前行标记。"""
        desc = _get_setting_description(self._view, self._full_key)
        if hasattr(self._view, '_show_setting_description'):
            self._view._show_setting_description(self._full_key, self._title, desc)

        for old in getattr(self._view, "_highlighted_rows", []):
            try:
                old._set_selected(False)
            except (AttributeError, RuntimeError):
                pass
        self._set_selected(True)
        self._view._highlighted_rows = [self]

    def setText(self, text: str):
        self._title = str(text or "").rstrip(":：")
        self._title_label.setText(f"{self._title}:")

    def _set_selected(self, selected: bool):
        self._selected = selected
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._activate()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._selected:
            return

        from PyQt6.QtGui import QPainter

        accent = themeColor().toRgb()
        accent.setAlpha(220)

        painter = QPainter(self)
        painter.fillRect(0, 7, 3, max(1, self.height() - 14), accent)
        painter.end()
