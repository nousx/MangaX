from typing import Any, Dict, List, Optional

from manga_translator.config import (
    VALID_LAYOUT_MODES,  # noqa: F401 - kept importable from here
    CliFields,
    ColorizerFields,
    DetectorFields,
    InpainterFields,
    OcrFields,
    RenderFields,
    TranslatorFields,
    UpscaleFields,
)
from manga_translator.custom_api_params import migrate_legacy_custom_api_params_config
from pydantic import BaseModel, Field, model_validator

from theme_registry import VALID_THEME_PREFERENCES as REGISTERED_THEME_PREFERENCES
from theme_registry import VALID_THEMES as REGISTERED_THEMES

# The settings shared with the backend are declared once, in the *Fields
# classes of manga_translator.config. Each class below adds what differs for
# the desktop app: plain strings where the backend uses an enum, the defaults
# a new installation starts with, and the switches only the desktop has.


class TranslatorSettings(TranslatorFields):
    translator: str = "openai_hq"
    target_lang: str = "CHS"
    # Relative path; the backend joins it with BASE_PATH (next to app.exe when packaged, the project root in development)
    high_quality_prompt_path: Optional[str] = "dict/prompt_example.yaml"


class OcrSettings(OcrFields):
    ocr: str = "48px"
    use_hybrid_ocr: bool = True
    secondary_ocr: str = "mocr"
    prob: float = 0.1


class DetectorSettings(DetectorFields):
    detector: str = "default"
    box_threshold: float = 0.5
    unclip_ratio: float = 2.5


class InpainterSettings(InpainterFields):
    inpainter: str = "lama_mpe"
    inpainting_precision: str = "fp32"


class RenderSettings(RenderFields):
    renderer: str = "default"
    alignment: str = "auto"
    disable_auto_wrap: bool = True
    font_size_minimum: int = 0
    direction: str = "auto"
    font_family: str = ""
    disable_system_fonts: bool = False
    line_spacing: Optional[float] = 1.0  # Line spacing multiplier, 1.0 by default
    letter_spacing: Optional[float] = 1.0  # Letter spacing multiplier, 1.0 by default


class UpscaleSettings(UpscaleFields):
    upscaler: str = "esrgan"


class ColorizerSettings(ColorizerFields):
    colorizer: str = "none"


class CliSettings(CliFields):
    format: str = "不指定"
    overwrite: bool = True
    save_text: bool = True
    load_text: bool = False
    template: bool = False
    generate_and_export: bool = False
    colorize_only: bool = False
    upscale_only: bool = False  # Upscale-only mode
    inpaint_only: bool = False  # Mode that only outputs the inpainted image


_LEGACY_THEME_MIGRATIONS = {
    ("dark", "teal"): "ocean",
    ("gray", "green"): "forest",
    ("gray", "orange"): "sunset",
    ("dark", "rose"): "rose",
}

_ACCENT_ONLY_THEME_FALLBACKS = {
    "teal": "ocean",
    "green": "forest",
    "orange": "sunset",
    "rose": "rose",
}

_VALID_THEMES = set(REGISTERED_THEMES)
_VALID_THEME_PREFERENCES = set(REGISTERED_THEME_PREFERENCES)


class AppSection(BaseModel):
    last_open_dir: str = "."
    last_output_path: str = ""
    favorite_folders: Optional[List[str]] = None
    folder_dialog_sort: str = "name_ascending"
    theme: str = "light"  # The theme options are defined in one place, theme_registry.py
    theme_user_preference: str = "light"
    ui_language: str = "auto"  # UI language: auto (detected automatically), zh_CN, en_US, ja_JP, ko_KR and so on
    auto_check_updates: bool = True  # Whether to check for a new version at start-up
    use_system_proxy: bool = False  # Whether network requests use the proxy settings of the operating system
    current_preset: str = "默认"  # Name of the preset in use
    editor_ocr: str = "mocr"  # OCR model used by the editor's property panel, separate from the OCR setting of the main page
    editor_translator: str = "openai"  # Translator used by the editor's property panel, separate from the translation setting of the main page
    editor_snap_enabled: bool = False  # Whether snapping is on when a text box is moved or rotated in the editor
    editor_rich_text_popup_enabled: bool = True  # Whether the floating rich-text popup of the editor is shown
    editor_rich_text_popup_pinned: bool = False  # Whether the rich-text popup is pinned in place and kept from hiding automatically
    editor_auto_save_on_switch: bool = True  # Save the project data automatically when switching images
    editor_auto_export_on_switch: bool = True  # Export the rendered image automatically when switching images
    editor_suppress_unsaved_warning: bool = False  # Do not warn about unsaved edits when switching images
    editor_auto_rich_text_rules: bool = (
        True  # Apply the rich-text rules automatically while editing a translation (a paragraph that already has manual rich text is skipped as a whole)
    )
    editor_delete_and_recover: bool = False  # When a text box is deleted, also remove its mask and restore the original image
    unload_models_after_translation: bool = (
        False  # Unload the models after translation (frees more memory, but they have to be loaded again next time)
    )
    saved_colors: Optional[List[str]] = None  # Saved list of frequently used colours
    saved_style_presets: Optional[Dict[str, Dict[str, Any]]] = (
        None  # Style combinations saved by the editor
    )
    saved_rich_text_presets: Optional[Dict[str, Dict[str, Any]]] = (
        None  # Style presets for rich-text fragments
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_theme_variants(cls, data: Any):
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        theme_accent = normalized.get("theme_accent")
        theme_value = normalized.get("theme")
        theme_user_preference = normalized.get("theme_user_preference")

        if theme_value == "system":
            mapped_user_pref = _LEGACY_THEME_MIGRATIONS.get(
                (theme_user_preference, theme_accent)
            )
            if mapped_user_pref:
                normalized["theme_user_preference"] = mapped_user_pref
            elif (
                theme_user_preference not in _VALID_THEME_PREFERENCES
                and theme_accent in _ACCENT_ONLY_THEME_FALLBACKS
            ):
                normalized["theme_user_preference"] = _ACCENT_ONLY_THEME_FALLBACKS[
                    theme_accent
                ]
        else:
            mapped_theme = _LEGACY_THEME_MIGRATIONS.get((theme_value, theme_accent))
            if mapped_theme:
                normalized["theme"] = mapped_theme
            elif (
                theme_value not in _VALID_THEMES
                and theme_accent in _ACCENT_ONLY_THEME_FALLBACKS
            ):
                normalized["theme"] = _ACCENT_ONLY_THEME_FALLBACKS[theme_accent]

            if normalized.get("theme_user_preference") not in _VALID_THEME_PREFERENCES:
                normalized["theme_user_preference"] = normalized.get("theme", "light")

        if normalized.get("theme") not in _VALID_THEMES:
            normalized["theme"] = "light"
        if normalized.get("theme_user_preference") not in _VALID_THEME_PREFERENCES:
            normalized["theme_user_preference"] = "light"
        return normalized


class AppSettings(BaseModel):
    app: AppSection = Field(default_factory=AppSection)
    filter_text_enabled: bool = True  # Whether the filter list is on
    kernel_size: int = 3
    mask_dilation_offset: int = 70
    use_custom_api_params: bool = False  # Whether the custom API parameter file is used (general)
    translator: TranslatorSettings = Field(default_factory=TranslatorSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    inpainter: InpainterSettings = Field(default_factory=InpainterSettings)
    render: RenderSettings = Field(default_factory=RenderSettings)
    upscale: UpscaleSettings = Field(default_factory=UpscaleSettings)
    colorizer: ColorizerSettings = Field(default_factory=ColorizerSettings)
    cli: CliSettings = Field(default_factory=CliSettings)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_custom_api_params(cls, data: Any):
        return migrate_legacy_custom_api_params_config(data)
