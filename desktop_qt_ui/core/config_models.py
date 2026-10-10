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
    # 相对路径，后端会用 BASE_PATH 拼接（打包后=app.exe 同级，开发时=项目根目录）
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
    line_spacing: Optional[float] = 1.0  # 行间距倍率，默认1.0
    letter_spacing: Optional[float] = 1.0  # 字间距倍率，默认1.0


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
    upscale_only: bool = False  # 仅超分模式
    inpaint_only: bool = False  # 仅输出修复图片模式


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
    theme: str = "light"  # 主题选项由 theme_registry.py 统一定义
    theme_user_preference: str = "light"
    ui_language: str = "auto"  # UI语言：auto(自动检测), zh_CN, en_US, ja_JP, ko_KR 等
    auto_check_updates: bool = True  # 启动时是否自动检查新版本
    use_system_proxy: bool = False  # 网络请求是否使用操作系统代理配置
    current_preset: str = "默认"  # 当前使用的预设名称
    editor_ocr: str = "mocr"  # 编辑器属性面板使用的 OCR 模型，与主页 OCR 设置分离
    editor_translator: str = "openai"  # 编辑器属性面板使用的翻译器，与主页翻译设置分离
    editor_snap_enabled: bool = False  # 编辑器文本框移动/旋转时是否启用吸附
    editor_rich_text_popup_enabled: bool = True  # 是否显示编辑器富文本浮动弹窗
    editor_rich_text_popup_pinned: bool = False  # 是否固定富文本浮窗位置并阻止自动隐藏
    editor_auto_save_on_switch: bool = True  # 切图时自动保存工程数据
    editor_auto_export_on_switch: bool = True  # 切图时自动导出渲染图片
    editor_suppress_unsaved_warning: bool = False  # 切图时不再提醒未保存编辑
    editor_auto_rich_text_rules: bool = (
        True  # 编辑译文时自动应用富文本规则（命中已带手工富文本则整段跳过）
    )
    editor_delete_and_recover: bool = False  # 删除文本框时同时移除其蒙版并恢复原图
    unload_models_after_translation: bool = (
        False  # 翻译完成后卸载模型（释放内存更彻底，但下次使用需要重新加载）
    )
    saved_colors: Optional[List[str]] = None  # 保存的常用颜色列表
    saved_style_presets: Optional[Dict[str, Dict[str, Any]]] = (
        None  # 编辑器保存的样式组合
    )
    saved_rich_text_presets: Optional[Dict[str, Dict[str, Any]]] = (
        None  # 富文本片段样式预设
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
    filter_text_enabled: bool = True  # 是否启用过滤列表
    kernel_size: int = 3
    mask_dilation_offset: int = 70
    use_custom_api_params: bool = False  # 是否使用自定义API参数配置文件（通用）
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
