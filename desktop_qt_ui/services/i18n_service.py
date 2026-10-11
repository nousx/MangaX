"""
Internationalisation support module.
Provides translation into several languages and localisation
"""
import json
import locale
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class LocaleInfo:
    """Locale information"""
    code: str  # Language code, such as 'zh_CN'
    name: str  # Language name in the language itself
    english_name: str  # English name, such as 'Simplified Chinese'
    direction: str = "ltr"  # Text direction: ltr (left to right) or rtl (right to left)

class I18nManager:
    """Internationalisation manager"""
    
    def __init__(self, locale_dir: str = "locales", fallback_locale: str = "zh_CN", config_language: str = "auto"):
        if not os.path.isabs(locale_dir):
            # Relative paths are based on the folder of this service.
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.locale_dir = os.path.join(current_dir, '..', locale_dir)
        else:
            self.locale_dir = locale_dir
        
        self.fallback_locale = fallback_locale
        self.translations: Dict[str, Dict[str, str]] = {}
        self.available_locales: Dict[str, LocaleInfo] = {}
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(self.locale_dir, exist_ok=True)
        
        # Initialise the supported languages
        self._init_supported_locales()
        
        # Decide the language from the configuration
        if config_language == "auto":
            # Detect the system language automatically
            system_locale = self._detect_system_locale()
            if system_locale and system_locale in self.available_locales:
                self.current_locale = system_locale
            else:
                self.current_locale = fallback_locale
        else:
            # Use the configured language
            if config_language in self.available_locales:
                self.current_locale = config_language
            else:
                self.current_locale = fallback_locale
        
        # Load the translations
        self._load_all_translations()
        
    
    def _init_supported_locales(self):
        """Initialise the list of supported languages"""
        self.available_locales = {
            "zh_CN": LocaleInfo("zh_CN", "简体中文", "Simplified Chinese"),
            "zh_TW": LocaleInfo("zh_TW", "繁體中文", "Traditional Chinese"),
            "en_US": LocaleInfo("en_US", "English", "English"),
            "th_TH": LocaleInfo("th_TH", "ไทย", "Thai"),
            "ja_JP": LocaleInfo("ja_JP", "日本語", "Japanese"),
            "ko_KR": LocaleInfo("ko_KR", "한국어", "Korean"),
            "es_ES": LocaleInfo("es_ES", "Español", "Spanish"),
        }
    
    def _detect_system_locale(self) -> str:
        """Detect the system language"""
        try:
            # Try to get the system language
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                # Normalise the language code
                if '_' not in system_locale and len(system_locale) == 2:
                    # With a language code only, add the default country code
                    lang_country_map = {
                        'zh': 'zh_CN',
                        'en': 'en_US',
                        'th': 'th_TH',
                        'ja': 'ja_JP',
                        'ko': 'ko_KR',
                        'es': 'es_ES',
                        'fr': 'fr_FR',
                        'de': 'de_DE',
                        'it': 'it_IT',
                        'pt': 'pt_BR',
                        'ru': 'ru_RU',
                        'ar': 'ar_SA'
                    }
                    system_locale = lang_country_map.get(system_locale, self.fallback_locale)
                
                return system_locale
                
        except Exception as e:
            self.logger.warning(f"Failed to detect system language: {e}")
        
        return self.fallback_locale
    
    def _load_all_translations(self):
        """Load the translations of all languages"""
        for locale_code in self.available_locales.keys():
            self._load_locale_translation(locale_code)
    
    def _load_locale_translation(self, locale_code: str):
        """Load the translations of a specific language"""
        try:
            translation_file = os.path.join(self.locale_dir, f"{locale_code}.json")
            
            if os.path.exists(translation_file):
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self.translations[locale_code] = json.load(f)
                self.logger.debug(f"Loading translation file: {translation_file}")
            else:
                # When the translation file does not exist, create an empty translation dictionary
                self.translations[locale_code] = {}
                
                # Create basic translation files for the main languages
                if locale_code in ['zh_CN', 'en_US']:
                    self._create_base_translation_file(locale_code)
                    
        except Exception as e:
            self.logger.error(f"Failed to load translation file for {locale_code}: {e}")
            self.translations[locale_code] = {}
    
    def _create_base_translation_file(self, locale_code: str):
        """Create the base translation file"""
        try:
            base_translations = self._get_base_translations(locale_code)
            
            translation_file = os.path.join(self.locale_dir, f"{locale_code}.json")
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(base_translations, f, ensure_ascii=False, indent=2)
            
            self.translations[locale_code] = base_translations
            self.logger.info(f"Creating base translation file: {translation_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create base translation file: {e}")
    
    def _get_base_translations(self, locale_code: str) -> Dict[str, str]:
        """Get the base translation content"""
        if locale_code == "zh_CN":
            return {
                # Menus and buttons
                "File": "文件",
                "Edit": "编辑",
                "View": "视图",
                "Tools": "工具",
                "Help": "帮助",
                "Open": "打开",
                "Save": "保存",
                "Exit": "退出",
                "Cancel": "取消",
                "OK": "确定",
                "Yes": "是",
                "No": "否",
                
                # Application title and interface
                "Manga Image Translator UI": "漫画图片翻译器 UI",
                "Main View": "主视图",
                "Editor View": "编辑器视图",
                "Settings": "设置",
                "About": "关于",
                
                # Translation
                "Start Translation": "开始翻译",
                "Stop Translation": "停止翻译",
                "Stopping...": "停止中...",
                "Translation Settings": "翻译设置",
                "Translator": "翻译引擎",
                "Target Language": "目标语言",
                "Source Language": "源语言",
                "Translation Progress": "翻译进度",
                "Translation Complete": "翻译完成",
                "Translation Failed": "翻译失败",
                "Task Completed": "任务完成",
                "Translation completed, {count} files saved.\n\nOpen results in editor?": "翻译完成，已保存 {count} 个文件。\n\n是否在编辑器中打开结果？",
                
                # File operations
                "Add Files": "添加文件",
                "Add Folder": "添加文件夹",
                "Clear List": "清空列表",
                "Remove Selected": "删除选中",
                "Select All": "全选",
                "File List": "文件列表",
                "Output Folder": "输出文件夹",
                
                # Progress and status
                "Progress": "进度",
                "Status": "状态",
                "Ready": "就绪",
                "Processing": "处理中",
                "Completed": "已完成",
                "Error": "错误",
                "Warning": "警告",
                "Information": "信息",
                
                # Editor
                "Editor": "编辑器",
                "Original Text": "原文",
                "Translated Text": "译文",
                "Font Size": "字体大小",
                "Font Color": "字体颜色",
                "Stroke Color": "描边颜色",
                "Stroke Width": "描边宽度",
                "Line Spacing": "行间距",
                "Letter Spacing": "字间距",
                "Rotation": "旋转",
                "Position": "位置",
                "Copy": "复制",
                "Translate": "翻译",
                "Paste": "粘贴",
                "Undo": "撤销",
                "Redo": "重做",
                
                # Configuration
                "Configuration": "配置",
                "Load Config": "加载配置",
                "Save Config": "保存配置",
                "Reset Config": "重置配置",
                "API Settings": "API设置",
                "Advanced Settings": "高级设置",
                
                # Error messages
                "Error occurred": "发生错误",
                "File not found": "文件未找到",
                "Invalid file format": "无效的文件格式",
                "Network error": "网络错误",
                "API error": "API错误",
                "Configuration error": "配置错误",
                
                # Success messages
                "Operation successful": "操作成功",
                "File saved successfully": "文件保存成功",
                "Configuration loaded": "配置已加载",
                "Translation completed successfully": "翻译完成",
                
                # API test
                "Test": "测试",
                "Testing": "测试中",
                "Get Models": "获取模型",
                "Select Model": "选择模型",
                "Available models:": "可用模型：",
                "Please enter API key first": "请先输入API密钥",
                "Testing API connection, please wait...": "正在测试API连接，请稍候...",
                "API connection test successful!": "API连接测试成功！",
                "API connection test failed": "API连接测试失败",
                "Fetching models, please wait...": "正在获取模型列表，请稍候...",
                "Failed to get models": "获取模型列表失败",
            }
        elif locale_code == "en_US":
            return {
                # Menus and buttons
                "File": "File",
                "Edit": "Edit",
                "View": "View",
                "Tools": "Tools",
                "Help": "Help",
                "Open": "Open",
                "Save": "Save",
                "Exit": "Exit",
                "Cancel": "Cancel",
                "OK": "OK",
                "Yes": "Yes",
                "No": "No",
                
                # Application title and interface
                "Manga Image Translator UI": "Manga Image Translator UI",
                "Main View": "Main View",
                "Editor View": "Editor View",
                "Settings": "Settings",
                "About": "About",
                
                # Translation
                "Start Translation": "Start Translation",
                "Stop Translation": "Stop Translation",
                "Translation Settings": "Translation Settings",
                "Translator": "Translator",
                "Target Language": "Target Language",
                "Source Language": "Source Language",
                "Translation Progress": "Translation Progress",
                "Translation Complete": "Translation Complete",
                "Translation Failed": "Translation Failed",
                
                # The rest stays in English as it is
                "Add Files": "Add Files",
                "Add Folder": "Add Folder",
                "Clear List": "Clear List",
                "Remove Selected": "Remove Selected",
                "Select All": "Select All",
                "File List": "File List",
                "Output Folder": "Output Folder",
                "Copy": "Copy",
                "Translate": "Translate",
                "Paste": "Paste",
            }
        else:
            # Other languages return an empty dictionary and use the fallback mechanism
            return {}
    
    def set_locale(self, locale_code: str) -> bool:
        """Set the current language"""
        if locale_code not in self.available_locales:
            self.logger.warning(f"Unsupported language: {locale_code}")
            return False
        
        old_locale = self.current_locale
        self.current_locale = locale_code
        
        # The translations are reloaded on every language switch, so entries updated while running take effect at once
        self._load_locale_translation(locale_code)
        
        self.logger.info(f"Switching language: {old_locale} -> {locale_code}")
        return True
    
    def get_current_locale(self) -> str:
        """Get the code of the current language"""
        return self.current_locale
    
    def get_locale_info(self, locale_code: str = None) -> Optional[LocaleInfo]:
        """Get the information of a language"""
        if locale_code is None:
            locale_code = self.current_locale
        return self.available_locales.get(locale_code)
    
    def get_available_locales(self) -> Dict[str, LocaleInfo]:
        """Get all available languages"""
        return self.available_locales.copy()
    
    def translate(self, key: str, locale_code: str = None, **kwargs) -> str:
        """Translate a text"""
        if locale_code is None:
            locale_code = self.current_locale
        
        # Check whether the key exists in the translations of the current language
        locale_translations = self.translations.get(locale_code, {})
        
        if key in locale_translations:
            # The key exists: use the translation of the current language
            translation = locale_translations[key]
        elif locale_code != self.fallback_locale:
            # The key does not exist and this is not the fallback language: try the fallback language
            fallback_translations = self.translations.get(self.fallback_locale, {})
            translation = fallback_translations.get(key, key)
        else:
            # The key does not exist and this already is the fallback language: return the key itself
            translation = key
        
        # Format the translation (supports parameter substitution)
        if kwargs and translation != key:
            try:
                translation = translation.format(**kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to format translation {key}: {e}")
        
        return translation
    
    def _get_translation(self, key: str, locale_code: str) -> str:
        """Get a translation"""
        locale_translations = self.translations.get(locale_code, {})
        return locale_translations.get(key, key)
    
    def add_translation(self, key: str, value: str, locale_code: str = None):
        """Add a translation"""
        if locale_code is None:
            locale_code = self.current_locale
        
        if locale_code not in self.translations:
            self.translations[locale_code] = {}
        
        self.translations[locale_code][key] = value
    
    def add_translations(self, translations: Dict[str, str], locale_code: str = None):
        """Add translations as a batch"""
        if locale_code is None:
            locale_code = self.current_locale
        
        if locale_code not in self.translations:
            self.translations[locale_code] = {}
        
        self.translations[locale_code].update(translations)
    
    def save_translations(self, locale_code: str = None) -> bool:
        """Save the translations to files"""
        try:
            if locale_code is None:
                # Save all languages
                for code in self.translations.keys():
                    self._save_locale_translation(code)
                return True
            else:
                return self._save_locale_translation(locale_code)
                
        except Exception as e:
            self.logger.error(f"Failed to save translations: {e}")
            return False
    
    def _save_locale_translation(self, locale_code: str) -> bool:
        """Save the translations of a specific language"""
        try:
            translation_file = os.path.join(self.locale_dir, f"{locale_code}.json")
            translations = self.translations.get(locale_code, {})
            
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"Saving translation file: {translation_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save translation file for {locale_code}: {e}")
            return False
    
    def export_missing_keys(self, locale_code: str, output_file: str) -> bool:
        """Export the missing translation keys"""
        try:
            # Get every key of the default language
            default_keys = set(self.translations.get(self.fallback_locale, {}).keys())
            
            # Get the keys of the target language
            target_keys = set(self.translations.get(locale_code, {}).keys())
            
            # Find the missing keys
            missing_keys = default_keys - target_keys
            
            if missing_keys:
                missing_translations = {}
                for key in missing_keys:
                    missing_translations[key] = ""  # An empty value, waiting for translation
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(missing_translations, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"Exporting {len(missing_keys)} missing translations to: {output_file}")
                return True
            else:
                self.logger.info(f"No missing translations for language {locale_code}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export missing translations: {e}")
            return False
    
    def get_text_direction(self, locale_code: str = None) -> str:
        """Get the text direction"""
        if locale_code is None:
            locale_code = self.current_locale
        
        locale_info = self.get_locale_info(locale_code)
        return locale_info.direction if locale_info else "ltr"
    
    def is_rtl_language(self, locale_code: str = None) -> bool:
        """Whether the language is written right to left"""
        return self.get_text_direction(locale_code) == "rtl"

# Global internationalisation manager
_i18n_manager = None

def get_i18n_manager() -> I18nManager:
    """Get the global internationalisation manager"""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
    return _i18n_manager

def setup_i18n(locale_dir: str = "locales", fallback_locale: str = "zh_CN", config_language: str = "auto") -> I18nManager:
    """Set up internationalisation"""
    global _i18n_manager
    _i18n_manager = I18nManager(locale_dir, fallback_locale, config_language)
    return _i18n_manager

# Convenience functions
def _(key: str, **kwargs) -> str:
    """Short alias of the translation function"""
    return get_i18n_manager().translate(key, **kwargs)

def set_language(locale_code: str) -> bool:
    """Convenience function that sets the language"""
    return get_i18n_manager().set_locale(locale_code)

def get_current_language() -> str:
    """Convenience function that gets the current language"""
    return get_i18n_manager().get_current_locale()

def get_available_languages() -> Dict[str, LocaleInfo]:
    """Convenience function that gets the available languages"""
    return get_i18n_manager().get_available_locales()
