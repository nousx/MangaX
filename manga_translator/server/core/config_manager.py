"""
Configuration management module

Responsible for loading, saving and managing the server configuration and the administrator configuration.
"""

import json
import os
from contextlib import contextmanager
from typing import Optional

from manga_translator import Config
from manga_translator.custom_api_params import migrate_legacy_custom_api_params_config
from manga_translator.runtime_paths import get_config_path
from manga_translator.server_paths import ADMIN_CONFIG_FILE, ensure_server_data_layout
from manga_translator.utils import BASE_PATH

# Configuration file paths
ADMIN_CONFIG_PATH = str(ADMIN_CONFIG_FILE)
# Default configuration file: config/config.json next to app.exe when packaged.
SERVER_CONFIG_PATH = get_config_path('config.json')

# Folder paths
FONTS_DIR = os.path.join(BASE_PATH, 'fonts')
PROMPTS_DIR = os.path.join(BASE_PATH, 'dict')
PROMPTS_DIR = os.path.abspath(PROMPTS_DIR)

# Make sure the folders exist
os.makedirs(FONTS_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)
ensure_server_data_layout()

# i18n paths
desktop_locales_dir = os.path.join(BASE_PATH, 'desktop_qt_ui', 'locales')
desktop_locales_dir = os.path.abspath(desktop_locales_dir)

# Global cache of the translation dictionaries
translations_cache = {}

print(f"[INFO] i18n locales directory: {desktop_locales_dir}")
print(f"[INFO] Fonts directory: {FONTS_DIR}")
print(f"[INFO] Prompts directory: {PROMPTS_DIR}")

# Default admin configuration
DEFAULT_ADMIN_SETTINGS = {
    'visible_sections': ['translator', 'cli', 'detector', 'ocr', 'inpainter', 'render', 'upscale', 'colorizer'],
    'hidden_keys': [
        'upscale.realcugan_model',
        # CLI configuration is hidden by default
        'cli.format',
        'cli.save_quality',
        'cli.overwrite',
        'cli.skip_no_text',
        'cli.save_text',
        'cli.export_from_local_json',
        'cli.load_text',
        'cli.translate_json_only',
        'cli.template',
        # 'cli.attempts',  # no longer hidden, so users can set the number of retries
        'cli.ignore_errors',
        'cli.batch_size',
        'cli.batch_concurrent',
        'cli.use_gpu',
        'cli.verbose',
        'cli.psd_script_only',  # The web UI hides the PSD script mode parameter
        'cli.generate_and_export',
        'cli.colorize_only',
        'cli.upscale_only',
        'cli.inpaint_only',
        # Parameters only for the Qt UI (replace-translation mode)
        'cli.replace_translation',
        'render.enable_template_alignment',
        'render.paste_mask_dilation_pixels',
        # Advanced translator configuration
        'translator.enable_post_translation_check',
        'translator.post_check_max_retry_attempts',
        'translator.post_check_repetition_threshold',
        'translator.post_check_target_lang_threshold',
        'translator.translator_chain',
        'translator.selective_translation',
        'translator.skip_lang',
         'use_custom_api_params',  # Server side only; not shown to users of the web UI
         'render.gimp_font',
         'detector.import_yolo_labels',  # Qt UI only - import fixed YOLO boxes
      ],
    'readonly_keys': [],
    'default_values': {},
    'allowed_translators': [],
    'allowed_languages': [],
    'allowed_workflows': [],
    'permissions': {
        'can_upload_fonts': True,
        'can_delete_fonts': True,
        'can_upload_prompts': True,
        'can_delete_prompts': True,
        'can_add_folders': True,
    },
    'upload_limits': {
        'max_image_size_mb': 10,
        'max_images_per_batch': 50,
    },
    'user_access': {
        'require_password': False,
        'user_password': '',
    },
    'api_key_policy': {
        'require_user_keys': False,
        'allow_server_keys': True,
        'save_user_keys_to_server': False,
    },
    'show_env_to_users': False,
    'announcement': {
        'enabled': False,
        'message': '',
        'type': 'info',
    },
    'registration': {
        'enabled': False,  # Whether user registration is open
        'default_group': 'default',  # Default user group of newly registered users
        'require_approval': False,  # Whether administrator approval is required (reserved)
    },
}

# All available translation workflows
AVAILABLE_WORKFLOWS = [
    'normal',
    'export_trans',
    'export_raw',
    'import_trans',
    'colorize',
    'upscale',
    'inpaint',
]


def load_admin_settings() -> dict:
    """Load the administrator configuration from the file"""
    if os.path.exists(ADMIN_CONFIG_PATH):
        try:
            with open(ADMIN_CONFIG_PATH, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)
                print(f"[INFO] Loaded admin settings from: {ADMIN_CONFIG_PATH}")
                # Merge the default configuration with the loaded one
                import copy
                settings = copy.deepcopy(DEFAULT_ADMIN_SETTINGS)
                for key, value in loaded_settings.items():
                    if key in settings and isinstance(settings[key], list) and isinstance(value, list):
                        # List type: start from the defaults and append items the file has but the defaults do not
                        default_set = set(settings[key])
                        merged = list(settings[key])
                        for item in value:
                            if item not in default_set:
                                merged.append(item)
                        settings[key] = merged
                    elif key in settings and isinstance(settings[key], dict) and isinstance(value, dict):
                        # Dictionary type: deep merge
                        merged_dict = copy.deepcopy(settings[key])
                        merged_dict.update(value)
                        settings[key] = merged_dict
                    else:
                        settings[key] = value
                
                # When the configuration file has no password, try the environment variable
                if not settings.get('admin_password'):
                    env_password = os.environ.get('MANGA_TRANSLATOR_ADMIN_PASSWORD')
                    if env_password and len(env_password) >= 6:
                        settings['admin_password'] = env_password
                        # Save to the configuration file
                        save_admin_settings(settings)
                        print("[INFO] Admin password set from environment variable MANGA_TRANSLATOR_ADMIN_PASSWORD")
                    elif env_password:
                        print("[WARNING] MANGA_TRANSLATOR_ADMIN_PASSWORD is too short (minimum 6 characters)")
                
                return settings
        except Exception as e:
            print(f"[ERROR] Failed to load admin settings: {e}")
            return DEFAULT_ADMIN_SETTINGS.copy()
    else:
        print(f"[INFO] Admin config file not found, using defaults: {ADMIN_CONFIG_PATH}")
        settings = DEFAULT_ADMIN_SETTINGS.copy()
        
        # On first start, try to read the password from the environment variable
        env_password = os.environ.get('MANGA_TRANSLATOR_ADMIN_PASSWORD')
        if env_password and len(env_password) >= 6:
            settings['admin_password'] = env_password
            # Save to the configuration file
            save_admin_settings(settings)
            print("[INFO] Admin password set from environment variable MANGA_TRANSLATOR_ADMIN_PASSWORD")
        elif env_password:
            print("[WARNING] MANGA_TRANSLATOR_ADMIN_PASSWORD is too short (minimum 6 characters)")
        
        return settings


def save_admin_settings(settings: dict) -> bool:
    """Save the administrator configuration to the file"""
    try:
        os.makedirs(os.path.dirname(ADMIN_CONFIG_PATH), exist_ok=True)
        with open(ADMIN_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved admin settings to: {ADMIN_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save admin settings: {e}")
        return False


def load_default_config_dict() -> dict:
    """Load the default configuration file; returns a dictionary (with the full configuration of the Qt UI)"""
    if os.path.exists(SERVER_CONFIG_PATH):
        try:
            with open(SERVER_CONFIG_PATH, 'r', encoding='utf-8') as f:
                config_dict = migrate_legacy_custom_api_params_config(json.load(f))
            return config_dict
        except Exception as e:
            print(f"[WARNING] Failed to load default config from {SERVER_CONFIG_PATH}: {e}")
            return {}
    else:
        print(f"[WARNING] Default config file not found: {SERVER_CONFIG_PATH}")
        return {}


def load_default_config() -> Config:
    """Load the default configuration file; returns a Config object"""
    config_dict = load_default_config_dict()
    if config_dict:
        try:
            config = Config.model_validate(config_dict)
            return config
        except Exception as e:
            print(f"[WARNING] Failed to parse config: {e}")
            return Config()
    return Config()


def parse_config(config_str: str) -> Config:
    """Parse a configuration; the default configuration is used when it is empty"""
    if not config_str or config_str.strip() in ('{}', ''):
        print("[INFO] No config provided, using default config from config/config.json")
        return load_default_config()
    else:
        config = Config.parse_raw(config_str)
        # Config now has a cli attribute; cli.attempts is stored automatically
        return config


def get_available_workflows(mode: str = 'user', admin_settings: Optional[dict] = None) -> list:
    """
    Get the list of available workflows

    Args:
        mode: 'user' or 'admin'
        admin_settings: dictionary of the administrator settings (optional)

    Returns:
        The list of available workflows
    """
    # In user mode, when the administrator has set a list of allowed workflows
    if mode == 'user' and admin_settings and admin_settings.get('allowed_workflows'):
        allowed = admin_settings['allowed_workflows']
        return [wf for wf in AVAILABLE_WORKFLOWS if wf in allowed]
    
    return AVAILABLE_WORKFLOWS


@contextmanager
def temp_env_vars(env_vars: dict):
    """
    Context manager that sets environment variables temporarily

    Note: this function no longer uses a global lock, because:
    1. concurrency is controlled by translation_semaphore
    2. a global lock would serialise all translation tasks and hurt performance badly
    3. if per-user isolation of API keys is needed, it should be handled at the translator level

    Args:
        env_vars: dictionary of the environment variables to set temporarily
    """
    import logging
    logger = logging.getLogger('manga_translator.server')
    
    if not env_vars:
        # No user environment variables: use the server defaults directly
        yield
        return
    
    logger.debug(f"[TempEnv] Setting temporary env vars: {list(env_vars.keys())}")
    
    # Keep the original values
    original_values = {}
    for key in env_vars:
        original_values[key] = os.environ.get(key)
    
    try:
        # Set the new values
        for key, value in env_vars.items():
            if value:  # Only non-empty values are set
                os.environ[key] = str(value)
                logger.debug(f"[TempEnv] Set {key}=***")
        
        # Clear the translator cache to force it to be created again (only then are the new environment variables read)
        try:
            from manga_translator.translators import translator_cache
            translator_cache.clear()
            logger.debug("[TempEnv] Cleared translator cache")
        except Exception as e:
            logger.warning(f"[TempEnv] Failed to clear translator cache: {e}")
        
        yield
    finally:
        # Restore the original values
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        
        logger.debug("[TempEnv] Restored original env vars")
        
        # Clear the cache once more, so the server's values are used next time
        try:
            from manga_translator.translators import translator_cache
            translator_cache.clear()
        except Exception:
            pass


def init_server_config_file():
    """Initialise the server configuration file (copied from the template when it does not exist)"""
    if not os.path.exists(SERVER_CONFIG_PATH):
        EXAMPLE_CONFIG_PATH = get_config_path('config-example.json')
        if os.path.exists(EXAMPLE_CONFIG_PATH):
            import shutil
            shutil.copy(EXAMPLE_CONFIG_PATH, SERVER_CONFIG_PATH)
            print(f"[INFO] Created server config from template: {SERVER_CONFIG_PATH}")
        else:
            print("[WARNING] Template config not found, will use default Config()")


# ============================================================================
# i18n Functions
# ============================================================================

def load_translation(locale: str) -> dict:
    """Load the translation file of the given language"""
    if locale in translations_cache:
        return translations_cache[locale]
    
    locales_dir = os.path.realpath(desktop_locales_dir)
    locale_file = os.path.realpath(os.path.join(locales_dir, f"{locale}.json"))
    if not locale_file.startswith(locales_dir + os.sep):
        print(f"[WARNING] Invalid locale: {locale}")
        return {}
    if os.path.dirname(locale_file) != locales_dir:
        print(f"[WARNING] Invalid locale: {locale}")
        return {}
    if os.path.isfile(locale_file):
        try:
            with open(locale_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                translations_cache[locale] = translations
                print(f"[INFO] Loaded {len(translations)} translations for {locale}")
                return translations
        except Exception as e:
            print(f"[ERROR] Failed to load translation file {locale_file}: {e}")
            return {}
    else:
        print(f"[WARNING] Translation file not found: {locale_file}")
        return {}


def get_available_locales() -> dict:
    """Get the list of available languages"""
    locales = {}
    if os.path.exists(desktop_locales_dir):
        for filename in os.listdir(desktop_locales_dir):
            if filename.endswith('.json'):
                locale_code = filename[:-5]  # Remove .json
                locales[locale_code] = locale_code
    return locales


# ============================================================================
# Hot reloading of the configuration
# ============================================================================

# Record the modification time of the configuration file, to detect changes
_admin_config_mtime = 0


def reload_admin_settings_if_changed() -> bool:
    """
    Check whether the configuration file changed, and reload it when it did.

    Returns:
        bool: whether the configuration was reloaded
    """
    global admin_settings, _admin_config_mtime
    
    try:
        if not os.path.exists(ADMIN_CONFIG_PATH):
            return False
        
        current_mtime = os.path.getmtime(ADMIN_CONFIG_PATH)
        
        if current_mtime > _admin_config_mtime:
            old_concurrent = admin_settings.get('max_concurrent_tasks', 3)
            
            # Reload the configuration
            admin_settings = load_admin_settings()
            _admin_config_mtime = current_mtime
            
            new_concurrent = admin_settings.get('max_concurrent_tasks', 3)
            
            # When the concurrency changed, update the semaphore
            if old_concurrent != new_concurrent:
                from .task_manager import update_server_config
                update_server_config({'max_concurrent_tasks': new_concurrent})
                print(f"[INFO] Configuration hot reload: max_concurrent_tasks {old_concurrent} -> {new_concurrent}")
            
            return True
        
        return False
    except Exception as e:
        print(f"[WARNING] Configuration hot reload failed: {e}")
        return False


def get_admin_settings() -> dict:
    """
    Get the administrator configuration (the hot reload is checked automatically)
    """
    reload_admin_settings_if_changed()
    return admin_settings


# ============================================================================
# Module Initialization
# ============================================================================

# Load the admin configuration (module level)
admin_settings = load_admin_settings()

# Initialise the modification time of the configuration file
if os.path.exists(ADMIN_CONFIG_PATH):
    _admin_config_mtime = os.path.getmtime(ADMIN_CONFIG_PATH)

print(f"[INFO] Available locales: {list(get_available_locales().keys())}")
