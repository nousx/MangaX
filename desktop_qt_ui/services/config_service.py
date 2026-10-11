"""
Configuration management service.
Responsible for loading, saving and validating the application configuration and for managing environment variables
"""

import json
import logging
import os
import re
import sys
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv.parser import parse_stream
from manga_translator.api_key_rotation import (
    env_has_any_indexed_value,
    get_rotation_env_keys,
    get_rotation_slot_count,
)
from manga_translator.colorization.prompt_loader import ensure_ai_colorizer_prompt_file
from manga_translator.custom_api_params import (
    ensure_custom_api_params_file,
    migrate_legacy_custom_api_params_config,
)
from manga_translator.ocr.prompt_loader import ensure_ai_ocr_prompt_file
from manga_translator.rendering.prompt_loader import ensure_ai_renderer_prompt_file
from manga_translator.runtime_paths import get_config_path
from manga_translator.utils.dotenv_utils import (
    APP_DOTENV_PATH_ENV,
    format_env_line,
    load_app_dotenv,
    read_dotenv_file,
    remove_invalid_dotenv_lines,
    validate_env_key,
)
from manga_translator.utils.openai_compat import resolve_openai_compatible_api_key

from core.config_models import AppSettings
from core.workflow_requirements import required_api_sections

PRESET_SPECIAL_ENV_VARS = [
    "OCR_OPENAI_API_KEY",
    "OCR_OPENAI_MODEL",
    "OCR_OPENAI_API_BASE",
    "OCR_GEMINI_API_KEY",
    "OCR_GEMINI_MODEL",
    "OCR_GEMINI_API_BASE",
    "COLOR_OPENAI_API_KEY",
    "COLOR_OPENAI_MODEL",
    "COLOR_OPENAI_API_BASE",
    "COLOR_GEMINI_API_KEY",
    "COLOR_GEMINI_MODEL",
    "COLOR_GEMINI_API_BASE",
    "RENDER_OPENAI_API_KEY",
    "RENDER_OPENAI_MODEL",
    "RENDER_OPENAI_API_BASE",
    "RENDER_GEMINI_API_KEY",
    "RENDER_GEMINI_MODEL",
    "RENDER_GEMINI_API_BASE",
]

API_ROTATION_ENV_GROUPS = [
    ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_MODEL"),
    ("GEMINI_API_KEY", "GEMINI_API_BASE", "GEMINI_MODEL"),
    ("OCR_OPENAI_API_KEY", "OCR_OPENAI_API_BASE", "OCR_OPENAI_MODEL"),
    ("OCR_GEMINI_API_KEY", "OCR_GEMINI_API_BASE", "OCR_GEMINI_MODEL"),
    ("COLOR_OPENAI_API_KEY", "COLOR_OPENAI_API_BASE", "COLOR_OPENAI_MODEL"),
    ("COLOR_GEMINI_API_KEY", "COLOR_GEMINI_API_BASE", "COLOR_GEMINI_MODEL"),
    ("RENDER_OPENAI_API_KEY", "RENDER_OPENAI_API_BASE", "RENDER_OPENAI_MODEL"),
    ("RENDER_GEMINI_API_KEY", "RENDER_GEMINI_API_BASE", "RENDER_GEMINI_MODEL"),
]

RUNTIME_API_REQUIREMENTS = {
    "openai": {
        "display_name": "OpenAI",
        "accepted_env_vars": ["OPENAI_API_KEY"],
        "accepted_base_env_vars": ["OPENAI_API_BASE"],
        "allow_empty_api_key_for_local_base": True,
    },
    "openai_hq": {
        "display_name": "OpenAI HQ",
        "accepted_env_vars": ["OPENAI_API_KEY"],
        "accepted_base_env_vars": ["OPENAI_API_BASE"],
        "allow_empty_api_key_for_local_base": True,
    },
    "gemini": {
        "display_name": "Gemini",
        "accepted_env_vars": ["GEMINI_API_KEY"],
    },
    "gemini_hq": {
        "display_name": "Gemini HQ",
        "accepted_env_vars": ["GEMINI_API_KEY"],
    },
    "openai_ocr": {
        "display_name": "OpenAI OCR",
        "accepted_env_vars": ["OCR_OPENAI_API_KEY", "OPENAI_API_KEY"],
        "accepted_base_env_vars": ["OCR_OPENAI_API_BASE", "OPENAI_API_BASE"],
        "allow_empty_api_key_for_local_base": True,
    },
    "gemini_ocr": {
        "display_name": "Gemini OCR",
        "accepted_env_vars": ["OCR_GEMINI_API_KEY", "GEMINI_API_KEY"],
    },
    "openai_colorizer": {
        "display_name": "OpenAI Colorizer",
        "accepted_env_vars": ["COLOR_OPENAI_API_KEY", "OPENAI_API_KEY"],
        "accepted_base_env_vars": ["COLOR_OPENAI_API_BASE", "OPENAI_API_BASE"],
        "allow_empty_api_key_for_local_base": True,
    },
    "gemini_colorizer": {
        "display_name": "Gemini Colorizer",
        "accepted_env_vars": ["COLOR_GEMINI_API_KEY", "GEMINI_API_KEY"],
    },
    "openai_renderer": {
        "display_name": "OpenAI Renderer",
        "accepted_env_vars": ["RENDER_OPENAI_API_KEY", "OPENAI_API_KEY"],
        "accepted_base_env_vars": ["RENDER_OPENAI_API_BASE", "OPENAI_API_BASE"],
        "allow_empty_api_key_for_local_base": True,
    },
    "gemini_renderer": {
        "display_name": "Gemini Renderer",
        "accepted_env_vars": ["RENDER_GEMINI_API_KEY", "GEMINI_API_KEY"],
    },
}


@dataclass
class TranslatorConfig:
    """Translator configuration information"""

    name: str
    display_name: str
    required_env_vars: List[str]
    optional_env_vars: List[str] = field(default_factory=list)
    validation_rules: Dict[str, str] = field(default_factory=dict)


from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot


class ConfigService(QObject):
    """Configuration management service"""

    config_changed = pyqtSignal(dict)
    write_failed = pyqtSignal(str)
    SAVE_DEBOUNCE_MS = 250

    def __init__(self, root_dir: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.root_dir = root_dir
        # The .env file belongs in the folder of the exe (a writable location)
        # Packaged: <install folder>/.env
        # Development: <project root>/.env
        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(sys.executable)
            self.env_path = os.path.join(exe_dir, ".env")
        else:
            self.env_path = os.path.join(self.root_dir, ".env")
        os.environ[APP_DOTENV_PATH_ENV] = self.env_path
        self._deferred_write_error: Optional[str] = None
        self._remove_invalid_env_lines()
        try:
            self._env_values = read_dotenv_file(self.env_path)
            load_app_dotenv(self.env_path, override=True)
        except Exception as exc:
            self.logger.error(f"Failed to load .env: {exc}")
            self._env_values = {}

        # Use get_default_config_path() for PyInstaller compatibility
        # Temporarily set a placeholder, will be properly set after initialization
        self.default_config_path = None
        self.user_config_path = None

        self.config_path = None  # This will hold the path of a loaded file
        self.current_config: AppSettings = AppSettings()

        # Set the correct default config path
        self.default_config_path = self.get_default_config_path()
        self.user_config_path = self.get_user_config_path()
        try:
            ensure_custom_api_params_file(logger=self.logger)
            ensure_ai_ocr_prompt_file()
            ensure_ai_renderer_prompt_file()
            ensure_ai_colorizer_prompt_file()
        except Exception as exc:
            self.logger.error(f"Failed to create local configuration template: {exc}")
        self.logger.debug(f"Default configuration: {os.path.basename(self.default_config_path)}")
        self.logger.debug(f"User configuration: {os.path.basename(self.user_config_path)}")
        self.logger.debug(f"Default configuration exists: {os.path.exists(self.default_config_path)}")
        self.logger.debug(f"User configuration exists: {os.path.exists(self.user_config_path)}")
        if getattr(sys, "frozen", False):
            self.logger.debug(
                f"Packaged environment; external configuration directory = {os.path.dirname(self.user_config_path)}"
            )

        # Loading order of the configuration: user configuration > default configuration > defaults in the code
        self._load_configs_with_priority()

        self._translator_configs = None
        self._env_cache = None
        self._config_cache = None
        self._initialize_write_pipeline()

    def _remove_invalid_env_lines(self) -> None:
        try:
            removed = remove_invalid_dotenv_lines(self.env_path)
        except Exception as exc:
            self._deferred_write_error = f"{self.env_path}: {exc}"
            self.logger.error(f"Failed to remove unparseable .env lines: {exc}")
            return
        if removed:
            self.logger.warning(f"Removed {removed} unparseable configuration lines from .env")

    def take_deferred_write_error(self) -> Optional[str]:
        error = self._deferred_write_error
        self._deferred_write_error = None
        return error

    def _initialize_write_pipeline(self) -> None:
        self._write_lock = threading.RLock()
        self._pending_config_writes: Dict[str, Dict[str, Any]] = {}
        self._pending_env_updates: Dict[str, Optional[str]] = {}
        self._pending_env_replacement: Optional[Dict[str, str]] = None
        self._write_futures: set[Future] = set()
        self._write_errors: list[Exception] = []
        self._env_write_failed = False
        self._writer_closed = False
        self._writer_shutdown_started = False
        self._write_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="config-writer",
        )
        self._write_timer = QTimer(self)
        self._write_timer.setSingleShot(True)
        self._write_timer.setInterval(self.SAVE_DEBOUNCE_MS)
        self._write_timer.timeout.connect(self._submit_pending_writes)

    @property
    def translator_configs(self):
        """Load the translator configurations lazily"""
        if self._translator_configs is None:
            self._translator_configs = self._init_translator_configs()
        return self._translator_configs

    def _init_translator_configs(self) -> Dict[str, TranslatorConfig]:
        """Initialise the registry of translator configurations from the JSON file"""
        configs = {}

        config_path = get_config_path("config", "translators.json")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, config_data in data.items():
                configs[name] = TranslatorConfig(**config_data)
        except FileNotFoundError:
            self.logger.error(f"Translator config file not found at: {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load translator configs: {e}")
        return configs

    def get_translator_configs(self) -> Dict[str, TranslatorConfig]:
        """Get all translator configurations"""
        return self.translator_configs

    def get_translator_config(self, translator_name: str) -> Optional[TranslatorConfig]:
        """Get the configuration of a specific translator"""
        return self.translator_configs.get(translator_name)

    def get_required_env_vars(self, translator_name: str) -> List[str]:
        """Get the environment variables a translator requires"""
        config = self.get_translator_config(translator_name)
        return config.required_env_vars if config else []

    def get_all_env_vars(self, translator_name: str) -> List[str]:
        """Get all environment variables related to a translator"""
        config = self.get_translator_config(translator_name)
        if not config:
            return []
        return config.required_env_vars + config.optional_env_vars

    def get_all_preset_env_vars(self) -> List[str]:
        """Get every API environment variable a preset should contain."""
        env_keys: List[str] = []
        seen = set()

        for translator_config in self.translator_configs.values():
            for key in (
                translator_config.required_env_vars
                + translator_config.optional_env_vars
            ):
                if key and key not in seen:
                    seen.add(key)
                    env_keys.append(key)

        for key in PRESET_SPECIAL_ENV_VARS:
            if key not in seen:
                seen.add(key)
                env_keys.append(key)

        current_env_vars = self.load_env_vars()
        for api_key_env, api_base_env, model_env in API_ROTATION_ENV_GROUPS:
            slots = get_rotation_slot_count(
                current_env_vars,
                (api_key_env, api_base_env, model_env),
            )
            for key in get_rotation_env_keys(
                api_key_env, api_base_env, model_env, slots=slots
            ):
                if key not in seen:
                    seen.add(key)
                    env_keys.append(key)

        return env_keys

    @staticmethod
    def _has_env_value(env_vars: Dict[str, str], key: str) -> bool:
        return env_has_any_indexed_value(env_vars, key)

    def get_missing_runtime_api_requirements(
        self,
        config: AppSettings,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get the run-time API key requirements that are missing under the current configuration."""
        merged_env_vars = {
            key: str(value or "") for key, value in self.load_env_vars().items()
        }
        if env_vars:
            for key, value in env_vars.items():
                merged_env_vars[key] = str(value or "")

        checks = [
            (
                "translator",
                "translator",
                getattr(config.translator, "translator", None),
            ),
            ("ocr", "ocr", getattr(config.ocr, "ocr", None)),
            ("colorizer", "colorizer", getattr(config.colorizer, "colorizer", None)),
            ("render", "renderer", getattr(config.render, "renderer", None)),
        ]

        if bool(getattr(config.ocr, "use_hybrid_ocr", False)):
            checks.append(
                ("ocr", "secondary_ocr", getattr(config.ocr, "secondary_ocr", None))
            )

        missing: List[Dict[str, Any]] = []
        active_sections = required_api_sections(config)
        for section, setting, selected_value in checks:
            if section not in active_sections:
                continue
            feature_name = str(selected_value or "").strip()
            if not feature_name:
                continue

            requirement = RUNTIME_API_REQUIREMENTS.get(feature_name)
            if not requirement:
                continue

            accepted_env_vars = list(requirement.get("accepted_env_vars", []))
            if any(
                self._has_env_value(merged_env_vars, key) for key in accepted_env_vars
            ):
                continue

            accepted_base_env_vars = list(requirement.get("accepted_base_env_vars", []))
            if requirement.get("allow_empty_api_key_for_local_base") and any(
                resolve_openai_compatible_api_key("", merged_env_vars.get(key, ""))
                for key in accepted_base_env_vars
            ):
                continue

            missing.append(
                {
                    "section": section,
                    "setting": setting,
                    "selected_value": feature_name,
                    "display_name": requirement.get("display_name", feature_name),
                    "accepted_env_vars": accepted_env_vars,
                }
            )

        return missing

    def validate_api_key(self, key: str, var_name: str, translator_name: str) -> bool:
        """Validate the format of an API key"""
        config = self.get_translator_config(translator_name)
        if not config or var_name not in config.validation_rules:
            return True  # Without a validation rule it counts as valid

        pattern = config.validation_rules[var_name]
        return bool(re.match(pattern, key))

    def load_config_file(self, config_path: str) -> bool:
        """Load the JSON configuration file and merge it with the default settings, validating key by key; a wrong key gets its default value"""
        try:
            if not os.path.exists(config_path):
                self.logger.error(f"Configuration file does not exist: {config_path}")
                return False

            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
                loaded_data = migrate_legacy_custom_api_params_config(
                    json.loads(content)
                )

            # Take the default configuration as the base
            default_config = AppSettings()
            new_config_dict = default_config.model_dump()

            # Merge key by key, safely, validating each value
            error_keys = []

            def safe_deep_update(target, source, path=""):
                """Safe deep merge, validating key by key"""
                for key, value in source.items():
                    current_path = f"{path}.{key}" if path else key
                    try:
                        if (
                            isinstance(value, dict)
                            and key in target
                            and isinstance(target[key], dict)
                        ):
                            # Handle nested dictionaries recursively
                            safe_deep_update(target[key], value, current_path)
                        else:
                            # Try to set the value, to see whether it is valid
                            old_value = target.get(key)
                            target[key] = value

                            # Try to create a configuration object with the new value, to validate it
                            try:
                                AppSettings.model_validate(new_config_dict)
                            except Exception as validate_err:
                                # Validation failed: restore the default value
                                target[key] = old_value
                                error_keys.append(
                                    (current_path, value, str(validate_err))
                                )
                                self.logger.warning(
                                    f"Invalid value for configuration key '{current_path}': {value}; using default: {old_value}"
                                )
                    except Exception as e:
                        error_keys.append((current_path, value, str(e)))
                        self.logger.warning(
                            f"Failed to load configuration key '{current_path}': {e}; keeping default value"
                        )

            safe_deep_update(new_config_dict, loaded_data)

            # Final validation, then create the configuration object
            try:
                self.current_config = AppSettings.model_validate(new_config_dict)
            except Exception as final_err:
                self.logger.error(f"Configuration validation failed; using defaults: {final_err}")
                self.current_config = AppSettings()

            # Report the keys that were wrong
            if error_keys:
                self.logger.warning(
                    f"Replaced {len(error_keys)} invalid configuration entries with default values:"
                )
                for key_path, bad_value, err in error_keys[:5]:  # Only the first 5 are shown
                    self.logger.warning(f"  - {key_path}: {bad_value}")
                if len(error_keys) > 5:
                    self.logger.warning(f"  ... and {len(error_keys) - 5} more")

            self.config_path = config_path
            self.logger.debug(f"Loading configuration: {os.path.basename(config_path)}")
            config_dict = self.current_config.model_dump()
            self.config_changed.emit(config_dict)
            return True

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid configuration JSON: {e}; using default configuration")
            self.current_config = AppSettings()
            return False
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {e}; using default configuration")
            self.current_config = AppSettings()
            return False

    def _build_config_payload(self, save_path: str) -> Dict[str, Any]:
        config_dict = self.current_config.model_dump()
        is_default_config = save_path == self.default_config_path
        if is_default_config:
            config_dict.setdefault("detector", {})["min_box_area_ratio"] = 0

        if is_default_config and not getattr(sys, "frozen", False):
            app = config_dict.setdefault("app", {})
            app.update(
                {
                    "last_open_dir": ".",
                    "last_output_path": "",
                    "favorite_folders": None,
                    "folder_dialog_sort": "name_ascending",
                    "theme": "light",
                    "ui_language": "auto",
                    "current_preset": "默认",
                    "editor_snap_enabled": False,
                    "editor_center_scale_enabled": False,
                    "editor_rich_text_popup_enabled": True,
                    "editor_rich_text_popup_pinned": False,
                    "editor_auto_save_on_switch": True,
                    "editor_auto_export_on_switch": True,
                    "editor_suppress_unsaved_warning": False,
                    "editor_auto_rich_text_rules": True,
                    "editor_delete_and_recover": False,
                    "saved_colors": None,
                    "saved_style_presets": None,
                    "saved_rich_text_presets": None,
                }
            )
            config_dict.setdefault("cli", {})["verbose"] = False
            render = config_dict.setdefault("render", {})
            render.update(
                {
                    "font_family": "Microsoft YaHei UI",
                    "disable_auto_wrap": False,
                    "center_text_in_bubble": False,
                    "optimize_line_breaks": False,
                    "semantic_linebreak": False,
                    "remove_linebreak_punctuation": False,
                    "recompute_line_breaks": False,
                    "check_br_and_retry": False,
                    "strict_smart_scaling": False,
                    "balloon_fill_mask_layout": False,
                }
            )
            config_dict.setdefault("translator", {})["high_quality_prompt_path"] = (
                "dict/prompt_example.yaml"
            )
            config_dict.setdefault("ocr", {})["use_hybrid_ocr"] = False
        return config_dict

    @staticmethod
    def _atomic_write_text(path: str, content: str) -> None:
        target = os.path.abspath(path)
        directory = os.path.dirname(target)
        os.makedirs(directory, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=directory,
                prefix=f".{os.path.basename(target)}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, target)
            temp_path = None
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    @classmethod
    def _merge_dotenv_updates(
        cls,
        path: str,
        updates: Dict[str, Optional[str]],
    ) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as source:
                for mapping in parse_stream(source):
                    key = mapping.key
                    if mapping.error:
                        continue
                    if key in updates:
                        if key not in seen and updates[key] is not None:
                            lines.append(format_env_line(key, updates[key]))
                        seen.add(key)
                    else:
                        lines.append(mapping.original.string)

        for key, value in updates.items():
            if key in seen or value is None:
                continue
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines.append("\n")
            lines.append(format_env_line(key, value))
        return "".join(lines)

    @classmethod
    def _write_snapshots(
        cls,
        config_writes: Dict[str, Dict[str, Any]],
        env_write: Optional[tuple[bool, Dict[str, Optional[str]]]],
        env_path: str,
    ) -> None:
        errors: list[str] = []
        for save_path, payload in config_writes.items():
            try:
                content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
                cls._atomic_write_text(save_path, content)
            except Exception as exc:
                errors.append(f"{save_path}: {exc}")
        if env_write is not None:
            try:
                replace, payload = env_write
                if replace:
                    content = "".join(
                        format_env_line(key, value) for key, value in payload.items()
                    )
                else:
                    content = cls._merge_dotenv_updates(env_path, payload)
                cls._atomic_write_text(env_path, content)
            except Exception as exc:
                errors.append(f"{env_path}: {exc}")
        if errors:
            raise OSError("; ".join(errors))

    def _take_pending_writes(self):
        with self._write_lock:
            config_writes = self._pending_config_writes
            self._pending_config_writes = {}
            if self._pending_env_replacement is not None:
                env_write = (True, self._pending_env_replacement)
            elif self._pending_env_updates:
                env_write = (False, self._pending_env_updates)
            else:
                env_write = None
            self._pending_env_replacement = None
            self._pending_env_updates = {}
        return config_writes, env_write

    @pyqtSlot()
    def _submit_pending_writes(self) -> Optional[Future]:
        if self._writer_closed:
            return None
        config_writes, env_write = self._take_pending_writes()
        if not config_writes and env_write is None:
            return None

        future = self._write_executor.submit(
            self._write_snapshots,
            config_writes,
            env_write,
            self.env_path,
        )
        with self._write_lock:
            self._write_futures.add(future)

        had_env_write = env_write is not None

        def on_done(done_future: Future) -> None:
            error = None
            try:
                done_future.result()
            except Exception as exc:
                error = exc
            with self._write_lock:
                self._write_futures.discard(done_future)
                if error is not None:
                    self._write_errors.append(error)
                    if had_env_write:
                        self._env_write_failed = True
                elif had_env_write:
                    self._env_write_failed = False
            if error is not None:
                message = f"Failed to save configuration in background: {error}"
                self.logger.error(
                    message,
                    exc_info=(type(error), error, error.__traceback__),
                )
                self.write_failed.emit(str(error))

        future.add_done_callback(on_done)
        return future

    def _schedule_write(self) -> bool:
        if self._writer_shutdown_started or self._writer_closed:
            self.logger.warning("Configuration writer is closed; ignoring save request")
            return False
        self._write_timer.start(self.SAVE_DEBOUNCE_MS)
        return True

    def request_save(self) -> bool:
        """Queue the current default config snapshots for a coalesced write."""
        return self.save_config_file()

    def save_config_file(self, config_path: Optional[str] = None) -> bool:
        """Queue a coalesced save; explicit export paths retain synchronous status."""
        try:
            if config_path:
                save_paths = [config_path]
            elif getattr(sys, "frozen", False):
                save_paths = [self.user_config_path]
            else:
                save_paths = [self.user_config_path, self.default_config_path]

            payloads = {
                os.path.abspath(path): self._build_config_payload(path)
                for path in save_paths
                if path
            }
            if not payloads:
                return False
            with self._write_lock:
                self._pending_config_writes.update(payloads)
            self.config_path = self.user_config_path
            if not self._schedule_write():
                return False
            return self.flush_pending_writes() if config_path else True
        except Exception as e:
            self.logger.error(f"Failed to save configuration file: {e}")
            return False

    def reload_config(self):
        """
        Force a complete reload of the configuration from the .env and JSON files.
        This makes sure any change made to the files from outside takes effect in the program.
        """
        self.logger.info("Forcing configuration reload...")
        self.flush_pending_writes()
        self._remove_invalid_env_lines()

        # 1. Reload the .env file into os.environ. The translation engine reads from there automatically.
        load_app_dotenv(self.env_path, override=True)
        with self._write_lock:
            self._env_values = read_dotenv_file(self.env_path)
        self.logger.info(f"Reloaded .env from {self.env_path}; environment variables updated.")

        # 2. Create the AppSettings object again (for the UI settings)
        self.current_config = AppSettings()

        # 3. Reload the configuration files in order of priority
        self._load_configs_with_priority()

        # 4. Tell every listener that the configuration changed
        config_dict = self.current_config.model_dump()
        self.config_changed.emit(config_dict)
        self.logger.info("Configuration reload completed.")

    def reload_from_disk(self):
        """
        Force a reload of the configuration from the currently set config_path, and notify all listeners.
        """
        self.flush_pending_writes()
        if self.config_path and os.path.exists(self.config_path):
            self.logger.debug(f"Reloading configuration from disk: {os.path.basename(self.config_path)}")
            self.load_config_file(self.config_path)
        else:
            self.logger.warning("Cannot reload configuration: config_path is unset or the file does not exist.")

    def get_config(self) -> AppSettings:
        """Get a deep copy of the current configuration model"""
        return self.current_config.model_copy(deep=True)

    def get_config_reference(self) -> AppSettings:
        """Get a direct reference to the current configuration model; use with care."""
        return self.current_config

    def get_current_preset(self) -> str:
        """Get the name of the current preset"""
        return getattr(self.current_config.app, "current_preset", "默认")

    def set_current_preset(self, preset_name: str) -> bool:
        """Set the name of the current preset and save it to the configuration file"""
        try:
            self.current_config.app.current_preset = preset_name
            self.save_config_file()
            # Not logged, to avoid flooding the log
            return True
        except Exception as e:
            self.logger.error(f"Failed to save current preset: {e}")
            return False

    def _convert_config_for_ui(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deprecated: None for upscale_ratio used to be changed into a display string meaning "not used",
        but all the UI downstream branches on `value is None`, and the conversion only confused it (mangajanai fell into else => 'x4').
        The empty implementation is kept for compatibility only; no conversion is done any more.
        """
        return config_dict

    def set_config(self, config: AppSettings) -> None:
        """Set the configuration and notify the listeners"""
        self.current_config = config.model_copy(deep=True)
        self.logger.debug("Configuration updated; notifying listeners...")
        config_dict = self.current_config.model_dump()
        self.config_changed.emit(config_dict)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update part of the configuration"""
        new_config_dict = self.current_config.model_dump()

        def deep_update(target, source):
            for key, value in source.items():
                if (
                    isinstance(value, dict)
                    and key in target
                    and isinstance(target[key], dict)
                ):
                    deep_update(target[key], value)
                else:
                    target[key] = value

        deep_update(new_config_dict, updates)

        self.current_config = AppSettings.model_validate(new_config_dict)
        self.logger.debug("Configuration updated; notifying listeners...")
        config_dict = self.current_config.model_dump()
        self.config_changed.emit(config_dict)

    def load_env_vars(self) -> Dict[str, str]:
        """Return the current in-memory environment snapshot."""
        with self._write_lock:
            return dict(self._env_values)

    def save_env_var(self, key: str, value: str) -> bool:
        """Update memory/os.environ immediately and coalesce the disk write."""
        try:
            return self.save_env_vars({key: value})
        except Exception as e:
            self.logger.error(f"Failed to save environment variable: {e}")
            return False

    def save_env_vars(self, env_vars: Dict[str, str]) -> bool:
        """Apply a batch in memory and persist it with one atomic rewrite."""
        try:
            normalized = {
                validate_env_key(str(key)): (
                    "" if value is None else str(value).strip()
                )
                for key, value in env_vars.items()
            }
            with self._write_lock:
                self._env_values.update(normalized)
                if self._env_write_failed:
                    self._pending_env_replacement = dict(self._env_values)
                    self._pending_env_updates.clear()
                elif self._pending_env_replacement is not None:
                    self._pending_env_replacement.update(normalized)
                else:
                    self._pending_env_updates.update(normalized)
            os.environ.update(normalized)
            self._env_cache = None
            return self._schedule_write()
        except Exception as e:
            self.logger.error(f"Failed to save environment variables in batch: {e}")
            return False

    def delete_env_vars(self, keys: list[str] | tuple[str, ...] | set[str]) -> bool:
        """Delete several environment variables and sync to the running environment at once."""
        try:
            normalized_keys = [validate_env_key(str(key)) for key in keys]
            with self._write_lock:
                for key in normalized_keys:
                    self._env_values.pop(key, None)
                    if self._env_write_failed:
                        self._pending_env_replacement = dict(self._env_values)
                        self._pending_env_updates.clear()
                    elif self._pending_env_replacement is not None:
                        self._pending_env_replacement.pop(key, None)
                    else:
                        self._pending_env_updates[key] = None
            for key in normalized_keys:
                os.environ.pop(key, None)
            self._env_cache = None
            return self._schedule_write()
        except Exception as e:
            self.logger.error(f"Failed to delete environment variable: {e}")
            return False

    def replace_env_file(self, env_vars: Dict[str, str]) -> bool:
        """Replace the content of the .env file completely"""
        try:
            normalized_env_vars = {
                validate_env_key(str(key)): (
                    "" if value is None else str(value).strip()
                )
                for key, value in env_vars.items()
            }
            with self._write_lock:
                old_keys = set(self._env_values)
                self._env_values = dict(normalized_env_vars)
                self._pending_env_replacement = dict(normalized_env_vars)
                self._pending_env_updates.clear()
            for key in old_keys - normalized_env_vars.keys():
                os.environ.pop(key, None)
            os.environ.update(normalized_env_vars)
            self._env_cache = None
            return self._schedule_write()
        except Exception as e:
            self.logger.error(f"Failed to replace .env file: {e}")
            return False

    def flush_pending_writes(self) -> bool:
        """Submit pending snapshots and wait until all accepted writes finish."""
        if QThread.currentThread() is self.thread():
            self._write_timer.stop()
        success = True
        while True:
            self._submit_pending_writes()
            with self._write_lock:
                futures = list(self._write_futures)
                has_pending = (
                    bool(self._pending_config_writes)
                    or bool(self._pending_env_updates)
                    or self._pending_env_replacement is not None
                )
            if not futures and not has_pending:
                with self._write_lock:
                    if self._write_errors:
                        success = False
                        self._write_errors.clear()
                return success
            for future in futures:
                try:
                    future.result()
                except Exception:
                    success = False

    def shutdown(self) -> bool:
        """Flush coalesced writes and stop the writer thread. Idempotent."""
        if self._writer_closed:
            return True
        self._writer_shutdown_started = True
        success = self.flush_pending_writes()
        self._write_executor.shutdown(wait=True, cancel_futures=False)
        self._writer_closed = True
        return success

    def validate_translator_env_vars(self, translator_name: str) -> Dict[str, bool]:
        """Check whether the environment variables of a translator are complete"""
        env_vars = self.load_env_vars()
        required_vars = self.get_required_env_vars(translator_name)

        validation_result = {}
        for var in required_vars:
            value = env_vars.get(var, "")
            is_present = bool(value.strip())
            is_valid_format = (
                self.validate_api_key(value, var, translator_name)
                if is_present
                else True
            )
            validation_result[var] = is_present and is_valid_format

        return validation_result

    def get_missing_env_vars(self, translator_name: str) -> List[str]:
        """Get the missing environment variables"""
        validation_result = self.validate_translator_env_vars(translator_name)
        return [var for var, is_valid in validation_result.items() if not is_valid]

    def is_translator_configured(self, translator_name: str) -> bool:
        """Check whether a translator is fully configured"""
        missing_vars = self.get_missing_env_vars(translator_name)
        return len(missing_vars) == 0

    def get_default_config_path(self) -> str:
        """
        Get the path of the default configuration file

        Packaged: config/config-example.json next to app.exe
        Development: config/config-example.json in the project root
        """
        return get_config_path("config-example.json")

    def get_user_config_path(self) -> str:
        """
        Get the path of the user configuration file

        Packaged: config/config.json next to app.exe (writable)
        Development: in the config folder of the project root
        """
        return get_config_path("config.json")

    def _load_configs_with_priority(self):
        """
        Load the configuration files by priority.
        Priority: user configuration > default configuration > defaults in the code
        """
        # 1. Load the default configuration first (when it exists)
        if os.path.exists(self.default_config_path):
            self.logger.info(f"Loading default configuration: {self.default_config_path}")
            self.load_config_file(self.default_config_path)
        else:
            self.logger.warning(f"Default configuration does not exist: {self.default_config_path}")

        # 2. Then load the user configuration (when it exists), which overrides the defaults
        if os.path.exists(self.user_config_path):
            self.logger.info(f"Loading user configuration: {self.user_config_path}")
            self.load_config_file(self.user_config_path)
            self.config_path = self.user_config_path
        else:
            self.logger.info(f"User configuration does not exist: {self.user_config_path}")
            # When there is no user configuration, create one from the default configuration
            if os.path.exists(self.default_config_path):
                self.logger.info("Creating user configuration from defaults")
                try:
                    # Copy the default configuration to the user configuration location
                    os.makedirs(os.path.dirname(self.user_config_path), exist_ok=True)
                    with open(self.default_config_path, "r", encoding="utf-8") as src:
                        config_data = json.load(src)
                    with open(self.user_config_path, "w", encoding="utf-8") as dst:
                        json.dump(config_data, dst, indent=2, ensure_ascii=False)
                    self.logger.info(f"User configuration created: {self.user_config_path}")
                    self.config_path = self.user_config_path
                except Exception as e:
                    self.logger.error(f"Failed to create user configuration: {e}")
                    self.config_path = self.default_config_path
            else:
                self.config_path = self.user_config_path

        # 3. Sync the user configuration (add new fields, remove old ones)
        self._sync_user_config()

    def _sync_user_config(self):
        """
        Sync the user configuration file
        - a field added to the default configuration → added to the user configuration
        - a field removed from the default configuration → removed from the user configuration
        - values the user changed stay as they are
        """
        if not os.path.exists(self.default_config_path):
            self.logger.warning("Default configuration does not exist; skipping synchronization")
            return

        if not os.path.exists(self.user_config_path):
            self.logger.info("User configuration does not exist; skipping synchronization")
            return

        try:
            # Read the default configuration (as the template)
            with open(self.default_config_path, "r", encoding="utf-8") as f:
                default_data = migrate_legacy_custom_api_params_config(json.load(f))

            # Read the user configuration
            with open(self.user_config_path, "r", encoding="utf-8") as f:
                user_data = migrate_legacy_custom_api_params_config(json.load(f))

            # Sync the configuration (nested dictionaries are handled recursively)
            synced_data = self._sync_dict(default_data, user_data)

            # When something changed, save back to the user configuration
            if synced_data != user_data:
                self.logger.info("Configuration structure changed; synchronizing user configuration")
                with open(self.user_config_path, "w", encoding="utf-8") as f:
                    json.dump(synced_data, f, indent=2, ensure_ascii=False)
                self.logger.info("User configuration synchronized")

        except Exception as e:
            self.logger.error(f"Failed to synchronize user configuration: {e}")

    def _sync_dict(self, template: dict, user: dict) -> dict:
        """
        Sync a dictionary recursively
        - keys that exist in the template are kept
        - keys that do not exist in the template are removed
        - values the user set are kept
        """
        result = {}

        for key in template.keys():
            if key in user:
                # The user configuration has this key
                if isinstance(template[key], dict) and isinstance(user[key], dict):
                    # Handle nested dictionaries recursively
                    result[key] = self._sync_dict(template[key], user[key])
                else:
                    # Use the user's value
                    result[key] = user[key]
            else:
                # The user configuration does not have this key: use the value of the template
                result[key] = template[key]

        return result

    def load_default_config(self) -> bool:
        """Load the default configuration"""
        default_path = self.get_default_config_path()
        return self.load_config_file(default_path)
