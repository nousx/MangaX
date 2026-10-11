"""
预设管理服务
用于管理.env配置预设
"""
import json
import logging
import os
from typing import Dict, List, Optional

from manga_translator.runtime_paths import get_application_dir

DEFAULT_PRESET_NAME = "默认"


class PresetService:
    """预设管理服务"""
    
    def __init__(self, presets_dir: str = None, config_service=None):
        self.logger = logging.getLogger(__name__)
        self.config_service = config_service
        
        # Folder where presets are stored
        if presets_dir is None:
            # Next to app.exe when packaged, in the project root in development.
            self.presets_dir = os.path.join(get_application_dir(), "presets")
        else:
            self.presets_dir = presets_dir
        
        # Make sure the preset folder exists
        os.makedirs(self.presets_dir, exist_ok=True)
        
        # Create the default preset (when it does not exist)
        self._create_default_preset()
        

    def _get_known_preset_env_keys(self) -> List[str]:
        """获取预设应覆盖的全部 env 键。"""
        if self.config_service and hasattr(self.config_service, "get_all_preset_env_vars"):
            try:
                return self.config_service.get_all_preset_env_vars()
            except Exception as e:
                self.logger.warning(f"Failed to get preset environment keys; using fallback list: {e}")

        return [
            "OPENAI_API_KEY",
            "OPENAI_API_BASE",
            "OPENAI_MODEL",
            "GEMINI_API_KEY",
            "GEMINI_API_BASE",
            "GEMINI_MODEL",
            "SAKURA_API_BASE",
            "SAKURA_DICT_PATH",
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

    def _normalize_preset_env_vars(self, env_vars: Optional[Dict[str, str]]) -> Dict[str, str]:
        """补齐所有已知 API env 键，并保留额外自定义 env 键。"""
        source = env_vars or {}
        normalized: Dict[str, str] = {}

        for key in self._get_known_preset_env_keys():
            value = source.get(key, "")
            normalized[key] = "" if value is None else str(value)

        for key, value in source.items():
            if key not in normalized:
                normalized[key] = "" if value is None else str(value)

        return normalized

    def _build_default_preset_env(self) -> Dict[str, str]:
        """构建默认预设内容。"""
        default_env = self._normalize_preset_env_vars({})
        default_env["OPENAI_API_BASE"] = "https://api.openai.com/v1"
        default_env["OPENAI_MODEL"] = "gpt-4o"
        return default_env
    
    def _create_default_preset(self):
        """创建默认预设"""
        default_preset_path = os.path.join(self.presets_dir, f"{DEFAULT_PRESET_NAME}.json")
        if not os.path.exists(default_preset_path):
            default_env = self._build_default_preset_env()
            try:
                with open(default_preset_path, 'w', encoding='utf-8') as f:
                    json.dump(default_env, f, indent=2, ensure_ascii=False)
                self.logger.info("Default preset created")
            except Exception as e:
                self.logger.error(f"Failed to create default preset: {e}")
    
    def get_presets_list(self) -> List[str]:
        """获取所有预设名称列表"""
        try:
            if not os.path.exists(self.presets_dir):
                return []
            
            presets = []
            for filename in os.listdir(self.presets_dir):
                if filename.endswith('.json'):
                    preset_name = filename[:-5]  # Remove the .json suffix
                    presets.append(preset_name)
            
            return sorted(presets)
        except Exception as e:
            self.logger.error(f"Failed to get preset list: {e}")
            return []
    
    def save_preset(self, preset_name: str, env_vars: Dict[str, str]) -> bool:
        """保存预设"""
        try:
            if not preset_name or not preset_name.strip():
                self.logger.error("Preset name cannot be empty")
                return False
            
            # Clean the preset name, removing invalid characters
            preset_name = self._sanitize_filename(preset_name.strip())
            
            preset_path = os.path.join(self.presets_dir, f"{preset_name}.json")
            normalized_env_vars = self._normalize_preset_env_vars(env_vars)
            
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(normalized_env_vars, f, indent=2, ensure_ascii=False)
            
            # Not logged, to avoid flooding the log
            return True
        except Exception as e:
            self.logger.error(f"Failed to save preset: {e}")
            return False
    
    def load_preset(self, preset_name: str) -> Optional[Dict[str, str]]:
        """加载预设"""
        try:
            preset_path = os.path.join(self.presets_dir, f"{preset_name}.json")
            
            if not os.path.exists(preset_path):
                self.logger.error(f"Preset does not exist: {preset_name}")
                return None
            
            with open(preset_path, 'r', encoding='utf-8') as f:
                env_vars = json.load(f)
            
            return self._normalize_preset_env_vars(env_vars)
        except Exception as e:
            self.logger.error(f"Failed to load preset: {e}")
            return None
    
    def delete_preset(self, preset_name: str) -> bool:
        """删除预设"""
        if preset_name == DEFAULT_PRESET_NAME:
            self.logger.warning("The default preset cannot be deleted")
            return False

        try:
            preset_path = os.path.join(self.presets_dir, f"{preset_name}.json")
            
            if not os.path.exists(preset_path):
                self.logger.error(f"Preset does not exist: {preset_name}")
                return False
            
            os.remove(preset_path)
            self.logger.info(f"Preset deleted: {preset_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete preset: {e}")
            return False
    
    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，移除非法字符"""
        # Remove the characters that are invalid on Windows and Unix
        illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in illegal_chars:
            filename = filename.replace(char, '_')
        return filename
