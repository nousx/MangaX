"""
文件服务层
处理文件和文件夹的选择、验证、拖拽等操作
"""
import base64
import json
import logging
import mimetypes
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

# Add the project root to the path, so path_manager can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from manga_translator.image_formats import SUPPORTED_IMAGE_EXTENSIONS
from manga_translator.utils import open_pil_image
from manga_translator.utils.path_manager import find_json_path


class FileService:
    """文件操作服务"""
    
    def __init__(self):
        from services import get_config_service
        self.logger = logging.getLogger(__name__)
        self.config_service = get_config_service()
        # Supported image formats
        self.supported_image_extensions = set(SUPPORTED_IMAGE_EXTENSIONS)
        # Supported archive and document formats
        self.supported_archive_extensions = {
            '.pdf', '.epub', '.cbz', '.cbr', '.zip'
        }
        # Supported configuration file formats
        self.supported_config_extensions = {
            '.json', '.yaml', '.yml', '.toml'
        }

    def load_translation_json(self, image_path: str, image: Image.Image = None) -> Tuple[List[dict], Optional[np.ndarray], Optional[Tuple[int, int]], Dict[str, Any]]:
        """
        根据给定的图片路径，加载关联的 _translations.json 文件。
        优先从新目录结构加载，支持向后兼容。
        返回 regions, raw_mask, original_size, overlays。
        overlays 为 {'paint': RGBA数组|None, 'stamp': RGBA数组|None,
                    'paste_overlays': [贴片字典...]}（base64 PNG 解码，未对齐尺寸）。
        """
        # Find the JSON file with path_manager (the new location is preferred)
        json_path = find_json_path(image_path)
        regions = []
        raw_mask = None
        original_size = None
        overlays: Dict[str, Any] = {
            'paint': None,
            'stamp': None,
            'paste_overlays': [],
        }

        if not json_path:
            self.logger.warning(f"JSON file not found for {os.path.basename(image_path)}")
            return regions, raw_mask, original_size, overlays

        self.logger.debug(f"Loading JSON from: {json_path}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_key = os.path.abspath(image_path)
            
            if image_key not in data:
                if data:
                    first_key = next(iter(data))
                    self.logger.warning(f"Exact image path '{image_key}' not found in JSON. Using first available key '{first_key}'.")
                    image_data = data[first_key]
                else:
                    image_data = {}
            else:
                image_data = data[image_key]

            regions = image_data.get('regions', [])

            config = self.config_service.get_config()
            default_target_lang = config.translator.target_lang if config else None

            if default_target_lang:
                for region in regions:
                    if not region.get('target_lang'):
                        region['target_lang'] = default_target_lang

            # For old JSON files: when translation_raw is missing it is filled from translation,
            # so the editor's "translation before replacement" box always has a value to show
            for region in regions:
                if 'translation_raw' not in region:
                    region['translation_raw'] = region.get('translation', '')

            mask_data = image_data.get('mask_raw')
            if isinstance(mask_data, str):
                try:
                    img_bytes = base64.b64decode(mask_data)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    raw_mask = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                except Exception as e:
                    self.logger.error(f"Failed to decode base64 mask in {os.path.basename(json_path)}: {e}")
                    raw_mask = None
            elif isinstance(mask_data, list):
                raw_mask = np.array(mask_data, dtype=np.uint8)
            
            original_size = (image_data.get('original_width'), image_data.get('original_height'))

            # Paint layer / stamp layer (base64 PNG, RGBA)
            for overlay_name, json_key in (('paint', 'paint_overlay'), ('stamp', 'stamp_overlay')):
                overlay_b64 = image_data.get(json_key)
                if not isinstance(overlay_b64, str) or not overlay_b64:
                    continue
                try:
                    overlay_bytes = np.frombuffer(base64.b64decode(overlay_b64), dtype=np.uint8)
                    overlay_bgra = cv2.imdecode(overlay_bytes, cv2.IMREAD_UNCHANGED)
                    if overlay_bgra is not None and overlay_bgra.ndim == 3 and overlay_bgra.shape[2] == 4:
                        overlays[overlay_name] = cv2.cvtColor(overlay_bgra, cv2.COLOR_BGRA2RGBA)
                except Exception as e:
                    self.logger.error(f"Failed to decode base64 {json_key} in {os.path.basename(json_path)}: {e}")

            self.logger.debug(f"Loaded {len(regions)} regions from {os.path.basename(json_path)}")

            # List of paste overlays (image patches laid on top): normalised item by item; bad data is skipped with a warning
            paste_raw = image_data.get('paste_overlays')
            if isinstance(paste_raw, list) and paste_raw:
                try:
                    from editor.paste_overlay_state import parse_page_paste_overlays

                    overlays['paste_overlays'] = parse_page_paste_overlays(image_data)
                except Exception as e:
                    self.logger.error(f"Failed to parse paste_overlays in {os.path.basename(json_path)}: {e}")

        except Exception as e:
            import traceback
            self.logger.error(f"Failed to load or parse JSON file {json_path}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return [], None, None, {'paint': None, 'stamp': None, 'paste_overlays': []}

        return regions, raw_mask, original_size, overlays
        
    def validate_image_file(self, file_path: str) -> bool:
        """验证是否为有效的图片文件或压缩包文件"""
        try:
            if not os.path.exists(file_path):
                return False
                
            # Check the file extension
            _, ext = os.path.splitext(file_path)
            ext_lower = ext.lower()
            
            # Archive formats are supported
            if ext_lower in self.supported_archive_extensions:
                return os.access(file_path, os.R_OK)
            
            if ext_lower not in self.supported_image_extensions:
                return False
                
            # Check the MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and not mime_type.startswith('image/'):
                return False
                
            # Check whether the file is readable
            if not os.access(file_path, os.R_OK):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate image file {file_path}: {e}")
            return False
    
    def is_archive_file(self, file_path: str) -> bool:
        """检查文件是否是压缩包/文档格式"""
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.supported_archive_extensions
    
    def validate_config_file(self, file_path: str) -> bool:
        """验证是否为有效的配置文件"""
        try:
            if not os.path.exists(file_path):
                return False
                
            _, ext = os.path.splitext(file_path)
            return ext.lower() in self.supported_config_extensions
            
        except Exception as e:
            self.logger.error(f"Failed to validate configuration file {file_path}: {e}")
            return False
    
    def _natural_sort_key(self, path: str):
        """
        生成自然排序的键，支持数字排序
        例如: file1.jpg, file2.jpg, file10.jpg 会按 1, 2, 10 排序
        而不是按字符串 1, 10, 2 排序
        
        对于包含路径的文件，会对整个路径进行自然排序，确保子文件夹也能正确排序
        例如: 第1话/001.jpg, 第2话/001.jpg, 第10话/001.jpg 会按 1, 2, 10 排序
        """
        import re
        
        # Normalise the path separators
        normalized_path = path.replace('\\', '/')
        
        # Split the whole path into text and number parts
        # A tuple keeps the types safe: (is it a number, sort value)
        # Numbers sort as integers and text as strings; the first element tells the types apart, so no comparison crosses types
        parts = []
        for part in re.split(r'(\d+)', normalized_path):
            if part.isdigit():
                # Number part: (False, integer value) - False sorts before True
                parts.append((False, int(part)))
            elif part:  # Ignore empty strings
                # Text part: (True, lower-case text) - True sorts after False
                parts.append((True, part.lower()))
        
        return parts
    
    def get_supported_files_from_folder(
        self, folder_path: str, recursive: bool = True
    ) -> tuple[List[str], List[str]]:
        """一次遍历返回图片与压缩包，忽略 manga_translator_work。"""
        image_files: List[str] = []
        archive_files: List[str] = []
        try:
            if not os.path.isdir(folder_path):
                return image_files, archive_files

            entries = os.walk(folder_path) if recursive else [(folder_path, [], os.listdir(folder_path))]
            for root, dirs, files in entries:
                if 'manga_translator_work' in dirs:
                    dirs.remove('manga_translator_work')
                dirs.sort(key=self._natural_sort_key)
                current_images: List[str] = []
                current_archives: List[str] = []
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    if ext in self.supported_image_extensions and os.path.isfile(file_path):
                        current_images.append(file_path)
                    elif ext in self.supported_archive_extensions and os.path.isfile(file_path):
                        current_archives.append(file_path)
                current_images.sort(key=self._natural_sort_key)
                current_archives.sort(key=self._natural_sort_key)
                image_files.extend(current_images)
                archive_files.extend(current_archives)
        except Exception as e:
            self.logger.error(f"Failed to list supported files in folder {folder_path}: {e}")
        return image_files, archive_files

    def get_image_files_from_folder(self, folder_path: str, recursive: bool = True) -> List[str]:
        return self.get_supported_files_from_folder(folder_path, recursive)[0]

    def get_archive_files_from_folder(self, folder_path: str, recursive: bool = True) -> List[str]:
        return self.get_supported_files_from_folder(folder_path, recursive)[1]
    
    def filter_valid_image_files(self, file_paths: List[str]) -> List[str]:
        """过滤出有效的图片文件"""
        valid_files = []
        
        for file_path in file_paths:
            if self.validate_image_file(file_path):
                valid_files.append(file_path)
            else:
                self.logger.warning(f"Skipping invalid file: {file_path}")
                
        return valid_files
    
    def process_dropped_files(self, dropped_data: str) -> Tuple[List[str], List[str]]:
        """处理拖拽的文件数据
        
        Returns:
            Tuple[List[str], List[str]]: (有效的图片文件列表, 错误信息列表)
        """
        image_files = []
        errors = []
        
        try:
            # Parse the dropped data
            file_paths = self._parse_drop_data(dropped_data)
            
            for file_path in file_paths:
                if os.path.isfile(file_path):
                    if self.validate_image_file(file_path):
                        image_files.append(file_path)
                    else:
                        errors.append(f"不支持的图片格式: {os.path.basename(file_path)}")
                        
                elif os.path.isdir(file_path):
                    # Handle folders
                    folder_images = self.get_image_files_from_folder(file_path)
                    if folder_images:
                        image_files.extend(folder_images)
                    else:
                        errors.append(f"文件夹中没有找到图片: {os.path.basename(file_path)}")
                else:
                    errors.append(f"文件不存在: {os.path.basename(file_path)}")
                    
        except Exception as e:
            self.logger.error(f"Failed to process dropped files: {e}")
            errors.append(f"处理拖拽文件时出错: {str(e)}")
            
        return image_files, errors
    
    def _parse_drop_data(self, dropped_data: str) -> List[str]:
        """解析拖拽数据，提取文件路径"""
        file_paths = []
        
        # Handle the line endings of different operating systems
        lines = dropped_data.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove a possible URI prefix
                if line.startswith('file:///'):
                    line = line[8:]  # Remove 'file:///'
                elif line.startswith('file://'):
                    line = line[7:]  # Remove 'file://'
                
                # URL-decode
                try:
                    import urllib.parse
                    line = urllib.parse.unquote(line)
                except Exception:
                    pass
                
                if os.path.exists(line):
                    file_paths.append(os.path.abspath(line))
                    
        return file_paths
    
    def get_file_info(self, file_path: str) -> dict:
        """获取文件信息"""
        try:
            if not os.path.exists(file_path):
                return {'error': '文件不存在'}
                
            stat = os.stat(file_path)
            file_info = {
                'name': os.path.basename(file_path),
                'path': os.path.abspath(file_path),
                'size': stat.st_size,
                'size_human': self._format_file_size(stat.st_size),
                'modified': stat.st_mtime,
                'is_readable': os.access(file_path, os.R_OK),
                'is_writable': os.access(file_path, os.W_OK)
            }
            
            if self.validate_image_file(file_path):
                file_info['type'] = 'image'
                # Get the image size
                try:
                    with open_pil_image(file_path, eager=False) as img:
                        file_info['width'] = img.width
                        file_info['height'] = img.height
                        file_info['format'] = img.format
                except Exception as e:
                    self.logger.warning(f"Failed to get image information for {file_path}: {e}")
                    
            return file_info
            
        except Exception as e:
            self.logger.error(f"Failed to get file information for {file_path}: {e}")
            return {'error': str(e)}
    
    def _format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
    
    def create_backup(self, file_path: str, backup_dir: Optional[str] = None) -> str:
        """创建文件备份"""
        try:
            if backup_dir is None:
                backup_dir = os.path.join(os.path.dirname(file_path), 'backups')
                
            os.makedirs(backup_dir, exist_ok=True)
            
            # Build the backup file name
            import time
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            name, ext = os.path.splitext(os.path.basename(file_path))
            backup_name = f"{name}_{timestamp}{ext}"
            backup_path = os.path.join(backup_dir, backup_name)
            
            # Copy the file
            shutil.copy2(file_path, backup_path)
            self.logger.info(f"Creating backup: {backup_path}")
            
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {e}")
            raise
    
    def cleanup_temp_files(self, temp_dir: str, max_age_hours: int = 24) -> None:
        """清理临时文件"""
        try:
            if not os.path.exists(temp_dir):
                return
                
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if current_time - os.path.getmtime(file_path) > max_age_seconds:
                            os.remove(file_path)
                            self.logger.info(f"Deleting expired temporary file: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete temporary file {file_path}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to clean up temporary files: {e}")
    
    def get_supported_image_extensions(self) -> Set[str]:
        """获取支持的图片文件扩展名"""
        return self.supported_image_extensions.copy()
    
    def get_supported_config_extensions(self) -> Set[str]:
        """获取支持的配置文件扩展名"""
        return self.supported_config_extensions.copy()
    
    def normalize_path(self, path: str) -> str:
        """标准化路径"""
        return os.path.normpath(os.path.abspath(path))
