"""
资源管理服务 (ResourceManagementService)

管理用户资源的上传、存储、检索和删除。
支持提示词和字体文件的管理。
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from manga_translator.server.models.resource_models import FontResource, PromptResource
from manga_translator.server.repositories.resource_repository import ResourceRepository
from manga_translator.server_paths import USER_RESOURCES_RELATIVE_DIR

logger = logging.getLogger(__name__)


class ResourceManagementService:
    """资源管理服务"""
    
    # Supported file formats
    PROMPT_FORMATS = {'.txt', '.json'}
    FONT_FORMATS = {'.ttf', '.otf', '.ttc'}
    
    def __init__(
        self,
        prompts_repo: ResourceRepository,
        fonts_repo: ResourceRepository,
        base_path: str = USER_RESOURCES_RELATIVE_DIR
    ):
        """
        初始化资源管理服务
        
        Args:
            prompts_repo: 提示词资源仓库
            fonts_repo: 字体资源仓库
            base_path: 资源存储基础路径
        """
        self.prompts_repo = prompts_repo
        self.fonts_repo = fonts_repo
        self.base_path = Path(base_path)
        
        # Make sure the folders exist
        self.prompts_path = self.base_path / "prompts"
        self.fonts_path = self.base_path / "fonts"
        self.prompts_path.mkdir(parents=True, exist_ok=True)
        self.fonts_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ResourceManagementService initialized with base path: {base_path}")
    
    async def upload_prompt(self, user_id: str, file) -> PromptResource:
        """
        上传提示词文件
        
        Args:
            user_id: 用户ID
            file: 上传的文件对象 (FastAPI UploadFile)
        
        Returns:
            PromptResource: 创建的提示词资源
        
        Raises:
            ValueError: 如果文件格式不支持或文件无效
        """
        # Validate the file
        if not file or not file.filename:
            raise ValueError("无效的文件")
        
        # Validate the file format
        file_format = self._get_file_extension(file.filename)
        if not self.validate_file_format(file.filename, 'prompt'):
            raise ValueError(
                f"不支持的提示词文件格式: {file_format}. "
                f"支持的格式: {', '.join(self.PROMPT_FORMATS)}"
            )
        
        # Create the user folder
        user_dir = self.prompts_path / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Build a safe file name (to prevent path traversal attacks)
        safe_filename = self._sanitize_filename(file.filename)
        file_path = user_dir / safe_filename
        
        # When the file already exists, add a number suffix
        file_path = self._get_unique_filepath(file_path)
        
        try:
            # Save the file
            content = await file.read()
            with open(file_path, 'wb') as f:
                f.write(content)
            
            file_size = file_path.stat().st_size
            
            # Create the resource record
            resource = PromptResource.create(
                user_id=user_id,
                filename=file_path.name,
                file_path=str(file_path.relative_to(self.base_path)),
                file_size=file_size,
                file_format=file_format
            )
            
            # Save to the index
            self.prompts_repo.add_resource(resource)
            
            logger.info(f"Uploaded prompt for user {user_id}: {file_path.name}")
            return resource
            
        except Exception as e:
            # When saving fails, remove the file
            if file_path.exists():
                file_path.unlink()
            logger.error(f"Failed to upload prompt: {e}")
            raise ValueError(f"上传提示词失败: {str(e)}")
    
    async def upload_font(self, user_id: str, file) -> FontResource:
        """
        上传字体文件
        
        Args:
            user_id: 用户ID
            file: 上传的文件对象 (FastAPI UploadFile)
        
        Returns:
            FontResource: 创建的字体资源
        
        Raises:
            ValueError: 如果文件格式不支持或文件无效
        """
        # Validate the file
        if not file or not file.filename:
            raise ValueError("无效的文件")
        
        # Validate the file format
        file_format = self._get_file_extension(file.filename)
        if not self.validate_file_format(file.filename, 'font'):
            raise ValueError(
                f"不支持的字体文件格式: {file_format}. "
                f"支持的格式: {', '.join(self.FONT_FORMATS)}"
            )
        
        # Create the user folder
        user_dir = self.fonts_path / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Build a safe file name
        safe_filename = self._sanitize_filename(file.filename)
        file_path = user_dir / safe_filename
        
        # When the file already exists, add a number suffix
        file_path = self._get_unique_filepath(file_path)
        
        try:
            # Save the file
            content = await file.read()
            with open(file_path, 'wb') as f:
                f.write(content)
            
            file_size = file_path.stat().st_size
            
            # Try to extract the font family name (optional)
            font_family = self._extract_font_family(file_path)
            
            # Create the resource record
            resource = FontResource.create(
                user_id=user_id,
                filename=file_path.name,
                file_path=str(file_path.relative_to(self.base_path)),
                file_size=file_size,
                file_format=file_format,
                font_family=font_family
            )
            
            # Save to the index
            self.fonts_repo.add_resource(resource)
            
            logger.info(f"Uploaded font for user {user_id}: {file_path.name}")
            return resource
            
        except Exception as e:
            # When saving fails, remove the file
            if file_path.exists():
                file_path.unlink()
            logger.error(f"Failed to upload font: {e}")
            raise ValueError(f"上传字体失败: {str(e)}")
    
    def get_user_prompts(self, user_id: str) -> List[PromptResource]:
        """
        获取用户的所有提示词
        
        Args:
            user_id: 用户ID
        
        Returns:
            List[PromptResource]: 提示词资源列表
        """
        resources_data = self.prompts_repo.get_user_resources(user_id)
        return [PromptResource.from_dict(data) for data in resources_data]
    
    def get_user_fonts(self, user_id: str) -> List[FontResource]:
        """
        获取用户的所有字体
        
        Args:
            user_id: 用户ID
        
        Returns:
            List[FontResource]: 字体资源列表
        """
        resources_data = self.fonts_repo.get_user_resources(user_id)
        return [FontResource.from_dict(data) for data in resources_data]
    
    def delete_prompt(self, resource_id: str, user_id: str) -> bool:
        """
        删除提示词资源
        
        Args:
            resource_id: 资源ID
            user_id: 用户ID（用于验证所有权）
        
        Returns:
            bool: 删除是否成功
        
        Raises:
            ValueError: 如果资源不存在或用户无权删除
        """
        return self._delete_resource(
            resource_id, user_id, self.prompts_repo, "prompt"
        )
    
    def delete_font(self, resource_id: str, user_id: str) -> bool:
        """
        删除字体资源
        
        Args:
            resource_id: 资源ID
            user_id: 用户ID（用于验证所有权）
        
        Returns:
            bool: 删除是否成功
        
        Raises:
            ValueError: 如果资源不存在或用户无权删除
        """
        return self._delete_resource(
            resource_id, user_id, self.fonts_repo, "font"
        )
    
    def _delete_resource(
        self,
        resource_id: str,
        user_id: str,
        repo: ResourceRepository,
        resource_type: str
    ) -> bool:
        """
        删除资源的通用方法
        
        Args:
            resource_id: 资源ID
            user_id: 用户ID
            repo: 资源仓库
            resource_type: 资源类型（用于日志）
        
        Returns:
            bool: 删除是否成功
        
        Raises:
            ValueError: 如果资源不存在或用户无权删除
        """
        # Get the resource
        resource_data = repo.get_resource_by_id(resource_id)
        if not resource_data:
            raise ValueError(f"资源不存在: {resource_id}")
        
        # Verify ownership
        if resource_data['user_id'] != user_id:
            raise ValueError("无权删除此资源")
        
        # Delete the file
        file_path = self.base_path / resource_data['file_path']
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted {resource_type} file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete {resource_type} file: {e}")
            # Carry on and delete the index record even when deleting the file failed
        
        # Remove from the index
        success = repo.delete_resource(resource_id)
        if success:
            logger.info(f"Deleted {resource_type} resource: {resource_id}")
        
        return success
    
    def validate_file_format(self, filename: str, resource_type: str) -> bool:
        """
        验证文件格式
        
        Args:
            filename: 文件名
            resource_type: 资源类型 ('prompt' 或 'font')
        
        Returns:
            bool: 文件格式是否有效
        """
        ext = self._get_file_extension(filename)
        
        if resource_type == 'prompt':
            return ext in self.PROMPT_FORMATS
        elif resource_type == 'font':
            return ext in self.FONT_FORMATS
        else:
            return False
    
    def _get_file_extension(self, filename: str) -> str:
        """
        获取文件扩展名（小写，包含点）
        
        Args:
            filename: 文件名
        
        Returns:
            str: 文件扩展名
        """
        return Path(filename).suffix.lower()
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        清理文件名，防止路径遍历攻击
        
        Args:
            filename: 原始文件名
        
        Returns:
            str: 安全的文件名
        """
        # Keep only the file name part, dropping the path
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        dangerous_chars = ['..', '/', '\\', '\0']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        return filename
    
    def _get_unique_filepath(self, file_path: Path) -> Path:
        """
        获取唯一的文件路径（如果文件已存在，添加数字后缀）
        
        Args:
            file_path: 原始文件路径
        
        Returns:
            Path: 唯一的文件路径
        """
        if not file_path.exists():
            return file_path
        
        # Split the file name and the extension
        stem = file_path.stem
        suffix = file_path.suffix
        parent = file_path.parent
        
        # Add a number suffix
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1
    
    def _extract_font_family(self, file_path: Path) -> Optional[str]:
        """
        尝试从字体文件中提取字体族名称
        
        Args:
            file_path: 字体文件路径
        
        Returns:
            Optional[str]: 字体族名称，如果提取失败返回 None
        """
        try:
            # The fontTools library could be used here to extract the font information
            # To keep it simple, None is returned for now
            # TODO: implement extraction of the font family name
            return None
        except Exception as e:
            logger.debug(f"Failed to extract font family: {e}")
            return None
    
    def get_resource_stats(self, user_id: str) -> dict:
        """
        获取用户的资源统计信息
        
        Args:
            user_id: 用户ID
        
        Returns:
            dict: 统计信息
        """
        prompts = self.get_user_prompts(user_id)
        fonts = self.get_user_fonts(user_id)
        
        prompt_size = sum(p.file_size for p in prompts)
        font_size = sum(f.file_size for f in fonts)
        
        return {
            'prompt_count': len(prompts),
            'font_count': len(fonts),
            'total_prompt_size': prompt_size,
            'total_font_size': font_size,
            'total_size': prompt_size + font_size
        }
