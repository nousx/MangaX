"""
Resource management service (ResourceManagementService)

Manages uploading, storing, retrieving and deleting user resources.
Supports prompt files and font files.
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
    """Resource management service"""
    
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
        Initialise the resource management service

        Args:
            prompts_repo: the prompt resource repository
            fonts_repo: the font resource repository
            base_path: base path of the resource storage
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
        Upload a prompt file

        Args:
            user_id: the user ID
            file: the uploaded file object (FastAPI UploadFile)

        Returns:
            PromptResource: the prompt resource that was created

        Raises:
            ValueError: when the file format is not supported or the file is invalid
        """
        # Validate the file
        if not file or not file.filename:
            raise ValueError("Invalid file")
        
        # Validate the file format
        file_format = self._get_file_extension(file.filename)
        if not self.validate_file_format(file.filename, 'prompt'):
            raise ValueError(
                f"Unsupported prompt file format: {file_format}. "
                f"Supported formats: {', '.join(self.PROMPT_FORMATS)}"
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
            raise ValueError(f"Uploading the prompt failed: {str(e)}")
    
    async def upload_font(self, user_id: str, file) -> FontResource:
        """
        Upload a font file

        Args:
            user_id: the user ID
            file: the uploaded file object (FastAPI UploadFile)

        Returns:
            FontResource: the font resource that was created

        Raises:
            ValueError: when the file format is not supported or the file is invalid
        """
        # Validate the file
        if not file or not file.filename:
            raise ValueError("Invalid file")
        
        # Validate the file format
        file_format = self._get_file_extension(file.filename)
        if not self.validate_file_format(file.filename, 'font'):
            raise ValueError(
                f"Unsupported font file format: {file_format}. "
                f"Supported formats: {', '.join(self.FONT_FORMATS)}"
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
            raise ValueError(f"Uploading the font failed: {str(e)}")
    
    def get_user_prompts(self, user_id: str) -> List[PromptResource]:
        """
        Get all prompts of a user

        Args:
            user_id: the user ID

        Returns:
            List[PromptResource]: list of prompt resources
        """
        resources_data = self.prompts_repo.get_user_resources(user_id)
        return [PromptResource.from_dict(data) for data in resources_data]
    
    def get_user_fonts(self, user_id: str) -> List[FontResource]:
        """
        Get all fonts of a user

        Args:
            user_id: the user ID

        Returns:
            List[FontResource]: list of font resources
        """
        resources_data = self.fonts_repo.get_user_resources(user_id)
        return [FontResource.from_dict(data) for data in resources_data]
    
    def delete_prompt(self, resource_id: str, user_id: str) -> bool:
        """
        Delete a prompt resource

        Args:
            resource_id: the resource ID
            user_id: the user ID (to verify ownership)

        Returns:
            bool: whether the deletion succeeded

        Raises:
            ValueError: when the resource does not exist or the user may not delete it
        """
        return self._delete_resource(
            resource_id, user_id, self.prompts_repo, "prompt"
        )
    
    def delete_font(self, resource_id: str, user_id: str) -> bool:
        """
        Delete a font resource

        Args:
            resource_id: the resource ID
            user_id: the user ID (to verify ownership)

        Returns:
            bool: whether the deletion succeeded

        Raises:
            ValueError: when the resource does not exist or the user may not delete it
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
        Common method for deleting a resource

        Args:
            resource_id: the resource ID
            user_id: the user ID
            repo: the resource repository
            resource_type: the resource type (for logging)

        Returns:
            bool: whether the deletion succeeded

        Raises:
            ValueError: when the resource does not exist or the user may not delete it
        """
        # Get the resource
        resource_data = repo.get_resource_by_id(resource_id)
        if not resource_data:
            raise ValueError(f"The resource does not exist: {resource_id}")
        
        # Verify ownership
        if resource_data['user_id'] != user_id:
            raise ValueError("You may not delete this resource")
        
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
        Validate the file format

        Args:
            filename: the file name
            resource_type: the resource type ('prompt' or 'font')

        Returns:
            bool: whether the file format is valid
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
        Get the file extension (lower case, with the dot)

        Args:
            filename: the file name

        Returns:
            str: the file extension
        """
        return Path(filename).suffix.lower()
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Clean a file name, to prevent path traversal attacks

        Args:
            filename: the original file name

        Returns:
            str: the safe file name
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
        Get a unique file path (a numeric suffix is added when the file already exists)

        Args:
            file_path: the original file path

        Returns:
            Path: the unique file path
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
        Try to extract the font family name from a font file

        Args:
            file_path: path of the font file

        Returns:
            Optional[str]: the font family name, or None when extraction fails
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
        Get the resource statistics of a user

        Args:
            user_id: the user ID

        Returns:
            dict: the statistics
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
