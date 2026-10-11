"""
File list model - one place for the original-image entries of the editor
"""

import os
from dataclasses import dataclass
from enum import Enum

from manga_translator.image_formats import SUPPORTED_IMAGE_EXTENSIONS

from services.file_list_data_service import canonical_path_key


class FileType(Enum):
    """File type enumeration"""

    SOURCE = "source"  # Original image (with JSON)
    UNTRANSLATED = "untranslated"  # Untranslated original image (no JSON yet)


@dataclass
class FileItem:
    """Data class of a file item"""

    path: str  # File path
    file_type: FileType  # File type
    json_path: str | None = None  # JSON path (for an original image)


class FileListModel:
    """In-memory image metadata projected from a background catalog snapshot."""

    def __init__(self):
        self.files: list[FileItem] = []
        self._path_index: dict[str, FileItem] = {}

    @staticmethod
    def is_supported_image_file(file_path: str) -> bool:
        """Check whether it is an image file the editor supports."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in SUPPORTED_IMAGE_EXTENSIONS

    def clear(self):
        """Clear the file list"""
        self.files.clear()
        self._path_index.clear()

    def remove_file(self, file_path: str) -> bool:
        """
        Remove a file

        Args:
            file_path: the file path

        Returns:
            Whether it was removed
        """
        path_key = canonical_path_key(file_path)
        item = self._path_index.pop(path_key, None)
        if item is None:
            return False
        self.files.remove(item)
        return True

    def get_file_item(self, file_path: str) -> FileItem | None:
        """Return the item with the same canonical path identity."""
        return self._path_index.get(canonical_path_key(file_path))

    def replace_from_snapshot(self, snapshot) -> None:
        """Replace the editor list in one go with a background snapshot, without reading metadata on the GUI thread."""
        self.clear()
        for file_path in snapshot.editor_files:
            normalized = os.path.abspath(os.path.normpath(file_path))
            json_path = snapshot.json_by_file.get(normalized)
            item = FileItem(
                path=normalized,
                file_type=FileType.SOURCE if json_path else FileType.UNTRANSLATED,
                json_path=json_path,
            )
            self.files.append(item)
            self._path_index[canonical_path_key(normalized)] = item
