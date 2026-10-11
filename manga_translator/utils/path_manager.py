#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path management module.
One place for building and finding file paths; supports the new folder structure and stays backward compatible
"""

import os
from typing import Optional, Tuple

from manga_translator.image_formats import SUPPORTED_IMAGE_EXTENSIONS
from manga_translator.utils.translation_template import (
    get_translation_output_format,
    normalize_translation_output_format,
)

# Constants for the work folder names
WORK_DIR_NAME = "manga_translator_work"
JSON_SUBDIR = "json"
TRANSLATIONS_SUBDIR = "translations"
ORIGINALS_SUBDIR = "originals"
YOLO_LABELS_SUBDIR = "yolo_labels"
INPAINTED_SUBDIR = "inpainted"
PAINT_OVERLAY_SUBDIR = "paint_overlay"  # Folder of the colour brush doodle layers
TRANSLATED_IMAGES_SUBDIR = "translated_images"  # Folder of translated images (used by replace-translation mode)
EDITOR_BASE_SUBDIR = "editor_base"
WORK_DIR_RESERVED_NAMES = {
    JSON_SUBDIR,
    TRANSLATIONS_SUBDIR,
    ORIGINALS_SUBDIR,
    YOLO_LABELS_SUBDIR,
    INPAINTED_SUBDIR,
    PAINT_OVERLAY_SUBDIR,
    TRANSLATED_IMAGES_SUBDIR,
    EDITOR_BASE_SUBDIR,
}


def normalize_image_path(image_path: str) -> str:
    """Normalise an image path."""
    return os.path.normpath(os.path.abspath(image_path))


def is_work_image_path(image_path: str) -> bool:
    """
    Whether the path is a colorized or upscaled base image kept for the editor.
    """
    norm_path = normalize_image_path(image_path)
    parent_dir = os.path.dirname(norm_path)
    grandparent_dir = os.path.dirname(parent_dir)

    # New structure: manga_translator_work/editor_base/xxx.png
    if (
        os.path.basename(parent_dir) == EDITOR_BASE_SUBDIR and
        os.path.basename(grandparent_dir) == WORK_DIR_NAME
    ):
        return True

    # Still accepts temporary base images that were placed in the root folder earlier
    if os.path.basename(parent_dir) == WORK_DIR_NAME:
        return os.path.basename(norm_path) not in WORK_DIR_RESERVED_NAMES

    return False


def resolve_original_image_path(image_path: str) -> str:
    """
    Turn the path of a unified base image in the work folder back into the path of the original image; other paths are returned as they are.
    """
    norm_path = normalize_image_path(image_path)
    if not is_work_image_path(norm_path):
        return norm_path

    parent_dir = os.path.dirname(norm_path)
    grandparent_dir = os.path.dirname(parent_dir)

    if (
        os.path.basename(parent_dir) == EDITOR_BASE_SUBDIR and
        os.path.basename(grandparent_dir) == WORK_DIR_NAME
    ):
        source_dir = os.path.dirname(grandparent_dir)
        return os.path.join(source_dir, os.path.basename(norm_path))

    source_dir = os.path.dirname(parent_dir)
    return os.path.join(source_dir, os.path.basename(norm_path))


def get_work_dir(image_path: str) -> str:
    """
    Get the work folder path that belongs to an image

    Args:
        image_path: path of the original image

    Returns:
        The absolute path of the work folder
    """
    image_dir = os.path.dirname(resolve_original_image_path(image_path))
    return os.path.join(image_dir, WORK_DIR_NAME)


def get_work_image_path(image_path: str, create_dir: bool = True) -> str:
    """
    Get the path of the colorized or upscaled base image kept for the editor.
    """
    if is_work_image_path(image_path):
        work_image_path = normalize_image_path(image_path)
        if create_dir:
            os.makedirs(os.path.dirname(work_image_path), exist_ok=True)
        return work_image_path

    original_path = resolve_original_image_path(image_path)
    work_dir = get_work_dir(original_path)
    editor_base_dir = os.path.join(work_dir, EDITOR_BASE_SUBDIR)
    if create_dir:
        os.makedirs(editor_base_dir, exist_ok=True)
    return os.path.join(editor_base_dir, os.path.basename(original_path))


def find_work_image_path(image_path: str) -> Optional[str]:
    """Find the colorized or upscaled base image kept for the editor."""
    work_image_path = get_work_image_path(image_path, create_dir=False)
    if os.path.exists(work_image_path):
        return work_image_path

    # Still accepts base images that may have been placed in the root folder earlier
    original_path = resolve_original_image_path(image_path)
    legacy_root_work_image = os.path.join(get_work_dir(original_path), os.path.basename(original_path))
    if os.path.exists(legacy_root_work_image):
        return legacy_root_work_image

    return None


def get_legacy_inpainted_path(image_path: str, create_dir: bool = True) -> str:
    """
    Get the path of the old-style inpainted image (manga_translator_work/inpainted/*_inpainted.ext).
    """
    original_path = resolve_original_image_path(image_path)
    work_dir = get_work_dir(original_path)
    inpainted_dir = os.path.join(work_dir, INPAINTED_SUBDIR)

    if create_dir:
        os.makedirs(inpainted_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(original_path))[0]
    ext = os.path.splitext(original_path)[1]
    return os.path.join(inpainted_dir, f"{base_name}_inpainted{ext}")


def get_json_path(image_path: str, create_dir: bool = True) -> str:
    """
    Get the path of the JSON file

    Args:
        image_path: path of the original image
        create_dir: whether the folder is created automatically

    Returns:
        The absolute path of the JSON file
    """
    work_dir = get_work_dir(image_path)
    json_dir = os.path.join(work_dir, JSON_SUBDIR)
    
    if create_dir:
        os.makedirs(json_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(json_dir, f"{base_name}_translations.json")


def _resolve_text_output_format(output_format: Optional[str] = None) -> str:
    if output_format is None:
        return get_translation_output_format()
    return normalize_translation_output_format(output_format)


def get_original_txt_path(
    image_path: str,
    create_dir: bool = True,
    output_format: Optional[str] = None,
) -> str:
    """
    Get the path of the original-text export file.

    Args:
        image_path: path of the original image
        create_dir: whether the folder is created automatically

    Returns:
        The absolute path of the original-text export file
    """
    work_dir = get_work_dir(image_path)
    originals_dir = os.path.join(work_dir, ORIGINALS_SUBDIR)
    
    if create_dir:
        os.makedirs(originals_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    extension = _resolve_text_output_format(output_format)
    return os.path.join(originals_dir, f"{base_name}_original.{extension}")


def get_translated_txt_path(
    image_path: str,
    create_dir: bool = True,
    output_format: Optional[str] = None,
) -> str:
    """
    Get the path of the translation export file.

    Args:
        image_path: path of the original image
        create_dir: whether the folder is created automatically

    Returns:
        The absolute path of the translation export file
    """
    work_dir = get_work_dir(image_path)
    translations_dir = os.path.join(work_dir, TRANSLATIONS_SUBDIR)
    
    if create_dir:
        os.makedirs(translations_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    extension = _resolve_text_output_format(output_format)
    return os.path.join(translations_dir, f"{base_name}_translated.{extension}")


def get_yolo_labels_dir(image_path: str, create_dir: bool = True) -> str:
    """
    Get the path of the YOLO label folder.

    Args:
        image_path: path of the original image
        create_dir: whether the folder is created automatically

    Returns:
        The absolute path of the YOLO label folder
    """
    work_dir = get_work_dir(image_path)
    yolo_labels_dir = os.path.join(work_dir, YOLO_LABELS_SUBDIR)

    if create_dir:
        os.makedirs(yolo_labels_dir, exist_ok=True)

    return yolo_labels_dir


def get_yolo_label_path(image_path: str, create_dir: bool = True) -> str:
    """
    Get the path of the YOLO label file that belongs to an image.

    Args:
        image_path: path of the original image
        create_dir: whether the folder is created automatically

    Returns:
        The absolute path of the YOLO label file
    """
    yolo_labels_dir = get_yolo_labels_dir(image_path, create_dir=create_dir)
    base_name = os.path.splitext(os.path.basename(resolve_original_image_path(image_path)))[0]
    return os.path.join(yolo_labels_dir, f"{base_name}.txt")


def find_yolo_label_path(image_path: str) -> Optional[str]:
    """
    Find the YOLO label file that belongs to an image.

    Args:
        image_path: path of the original image

    Returns:
        The path of the YOLO label file that was found, or None when it does not exist
    """
    original_path = resolve_original_image_path(image_path)
    yolo_label_path = get_yolo_label_path(original_path, create_dir=False)
    if os.path.exists(yolo_label_path):
        return yolo_label_path

    legacy_yolo_label_path = os.path.splitext(original_path)[0] + ".txt"
    if os.path.exists(legacy_yolo_label_path):
        return legacy_yolo_label_path

    return None


def get_inpainted_path(image_path: str, create_dir: bool = True) -> str:
    """
    Get the path of the inpainted image

    Args:
        image_path: path of the original image
        create_dir: whether the folder is created automatically

    Returns:
        The absolute path of the inpainted image
    """
    return get_legacy_inpainted_path(image_path, create_dir=create_dir)


def get_translated_images_dir(image_path: str, create_dir: bool = True) -> str:
    """
    Get the path of the folder of translated images

    Args:
        image_path: path of the original image
        create_dir: whether the folder is created automatically

    Returns:
        The absolute path of the folder of translated images
    """
    work_dir = get_work_dir(image_path)
    translated_dir = os.path.join(work_dir, TRANSLATED_IMAGES_SUBDIR)
    
    if create_dir:
        os.makedirs(translated_dir, exist_ok=True)
    
    return translated_dir


def find_translated_source_json(target_image_path: str, translated_dir: str) -> Optional[str]:
    """
    In the folder of translated images, find the translation data JSON of the image with the same name as the target

    For replace-translation mode: by the file name of the raw image, find the JSON of the image with the same name in the translated folder

    Args:
        target_image_path: path of the target (raw) image
        translated_dir: folder that holds the translated images

    Returns:
        The path of the JSON file that was found, or None when it does not exist
    """
    if not translated_dir or not os.path.isdir(translated_dir):
        return None
    
    # Base file name of the target image (without extension)
    target_basename = os.path.splitext(os.path.basename(target_image_path))[0]
    
    # Look for an image with the same name in the translated folder
    # Try manga_translator_work/json/<file name>_translations.json
    translated_work_dir = os.path.join(translated_dir, WORK_DIR_NAME, JSON_SUBDIR)
    if os.path.isdir(translated_work_dir):
        json_path = os.path.join(translated_work_dir, f"{target_basename}_translations.json")
        if os.path.exists(json_path):
            return json_path
    
    # Backward compatibility: look for <translated folder>/<file name>_translations.json
    old_json_path = os.path.join(translated_dir, f"{target_basename}_translations.json")
    if os.path.exists(old_json_path):
        return old_json_path
    
    # Try any supported image extension
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        # Build the possible path of the translated image
        possible_translated_image = os.path.join(translated_dir, f"{target_basename}{ext}")
        if os.path.exists(possible_translated_image):
            # Find the JSON of that image
            json_path = find_json_path(possible_translated_image)
            if json_path:
                return json_path
    
    return None


def find_json_path(image_path: str) -> Optional[str]:
    """
    Find the JSON file, looking in the new location first, with backward compatibility

    Args:
        image_path: path of the original image

    Returns:
        The path of the JSON file that was found, or None when it does not exist
    """
    original_path = resolve_original_image_path(image_path)

    # 1. Look in the new location first
    new_json_path = get_json_path(original_path, create_dir=False)
    if os.path.exists(new_json_path):
        return new_json_path
    
    # 2. Backward compatibility: look in the old location (next to the image)
    old_json_path = os.path.splitext(original_path)[0] + '_translations.json'
    if os.path.exists(old_json_path):
        return old_json_path
    
    return None


def find_inpainted_path(image_path: str) -> Optional[str]:
    """
    Find the inpainted image file

    Args:
        image_path: path of the original image

    Returns:
        The path of the inpainted image that was found, or None when it does not exist
    """
    inpainted_path = get_inpainted_path(image_path, create_dir=False)
    if os.path.exists(inpainted_path):
        return inpainted_path
    
    return None


def get_paint_overlay_path(image_path: str, create_dir: bool = True) -> str:
    """
    Get the path where the colour brush doodle layer (paint overlay) is saved.

    It is stored at manga_translator_work/paint_overlay/<basename>_overlay.png.
    PNG is always used, to keep the alpha channel.

    Args:
        image_path: path of the original image
        create_dir: whether the folder is created automatically

    Returns:
        The absolute path of the paint overlay image
    """
    original_path = resolve_original_image_path(image_path)
    work_dir = get_work_dir(original_path)
    overlay_dir = os.path.join(work_dir, PAINT_OVERLAY_SUBDIR)

    if create_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(original_path))[0]
    return os.path.join(overlay_dir, f"{base_name}_overlay.png")


def find_paint_overlay_path(image_path: str) -> Optional[str]:
    """Find the saved colour brush doodle layer file."""
    overlay_path = get_paint_overlay_path(image_path, create_dir=False)
    if os.path.exists(overlay_path):
        return overlay_path
    return None


def find_txt_files(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the original-text and translation export files of the current template format.

    Args:
        image_path: path of the original image

    Returns:
        (original-text path, translation path); None for one that does not exist
    """
    original_path = get_original_txt_path(image_path, create_dir=False)
    translated_path = get_translated_txt_path(image_path, create_dir=False)

    original_exists = original_path if os.path.exists(original_path) else None
    translated_exists = translated_path if os.path.exists(translated_path) else None
    
    return original_exists, translated_exists


def get_legacy_json_path(image_path: str) -> str:
    """
    Get the path of the old-style JSON file (next to the image).
    For backward compatibility

    Args:
        image_path: path of the original image

    Returns:
        The path of the old-style JSON file
    """
    return os.path.splitext(image_path)[0] + '_translations.json'


def migrate_legacy_files(image_path: str, move_files: bool = False) -> dict:
    """
    Migrate old-style files to the new folder structure

    Args:
        image_path: path of the original image
        move_files: whether files are moved (True) or copied (False)

    Returns:
        A dictionary with the migration result, holding the lists of files that succeeded and failed
    """
    import shutil
    
    result = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    # Check for a JSON file and migrate it
    old_json = get_legacy_json_path(image_path)
    if os.path.exists(old_json):
        new_json = get_json_path(image_path, create_dir=True)
        if not os.path.exists(new_json):
            try:
                if move_files:
                    shutil.move(old_json, new_json)
                else:
                    shutil.copy2(old_json, new_json)
                result['success'].append(('json', old_json, new_json))
            except Exception as e:
                result['failed'].append(('json', old_json, str(e)))
        else:
            result['skipped'].append(('json', old_json, 'target exists'))
    
    # Check for an old TXT file and migrate it
    old_txt = os.path.splitext(image_path)[0] + '_translations.txt'
    if os.path.exists(old_txt):
        new_txt = get_translated_txt_path(image_path, create_dir=True)
        if not os.path.exists(new_txt):
            try:
                if move_files:
                    shutil.move(old_txt, new_txt)
                else:
                    shutil.copy2(old_txt, new_txt)
                result['success'].append(('txt', old_txt, new_txt))
            except Exception as e:
                result['failed'].append(('txt', old_txt, str(e)))
        else:
            result['skipped'].append(('txt', old_txt, 'target exists'))
    
    return result
