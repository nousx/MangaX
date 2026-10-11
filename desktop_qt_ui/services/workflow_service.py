import glob
import json
import logging
import os
import re
import sys
from typing import List, Tuple

# Add the project root to the path, so path_manager can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from manga_translator.image_formats import (
    IMAGE_FILE_GLOB_PATTERNS,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from manga_translator.runtime_paths import get_application_dir, get_config_path
from manga_translator.utils.path_manager import (
    find_json_path,
    find_txt_files,
    get_original_txt_path,
    get_translated_txt_path,
)
from manga_translator.utils.translation_template import (
    ensure_translation_template_exists,
    get_translation_output_format,
    parse_translation_template_config,
)

logger = logging.getLogger(__name__)


def restore_translation_to_text(json_path: str) -> bool:
    """
    在加载文本+模板模式下，将翻译结果写回到原文字段
    确保模板模式输出翻译而不是原文
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        bool: 是否有修改并成功写回
    """
    try:
        if not os.path.exists(json_path):
            logger.warning(f"JSON file not found: {json_path}")
            return False
            
        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        processed_regions = 0
        
        # Go through the data of every image
        for image_key, image_data in data.items():
            if isinstance(image_data, dict) and 'regions' in image_data:
                regions = image_data['regions']
                
                for region in regions:
                    if isinstance(region, dict):
                        # Get the translation and the original text
                        translation = region.get('translation', '').strip()
                        original_text = region.get('text', '').strip()
                        
                        # Written back only when the translation is not empty and differs from the original
                        if translation and translation != original_text:
                            # Write the translation back to the original text field
                            region['text'] = translation
                            
                            # Update the texts array as well
                            if 'texts' in region and isinstance(region['texts'], list):
                                if len(region['texts']) > 0:
                                    region['texts'][0] = translation
                                else:
                                    region['texts'] = [translation]
                            
                            modified = True
                            processed_regions += 1
                            logger.debug(f"Restored translation to text: '{original_text}' -> '{translation}'")
        
        # When something changed, write the file back
        if modified:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Processed {processed_regions} regions in {os.path.basename(json_path)}")
            
        return modified
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {json_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return False


def batch_process_json_folder(folder_path: str, pattern: str = "*_translations.json") -> Tuple[int, int]:
    """
    批量处理文件夹中的JSON文件，将翻译写回原文
    
    Args:
        folder_path: 文件夹路径
        pattern: 文件匹配模式
        
    Returns:
        Tuple[int, int]: (成功处理的文件数, 总文件数)
    """
    import glob
    
    if not os.path.isdir(folder_path):
        logger.warning(f"Folder not found: {folder_path}")
        return 0, 0
    
    # Find all matching JSON files
    search_pattern = os.path.join(folder_path, "**", pattern)
    json_files = glob.glob(search_pattern, recursive=True)
    
    successful = 0
    total = len(json_files)
    
    logger.info(f"Found {total} JSON files in {folder_path}")
    
    for json_file in json_files:
        try:
            if restore_translation_to_text(json_file):
                successful += 1
                logger.info(f"Successfully processed: {os.path.basename(json_file)}")
            else:
                logger.debug(f"No changes needed for: {os.path.basename(json_file)}")
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {e}")
    
    return successful, total


def process_json_file_list(file_paths: List[str]) -> Tuple[int, int]:
    """
    处理指定的图片文件列表，查找对应的JSON文件并处理翻译写回
    
    Args:
        file_paths: 图片文件路径列表
        
    Returns:
        Tuple[int, int]: (成功处理的文件数, 总文件数)
    """
    successful = 0
    total = 0
    
    for file_path in file_paths:
        # Build the path of the matching JSON file
        json_path = os.path.splitext(file_path)[0] + "_translations.json"
        
        if os.path.exists(json_path):
            total += 1
            try:
                if restore_translation_to_text(json_path):
                    successful += 1
                    logger.info(f"Processed JSON for: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Failed to process JSON for {file_path}: {e}")
        else:
            logger.debug(f"No JSON file found for: {os.path.basename(file_path)}")
    
    return successful, total


def should_restore_translation_to_text(load_text_enabled: bool, template_enabled: bool) -> bool:
    """
    检查是否应该执行翻译写回原文的预处理
    
    Args:
        load_text_enabled: 是否启用了加载文本模式
        template_enabled: 是否启用了模板模式
        
    Returns:
        bool: 是否应该处理
    """
    result = load_text_enabled and template_enabled
    logger.debug(f"DEBUG: should_restore_translation_to_text - load_text={load_text_enabled}, template={template_enabled}, result={result}")
    return result

def parse_template(template_string: str):
    """
    Parses a free-form text template to find prefix, suffix, item_template, and separator.
    An 'item' is defined as a line containing the <original> placeholder.
    """
    output_format, template_string = parse_translation_template_config(template_string)
    logger.debug(
        f"Parsing template (output_format={output_format}):\n---\n"
        f"{template_string[:200]}...\n---"
    )
    # Find all lines containing <original>
    lines = template_string.splitlines(True) # Keep endings to preserve original spacing
    item_line_indices = [i for i, line in enumerate(lines) if "<original>" in line]
    logger.debug(f"Found {len(item_line_indices)} lines with <original>: {item_line_indices}")

    if not item_line_indices:
        raise ValueError("Template must contain at least one '<original>' placeholder.")

    # Define the item_template from the first found item line
    first_item_line_index = item_line_indices[0]
    first_item_line = lines[first_item_line_index]

    # Extract item_template: from the start of <original> to the end of <translated> (without the leading spaces, the trailing comma and so on)
    # Find where <original> starts
    original_placeholder = "<original>"
    original_start_index = first_item_line.find(original_placeholder)

    # Find where <translated> ends
    translated_placeholder = "<translated>"
    translated_end_index = first_item_line.find(translated_placeholder)
    if translated_end_index != -1:
        translated_end_index += len(translated_placeholder)
        item_template = first_item_line[original_start_index:translated_end_index]
    else:
        item_template = first_item_line[original_start_index:]

    # Extract the leading spaces (from the start of the line to <original>)
    leading_spaces = first_item_line[:original_start_index]

    # Define prefix (includes the leading spaces of the first item)
    prefix_lines = lines[:first_item_line_index]
    prefix = "".join(prefix_lines) + leading_spaces

    # Define separator and suffix
    if len(item_line_indices) > 1:
        # Separator is the content between the end of <translated> and the start of next <original>
        # This includes trailing characters on the first line (like comma) and content between lines
        second_item_line_index = item_line_indices[1]
        second_item_line = lines[second_item_line_index]

        # From the end of <translated> on the first line to the end of that line
        separator_from_first_line = first_item_line[translated_end_index:]

        # What lies between the first and the second line
        separator_lines = lines[first_item_line_index + 1 : second_item_line_index]
        separator_between_lines = "".join(separator_lines)

        # The second line from its start to where <original> begins (leading spaces)
        second_original_start_index = second_item_line.find("<original>")
        if second_original_start_index > 0:
            separator_to_second_line = second_item_line[:second_original_start_index]
        else:
            separator_to_second_line = ""

        # Put the separator together
        separator = separator_from_first_line + separator_between_lines + separator_to_second_line

        # Suffix is the content after the last item line's <translated>
        last_item_line_index = item_line_indices[-1]
        last_item_line = lines[last_item_line_index]
        last_translated_end_index = last_item_line.find(translated_placeholder)
        if last_translated_end_index != -1:
            last_translated_end_index += len(translated_placeholder)
            suffix_from_last_line = last_item_line[last_translated_end_index:]
        else:
            suffix_from_last_line = last_item_line

        suffix_lines = lines[last_item_line_index + 1:]
        suffix = suffix_from_last_line + "".join(suffix_lines)
    else:
        # Only one item, so no separator, and suffix is everything after <translated>
        separator = ""
        suffix_from_first_line = first_item_line[translated_end_index:]
        suffix_lines = lines[first_item_line_index + 1:]
        suffix = suffix_from_first_line + "".join(suffix_lines)
    
    logger.debug(f"Parsed template parts: prefix='{prefix.strip()}', separator='{separator.strip()}', suffix='{suffix.strip()}'")
    logger.debug(f"Item template: '{item_template}'")
    logger.debug(f"Item template (repr): {repr(item_template)}")
    logger.debug(f"Prefix (repr): {repr(prefix)}")
    logger.debug(f"Separator (repr): {repr(separator)}")
    logger.debug(f"Prefix spaces: {prefix.count(' ')}")
    logger.debug(f"Separator spaces: {separator.count(' ')}")
    return prefix, item_template, separator, suffix


def _load_template_definition(template_path: str = None):
    """统一加载输出格式与占位符模板；所有扩展名共用同一套解析逻辑。"""
    output_format = get_translation_output_format(template_path)
    if not template_path or not os.path.exists(template_path):
        return output_format, None

    with open(template_path, 'r', encoding='utf-8') as f:
        template_string = f.read()
    return output_format, parse_template(template_string)


def _render_template_items(items, template_parts, fallback_field: str) -> str:
    if template_parts is None:
        return '\n'.join(item[fallback_field] for item in items)

    prefix, item_template, separator, suffix = template_parts
    if not items:
        return prefix + suffix

    formatted_items = []
    for item in items:
        formatted_item = item_template.replace('<original>', item['original'])
        formatted_item = formatted_item.replace('<translated>', item['translated'])
        formatted_items.append(formatted_item)
    return prefix + separator.join(formatted_items) + suffix

def generate_original_text(
    detailed_json_path: str,
    template_path: str = None,
    output_path: str = None
) -> str:
    """
    导出原文到TXT文件

    Args:
        detailed_json_path: JSON文件路径
        template_path: 模板文件路径（可选，用于格式化）
        output_path: 输出文件路径（可选，默认使用path_manager生成）

    Returns:
        输出文件路径或错误信息
    """
    try:
        with open(detailed_json_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except Exception as e:
        return f"Error reading JSON file: {e}"

    image_data = next(iter(source_data.values()), None)
    if not image_data or 'regions' not in image_data:
        return "Error: Could not find 'regions' list in source JSON."
    regions = image_data.get('regions', [])

    # Collect original texts and translations (when exporting the original text, the translation field is filled with the translation from the JSON)
    items = []
    for region in regions:
        original_text = region.get('text', '').replace('[BR]', '')
        translated_text = region.get('translation', '').replace('[BR]', '')
        if original_text.strip():
            items.append({
                'original': original_text,
                'translated': translated_text if translated_text else original_text  # When translation is empty, use the original text as a placeholder
            })

    try:
        output_format, template_parts = _load_template_definition(template_path)
    except Exception as e:
        return f"Error reading template file: {e}"
    
    # Record whether there is any text
    if not items:
        logger.info(
            f"No text regions found in {detailed_json_path}, "
            f"will create empty .{output_format} file"
        )

    # Build the output path
    if output_path is None:
        # Infer the image path from the JSON path
        json_dir = os.path.dirname(detailed_json_path)
        json_basename = os.path.basename(detailed_json_path)

        # Check whether it is in the new folder structure
        if json_dir.endswith(os.path.join('manga_translator_work', 'json')):
            # Infer the path of the original image
            work_dir = os.path.dirname(json_dir)
            image_dir = os.path.dirname(work_dir)
            image_name = json_basename.replace('_translations.json', '')
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_path = os.path.join(image_dir, image_name + ext)
                if os.path.exists(image_path):
                    output_path = get_original_txt_path(
                        image_path,
                        output_format=output_format,
                    )
                    break
            if output_path is None:
                # When the image is not found, use the folder of the JSON
                output_path = os.path.splitext(detailed_json_path)[0] + f'_original.{output_format}'
        else:
            # Old format: use the folder of the JSON
            output_path = os.path.splitext(detailed_json_path)[0] + f'_original.{output_format}'

    # Format the output with the template
    try:
        output_content = _render_template_items(
            items,
            template_parts,
            fallback_field='original',
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        logger.info(f"Original text exported to: {output_path}")
    except Exception as e:
        return f"Error writing to output file: {e}"

    return output_path


def generate_translated_text(
    detailed_json_path: str,
    template_path: str = None,
    output_path: str = None
) -> str:
    """
    导出翻译到TXT文件

    Args:
        detailed_json_path: JSON文件路径
        template_path: 模板文件路径（可选，用于格式化）
        output_path: 输出文件路径（可选，默认使用path_manager生成）

    Returns:
        输出文件路径或错误信息
    """
    try:
        with open(detailed_json_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except Exception as e:
        return f"Error reading JSON file: {e}"

    image_data = next(iter(source_data.values()), None)
    if not image_data or 'regions' not in image_data:
        return "Error: Could not find 'regions' list in source JSON."
    regions = image_data.get('regions', [])

    # Collect original texts and translations
    items = []
    for region in regions:
        original_text = region.get('text', '').replace('[BR]', '')
        translated_text = region.get('translation', '').replace('[BR]', '')
        if original_text.strip():
            items.append({
                'original': original_text,
                'translated': translated_text  # When exporting the translation, the translation field is the real translation
            })

    try:
        output_format, template_parts = _load_template_definition(template_path)
    except Exception as e:
        return f"Error reading template file: {e}"

    # Build the output path
    if output_path is None:
        # Infer the image path from the JSON path
        json_dir = os.path.dirname(detailed_json_path)
        json_basename = os.path.basename(detailed_json_path)

        # Check whether it is in the new folder structure
        if json_dir.endswith(os.path.join('manga_translator_work', 'json')):
            # Infer the path of the original image
            work_dir = os.path.dirname(json_dir)
            image_dir = os.path.dirname(work_dir)
            image_name = json_basename.replace('_translations.json', '')
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_path = os.path.join(image_dir, image_name + ext)
                if os.path.exists(image_path):
                    output_path = get_translated_txt_path(
                        image_path,
                        output_format=output_format,
                    )
                    break
            if output_path is None:
                # When the image is not found, use the folder of the JSON
                output_path = os.path.splitext(detailed_json_path)[0] + f'_translated.{output_format}'
        else:
            # Old format: use the folder of the JSON
            output_path = os.path.splitext(detailed_json_path)[0] + f'_translated.{output_format}'

    # Format the output with the template
    try:
        output_content = _render_template_items(
            items,
            template_parts,
            fallback_field='translated',
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        logger.info(f"Translated text exported to: {output_path}")
    except Exception as e:
        return f"Error writing to output file: {e}"

    return output_path


def generate_text_from_template(
    detailed_json_path: str,
    template_path: str
) -> str:
    """
    Generates a custom text format based on a free-form text template file.
    保留用于向后兼容，现在会同时生成原文和翻译两个文件
    """
    # Generate the original text
    original_result = generate_original_text(detailed_json_path, template_path)
    if original_result.startswith("Error"):
        logger.warning(f"Failed to generate original text: {original_result}")

    # Generate the translation
    translated_result = generate_translated_text(detailed_json_path, template_path)
    if translated_result.startswith("Error"):
        return translated_result

    # Return the path of the translation file (kept for backward compatibility)
    return translated_result

def get_template_path_from_config(custom_path: str = None) -> str:
    """
    获取模板文件路径，支持自定义路径
    
    Args:
        custom_path: 用户指定的自定义模板路径
        
    Returns:
        str: 最终使用的模板路径
    """
    base_path = get_application_dir()

    # Priority: given by the user > environment variable > default path
    if custom_path:
        path_to_check = custom_path if os.path.isabs(custom_path) else os.path.join(base_path, custom_path)
        if os.path.exists(path_to_check):
            logger.debug(f"Using user-provided template path: {path_to_check}")
            return path_to_check
    
    env_template = os.environ.get('MANGA_TEMPLATE_PATH')
    if env_template:
        path_to_check = env_template if os.path.isabs(env_template) else os.path.join(base_path, env_template)
        if os.path.exists(path_to_check):
            logger.debug(f"Using environment variable template path: {path_to_check}")
            return path_to_check
    
    default_path = get_default_template_path()
    logger.debug(f"Using default template path: {default_path}")
    return default_path


def create_template_selection_dialog(parent=None):
    """
    创建模板选择对话框
    
    Args:
        parent: 父窗口
        
    Returns:
        str: 选择的模板文件路径，如果取消则返回None
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Without a parent window, create a hidden root window
        if parent is None:
            root = tk.Tk()
            root.withdraw()
            parent = root
        
        # Open the file dialog
        template_path = filedialog.askopenfilename(
            parent=parent,
            title="选择翻译模板文件",
            filetypes=[
                ("JSON模板文件", "*.json"),
                ("文本模板文件", "*.txt"),
                ("所有文件", "*.*")
            ],
            initialdir=os.path.dirname(get_default_template_path())
        )
        
        return template_path if template_path else None
        
    except ImportError:
        logger.warning("Cannot import tkinter; file selection dialog is unavailable")
        return None
    except Exception as e:
        logger.error(f"Failed to create template selection dialog: {e}")
        return None


def export_with_custom_template(
    json_path: str, 
    template_path: str = None,
    output_path: str = None
) -> str:
    """
    使用自定义模板导出翻译文件
    
    Args:
        json_path: JSON文件路径
        template_path: 模板文件路径
        output_path: 输出文件路径，如果为None则自动生成
        
    Returns:
        str: 导出结果或错误信息
    """
    if not os.path.exists(json_path):
        return f"错误：JSON文件不存在: {json_path}"
    
    # Get the template path
    final_template_path = get_template_path_from_config(template_path)
    if not os.path.exists(final_template_path):
        return f"错误：模板文件不存在: {final_template_path}"
    
    # Build the output path
    if output_path is None:
        base_name = os.path.splitext(json_path)[0]
        if base_name.endswith("_translations"):
            # From "image_translations.json" build "image_translations.txt"
            output_path = base_name + ".txt"
        else:
            output_path = base_name + ".txt"
    
    try:
        result_path = generate_text_from_template(json_path, final_template_path)
        if result_path and os.path.exists(result_path):
            return f"成功导出到: {result_path}"
        else:
            return f"导出失败: {result_path}"
    except Exception as e:
        return f"导出过程中出错: {e}"


def import_with_custom_template(
    txt_path: str,
    json_path: str = None, 
    template_path: str = None
) -> str:
    """
    使用自定义模板从TXT文件导入翻译到JSON
    
    Args:
        txt_path: TXT文件路径
        json_path: JSON文件路径，如果为None则自动推断
        template_path: 模板文件路径
        
    Returns:
        str: 导入结果或错误信息
    """
    if not os.path.exists(txt_path):
        return f"错误：TXT文件不存在: {txt_path}"
    
    # Infer the JSON path automatically
    if json_path is None:
        base_name = os.path.splitext(txt_path)[0]
        json_path = base_name + ".json"
    
    if not os.path.exists(json_path):
        return f"错误：JSON文件不存在: {json_path}"
    
    # Get the template path
    final_template_path = get_template_path_from_config(template_path)
    if not os.path.exists(final_template_path):
        return f"错误：模板文件不存在: {final_template_path}"
    
    try:
        result = safe_update_large_json_from_text(txt_path, json_path, final_template_path)
        return result
    except Exception as e:
        return f"导入过程中出错: {e}"


def get_default_template_path() -> str:
    """获取默认模板文件路径"""
    return get_config_path("translation_template.json")


def ensure_default_template_exists() -> str:
    """确保默认模板存在，并复用核心层的统一默认内容。"""
    return ensure_translation_template_exists()


def smart_update_translations_from_images(
    image_file_paths: List[str],
    template_path: str = None
) -> str:
    """
    根据加载的图片文件路径，智能匹配对应的JSON和TXT文件进行翻译更新
    支持新的目录结构和向后兼容

    Args:
        image_file_paths: 图片文件路径列表
        template_path: 模板文件路径，如果为None则使用默认模板

    Returns:
        str: 处理结果报告
    """
    if not image_file_paths:
        return "错误：未提供图片文件路径"

    # Use the default template when none is given, and make sure the template file exists
    if template_path is None:
        template_path = ensure_default_template_exists()
        if template_path is None:
            return "错误：无法创建或找到默认模板文件"

    if not os.path.exists(template_path):
        return f"错误：模板文件不存在: {template_path}"

    results = []

    for image_path in image_file_paths:
        if not os.path.exists(image_path):
            results.append(f"✗ {os.path.basename(image_path)}: 图片文件不存在")
            continue

        # Find the JSON and TXT files with path_manager (supports the new folder structure)
        json_path = find_json_path(image_path)
        original_txt_path, translated_txt_path = find_txt_files(image_path)

        # Check that the files exist
        if not json_path:
            results.append(f"- {os.path.basename(image_path)}: 未找到JSON文件")
            continue

        # Use the original-text TXT (imported after the user edited the original text)
        txt_path = original_txt_path if original_txt_path else translated_txt_path

        if not txt_path:
            results.append(f"- {os.path.basename(image_path)}: 未找到TXT文件")
            continue

        # Update the translations
        try:
            result = safe_update_large_json_from_text(txt_path, json_path, template_path)
            results.append(f"✓ {os.path.basename(image_path)}: {result}")
        except Exception as e:
            results.append(f"✗ {os.path.basename(image_path)}: 更新失败 - {e}")

    if not results:
        return "未找到任何可处理的文件"

    # Count the results
    successful = len([r for r in results if r.startswith("✓")])
    total = len(results)

    summary = f"批量翻译更新完成 (成功: {successful}/{total}):\n" + "\n".join(results)
    return summary


def auto_detect_and_update_translations(
    directory_or_files,
    template_path: str = None
) -> str:
    """
    自动检测并更新翻译 - 支持目录或文件列表
    
    Args:
        directory_or_files: 目录路径(str) 或 图片文件路径列表(List[str])
        template_path: 模板文件路径
        
    Returns:
        str: 处理结果报告
    """
    if isinstance(directory_or_files, str):
        # For a folder path, scan the folder for image files
        if os.path.isdir(directory_or_files):
            import glob
            
            image_files = []
            
            for ext in IMAGE_FILE_GLOB_PATTERNS:
                pattern = os.path.join(directory_or_files, "**", ext)
                image_files.extend(glob.glob(pattern, recursive=True))
                # Search for upper-case extensions as well
                pattern = os.path.join(directory_or_files, "**", ext.upper())
                image_files.extend(glob.glob(pattern, recursive=True))
            
            if not image_files:
                return f"在目录 {directory_or_files} 中未找到任何图片文件"
            
            return smart_update_translations_from_images(image_files, template_path)
        else:
            return f"错误：目录不存在: {directory_or_files}"
    
    elif isinstance(directory_or_files, list):
        # For a file list, process it directly
        return smart_update_translations_from_images(directory_or_files, template_path)
    
    else:
        return "错误：参数类型不正确，需要目录路径或图片文件路径列表"


def _load_large_json_optimized(json_file_path: str):
    """优化的大文件JSON加载"""
    import ijson
    try:
        # Parse as a stream with ijson and materialise as a dictionary at once
        with open(json_file_path, 'rb') as f:
            return dict(ijson.kvitems(f, ''))
    except ImportError:
        # Without ijson, fall back to the standard method but read in chunks
        logger.warning("ijson is unavailable; using standard loading for large files")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def _strip_legacy_horizontal_tags(text):
    """剥除已废除的 <H>...</H> 局部横排标记（保留内文）。

    渲染管线已删除全部 <H> 消费方，字面标记会被当普通字符画上成品图；
    局部横排改用富文本 tcy（旧 <H> 协议已废除）。
    """
    if not isinstance(text, str):
        return text
    if '<H>' not in text and '</H>' not in text:
        return text
    return text.replace('<H>', '').replace('</H>', '')


def safe_update_large_json_from_text(
    text_file_path: str,
    json_file_path: str,
    template_path: str
) -> str:
    """
    安全地更新大型JSON文件，保护原始数据完整性
    """
    logger.debug(f"Starting safe update. TXT: '{os.path.basename(text_file_path)}', JSON: '{os.path.basename(json_file_path)}'")
    import shutil
    import tempfile
    import time
    
    # Check that the file exists
    for file_path, name in [(text_file_path, "TXT"), (json_file_path, "JSON"), (template_path, "模板")]:
        if not os.path.exists(file_path):
            return f"错误：{name}文件不存在: {file_path}"
    
    # Get the file size
    json_size_mb = os.path.getsize(json_file_path) / (1024 * 1024)
    logger.info(f"Processing JSON file: {os.path.basename(json_file_path)} ({json_size_mb:.2f} MB)")
    
    try:
        # 1. Parse the template and the TXT file
        logger.debug("Reading template and text files.")
        with open(template_path, 'r', encoding='utf-8') as f:
            template_string = f.read()
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except Exception as e:
        return f"错误：读取输入文件失败: {e}"

    try:
        prefix, item_template, separator, suffix = parse_template(template_string)
    except ValueError as e:
        return f"错误：解析模板失败: {e}"

    # 2. Parse the translation content
    logger.debug("Parsing translations from text content.")
    translations = {}
    
    # First try to parse it directly as JSON (supports the compact format)
    try:
        parsed_json = json.loads(text_content)
        if isinstance(parsed_json, dict):
            translations = parsed_json
            logger.info(f"Parsed JSON directly; found {len(translations)} translations")
        else:
            raise ValueError("Not a dict")
    except (json.JSONDecodeError, ValueError):
        # When JSON parsing fails, use the original template parsing logic
        # Remove the prefix and the suffix
        if prefix and text_content.startswith(prefix):
            text_content = text_content[len(prefix):]
        if suffix and text_content.endswith(suffix):
            text_content = text_content[:-len(suffix)]

        # Split into entries
        if separator:
            # Try to split with the separator
            items = text_content.split(separator)
            # When only 1 item comes out, it may be the compact format (no line breaks): try splitting on commas
            if len(items) == 1 and ',' in text_content:
                # Split with a regular expression: matches the pattern "key": "value",
                items = re.split(r'",\s*"', text_content)
        else:
            items = [text_content] if text_content.strip() else []
        logger.debug(f"Found {len(items)} items in text file.")

        # Parse each entry
        parts = re.split(f'({re.escape("<original>")}|{re.escape("<translated>")})', item_template)
        parser_regex_str = ""
        group_order = []
        for part in parts:
            if part == "<original>":
                parser_regex_str += "(.+?)"  # The original text must have at least one character
                group_order.append("original")
            elif part == "<translated>":
                parser_regex_str += "(.*)"  # The translation may be empty; match to the end
                group_order.append("translated")
            else:
                parser_regex_str += re.escape(part)
        
        # Add an end anchor, so it matches to the end of the string
        parser_regex_str += "$"
        parser_regex = re.compile(parser_regex_str, re.DOTALL)

        for item in items:
            item_stripped = item.strip()
            if not item_stripped:
                continue
            
            match = parser_regex.search(item)
            if match:
                try:
                    result = {}
                    for j, group_name in enumerate(group_order):
                        captured_string = match.group(j + 1)
                        result[group_name] = captured_string
                    translations[result['original']] = result['translated']
                except (IndexError, KeyError):
                    continue  # Skip entries that fail to parse

    if not translations:
        logger.warning(f"Could not parse any translations from '{os.path.basename(text_file_path)}'.")
        return "错误：未能从TXT文件中解析出任何翻译内容"

    logger.info(f"Parsed {len(translations)} translations")

    # 2.5. Create a normalised mapping (for fuzzy matching)
    def normalize_text(text):
        """标准化文本：去除特殊字符、统一空白字符"""
        import unicodedata
        # Remove control characters (C), format characters (Cf), the replacement character (U+FFFD) and so on
        # Keep letters (L), numbers (N), punctuation (P), symbols (S) and marks (M)
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] not in ['C', 'Z'] and ch != '\ufffd')
        # Normalise whitespace
        text = ' '.join(text.split())
        return text

    # Create the normalised mapping: normalized_text -> original_text
    normalized_to_original = {}
    for original_text in translations.keys():
        normalized = normalize_text(original_text)
        normalized_to_original[normalized] = original_text
        if len(normalized_to_original) <= 3:  # Only the first 3 are logged
            logger.debug(f"Normalized mapping: '{original_text}' -> '{normalized}'")

    # 3. Create a temporary backup file
    backup_path = None
    temp_path = None
    try:
        # Create the backup
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # backup_path = f"{json_file_path}.backup_{timestamp}"
        # shutil.copy2(json_file_path, backup_path)
        # logger.info(f"Created backup: {os.path.basename(backup_path)}")

        # 4. Load and update the JSON in a memory-saving way
        pass
        start_time = time.time()
        
        # Large files are processed as a stream, to use less memory
        if json_size_mb > 50:  # Above 50MB the optimised processing is used
            logger.debug(f"Loading large file with streaming parser: {os.path.basename(json_file_path)}")
            source_data = _load_large_json_optimized(json_file_path)
        else:
            logger.debug(f"Loading JSON file into memory: {os.path.basename(json_file_path)}")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
        
        load_time = time.time() - start_time
        logger.info(f"JSON loaded in {load_time:.2f} seconds")

        # 5. Update the translations
        logger.debug("Updating translations in memory.")
        updated_count = 0
        image_key = next(iter(source_data.keys()), None)
        
        if not image_key or 'regions' not in source_data[image_key]:
            return "错误：JSON文件格式不正确，找不到regions数据"

        start_time = time.time()
        
        for region in source_data[image_key]['regions']:
            original_text = region.get('text', '')

            # Try an exact match first
            if original_text in translations:
                old_translation = region.get('translation', '')
                new_translation = _strip_legacy_horizontal_tags(translations[original_text])

                # The translation field is always updated, even when original and translation are the same
                if old_translation != new_translation:
                    region['translation'] = new_translation
                    # Writing a plain-text translation must also invalidate the rich-text document (rich takes precedence when rendering;
                    # without clearing it the output image would show the old translation; this matches how editor_controller writes)
                    region.pop('translation_rich', None)
                    updated_count += 1
                    logger.debug(f"Updating translation: '{original_text[:30]}...' -> '{new_translation[:30]}...'")
            else:
                # When the exact match fails, try a fuzzy match
                normalized = normalize_text(original_text)
                logger.debug(f"Exact match failed; trying fuzzy match: '{original_text}' -> '{normalized}'")
                if normalized in normalized_to_original:
                    matched_original = normalized_to_original[normalized]
                    old_translation = region.get('translation', '')
                    new_translation = _strip_legacy_horizontal_tags(translations[matched_original])

                    logger.debug(f"Fuzzy match succeeded: '{original_text}' -> '{matched_original}', old='{old_translation}', new='{new_translation}'")

                    # The translation field is always updated, even when original and translation are the same
                    if old_translation != new_translation:
                        region['translation'] = new_translation
                        # As above: pop translation_rich when writing translation
                        region.pop('translation_rich', None)
                        updated_count += 1
                else:
                    logger.debug(f"Fuzzy match also failed: '{normalized}' not in normalized_to_original")

        update_time = time.time() - start_time
        logger.info(f"Update completed in {update_time:.2f} seconds; updated {updated_count} entries")

        # Import translation and render: whether or not the imported content is identical to the existing translation,
        # once this import has run, later rendering should scale the text again.
        source_data[image_key]['skip_font_scaling'] = False

        # 6. Write the file back (through a temporary file, for atomicity)
        logger.debug("Writing updated data to temporary file.")
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, 
                                       dir=os.path.dirname(json_file_path), 
                                       suffix='.tmp') as temp_file:
            temp_path = temp_file.name
            
            # Use the optimised JSON encoder
            class OptimizedJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, 'tolist'):  # numpy array
                        return obj.tolist()
                    if hasattr(obj, '__int__'):  # numpy integer
                        return int(obj)
                    if hasattr(obj, '__float__'):  # numpy float
                        return float(obj)
                    return super().default(obj)
            
            start_time = time.time()
            json.dump(source_data, temp_file, ensure_ascii=False, indent=4, 
                     cls=OptimizedJSONEncoder)
        
        write_time = time.time() - start_time
        logger.info(f"Temporary file written in {write_time:.2f} seconds")

        # 7. Replace the original file atomically
        logger.debug(f"Atomically moving temporary file to final destination: {os.path.basename(json_file_path)}")
        if os.name == 'nt':  # Windows
            # On Windows the target file has to be deleted first
            if os.path.exists(json_file_path):
                os.remove(json_file_path)
        
        shutil.move(temp_path, json_file_path)
        temp_path = None  # Mark as moved, so it is not deleted a second time
        
        # 8. Free memory
        logger.debug("Clearing source data from memory.")
        del source_data
        pass
        # 9. Verify the integrity of the file
        try:
            logger.debug("Verifying integrity of written JSON file.")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            logger.info("File integrity verification passed")
        except Exception:
            # When verification fails, restore the backup
            logger.error("File integrity check failed! Restoring backup.")
            if backup_path and os.path.exists(backup_path):
                shutil.copy2(backup_path, json_file_path)
                return "错误：文件写入后验证失败，已恢复备份。请检查磁盘空间和文件权限。"
        
        # 10. Remove old backups (optional; the 3 most recent are kept)
        try:
            logger.debug("Cleaning up old backups.")
            backup_pattern = f"{json_file_path}.backup_*"
            backup_files = sorted(glob.glob(backup_pattern), reverse=True)
            for old_backup in backup_files[3:]:  # Keep the 3 most recent backups
                try:
                    os.remove(old_backup)
                    logger.debug(f"Deleting old backup: {os.path.basename(old_backup)}")
                except Exception:
                    pass
        except Exception:
            pass

        return f"成功更新 {updated_count} 条翻译 (总时间: {load_time + update_time + write_time:.2f}秒)"

    except Exception as e:
        # Error recovery
        error_msg = f"错误：更新过程中出现异常: {e}"
        backup_recovery = "not attempted"
        
        # Remove the temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                logger.debug(f"Cleaning up temporary file: {temp_path}")
                os.remove(temp_path)
            except Exception:
                pass
        
        # Try to restore the backup
        if backup_path and os.path.exists(backup_path):
            try:
                logger.warning("Exception occurred, attempting to restore backup.")
                shutil.copy2(backup_path, json_file_path)
                error_msg += " (已恢复备份文件)"
                backup_recovery = "restored"
            except Exception:
                error_msg += " (备份恢复失败，请手动恢复)"
                backup_recovery = "failed"
        
        logger.error("Error updating translations: %s (backup recovery: %s)", e, backup_recovery)
        return error_msg

    finally:
        # Force garbage collection
        logger.debug("Running final garbage collection.")
        pass
def batch_update_directory_translations(
    directory_path: str,
    template_path: str = None,
    pattern: str = "*_translations.json"
) -> str:
    """
    批量更新目录中所有JSON文件的翻译
    
    Args:
        directory_path: 目录路径
        template_path: 模板文件路径
        pattern: JSON文件匹配模式
        
    Returns:
        str: 批量处理结果报告
    """
    logger.debug(f"Starting batch update in directory: '{directory_path}' with pattern '{pattern}'")
    import glob

    if not os.path.isdir(directory_path):
        return f"错误：目录不存在: {directory_path}"

    # Use the default template when none is given, and make sure the template file exists
    if template_path is None:
        logger.debug("No template path provided, using default.")
        template_path = ensure_default_template_exists()
        if template_path is None:
            return "错误：无法创建或找到默认模板文件"

    if not os.path.exists(template_path):
        return f"错误：模板文件不存在: {template_path}"
    logger.debug(f"Using template: {template_path}")

    search_pattern = os.path.join(directory_path, "**", pattern)
    json_files = glob.glob(search_pattern, recursive=True)
    logger.debug(f"Found {len(json_files)} JSON files: {json_files}")

    if not json_files:
        return f"在目录 {directory_path} 中未找到匹配 '{pattern}' 的JSON文件"

    results = []
    for json_path in json_files:
        logger.debug(f"Processing file: {json_path}")

        # Infer the image path from the JSON path, then get the TXT path with path_manager
        json_dir = os.path.dirname(json_path)
        json_basename = os.path.basename(json_path)

        # Check whether it is in the new folder structure
        if json_dir.endswith(os.path.join('manga_translator_work', 'json')):
            # Infer the path of the original image
            work_dir = os.path.dirname(json_dir)
            image_dir = os.path.dirname(work_dir)
            image_name = json_basename.replace('_translations.json', '')

            image_path = None
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                candidate = os.path.join(image_dir, image_name + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break

            if image_path:
                from manga_translator.utils.path_manager import get_original_txt_path
                txt_path = get_original_txt_path(image_path, create_dir=False)
            else:
                # When the image is not found, use the folder of the JSON
                txt_path = os.path.splitext(json_path)[0] + ".txt"
        else:
            # Old format: use the folder of the JSON
            txt_path = os.path.splitext(json_path)[0] + ".txt"

        if not os.path.exists(txt_path):
            logger.warning(f"Could not find matching TXT file for '{os.path.basename(json_path)}', skipping.")
            results.append(f"- {os.path.basename(json_path)}: 未找到对应的TXT文件 ({os.path.basename(txt_path)})")
            continue

        try:
            result = safe_update_large_json_from_text(txt_path, json_path, template_path)
            results.append(f"✓ {os.path.basename(json_path)}: {result}")
        except Exception as e:
            logger.error(f"An exception occurred while processing '{os.path.basename(json_path)}': {e}", exc_info=True)
            results.append(f"✗ {os.path.basename(json_path)}: 更新失败 - {e}")

    successful = len([r for r in results if r.startswith("✓")])
    total = len(json_files)
    summary = f"批量更新完成 (处理: {successful}/{total}):\n" + "\n".join(results)
    logger.debug("Batch update completed (processed: %s/%s); files: %s", successful, total, json_files)
    return summary
