"""
Unified prompt file loader

YAML (.yaml/.yml) and JSON (.json) are supported, and YAML is loaded first.
When a .yaml and a .json file of the same name both exist, the .yaml one is used.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger('manga_translator')

# Cache of the loaded yaml module
_yaml_module = None
_yaml_available = None


def _get_yaml():
    """Load the yaml module lazily, to avoid an error at start-up"""
    global _yaml_module, _yaml_available
    if _yaml_available is None:
        try:
            import yaml
            _yaml_module = yaml
            _yaml_available = True
        except ImportError:
            _yaml_available = False
            logger.warning("PyYAML not installed. YAML prompt files will not be supported. Install with: pip install pyyaml")
    return _yaml_module if _yaml_available else None


def load_prompt_file(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a single prompt file (the format is detected automatically)

    Args:
        path: file path (.yaml/.yml/.json)

    Returns:
        The parsed dictionary, or None when loading fails
    """
    if not path or not os.path.exists(path):
        return None

    ext = os.path.splitext(path)[1].lower()

    try:
        with open(path, 'r', encoding='utf-8') as f:
            if ext in ('.yaml', '.yml'):
                yaml = _get_yaml()
                if yaml is None:
                    logger.error(f"Cannot load YAML file {path}: PyYAML not installed")
                    return None
                data = yaml.safe_load(f)
            elif ext == '.json':
                data = json.load(f)
            else:
                logger.warning(f"Unsupported prompt file format: {ext}")
                return None

        if not isinstance(data, dict):
            logger.warning(f"Prompt file {path} did not parse to a dict (got {type(data).__name__})")
            return None

        return data

    except Exception as e:
        logger.error(f"Failed to load prompt file {path}: {e}")
        return None


def resolve_prompt_path(base_dir: str, stem: str) -> Optional[str]:
    """
    Find a prompt file by its name without extension, YAML first.

    Search order: .yaml → .yml → .json

    Args:
        base_dir: folder path
        stem: file name without extension, for example "system_prompt_hq"

    Returns:
        The full path of the file that was found, or None when there is none
    """
    for ext in ('.yaml', '.yml', '.json'):
        path = os.path.join(base_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def load_prompt_by_stem(base_dir: str, stem: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load a prompt by its file name without extension, YAML first.

    Args:
        base_dir: folder path
        stem: file name without extension

    Returns:
        (data, path) - the loaded data and the actual file path, or (None, None) on failure
    """
    path = resolve_prompt_path(base_dir, stem)
    if path is None:
        return None, None
    data = load_prompt_file(path)
    return data, path


def load_system_prompt_hq(dict_dir: str) -> str:
    """
    Load the HQ system prompt

    Args:
        dict_dir: path of the dict/ folder

    Returns:
        The prompt text, or an empty string when loading fails
    """
    data, path = load_prompt_by_stem(dict_dir, 'system_prompt_hq')
    if data is None:
        return ""
    prompt = data.get('system_prompt', '')
    if prompt and path:
        logger.debug(f"Loaded HQ system prompt from: {path}")
    return prompt


def load_system_prompt_hq_format(dict_dir: str, target_lang: str, extract_glossary: bool = False) -> str:
    """
    Load the HQ output format prompt.

    Args:
        dict_dir: path of the dict/ folder
        target_lang: name of the target language
        extract_glossary: whether new_terms is required as extra output

    Returns:
        The prompt text with the placeholders replaced, or an empty string when loading fails
    """
    data, path = load_prompt_by_stem(dict_dir, 'system_prompt_hq_format')
    if data is None:
        return ""

    prompt = data.get('system_prompt_hq_format', '')
    if not prompt:
        return ""

    prompt = prompt.replace("{{{target_lang}}}", target_lang)

    if extract_glossary:
        optional_new_terms_rule = (
            '    -   The object MUST also contain a key "new_terms" which is a list of objects.\n'
            '    -   If no new terms are found, return `"new_terms": []`.\n'
            '    -   Each item in "new_terms" MUST have "original", "category", and "aliases".\n'
            '    -   "aliases" MUST be a list. Each alias MUST have "original" and "translations".\n'
            '    -   Each alias "translations" list MUST contain exactly one object with only "text".\n'
        )
        optional_new_terms_example_suffix = (
            ',\n'
            '  "new_terms": [\n'
            '    {\n'
            '      "original": "Excalibur",\n'
            '      "category": "Item",\n'
            '      "aliases": [\n'
            '        {\n'
            '          "original": "Excalibur",\n'
            '          "translations": [\n'
            f'            {{ "text": "<{target_lang} translation>" }}\n'
            '          ]\n'
            '        }\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
        )
        optional_new_terms_final_instruction = ' Return "new_terms": [] when no new terms are found.'
    else:
        optional_new_terms_rule = ""
        optional_new_terms_example_suffix = ""
        optional_new_terms_final_instruction = ""

    prompt = prompt.replace("{{{optional_new_terms_rule}}}", optional_new_terms_rule)
    prompt = prompt.replace("{{{optional_new_terms_example_suffix}}}", optional_new_terms_example_suffix)
    prompt = prompt.replace("{{{optional_new_terms_final_instruction}}}", optional_new_terms_final_instruction)

    if path:
        logger.debug(f"Loaded HQ output format prompt from: {path}")
    return prompt


def load_line_break_prompt(dict_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load the AI line-breaking prompt

    Args:
        dict_dir: path of the dict/ folder

    Returns:
        The prompt dictionary (with the line_break_prompt field), or None when loading fails
    """
    data, path = load_prompt_by_stem(dict_dir, 'system_prompt_line_break')
    if data and path:
        logger.debug(f"Loaded line break prompt from: {path}")
    return data


def load_glossary_extraction_prompt(dict_dir: str, target_lang: str) -> str:
    """
    Load the glossary extraction prompt

    Args:
        dict_dir: path of the dict/ folder
        target_lang: name of the target language

    Returns:
        The prompt text with the placeholders replaced, or an empty string when loading fails
    """
    data, path = load_prompt_by_stem(dict_dir, 'glossary_extraction_prompt')
    if data is None:
        return ""
    prompt = data.get('glossary_extraction_prompt', '')
    if prompt:
        prompt = prompt.replace("{{{target_lang}}}", target_lang)
        if path:
            logger.debug(f"Loaded glossary extraction prompt from: {path}")
    return prompt


def load_glossary_output_format_prompt(dict_dir: str, target_lang: str) -> str:
    """
    For old callers: load the HQ output format prompt used when glossary extraction is on.

    Args:
        dict_dir: path of the dict/ folder
        target_lang: name of the target language

    Returns:
        The prompt text with the placeholders replaced, or an empty string when loading fails
    """
    return load_system_prompt_hq_format(dict_dir, target_lang, extract_glossary=True)


def load_custom_prompt(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a prompt file defined by the user (.yaml/.yml/.json are supported)

    When the given path does not exist, the other extension is tried once.
    For example, when xxx.json does not exist, xxx.yaml is tried.

    Args:
        path: path of the prompt file given by the user

    Returns:
        The parsed dictionary, or None when loading fails
    """
    if not path:
        return None

    # The path exists as given
    if os.path.exists(path):
        return load_prompt_file(path)

    # Try with the other extension
    base, ext = os.path.splitext(path)
    alt_exts = ['.yaml', '.yml', '.json']
    for alt_ext in alt_exts:
        if alt_ext != ext:
            alt_path = base + alt_ext
            if os.path.exists(alt_path):
                logger.info(f"Prompt file {path} not found, using {alt_path} instead")
                return load_prompt_file(alt_path)

    logger.warning(f"Custom prompt file not found: {path}")
    return None


def list_prompt_files(dict_dir: str, exclude_system: bool = True) -> list:
    """
    List the prompt files in the dict/ folder

    Args:
        dict_dir: path of the dict/ folder
        exclude_system: whether the system prompt files are left out

    Returns:
        The list of file names
    """
    system_stems = {
        'system_prompt_hq',
        'system_prompt_hq_format',
        'system_prompt_line_break',
        'glossary_extraction_prompt'
    }
    prompt_exts = {'.json', '.yaml', '.yml'}

    if not os.path.exists(dict_dir):
        return []

    files = []
    for f in os.listdir(dict_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext not in prompt_exts:
            continue
        if exclude_system:
            stem = os.path.splitext(f)[0]
            if stem in system_stems:
                continue
        files.append(f)

    return sorted(files)
