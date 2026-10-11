# Text filtering tools
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from manga_translator.runtime_paths import get_config_dir

from . import get_logger

logger = get_logger('TextFilter')

# Filter list cache: (contains filter list, exact filter list)
_filter_lists: Optional[Tuple[List[str], List[str]]] = None

_FILTER_LIST_FILENAME = 'filter_list.json'
_LEGACY_FILTER_LIST_FILENAME = 'filter_list.txt'

_DEFAULT_FILTER_LIST_DATA = {
    "contains": [],
    "exact": [],
}


def _get_filter_list_path() -> str:
    """
    Get the path of the JSON filter list file

    Packaged: config/filter_list.json next to the executable
    Development: config/filter_list.json in the project root
    """
    return os.path.join(get_config_dir(), _FILTER_LIST_FILENAME)


def _get_legacy_filter_list_path() -> str:
    """Get the path of the old TXT filter list file."""
    return os.path.join(get_config_dir(), _LEGACY_FILTER_LIST_FILENAME)


def _sanitize_rule_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []

    sanitized: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            sanitized.append(text)
    return sanitized


def _normalize_rule_list(values: Any) -> List[str]:
    return [item.lower() for item in _sanitize_rule_list(values)]


def _write_filter_list_json(path: str, data: Dict[str, Any]) -> None:
    payload = dict(data) if isinstance(data, dict) else {}
    payload["contains"] = _sanitize_rule_list(payload.get("contains", []))
    payload["exact"] = _sanitize_rule_list(payload.get("exact", []))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def _parse_legacy_filter_list(path: str) -> Tuple[List[str], List[str]]:
    contains_list: List[str] = []
    exact_list: List[str] = []
    current_section = None

    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line == '[包含过滤]':
                current_section = 'contains'
                continue
            if line == '[精确过滤]':
                current_section = 'exact'
                continue

            if current_section == 'contains':
                contains_list.append(line)
            elif current_section == 'exact':
                exact_list.append(line)

    return contains_list, exact_list


def _migrate_legacy_filter_list() -> bool:
    json_path = _get_filter_list_path()
    legacy_path = _get_legacy_filter_list_path()

    if os.path.exists(json_path) or not os.path.exists(legacy_path):
        return False

    try:
        contains_list, exact_list = _parse_legacy_filter_list(legacy_path)
        _write_filter_list_json(json_path, {
            "contains": contains_list,
            "exact": exact_list,
        })
        logger.info(f"Migrated legacy filter list to JSON: {json_path}")
        return True
    except Exception as exc:
        logger.error(f"Failed to migrate legacy filter list: {exc}")
        return False


def ensure_filter_list_exists() -> str:
    """
    Make sure the filter list file exists; when it does not, create the default JSON file.

    Returns:
        The path of the filter list file
    """
    filter_path = _get_filter_list_path()

    if os.path.exists(filter_path):
        return filter_path

    if _migrate_legacy_filter_list():
        return filter_path

    try:
        _write_filter_list_json(filter_path, _DEFAULT_FILTER_LIST_DATA)
        logger.info(f"Created filter list file: {filter_path}")
    except Exception as exc:
        logger.error(f"Failed to create filter list file: {exc}")

    return filter_path


def load_filter_list_config() -> Dict[str, List[str]]:
    """
    Load the filter list JSON configuration, keeping the original letter case.
    """
    filter_path = ensure_filter_list_exists()

    try:
        with open(filter_path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("filter list json root must be an object")
        return {
            "contains": _sanitize_rule_list(data.get("contains", [])),
            "exact": _sanitize_rule_list(data.get("exact", [])),
        }
    except Exception as exc:
        logger.error(f"Failed to load filter list configuration: {exc}")
        return {
            "contains": [],
            "exact": [],
        }


def save_filter_list_config(data: Dict[str, Any]) -> str:
    """
    Save the filter list JSON configuration.
    """
    global _filter_lists

    filter_path = _get_filter_list_path()
    _write_filter_list_json(filter_path, data)
    _filter_lists = None
    return filter_path


def load_filter_list(force_reload: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load the filter list.

    Args:
        force_reload: whether a reload is forced

    Returns:
        (contains filter list, exact filter list), both in lower case
    """
    global _filter_lists

    if _filter_lists is not None and not force_reload:
        return _filter_lists

    try:
        config = load_filter_list_config()
        contains_list = _normalize_rule_list(config.get("contains", []))
        exact_list = _normalize_rule_list(config.get("exact", []))

        if contains_list or exact_list:
            logger.info(f"Loaded filter rules: {len(contains_list)} substring filters, {len(exact_list)} exact match filters")

        _filter_lists = (contains_list, exact_list)
    except Exception as exc:
        logger.error(f"Failed to load filter list: {exc}")
        _filter_lists = ([], [])

    return _filter_lists


def match_filter(text: str) -> Optional[Tuple[str, str]]:
    """
    Check whether a text matches the filter list.

    Args:
        text: the text to check

    Returns:
        (the filter word that matched, the kind of match), or None when nothing matches.
        The kind of match is one of two fixed strings, for a "contains" match and for an "exact" match
    """
    if not text:
        return None

    contains_list, exact_list = load_filter_list()
    text_lower = text.lower()

    for filter_word in exact_list:
        if text_lower == filter_word:
            return (filter_word, "精确")

    for filter_word in contains_list:
        if filter_word in text_lower:
            return (filter_word, "包含")

    return None


def should_filter(text: str) -> bool:
    """
    Check whether a text should be filtered out.

    Args:
        text: the text to check

    Returns:
        True when it should be filtered out, otherwise False
    """
    return match_filter(text) is not None
