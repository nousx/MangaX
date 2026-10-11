"""Batch edit engine - evaluating conditions, batch actions, and reading, changing and writing ``_translations.json``.

Pure logic with no Qt dependency; the threading is wrapped in ``batch_edit_service``. This way the condition evaluation and the write-back
semantics can be unit-tested directly (the same reason ``file_list_data_service.build_file_catalog_snapshot``
keeps its pure-function entry separate).

Three conventions that must hold (from studying the existing chain):

1. **Read everything → change locally → write everything.** ``export_service`` rebuilds the dict from scratch when it saves
   and loses ``mask_raw`` / ``original_width|height`` / overlays. Here only the matched
   region entries are replaced, and every other key is kept as it is.
2. **Matching runs on the rich-text body (``\\n``)**, not on ``translation`` (``[BR]``),
   otherwise the four characters of ``[BR]`` would pollute the character indexes.
3. **After the text changes, the rich text has to follow.** ``apply_text_change`` splices ranges (unchanged
   characters keep their own style), and when the result has no style the ``translation_rich`` field is deleted instead of leaving an
   empty document.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from editor.rich_text_editing import (
    apply_ruby_to_range,
    apply_style_to_range,
    apply_tcy_to_range,
    clear_styles_from_range,
    document_from_region,
    normalize_text_style,
    plain_text_to_storage_text,
    storage_text_to_editor_text,
    styled_segments_for_range,
    visible_text_from_document,
)
from editor.rich_text_presets import normalize_rich_text_preset
from manga_translator.rendering.rich_text import ensure_rich_text_document
from manga_translator.rendering.rich_text_sync import document_has_styling
from utils.json_encoder import CustomJSONEncoder

from .batch_edit_schemes import (
    ACTION_REPLACE_TEXT,
    ACTION_RICH_TEXT,
    ACTION_SET_FIELDS,
    LOGIC_ALL,
    LOGIC_ANY,
    RICH_MODE_FILL,
    RICH_MODE_OVERWRITE,
    RICH_MODE_REPLACE,
)


class BatchEditCancelled(RuntimeError):
    """The scan or the run was cancelled by the caller."""


# ─── Field table ───

KIND_TEXT = "text"
KIND_ENUM = "enum"
KIND_NUMBER = "number"
KIND_COLOR = "color"
KIND_BOOL = "bool"


@dataclass(frozen=True)
class FieldSpec:
    key: str
    kind: str
    label: str
    choices: tuple[str, ...] = ()
    #: Derived fields can only be used in conditions; set_fields cannot write them
    writable: bool = True
    integer: bool = False


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("translation", KIND_TEXT, "Translation"),
    FieldSpec("text", KIND_TEXT, "Source Text", writable=False),
    FieldSpec("translation_raw", KIND_TEXT, "Translation (pre-replacement)"),
    FieldSpec("font_family", KIND_TEXT, "Font Family"),
    FieldSpec("target_lang", KIND_TEXT, "Target Language"),
    FieldSpec("source_lang", KIND_TEXT, "Source Language"),
    FieldSpec("direction", KIND_ENUM, "Direction", ("h", "v", "auto")),
    FieldSpec("alignment", KIND_ENUM, "Alignment", ("left", "center", "right", "auto")),
    FieldSpec("font_size", KIND_NUMBER, "Font Size", integer=True),
    FieldSpec("angle", KIND_NUMBER, "Angle"),
    FieldSpec("line_spacing", KIND_NUMBER, "Line Spacing"),
    FieldSpec("letter_spacing", KIND_NUMBER, "Letter Spacing"),
    FieldSpec("stroke_width", KIND_NUMBER, "Stroke Width"),
    FieldSpec("prob", KIND_NUMBER, "OCR Confidence", writable=False),
    FieldSpec("fg_colors", KIND_COLOR, "Text Color"),
    FieldSpec("bg_colors", KIND_COLOR, "Stroke Color"),
    # The region-level bold/italic/underline/font_weight are in TextBlock.to_dict(),
    # but no UI writes them, and on the rendering side nothing reads italic/underline/font_weight at all
    # (bold is only read by rendering/__init__.py, and its value is always the default False). The bold and italic
    # that really take effect live in the rich-text styles, so these four fields are kept out of the batch table; choosing them would do nothing.
    FieldSpec("has_rich_text", KIND_BOOL, "Has Rich Text", writable=False),
    FieldSpec("line_count", KIND_NUMBER, "Line Count", writable=False, integer=True),
    FieldSpec("region_index", KIND_NUMBER, "Region Index", writable=False, integer=True),
)

FIELDS_BY_KEY: dict[str, FieldSpec] = {spec.key: spec for spec in FIELDS}

OPS_BY_KIND: dict[str, tuple[str, ...]] = {
    KIND_TEXT: ("contains", "not_contains", "eq", "ne", "regex", "not_regex", "empty", "not_empty"),
    KIND_ENUM: ("eq", "ne"),
    KIND_NUMBER: ("eq", "ne", "gt", "gte", "lt", "lte", "between"),
    KIND_COLOR: ("color_eq", "color_near"),
    KIND_BOOL: ("is_true", "is_false"),
}

#: These operators need no value; the UI should hide the value editor
VALUELESS_OPS = frozenset({"empty", "not_empty", "is_true", "is_false"})

OP_LABELS: dict[str, str] = {
    "contains": "contains",
    "not_contains": "does not contain",
    "eq": "equals",
    "ne": "not equal to",
    "regex": "matches regex",
    "not_regex": "does not match regex",
    "empty": "is empty",
    "not_empty": "is not empty",
    "gt": "greater than",
    "gte": "at least",
    "lt": "less than",
    "lte": "at most",
    "between": "between",
    "color_eq": "equals color",
    "color_near": "close to color",
    "is_true": "is yes",
    "is_false": "is no",
}

_DIRECTION_ALIASES = {
    "horizontal": "h",
    "vertical": "v",
    "h": "h",
    "v": "v",
    "hr": "h",  # For old data; the reading order is decided by the language.
    "vr": "v",
    "auto": "auto",
}

#: On save the editor wrote fg_colors/bg_colors as font_color/bg_color hex strings;
#: both forms have to be recognised, otherwise only half of a batch of images would match.
_COLOR_FALLBACKS = {"fg_colors": "font_color", "bg_colors": "bg_color"}


# ─── Reading values / normalisation ───


def region_visible_text(region: dict) -> str:
    """The translation body of a region, in ``\\n`` convention (rich text first, then the BR of ``translation``)."""
    try:
        return visible_text_from_document(document_from_region(region))
    except (TypeError, ValueError):
        return str(region.get("translation", "") or "")


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_rgb(value: Any) -> Optional[tuple[float, float, float]]:
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if len(text) == 6:
            try:
                return tuple(float(int(text[i:i + 2], 16)) for i in (0, 2, 4))  # type: ignore[return-value]
            except ValueError:
                return None
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        channels = [_to_float(channel) for channel in value[:3]]
        if all(channel is not None for channel in channels):
            return tuple(channels)  # type: ignore[return-value]
    return None


def _normalize_direction(value: Any) -> str:
    return _DIRECTION_ALIASES.get(str(value or "").strip().lower(), str(value or "").strip().lower())


def region_field_value(region: dict, key: str, region_index: int = 0) -> Any:
    """Read a value by the field table, absorbing the differences between old forms of a region."""
    if key == "translation":
        return region_visible_text(region)
    if key == "has_rich_text":
        try:
            document = ensure_rich_text_document(region.get("translation_rich"))
        except (TypeError, ValueError):
            return False
        return document_has_styling(document)
    if key == "line_count":
        return region_visible_text(region).count("\n") + 1
    if key == "region_index":
        return region_index
    if key == "direction":
        return _normalize_direction(region.get("direction"))
    if key in _COLOR_FALLBACKS:
        value = region.get(key)
        if value in (None, "", []):
            value = region.get(_COLOR_FALLBACKS[key])
        return value
    return region.get(key)


# ─── Evaluating conditions ───


def _match_text(value: Any, op: str, expected: Any) -> bool:
    text = "" if value is None else str(value)
    if op == "empty":
        return not text.strip()
    if op == "not_empty":
        return bool(text.strip())
    needle = "" if expected is None else str(expected)
    if op == "contains":
        return needle in text
    if op == "not_contains":
        return needle not in text
    if op == "eq":
        return text == needle
    if op == "ne":
        return text != needle
    if op in ("regex", "not_regex"):
        try:
            hit = re.search(needle, text) is not None
        except re.error:
            return False
        return hit if op == "regex" else not hit
    return False


def _match_enum(value: Any, op: str, expected: Any) -> bool:
    left = str(value or "").strip().lower()
    right = str(expected or "").strip().lower()
    if left in _DIRECTION_ALIASES or right in _DIRECTION_ALIASES:
        left = _DIRECTION_ALIASES.get(left, left)
        right = _DIRECTION_ALIASES.get(right, right)
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    return False


def _match_number(value: Any, op: str, expected: Any) -> bool:
    left = _to_float(value)
    if left is None:
        return False
    if op == "between":
        bounds = expected
        if isinstance(bounds, dict):
            low, high = _to_float(bounds.get("min")), _to_float(bounds.get("max"))
        elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
            low, high = _to_float(bounds[0]), _to_float(bounds[1])
        else:
            return False
        if low is None or high is None:
            return False
        if low > high:
            low, high = high, low
        return low <= left <= high
    right = _to_float(expected)
    if right is None:
        return False
    if op == "eq":
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    if op == "ne":
        return not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    return False


def _match_color(value: Any, op: str, expected: Any) -> bool:
    left = _to_rgb(value)
    if left is None:
        return False
    if isinstance(expected, dict):
        right = _to_rgb(expected.get("color"))
        tolerance = _to_float(expected.get("tolerance"))
    else:
        right = _to_rgb(expected)
        tolerance = None
    if right is None:
        return False
    distance = math.dist(left, right)
    if op == "color_eq":
        return distance == 0
    if op == "color_near":
        return distance <= (tolerance if tolerance is not None else 30.0)
    return False


def _match_bool(value: Any, op: str) -> bool:
    truthy = bool(value)
    return truthy if op == "is_true" else not truthy


def _style_value_matches(actual: Any, expected: Any) -> bool:
    """Comparison of rich-text style values; nested objects are compared by containment of the chosen sub-items."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _style_value_matches(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, str) and isinstance(expected, str) \
            and actual.startswith("#") and expected.startswith("#"):
        return actual.casefold() == expected.casefold()
    return actual == expected


def _rich_text_style_criteria(value: Any) -> Optional[list[tuple[str, Any]]]:
    """Turn the flat style values of a card into standard properties that can be compared item by item."""
    if not isinstance(value, dict):
        return None
    selected = copy.deepcopy(value)
    ruby = selected.pop("ruby", None)
    tcy = selected.pop("tcy", None)
    try:
        style = normalize_text_style(selected)
    except (TypeError, ValueError):
        return None
    criteria = list(style.items())
    if isinstance(ruby, str) and ruby:
        criteria.append(("ruby", ruby))
    if bool(tcy):
        criteria.append(("tcy", True))
    return criteria


def _segment_matches_style_criterion(segment: Any, key: str, expected: Any) -> bool:
    if key == "ruby":
        return segment.node_type == "ruby" and segment.ruby_text == expected
    if key == "tcy":
        return segment.node_type == "tcy"
    return key in segment.style and _style_value_matches(segment.style[key], expected)


def _matching_rich_text_spans(
    document: dict,
    start: int,
    end: int,
    expected: Any,
    logic: str,
) -> list[tuple[int, int]]:
    """Return the character fragments inside a text range that also satisfy the existing rich-text conditions."""
    criteria = _rich_text_style_criteria(expected)
    if not criteria:
        return []
    segments = styled_segments_for_range(document, start, end, expand_empty=False)
    matches: list[tuple[int, int]] = []
    for segment in segments:
        results = [
            _segment_matches_style_criterion(segment, key, value)
            for key, value in criteria
        ]
        if any(results) if logic == LOGIC_ANY else all(results):
            matches.append((max(start, segment.start), min(end, segment.end)))
    return [(span_start, span_end) for span_start, span_end in matches if span_start < span_end]


def evaluate_condition(region: dict, condition: dict, region_index: int = 0) -> bool:
    spec = FIELDS_BY_KEY.get(str(condition.get("field", "")))
    if spec is None:
        return False
    op = str(condition.get("op", ""))
    if op not in OPS_BY_KIND.get(spec.kind, ()):
        return False
    value = region_field_value(region, spec.key, region_index)
    expected = condition.get("value")
    if spec.kind == KIND_TEXT:
        return _match_text(value, op, expected)
    if spec.kind == KIND_ENUM:
        return _match_enum(value, op, expected)
    if spec.kind == KIND_NUMBER:
        return _match_number(value, op, expected)
    if spec.kind == KIND_COLOR:
        return _match_color(value, op, expected)
    if spec.kind == KIND_BOOL:
        return _match_bool(value, op)
    return False


def evaluate_conditions(region: dict, match: dict, region_index: int = 0) -> bool:
    """No conditions = every region matches (the intuitive meaning of "the filter is left empty")."""
    conditions = (match or {}).get("conditions") or []
    if not conditions:
        return True
    results = (evaluate_condition(region, condition, region_index) for condition in conditions)
    if str((match or {}).get("logic", "")).lower() == LOGIC_ANY:
        return any(results)
    return all(results)


# ─── Actions ───


def _compile_pattern(action: dict) -> Optional[re.Pattern]:
    pattern = str(action.get("pattern", "") or "")
    if not pattern:
        return None
    try:
        return re.compile(pattern if action.get("regex") else re.escape(pattern))
    except re.error:
        return None


def _coerce_field_value(spec: Optional[FieldSpec], value: Any) -> Any:
    if spec is None:
        return value
    if spec.key == "direction":
        return _normalize_direction(value)
    if spec.kind == KIND_BOOL:
        return bool(value)
    if spec.kind == KIND_NUMBER:
        number = _to_float(value)
        if number is None:
            return value
        return int(round(number)) if spec.integer else number
    if spec.kind == KIND_COLOR:
        rgb = _to_rgb(value)
        return [int(round(channel)) for channel in rgb] if rgb else value
    return str(value)


_COLLAPSE_BREAKS_RE = re.compile(r"\n+")


def _region_direction(region: dict) -> Any:
    return _normalize_direction(region.get("direction", "h"))


def _sync_translation(
    region: dict,
    *,
    ops: Optional[list],
    pre_text: str,
    post_text: str,
    keep_raw: bool = False,
) -> None:
    """Write back the three translation fields; rich text goes through the editor's ``sync_region_rich_translation``.

    When ``ops`` gives a sequence of ``[pos, removed_len, inserted_text]`` (the same convention as
    ``manga_translator/utils/text_edit_ops.py``), the edit is replayed: unchanged characters
    keep their own style and ruby/tcy node membership in place, and only the few characters that were replaced lose their style.
    ``ops=None`` means the whole translation was rewritten; the old rich text no longer matches the new body and can only be dropped.

    Both paths end with "delete translation_rich when the result has no style" - rich text takes precedence over
    plain text when rendering, so keeping the old one would make the change pointless.
    """
    storage_text = plain_text_to_storage_text(post_text)
    rich = None
    if ops is not None:
        from manga_translator.rendering.rich_text_sync import sync_region_rich_translation

        try:
            rich = sync_region_rich_translation(
                region.get("translation_rich"),
                {"ops": ops, "pre_text": pre_text, "post_text": post_text},
                raw_mode=False,
                new_translation=storage_text,
                direction_value=_region_direction(region),
                old_translation=str(region.get("translation", "") or ""),
            )
        except Exception:
            rich = None

    region["translation"] = storage_text
    if not keep_raw:
        region["translation_raw"] = storage_text
    if rich is not None:
        region["translation_rich"] = rich
    else:
        region.pop("translation_rich", None)


def _apply_set_fields(region: dict, action: dict) -> None:
    translation_value: Optional[str] = None
    raw_written = False
    for key, value in (action.get("fields") or {}).items():
        spec = FIELDS_BY_KEY.get(key)
        if spec is not None and not spec.writable:
            continue
        if key == "translation":
            translation_value = str(value)
            continue  # Deferred, so everything goes through the sync pipeline together
        if key == "translation_raw":
            region[key] = plain_text_to_storage_text(storage_text_to_editor_text(str(value)))
            raw_written = True
            continue
        region[key] = _coerce_field_value(spec, value)

    if translation_value is not None:
        # The user may type [BR]/<br> directly; normalise to \n first, then go through the pipeline
        post_text = storage_text_to_editor_text(translation_value)
        _sync_translation(
            region,
            ops=None,
            pre_text="",
            post_text=post_text,
            keep_raw=raw_written,
        )


def _store_document(region: dict, document: dict) -> None:
    """A document without styling is not worth a field - it is deleted instead of leaving an empty shell."""
    try:
        has_styling = document_has_styling(ensure_rich_text_document(document))
    except (TypeError, ValueError):
        has_styling = False
    if has_styling:
        region["translation_rich"] = document
    else:
        region.pop("translation_rich", None)


def _expand_replacement(match: re.Match, action: dict) -> str:
    replacement = str(action.get("replace", "") or "")
    if not action.get("regex"):
        return replacement
    try:
        return match.expand(replacement)
    except (re.error, IndexError):
        # The user wrote something like \d into the replacement string: treat it as a literal, so the whole batch does not crash here
        return replacement


def _collapsed_index_map(raw_text: str) -> list[int]:
    """Index after collapsing line breaks → index in the original document.

    ops run in the coordinate system where "consecutive line breaks are collapsed into one", while the styles of the original document have to be read with the original indexes;
    how far the two differ depends on how many runs of line breaks the text has, so a character-by-character table is the only way.
    """
    mapping: list[int] = []
    previous_is_break = False
    for index, char in enumerate(raw_text):
        if char == "\n" and previous_is_break:
            continue
        mapping.append(index)
        previous_is_break = char == "\n"
    return mapping


def _style_at(document: dict, index: int) -> tuple[dict, Optional[str], str]:
    """Get the style and the ruby/tcy membership at a position in the body; empty when there is no style."""
    for segment in styled_segments_for_range(document, index, index + 1, expand_empty=False):
        if segment.start <= index < segment.end:
            return segment.style or {}, segment.node_type, segment.ruby_text
    return {}, None, ""


def _restore_replaced_styles(region: dict, carried: Sequence[tuple]) -> None:
    """Attach the style of the text that was replaced to the new characters that replace it.

    Replaying ops lets new characters inherit a style only when "the neighbours before and after the insertion point have the same style", and the new characters of a replacement
    mostly sit on a style boundary and inherit nothing - so a styled word would lose its style as soon as it is replaced.
    This adds the rule the user decided on: the whole span takes the style of the first character of the matched range (when the range had several
    styles, only one can be taken).
    """
    if not carried:
        return
    document = document_from_region(region)
    text_length = len(visible_text_from_document(document))
    changed = False
    for start, end, style, node_type, ruby_text in carried:
        if start >= end or end > text_length:
            continue
        if style:
            document = apply_style_to_range(document, start, end, style)
            changed = True
        if node_type == "ruby" and ruby_text:
            document = apply_ruby_to_range(document, start, end, ruby_text)
            changed = True
        elif node_type == "tcy":
            document = apply_tcy_to_range(document, start, end)
            changed = True
    if changed:
        _store_document(region, document)


def _apply_replace_text(region: dict, action: dict) -> None:
    pattern = _compile_pattern(action)
    if pattern is None:
        return
    document = document_from_region(region)
    raw_text = visible_text_from_document(document)
    # Coordinate convention of ops = the document body with consecutive line breaks collapsed into one (as in _collapse_linebreak_entries)
    pre_text = _COLLAPSE_BREAKS_RE.sub("\n", raw_text)
    matches = [item for item in pattern.finditer(pre_text) if item.start() != item.end()]
    if not matches:
        return

    raw_index = _collapsed_index_map(raw_text)

    # Matches are produced in ascending order; the position of each op is in the coordinate system where "all earlier ops have been replayed"
    ops: list[list] = []
    carried: list[tuple] = []
    post_text = pre_text
    shift = 0
    for item in matches:
        start, end = item.span()
        replacement = _expand_replacement(item, action)
        ops.append([start + shift, end - start, replacement])
        post_text = post_text[:start + shift] + replacement + post_text[end + shift:]
        if replacement:
            style, node_type, ruby_text = _style_at(document, raw_index[start])
            if style or node_type:
                carried.append((
                    start + shift, start + shift + len(replacement), style, node_type, ruby_text,
                ))
        shift += len(replacement) - (end - start)

    _sync_translation(
        region,
        ops=ops,
        pre_text=pre_text,
        post_text=post_text,
    )
    _restore_replaced_styles(region, carried)


def _rich_text_spans(document: dict, action: dict) -> list[tuple[int, int]]:
    """Target ranges of a rich-text action in the body.

    An empty pattern = all the text of the whole region - choosing which regions is the job of the match conditions;
    here only the substrings inside the selected regions are located.
    """
    text = visible_text_from_document(document)
    if not text:
        return []
    if not str(action.get("pattern", "") or ""):
        spans = [(0, len(text))]
    else:
        pattern = _compile_pattern(action)
        if pattern is None:
            return []
        spans = [item.span() for item in pattern.finditer(text) if item.start() != item.end()]

    match_style = action.get("match_style")
    if not isinstance(match_style, dict) or not match_style:
        return spans
    logic = str(action.get("match_style_logic", LOGIC_ALL) or LOGIC_ALL).lower()
    matched_spans = [
        matched
        for start, end in spans
        for matched in _matching_rich_text_spans(document, start, end, match_style, logic)
    ]
    merged: list[tuple[int, int]] = []
    for start, end in sorted(matched_spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _uniform_style_spans(document: dict, start: int, end: int) -> list[tuple[int, int, dict, bool]]:
    """Cut ``[start, end)`` into sub-ranges that each have one consistent style.

    ``styled_segments_for_range`` only reports the styled fragments, so the blank stretches in between have to be filled back in here
    - fill mode uses this partition to decide which items each position lacks, and a missed blank stretch would be a missed fill.
    Each item is ``(start, end, the existing style of the stretch, whether the stretch is already inside a ruby/tcy node)``.
    """
    spans: list[tuple[int, int, dict, bool]] = []
    cursor = start
    for segment in styled_segments_for_range(document, start, end, expand_empty=False):
        seg_start, seg_end = max(segment.start, start), min(segment.end, end)
        if seg_start >= seg_end:
            continue
        if cursor < seg_start:
            spans.append((cursor, seg_start, {}, False))
        spans.append((seg_start, seg_end, segment.style or {}, segment.node_type is not None))
        cursor = seg_end
    if cursor < end:
        spans.append((cursor, end, {}, False))
    return spans


def _overwrite_rich_text(document: dict, start: int, end: int, preset: dict) -> dict:
    """overwrite: the items you edited win, and other items on the matched range are kept as they are."""
    if preset["style"]:
        document = apply_style_to_range(document, start, end, preset["style"])
    if preset["ruby"]:
        document = apply_ruby_to_range(document, start, end, preset["ruby"])
    elif preset["tcy"]:
        document = apply_tcy_to_range(document, start, end)
    return document


def _fill_rich_text(document: dict, start: int, end: int, preset: dict) -> dict:
    """fill: an existing item of the same name on the matched range wins; only what it lacks is added."""
    spans = _uniform_style_spans(document, start, end)
    for span_start, span_end, existing, _ in reversed(spans):
        # Nested items (stroke/glow/transform...) are judged by their top-level key: when a stroke already exists, it is left alone entirely
        patch = {key: value for key, value in preset["style"].items() if key not in existing}
        if patch:
            document = apply_style_to_range(document, span_start, span_end, patch)
    # ruby/tcy is one node for the whole span and cannot be added piecewise per style sub-range (it would break into several ruby nodes of the same name),
    # so when the range already has any node, the whole span gives way
    if not any(has_node for _, _, _, has_node in spans):
        if preset["ruby"]:
            document = apply_ruby_to_range(document, start, end, preset["ruby"])
        elif preset["tcy"]:
            document = apply_tcy_to_range(document, start, end)
    return document


def _replace_rich_text(document: dict, start: int, end: int, preset: dict) -> dict:
    """replace: the existing styles and nodes of the matched range are cleared, then the new style is applied."""
    document = clear_styles_from_range(document, start, end)
    return _overwrite_rich_text(document, start, end, preset)


def _apply_rich_text(region: dict, action: dict) -> None:
    document = document_from_region(region)
    spans = _rich_text_spans(document, action)
    if not spans:
        return

    mode = str(action.get("mode", "") or RICH_MODE_OVERWRITE)
    preset = normalize_rich_text_preset({
        "style": action.get("style") or {},
        "ruby": action.get("ruby", ""),
        "tcy": action.get("tcy", False),
    })
    if preset is None:
        return
    apply_span = {
        RICH_MODE_FILL: _fill_rich_text,
        RICH_MODE_REPLACE: _replace_rich_text,
    }.get(mode, _overwrite_rich_text)
    for start, end in reversed(spans):
        document = apply_span(document, start, end, preset)
    _store_document(region, document)


_ACTION_HANDLERS: dict[str, Callable[[dict, dict], None]] = {
    ACTION_SET_FIELDS: _apply_set_fields,
    ACTION_REPLACE_TEXT: _apply_replace_text,
    ACTION_RICH_TEXT: _apply_rich_text,
}


def region_is_sane(region: Any) -> bool:
    """Whether the backend's ``TextBlock(**region)`` can take this region.

    When ``texts`` is empty or ``lines`` has the wrong shape, the backend skips the whole region (and trips the fuse that stops the write);
    such a region is not touched here either - changing it might turn it into bad data that looks valid.
    """
    if not isinstance(region, dict):
        return False
    texts = region.get("texts")
    if not isinstance(texts, list) or not texts:
        return False
    lines = region.get("lines")
    if not isinstance(lines, list) or not lines:
        return False
    for line in lines:
        if not isinstance(line, list) or len(line) < 4:
            return False
        for point in line[:4]:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return False
    return True


def apply_scheme_to_region(region: dict, scheme: dict) -> Optional[dict]:
    """Return the changed copy of a region; ``None`` when nothing changed."""
    updated = copy.deepcopy(region)
    for action in scheme.get("actions") or []:
        handler = _ACTION_HANDLERS.get(str(action.get("type", "")))
        if handler is not None:
            handler(updated, action)
    return None if updated == region else updated


# ─── Scanning ───


@dataclass(frozen=True, slots=True)
class MatchItem:
    json_path: str
    image_key: str
    region_index: int
    image_name: str
    before_text: str
    after_text: str
    summary: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.json_path, self.image_key, self.region_index)


@dataclass
class ScanResult:
    matches: list[MatchItem] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    scanned_files: int = 0
    scanned_regions: int = 0
    skipped_regions: int = 0

    @property
    def file_count(self) -> int:
        return len({item.json_path for item in self.matches})


@dataclass
class ApplyReport:
    written_files: list[str] = field(default_factory=list)
    changed_regions: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    backups: list[str] = field(default_factory=list)


def _summarize(before: dict, after: dict) -> str:
    changes: list[str] = []
    for key in sorted(set(before) | set(after)):
        if key == "translation_rich":
            had, has = "translation_rich" in before, "translation_rich" in after
            if before.get(key) != after.get(key):
                changes.append("rich text" if has else ("rich text removed" if had else "rich text"))
            continue
        if before.get(key) != after.get(key):
            changes.append(key)
    return ", ".join(changes)


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise BatchEditCancelled()


def iter_pages(data: Any) -> Iterable[tuple[str, dict]]:
    """Go through every image entry at the top level.

    The existing writers all produce a single key, but after moving to another machine the key is an absolute path of the old machine, and readers always
    fall back to "take the first value"; going through all of them here is more robust than picking one.
    """
    if not isinstance(data, dict):
        return
    for image_key, page in data.items():
        if isinstance(page, dict) and isinstance(page.get("regions"), list):
            yield str(image_key), page


def detect_indent(raw: str, default: int = 4) -> int:
    """Detect the indentation of the original file and keep it (the backend writes 4, the editor writes 2), for the smallest diff."""
    for line in raw.splitlines()[1:]:
        stripped = line.lstrip(" ")
        if not stripped:
            continue
        return len(line) - len(stripped) or default
    return default


def read_json_document(json_path: str) -> tuple[Any, int]:
    with open(json_path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    return json.loads(raw), detect_indent(raw)


def write_json_document(json_path: str, data: Any, indent: int = 4, backup: bool = True) -> Optional[str]:
    """Atomic write; returns the backup path (``None`` when no backup was made).

    The existing chain has no backup mechanism at all, and a batch change is a destructive operation on dozens or hundreds of files at once,
    so a ``.bak`` is made by default.
    """
    backup_path = None
    if backup and os.path.exists(json_path):
        backup_path = json_path + ".bak"
        shutil.copy2(json_path, backup_path)

    directory = os.path.dirname(os.path.abspath(json_path)) or "."
    handle_fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".batch_edit_", suffix=".tmp")
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=indent, ensure_ascii=False, cls=CustomJSONEncoder)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, json_path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
    return backup_path


def scan_file(json_path: str, scheme: dict, result: Optional[ScanResult] = None) -> ScanResult:
    result = result if result is not None else ScanResult()
    try:
        data, _indent = read_json_document(json_path)
    except (OSError, ValueError) as exc:
        result.errors.append((json_path, str(exc)))
        return result

    result.scanned_files += 1
    for image_key, page in iter_pages(data):
        image_name = os.path.basename(image_key) or image_key
        for index, region in enumerate(page.get("regions") or []):
            if not region_is_sane(region):
                result.skipped_regions += 1
                continue
            result.scanned_regions += 1
            if not evaluate_conditions(region, scheme.get("match") or {}, index):
                continue
            updated = apply_scheme_to_region(region, scheme)
            if updated is None:
                continue
            result.matches.append(MatchItem(
                json_path=json_path,
                image_key=image_key,
                region_index=index,
                image_name=image_name,
                before_text=region_visible_text(region),
                after_text=region_visible_text(updated),
                summary=_summarize(region, updated),
            ))
    return result


def scan_matches(
    json_paths: Sequence[str],
    scheme: dict,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ScanResult:
    result = ScanResult()
    total = len(json_paths)
    for position, json_path in enumerate(json_paths, start=1):
        _check_cancelled(cancel_event)
        scan_file(json_path, scheme, result)
        if progress is not None:
            progress(position, total)
    return result


def apply_matches(
    selected: Iterable[tuple[str, str, int]],
    scheme: dict,
    backup: bool = True,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ApplyReport:
    """Apply a scheme to the selected ``(json_path, image_key, region_index)`` items.

    When it runs, the files are read from disk again and the scheme is run once more (instead of applying the cached preview result), so a file
    changed by another process between preview and run does not get a result based on stale data.
    """
    grouped: dict[str, dict[str, set[int]]] = {}
    for json_path, image_key, region_index in selected:
        grouped.setdefault(json_path, {}).setdefault(image_key, set()).add(int(region_index))

    report = ApplyReport()
    total = len(grouped)
    for position, (json_path, pages) in enumerate(sorted(grouped.items()), start=1):
        _check_cancelled(cancel_event)
        try:
            data, indent = read_json_document(json_path)
        except (OSError, ValueError) as exc:
            report.errors.append((json_path, str(exc)))
            continue

        changed = 0
        for image_key, page in iter_pages(data):
            wanted = pages.get(image_key)
            if not wanted:
                continue
            regions = page.get("regions") or []
            for index in sorted(wanted):
                if index >= len(regions):
                    continue
                region = regions[index]
                if not region_is_sane(region):
                    continue
                if not evaluate_conditions(region, scheme.get("match") or {}, index):
                    continue
                updated = apply_scheme_to_region(region, scheme)
                if updated is None:
                    continue
                regions[index] = updated
                changed += 1

        if not changed:
            continue
        try:
            backup_path = write_json_document(json_path, data, indent=indent, backup=backup)
        except OSError as exc:
            report.errors.append((json_path, str(exc)))
            continue
        report.written_files.append(json_path)
        report.changed_regions += changed
        if backup_path:
            report.backups.append(backup_path)
        if progress is not None:
            progress(position, total)
    return report


# ─── Restoring ───


def backup_path_for(json_path: str) -> str:
    return json_path + ".bak"


def has_backup(json_path: str) -> bool:
    return os.path.isfile(backup_path_for(json_path))


@dataclass
class RestoreReport:
    restored_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


def restore_files(
    json_paths: Sequence[str],
    progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RestoreReport:
    """Restore each json from the ``.bak`` next to it; the ``.bak`` is used up in place by the restore.

    ``os.replace`` is used instead of "read it and write it back atomically": only the directory entry changes and no data moves,
    which is faster than copying byte by byte and is atomic in itself - there is no trade-off.
    """
    report = RestoreReport()
    paths = sorted({os.path.abspath(path) for path in json_paths})
    total = len(paths)
    for position, json_path in enumerate(paths, start=1):
        _check_cancelled(cancel_event)
        backup = backup_path_for(json_path)
        if not os.path.isfile(backup):
            report.missing_files.append(json_path)
            continue
        try:
            os.replace(backup, json_path)
        except OSError as exc:
            report.errors.append((json_path, str(exc)))
            continue
        report.restored_files.append(json_path)
        if progress is not None:
            progress(position, total)
    return report
