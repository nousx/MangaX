"""Helpers for editing structured rich text from the Qt floating editor.

The one implementation of parsing and serialising the richtext.v1 protocol is in manga_translator.rendering.rich_text
(gathered there by F11); this module is only responsible for the structured edit operations on the editor side: inserting and deleting at the cursor
(apply_text_change), style patches (apply_style_to_range), and wrapping and unwrapping ruby/tcy.

All edit operations share one "flattened node membership" representation (F01/F17):
document → one _CharEntry(char, style, run, node) per visible character, where
style/run/node are shared references into the dicts of the original document (read-only, no deep copy per character, F25);
an edit = splicing or rewriting the entry sequence, which _document_from_entries then rebuilds -
consecutive characters with the same node are regrouped into a node of the original type (ruby keeps its original ruby text, tcy rebuilds content),
and stretches with node None are grouped by style into text runs.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from manga_translator.rendering.rich_text import (
    RICH_TEXT_FORMAT,
    TextStyle,
    ensure_rich_text_document,
    is_rich_text_document,
    legacy_line_breaks_to_document,
    normalize_rich_linebreaks,
    plain_text_of,
)


def editor_text_to_plain_text(text: str) -> str:
    return str(text or "").replace("↵", "\n")


def utf16_length(text: str) -> int:
    """Return the number of UTF-16 code units, as used by the Qt text API."""
    return len(str(text or "").encode("utf-16-le")) // 2


def python_index_to_utf16_offset(text: str, index: int) -> int:
    """Convert a Python character index to a QTextCursor/contentsChange offset."""
    text = str(text or "")
    index = max(0, min(int(index), len(text)))
    return utf16_length(text[:index])


def utf16_offset_to_python_index(
    text: str, offset: int, *, round_up: bool = False
) -> int:
    """Convert a UTF-16 offset to a Python character index.

    Qt normally only gives character boundaries. When a caller passes a position in the middle of a surrogate pair, the start of a range is rounded
    down and the end of a range can be rounded up with ``round_up=True``, so a non-BMP character is not split.
    """
    text = str(text or "")
    offset = max(0, min(int(offset), utf16_length(text)))
    units = 0
    for index, char in enumerate(text):
        next_units = units + (2 if ord(char) > 0xFFFF else 1)
        if offset < next_units:
            return index + 1 if round_up else index
        if offset == next_units:
            return index + 1
        units = next_units
    return len(text)


def utf16_range_to_python_range(text: str, start: int, end: int) -> tuple[int, int]:
    """Convert a Qt UTF-16 selection safely to a Python half-open interval."""
    start, end = sorted((int(start), int(end)))
    py_start = utf16_offset_to_python_index(text, start)
    py_end = utf16_offset_to_python_index(text, end, round_up=True)
    return py_start, max(py_start, py_end)


def plain_text_to_storage_text(text: str) -> str:
    # Kept on purpose: line breaks are written back to the translation field as [BR] markers, so PSD export and other
    # downstream code that expects the [BR] form keeps working (see the note on F31 in the review report).
    return re.sub(r"\n+", "[BR]", str(text or ""))


def storage_text_to_editor_text(text: Any) -> str:
    if is_rich_text_document(text):
        return plain_text_of(text)
    # Thin wrapper: the one implementation of BR marker -> line break is in rich_text.py (F14).
    return normalize_rich_linebreaks(str(text or ""))


def document_from_region(region_data: dict) -> dict:
    rich = region_data.get("translation_rich")
    if is_rich_text_document(rich):
        try:
            # Strict parsing + to_dict: it makes an isolated copy and also normalises a RichTextDocument instance
            # into the dict form the editor works with; an invalid document degrades to plain text instead of making
            # the editor crash (for the whole-document fallback at the load boundary see F04).
            return ensure_rich_text_document(rich).to_dict()
        except (ValueError, TypeError):
            pass
    # The one implementation of line break / BR marker -> paragraph is in rich_text.py (F11).
    return legacy_line_breaks_to_document(
        str(region_data.get("translation", "") or "")
    ).to_dict()


def visible_text_from_document(document: Any) -> str:
    # Thin wrapper (F11): plain_text_of accepts both a RichTextDocument instance and a dict,
    # and returns a plain string unchanged - there is no longer a branch that outputs str(document) garbage.
    return plain_text_of(document)


def normalize_text_style(style: Any) -> dict:
    """Entry point for style validation and normalisation, shared by the rich-text editor and the rules editor."""
    if not isinstance(style, dict):
        style = {}
    return TextStyle.from_dict(copy.deepcopy(style)).to_dict()


def text_style_to_control_values(style: Any) -> dict:
    """Expand a nested richtext.v1 style into flat values the controls can read and write directly."""
    style = normalize_text_style(style)
    transform = style.get("transform") or {}
    stroke = style.get("stroke") or {}
    outer_stroke = style.get("outerStroke") or {}
    glow = style.get("glow") or {}
    italic = style.get("italic")
    # The renderer treats the legacy boolean form ``italic: true`` as the
    # reference 15-degree shear.  Expose that same numeric value in the rule
    # editor's angle control instead of coercing ``True`` to 1 degree.
    if italic is True:
        italic = 15.0
    return {
        "bold": bool(style.get("bold", False)),
        "underline": bool(style.get("underline", False)),
        "strikethrough": bool(style.get("strikethrough", False)),
        "emphasis": bool(style.get("emphasis", False)),
        "verticalAdvance": style.get("verticalAdvance"),
        "italic": italic,
        "color": style.get("color"),
        "fontSize": style.get("fontSize"),
        "scale": style.get("scale"),
        "fontFamily": style.get("fontFamily"),
        "stroke": copy.deepcopy(stroke) or None,
        "outerStroke": copy.deepcopy(outer_stroke) or None,
        "glow": copy.deepcopy(glow) or None,
        "kerning": style.get("kerning"),
        "preKerning": style.get("preKerning"),
        "lineKerning": style.get("lineKerning"),
        "nextKerning": style.get("nextKerning"),
        "rotation": transform.get("rotation"),
        "offsetX": transform.get("offsetX"),
        "offsetY": transform.get("offsetY"),
        "scaleX": transform.get("scaleX"),
        "scaleY": transform.get("scaleY"),
    }


def text_style_from_control_values(values: dict, enabled: set[str]) -> dict:
    """Build a strict richtext.v1 style from the shared control values. Fields that are not enabled are not written."""
    style: dict[str, Any] = {}
    for key in (
        "bold",
        "underline",
        "strikethrough",
        "emphasis",
        "italic",
        "color",
        "fontSize",
        "scale",
        "fontFamily",
        "stroke",
        "outerStroke",
        "glow",
        "kerning",
        "preKerning",
        "lineKerning",
        "nextKerning",
        "verticalAdvance",
    ):
        if key in enabled:
            style[key] = copy.deepcopy(values.get(key))
    transform = {
        key: values.get(key)
        for key in ("rotation", "offsetX", "offsetY", "scaleX", "scaleY")
        if key in enabled
    }
    if transform:
        style["transform"] = transform
    return normalize_text_style(style)


# ---------------------------------------------------------------------------
# Flatten node membership (shared walk, F01/F17/F25)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CharEntry:
    """One visible character with the original text run and the ruby/tcy node it belongs to."""

    char: str
    style: dict
    run: dict | None
    node: dict | None


def _visible_entries(document: Any) -> list[_CharEntry]:
    """Flatten a document into membership entries, one per visible character (the only walk of the blocks→inlines cursor).

    The ruby text runs take no visible position and produce no entries; by the protocol, base/content only hold
    text runs, and illegally nested nodes are ignored here (strict parsing on the render side rejects them anyway).
    """
    entries: list[_CharEntry] = []
    if not is_rich_text_document(document):
        return entries
    if not isinstance(document, dict):
        # RichTextDocument instance -> the editor always works with the dict form
        document = ensure_rich_text_document(document).to_dict()
    blocks = document.get("blocks", [])
    if not isinstance(blocks, list):
        return entries
    for block_index, block in enumerate(blocks):
        inlines = block.get("inlines", []) if isinstance(block, dict) else []
        for inline in inlines:
            if not isinstance(inline, dict):
                continue
            inline_type = inline.get("type", "text")
            if inline_type == "ruby":
                _append_run_entries(inline.get("base", []), inline, entries)
            elif inline_type == "tcy":
                _append_run_entries(inline.get("content", []), inline, entries)
            else:
                _append_run_entries([inline], None, entries)
        if block_index < len(blocks) - 1:
            entries.append(_CharEntry("\n", {}, None, None))
    return entries


def _append_run_entries(
    runs: Any, node: dict | None, entries: list[_CharEntry]
) -> None:
    if not isinstance(runs, list):
        return
    for run in runs:
        if not isinstance(run, dict) or run.get("type", "text") != "text":
            continue
        style = run.get("style") if isinstance(run.get("style"), dict) else {}
        for char in str(run.get("text", "")):
            entries.append(_CharEntry(char, style, run, node))


def _document_from_entries(entries: list[_CharEntry]) -> dict:
    """Rebuild paragraphs and inline nodes from the canonical visible entries."""
    blocks: list[dict] = []
    line_start = 0
    while True:
        line_end = line_start
        while line_end < len(entries) and entries[line_end].char != "\n":
            line_end += 1

        inlines: list[dict] = []
        index = line_start
        while index < line_end:
            node = entries[index].node
            group_start = index
            index += 1
            while index < line_end and entries[index].node is node:
                index += 1
            runs = _runs_from_group(entries, group_start, index)
            if node is None:
                inlines.extend(runs)
            elif runs:
                if node.get("type") == "ruby":
                    inlines.append(
                        {
                            "type": "ruby",
                            "base": runs,
                            "text": copy.deepcopy(node.get("text", [])),
                        }
                    )
                else:
                    inlines.append({"type": "tcy", "content": runs})
        blocks.append({"type": "paragraph", "inlines": inlines})
        if line_end == len(entries):
            break
        line_start = line_end + 1

    return {"format": RICH_TEXT_FORMAT, "blocks": blocks}


def _runs_from_group(entries: list[_CharEntry], start: int, end: int) -> list[dict]:
    """Group adjacent entries with equal styles into the canonical text runs."""
    if start >= end:
        return []
    runs: list[dict] = []
    run_start = start
    current_style = entries[start].style
    for index in range(start + 1, end):
        style = entries[index].style
        if style is current_style or style == current_style:
            continue
        runs.append(
            {
                "type": "text",
                "text": "".join(
                    entries[offset].char for offset in range(run_start, index)
                ),
                "style": copy.deepcopy(current_style or {}),
            }
        )
        run_start = index
        current_style = style
    runs.append(
        {
            "type": "text",
            "text": "".join(entries[offset].char for offset in range(run_start, end)),
            "style": copy.deepcopy(current_style or {}),
        }
    )
    return runs


def _normalize_range(
    entries: list[_CharEntry], start: int, end: int, expand_empty: bool
) -> tuple[int, int]:
    length = len(entries)
    start = int(start)
    end = int(end)
    if expand_empty and start == end:
        return 0, length
    start = max(0, min(start, length))
    end = max(start, min(end, length))
    return start, end


def _runs_text(runs: Any) -> str:
    if not isinstance(runs, list):
        return ""
    return "".join(str(run.get("text", "")) for run in runs if isinstance(run, dict))


# ---------------------------------------------------------------------------
# Edit operations
# ---------------------------------------------------------------------------


def apply_text_change(
    document: dict,
    editor_text: str,
    position: int,
    chars_removed: int,
    chars_added: int,
) -> dict:
    """Update the document with the semantics of QTextDocument.contentsChange, keeping the ruby/tcy nodes (F01)."""
    return _apply_plain_text_change(
        document,
        editor_text_to_plain_text(editor_text),
        position,
        chars_removed,
        chars_added,
    )


def _apply_plain_text_change(
    document: dict,
    new_text: str,
    position: int,
    chars_removed: int,
    chars_added: int,
) -> dict:
    entries = _visible_entries(document)
    position = max(0, min(int(position), len(entries)))
    chars_removed = max(0, int(chars_removed))
    chars_added = max(0, int(chars_added))
    inherited_style, inherited_node = _insertion_inheritance(entries, position)
    inserted = [
        _CharEntry(char, inherited_style, None, inherited_node)
        for char in new_text[position : position + chars_added]
    ]
    new_entries = entries[:position] + inserted + entries[position + chars_removed :]
    for index, char in enumerate(new_text):
        if index < len(new_entries):
            new_entries[index].char = char
        else:
            new_entries.append(_CharEntry(char, {}, None, None))
    del new_entries[len(new_text) :]
    return _document_from_entries(new_entries)


def apply_qt_text_change(
    document: dict,
    old_editor_text: str,
    new_editor_text: str,
    position: int,
    chars_removed: int,
    chars_added: int,
) -> dict:
    """Update the document with the UTF-16 semantics of ``QTextDocument.contentsChange``.

    The edit logic of ``apply_text_change`` uses Python character indexes; here the text before and after the change
    are both examined, and the position/removed/added code units Qt gives are converted to a Python interval that does not split
    non-BMP characters such as emoji.
    """
    old_text = editor_text_to_plain_text(old_editor_text)
    new_text = editor_text_to_plain_text(new_editor_text)
    position = max(0, int(position))
    chars_removed = max(0, int(chars_removed))
    chars_added = max(0, int(chars_added))

    old_start = utf16_offset_to_python_index(old_text, position)
    old_end = utf16_offset_to_python_index(
        old_text,
        position + chars_removed,
        round_up=True,
    )
    new_start = utf16_offset_to_python_index(new_text, position)
    new_end = utf16_offset_to_python_index(
        new_text,
        position + chars_added,
        round_up=True,
    )
    # The text before the change point is identical, so in theory old_start == new_start. With odd signal
    # arguments the smaller value is taken, which still keeps the range inside the common prefix of the two texts.
    change_start = min(old_start, new_start)
    removed_seg = old_text[change_start:old_end]
    added_seg = new_text[change_start:new_end]
    # An IME commit may report the whole document as one "full replacement". Compare the text before and after and trim
    # the unchanged start and end of the reported range, narrowing it to the smallest real operation - unchanged characters keep their own style
    # and node membership in place, instead of being rebuilt as newly inserted text (which loses the styling).
    prefix = 0
    limit = min(len(removed_seg), len(added_seg))
    while prefix < limit and removed_seg[prefix] == added_seg[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < limit - prefix
        and removed_seg[len(removed_seg) - 1 - suffix]
        == added_seg[len(added_seg) - 1 - suffix]
    ):
        suffix += 1
    return _apply_plain_text_change(
        document,
        new_text,
        change_start + prefix,
        len(removed_seg) - prefix - suffix,
        len(added_seg) - prefix - suffix,
    )


def _insertion_inheritance(
    entries: list[_CharEntry], position: int
) -> tuple[dict, dict | None]:
    """Style and node membership of inserted characters.

    Style: the style of a text run is inherited only when the insertion point is strictly inside that run (the characters before and after are in the same run);
    at the edge of a run nothing is inherited - the same meaning as in the old implementation.
    Node: the characters join a ruby/tcy node only when the insertion point is strictly inside that node (the characters before and after are in the same
    node); inserted at the front or back edge of a node, or in plain text → ordinary text (decision F01).
    The neighbours are the characters before and after the insertion point in the old document (position-1 / position).
    """
    if position <= 0 or position >= len(entries):
        return {}, None
    prev = entries[position - 1]
    nxt = entries[position]
    style = prev.style if (prev.run is not None and prev.run is nxt.run) else {}
    node = prev.node if (prev.node is not None and prev.node is nxt.node) else None
    return style, node


def _mutate_range(document: dict, start: int, end: int, mutate) -> dict:
    """Apply ``mutate`` to every non-line-break entry in the range, then rebuild the document.

    For an empty range a deep copy of the original document is returned; an empty range is no longer expanded to the whole text - every caller already
    holds a real selection, and rewriting the whole text implicitly would only hide a missing guard higher up.
    """
    entries = _visible_entries(document)
    start, end = _normalize_range(entries, start, end, expand_empty=False)
    if start == end:
        return copy.deepcopy(document)
    for entry in entries[start:end]:
        if entry.char != "\n":
            mutate(entry)
    return _document_from_entries(entries)


def _drop_node_of_type(node_type: str):
    def drop(entry: _CharEntry) -> None:
        if isinstance(entry.node, dict) and entry.node.get("type") == node_type:
            entry.node = None

    return drop


def apply_style_to_range(document: dict, start: int, end: int, patch: dict) -> dict:
    merged_by_style: dict[int, dict] = {}

    def restyle(entry: _CharEntry) -> None:
        merged = merged_by_style.get(id(entry.style))
        if merged is None:
            # Style editing and run inspection must share the exact same
            # richtext.v1 canonical form.  In particular, neutral transform
            # and spacing values (rotation/offset/kerning = 0) are omitted by
            # the protocol and must not survive only in the editor document.
            merged = normalize_text_style(_merge_style(entry.style, patch))
            merged_by_style[id(entry.style)] = merged
        entry.style = merged

    return _mutate_range(document, start, end, restyle)


def apply_tcy_to_range(document: dict, start: int, end: int) -> dict:
    return _wrap_range_as_node(document, start, end, "tcy")


def apply_ruby_to_range(document: dict, start: int, end: int, ruby_text: str) -> dict:
    ruby_text = str(ruby_text or "")
    if not ruby_text:
        return remove_ruby_from_range(document, start, end)
    return _wrap_range_as_node(document, start, end, "ruby", ruby_text=ruby_text)


def remove_ruby_from_range(document: dict, start: int, end: int) -> dict:
    return _mutate_range(document, start, end, _drop_node_of_type("ruby"))


def remove_tcy_from_range(document: dict, start: int, end: int) -> dict:
    return _mutate_range(document, start, end, _drop_node_of_type("tcy"))


def clear_styles_from_range(document: dict, start: int, end: int) -> dict:
    """Remove every inline style and ruby/tcy wrapper from a visible range."""

    def reset(entry: _CharEntry) -> None:
        entry.style = {}
        entry.node = None

    return _mutate_range(document, start, end, reset)


def _wrap_range_as_node(
    document: dict, start: int, end: int, node_type: str, ruby_text: str = ""
) -> dict:
    entries = _visible_entries(document)
    start, end = _normalize_range(entries, start, end, expand_empty=False)
    if start >= end:
        return copy.deepcopy(document)

    # Existing ruby/tcy outside the range is kept as it is (the entries carry node membership and it is restored on rebuild).
    # The range overlaps existing nodes partly or fully: break up the overlapped old nodes - all their characters degrade to
    # ordinary text carrying the original style (the protocol does not allow nested nodes, and keeping half a node on a partial overlap would
    # be ambiguous), then wrap the range as a new node.
    overlapped_ids = {
        id(entry.node) for entry in entries[start:end] if entry.node is not None
    }
    if overlapped_ids:
        for entry in entries:
            if entry.node is not None and id(entry.node) in overlapped_ids:
                entry.node = None

    if node_type == "ruby":
        new_node = {
            "type": "ruby",
            "base": [],
            "text": [{"type": "text", "text": ruby_text, "style": {}}],
        }
    else:
        new_node = {"type": "tcy", "content": []}
    for entry in entries[start:end]:
        entry.node = new_node
    # Note: a range that spans paragraphs is regrouped into one node per paragraph (the ruby text is copied to each paragraph),
    # the same as the line-by-line wrapping of the old implementation.
    return _document_from_entries(entries)


# ---------------------------------------------------------------------------
# Queries (toolbar state of the floating editor)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StyledTextSegment:
    """A real, contiguous range of visible text in the document that carries local styling."""

    start: int
    end: int
    text: str
    style: dict
    node_type: str | None = None
    ruby_text: str = ""
    node_start: int | None = None
    node_end: int | None = None


def styled_segments_for_range(
    document: dict,
    start: int = 0,
    end: int | None = None,
    *,
    expand_empty: bool = True,
) -> list[StyledTextSegment]:
    """Return the real style fragments in text order, without text in the default or an empty style.

    Only adjacent characters whose style, node type and ruby text are exactly the same are merged. As soon as unstyled text
    or another style lies between them, two fragments are kept even when the style values at both ends are equal.
    """
    entries = _visible_entries(document)
    if end is None:
        end = len(entries)
    start, end = _normalize_range(entries, start, end, expand_empty=expand_empty)
    if start >= end:
        return []

    normalized_styles: dict[int, dict] = {}
    node_ranges: dict[int, tuple[int, int]] = {}
    for index, entry in enumerate(entries):
        if entry.node is None or entry.char == "\n":
            continue
        node_id = id(entry.node)
        node_start, node_end = node_ranges.get(node_id, (index, index + 1))
        node_ranges[node_id] = (min(node_start, index), max(node_end, index + 1))
    segments: list[StyledTextSegment] = []
    current: dict[str, Any] | None = None

    def finish_current() -> None:
        nonlocal current
        if current is None:
            return
        segments.append(
            StyledTextSegment(
                start=current["start"],
                end=current["end"],
                text="".join(current["chars"]),
                style=copy.deepcopy(current["style"]),
                node_type=current["node_type"],
                ruby_text=current["ruby_text"],
                node_start=current["node_start"],
                node_end=current["node_end"],
            )
        )
        current = None

    for index in range(start, end):
        entry = entries[index]
        if entry.char == "\n":
            finish_current()
            continue

        style_id = id(entry.style)
        style = normalized_styles.get(style_id)
        if style is None:
            style = normalize_text_style(entry.style)
            normalized_styles[style_id] = style

        node = entry.node if isinstance(entry.node, dict) else None
        node_type = node.get("type") if node is not None else None
        ruby_text = _runs_text(node.get("text", [])) if node_type == "ruby" else ""
        node_range = node_ranges.get(id(node)) if node is not None else None
        if not style and node_type not in {"ruby", "tcy"}:
            finish_current()
            continue

        can_merge = (
            current is not None
            and current["end"] == index
            and current["style"] == style
            and current["node_type"] == node_type
            and current["ruby_text"] == ruby_text
            and (
                node_type is None
                or (current["node_start"], current["node_end"]) == node_range
            )
        )
        if not can_merge:
            finish_current()
            current = {
                "start": index,
                "end": index + 1,
                "chars": [entry.char],
                "style": style,
                "node_type": node_type,
                "ruby_text": ruby_text,
                "node_start": node_range[0] if node_range else None,
                "node_end": node_range[1] if node_range else None,
            }
        else:
            current["end"] = index + 1
            current["chars"].append(entry.char)

    finish_current()
    return segments


def selected_range_from_editor(text_edit) -> tuple[int, int]:
    cursor = text_edit.textCursor()
    return utf16_range_to_python_range(
        text_edit.toPlainText(),
        cursor.selectionStart(),
        cursor.selectionEnd(),
    )


def style_for_range(document: dict, start: int, end: int) -> dict:
    entries = _visible_entries(document)
    start, end = _normalize_range(entries, start, end, expand_empty=True)
    styles = _styles_in_range(entries, start, end)
    result: dict[str, Any] = {}
    ruby_texts = _ruby_texts_in_range(entries, start, end)
    if ruby_texts:
        result["ruby"] = True
        result["rubyText"] = ruby_texts[0]
    if any(
        isinstance(entry.node, dict) and entry.node.get("type") == "tcy"
        for entry in entries[start:end]
    ):
        result["tcy"] = True
    for style in styles:
        if not isinstance(style, dict):
            continue
        for key in (
            "bold",
            "italic",
            "underline",
            "strikethrough",
            "scale",
            "emphasis",
            "noTcy",
            "verticalAdvance",
            "kerning",
            "preKerning",
            "lineKerning",
            "nextKerning",
        ):
            if key in style and key not in result:
                result[key] = style.get(key)
        if "color" in style and "color" not in result:
            result["color"] = style.get("color")
        if "fontSize" in style and "fontSize" not in result:
            result["fontSize"] = style.get("fontSize")
        if "fontFamily" in style and "fontFamily" not in result:
            result["fontFamily"] = style.get("fontFamily")
        stroke = style.get("stroke")
        if isinstance(stroke, dict):
            if "color" in stroke and "strokeColor" not in result:
                result["strokeColor"] = stroke.get("color")
            if "width" in stroke and "strokeWidth" not in result:
                result["strokeWidth"] = stroke.get("width")
        outer_stroke = style.get("outerStroke")
        if isinstance(outer_stroke, dict):
            if "color" in outer_stroke and "outerStrokeColor" not in result:
                result["outerStrokeColor"] = outer_stroke.get("color")
            if "width" in outer_stroke and "outerStrokeWidth" not in result:
                result["outerStrokeWidth"] = outer_stroke.get("width")
        glow = style.get("glow")
        if isinstance(glow, dict):
            if "color" in glow and "glowColor" not in result:
                result["glowColor"] = glow.get("color")
            if "blur" in glow and "glowBlur" not in result:
                result["glowBlur"] = glow.get("blur")
        transform = style.get("transform")
        if isinstance(transform, dict):
            for source, target in (
                ("offsetX", "offsetX"),
                ("offsetY", "offsetY"),
                ("rotation", "rotation"),
                ("mirrorX", "mirrorX"),
                ("mirrorY", "mirrorY"),
            ):
                if source in transform and target not in result:
                    result[target] = transform.get(source)
    return result


def style_row_coverage(
    document: dict, start: int, end: int, row_key: str
) -> tuple[bool, bool]:
    """Return (used by any text, used by all text) for a style within the range.

    An empty selection keeps the toolbar's whole-text meaning; line breaks take no part in the coverage calculation.
    """
    entries = _visible_entries(document)
    start, end = _normalize_range(entries, start, end, expand_empty=True)
    visible_entries = [entry for entry in entries[start:end] if entry.char != "\n"]
    if not visible_entries:
        return False, False

    matched = [
        _style_row_value(entry, row_key) is not _UNSET for entry in visible_entries
    ]
    return any(matched), all(matched)


def _styles_in_range(entries: list[_CharEntry], start: int, end: int) -> list[dict]:
    """The styles of the text runs in the range (shared references, read-only), with adjacent duplicates removed, in order of appearance."""
    styles: list[dict] = []
    last_run: Any = _UNSET
    for entry in entries[start:end]:
        if entry.char == "\n":
            continue
        if entry.run is not last_run:
            styles.append(entry.style)
            last_run = entry.run
    return styles


_UNSET = object()
_BOOLEAN_STYLE_ROWS = {
    "B": "bold",
    "U": "underline",
    "ST": "strikethrough",
    "D": "emphasis",
}
_DIRECT_STYLE_ROWS = {
    "C": "color",
    "I": "italic",
    "S": "fontSize",
    "%": "scale",
    "F": "fontFamily",
    "FA": "verticalAdvance",
    "K": "kerning",
    "PK": "preKerning",
    "LK": "lineKerning",
    "NK": "nextKerning",
}
_NESTED_STYLE_ROWS = {"O": "stroke", "G": "glow", "OS": "outerStroke"}


def _ruby_texts_in_range(entries: list[_CharEntry], start: int, end: int) -> list[str]:
    texts: list[str] = []
    seen: set[int] = set()
    for entry in entries[start:end]:
        node = entry.node
        if (
            not (isinstance(node, dict) and node.get("type") == "ruby")
            or id(node) in seen
        ):
            continue
        seen.add(id(node))
        ruby_text = _runs_text(node.get("text", []))
        if ruby_text:
            texts.append(ruby_text)
    return texts


def styled_text_for_key(document: dict, start: int, end: int, row_key: str) -> str:
    entries = _visible_entries(document)
    start, end = _normalize_range(entries, start, end, expand_empty=True)
    matches: list[str] = []
    current_chars: list[str] = []
    current_signature: Any = _UNSET

    def finish_match() -> None:
        nonlocal current_signature
        if not current_chars:
            return
        matches.append(_compact_display_text("".join(current_chars)))
        current_chars.clear()
        current_signature = _UNSET

    for entry in entries[start:end]:
        if entry.char == "\n":
            finish_match()
            continue
        signature = _style_row_value(entry, row_key)
        if signature is _UNSET:
            finish_match()
            continue
        if current_chars and signature != current_signature:
            finish_match()
        current_signature = signature
        current_chars.append(entry.char)
    finish_match()
    return " / ".join(item for item in matches if item)


def _style_row_value(entry: _CharEntry, row_key: str) -> Any:
    """Return a row's canonical value, or ``_UNSET`` when it is not active."""
    node = entry.node if isinstance(entry.node, dict) else None
    if row_key == "R":
        if node is not None and node.get("type") == "ruby":
            return ("ruby", _runs_text(node.get("text", [])))
        return _UNSET
    if row_key == "T":
        return True if node is not None and node.get("type") == "tcy" else _UNSET

    style = entry.style or {}
    boolean_key = _BOOLEAN_STYLE_ROWS.get(row_key)
    if boolean_key is not None:
        return True if style.get(boolean_key) else _UNSET

    direct_key = _DIRECT_STYLE_ROWS.get(row_key)
    if direct_key is not None:
        return style[direct_key] if direct_key in style else _UNSET

    nested_key = _NESTED_STYLE_ROWS.get(row_key)
    if nested_key is not None:
        value = style.get(nested_key)
        return copy.deepcopy(value) if isinstance(value, dict) and value else _UNSET

    transform = style.get("transform")
    if not isinstance(transform, dict):
        return _UNSET
    if row_key == "Rot":
        return transform["rotation"] if "rotation" in transform else _UNSET
    if row_key == "XY":
        if "offsetX" in transform or "offsetY" in transform:
            return (transform.get("offsetX"), transform.get("offsetY"))
        return _UNSET
    if row_key == "WH":
        if "scaleX" in transform or "scaleY" in transform:
            return (transform.get("scaleX"), transform.get("scaleY"))
        return _UNSET
    if row_key in {"M", "MV"}:
        key = "mirrorX" if row_key == "M" else "mirrorY"
        return True if transform.get(key) else _UNSET
    return _UNSET


def _compact_display_text(text: str, limit: int = 12) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def _merge_style(style: dict, patch: dict) -> dict:
    merged = copy.deepcopy(style) if isinstance(style, dict) else {}
    for key, value in patch.items():
        if key in {"stroke", "outerStroke", "glow", "transform"}:
            if value is None:
                merged.pop(key, None)
                continue
            nested = merged.get(key) if isinstance(merged.get(key), dict) else {}
            nested.update({k: v for k, v in value.items() if v is not None})
            for nested_key, nested_value in list(value.items()):
                if nested_value is None:
                    nested.pop(nested_key, None)
            if nested:
                merged[key] = nested
            else:
                merged.pop(key, None)
        elif value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged
