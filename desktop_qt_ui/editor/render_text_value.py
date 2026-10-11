"""Shared text selection helpers for editor render and measurement paths."""

from __future__ import annotations

from typing import Any

from manga_translator.rendering.rich_text import (
    has_content,
    has_legacy_line_breaks,
    is_rich_text_document,
    legacy_line_breaks_to_document,
)


def has_renderable_text(value: Any) -> bool:
    # Thin wrapper: the one implementation of "has renderable content" for rich text and plain text is in rich_text.py (F12).
    return has_content(value)


def render_text_value_from_region(region_data: dict) -> Any:
    rich = (
        region_data.get("translation_rich") if isinstance(region_data, dict) else None
    )
    if is_rich_text_document(rich):
        return rich
    return region_data.get("translation", "") if isinstance(region_data, dict) else ""


def render_text_value_from_text_block(text_block) -> Any:
    """Return the canonical value shared by editor measurement and drawing."""
    if hasattr(text_block, "get_translation_for_rendering"):
        value = text_block.get_translation_for_rendering()
        if not has_renderable_text(value):
            # A region that is not translated yet (detection/OCR only) falls back to a preview of the original text instead of a blank canvas (F29)
            value = getattr(text_block, "text", "")
    else:
        value = getattr(text_block, "translation", "") or getattr(
            text_block, "text", ""
        )
    if isinstance(value, str) and has_legacy_line_breaks(value):
        return legacy_line_breaks_to_document(value).to_dict()
    return value
