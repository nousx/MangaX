"""Facade of the text_render package.

The public API and the symbols existing consumers depend on are gathered here; the implementation is layered by responsibility:
_shared (tools / cache primitives) → _fonts (Qt font runtime) → _glyphs (glyph rasterisation),
_compose (layer compositing and effects, independent) → _layout (horizontal and vertical layout and envelope geometry) →
_render (the put_text_*/measure_*/calc_* entry points).

Outside consumers (rendering/__init__.py, auto_linebreak, the editor
backend, the server routes, the tests) should only access it through this namespace.
"""
from ..rich_text import (
    RenderSpan,
    RichTextDocument,
    TextStyle,
    ensure_rich_text_document,
    is_rich_text_document,
    normalize_rich_linebreaks,
)
from ._compose import (
    DEFAULT_ITALIC_ANGLE,
    _paste_bitmap,
    _style_font_size,
    add_color,
)
from ._fonts import (
    DEFAULT_FONT_FAMILY,
    _sanitized_font_bytes,
    _state,
    _style_font_scope,
    font_registry_revision,
    load_font_file,
    qt_family_is_ambiguous,
    register_font_file,
    select_hyphenator,
    set_bold,
    set_font,
    legacy_font_family,
    strip_qt_foundry_brackets,
    unregister_font_file,
)
from ._layout import (
    CJK_Compatibility_Forms_translate,
    _build_rich_horizontal_layout,
    _build_rich_vertical_layout,
    _line_metrics,
    _line_surface,
    _measure_horizontal_text_width,
    _rich_horizontal_layout_geometry,
    _rich_vertical_column_positions,
    _rich_vertical_layout_geometry,
    _vertical_base,
    _vertical_char_bitmap_x,
    calc_horizontal_block_height,
    calc_horizontal_line_spacing_px,
    calc_vertical_line_spacing_px,
    get_vertical_char_bitmap_width,
)
from ._render import (
    calc_horizontal,
    calc_vertical,
    calc_vertical_metrics,
    get_char_offset_x,
    get_char_offset_y,
    get_string_height,
    get_string_width,
    measure_rich_text_horizontal,
    measure_rich_text_metrics,
    measure_rich_text_vertical,
    put_text_horizontal,
    put_text_vertical,
)
