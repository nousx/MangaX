"""Central rich-text rendering ratios.

These values describe layout policy rather than implementation details.  Keep
them here so measurement and painting cannot silently drift apart.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RichTextRenderPolicy:
    horizontal_ruby_size: float = 0.42
    vertical_ruby_size: float = 0.36
    vertical_ruby_side_space: float = 0.45
    emphasis_side_space: float = 0.35
    decoration_gap: float = 0.08
    emphasis_radius: float = 0.055
    vertical_emphasis_offset: float = 0.20
    # Underline: thickness and offset are both ratios of the base font size (the same convention as the stroke's stroke_ratio).
    # underline_offset is the distance from the horizontal baseline to the top edge of the line; vertical_underline_offset
    # is the distance from the body edge of a vertical column to the centre of the line (the same convention as vertical_emphasis_offset,
    # with a smaller value, so the underline falls between the body and the emphasis marks without overlapping).
    underline_thickness: float = 0.06
    underline_offset: float = 0.14
    vertical_underline_offset: float = 0.10
    # Position of the centre of a horizontal strikethrough relative to the baseline; the thickness reuses underline_thickness.
    strikethrough_offset: float = -0.30
    ruby_overflow_ratio: float = 1.20
    # Maximum ink width allowed for a tate-chu-yoko block (in multiples of the base font size); beyond it the whole group is compressed horizontally in proportion.
    # As in the reference implementation (mtu-json-gui): a margin of 1.1 is left, so full-width digits are not squeezed too much.
    tcy_max_width: float = 1.10


RICH_TEXT_POLICY = RichTextRenderPolicy()
