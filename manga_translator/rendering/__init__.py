import base64
import math
import os
import re
from typing import List, Optional, Tuple

import cv2

# import logging
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

from ..config import Config, Renderer

# Only the Qt off-screen renderer is used
from ..utils import (
    TextBlock,
    build_region_reference_mask as _build_region_reference_mask,
    fg_bg_compare,
    get_logger,
    rotate_polygons,
)
from . import text_render
from .auto_linebreak import (
    _is_chinese_lang,
    _solve_unified_no_br_layout,
    solve_no_br_layout,
    should_force_no_wrap_single_region,
    strip_linebreak_edge_punctuation,
)
from .chinese_linebreak import (
    BubbleLinebreakEvaluation,
    append_chinese_linebreak_debug_record,
    build_chinese_linebreak_debug_snapshot,
    bubble_mask_overflow_pixels,
    choose_chinese_bubble_linebreak_with_trace,
    download_chinese_linebreak_models_if_enabled,
)
from .text_replacement_layout import prepare_text_replacements_for_layout, sync_translation_raw_from_layout
from .auto_linebreak import unwrapped_translation
from .text_render_eng import apply_manga2eng_line_breaks
from .rich_text import (
    ensure_rich_text_document,
    has_content as rich_text_has_content,
    is_rich_text_document,
    plain_equivalent_text,
    plain_text_of,
)
from manga_translator.utils.swallowed import note_ignored_error

logger = get_logger('render')

# Base font size, used to simulate text blocks
BASE_FONT_SIZE = 100


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _encode_mask_png_base64(mask: Optional[np.ndarray]) -> str:
    if mask is None:
        return ""
    try:
        mask_u8 = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
        ok, buffer = cv2.imencode(".png", mask_u8)
        if not ok:
            return ""
        return base64.b64encode(buffer).decode("ascii")
    except Exception as ignored_error:
        note_ignored_error(ignored_error, "manga_translator/rendering/__init__.py:_encode_mask_png_base64")
        return ""


def _resolve_effective_stroke_width(
    config: Config = None,
    stroke_width: float = None,
) -> float:
    """Resolve the one stroke ratio consumed by measurement and drawing."""
    render_cfg = getattr(config, 'render', None) if config is not None else None
    if bool(getattr(render_cfg, 'disable_font_border', False)):
        return 0.0
    if stroke_width is not None:
        return max(_safe_float(stroke_width, 0.0), 0.0)
    return max(_safe_float(getattr(render_cfg, 'stroke_width', 0.07), 0.07), 0.0)


def _estimate_effect_padding(
    font_size: int,
    config: Config = None,
    stroke_width: float = None,
) -> float:
    """Estimate the extra edge pixels that text effects (mainly the stroke, for now) add."""
    if font_size <= 0:
        return 0.0

    stroke_ratio = _resolve_effective_stroke_width(config, stroke_width)
    if stroke_ratio <= 0.0:
        return 0.0

    # Kept in line with the bg_size calculation in text_render.py
    return float(max(int(font_size * stroke_ratio), 1))


def _has_explicit_line_breaks(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r'(\[BR\]|【BR】|<br\s*/?>|\r\n|\r|\n)', text or '', flags=re.IGNORECASE))


def _rich_text_has_content(value) -> bool:
    # F12: thin wrapper around rich_text.has_content (at this layer only rich-text documents are considered)
    return is_rich_text_document(value) and rich_text_has_content(value)


def _translation_preview(value, limit: int = 80) -> str:
    # F12: thin wrapper around rich_text.plain_text_of (any translation value -> plain text)
    return plain_text_of(value)[:limit]


def _region_render_value(region: TextBlock):
    if hasattr(region, 'get_translation_for_rendering'):
        return region.get_translation_for_rendering()
    return getattr(region, 'translation', '')


def _resolve_region_stroke_width(region: TextBlock, config: Config = None) -> float:
    """Return the exact region stroke ratio consumed by layout and drawing."""
    render_cfg = getattr(config, 'render', None) if config is not None else None
    if bool(getattr(render_cfg, 'disable_font_border', False)):
        return 0.0
    return max(_safe_float(getattr(region, 'stroke_width', 0.0), 0.0), 0.0)


def _should_apply_default_english_line_break_method(region: TextBlock, config: Config = None) -> bool:
    if is_rich_text_document(_region_render_value(region)):
        return False
    if not isinstance(getattr(region, 'translation', ''), str):
        return False
    # With the switch on, it applies to every language (forced horizontal + bubble layout)
    render_cfg = getattr(config, 'render', None) if config is not None else None
    bubble_layout = bool(getattr(render_cfg, 'bubble_layout_english', False)) if render_cfg is not None else False
    if bubble_layout:
        return not _has_explicit_line_breaks(getattr(region, 'translation', ''))
    # Default behaviour: only for horizontal English text
    return (
        str(getattr(region, 'target_lang', '') or '').upper() == 'ENG'
        and _resolve_region_render_horizontal(region)
        and not _has_explicit_line_breaks(getattr(region, 'translation', ''))
    )


def _apply_default_english_case_preferences(region: TextBlock, config: Config = None) -> bool:
    if is_rich_text_document(_region_render_value(region)):
        return False
    if not isinstance(getattr(region, 'translation', ''), str):
        return False
    if str(getattr(region, 'target_lang', '') or '').upper() != 'ENG':
        return False
    if not _resolve_region_render_horizontal(region):
        return False

    render_cfg = getattr(config, 'render', None) if config is not None else None
    uppercase = bool(getattr(render_cfg, 'uppercase', False)) if render_cfg is not None else False
    lowercase = bool(getattr(render_cfg, 'lowercase', False)) if render_cfg is not None else False

    original_translation = str(getattr(region, 'translation', '') or '')
    updated_translation = original_translation
    if uppercase:
        updated_translation = original_translation.upper()
    elif lowercase:
        updated_translation = original_translation.lower()

    if updated_translation == original_translation:
        return False

    region.translation = updated_translation
    return True


def _apply_default_english_line_break_method(
    region: TextBlock,
    target_font_size: int,
    original_img: np.ndarray = None,
    config: Config = None,
) -> bool:
    if not _should_apply_default_english_line_break_method(region, config):
        return False

    # With the switch on, force horizontal text
    render_cfg = getattr(config, 'render', None) if config is not None else None
    bubble_layout = bool(getattr(render_cfg, 'bubble_layout_english', False)) if render_cfg is not None else False
    if bubble_layout:
        # Force it to horizontal
        region._direction = 'h'

    applied = apply_manga2eng_line_breaks(
        region,
        original_img=original_img,
        seed_font_size=target_font_size,
        config=config,
        letter_spacing=_resolve_letter_spacing_multiplier(region, config),
    )
    if applied:
        logger.debug("[BUBBLE LAYOUT] Applied bubble-based line breaking (force horizontal)")
    return applied


def calc_text_block_dimensions(text: str, is_horizontal: bool, line_spacing: float = 1.0,
                                config: Config = None, target_lang: str = None,
                                font_size: int = BASE_FONT_SIZE, letter_spacing: float = 1.0,
                                stroke_width: float = None) -> tuple:
    """
    Simulate rendering a text block at the given font size and return its exact pixel size

    Measuring and rendering share one envelope geometry (the normalised convention of put_text_*): a plain string
    is turned into a single-style richtext paragraph at the measure entry, and measured box == size of the rendered surface.

    Args:
        text: the text. In a plain string, [BR]/<br>/【BR】/line breaks are normalised into paragraphs.
        is_horizontal: True = horizontal, False = vertical
        line_spacing: line spacing multiplier
        config: the configuration object
        target_lang: the target language

    Returns:
        (base_width, base_height, n_lines) - the base size and the number of lines or columns
    """
    _ = target_lang
    base_font = max(1, int(font_size))
    if is_horizontal:
        return text_render.measure_rich_text_horizontal(
            base_font,
            text,
            line_spacing,
            config=config,
            stroke_width=stroke_width,
            letter_spacing=letter_spacing,
        )
    return text_render.measure_rich_text_vertical(
        base_font,
        text,
        line_spacing,
        config=config,
        stroke_width=stroke_width,
        letter_spacing=letter_spacing,
    )


def calc_font_from_box(width: float, height: float, text: str, is_horizontal: bool,
                       line_spacing: float = 1.0, config: Config = None,
                       target_lang: str = None, letter_spacing: float = 1.0,
                       stroke_width: float = None) -> int:
    """
    Box -> font: binary search, on real measurements, for the largest font size that fits

    Args:
        width: box width (pixels)
        height: box height (pixels)
        text: the text
        is_horizontal: True = horizontal, False = vertical
        line_spacing: line spacing multiplier
        config: the configuration object
        target_lang: the target language

    Returns:
        The largest font size that fits in the box (pixels)
    """
    if width <= 0 or height <= 0:
        return 1

    if is_rich_text_document(text):
        if not _rich_text_has_content(text):
            return 1
        # F24: parse once and pass the instance down, so the dict is not parsed again inside the bisection
        text = ensure_rich_text_document(text)
    else:
        text = (text or '').strip()
        if not text:
            return 1

    def _fits(fs: int) -> bool:
        req_w, req_h, _, _ = calc_box_from_font(
            fs,
            text,
            is_horizontal,
            line_spacing,
            config,
            target_lang,
            center=None,
            angle=0,
            letter_spacing=letter_spacing,
            stroke_width=stroke_width,
        )
        return req_w <= width and req_h <= height

    if not _fits(1):
        return 1

    lo = 1
    hi = max(1, int(min(width, height)))
    hi = min(hi, 8192)

    while hi < 8192 and _fits(hi):
        lo = hi
        hi = min(hi * 2, 8192)
        if hi == lo:
            break

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _fits(mid):
            lo = mid
        else:
            hi = mid - 1

    return max(1, int(lo))


def _select_preserved_line_layout_font(
    base_font_size: int,
    width: float,
    height: float,
    text: str,
    is_horizontal: bool,
    line_spacing: float = 1.0,
    config: Config = None,
    target_lang: str = None,
    letter_spacing: float = 1.0,
    stroke_width: float = None,
) -> Tuple[int, int]:
    base_font = max(int(base_font_size), 1)
    line_font = max(
        int(
            calc_font_from_box(
                width=width,
                height=height,
                text=text,
                is_horizontal=is_horizontal,
                line_spacing=line_spacing,
                config=config,
                target_lang=target_lang,
                letter_spacing=letter_spacing,
                stroke_width=stroke_width,
            )
        ),
        1,
    )
    return max(base_font, line_font), line_font






def calc_text_block_metrics(text, is_horizontal: bool, line_spacing: float,
                            config: Config = None, target_lang: str = None,
                            font_size: int = None, letter_spacing: float = 1.0,
                            stroke_width: float = None) -> tuple:
    """Size + body centre: calc_text_block_dimensions plus the centre point of the body box.

    Returns:
        (base_width, base_height, n_lines, body_center)
        body_center is the centre of the body box inside the render box (relative to its top-left corner).
        Plain text has no decoration outside the box, so it is always the exact centre of the render box.
    """
    _ = target_lang
    base_font = max(1, int(font_size))
    metrics = text_render.measure_rich_text_metrics(
        base_font, text, is_horizontal, line_spacing,
        config=config, stroke_width=stroke_width, letter_spacing=letter_spacing,
    )
    return metrics['width'], metrics['height'], metrics['n_lines'], metrics['body_center']


def calc_box_from_font(font_size: int, text: str, is_horizontal: bool,
                       line_spacing: float = 1.0, config: Config = None,
                       target_lang: str = None, center: tuple = None,
                       angle: float = 0, letter_spacing: float = 1.0,
                       stroke_width: float = None) -> tuple:
    """
    Font -> box: measure the pixel size of the text directly at the target font size and give the body centre

    Args:
        font_size: font size (pixels)
        text: the text
        is_horizontal: True = horizontal, False = vertical
        line_spacing: line spacing multiplier
        config: the configuration object
        target_lang: the target language
        center: centre point (cx, cy); when given, dst_points is returned
        angle: rotation angle (degrees), only used when center is not None

    Returns:
        center is None: (required_width, required_height, n_lines, body_center)
            body_center = the centre of the body box inside the render box (relative to its top-left corner).
            For horizontal text it is computed from the actual ink of the main text, and decorations such as
            ruby/emphasis can move it off the exact centre of the render box; for vertical text it shifts
            because of decorations such as the ruby of the first column.
        center is not None: (dst_points, body_center_world)
            dst_points has shape (1, 4, 2), and the render box has center as its exact centre;
            body_center_world = the body centre in world coordinates (already rotated by angle).
            (None, None) is returned when the text is empty.
    """
    font_size = max(1, int(font_size))
    base_w, base_h, n_lines, (body_x, body_y) = calc_text_block_metrics(
        text, is_horizontal, line_spacing, config, target_lang,
        font_size=font_size, letter_spacing=letter_spacing,
        stroke_width=stroke_width,
    )

    if base_w <= 0 or base_h <= 0:
        if center is not None:
            return None, None
        return 0, 0, 0, (0.0, 0.0)

    # Measure at the target font size directly, instead of scaling linearly from the base size
    req_width = math.ceil(base_w)
    req_height = math.ceil(base_h)
    body_x = float(body_x)
    body_y = float(body_y)

    # For horizontal text the stroke and rich-text effects are part of the real ink plan; vertical text still uses the outer padding.
    effect_padding = 0.0 if is_horizontal else _estimate_effect_padding(
        font_size,
        config,
        stroke_width,
    )
    if effect_padding > 0.0:
        pad_total = int(effect_padding * 2.0)
        req_width += pad_total
        req_height += pad_total
        body_x += pad_total / 2.0
        body_y += pad_total / 2.0

    # Without a centre point, return the size and the body centre (coordinates inside the box)
    if center is None:
        return req_width, req_height, n_lines, (body_x, body_y)

    # With a centre point, build dst_points
    cx, cy = center
    half_w = req_width / 2
    half_h = req_height / 2

    # The four corners of the unrotated rectangle
    unrotated_points = np.array([
        [cx - half_w, cy - half_h],
        [cx + half_w, cy - half_h],
        [cx + half_w, cy + half_h],
        [cx - half_w, cy + half_h]
    ], dtype=np.float32)
    # Body centre (unrotated world coordinates): top-left corner of the box + coordinates inside the box
    unrotated_body = np.array([
        [cx - half_w + body_x, cy - half_h + body_y],
    ] * 4, dtype=np.float32)

    # Apply the rotation (the body point goes through exactly the same transform as the corners, so the conventions agree)
    if angle != 0:
        dst_points = rotate_polygons(
            center, unrotated_points.reshape(1, -1),
            -angle, to_int=False
        ).reshape(-1, 4, 2)
        rotated_body = rotate_polygons(
            center, unrotated_body.reshape(1, -1),
            -angle, to_int=False
        ).reshape(-1, 4, 2)
        body_world = (float(rotated_body[0, 0, 0]), float(rotated_body[0, 0, 1]))
    else:
        dst_points = unrotated_points.reshape(-1, 4, 2)
        body_world = (float(unrotated_body[0, 0]), float(unrotated_body[0, 1]))

    return dst_points, body_world

def find_largest_inscribed_rect(mask: np.ndarray) -> tuple:
    """Return the largest axis-aligned rectangle fully covered by ``mask``.

    Crop to the foreground bounds and combine consecutive identical rows and
    columns before running the histogram/stack algorithm. Group weights retain
    the original pixel dimensions; neither the mask nor the result is scaled.
    """
    binary = np.asarray(mask) > 0
    if binary.ndim != 2 or not np.any(binary):
        return 0, 0, 0, 0

    offset_x, offset_y, roi_width, roi_height = cv2.boundingRect(binary.astype(np.uint8))
    roi = binary[offset_y:offset_y + roi_height, offset_x:offset_x + roi_width]
    row_starts = np.r_[0, np.flatnonzero(np.any(roi[1:] != roi[:-1], axis=1)) + 1]
    row_ends = np.r_[row_starts[1:], roi_height]
    row_weights = row_ends - row_starts
    selected_rows = roi[row_starts]
    col_starts = np.r_[
        0,
        np.flatnonzero(np.any(selected_rows[:, 1:] != selected_rows[:, :-1], axis=0)) + 1,
    ]
    col_bounds = col_starts.tolist() + [roi_width]
    compact = selected_rows[:, col_starts]
    height, width = compact.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_rect = (0, 0, 0, 0)

    for y in range(height):
        # A group's last row dominates its earlier rows. Identical columns can
        # likewise be evaluated at their full width. Keep scan order and strict
        # area updates so equal-area rectangles retain the original tie break.
        heights = np.where(compact[y], heights + int(row_weights[y]), 0)
        stack: list[int] = []
        for x in range(width + 1):
            current_height = int(heights[x]) if x < width else 0
            while stack and current_height < int(heights[stack[-1]]):
                bar_x = stack.pop()
                rect_height = int(heights[bar_x])
                left = stack[-1] + 1 if stack else 0
                rect_width = col_bounds[x] - col_bounds[left]
                area = rect_width * rect_height
                if area <= best_area:
                    continue
                best_area = area
                best_rect = (
                    offset_x + col_bounds[left],
                    offset_y + int(row_ends[y]) - rect_height,
                    rect_width,
                    rect_height,
                )
            stack.append(x)

    return best_rect


def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    if path:
        parsed = path.split(',')
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed

def count_text_length(text: str) -> float:
    """Calculate text length, treating っッぁぃぅぇぉ as 0.5 characters"""
    half_width_chars = 'っッぁぃぅぇぉ'  
    length = 0.0
    for char in text.strip():
        if char in half_width_chars:
            length += 0.5
        else:
            length += 1.0
    return length

def generate_line_break_combinations(text: str):
    """
    Generate line break combinations using a smart pruning strategy.
    
    Strategy:
    1. For small n (<=10): Use exhaustive search (original algorithm)
    2. For medium n (11-20): Use beam search with top-k pruning
    3. For large n (>20): Use greedy + sampling strategy
    
    This balances quality and performance.
    """
    import itertools
    import random
    
    # Standardize all break markers to [BR] (including full-width brackets)
    text = re.sub(r'\s*(<br>|【BR】)\s*', '[BR]', text, flags=re.IGNORECASE)
    
    # Find all [BR] positions
    breaks = []
    pattern = r'\[BR\]'
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        breaks.append((match.start(), match.end()))
    
    if not breaks:
        return [(text, "no_breaks", None)]
    
    n_breaks = len(breaks)
    combinations = []
    
    # Strategy 1: Small n - exhaustive search (original algorithm)
    if n_breaks <= 10:
        logger.debug(f"[OPTIMIZE_LINE_BREAKS] Using exhaustive search (n={n_breaks})")
        
        # Add original (keep all breaks)
        combinations.append((text, "all_breaks", None))
        
        # Generate all possible combinations
        for r in range(1, n_breaks + 1):
            for combo in itertools.combinations(range(n_breaks), r):
                segments = re.split(pattern, text, flags=re.IGNORECASE)
                
                if 0 in combo and len(segments[0].strip()) <= 2:
                    combinations.append((None, f"remove_{combo}", "first_segment_too_short"))
                    continue
                
                modified_text = text
                for idx in sorted(combo, reverse=True):
                    start, end = breaks[idx]
                    modified_text = modified_text[:start] + modified_text[end:]
                
                combinations.append((modified_text, f"remove_{combo}", None))
        
        return combinations
    
    # Strategy 2: Medium n - beam search with sampling
    elif n_breaks <= 20:
        logger.debug(f"[OPTIMIZE_LINE_BREAKS] Using beam search (n={n_breaks})")
        
        combinations.append((text, "all_breaks", None))
        
        # Sample combinations: all singles, all pairs, some triples, and remove_all
        # Singles: remove each break individually
        for i in range(n_breaks):
            if i == 0:
                segments = re.split(pattern, text, flags=re.IGNORECASE)
                if len(segments[0].strip()) <= 2:
                    combinations.append((None, f"remove_({i},)", "first_segment_too_short"))
                    continue
            
            modified_text = text
            start, end = breaks[i]
            modified_text = modified_text[:start] + modified_text[end:]
            combinations.append((modified_text, f"remove_({i},)", None))
        
        # Pairs: remove adjacent breaks
        for i in range(n_breaks - 1):
            if i == 0:
                segments = re.split(pattern, text, flags=re.IGNORECASE)
                if len(segments[0].strip()) <= 2:
                    continue
            
            modified_text = text
            for idx in [i+1, i]:
                start, end = breaks[idx]
                if idx == i+1:
                    modified_text = modified_text[:start] + modified_text[end:]
                else:
                    # Recalculate position after first removal
                    offset = breaks[i+1][1] - breaks[i+1][0]
                    start -= offset
                    end -= offset
                    modified_text = modified_text[:start] + modified_text[end:]
            
            combinations.append((modified_text, f"remove_({i},{i+1})", None))
        
        # Sample some triples (every 3rd combination)
        for r in [3]:
            sampled = list(itertools.combinations(range(n_breaks), r))
            # Sample at most 20 combinations
            if len(sampled) > 20:
                sampled = random.sample(sampled, 20)
            
            for combo in sampled:
                if 0 in combo:
                    segments = re.split(pattern, text, flags=re.IGNORECASE)
                    if len(segments[0].strip()) <= 2:
                        continue
                
                modified_text = text
                for idx in sorted(combo, reverse=True):
                    start, end = breaks[idx]
                    modified_text = modified_text[:start] + modified_text[end:]
                
                combinations.append((modified_text, f"remove_{combo}", None))
        
        # Remove all
        modified_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        combinations.append((modified_text, "remove_all", None))
        
        return combinations
    
    # Strategy 3: Large n - greedy + sampling
    else:
        logger.debug(f"[OPTIMIZE_LINE_BREAKS] Using greedy+sampling (n={n_breaks})")
        
        combinations.append((text, "all_breaks", None))
        
        # Singles: sample every 2nd break
        for i in range(0, n_breaks, 2):
            if i == 0:
                segments = re.split(pattern, text, flags=re.IGNORECASE)
                if len(segments[0].strip()) <= 2:
                    continue
            
            modified_text = text
            start, end = breaks[i]
            modified_text = modified_text[:start] + modified_text[end:]
            combinations.append((modified_text, f"remove_({i},)", None))
        
        # Pairs: sample every 3rd adjacent pair
        for i in range(0, n_breaks - 1, 3):
            if i == 0:
                segments = re.split(pattern, text, flags=re.IGNORECASE)
                if len(segments[0].strip()) <= 2:
                    continue
            
            modified_text = text
            for idx in [i+1, i]:
                start, end = breaks[idx]
                if idx == i+1:
                    modified_text = modified_text[:start] + modified_text[end:]
                else:
                    offset = breaks[i+1][1] - breaks[i+1][0]
                    start -= offset
                    end -= offset
                    modified_text = modified_text[:start] + modified_text[end:]
            
            combinations.append((modified_text, f"remove_({i},{i+1})", None))
        
        # Remove all
        modified_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        combinations.append((modified_text, "remove_all", None))
        
        return combinations

def calculate_uniformity(lines: List[str]) -> float:
    """
    Calculate uniformity score for line lengths.
    Lower score = more uniform (better).
    Uses coefficient of variation (std/mean).
    """
    if not lines or len(lines) <= 1:
        return 0.0
    
    lengths = [len(line.strip()) for line in lines]
    if not lengths or sum(lengths) == 0:
        return float('inf')
    
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    # Coefficient of variation
    cv = std_length / mean_length if mean_length > 0 else float('inf')
    return cv

def optimize_line_breaks_for_region(region: TextBlock, config: Config, target_font_size: int, bubble_width: float, bubble_height: float):
    """
    Optimize line breaks for a single region by testing all combinations.
    Returns the best text variant and the font size it achieves.
    """
    original_translation = region.translation
    combinations = generate_line_break_combinations(original_translation)
    
    best_text = original_translation
    best_font_size = 0
    best_uniformity = float('inf')
    
    layout_mode = config.render.layout_mode if config and hasattr(config.render, 'layout_mode') else 'default'
    logger.debug(f"[OPTIMIZE_LINE_BREAKS] Testing {len(combinations)} combinations, layout_mode={layout_mode}")
    render_horizontally = _resolve_region_render_horizontal(region)
    
    for text_variant, combo_desc, skip_reason in combinations:
        if skip_reason:
            logger.debug(f"[OPTIMIZE_LINE_BREAKS] Skipping {combo_desc}: {skip_reason}")
            continue
        
        # Convert [BR] to \n for calculation
        text_for_calc = re.sub(r'\s*\[BR\]\s*', '\n', text_variant, flags=re.IGNORECASE)
        
        # Strict smart scaling: if removing every line break (no \n) makes the text box grow, this option is rejected
        strict_smart_scaling = getattr(config.render, 'strict_smart_scaling', False) if config and hasattr(config, 'render') else False
        if layout_mode == 'smart_scaling' and strict_smart_scaling:
            if '\n' not in text_for_calc:
                logger.debug(f"[OPTIMIZE_LINE_BREAKS] Skipping {combo_desc}: removing line breaks would expand the text box in strict smart scaling mode")
                continue
        
        try:
            line_spacing_multiplier = _resolve_line_spacing_multiplier(region, config)
            letter_spacing_multiplier = _resolve_letter_spacing_multiplier(region, config)
            # Calculate required dimensions
            if render_horizontally:
                lines, widths = text_render.calc_horizontal(
                    target_font_size, text_for_calc, 
                    max_width=99999, max_height=99999, 
                    language=region.target_lang,
                    letter_spacing=letter_spacing_multiplier
                )
                if widths:
                    spacing_y = text_render.calc_horizontal_line_spacing_px(
                        target_font_size,
                        line_spacing_multiplier,
                    )
                    required_width = max(widths)
                    required_height = target_font_size * len(lines) + spacing_y * max(0, len(lines) - 1)
                else:
                    continue
            else:  # Vertical
                lines, heights, line_widths = text_render.calc_vertical_metrics(
                    target_font_size,
                    text_for_calc,
                    max_height=99999,
                    config=config,
                    letter_spacing=letter_spacing_multiplier,
                )
                if heights:
                    spacing_x = int(target_font_size * 0.2 * line_spacing_multiplier)
                    required_height = max(heights)
                    required_width = sum(line_widths) + spacing_x * max(0, len(lines) - 1)
                else:
                    continue
            
            # Calculate how much the text fits in the bubble
            # Larger font size is better
            width_ratio = bubble_width / required_width if required_width > 0 else 1.0
            height_ratio = bubble_height / required_height if required_height > 0 else 1.0
            fit_ratio = min(width_ratio, height_ratio)
            
            # Calculate effective font size for this combination
            effective_font_size = target_font_size * fit_ratio
            
            # Calculate uniformity
            uniformity = calculate_uniformity(lines)
            
            logger.debug(f"[OPTIMIZE_LINE_BREAKS] {combo_desc}: font_size={effective_font_size:.1f}, uniformity={uniformity:.3f}")
            
            # Choose the best: prioritize font size, then uniformity
            is_better = False
            if effective_font_size > best_font_size + 0.5:  # Significantly larger font
                is_better = True
            elif abs(effective_font_size - best_font_size) <= 0.5:  # Similar font size
                if uniformity < best_uniformity:  # Better uniformity
                    is_better = True
            
            if is_better:
                best_text = text_variant
                best_font_size = effective_font_size
                best_uniformity = uniformity
                logger.debug(f"[OPTIMIZE_LINE_BREAKS] New best: {combo_desc}")
        
        except Exception as e:
            logger.warning(f"[OPTIMIZE_LINE_BREAKS] Error evaluating {combo_desc}: {e}")
            continue
    
    # Compare and log optimization results
    # Count with one regular expression that matches every BR variant
    br_pattern = r'(\[BR\]|【BR】|<br>)'
    original_br_count = len(re.findall(br_pattern, original_translation, flags=re.IGNORECASE))
    optimized_br_count = len(re.findall(br_pattern, best_text, flags=re.IGNORECASE))
    
    # Apply the optimisation only when the number of BRs really changed
    if optimized_br_count != original_br_count:
        br_change = optimized_br_count - original_br_count
        if br_change > 0:
            change_desc = f"added {br_change}"
        elif br_change < 0:
            change_desc = f"removed {-br_change}"
        else:
            change_desc = "repositioned"
        logger.debug(f"[AI Line Break Font Scaling] Optimization complete: {change_desc} line breaks; font size increased to {best_font_size:.1f}px")
        logger.debug(f"[AI Line Break Font Scaling] Original text: {original_translation}")
        logger.debug(f"[AI Line Break Font Scaling] Optimized text: {best_text}")
        return best_text, best_font_size
    else:
        logger.debug(f"[AI Line Break Font Scaling] No optimization applied: the original line breaks are optimal; font size {best_font_size:.1f}px")
        # Even with the same count, return the normalised text (full-width to half-width)
        return best_text, best_font_size

def _resolve_region_render_horizontal(region: TextBlock) -> bool:
    forced_direction = region._direction if hasattr(region, '_direction') else region.direction
    if forced_direction != 'auto':
        if forced_direction in ['horizontal', 'h']:
            return True
        if forced_direction in ['vertical', 'v']:
            return False
    return region.horizontal


def _polygon_fully_inside_mask(points: np.ndarray, bubble_mask: np.ndarray) -> bool:
    if points is None or points.size == 0 or bubble_mask is None:
        return False
    h, w = bubble_mask.shape[:2]
    if h <= 0 or w <= 0:
        return False

    pts = np.asarray(points, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return False
    pts[:, 0] = np.clip(pts[:, 0], 0, max(w - 1, 0))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(h - 1, 0))

    x, y, box_w, box_h = cv2.boundingRect(pts)
    poly_mask = np.zeros((box_h, box_w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [pts - (x, y)], 255)
    poly_pixels = poly_mask > 0
    if not np.any(poly_pixels):
        return False
    return bool(np.all(bubble_mask[y:y + box_h, x:x + box_w][poly_pixels] > 0))


def _region_lines_fully_inside_mask(region: TextBlock, bubble_mask: np.ndarray) -> bool:
    lines = np.asarray(region.lines)
    if lines.size == 0:
        return False

    if lines.ndim == 2 and lines.shape[1] == 8:
        polys = lines.reshape(-1, 4, 2)
    elif lines.ndim == 3 and lines.shape[1:] == (4, 2):
        polys = lines
    else:
        return False

    for poly in polys:
        if not _polygon_fully_inside_mask(poly, bubble_mask):
            return False
    return True


def _resolve_line_spacing_multiplier(region: TextBlock, config: Config) -> float:
    region_line_spacing = getattr(region, 'line_spacing', None)
    if isinstance(region_line_spacing, (int, float)) and region_line_spacing > 0:
        return float(region_line_spacing)
    cfg_val = config.render.line_spacing if config and hasattr(config, 'render') else None
    if isinstance(cfg_val, (int, float)) and cfg_val > 0:
        return float(cfg_val)
    return 1.0


def _resolve_letter_spacing_multiplier(region: TextBlock, config: Config) -> float:
    region_letter_spacing = getattr(region, 'letter_spacing', None)
    if isinstance(region_letter_spacing, (int, float)) and region_letter_spacing > 0:
        return float(region_letter_spacing)
    cfg_val = config.render.letter_spacing if config and hasattr(config, 'render') else None
    if isinstance(cfg_val, (int, float)) and cfg_val > 0:
        return float(cfg_val)
    return 1.0


def _resolve_configured_min_font_size(config: Config) -> int:
    render_cfg = getattr(config, 'render', None) if config is not None else None
    raw_min_font_size = getattr(render_cfg, 'font_size_minimum', 0) if render_cfg is not None else 0
    if isinstance(raw_min_font_size, (int, float)) and raw_min_font_size > 0:
        return max(int(raw_min_font_size), 1)
    return 0


def _resolve_configured_fixed_font_size(config: Config) -> int:
    render_cfg = getattr(config, 'render', None) if config is not None else None
    raw_font_size = getattr(render_cfg, 'font_size', None) if render_cfg is not None else None
    if isinstance(raw_font_size, (int, float)) and raw_font_size > 0:
        return max(int(raw_font_size), 1)
    return 0


def _balloon_fill_mask_layout_enabled(config: Config) -> bool:
    render_config = getattr(config, 'render', None) if config is not None else None
    return bool(getattr(render_config, 'balloon_fill_mask_layout', False))


def _polygons_overlap(left_points: np.ndarray, right_points: np.ndarray, min_area: float = 0.5) -> bool:
    """Return whether two convex render boxes share positive area."""
    left = np.asarray(left_points, dtype=np.float32).reshape(-1, 2)
    right = np.asarray(right_points, dtype=np.float32).reshape(-1, 2)
    if left.shape[0] < 3 or right.shape[0] < 3:
        return False
    try:
        intersection_area, _ = cv2.intersectConvexConvex(left, right)
    except cv2.error:
        return False
    return float(intersection_area) > float(min_area)


def _layout_regions_conflict(
    points: np.ndarray,
    bubble_mask: Optional[np.ndarray],
    placed_regions: list[tuple[np.ndarray, Optional[np.ndarray]]],
) -> bool:
    """Check render-box collisions; this is opt-in because masks may merge bubbles."""
    for placed_points, placed_mask in placed_regions:
        if _polygons_overlap(points, placed_points):
            return True
    return False


def _shrink_font_for_layout_collisions(
    region: TextBlock,
    start_font_size: int,
    min_font_size: int,
    render_horizontally: bool,
    line_spacing_multiplier: float,
    letter_spacing_multiplier: float,
    config: Config,
    anchor_mode: str,
    bubble_mask: Optional[np.ndarray],
    placed_regions: list[tuple[np.ndarray, Optional[np.ndarray]]],
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """Find the largest smaller font whose render box avoids prior bubbles."""
    for font_size in range(max(int(start_font_size), 1), max(int(min_font_size), 1) - 1, -1):
        points = _calc_region_dst_points_for_font(
            region=region,
            font_size=font_size,
            render_horizontally=render_horizontally,
            line_spacing_multiplier=line_spacing_multiplier,
            letter_spacing_multiplier=letter_spacing_multiplier,
            config=config,
            anchor_mode=anchor_mode,
        )
        if points is not None and not _layout_regions_conflict(points, bubble_mask, placed_regions):
            return font_size, points
    return None, None


def _resolve_initial_layout_font_size(region: TextBlock, img: np.ndarray, config: Config) -> int:
    region_font_size = getattr(region, 'font_size', 0)
    if isinstance(region_font_size, (int, float)) and region_font_size > 0:
        return max(int(region_font_size), 1)

    if img is not None and hasattr(img, 'shape') and len(img.shape) >= 2:
        return max(round((img.shape[0] + img.shape[1]) / 200), 1)
    return 24


def _resolve_balloon_fill_search_font_size(
    preferred_font_size: int,
    target_font_size: int,
    line_box_width: float,
    line_box_height: float,
    bubble_width: float,
    bubble_height: float,
) -> int:
    """Return a bounded upper limit spanning both OCR and bubble geometry."""
    candidates = (
        preferred_font_size,
        target_font_size,
        line_box_width,
        line_box_height,
        bubble_width,
        bubble_height,
    )
    valid_candidates = []
    for value in candidates:
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0:
            valid_candidates.append(int(value))
    return max(1, min(max(valid_candidates, default=1), 8192))


def _apply_final_font_constraints(layout_font_size: int, config: Config) -> int:
    final_font_size = max(int(layout_font_size), 1)
    render_cfg = getattr(config, 'render', None) if config is not None else None

    configured_font_size = _resolve_configured_fixed_font_size(config)
    if configured_font_size > 0:
        final_font_size = configured_font_size

    font_size_offset = getattr(render_cfg, 'font_size_offset', 0) if render_cfg is not None else 0
    if isinstance(font_size_offset, (int, float)) and font_size_offset != 0:
        final_font_size = max(int(final_font_size + float(font_size_offset)), 1)

    font_scale_ratio = getattr(render_cfg, 'font_scale_ratio', 1.0) if render_cfg is not None else 1.0
    if not isinstance(font_scale_ratio, (int, float)) or font_scale_ratio <= 0:
        font_scale_ratio = 1.0
    final_font_size = max(int(final_font_size * float(font_scale_ratio)), 1)

    configured_min_font_size = _resolve_configured_min_font_size(config)
    if configured_min_font_size > 0:
        final_font_size = max(final_font_size, configured_min_font_size)

    max_font_size = getattr(render_cfg, 'max_font_size', 0) if render_cfg is not None else 0
    if isinstance(max_font_size, (int, float)) and max_font_size > 0:
        final_font_size = min(final_font_size, int(max_font_size))

    return max(final_font_size, 1)


def _resolve_strict_layout_font_size(
    region: TextBlock,
    config: Config,
    layout_candidate_font_size: int,
    box_fit_font_size: int,
) -> int:
    """Font size of the strict layout: the font size fitted to the OCR box is the layout limit for the final text.

    When box_fit_font_size <= 0 (no box-fitted result was obtained) the candidate font size is used directly,
    and the font size setting is still applied in one place by the outer code. In replace-translation mode, regions forced to a single line
    (a single OCR line whose direction was not rewritten) are exempt from the box limit: the line-break markers are removed and the text is rendered
    at the candidate font size, and may extend beyond the OCR box.
    """
    min_shrink_font_size = 8
    is_replace_mode = config.cli.replace_translation if (config and hasattr(config, 'cli')) else False
    force_single_line_no_wrap = is_replace_mode and should_force_no_wrap_single_region(region)
    if is_replace_mode and len(region.lines) == 1 and not force_single_line_no_wrap:
        logger.debug("[STRICT MODE] Direction override detected for a single-line region in replacement mode; allowing automatic line wrapping")
    if force_single_line_no_wrap:
        logger.debug("[STRICT MODE] Disabling line wrapping for a single-line region in replacement mode (OCR lines=1); rendering at the candidate font size")
        region.translation = re.sub(r'(\n|\[BR\]|【BR】|<br>)', '', region.translation, flags=re.IGNORECASE)
        return max(int(layout_candidate_font_size), min_shrink_font_size)
    if isinstance(box_fit_font_size, (int, float)) and box_fit_font_size > 0:
        return max(min(int(layout_candidate_font_size), int(box_fit_font_size)), min_shrink_font_size)
    return max(int(layout_candidate_font_size), min_shrink_font_size)


def _resolve_balloon_fill_fallback_font_size(
    config: Config,
    layout_candidate_font_size: int,
    box_fit_font_size: int,
    original_region_font_size: int,
    lines_fully_enclosed: bool,
    region: Optional[TextBlock] = None,
) -> int:
    """Choose the partial-mask fallback while keeping legacy behavior opt-in."""
    if _balloon_fill_mask_layout_enabled(config) and not lines_fully_enclosed:
        return max(int(layout_candidate_font_size), int(original_region_font_size), 1)
    if region is not None:
        return _resolve_strict_layout_font_size(
            region=region,
            config=config,
            layout_candidate_font_size=layout_candidate_font_size,
            box_fit_font_size=box_fit_font_size,
        )
    min_shrink_font_size = 8
    if isinstance(box_fit_font_size, (int, float)) and box_fit_font_size > 0:
        return max(min(int(layout_candidate_font_size), int(box_fit_font_size)), min_shrink_font_size)
    return max(int(layout_candidate_font_size), min_shrink_font_size)




def _compute_top_aligned_center(region: 'TextBlock', text_height: float) -> tuple:
    """Shift center toward bubble top so text is top-aligned within the bubble."""
    pts = region.min_rect  # (1, 4, 2)
    mid = (pts[:, [1, 2, 3, 0]] + pts) / 2
    top_mid = mid[0, 0]
    bot_mid = mid[0, 2]
    bubble_h = float(np.linalg.norm(bot_mid - top_mid))
    if bubble_h <= 0 or text_height >= bubble_h:
        return tuple(region.center)
    up = (top_mid - bot_mid) / bubble_h
    shift = (bubble_h - text_height) / 2.0
    nc = np.array(region.center, dtype=float) + shift * up
    return (float(nc[0]), float(nc[1]))


def _resolve_layout_anchor_mode(*, apply_bubble_centering: bool, skip_font_scaling: bool = False) -> str:
    """One strategy for the centre anchor.

    - skip_font_scaling: a layout authorised by the editor - region.center is the centre of the editor's white box (the render box)
      and is placed with the meaning "render box centre" (center_box, no body shift), pixel for pixel the same
      as the editor preview
    - normal rendering: the pipeline computes the anchor itself, with the meaning "where the body centre should be" - center when
      centring in the bubble applies, otherwise top
    """
    if skip_font_scaling:
        return 'center_box'
    return 'center' if apply_bubble_centering else 'top'


def _resolve_region_layout_center(
    region: TextBlock,
    font_size: int,
    render_horizontally: bool,
    line_spacing_multiplier: float,
    letter_spacing_multiplier: float,
    config: Config,
    anchor_mode: str = 'top',
    render_value=None,
) -> tuple:
    if anchor_mode in ('center', 'center_box'):
        return tuple(region.center)
    if anchor_mode != 'top':
        raise ValueError(f"Unsupported anchor_mode: {anchor_mode!r}")

    if render_value is None:
        render_value = _region_render_value(region)
    _, req_h, _, (_, body_y) = calc_box_from_font(
        int(max(font_size, 1)),
        render_value,
        render_horizontally,
        line_spacing_multiplier,
        config,
        region.target_lang,
        letter_spacing=letter_spacing_multiplier,
        stroke_width=_resolve_region_stroke_width(region, config),
    )
    # Top alignment uses the height of the body text (twice the distance from the body centre to the bottom edge), leaving out ruby and emphasis marks outside the box,
    # so the top of the body meets the top of the bubble, rather than the top of the ruby.
    body_height = 2.0 * (req_h - body_y)
    return _compute_top_aligned_center(region, body_height)


def _calc_region_dst_points_for_font(
    region: TextBlock,
    font_size: int,
    render_horizontally: bool,
    line_spacing_multiplier: float,
    letter_spacing_multiplier: float,
    config: Config,
    anchor_mode: str = 'top',
) -> Optional[np.ndarray]:
    # F24: the rich-text render value is parsed once here; the anchor calculation and the dst calculation (and every
    # step of the mask bisection) reuse the same instance, so the dict is not parsed again.
    render_value = _region_render_value(region)
    if is_rich_text_document(render_value):
        render_value = ensure_rich_text_document(render_value)
    # anchor is the world coordinate where the "body centre" should land (for plain text the body centre is the box centre).
    anchor = _resolve_region_layout_center(
        region=region,
        font_size=font_size,
        render_horizontally=render_horizontally,
        line_spacing_multiplier=line_spacing_multiplier,
        letter_spacing_multiplier=letter_spacing_multiplier,
        config=config,
        anchor_mode=anchor_mode,
        render_value=render_value,
    )
    dst_points, body_world = calc_box_from_font(
        int(max(font_size, 1)),
        render_value,
        render_horizontally,
        line_spacing_multiplier,
        config,
        region.target_lang,
        center=anchor,
        angle=region.angle,
        letter_spacing=letter_spacing_multiplier,
        stroke_width=_resolve_region_stroke_width(region, config),
    )
    if dst_points is None:
        return None
    # calc_box_from_font puts the "render box centre" at anchor. An anchor the pipeline computes itself (top/center)
    # means "where the body centre should be": the whole box is shifted so the body centre lands on anchor - for plain text
    # body_world == anchor and delta = 0, exactly as before; for rich text the ruby and emphasis marks
    # are pushed outside the box while the body stays anchored. center_box (a centre authorised by the editor) keeps the render box
    # centre meaning and is not shifted, so it lines up with the editor preview.
    if anchor_mode == 'center_box':
        return dst_points
    delta_x = float(anchor[0]) - float(body_world[0])
    delta_y = float(anchor[1]) - float(body_world[1])
    if delta_x or delta_y:
        dst_points = dst_points + np.array([delta_x, delta_y], dtype=dst_points.dtype)
    return dst_points


def _font_size_fits_bubble_mask(
    region: TextBlock,
    font_size: int,
    render_horizontally: bool,
    line_spacing_multiplier: float,
    letter_spacing_multiplier: float,
    config: Config,
    bubble_mask: np.ndarray,
    anchor_mode: str = 'top',
) -> Tuple[bool, Optional[np.ndarray]]:
    dst_points = _calc_region_dst_points_for_font(
        region=region,
        font_size=font_size,
        render_horizontally=render_horizontally,
        line_spacing_multiplier=line_spacing_multiplier,
        letter_spacing_multiplier=letter_spacing_multiplier,
        config=config,
        anchor_mode=anchor_mode,
    )
    if dst_points is None or dst_points.size == 0:
        return False, None
    fits = _polygon_fully_inside_mask(np.asarray(dst_points[0]), bubble_mask)
    return fits, dst_points


def _binary_search_font_for_bubble_mask(
    region: TextBlock,
    start_font_size: int,
    min_font_size: int,
    render_horizontally: bool,
    line_spacing_multiplier: float,
    letter_spacing_multiplier: float,
    config: Config,
    bubble_mask: np.ndarray,
    anchor_mode: str = 'top',
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    lo = max(int(min_font_size), 1)
    hi = max(int(start_font_size), lo)
    best_font: Optional[int] = None
    best_dst_points: Optional[np.ndarray] = None

    while lo <= hi:
        mid = (lo + hi) // 2
        fits, dst_points = _font_size_fits_bubble_mask(
            region=region,
            font_size=mid,
            render_horizontally=render_horizontally,
            line_spacing_multiplier=line_spacing_multiplier,
            letter_spacing_multiplier=letter_spacing_multiplier,
            config=config,
            bubble_mask=bubble_mask,
            anchor_mode=anchor_mode,
        )
        if fits:
            best_font = mid
            best_dst_points = dst_points
            lo = mid + 1
        else:
            hi = mid - 1

    return best_font, best_dst_points


def _finalize_region_font_sizes(
    text_regions: List[TextBlock],
    dst_points_list: list,
    anchor_modes: List[Optional[str]],
    config: Config,
    *,
    skip_font_scaling: bool = False,
    debug_img: Optional[np.ndarray] = None,
) -> None:
    """Apply global size settings once, then measure with the final rich-text styles."""
    for index, (region, anchor_mode) in enumerate(zip(text_regions, anchor_modes)):
        if region is None or anchor_mode is None:
            continue
        if not skip_font_scaling:
            region.font_size = _apply_final_font_constraints(region.font_size, config)

        config._current_region = region
        text_render.set_font(getattr(region, 'font_family', '') or text_render.DEFAULT_FONT_FAMILY)
        try:
            points = _calc_region_dst_points_for_font(
                region=region,
                font_size=region.font_size,
                render_horizontally=_resolve_region_render_horizontal(region),
                line_spacing_multiplier=_resolve_line_spacing_multiplier(region, config),
                letter_spacing_multiplier=_resolve_letter_spacing_multiplier(region, config),
                config=config,
                anchor_mode=anchor_mode,
            )
        except Exception as exc:
            logger.exception(f"Final font measurement failed for region {index}: {exc}")
            points = None
        if points is not None:
            dst_points_list[index] = points

        if debug_img is not None and dst_points_list[index] is not None:
            polygon = np.asarray(dst_points_list[index]).reshape(-1, 2).astype(np.int32)
            if polygon.shape[0] >= 4:
                cv2.polylines(debug_img, [polygon], True, (0, 255, 0), 2)
                cv2.putText(
                    debug_img,
                    f'B{index}:{region.font_size}',
                    tuple(polygon[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )


def resize_regions_to_font_size(
    img: np.ndarray,
    text_regions: List['TextBlock'],
    config: Config,
    original_img: np.ndarray = None,
    return_debug_img: bool = False,
    skip_font_scaling: bool = False,
    skip_text_replacements: bool = False,
    bubble_mask: Optional[np.ndarray] = None,
):
    """Finish automatic layout before applying global font-size settings."""
    dst_points_list, anchor_modes, debug_img = _layout_regions_to_font_size(
        img,
        text_regions,
        config,
        original_img,
        return_debug_img,
        skip_font_scaling=skip_font_scaling,
        skip_text_replacements=skip_text_replacements,
        bubble_mask=bubble_mask,
    )
    _finalize_region_font_sizes(
        text_regions,
        dst_points_list,
        anchor_modes,
        config,
        skip_font_scaling=skip_font_scaling,
        debug_img=debug_img,
    )
    if return_debug_img and debug_img is not None:
        return dst_points_list, debug_img
    return dst_points_list


def _layout_region_with_fixed_font(
    *,
    anchor_modes,
    config,
    dst_points_list,
    img,
    region,
    region_idx,
    skip_anchor_mode,
):
    """Use the font size stored on the region and only work out where its text goes."""
    anchor_modes[region_idx] = skip_anchor_mode
    fixed_font_size = region.font_size if region.font_size > 0 else round((img.shape[0] + img.shape[1]) / 200)
    logger.debug(f"[RESIZE] skip_font_scaling: region {region_idx} uses fixed font size {fixed_font_size}")

    # Compute the text box directly from the fixed font size
    # The forced direction override has to be taken into account (the same test as in render())
    actual_horizontal = _resolve_region_render_horizontal(region)

    line_spacing_multiplier = _resolve_line_spacing_multiplier(region, config)
    letter_spacing_multiplier = _resolve_letter_spacing_multiplier(region, config)

    dst_points = _calc_region_dst_points_for_font(
        region=region,
        font_size=fixed_font_size,
        render_horizontally=actual_horizontal,
        line_spacing_multiplier=line_spacing_multiplier,
        letter_spacing_multiplier=letter_spacing_multiplier,
        config=config,
        anchor_mode=skip_anchor_mode,
    )

    if dst_points is None:
        dst_points = region.min_rect

    region.font_size = fixed_font_size
    dst_points_list.append(dst_points)
    return


def _layout_rich_text_region(
    *,
    config,
    dst_points_list,
    layout_candidate_font_size,
    layout_min_font_size,
    letter_spacing_multiplier,
    line_box_height,
    line_box_width,
    line_spacing_multiplier,
    lines_fully_enclosed,
    mode,
    normal_anchor_mode,
    original_img,
    region,
    region_bubble_mask,
    region_idx,
    render_horizontally,
):
    """Fit a rich-text region by shrinking its font; its line structure is never rearranged."""
    # A rich-text document cannot be rearranged: no line-break optimisation or automatic wrapping (they would break the structured paragraphs
    # and the style boundaries), but font size fitting must still apply - the estimated size is no longer used as it is.
    # Parse once and pass the instance down, so the dict is not parsed again inside the font size bisection.
    rich_render_value = ensure_rich_text_document(_region_render_value(region))
    # 1) The largest font size that fits the unrotated bounding box, min with the estimated size (shrink only)
    box_fit_font_size = calc_font_from_box(
        width=float(line_box_width),
        height=float(line_box_height),
        text=rich_render_value,
        is_horizontal=render_horizontally,
        line_spacing=line_spacing_multiplier,
        config=config,
        target_lang=region.target_lang,
        letter_spacing=letter_spacing_multiplier,
        stroke_width=_resolve_region_stroke_width(region, config),
    )
    layout_font_size = max(
        min(layout_candidate_font_size, int(box_fit_font_size)),
        layout_min_font_size,
    )

    # 2) balloon_fill: shrink further with the bubble mask (_calc_region_dst_points_for_font
    #    already anchors the body of rich text); when the region is not fully inside the mask, keep the box-fitted result
    if mode == 'balloon_fill' and original_img is not None:
        try:
            if (
                region_bubble_mask is not None
                and np.count_nonzero(region_bubble_mask) > 0
                and lines_fully_enclosed
            ):
                best_font_size, _ = _binary_search_font_for_bubble_mask(
                    region=region,
                    start_font_size=layout_font_size,
                    min_font_size=layout_min_font_size,
                    render_horizontally=render_horizontally,
                    line_spacing_multiplier=line_spacing_multiplier,
                    letter_spacing_multiplier=letter_spacing_multiplier,
                    config=config,
                    bubble_mask=region_bubble_mask,
                    anchor_mode=normal_anchor_mode,
                )
                if best_font_size is not None:
                    layout_font_size = int(best_font_size)
        except Exception as exc:
            logger.warning(
                f"balloon_fill rich-text mask shrink failed for region {region_idx}: {exc}"
            )

    final_font_size = layout_font_size
    dst_points = _calc_region_dst_points_for_font(
        region=region,
        font_size=final_font_size,
        render_horizontally=render_horizontally,
        line_spacing_multiplier=line_spacing_multiplier,
        letter_spacing_multiplier=letter_spacing_multiplier,
        config=config,
        anchor_mode=normal_anchor_mode,
    )
    if dst_points is None:
        dst_points = region.min_rect

    region.font_size = final_font_size
    dst_points_list.append(dst_points)
    return


def _break_region_lines(
    *,
    bubble_layout_rect,
    config,
    has_br,
    layout_box_height,
    layout_box_width,
    layout_candidate_font_size,
    layout_min_font_size,
    letter_spacing_multiplier,
    line_box_height,
    line_box_width,
    line_spacing_multiplier,
    lines_fully_enclosed,
    no_br_source_text,
    region,
    region_bubble_mask,
    region_layout_mode,
    remove_linebreak_punctuation,
    render_horizontally,
):
    """Decide the line breaks of a region. Returns (layout box width, layout box height, bubble layout rect)."""
    if has_br:
        if remove_linebreak_punctuation:
            region.translation = strip_linebreak_edge_punctuation(region.translation)
        if config.render.optimize_line_breaks and (region_layout_mode != 'strict' or config.render.disable_auto_wrap):
            optimized_text, _ = optimize_line_breaks_for_region(
                region,
                config,
                layout_candidate_font_size,
                float(line_box_width),
                float(line_box_height),
            )
            region.translation = optimized_text
            if remove_linebreak_punctuation:
                region.translation = strip_linebreak_edge_punctuation(region.translation)
    else:
        mask_layout_active = (
            region_layout_mode == 'balloon_fill'
            and _balloon_fill_mask_layout_enabled(config)
        )
        mask_layout_applied = False
        if mask_layout_active and lines_fully_enclosed:
            _mask_x, _mask_y, layout_width, layout_height = find_largest_inscribed_rect(
                region_bubble_mask
            )
            bubble_layout_rect = (
                int(_mask_x), int(_mask_y), int(layout_width), int(layout_height)
            )

            if layout_width > 0 and layout_height > 0:
                layout_box_width = float(layout_width)
                layout_box_height = float(layout_height)
                line_layout_max_font_size = int(
                    max(layout_candidate_font_size, layout_box_width, layout_box_height, layout_min_font_size)
                )
                region.translation = _solve_unified_no_br_layout(
                    text=no_br_source_text,
                    render_horizontally=render_horizontally,
                    target_font_size=layout_candidate_font_size,
                    bubble_width=layout_box_width,
                    bubble_height=layout_box_height,
                    layout_min_font_size=layout_min_font_size,
                    line_spacing_multiplier=line_spacing_multiplier,
                    letter_spacing_multiplier=letter_spacing_multiplier,
                    config=config,
                    target_lang=region.target_lang,
                    max_font_size=line_layout_max_font_size,
                )
                mask_layout_applied = True

        # No usable bubble rectangle: keep automatic line breaking
        # against the OCR box, including text outside detected bubbles.
        if not mask_layout_applied:
            layout_box_width = float(line_box_width)
            layout_box_height = float(line_box_height)
            line_layout_max_font_size = int(
                max(layout_candidate_font_size, layout_box_width, layout_box_height, layout_min_font_size)
            )

            region.translation = _solve_unified_no_br_layout(
                text=region.translation,
                render_horizontally=render_horizontally,
                target_font_size=layout_candidate_font_size,
                bubble_width=layout_box_width,
                bubble_height=layout_box_height,
                layout_min_font_size=layout_min_font_size,
                line_spacing_multiplier=line_spacing_multiplier,
                letter_spacing_multiplier=letter_spacing_multiplier,
                config=config,
                target_lang=region.target_lang,
                max_font_size=line_layout_max_font_size,
            )
    return layout_box_width, layout_box_height, bubble_layout_rect


def _layout_region_balloon_fill(
    *,
    anchor_modes,
    box_fit_font_size,
    bubble_layout_rect,
    candidate_n,
    candidate_required_height,
    candidate_required_width,
    config,
    debug_img,
    dst_points_list,
    has_br,
    layout_candidate_font_size,
    layout_min_font_size,
    letter_spacing_multiplier,
    line_box_height,
    line_box_width,
    line_spacing_multiplier,
    lines_fully_enclosed,
    no_br_source_text,
    normal_anchor_mode,
    original_img,
    original_region_font_size,
    placed_regions,
    region,
    region_bubble_mask,
    region_idx,
    render_horizontally,
    target_font_size,
):
    """balloon_fill: fit the text of a region to the speech bubble that encloses it."""
    semantic_linebreak_debug = (
        bool(getattr(config.render, 'semantic_linebreak', False))
        and _is_chinese_lang(getattr(region, 'target_lang', '') or '')
    )
    if not semantic_linebreak_debug:
        logger.debug(f"=== balloon_fill mode activated for region {region_idx} ===")
        logger.debug(f"OCR box (xywh): {region.xywh}")
    min_font_size = layout_min_font_size

    if original_img is None:
        logger.warning("balloon_fill mode requires original_img, fallback to strict layout")
        fallback_font_size = _resolve_strict_layout_font_size(
            region=region,
            config=config,
            layout_candidate_font_size=layout_candidate_font_size,
            box_fit_font_size=box_fit_font_size,
        )
        fallback_dst_points = _calc_region_dst_points_for_font(
            region=region,
            font_size=fallback_font_size,
            render_horizontally=_resolve_region_render_horizontal(region),
            line_spacing_multiplier=_resolve_line_spacing_multiplier(region, config),
            letter_spacing_multiplier=_resolve_letter_spacing_multiplier(region, config),
            config=config,
            anchor_mode=normal_anchor_mode,
        )
        if fallback_dst_points is None:
            fallback_dst_points = region.min_rect
        region.font_size = fallback_font_size
        dst_points_list.append(fallback_dst_points)
        return

    try:
        chosen_dst_points = None
        chosen_font_size = int(max(target_font_size, layout_min_font_size))
        overflow_candidate_dst_points = None
        bubble_w = 0
        bubble_h = 0
        line_budget = 0.0
        search_bubble_width = 0
        search_bubble_height = 0

        if not lines_fully_enclosed:
            # The bubble mask is invalid or the region is not fully enclosed by a bubble: fall back to the strict layout.
            chosen_font_size = _resolve_balloon_fill_fallback_font_size(
                region=region,
                config=config,
                layout_candidate_font_size=layout_candidate_font_size,
                box_fit_font_size=box_fit_font_size,
                original_region_font_size=original_region_font_size,
                lines_fully_enclosed=lines_fully_enclosed,
            )
            if not semantic_linebreak_debug:
                logger.debug(f"balloon_fill region {region_idx}: not fully enclosed, fallback to strict")
        else:
            if (
                bool(getattr(config.render, 'semantic_linebreak', False))
                and _is_chinese_lang(getattr(region, 'target_lang', '') or '')
                and np.count_nonzero(region_bubble_mask) > 0
            ):
                _bubble_x, _bubble_y, bubble_w, bubble_h = find_largest_inscribed_rect(region_bubble_mask)
                line_budget = float(bubble_w if render_horizontally else bubble_h)
            if _balloon_fill_mask_layout_enabled(config) and has_br:
                _bubble_x, _bubble_y, search_bubble_width, search_bubble_height = cv2.boundingRect(
                    region_bubble_mask
                )
                normal_anchor_mode = 'center'
                anchor_modes[region_idx] = normal_anchor_mode


            if has_br:
                if not semantic_linebreak_debug:
                    logger.debug(
                        f"balloon_fill region {region_idx}: keep explicit breaks, "
                        f"candidate font={layout_candidate_font_size}, "
                        f"required={candidate_required_width:.1f}x{candidate_required_height:.1f}"
                    )
            else:
                if not semantic_linebreak_debug:
                    logger.debug(
                        f"balloon_fill region {region_idx}: unified no_br layout, "
                        f"result_segments={candidate_n}, font={layout_candidate_font_size}, "
                        f"required={candidate_required_width:.1f}x{candidate_required_height:.1f}"
                    )

            preferred_font_size = int(max(layout_candidate_font_size, layout_min_font_size))

            # For debugging: record the "out of range candidate box" (a larger font size candidate that fails the mask constraint)
            preferred_fits = False
            preferred_dst_points = _calc_region_dst_points_for_font(
                region=region,
                font_size=preferred_font_size,
                render_horizontally=render_horizontally,
                line_spacing_multiplier=line_spacing_multiplier,
                letter_spacing_multiplier=letter_spacing_multiplier,
                config=config,
                anchor_mode=normal_anchor_mode,
            )
            if preferred_dst_points is not None and preferred_dst_points.size > 0:
                preferred_fits = _polygon_fully_inside_mask(np.asarray(preferred_dst_points[0]), region_bubble_mask)
                if not preferred_fits:
                    overflow_candidate_dst_points = preferred_dst_points

            if (
                semantic_linebreak_debug
                and not has_br
                and bubble_w > 0
                and bubble_h > 0
                and line_budget > 0
            ):
                single_width, single_height, _, _ = calc_box_from_font(
                    preferred_font_size,
                    no_br_source_text,
                    render_horizontally,
                    line_spacing_multiplier,
                    config,
                    region.target_lang,
                    center=None,
                    angle=0,
                    letter_spacing=letter_spacing_multiplier,
                    stroke_width=_resolve_region_stroke_width(region, config),
                )
                total_budget = float(single_width if render_horizontally else single_height)
                linebreak_snapshot = build_chinese_linebreak_debug_snapshot(
                    no_br_source_text,
                    font_size=preferred_font_size,
                    target_segments=candidate_n,
                    total_budget=total_budget,
                    line_budget=line_budget,
                    horizontal=render_horizontally,
                    letter_spacing=letter_spacing_multiplier,
                )

                original_candidate_text = region.translation

                def evaluate_chinese_candidate(candidate_text: str) -> Optional[BubbleLinebreakEvaluation]:
                    region.translation = candidate_text
                    req_w, req_h, req_n, _ = calc_box_from_font(
                        preferred_font_size,
                        candidate_text,
                        render_horizontally,
                        line_spacing_multiplier,
                        config,
                        region.target_lang,
                        center=None,
                        angle=0,
                        letter_spacing=letter_spacing_multiplier,
                        stroke_width=_resolve_region_stroke_width(region, config),
                    )
                    candidate_dst_points = _calc_region_dst_points_for_font(
                        region=region,
                        font_size=preferred_font_size,
                        render_horizontally=render_horizontally,
                        line_spacing_multiplier=line_spacing_multiplier,
                        letter_spacing_multiplier=letter_spacing_multiplier,
                        config=config,
                        anchor_mode=normal_anchor_mode,
                    )
                    if candidate_dst_points is None or candidate_dst_points.size == 0:
                        return None
                    return BubbleLinebreakEvaluation(
                        text_with_br=candidate_text,
                        required_width=float(req_w),
                        required_height=float(req_h),
                        n_segments=int(req_n),
                        dst_points=candidate_dst_points,
                        overflow_pixels=bubble_mask_overflow_pixels(candidate_dst_points, region_bubble_mask),
                    )

                try:
                    semantic_choice = choose_chinese_bubble_linebreak_with_trace(
                        source_text=no_br_source_text,
                        current_text=region.translation,
                        font_size=preferred_font_size,
                        target_segments=candidate_n,
                        total_budget=total_budget,
                        line_budget=line_budget,
                        horizontal=render_horizontally,
                        letter_spacing=letter_spacing_multiplier,
                        evaluate=evaluate_chinese_candidate,
                    )
                finally:
                    region.translation = original_candidate_text

                if semantic_choice is not None and semantic_choice.selected is not None:
                    chosen_semantic_candidate = semantic_choice.selected
                    expected_candidate_n = candidate_n
                    region.translation = chosen_semantic_candidate.text_with_br
                    layout_candidate_font_size = preferred_font_size
                    candidate_required_width = chosen_semantic_candidate.required_width
                    candidate_required_height = chosen_semantic_candidate.required_height
                    candidate_n = chosen_semantic_candidate.n_segments
                    preferred_dst_points = chosen_semantic_candidate.dst_points
                    preferred_fits = chosen_semantic_candidate.fits
                    overflow_candidate_dst_points = None if preferred_fits else chosen_semantic_candidate.dst_points
                    append_chinese_linebreak_debug_record(
                        config,
                        {
                            "stage": "bubble_mask_choice",
                            "region_index": region_idx,
                            "input": no_br_source_text,
                            "current_candidate": original_candidate_text,
                            "direction": "h" if render_horizontally else "v",
                            "font_size": preferred_font_size,
                            "target_segments": expected_candidate_n,
                            "ocr_box_xywh": np.asarray(region.xywh).tolist() if getattr(region, "xywh", None) is not None else None,
                            "ocr_box_size": {"width": float(line_box_width), "height": float(line_box_height)},
                            "bubble_inscribed_rect": {
                                "width": float(bubble_w),
                                "height": float(bubble_h),
                                "line_budget": float(line_budget),
                            },
                            "single_line_required": {"width": float(single_width), "height": float(single_height)},
                            "total_budget": float(total_budget),
                            "mask": {
                                "encoding": "png_base64",
                                "width": int(region_bubble_mask.shape[1]) if region_bubble_mask is not None else 0,
                                "height": int(region_bubble_mask.shape[0]) if region_bubble_mask is not None else 0,
                                "nonzero_pixels": int(np.count_nonzero(region_bubble_mask)) if region_bubble_mask is not None else 0,
                                "data": _encode_mask_png_base64(region_bubble_mask),
                            },
                            "linebreak_snapshot": linebreak_snapshot,
                            "selected": {
                                "text_with_br": chosen_semantic_candidate.text_with_br,
                                "segments": int(chosen_semantic_candidate.n_segments),
                                "required": {
                                    "width": float(chosen_semantic_candidate.required_width),
                                    "height": float(chosen_semantic_candidate.required_height),
                                },
                                "fits": bool(chosen_semantic_candidate.fits),
                                "overflow_pixels": int(chosen_semantic_candidate.overflow_pixels),
                                "dst_points": np.asarray(chosen_semantic_candidate.dst_points).tolist()
                                if chosen_semantic_candidate.dst_points is not None
                                else None,
                            },
                            "candidate_evaluations": semantic_choice.evaluations,
                            "candidates": [
                                {
                                    "rank": rank,
                                    "score": list(score),
                                    "selected": candidate.text_with_br == chosen_semantic_candidate.text_with_br,
                                    "text_with_br": candidate.text_with_br,
                                    "segments": int(candidate.n_segments),
                                    "semantic_penalty": int(score[1]),
                                    "required": {
                                        "width": float(candidate.required_width),
                                        "height": float(candidate.required_height),
                                    },
                                    "fits": bool(candidate.fits),
                                    "overflow_pixels": int(candidate.overflow_pixels),
                                    "dst_points": np.asarray(candidate.dst_points).tolist()
                                    if candidate.dst_points is not None
                                    else None,
                                }
                                for rank, (score, candidate) in enumerate(semantic_choice.candidates, start=1)
                            ],
                        },
                    )
                else:
                    append_chinese_linebreak_debug_record(
                        config,
                        {
                            "stage": "bubble_mask_choice",
                            "region_index": region_idx,
                            "input": no_br_source_text,
                            "current_candidate": original_candidate_text,
                            "direction": "h" if render_horizontally else "v",
                            "font_size": preferred_font_size,
                            "target_segments": candidate_n,
                            "ocr_box_xywh": np.asarray(region.xywh).tolist() if getattr(region, "xywh", None) is not None else None,
                            "bubble_inscribed_rect": {
                                "width": float(bubble_w),
                                "height": float(bubble_h),
                                "line_budget": float(line_budget),
                            },
                            "single_line_required": {"width": float(single_width), "height": float(single_height)},
                            "total_budget": float(total_budget),
                            "mask": {
                                "encoding": "png_base64",
                                "width": int(region_bubble_mask.shape[1]) if region_bubble_mask is not None else 0,
                                "height": int(region_bubble_mask.shape[0]) if region_bubble_mask is not None else 0,
                                "nonzero_pixels": int(np.count_nonzero(region_bubble_mask)) if region_bubble_mask is not None else 0,
                                "data": _encode_mask_png_base64(region_bubble_mask),
                            },
                            "linebreak_snapshot": linebreak_snapshot,
                            "selected": None,
                            "candidate_evaluations": semantic_choice.evaluations if semantic_choice is not None else [],
                            "candidates": [],
                        },
                    )

            best_font_size, best_dst_points = _binary_search_font_for_bubble_mask(
                region=region,
                start_font_size=(
                    _resolve_balloon_fill_search_font_size(
                        preferred_font_size=preferred_font_size,
                        target_font_size=target_font_size,
                        line_box_width=line_box_width,
                        line_box_height=line_box_height,
                        bubble_width=search_bubble_width,
                        bubble_height=search_bubble_height,
                    )
                    if _balloon_fill_mask_layout_enabled(config) and has_br
                    else preferred_font_size
                ),
                min_font_size=min_font_size,
                render_horizontally=render_horizontally,
                line_spacing_multiplier=line_spacing_multiplier,
                letter_spacing_multiplier=letter_spacing_multiplier,
                config=config,
                bubble_mask=region_bubble_mask,
                anchor_mode=normal_anchor_mode,
            )
            if best_font_size is not None and best_dst_points is not None:
                chosen_font_size = int(best_font_size)
                chosen_dst_points = best_dst_points
                if not semantic_linebreak_debug:
                    logger.debug(
                        f"balloon_fill region {region_idx}: enclosed lines, binary-search font {preferred_font_size}->{chosen_font_size}"
                    )
            else:
                chosen_font_size = int(max(min_font_size, 1))
                chosen_dst_points = _calc_region_dst_points_for_font(
                    region=region,
                    font_size=chosen_font_size,
                    render_horizontally=render_horizontally,
                    line_spacing_multiplier=line_spacing_multiplier,
                    letter_spacing_multiplier=letter_spacing_multiplier,
                    config=config,
                    anchor_mode=normal_anchor_mode,
                )
                if chosen_dst_points is None:
                    chosen_font_size = preferred_font_size
                    chosen_dst_points = preferred_dst_points
                if not semantic_linebreak_debug:
                    logger.debug(
                        f"balloon_fill region {region_idx}: no mask-safe layout found, shrink to font={chosen_font_size}"
                    )

        if chosen_dst_points is None:
            chosen_dst_points = region.min_rect

        final_font_size = chosen_font_size
        final_dst_points = _calc_region_dst_points_for_font(
            region=region,
            font_size=final_font_size,
            render_horizontally=render_horizontally,
            line_spacing_multiplier=line_spacing_multiplier,
            letter_spacing_multiplier=letter_spacing_multiplier,
            config=config,
            anchor_mode=normal_anchor_mode,
        )
        if final_dst_points is None:
            final_dst_points = chosen_dst_points

        if _balloon_fill_mask_layout_enabled(config) and placed_regions:
            collision_font_size, collision_dst_points = _shrink_font_for_layout_collisions(
                region=region,
                start_font_size=final_font_size,
                min_font_size=layout_min_font_size,
                render_horizontally=render_horizontally,
                line_spacing_multiplier=line_spacing_multiplier,
                letter_spacing_multiplier=letter_spacing_multiplier,
                config=config,
                anchor_mode=normal_anchor_mode,
                bubble_mask=region_bubble_mask,
                placed_regions=placed_regions,
            )
            if collision_font_size is not None and collision_dst_points is not None:
                if collision_font_size < final_font_size:
                    logger.debug(
                        f"balloon_fill region {region_idx}: collision guard "
                        f"shrinks font {final_font_size}->{collision_font_size}"
                    )
                final_font_size = collision_font_size
                final_dst_points = collision_dst_points

        region.font_size = final_font_size
        chosen_dst_points = final_dst_points
        dst_points_list.append(chosen_dst_points)
        if _balloon_fill_mask_layout_enabled(config):
            placed_regions.append((chosen_dst_points, region_bubble_mask))

        if debug_img is not None:
            ocr_x1, ocr_y1, ocr_w, ocr_h = map(int, region.xywh)
            cv2.rectangle(debug_img, (ocr_x1, ocr_y1), (ocr_x1 + ocr_w, ocr_y1 + ocr_h), (0, 0, 255), 2)
            if bubble_layout_rect is not None:
                bx, by, bw, bh = bubble_layout_rect
                cv2.rectangle(
                    debug_img,
                    (bx, by),
                    (bx + bw - 1, by + bh - 1),
                    (255, 0, 255),
                    2,
                )
                cv2.putText(
                    debug_img,
                    f"B{region_idx}:MASK {bw}x{bh}",
                    (bx, max(12, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    1,
                )

            if np.count_nonzero(region_bubble_mask) > 0:
                component_contours, _ = cv2.findContours(region_bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if component_contours:
                    cv2.drawContours(debug_img, component_contours, -1, (0, 255, 255), 1)

            if overflow_candidate_dst_points is not None:
                overflow_poly = np.asarray(overflow_candidate_dst_points).reshape(-1, 2).astype(np.int32)
                if overflow_poly.shape[0] >= 4:
                    # BGR orange: the candidate box goes beyond the mask and is shrunk or dropped in the end
                    cv2.polylines(debug_img, [overflow_poly], True, (0, 165, 255), 2)
                    cv2.putText(
                        debug_img,
                        f'B{region_idx}:OVR',
                        tuple(overflow_poly[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 165, 255),
                        1,
                    )
    except Exception as e:
        logger.exception(f"Error in balloon_fill layout for region {region_idx}: {e}")
        dst_points_list.append(region.min_rect)
        region.font_size = target_font_size

    return


def _layout_region_strict(
    *,
    box_fit_font_size,
    config,
    dst_points_list,
    layout_candidate_font_size,
    letter_spacing_multiplier,
    line_spacing_multiplier,
    normal_anchor_mode,
    region,
    render_horizontally,
):
    """strict: keep the text of a region inside its detected box."""
    # The same rule with and without BR: the final text uses the font size fitted to the OCR box as the layout limit.
    layout_font_size = _resolve_strict_layout_font_size(
        region=region,
        config=config,
        layout_candidate_font_size=layout_candidate_font_size,
        box_fit_font_size=box_fit_font_size,
    )
    final_font_size = layout_font_size
    dst_points = _calc_region_dst_points_for_font(
        region=region,
        font_size=final_font_size,
        render_horizontally=render_horizontally,
        line_spacing_multiplier=line_spacing_multiplier,
        letter_spacing_multiplier=letter_spacing_multiplier,
        config=config,
        anchor_mode=normal_anchor_mode,
    )
    if dst_points is None:
        dst_points = region.min_rect

    region.font_size = final_font_size
    dst_points_list.append(dst_points)
    return


def _layout_region_smart_scaling(
    *,
    candidate_n,
    candidate_required_height,
    candidate_required_width,
    config,
    dst_points_list,
    has_br,
    layout_candidate_font_size,
    layout_min_font_size,
    letter_spacing_multiplier,
    line_box_height,
    line_box_width,
    line_spacing_multiplier,
    mode,
    normal_anchor_mode,
    region,
    region_idx,
    render_horizontally,
    target_font_size,
):
    """smart_scaling: let the box of a region grow when its text does not fit."""
    # Diagnostic logging
    logger.debug(f"[SMART_SCALING] Region {region_idx}: mode={mode}, has_br={has_br}")

    try:
        bubble_width = float(line_box_width)
        bubble_height = float(line_box_height)
        required_width = float(candidate_required_width)
        required_height = float(candidate_required_height)
        n = max(1, int(candidate_n))
        target_font_size = int(max(layout_candidate_font_size, layout_min_font_size))

        # Create base polygon for scaling
        try:
            unrotated_base_poly = Polygon(region.unrotated_min_rect[0])
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/rendering/__init__.py:_layout_regions_to_font_size")
            unrotated_base_poly = Polygon([(0, 0), (bubble_width, 0), (bubble_width, bubble_height), (0, bubble_height)])

        logger.debug(
            f"[SMART_SCALING] Region {region_idx}: candidate n={n}, "
            f"font={target_font_size}, required={required_width:.1f}x{required_height:.1f}"
        )

        # Check for overflow in either dimension
        width_overflow = max(0, required_width - bubble_width)
        height_overflow = max(0, required_height - bubble_height)

        dst_points = region.min_rect

        if width_overflow > 0 or height_overflow > 0:
            # Scale width and height independently (the same logic for a single column/line and for several)
            width_scale_factor = 1.0
            height_scale_factor = 1.0

            if width_overflow > 0:
                width_scale_needed = required_width / bubble_width if bubble_width > 0 else 1.0
                diff_ratio_w = width_scale_needed - 1.0
                box_expansion_ratio_w = diff_ratio_w / 2
                width_scale_factor = 1 + min(box_expansion_ratio_w, 1.0)

            if height_overflow > 0:
                height_scale_needed = required_height / bubble_height if bubble_height > 0 else 1.0
                diff_ratio_h = height_scale_needed - 1.0
                box_expansion_ratio_h = diff_ratio_h / 2
                height_scale_factor = 1 + min(box_expansion_ratio_h, 1.0)

            try:
                scaled_unrotated_poly = affinity.scale(unrotated_base_poly, xfact=width_scale_factor, yfact=height_scale_factor, origin='center')
                scaled_unrotated_points = np.array(scaled_unrotated_poly.exterior.coords[:4])
                dst_points = rotate_polygons(region.center, scaled_unrotated_points.reshape(1, -1), -region.angle, to_int=False).reshape(-1, 4, 2)
            except Exception as e:
                logger.warning(f"Failed to apply independent scaling: {e}")

            # Font scaling is based on the dimension that overflows the most
            scale_needed = max(required_width / bubble_width if bubble_width > 0 else 1.0,
                             required_height / bubble_height if bubble_height > 0 else 1.0)
            diff_ratio = scale_needed - 1.0
            font_shrink_ratio = diff_ratio / 2 / (1 + diff_ratio)
            font_scale_factor = 1 - min(font_shrink_ratio, 0.5)
            target_font_size = int(target_font_size * font_scale_factor)

            # Recompute required with the rounded font size
            if render_horizontally:
                final_total_width = text_render.get_string_width(
                    target_font_size,
                    region.translation,
                    letter_spacing=letter_spacing_multiplier,
                )
                final_spacing_y = text_render.calc_horizontal_line_spacing_px(
                    target_font_size,
                    line_spacing_multiplier,
                )
                required_width = final_total_width / n if n > 0 else final_total_width
                required_height = n * target_font_size + max(0, n - 1) * final_spacing_y
            else:
                required_width, required_height, n, _ = calc_box_from_font(
                    target_font_size,
                    region.translation,
                    False,
                    line_spacing_multiplier,
                    config,
                    region.target_lang,
                    center=None,
                    angle=0,
                    letter_spacing=letter_spacing_multiplier,
                    stroke_width=_resolve_region_stroke_width(region, config),
                )

            # Recompute the box growth with the new required
            width_scale_factor = required_width / bubble_width if bubble_width > 0 and required_width > bubble_width else 1.0
            height_scale_factor = required_height / bubble_height if bubble_height > 0 and required_height > bubble_height else 1.0

            try:
                scaled_unrotated_poly = affinity.scale(unrotated_base_poly, xfact=width_scale_factor, yfact=height_scale_factor, origin='center')
                scaled_unrotated_points = np.array(scaled_unrotated_poly.exterior.coords[:4])
                dst_points = rotate_polygons(region.center, scaled_unrotated_points.reshape(1, -1), -region.angle, to_int=False).reshape(-1, 4, 2)
            except Exception as e:
                logger.warning(f"Failed to apply final scaling: {e}")
        else:
            # No overflow, can enlarge font to fit better
            if required_width > 0 and required_height > 0:
                width_scale_factor = bubble_width / required_width
                height_scale_factor = bubble_height / required_height
                font_scale_factor = min(width_scale_factor, height_scale_factor)
                target_font_size = int(target_font_size * font_scale_factor)

            try:
                unrotated_points = np.array(unrotated_base_poly.exterior.coords[:4])
                dst_points = rotate_polygons(region.center, unrotated_points.reshape(1, -1), -region.angle, to_int=False).reshape(-1, 4, 2)
            except Exception as e:
                logger.warning(f"Failed to use base polygon: {e}")

    except Exception as e:
        logger.exception(f"Error in smart_scaling layout for region {region_idx}: {e}")
        # Fallback to a safe state
        target_font_size = getattr(region, 'layout_base_font_size', target_font_size)
        dst_points = region.min_rect

    final_font_size = target_font_size

    # Compute dst_points directly with the helper (it builds the rectangle and rotates it)
    line_spacing_multiplier = _resolve_line_spacing_multiplier(region, config)
    letter_spacing_multiplier = _resolve_letter_spacing_multiplier(region, config)
    dst_points = _calc_region_dst_points_for_font(
        region=region,
        font_size=final_font_size,
        render_horizontally=render_horizontally,
        line_spacing_multiplier=line_spacing_multiplier,
        letter_spacing_multiplier=letter_spacing_multiplier,
        config=config,
        anchor_mode=normal_anchor_mode,
    )

    # When the calculation fails, use the original detected box
    if dst_points is None:
        dst_points = region.min_rect

    region.font_size = final_font_size
    dst_points_list.append(dst_points)
    return


def _layout_regions_to_font_size(
    img: np.ndarray,
    text_regions: List['TextBlock'],
    config: Config,
    original_img: np.ndarray = None,
    return_debug_img: bool = False,
    skip_font_scaling: bool = False,
    skip_text_replacements: bool = False,
    bubble_mask: Optional[np.ndarray] = None,
):
    """
    Calculate automatic layout without applying global font-size settings.

    Args:
        return_debug_img: If True, prepares a debug image for balloon_fill mode
        skip_font_scaling: If True, skip font scaling algorithm and use font_size from region directly (for load_text mode)
    """
    mode = config.render.layout_mode
    if (
        mode == 'balloon_fill'
        and return_debug_img
        and bool(getattr(config.render, 'semantic_linebreak', False))
    ):
        config._chinese_linebreak_debug_records = []
    
    logger.debug(f"[RESIZE] Processing {len(text_regions)} regions")

    # Prepare debug image for balloon_fill mode (only when requested)
    debug_img = None
    if mode == 'balloon_fill' and original_img is not None and return_debug_img:
        # OpenCV drawing APIs use BGR colours; the debug image is converted to BGR so the colours match
        debug_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        logger.debug("Created debug image for balloon_fill visualization")

    balloon_fill_mask = None
    balloon_fill_label_map = None
    balloon_fill_label_count = None
    # With skip_font_scaling (a layout authorised by the editor) the center_box anchor is always used and the bubble mask takes no part in placement;
    # only the verbose debug image still needs the mask, for visualisation
    if mode == 'balloon_fill' and original_img is not None and (not skip_font_scaling or return_debug_img):
        try:
            if bubble_mask is None:
                logger.warning("balloon_fill image context has no bubble mask, skip global bubble mask")
                balloon_fill_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
            else:
                balloon_fill_mask = bubble_mask
                mask_pixels = int(np.count_nonzero(balloon_fill_mask))
                logger.debug(
                    f"balloon_fill mask from image context: mask_pixels={mask_pixels}"
                )
                if mask_pixels == 0 and debug_img is not None:
                    logger.warning("balloon_fill global bubble mask is empty (mask_pixels=0), blue overlay will not be visible")
                if mask_pixels > 0:
                    balloon_fill_label_count, balloon_fill_label_map = cv2.connectedComponents(
                        np.where(balloon_fill_mask > 0, 1, 0).astype(np.uint8),
                        connectivity=8,
                    )
                    if debug_img is not None:
                        # Draw the "blue mask area" (semi-transparent fill) and a blue outline on the debug image, to make it easier to see
                        mask_u8 = np.where(balloon_fill_mask > 0, 255, 0).astype(np.uint8)
                        mask_pixels_idx = mask_u8 > 0
                        if np.any(mask_pixels_idx):
                            overlay = debug_img.copy()
                            overlay[mask_pixels_idx] = (255, 0, 0)  # BGR blue
                            cv2.addWeighted(overlay, 0.22, debug_img, 0.78, 0, dst=debug_img)

                        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 2)
        except Exception as exc:
            logger.warning(f"balloon_fill bubble mask preparation failed, skip global bubble mask: {exc}")
            balloon_fill_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
            balloon_fill_label_map = None
            balloon_fill_label_count = None

    # Bubble mask for center_text_in_bubble: reuse the mask passed from the image context.
    # (with skip_font_scaling the anchor is fixed to center_box and centring in the bubble does not apply, so the mask is not built for nothing)
    center_check_mask = balloon_fill_mask
    center_check_label_map = balloon_fill_label_map
    center_check_label_count = balloon_fill_label_count
    if (
        center_check_mask is None
        and config.render.center_text_in_bubble
        and original_img is not None
        and not skip_font_scaling
    ):
        try:
            if bubble_mask is not None:
                center_check_mask = bubble_mask
                if center_check_mask is not None and np.count_nonzero(center_check_mask) > 0:
                    center_check_label_count, center_check_label_map = cv2.connectedComponents(
                        np.where(center_check_mask > 0, 1, 0).astype(np.uint8), connectivity=8
                    )
                else:
                    center_check_mask = None
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/rendering/__init__.py:_layout_regions_to_font_size")
            pass

    dst_points_list = []
    anchor_modes: List[Optional[str]] = [None] * len(text_regions)
    placed_regions: list[tuple[np.ndarray, Optional[np.ndarray]]] = []
    for region_idx, region in enumerate(text_regions):
        if region is None:
            logger.info(f"[RESIZE] Region {region_idx}: None; skipping")
            dst_points_list.append(None)
            continue
        region_font_family = getattr(region, 'font_family', '') or ''
        try:
            region_bubble_mask = None
            if config:
                config._current_region = region
                config._semantic_linebreak_current_region_idx = region_idx

            # The region font is applied before any layout measurement; later candidate sizes, scaling and the final dst_points all use the same font.
            if region_font_family:
                text_render.set_font(region_font_family)
            else:
                text_render.set_font(text_render.DEFAULT_FONT_FAMILY)

            # When translation is empty, return min_rect directly and avoid the complex layout calculation
            render_value = _region_render_value(region)
            if (
                not render_value
                or (isinstance(render_value, str) and not render_value.strip())
                or (is_rich_text_document(render_value) and not _rich_text_has_content(render_value))
            ):
                logger.info(f"[RESIZE] Region {region_idx}: translation is empty; using min_rect")
                dst_points_list.append(region.min_rect)
                continue

            _apply_default_english_case_preferences(region, config)
            prepare_text_replacements_for_layout(
                [region],
                config,
                resolve_render_horizontal=_resolve_region_render_horizontal,
                skip_text_replacements=skip_text_replacements,
            )

            # Whether to centre in the bubble: the setting is on and the region really is inside a detected bubble
            apply_bubble_centering = config.render.center_text_in_bubble
            if apply_bubble_centering and center_check_mask is not None and np.count_nonzero(center_check_mask) > 0:
                _rm = _build_region_reference_mask(
                    region, center_check_mask, center_check_label_map, center_check_label_count
                )
                apply_bubble_centering = np.count_nonzero(_rm) > 0
            normal_anchor_mode = _resolve_layout_anchor_mode(
                apply_bubble_centering=apply_bubble_centering,
                skip_font_scaling=False,
            )
            skip_anchor_mode = _resolve_layout_anchor_mode(
                apply_bubble_centering=apply_bubble_centering,
                skip_font_scaling=True,
            )
            anchor_modes[region_idx] = normal_anchor_mode

            # skip_font_scaling mode: use region.font_size as the final size and skip layout scaling entirely
            # In an editor export the font size the user set is the size rendered, with no scaling
            if skip_font_scaling:
                _layout_region_with_fixed_font(
                    anchor_modes=anchor_modes,
                    config=config,
                    dst_points_list=dst_points_list,
                    img=img,
                    region=region,
                    region_idx=region_idx,
                    skip_anchor_mode=skip_anchor_mode,
                )
                continue
            else:
                original_region_font_size = region.font_size if region.font_size > 0 else round((img.shape[0] + img.shape[1]) / 200)

                # Keep the original font size on the region object, for the JSON export
                if not hasattr(region, 'original_font_size'):
                    region.original_font_size = original_region_font_size

                layout_min_font_size = 1
                target_font_size = max(_resolve_initial_layout_font_size(region, img, config), layout_min_font_size)

                # At entry only the reference font size of the layout algorithm itself is kept:
                # region.font_size > the value estimated from the image
                # render.font_size is a fixed font size that overrides the layout result at the single exit.
                region.layout_base_font_size = int(target_font_size)

                english_auto_line_break_applied = _apply_default_english_line_break_method(
                    region=region,
                    target_font_size=target_font_size,
                    original_img=original_img,
                    config=config,
                )
                if english_auto_line_break_applied:
                    render_horizontally = _resolve_region_render_horizontal(region)
                    line_spacing_multiplier = _resolve_line_spacing_multiplier(region, config)
                    letter_spacing_multiplier = _resolve_letter_spacing_multiplier(region, config)
                    final_font_size = target_font_size
                    anchor_modes[region_idx] = skip_anchor_mode

                    dst_points = _calc_region_dst_points_for_font(
                        region=region,
                        font_size=final_font_size,
                        render_horizontally=render_horizontally,
                        line_spacing_multiplier=line_spacing_multiplier,
                        letter_spacing_multiplier=letter_spacing_multiplier,
                        config=config,
                        anchor_mode=skip_anchor_mode,
                    )
                    if dst_points is None:
                        dst_points = region.min_rect

                    region.font_size = final_font_size
                    dst_points_list.append(dst_points)
                    continue

            render_horizontally = _resolve_region_render_horizontal(region)
            line_spacing_multiplier = _resolve_line_spacing_multiplier(region, config)
            letter_spacing_multiplier = _resolve_letter_spacing_multiplier(region, config)
            if getattr(config.render, 'recompute_line_breaks', False):
                # Drop the breaks an earlier layout stored so this layout decides them again.
                region.translation = unwrapped_translation(
                    region.translation,
                    getattr(region, 'translation_unwrapped', ''),
                    region.target_lang,
                )
            no_br_source_text = region.translation
            # region.translation is always str (TextBlock._translation is only written through
            # _translation_plain_text), so no isinstance guard is needed
            has_br = bool(re.search(r'(\[BR\]|【BR】|<br>)', region.translation, flags=re.IGNORECASE))
            if not has_br:
                # Keep the text as it is before wrapping, so the breaks can be recomputed later.
                region.translation_unwrapped = region.translation

            line_box_width, line_box_height = region.unrotated_size
            if not (isinstance(line_box_width, (int, float)) and np.isfinite(line_box_width) and line_box_width > 0):
                line_box_width = float(max(region.xywh[2], 1))
            if not (isinstance(line_box_height, (int, float)) and np.isfinite(line_box_height) and line_box_height > 0):
                line_box_height = float(max(region.xywh[3], 1))
            layout_box_width = float(line_box_width)
            layout_box_height = float(line_box_height)

            region_bubble_mask = None
            bubble_layout_rect = None

            lines_fully_enclosed = False
            if mode == 'balloon_fill' and original_img is not None:
                try:
                    if balloon_fill_mask is not None and np.count_nonzero(balloon_fill_mask) > 0:
                        region_bubble_mask = _build_region_reference_mask(
                            region, balloon_fill_mask, balloon_fill_label_map, balloon_fill_label_count
                        )
                    lines_fully_enclosed = (
                        np.count_nonzero(region_bubble_mask) > 0
                        and _region_lines_fully_inside_mask(region, region_bubble_mask)
                    )
                except Exception as exc:
                    logger.warning(
                        f"balloon_fill bubble mask preparation failed for region {region_idx}: {exc}"
                    )


            layout_candidate_font_size = int(max(target_font_size, layout_min_font_size))
            remove_linebreak_punctuation = bool(getattr(config.render, 'remove_linebreak_punctuation', False))
            region_layout_mode = mode
            if mode == 'balloon_fill' and not lines_fully_enclosed:
                # Text outside a complete bubble uses the same line-breaking
                # and box-fit font rules as an explicitly selected strict mode.
                region_layout_mode = 'strict'
            if is_rich_text_document(_region_render_value(region)):
                _layout_rich_text_region(
                    config=config,
                    dst_points_list=dst_points_list,
                    layout_candidate_font_size=layout_candidate_font_size,
                    layout_min_font_size=layout_min_font_size,
                    letter_spacing_multiplier=letter_spacing_multiplier,
                    line_box_height=line_box_height,
                    line_box_width=line_box_width,
                    line_spacing_multiplier=line_spacing_multiplier,
                    lines_fully_enclosed=lines_fully_enclosed,
                    mode=mode,
                    normal_anchor_mode=normal_anchor_mode,
                    original_img=original_img,
                    region=region,
                    region_bubble_mask=region_bubble_mask,
                    region_idx=region_idx,
                    render_horizontally=render_horizontally,
                )
                continue

            # BR branch at entry: explicit BRs are kept; text without BR is broken into lines here.
            # balloon_fill + balloon_fill_mask_layout uses the rectangle inscribed in the bubble,
            # instead of breaking lines by the OCR box first and breaking again in the balloon_fill branch later.
            layout_box_width, layout_box_height, bubble_layout_rect = _break_region_lines(
                bubble_layout_rect=bubble_layout_rect,
                config=config,
                has_br=has_br,
                layout_box_height=layout_box_height,
                layout_box_width=layout_box_width,
                layout_candidate_font_size=layout_candidate_font_size,
                layout_min_font_size=layout_min_font_size,
                letter_spacing_multiplier=letter_spacing_multiplier,
                line_box_height=line_box_height,
                line_box_width=line_box_width,
                line_spacing_multiplier=line_spacing_multiplier,
                lines_fully_enclosed=lines_fully_enclosed,
                no_br_source_text=no_br_source_text,
                region=region,
                region_bubble_mask=region_bubble_mask,
                region_layout_mode=region_layout_mode,
                remove_linebreak_punctuation=remove_linebreak_punctuation,
                render_horizontally=render_horizontally,
            )



            # After line breaking there is no second branch on BR: the final text is used to compute the box-fitted font size
            # (the layout limit of strict) and the candidate font size and candidate dimensions (the scaling inputs of smart_scaling
            # and balloon_fill).
            layout_candidate_font_size, box_fit_font_size = _select_preserved_line_layout_font(
                base_font_size=layout_candidate_font_size,
                width=layout_box_width,
                height=layout_box_height,
                text=region.translation,
                is_horizontal=render_horizontally,
                line_spacing=line_spacing_multiplier,
                config=config,
                target_lang=region.target_lang,
                letter_spacing=letter_spacing_multiplier,
                stroke_width=_resolve_region_stroke_width(region, config),
            )
            layout_candidate_font_size = max(int(layout_candidate_font_size), layout_min_font_size)
            candidate_required_width, candidate_required_height, candidate_n, _ = calc_box_from_font(
                layout_candidate_font_size,
                region.translation,
                render_horizontally,
                line_spacing_multiplier,
                config,
                region.target_lang,
                center=None,
                angle=0,
                letter_spacing=letter_spacing_multiplier,
                stroke_width=_resolve_region_stroke_width(region, config),
            )

            # --- Mode 5: balloon_fill (MUST BE FIRST to override other modes) ---
            if region_layout_mode == 'balloon_fill':
                _layout_region_balloon_fill(
                    anchor_modes=anchor_modes,
                    box_fit_font_size=box_fit_font_size,
                    bubble_layout_rect=bubble_layout_rect,
                    candidate_n=candidate_n,
                    candidate_required_height=candidate_required_height,
                    candidate_required_width=candidate_required_width,
                    config=config,
                    debug_img=debug_img,
                    dst_points_list=dst_points_list,
                    has_br=has_br,
                    layout_candidate_font_size=layout_candidate_font_size,
                    layout_min_font_size=layout_min_font_size,
                    letter_spacing_multiplier=letter_spacing_multiplier,
                    line_box_height=line_box_height,
                    line_box_width=line_box_width,
                    line_spacing_multiplier=line_spacing_multiplier,
                    lines_fully_enclosed=lines_fully_enclosed,
                    no_br_source_text=no_br_source_text,
                    normal_anchor_mode=normal_anchor_mode,
                    original_img=original_img,
                    original_region_font_size=original_region_font_size,
                    placed_regions=placed_regions,
                    region=region,
                    region_bubble_mask=region_bubble_mask,
                    region_idx=region_idx,
                    render_horizontally=render_horizontally,
                    target_font_size=target_font_size,
                )
                continue

            # --- Mode: strict ---
            if region_layout_mode == 'strict':
                _layout_region_strict(
                    box_fit_font_size=box_fit_font_size,
                    config=config,
                    dst_points_list=dst_points_list,
                    layout_candidate_font_size=layout_candidate_font_size,
                    letter_spacing_multiplier=letter_spacing_multiplier,
                    line_spacing_multiplier=line_spacing_multiplier,
                    normal_anchor_mode=normal_anchor_mode,
                    region=region,
                    render_horizontally=render_horizontally,
                )
                continue

            # --- Mode: smart_scaling ---
            elif mode == 'smart_scaling':
                _layout_region_smart_scaling(
                    candidate_n=candidate_n,
                    candidate_required_height=candidate_required_height,
                    candidate_required_width=candidate_required_width,
                    config=config,
                    dst_points_list=dst_points_list,
                    has_br=has_br,
                    layout_candidate_font_size=layout_candidate_font_size,
                    layout_min_font_size=layout_min_font_size,
                    letter_spacing_multiplier=letter_spacing_multiplier,
                    line_box_height=line_box_height,
                    line_box_width=line_box_width,
                    line_spacing_multiplier=line_spacing_multiplier,
                    mode=mode,
                    normal_anchor_mode=normal_anchor_mode,
                    region=region,
                    region_idx=region_idx,
                    render_horizontally=render_horizontally,
                    target_font_size=target_font_size,
                )
                continue

            # --- Unsupported layout modes ---
            else:
                raise ValueError(
                    f"Unsupported render.layout_mode: {mode!r}. "
                    "Supported values: balloon_fill, smart_scaling, strict"
                )
        except Exception:
            raise
        
    # Add legend to debug image
    if return_debug_img and debug_img is not None:
        # Add legend in top-left corner
        legend_y = 30
        cv2.putText(debug_img, 'Balloon Fill Debug:', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, 'Red = OCR Box', (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(debug_img, 'Yellow = Region Bubble Component', (10, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(debug_img, 'Blue = Global Bubble Mask', (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(debug_img, 'Green = Render Box', (10, legend_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(debug_img, 'Orange = Overflow Candidate Box', (10, legend_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    return dst_points_list, anchor_modes, debug_img


async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    config: Config = None,
    original_img: np.ndarray = None,
    return_debug_img: bool = False,
    skip_font_scaling: bool = False,
    skip_text_replacements: bool = False,
    render_alpha: Optional[np.ndarray] = None,
    bubble_mask: Optional[np.ndarray] = None,
    ):

    if config is None:
        from ..config import Config
        config = Config()

    if config.render.renderer in (
        Renderer.openai_renderer,
        Renderer.gemini_renderer,
    ):
        prepare_text_replacements_for_layout(
            text_regions,
            config,
            resolve_render_horizontal=_resolve_region_render_horizontal,
            skip_text_replacements=skip_text_replacements,
        )
        from .model_api_renderer import dispatch_api_rendering

        result = await dispatch_api_rendering(img=img, text_regions=text_regions, config=config)
        sync_translation_raw_from_layout(
            text_regions,
            config,
            skip_text_replacements=skip_text_replacements,
        )
        return result

    await download_chinese_linebreak_models_if_enabled(config)

    text_render.set_font(getattr(config.render, 'font_family', None) or text_render.DEFAULT_FONT_FAMILY)
    text_regions = list(filter(lambda region: _region_render_value(region), text_regions))

    dst_points_list, anchor_modes, debug_img = _layout_regions_to_font_size(
        img,
        text_regions,
        config,
        original_img,
        return_debug_img,
        skip_font_scaling=skip_font_scaling,
        skip_text_replacements=skip_text_replacements,
        bubble_mask=bubble_mask,
    )
    sync_translation_raw_from_layout(
        text_regions,
        config,
        skip_text_replacements=skip_text_replacements,
    )

    # Automatic rich-text rules must wait until replacement, line breaking and automatic wrapping of the plain string are done before the document is generated;
    # the last measurement only updates geometry: the global font size setting is applied once, while local rich-text sizes and ratios keep priority.
    _finalize_region_font_sizes(
        text_regions,
        dst_points_list,
        anchor_modes,
        config,
        skip_font_scaling=skip_font_scaling,
        debug_img=debug_img,
    )
    for region in text_regions:
        if hasattr(region, '_rich_text_rules_applied'):
            delattr(region, '_rich_text_rules_applied')

    # As in the editor: each region is first composed in full, with effects, stroke and body, and then
    # laid on the canvas in region order, later regions covering earlier ones. Drawing in passes across regions is not possible:
    # the body of a lower region would float above the stroke of an upper one and break the stacking order of the text boxes.
    for region, dst_points in tqdm(
        zip(text_regions, dst_points_list),
        '[render]',
        total=len(text_regions),
    ):
        region.dst_points = dst_points
        render_value = _region_render_value(region)
        if not render_value or (
            isinstance(render_value, str) and not render_value.strip()
        ):
            logger.info(
                f"[RENDER] Skipping empty text region: text='{region.text[:20] if region.text else ''}', "
                f"translation='{_translation_preview(render_value, 20)}'"
            )
            continue

        line_spacing_multiplier = _resolve_line_spacing_multiplier(region, config)
        img = render(
            img,
            region,
            dst_points,
            not config.render.no_hyphenation,
            line_spacing_multiplier,
            config.render.disable_font_border,
            config,
            render_alpha=render_alpha,
            paint_part=None,
        )
    
    if return_debug_img and debug_img is not None:
        return img, debug_img
    return img

def _native_render_rect_points(center, width: int, height: int, angle: float) -> np.ndarray:
    """Build the actual rendered corners from the native RGBA size; angle is clockwise in image coordinates."""
    cx, cy = float(center[0]), float(center[1])
    hw, hh = float(width) / 2.0, float(height) / 2.0
    local = np.array(
        [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]],
        dtype=np.float32,
    )
    rad = math.radians(float(angle or 0.0))
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    points = np.empty_like(local)
    points[:, 0] = cx + local[:, 0] * cos_a - local[:, 1] * sin_a
    points[:, 1] = cy + local[:, 0] * sin_a + local[:, 1] * cos_a
    return points.reshape(1, 4, 2)


def _premultiply_rgba(rgba: np.ndarray) -> np.ndarray:
    premultiplied = rgba.copy()
    if premultiplied.ndim != 3 or premultiplied.shape[2] != 4:
        return premultiplied
    alpha = premultiplied[:, :, 3].astype(np.float32) / 255.0
    for channel in range(3):
        premultiplied[:, :, channel] = np.clip(
            premultiplied[:, :, channel].astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)
    return premultiplied


def _rotate_native_rgba(premultiplied: np.ndarray, angle: float) -> np.ndarray:
    """Rotate premultiplied RGBA at scale=1 and enlarge the canvas so nothing is cropped."""
    angle = float(angle or 0.0)
    if abs(angle) < 1e-6:
        return premultiplied

    height, width = premultiplied.shape[:2]
    center = (width / 2.0, height / 2.0)
    # A positive OpenCV angle is counter-clockwise in image coordinates; a positive project angle (and Qt angle) is clockwise.
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    abs_cos = abs(float(matrix[0, 0]))
    abs_sin = abs(float(matrix[0, 1]))
    bound_w = max(1, int(math.ceil(width * abs_cos + height * abs_sin)))
    bound_h = max(1, int(math.ceil(height * abs_cos + width * abs_sin)))
    matrix[0, 2] += bound_w / 2.0 - center[0]
    matrix[1, 2] += bound_h / 2.0 - center[1]
    return cv2.warpAffine(
        premultiplied,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def render(
    img,
    region: TextBlock,
    dst_points,
    hyphenate,
    line_spacing,
    disable_font_border,
    config: Config,
    render_alpha: Optional[np.ndarray] = None,
    paint_part: str | None = None,
):
    # A region only stores the family; the font files are registered at start-up or on import.
    region_font_family = getattr(region, 'font_family', '') or ''
    if region_font_family:
        text_render.set_font(region_font_family)
    else:
        text_render.set_font(text_render.DEFAULT_FONT_FAMILY)

    # --- START BRUTEFORCE COLOR FIX ---
    fg = (0, 0, 0) # Default to black
    try:
        # Priority 1: Check for the original hex string from the UI
        if hasattr(region, 'font_color') and isinstance(region.font_color, str) and region.font_color.startswith('#'):
            hex_c = region.font_color
            if len(hex_c) == 7:
                r = int(hex_c[1:3], 16)
                g = int(hex_c[3:5], 16)
                b = int(hex_c[5:7], 16)
                fg = (r, g, b)
        # Priority 2: Check for a pre-converted tuple
        elif hasattr(region, 'fg_colors') and isinstance(region.fg_colors, (tuple, list)) and len(region.fg_colors) == 3:
            fg = tuple(region.fg_colors)
        # Last resort: Use the method2
        else:
            fg, _ = region.get_font_colors()
    except Exception as ignored_error:
        note_ignored_error(ignored_error, "manga_translator/rendering/__init__.py:render")
        # If anything fails, fg remains black
        pass

    # Get background color separately
    _, bg = region.get_font_colors()
    # --- END BRUTEFORCE COLOR FIX ---

    # Convert hex color string to RGB tuple, if necessary
    if isinstance(fg, str) and fg.startswith('#') and len(fg) == 7:
        try:
            r = int(fg[1:3], 16)
            g = int(fg[3:5], 16)
            b = int(fg[5:7], 16)
            fg = (r, g, b)
        except ValueError:
            fg = (0, 0, 0)  # Default to black on error
    elif not isinstance(fg, (tuple, list)):
        fg = (0, 0, 0) # Default to black if format is unexpected

    if getattr(region, 'adjust_bg_color', True):
        fg, bg = fg_bg_compare(fg, bg)

    text_to_render = region.get_translation_for_rendering()
    has_br_in_text = isinstance(text_to_render, str) and bool(re.search(r'(\[BR\]|<br>|【BR】)', text_to_render, flags=re.IGNORECASE))
    if has_br_in_text:
        text_to_render = re.sub(r'\s*(\[BR\]|<br>|【BR】)\s*', '\n', text_to_render, flags=re.IGNORECASE)

    if disable_font_border :
        bg = None

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    render_horizontally = _resolve_region_render_horizontal(region)
    letter_spacing = _resolve_letter_spacing_multiplier(region, config)

    # Pass the current region to config, for detecting a direction mismatch
    if config:
        config._current_region = region

    # Use the Qt off-screen renderer. A document without styling that only comes from BR conversion keeps using
    # the equivalent multi-line string, so the existing plain-text layout behaviour stays the same.
    if is_rich_text_document(text_to_render):
        plain_equivalent = plain_equivalent_text(text_to_render)
        if plain_equivalent is not None:
            text_to_render = plain_equivalent
    if render_horizontally:
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            text_to_render,
            round(norm_h[0]),
            round(norm_v[0]),
            region.alignment,
            False,  # Qt shapes and orders horizontal text from logical Unicode.
            fg,
            bg,
            region.target_lang,
            hyphenate,
            line_spacing,
            config,
            len(region.lines),
            stroke_width=region.stroke_width,
            letter_spacing=letter_spacing,
            paint_part=paint_part,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size,
            text_to_render,
            round(norm_v[0]),
            region.alignment,
            fg,
            bg,
            line_spacing,
            config,
            len(region.lines),
            stroke_width=region.stroke_width,
            letter_spacing=letter_spacing,
            paint_part=paint_part,
        )
    
    if temp_box is None:
        # The effects and stroke passes may intentionally have no pixels when
        # the region has no rich-text effect or the stroke is disabled.  The
        # fill pass is the one that determines whether text rendering failed.
        if paint_part in (None, "fill"):
            logger.warning(f"[RENDER SKIPPED] Text rendering returned None. Text: '{_translation_preview(region.translation, 100)}...'")
        return img
    
    h, w, _ = temp_box.shape
    if h == 0 or w == 0:
        logger.warning(f"Skipping rendering for region with invalid dimensions (w={w}, h={h}). Text: '{region.translation}'")
        return img
    layout_points = np.asarray(dst_points, dtype=np.float32).reshape(-1, 2)
    anchor = np.mean(layout_points[:4], axis=0)
    edge = layout_points[1] - layout_points[0]
    angle = math.degrees(math.atan2(float(edge[1]), float(edge[0])))

    # dst_points no longer controls the pixel size; the actual four corners are derived back from the native RGBA width and height.
    actual_dst_points = _native_render_rect_points(anchor, w, h, angle)
    region.dst_points = actual_dst_points

    premultiplied = _premultiply_rgba(temp_box)
    rgba_region = _rotate_native_rgba(premultiplied, angle)
    rotated_h, rotated_w = rgba_region.shape[:2]
    SHRT_MAX = 32767
    if rotated_h > SHRT_MAX or rotated_w > SHRT_MAX:
        logger.error(
            f"[RENDER SKIPPED] Native text layer exceeds OpenCV limit (32767). "
            f"box={rgba_region.shape[:2]}, text='{_translation_preview(getattr(region, 'translation', None), 50)}...'"
        )
        return img

    img_h, img_w = img.shape[:2]
    dst_x1 = int(round(float(anchor[0]) - rotated_w / 2.0))
    dst_y1 = int(round(float(anchor[1]) - rotated_h / 2.0))
    dst_x2 = dst_x1 + rotated_w
    dst_y2 = dst_y1 + rotated_h

    clip_x1 = max(0, dst_x1)
    clip_y1 = max(0, dst_y1)
    clip_x2 = min(img_w, dst_x2)
    clip_y2 = min(img_h, dst_y2)
    if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
        logger.warning(
            f"Text region completely outside image bounds: center=({anchor[0]:.1f}, {anchor[1]:.1f}), "
            f"native_size=({rotated_w}, {rotated_h}), image_size=({img_w}, {img_h}). "
            f"Text: '{_translation_preview(getattr(region, 'translation', None), 50)}...'"
        )
        return img

    src_x1 = clip_x1 - dst_x1
    src_y1 = clip_y1 - dst_y1
    src_x2 = src_x1 + (clip_x2 - clip_x1)
    src_y2 = src_y1 + (clip_y2 - clip_y1)
    source = rgba_region[src_y1:src_y2, src_x1:src_x2]
    canvas_region = source[:, :, :3]
    mask_region = source[:, :, 3:4].astype(np.float32) / 255.0
    target_region = img[clip_y1:clip_y2, clip_x1:clip_x2]
    img[clip_y1:clip_y2, clip_x1:clip_x2] = np.clip(
        target_region.astype(np.float32) * (1.0 - mask_region)
        + canvas_region.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    if render_alpha is not None:
        try:
            alpha_region = source[:, :, 3]
            alpha_target = render_alpha[clip_y1:clip_y2, clip_x1:clip_x2]
            if alpha_region.shape == alpha_target.shape:
                np.maximum(alpha_target, alpha_region, out=alpha_target)
        except Exception as alpha_error:
            logger.debug(f"Failed to accumulate render alpha: {alpha_error}")
    
    return img
