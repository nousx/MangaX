"""Compositing layer: colouring, stroking and pasting alpha bitmaps, affine effects (italics, rotation, mirroring) and their pure-geometry counterparts.

It holds no render state and only does numpy/cv2 layer operations and TextStyle parsing.
"""

import math

import cv2
import numpy as np

from ...utils import parse_color
from ..rich_text import TextStyle


def add_color(bw_char_map, color, stroke_char_map, stroke_color):
    """Composite text and stroke into an RGBA layer.

    Key points (they fix grey and dirty edges, and keep the whole run from turning transparent when the stroke layer is missing):
    1. Force stroke_alpha = max(stroke_alpha, text_alpha), so the stroke fully covers every anti-aliased pixel of the text
       in space and the misalignment from two separate rasterisations disappears;
       it also guarantees that the output alpha still contains the body when the stroke layer is all zero (stroke_ratio=0 and similar cases).
    2. Treat the stroke as the "base colour" of the text: anti-aliased transition pixels are blended directly onto the solid stroke colour instead of
       compositing two semi-transparent layers with over - which was the cause of the old grey edges.
    """
    H, W = bw_char_map.shape[:2]
    if bw_char_map.size == 0:
        return np.zeros((H, W, 4), dtype=np.uint8)

    out = np.zeros((H, W, 4), dtype=np.uint8)
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)

    # No stroke: output the text layer directly
    if stroke_color is None or stroke_char_map is None:
        out[:, :, :3] = np.clip(color_arr, 0, 255).astype(np.uint8)
        out[:, :, 3] = bw_char_map
        return out

    stroke_color_arr = np.asarray(stroke_color, dtype=np.float32).reshape(1, 1, 3)

    # 1) Force stroke_alpha >= text_alpha - when the stroke layer is all zero, the output reduces to the text alone
    text_alpha_u8 = bw_char_map
    stroke_alpha_u8 = np.maximum(stroke_char_map, text_alpha_u8)

    # 2) The text is blended onto the solid stroke colour (locally opaque, no stacking of semi-transparent layers)
    text_af = (text_alpha_u8.astype(np.float32) / 255.0)[:, :, None]
    rgb = color_arr * text_af + stroke_color_arr * (1.0 - text_af)

    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[:, :, 3] = stroke_alpha_u8
    return out


def _parse_rgb(value, fallback=(0, 0, 0)) -> tuple[int, int, int]:
    """F16: thin wrapper around utils.generic.parse_color (#RGB/#RRGGBB/sequence, clamping, fallback).

    On top of the shared helper it guarantees that an RGB triple is always returned: the fallback itself may also be
    a hex string or a sequence, and when both levels fail to parse, black is returned.
    """
    color = parse_color(value, None)
    if color is None:
        color = parse_color(fallback, None)
    return tuple(color) if color is not None else (0, 0, 0)


def _style_font_size(base_font_size: int, style: TextStyle) -> int:
    if style.font_size is not None:
        return max(1, round(style.font_size))
    return max(1, round(base_font_size * (style.scale or 1.0)))


def _style_fill_color(style: TextStyle, fallback) -> tuple[int, int, int]:
    return _parse_rgb(style.color, fallback)


def _style_stroke_color(style: TextStyle, fallback):
    if style.stroke and style.stroke.color:
        return _parse_rgb(style.stroke.color, fallback or (0, 0, 0))
    return None if fallback is None else _parse_rgb(fallback, (0, 0, 0))


def _style_stroke_ratio(
    style: TextStyle, font_size: int, global_stroke_ratio: float, global_stroke_color
) -> float:
    if style.stroke:
        if style.stroke.width is not None:
            return max(0.0, float(style.stroke.width))
        return max(float(global_stroke_ratio), 0.07)
    return (
        max(float(global_stroke_ratio), 0.0) if global_stroke_color is not None else 0.0
    )


def _rgba_from_alpha_pair(
    text_alpha: np.ndarray, border_alpha: np.ndarray | None, fill_color, stroke_color
):
    if text_alpha is None or text_alpha.size == 0:
        return None
    if stroke_color is None or border_alpha is None or border_alpha.size == 0:
        border_alpha = None
    return add_color(text_alpha, fill_color, border_alpha, stroke_color)


def _rgba_for_paint_part(
    text_alpha, border_alpha, fill_color, stroke_color, paint_part: str
):
    if paint_part == "fill":
        return _colored_alpha_layer(text_alpha, fill_color)
    if paint_part == "stroke":
        if stroke_color is None or border_alpha is None or not border_alpha.size:
            return None
        return _colored_alpha_layer(border_alpha, stroke_color)
    return _rgba_from_alpha_pair(text_alpha, border_alpha, fill_color, stroke_color)


def _stroke_alpha_from_text_alpha(text_alpha: np.ndarray, stroke_px: int):
    if text_alpha is None or text_alpha.size == 0:
        return np.zeros((0, 0), dtype=np.uint8), 0, 0
    pad = max(1, int(stroke_px)) + 1
    padded = cv2.copyMakeBorder(
        text_alpha, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )
    fg_mask = (padded >= 128).astype(np.uint8) * 255
    bg_mask = cv2.bitwise_not(fg_mask)
    dist = cv2.distanceTransform(bg_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    stroke_region = np.clip((stroke_px + 0.5 - dist), 0.0, 1.0)
    stroke_alpha = (stroke_region * 255).astype(np.uint8)
    return np.maximum(stroke_alpha, padded), -pad, -pad


def _crop_rgba_fixed(canvas: np.ndarray, x0: int, x1: int, y0: int, y1: int):
    """Crop to the given envelope rectangle (not tightly to the ink).

    The "measured == size of the output surface" contract of the rich-text path: the envelope is given by the metric geometry, measuring and
    drawing share the same numbers, and the output surface must equal the measured box exactly; offsets, shear and the like may push the ink
    anywhere inside the box without it being cropped (when everything is transparent, None is still returned, meaning there is nothing to draw).
    """
    if canvas is None or canvas.size == 0 or canvas.shape[2] != 4:
        return None
    x0 = max(0, min(int(x0), canvas.shape[1]))
    x1 = max(x0, min(int(x1), canvas.shape[1]))
    y0 = max(0, min(int(y0), canvas.shape[0]))
    y1 = max(y0, min(int(y1), canvas.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return None
    region = canvas[y0:y1, x0:x1]
    return region if region[:, :, 3].any() else None


def _stroke_pad_px(font_size: int, stroke_ratio: float) -> int:
    """How far the stroke extends a glyph layer on each of its four sides (pixels).

    The same pad formula as _stroke_alpha_from_text_alpha, shared by measuring and drawing.
    """
    if stroke_ratio <= 0:
        return 0
    stroke_px = max(round(stroke_ratio * font_size), 1)
    return max(1, int(stroke_px)) + 1


def _paste_rgba(dst: np.ndarray, src: np.ndarray, x: int, y: int):
    if src is None or src.size == 0:
        return
    rows, width = src.shape[:2]
    x2, y2 = x + width, y + rows
    sx1, sy1, sx2, sy2 = (
        max(0, x),
        max(0, y),
        min(dst.shape[1], x2),
        min(dst.shape[0], y2),
    )
    if sx1 >= sx2 or sy1 >= sy2:
        return
    bx1, by1 = sx1 - x, sy1 - y
    src_view = src[by1 : by1 + (sy2 - sy1), bx1 : bx1 + (sx2 - sx1)]
    dst_view = dst[sy1:sy2, sx1:sx2]
    if not dst_view[:, :, 3].any():
        # F26 fast path: the target area is fully transparent (the usual case when pasting vertical text character by character),
        # so alpha-over reduces to a plain overwrite and the full floating-point blend is skipped
        dst_view[:] = src_view
        return
    src_crop = src_view.astype(np.float32)
    dst_crop = dst_view.astype(np.float32)
    src_a = src_crop[:, :, 3:4] / 255.0
    dst_a = dst_crop[:, :, 3:4] / 255.0
    out_a = src_a + dst_a * (1.0 - src_a)
    safe_a = np.where(out_a <= 0, 1.0, out_a)
    out_rgb = (
        src_crop[:, :, :3] * src_a + dst_crop[:, :, :3] * dst_a * (1.0 - src_a)
    ) / safe_a
    out = np.zeros_like(dst_crop)
    out[:, :, :3] = np.clip(out_rgb, 0, 255)
    out[:, :, 3:4] = np.clip(out_a * 255.0, 0, 255)
    dst[sy1:sy2, sx1:sx2] = out.astype(np.uint8)


def _warp_geometry(height: int, width: int, matrix: np.ndarray):
    """Pure geometry of an affine transform: the size and offset of the output box after an input box (height, width)
    is transformed by matrix. Shared by rendering (_warp_rgba_layer) and measuring, so the geometry agrees.
    """
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(
        -1, 1, 2
    )
    transformed = cv2.transform(corners, matrix).reshape(-1, 2)
    min_x, min_y = transformed.min(axis=0)
    max_x, max_y = transformed.max(axis=0)
    out_w = max(1, math.ceil(max_x - min_x))
    out_h = max(1, math.ceil(max_y - min_y))
    return out_h, out_w, min_x, min_y


def _warp_rgba_layer(layer: np.ndarray, matrix: np.ndarray):
    if layer is None or layer.size == 0:
        return layer, 0.0, 0.0
    h, w = layer.shape[:2]
    out_h, out_w, min_x, min_y = _warp_geometry(h, w, matrix)
    adjusted = matrix.copy()
    adjusted[0, 2] -= min_x
    adjusted[1, 2] -= min_y
    warped = cv2.warpAffine(
        layer,
        adjusted,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped, float(min_x), float(min_y)


DEFAULT_ITALIC_ANGLE = 10.0
_MAX_ITALIC_ANGLE = 85.0


def _style_italic_shear(style: TextStyle) -> float:
    """italic protocol value -> shear factor of the glyph outline (the meaning of faux italics in Photoshop).

    True = the default of 10 degrees measured for Photoshop faux italics (see test/ps_italic_angle.py); a number = the angle
    (degrees). The shear is applied at the glyph path stage (in the coordinate system with the baseline origin, x' = x + shear * y), so it is
    naturally about the baseline and the advance is unchanged; characters lying on their side in vertical text are sheared first and then rotated (R * S). The path
    y axis points down, so the factor for leaning right is -tan(angle).
    """
    italic = style.italic
    if italic is True:
        angle = DEFAULT_ITALIC_ANGLE
    elif isinstance(italic, (int, float)) and not isinstance(italic, bool):
        angle = float(italic)
    else:
        return 0.0
    angle = max(-_MAX_ITALIC_ANGLE, min(_MAX_ITALIC_ANGLE, angle))
    if not angle:
        return 0.0
    return -math.tan(math.radians(angle))


def _apply_style_layer_effects(layer: np.ndarray, style: TextStyle, font_size: int):
    """Geometric post-processing of a layer for mirroring and free rotation. Italics and stretching are already done at the glyph outline stage."""
    if layer is None or layer.size == 0:
        return layer, 0.0, 0.0

    result = layer
    offset_x = 0.0
    offset_y = 0.0

    if style.transform.mirror_x:
        result = cv2.flip(result, 1)
    if style.transform.mirror_y:
        result = cv2.flip(result, 0)

    if style.transform.rotation:
        height, width = result.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0),
            float(style.transform.rotation),
            1.0,
        )
        result, dx, dy = _warp_rgba_layer(result, matrix)
        offset_x += dx
        offset_y += dy

    return result, offset_x, offset_y


def _style_layer_effects_geometry(
    height: int,
    width: int,
    style: TextStyle,
    font_size: int,
    *,
    include_paint_effects: bool = True,
):
    """Pure-geometry version of _apply_style_layer_effects (for measuring, F21).

    mirror does not change the size; rotation computes the output box and offset from the corner points through _warp_geometry with the same matrix
    as the render path, without an actual warp. Italics and stretching are already done at the glyph outline stage.
    Returns (height, width, offset_x, offset_y).
    """
    if height <= 0 or width <= 0:
        return height, width, 0.0, 0.0

    paint_pad = (
        _style_paint_effect_pad(style, font_size) if include_paint_effects else 0
    )
    if paint_pad:
        height += paint_pad * 2
        width += paint_pad * 2
    offset_x = float(-paint_pad)
    offset_y = float(-paint_pad)

    if style.transform.rotation:
        matrix = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0), float(style.transform.rotation), 1.0
        )
        height, width, dx, dy = _warp_geometry(height, width, matrix)
        offset_x += float(dx)
        offset_y += float(dy)

    return height, width, offset_x, offset_y


def _style_paint_effect_pad(style: TextStyle, font_size: int) -> int:
    """Pixel envelope of a local outer stroke or glow, converted from its ratio of the font size."""
    outer_px = 0.0
    if style.outer_stroke and style.outer_stroke.width is not None:
        outer_px = max(0.0, float(style.outer_stroke.width)) * max(
            float(font_size), 1.0
        )
    glow_px = 0.0
    if style.glow:
        glow_px = max(0.0, float(style.glow.blur)) * max(float(font_size), 1.0)
    return math.ceil(max(outer_px + 1.0 if outer_px else 0.0, glow_px * 3.0))


def _colored_alpha_layer(alpha: np.ndarray, color) -> np.ndarray:
    layer = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    layer[:, :, :3] = np.asarray(_parse_rgb(color), dtype=np.uint8)
    layer[:, :, 3] = alpha
    return layer


def _apply_style_paint_effects(
    layer: np.ndarray, style: TextStyle, font_size: int, paint_part: str = "all"
):
    """Build effect and body layers of the same size, so the renderer can draw effects first and body second, globally."""
    if layer is None or layer.size == 0:
        return layer
    pad = _style_paint_effect_pad(style, font_size)
    if pad <= 0:
        return None if paint_part == "effects" else layer

    canvas = np.zeros(
        (layer.shape[0] + pad * 2, layer.shape[1] + pad * 2, 4), dtype=np.uint8
    )
    alpha = layer[:, :, 3]

    if paint_part != "body" and style.glow and style.glow.blur > 0:
        blur = max(0.0, float(style.glow.blur)) * max(float(font_size), 1.0)
        padded_alpha = cv2.copyMakeBorder(
            alpha, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        )
        glow_alpha = cv2.GaussianBlur(padded_alpha, (0, 0), sigmaX=blur, sigmaY=blur)
        _paste_rgba(
            canvas,
            _colored_alpha_layer(glow_alpha, style.glow.color or "#000000"),
            0,
            0,
        )

    if (
        paint_part != "body"
        and style.outer_stroke
        and style.outer_stroke.width is not None
        and style.outer_stroke.width > 0
    ):
        width = max(
            1, round(float(style.outer_stroke.width) * max(float(font_size), 1.0))
        )
        padded_alpha = cv2.copyMakeBorder(
            alpha, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1)
        )
        outer_alpha = cv2.dilate(padded_alpha, kernel)
        _paste_rgba(
            canvas,
            _colored_alpha_layer(outer_alpha, style.outer_stroke.color or "#000000"),
            0,
            0,
        )

    if paint_part != "effects":
        _paste_rgba(canvas, layer, pad, pad)
    return canvas


def _draw_rgba_disc(
    dst: np.ndarray, center_x: float, center_y: float, radius: float, color
):
    radius = max(1, round(radius))
    size = radius * 2 + 3
    alpha = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(alpha, (size // 2, size // 2), radius, 255, -1, lineType=cv2.LINE_AA)
    layer = np.zeros((size, size, 4), dtype=np.uint8)
    layer[:, :, :3] = np.asarray(_parse_rgb(color), dtype=np.uint8)
    layer[:, :, 3] = alpha
    _paste_rgba(dst, layer, round(center_x) - size // 2, round(center_y) - size // 2)


def _draw_rgba_bar(
    dst: np.ndarray, left: float, top: float, width: float, height: float, color
):
    """Solid rectangular bar (for the underline of vertical text; the rectangular counterpart of the emphasis dot).

    Same structure as _draw_rgba_disc: it only produces an opaque block of solid colour on the fill layer and takes no part in stroke or glow
    - so decorations and body glyphs keep the same layer responsibilities here.
    """
    width = max(1, round(width))
    height = max(1, round(height))
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    layer[:, :, :3] = np.asarray(_parse_rgb(color), dtype=np.uint8)
    layer[:, :, 3] = 255
    _paste_rgba(dst, layer, round(left), round(top))


def _paste_bitmap(
    canvas: np.ndarray, bitmap_arr: np.ndarray, x: int, y: int, mode: str = "max"
):
    if bitmap_arr is None or bitmap_arr.size == 0:
        return
    rows, width = bitmap_arr.shape
    x2, y2 = x + width, y + rows
    sx1, sy1, sx2, sy2 = (
        max(0, x),
        max(0, y),
        min(canvas.shape[1], x2),
        min(canvas.shape[0], y2),
    )
    if sx1 >= sx2 or sy1 >= sy2:
        return
    bx1, by1 = sx1 - x, sy1 - y
    bitmap = bitmap_arr[by1 : by1 + (sy2 - sy1), bx1 : bx1 + (sx2 - sx1)]
    target = canvas[sy1:sy2, sx1:sx2]
    if mode == "add":
        # Use cv2.add to avoid the specks caused by numpy uint8 addition overflow
        cv2.add(target, bitmap, dst=target)
    else:
        np.maximum(target, bitmap, out=target)


def _crop_pair(text_canvas: np.ndarray, border_canvas: np.ndarray):
    combined = cv2.add(text_canvas, border_canvas)
    if not np.any(combined):
        return None
    x, y, w, h = cv2.boundingRect(combined)
    return (
        None
        if w == 0 or h == 0
        else (
            text_canvas[y : y + h, x : x + w],
            border_canvas[y : y + h, x : x + w],
            x,
            y,
            w,
            h,
        )
    )


def _glyph_pair_rgba(
    char_bitmap: np.ndarray,
    border_bitmap: np.ndarray | None,
    fill,
    stroke,
    paint_part: str = "all",
):
    if char_bitmap is None or char_bitmap.size == 0:
        return None
    if border_bitmap is None or border_bitmap.size == 0 or stroke is None:
        layer = (
            None if paint_part == "stroke" else _colored_alpha_layer(char_bitmap, fill)
        )
        return layer, 0, 0
    border_x = -round((border_bitmap.shape[1] - char_bitmap.shape[1]) / 2.0)
    border_y = -round((border_bitmap.shape[0] - char_bitmap.shape[0]) / 2.0)
    left = min(0, border_x)
    top = min(0, border_y)
    right = max(char_bitmap.shape[1], border_x + border_bitmap.shape[1])
    bottom = max(char_bitmap.shape[0], border_y + border_bitmap.shape[0])
    text_alpha = np.zeros((bottom - top, right - left), dtype=np.uint8)
    border_alpha = np.zeros_like(text_alpha)
    _paste_bitmap(text_alpha, char_bitmap, -left, -top)
    _paste_bitmap(
        border_alpha, border_bitmap, border_x - left, border_y - top, mode="add"
    )
    return (
        _rgba_for_paint_part(text_alpha, border_alpha, fill, stroke, paint_part),
        left,
        top,
    )


def _bitmap_ink_rect(
    bitmap: np.ndarray | None,
) -> tuple[int, int, int, int] | None:
    if bitmap is None or bitmap.size == 0:
        return None
    nz = cv2.findNonZero(bitmap)
    return None if nz is None else tuple(map(int, cv2.boundingRect(nz)))


def _stroke_bitmap_from_alpha(
    text_alpha: np.ndarray, font_size: int, stroke_ratio: float
):
    stroke_px = max(round(stroke_ratio * font_size), 1)
    bitmap, _, _ = _stroke_alpha_from_text_alpha(text_alpha, stroke_px)
    return None if bitmap.size == 0 else bitmap
