"""
Data layer of paste overlays (image patches laid on top) - normalisation, serialisation and parsing of the page JSON.

Data model: each page has one ``paste_overlays`` list (at the same level as ``regions``), and each item is a PNG asset that can be
placed, scaled and rotated freely (a repair patch, effect lettering, a background patch and so on). The image content is embedded in the JSON
as base64 PNG (RGBA), the same way ``mask_raw`` / ``paint_overlay`` / ``stamp_overlay`` are stored,
so a whole page moves with the project file.

Storage key in the page JSON::

    "paste_overlays": [{
        "id": "…", "name": "…",
        "z": 0, "visible": true, "opacity": 1.0,
        "center_x": 0.0, "center_y": 0.0,
        "width": 0.0, "height": 0.0,
        "rotation": 0.0, "flip_h": false, "flip_v": false,
        "image": "<base64 PNG, RGBA>"
    }, …]

The geometry fields are all values (floats) at the resolution of the source image. This module only depends on ``numpy`` / ``cv2``, not on Qt.
"""

from __future__ import annotations

import base64
import copy
import logging
import math
import struct
import uuid
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

logger = logging.getLogger("manga_translator")

PAGE_KEY = "paste_overlays"

_ALPHA_ZERO = 0
_OPACITY_MIN = 0.0
_OPACITY_MAX = 1.0


def _to_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须为数字，收到: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值，收到: {value!r}")
    return number


def _to_int(value: Any, name: str) -> int:
    number = _to_float(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} 必须为整数，收到: {value!r}")
    return int(number)


def _to_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{name} 必须为布尔值，收到: {value!r}")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off", ""):
            return False
        raise ValueError(f"{name} 布尔字符串无法解析: {value!r}")
    raise ValueError(f"{name} 必须为布尔值，收到: {type(value).__name__}")


def _clamp_opacity(value: float) -> float:
    return max(_OPACITY_MIN, min(_OPACITY_MAX, value))


def _validate_image_field(image: Any) -> str:
    if not isinstance(image, str):
        raise ValueError(f"image 必须为 base64 PNG 字符串，收到: {type(image).__name__}")
    if not image:
        return ""
    try:
        base64.b64decode(image, validate=True)
    except Exception as error:
        raise ValueError("image 不是合法的 base64 数据") from error
    return image


def new_overlay_id() -> str:
    """Generate a paste overlay id (a uuid4 hex without hyphens)."""
    return uuid.uuid4().hex


def normalize_paste_overlay(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise any input into a paste overlay dictionary (Python values that are safe as plain JSON).

    Missing fields get defaults; numbers are coerced; ``opacity`` is clamped to [0, 1].
    Invalid input raises :class:`ValueError`, and the caller decides whether to skip it or report an error.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"贴片必须是字典，收到: {type(raw).__name__}")

    width = _to_float(raw.get("width", 0.0), "width")
    height = _to_float(raw.get("height", 0.0), "height")
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height 必须为正数，收到 width={width} height={height}")

    return {
        "id": str(raw.get("id") or "").strip() or new_overlay_id(),
        "name": str(raw.get("name", "")).strip() or "贴片",
        "z": _to_int(raw.get("z", 0), "z"),
        "visible": _to_bool(raw.get("visible", True), "visible"),
        "opacity": _clamp_opacity(_to_float(raw.get("opacity", 1.0), "opacity")),
        "center_x": _to_float(raw.get("center_x", 0.0), "center_x"),
        "center_y": _to_float(raw.get("center_y", 0.0), "center_y"),
        "width": width,
        "height": height,
        "rotation": _to_float(raw.get("rotation", 0.0), "rotation"),
        "flip_h": _to_bool(raw.get("flip_h", False), "flip_h"),
        "flip_v": _to_bool(raw.get("flip_v", False), "flip_v"),
        "image": _validate_image_field(raw.get("image", "")),
    }


def _assign_unique_ids(overlays: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for overlay in overlays:
        overlay_id = str(overlay.get("id") or "").strip()
        while not overlay_id or overlay_id in seen:
            overlay_id = new_overlay_id()
        seen.add(overlay_id)
        overlay["id"] = overlay_id


def serialize_paste_overlays(
    overlays: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise a paste overlay list into a plain Python structure that ``json.dump`` can take directly (used on the writing side)."""
    normalized = [normalize_paste_overlay(item) for item in overlays]
    _assign_unique_ids(normalized)
    return copy.deepcopy(normalized)


def parse_page_paste_overlays(
    image_data: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Read the ``paste_overlays`` key from a page JSON dictionary (used on the reading side).

    - key missing or empty → an empty list is returned;
    - wrong root type → :class:`ValueError` is raised;
    - a single invalid overlay → a warning is logged and it is skipped, so one bad record does not bring down the loading of the whole page.
    """
    raw = image_data.get(PAGE_KEY)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{PAGE_KEY} 必须是列表，收到: {type(raw).__name__}")
    overlays: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        try:
            overlays.append(normalize_paste_overlay(item))
        except Exception as error:
            logger.warning("Skipping invalid overlay #%s: %s", index, error)
    _assign_unique_ids(overlays)
    return overlays


def rgba_overlay_to_png_base64(image_rgba: Any) -> str:
    """RGBA uint8 array → base64 PNG string (the same encoding as the paint and stamp layers)."""
    array = np.asarray(image_rgba)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"贴片必须为 RGBA，收到 shape {array.shape}")
    bgra = cv2.cvtColor(array.astype(np.uint8, copy=False), cv2.COLOR_RGBA2BGRA)
    ok, encoded = cv2.imencode(".png", bgra)
    if not ok:
        raise ValueError("贴片 PNG 编码失败")
    return base64.b64encode(encoded).decode("utf-8")


def png_base64_to_rgba_overlay(image_b64: str) -> np.ndarray | None:
    """base64 PNG string → RGBA uint8 array; None when decoding fails or the result is not RGBA."""
    if not isinstance(image_b64, str) or not image_b64:
        return None
    try:
        image_bytes = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
        bgra = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        return None
    array = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    array.setflags(write=False)
    return array


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_base64_dimensions(image_b64: str) -> tuple[int, int] | None:
    """Parse the IHDR width and height of a base64 PNG (without decoding the pixels); None for a non-PNG or when parsing fails.

    Used before ``cv2.imdecode`` to stop a "decompression bomb" with a huge pixel count, so no giant image is allocated first.
    """
    try:
        data = base64.b64decode(image_b64)
    except Exception:
        return None
    if len(data) < 33 or not data.startswith(_PNG_SIGNATURE):
        return None
    try:
        width, height = struct.unpack(">II", data[16:24])
    except (struct.error, IndexError):
        return None
    return int(width), int(height)


def compose_paste_overlays(
    overlays: Iterable[Mapping[str, Any]],
    canvas_size: tuple[int, int],
) -> np.ndarray | None:
    """Pre-composite the paste overlays, each with its own geometry and alpha, onto one whole-page RGBA canvas (baked for export).

    ``canvas_size`` is the (width, height) pixel size of the source image; the returned array has the same structure as the paint and stamp overlay layers,
    and the backend alpha-composites it with the base image before rendering the text. None is returned when no overlay is visible.
    Overlays are composited in ascending ``z`` (larger z on top) with source-over alpha compositing,
    so a semi-transparent overlay does not hide the layer below; scaling is done by the affine matrix, without a large intermediate image.
    Note: for the on-screen direction of rotation and flipping, the canvas preview is the reference; if a direction turns out reversed later,
    only the sign of the rotation angle in this function needs changing.
    """
    width, height = (int(canvas_size[0]), int(canvas_size[1]))
    if width <= 0 or height <= 0:
        return None

    # The canvas itself is stored as straight-alpha uint8 (memory for a whole page = 4 bytes per pixel);
    # premultiplied alpha only appears in the local float calculation of each overlay's bounding box,
    # so a large page with many overlays no longer holds whole-page float32 copies at the same time.
    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    def _blend_premultiplied(base: np.ndarray, patch: np.ndarray) -> np.ndarray:
        """pre-mul source-over: RGB is already premultiplied, and the alpha channel holds the original 0..255 values."""
        patch_coverage = patch[..., 3:4] / 255.0
        merged = np.empty_like(base)
        merged[..., :3] = patch[..., :3] + base[..., :3] * (1.0 - patch_coverage)
        merged[..., 3:4] = patch[..., 3:4] + base[..., 3:4] * (1.0 - patch_coverage)
        return merged

    # Size limit for a single overlay image: guards against a hand-made oversized base64 (see CodeRabbit CWE-400)
    max_image_chars = 24_000_000
    # Largest side in pixels after decoding: beyond it the project is treated as abnormal and the overlay is skipped
    max_source_side = 8192

    items = [
        item
        for item in overlays
        if isinstance(item, Mapping) and item.get("visible", True)
    ]
    # Composite in ascending z: lower z is drawn first (below); equal z keeps the list order
    items.sort(key=lambda item: float(item.get("z", 0)) or 0.0)

    drawn = False
    for item in items:
        image_b64 = item.get("image", "")
        if not isinstance(image_b64, str) or not image_b64:
            continue
        if len(image_b64) > max_image_chars:
            logger.warning("Skipping oversized overlay image data (>%s base64 characters)", max_image_chars)
            continue
        png_size = _png_base64_dimensions(image_b64)
        if png_size is not None and max(png_size) > max_source_side:
            logger.warning("Skipping oversized overlay PNG (%dx%d)", png_size[0], png_size[1])
            continue
        source = png_base64_to_rgba_overlay(image_b64)
        if source is None or not np.any(source[..., 3]):
            continue
        source_h, source_w = source.shape[:2]
        if source_w <= 0 or source_h <= 0 or max(source_w, source_h) > max_source_side:
            logger.warning("Skipping overlay with invalid dimensions (%dx%d)", source_w, source_h)
            continue

        target_width = float(item.get("width", source_w))
        target_height = float(item.get("height", source_h))
        if target_width <= 0 or target_height <= 0:
            continue
        center_x = float(item.get("center_x", 0.0))
        center_y = float(item.get("center_y", 0.0))
        rotation = float(item.get("rotation", 0.0))
        flip_h = -1.0 if item.get("flip_h") else 1.0
        flip_v = -1.0 if item.get("flip_v") else 1.0
        try:
            opacity = float(item.get("opacity", 1.0))
        except (TypeError, ValueError):
            opacity = 1.0
        opacity = max(0.0, min(1.0, opacity))
        if opacity <= 0.0:
            continue
        if opacity < 1.0:
            source = source.copy()
            source[..., 3] = (source[..., 3].astype(np.float32) * opacity).astype(np.uint8)

        # Premultiply: RGB x (alpha/255); the RGB that takes part in interpolation is colour x coverage, 0 where transparent
        premul_source = np.empty((source_h, source_w, 4), dtype=np.float32)
        premul_source[..., 3:4] = source[..., 3:4].astype(np.float32)
        premul_source[..., :3] = (
            source[..., :3].astype(np.float32)
            * (premul_source[..., 3:4] / 255.0)
        )

        # Affine matrix: p_scene = center + R * S * (p_source - source_center)
        # S includes non-uniform scaling and horizontal/vertical flips, so no enlarged intermediate image is needed first
        scale_x = flip_h * (target_width / source_w)
        scale_y = flip_v * (target_height / source_h)
        theta = math.radians(rotation)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        a00 = cos_t * scale_x
        a01 = -sin_t * scale_y
        a10 = sin_t * scale_x
        a11 = cos_t * scale_y
        offset_x = center_x - (a00 * (source_w / 2.0) + a01 * (source_h / 2.0))
        offset_y = center_y - (a10 * (source_w / 2.0) + a11 * (source_h / 2.0))
        matrix = np.array(
            [[a00, a01, offset_x], [a10, a11, offset_y]], dtype=np.float64
        )

        # Warp and blend only the transformed bounding box: this avoids allocating a whole-page float32 canvas for every overlay
        # on a large page (a single 8192x8192 one is about 1 GiB; several overlays would run out of memory)
        src_corners = np.array(
            [[0.0, 0.0], [source_w, 0.0], [source_w, source_h], [0.0, source_h]],
            dtype=np.float64,
        )
        transformed_x = (
            matrix[0, 0] * src_corners[:, 0]
            + matrix[0, 1] * src_corners[:, 1]
            + matrix[0, 2]
        )
        transformed_y = (
            matrix[1, 0] * src_corners[:, 0]
            + matrix[1, 1] * src_corners[:, 1]
            + matrix[1, 2]
        )
        min_x = max(0, int(math.floor(transformed_x.min())) - 1)
        max_x = min(width, int(math.ceil(transformed_x.max())) + 1)
        min_y = max(0, int(math.floor(transformed_y.min())) - 1)
        max_y = min(height, int(math.ceil(transformed_y.max())) + 1)
        if max_x <= min_x or max_y <= min_y:
            # The overlay lies completely outside the canvas
            continue

        box_width = max_x - min_x
        box_height = max_y - min_y
        patch = np.zeros((box_height, box_width, 4), dtype=np.float32)
        sub_matrix = matrix.copy()
        sub_matrix[0, 2] -= min_x
        sub_matrix[1, 2] -= min_y
        patch = cv2.warpAffine(
            premul_source,
            sub_matrix,
            (box_width, box_height),
            dst=patch,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        roi = canvas[min_y:max_y, min_x:max_x]
        roi_float = roi.astype(np.float32)
        roi_alpha = roi_float[..., 3:4]
        premul_base = np.empty_like(roi_float)
        premul_base[..., 3:4] = roi_alpha
        premul_base[..., :3] = roi_float[..., :3] * (roi_alpha / 255.0)
        blended = _blend_premultiplied(premul_base, patch)
        out_alpha = blended[..., 3:4]
        out_coverage = out_alpha / 255.0
        safe = np.maximum(out_coverage, 1e-6)
        straight = np.empty_like(roi_float)
        straight[..., :3] = np.clip(blended[..., :3] / safe, 0, 255)
        straight[..., 3:4] = np.clip(out_alpha, 0, 255)
        roi[...] = straight.astype(np.uint8)
        drawn = True

    if not drawn:
        return None
    if not np.any(canvas[..., 3]):
        return None
    canvas.setflags(write=False)
    return canvas
