"""文字渲染管线 — 构建 TextBlock、render_params、执行渲染。"""
import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QPixmap, QTransform

from manga_translator.utils import TextBlock

logger = logging.getLogger('manga_translator')


def build_text_block_from_region(region_data: dict, font_size_override=None, log_tag: str = "") -> Optional[TextBlock]:
    args = region_data.copy()

    try:
        if "lines" in args and isinstance(args["lines"], list):
            args["lines"] = np.array(args["lines"])

        if args.get("texts") is None:
            args["texts"] = []

        if "font_color" in args and isinstance(args["font_color"], str):
            hex_color = args.pop("font_color")
            try:
                r = int(hex_color[1:3], 16)
                g = int(hex_color[3:5], 16)
                b = int(hex_color[5:7], 16)
                args["fg_color"] = (r, g, b)
            except (ValueError, TypeError):
                args["fg_color"] = (0, 0, 0)
        elif "fg_colors" in args:
            args["fg_color"] = args.pop("fg_colors")

        if "bg_colors" in args:
            args["bg_color"] = args.pop("bg_colors")

        if "direction" in args:
            d = args["direction"]
            if d == "horizontal":
                args["direction"] = "h"
            elif d == "vertical":
                args["direction"] = "v"

        # center is given explicitly by the upstream snapshot; no implicit offset is applied
        args["angle"] = 0
        if font_size_override is not None:
            args["font_size"] = font_size_override

        return TextBlock(**args)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        # Invalid region data is no longer swallowed silently: it is logged and the region skipped (the canvas stays empty there), so the user
        # can look into it (F04c). IndexError covers the existing crash where TextBlock reads
        # texts[0] of an empty texts list; the ValueError of np.array for an invalid lines shape
        # is "bad region data" as well and gets the same fallback.
        logger.warning("Failed to build TextBlock from region data%s: %s", log_tag, exc)
        return None


def build_region_render_params(
    render_parameter_service,
    _text_renderer_backend,
    region_index: int,
    region_data: dict,
    text_block: TextBlock,
) -> dict:
    render_params = render_parameter_service.export_parameters_for_backend(region_index, region_data)
    render_params["font_size"] = text_block.font_size
    region_font = (
        region_data.get("font_family")
        or getattr(text_block, "font_family", "")
        or render_params.get("font_family", "")
    )
    if region_font:
        render_params["font_family"] = region_font
        text_block.font_family = region_font

    return render_params


def render_region_text(text_renderer_backend, text_block: TextBlock, dst_points: np.ndarray, render_params: dict, total_regions: int):
    identity_transform = QTransform()
    return text_renderer_backend.render_text_for_region(
        text_block,
        dst_points,
        identity_transform,
        render_params,
        pure_zoom=1.0,
        total_regions=total_regions,
    )


def clear_region_text(item):
    item.update_text_pixmap(QPixmap(), QPointF(0, 0))
    item.set_dst_points(None)
