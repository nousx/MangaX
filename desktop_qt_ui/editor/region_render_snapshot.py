"""Render snapshot of a single region.

Goal: within one render, text_block / dst_points / render_params use the same geometry data,
so old data mixed between model and item cannot make the position jump.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .region_geometry_state import is_rect_like, normalize_region_geometry_data


def _resolve_effective_box_local(data: dict):
    custom_box = data.get("white_frame_rect_local")
    render_box = data.get("render_box_rect_local")
    has_custom = bool(data.get("has_custom_white_frame", False))

    # Unbound: once the user has set a white box by hand, the white box always decides the render centre;
    # render_box (the small box the font size is derived from) does not take over.
    if has_custom and is_rect_like(custom_box):
        return custom_box
    if is_rect_like(render_box):
        return render_box
    if is_rect_like(custom_box):
        return custom_box
    return None


@dataclass
class RegionRenderSnapshot:
    region_index: int
    region_data: Dict[str, Any]
    source_center: Tuple[float, float]  # Centre of the source region (world coordinates)
    render_center: Tuple[float, float]  # Centre of the white box (world coordinates), the anchor for placing the text
    white_frame_local: Optional[Tuple[float, float, float, float]]
    white_frame_world: Optional[np.ndarray]  # shape (1, 4, 2)

    @classmethod
    def from_sources(
        cls,
        region_index: int,
        region_data: Optional[dict],
        geo_state: Optional[object] = None,
    ) -> RegionRenderSnapshot:
        data = (
            normalize_region_geometry_data(copy.deepcopy(region_data))
            if isinstance(region_data, dict)
            else {}
        )

        # Merge the current geometry of the item first, so "old values of the model" do not flow back
        if geo_state is not None:
            try:
                data.update(geo_state.to_persisted_state_patch())
                data["center"] = list(geo_state.center)
            except Exception:
                pass

        center = data.get("center", [0.0, 0.0])
        if not (isinstance(center, (list, tuple)) and len(center) >= 2):
            center = [0.0, 0.0]
        cx, cy = float(center[0]), float(center[1])
        source_center = (cx, cy)

        wf_local_raw = _resolve_effective_box_local(data)
        white_frame_local = None
        white_frame_world = None
        render_center = source_center

        if is_rect_like(wf_local_raw):
            left, top, right, bottom = (float(v) for v in wf_local_raw)
            white_frame_local = (left, top, right, bottom)

            theta = math.radians(float(data.get("angle") or 0.0))
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            local_cx = (left + right) / 2.0
            local_cy = (top + bottom) / 2.0
            render_center = (
                cx + local_cx * cos_t - local_cy * sin_t,
                cy + local_cx * sin_t + local_cy * cos_t,
            )

            def _l2w(lx: float, ly: float):
                return [
                    float(cx + lx * cos_t - ly * sin_t),
                    float(cy + lx * sin_t + ly * cos_t),
                ]

            white_frame_world = np.array(
                [
                    [
                        _l2w(left, top),
                        _l2w(right, top),
                        _l2w(right, bottom),
                        _l2w(left, bottom),
                    ]
                ],
                dtype=np.float32,
            )

        # Write the render centre back to the snapshot data; the later pipeline no longer changes center implicitly
        data["center"] = [render_center[0], render_center[1]]

        return cls(
            region_index=region_index,
            region_data=data,
            source_center=source_center,
            render_center=render_center,
            white_frame_local=white_frame_local,
            white_frame_world=white_frame_world,
        )
