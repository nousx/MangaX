"""Helpers for solid-colour bubble filling and block-by-block inpainting."""
from typing import List, Tuple

import cv2
import numpy as np

from ..rendering.ballon_extractor import enlarge_window
from ..utils.bubble import calc_bbox_mask_overlap_ratio
from manga_translator.utils.swallowed import note_ignored_error


MODEL_BUBBLE_SHRINK_RATIO = 0.02


def solid_fill_pure_bubbles(
    img: np.ndarray,
    mask: np.ndarray,
    text_regions: List,
    mask_tight: np.ndarray,
    bubble_mask: np.ndarray,
    overlap_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    For matching solid-colour bubbles, fill directly with the median background colour only inside the intersection of the inpainting mask and the bubble mask, so bubble content outside the inpainting mask is not covered.

    Args:
        img: the RGB or RGBA working image
        mask: the refined (dilated) inpainting mask, the same height and width as img; the filled areas are cleared from it
        text_regions: list of text regions; the existing model bubble overlap logic picks the matching bubbles
        mask_tight: the dilated raw text mask, only used to cut the text pixels out of the bubble and sample the background colour
        bubble_mask: the mask output by the bubble model, already shrunk in proportion, used to identify the matching bubble components
        overlap_threshold: minimum overlap ratio for a text box to count as inside a model bubble

    Returns:
        (filled_img, remaining_mask, filled_region_count); the inputs are not modified.
    """
    filled_img = img.copy()
    remaining_mask = mask.copy()
    rgb = filled_img[:, :, :3] if filled_img.ndim == 3 and filled_img.shape[2] == 4 else filled_img
    tight_bin = np.where(mask_tight >= 127, 255, 0).astype(np.uint8)
    bubble_bin = np.where(bubble_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(bubble_bin):
        return filled_img, remaining_mask, 0

    num_labels, label_map = cv2.connectedComponents(
        np.where(bubble_bin > 0, 1, 0).astype(np.uint8),
        connectivity=8,
    )
    overlap_threshold = max(0.0, min(float(overlap_threshold), 1.0))

    region_bboxes = []
    for region in text_regions:
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in region.xyxy]
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/inpainting/ballon_fill.py:solid_fill_pure_bubbles")
            continue
        if x2 > x1 and y2 > y1:
            region_bboxes.append((x1, y1, x2 - x1, y2 - y1))

    filled_regions = set()
    for label_idx in range(1, num_labels):
        region_bubble = np.where(label_map == label_idx, 255, 0).astype(np.uint8)
        matched_regions = {
            idx for idx, bbox in enumerate(region_bboxes)
            if calc_bbox_mask_overlap_ratio(bbox, region_bubble) >= overlap_threshold
        }
        if not matched_regions:
            continue

        non_text_mask = cv2.bitwise_and(region_bubble, 255 - tight_bin)
        non_text_px = rgb[non_text_mask > 0]
        if not non_text_px.size:
            continue
        average_bg_color = np.median(non_text_px, axis=0)
        std_rgb = np.std(non_text_px - average_bg_color, axis=0)
        inpaint_thresh = 7 if np.std(std_rgb) > 1 else 10
        if np.max(std_rgb) >= inpaint_thresh:
            continue

        # The bubble mask only identifies candidate bubbles; the actual fill is strictly limited to the inpainting mask.
        fill_region = (region_bubble > 0) & (mask > 0)
        if not np.any(fill_region):
            continue
        rgb[fill_region] = np.clip(np.round(average_bg_color), 0, 255).astype(np.uint8)
        remaining_mask[fill_region] = 0
        filled_regions.update(matched_regions)

    return filled_img, remaining_mask, len(filled_regions)


async def inpaint_regions_per_block(img: np.ndarray, remaining_mask: np.ndarray,
                                    inpaint_fn) -> Tuple[np.ndarray, int]:
    """
    Block-by-block inpainting: each isolated connected component of the refined mask after filling
    is inpainted on its own in a window twice the size of its bounding box and pasted back.

    Differences from whole-page inpainting:
    - In a small per-block window the mask takes up a large share, and LaMa inpaints far better than with a long strip mask on a whole page (which leaves ghosts of the text)
    - Only the refined mask of the current component is passed in each time, unaffected by text line boxes and neighbouring masks
    - Image and mask are reflect-padded to a square together: the model gets enough context; the mask is reflected too,
      otherwise the mirrored text would have no mask and the model would paint the text back from the mirror image
    - After padding to a square the ordinary inpainting entry is called; the aspect ratio is 1, so the long-image tiling flow is not entered

    Args:
        inpaint_fn: async (crop, mask) -> inpainted crop
    Returns:
        (result_img, inpainted_block_count). img is not modified;
        the components that were inpainted are cleared in place from remaining_mask.
    """
    result = img.copy()
    im_h, im_w = result.shape[:2]
    mask_bin = np.where(remaining_mask > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    count = 0
    for label_idx in range(1, num_labels):
        x1, y1, w, h, area = map(int, stats[label_idx])
        if area <= 0:
            continue
        x2, y2 = x1 + w, y1 + h
        ex1, ey1, ex2, ey2 = enlarge_window([x1, y1, x2, y2], im_w, im_h, ratio=2.0)
        if ex2 <= ex1 or ey2 <= ey1:
            continue
        msk = np.where(labels[ey1:ey2, ex1:ex2] == label_idx, 255, 0).astype(np.uint8)
        crop = result[ey1:ey2, ex1:ex2].copy()
        ch, cw = crop.shape[:2]
        longer = max(ch, cw)
        pad_bottom, pad_right = longer - ch, longer - cw
        crop_sq = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        msk_sq = cv2.copyMakeBorder(np.ascontiguousarray(msk), 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        out = await inpaint_fn(crop_sq, msk_sq)
        result[ey1:ey2, ex1:ex2] = out[:ch, :cw]
        remaining_view = remaining_mask[y1:y2, x1:x2]
        remaining_view[labels[y1:y2, x1:x2] == label_idx] = 0
        count += 1
    return result, count
