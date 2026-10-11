from typing import List, Optional

import cv2
import numpy as np

from ..config import Detector
from ..utils import (
    Quadrilateral,
    build_bubble_mask_from_mangalens_result,
    calc_bbox_mask_overlap_ratio,
    detect_bubbles_with_mangalens,
)
from ..utils.log import get_logger
from .common import CommonDetector, OfflineDetector
from .craft import CRAFTDetector
from .ctd import ComicTextDetector
from .dbnet_convnext import DBConvNextDetector
from .default import DefaultDetector

# from .paddle_rust import PaddleDetector  # removed
from .none import NoneDetector
from .yolo_obb import YOLOOBBDetector

DETECTORS = {
    Detector.default: DefaultDetector,
    Detector.dbconvnext: DBConvNextDetector,
    Detector.ctd: ComicTextDetector,
    Detector.craft: CRAFTDetector,
    # Detector.paddle: PaddleDetector,  # removed
    Detector.none: NoneDetector,
}
detector_cache = {}

def get_detector(key: Detector, *args, **kwargs) -> CommonDetector:
    if key not in DETECTORS:
        raise ValueError(f'Could not find detector for: "{key}". Choose from the following: %s' % ','.join(DETECTORS))
    if not detector_cache.get(key):
        detector = DETECTORS[key]
        detector_cache[key] = detector(*args, **kwargs)
    return detector_cache[key]

async def prepare(detector_key: Detector):
    detector = get_detector(detector_key)
    if isinstance(detector, OfflineDetector):
        await detector.download()

async def dispatch(detector_key: Detector, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float, unclip_ratio: float,
                   device: str = 'cpu', verbose: bool = False,
                   use_yolo_obb: bool = False, yolo_obb_conf: float = 0.4, yolo_obb_overlap_threshold: float = 0.1, min_box_area_ratio: float = 0.0009,
                   result_path_fn=None, det_rearrange_min_effective_short_side: float = 341.0,
                   use_sfx_filter: bool = False, sfx_filter_include_bubble_text: bool = False,
                   bubble_mask: Optional[np.ndarray] = None):
    """
    检测调度函数，支持混合检测模式
    
    Args:
        use_yolo_obb: 是否启用YOLO OBB辅助检测器
        use_sfx_filter: 是否过滤既未被 other 包裹、也未与 YOLO 文本框重叠的主检测框
        sfx_filter_include_bubble_text: 是否让气泡内文本也参与拟声词过滤
        yolo_obb_conf: YOLO OBB检测器的置信度阈值
        min_box_area_ratio: 最小检测框面积占比（相对图片总像素）
        result_path_fn: 结果路径生成函数（用于保存调试图）
        det_rearrange_min_effective_short_side: 长图检测重排后的最低有效短边分辨率
    """
    # Run the main detector
    detector = get_detector(detector_key)
    if isinstance(detector, OfflineDetector):
        await detector.load(device)
    main_textlines, mask, raw_image = await detector.detect(
        image,
        detect_size,
        text_threshold,
        box_threshold,
        unclip_ratio,
        verbose,
        min_box_area_ratio,
        result_path_fn,
        det_rearrange_min_effective_short_side,
    )
    
    # Without YOLO OBB, return the result of the main detector directly
    if not use_yolo_obb:
        return main_textlines, mask, raw_image
    
    # YOLO OBB auxiliary detection
    try:
        yolo_detector = get_detector_instance('yolo_obb', YOLOOBBDetector)
        await yolo_detector.load(device)
        
        # YOLO OBB detection (yolo_obb_conf is used as text_threshold)
        yolo_textlines, _, _ = await yolo_detector.detect(
            image, detect_size, yolo_obb_conf, box_threshold, unclip_ratio,
            verbose, min_box_area_ratio, result_path_fn,
            det_rearrange_min_effective_short_side,
        )
        
        # Smart merge: a YOLO box can replace a main detector box that is too small, or add a new box
        combined_textlines = merge_detection_boxes(
            yolo_textlines,
            main_textlines,
            overlap_threshold=yolo_obb_overlap_threshold,
            use_sfx_filter=use_sfx_filter,
            sfx_filter_include_bubble_text=sfx_filter_include_bubble_text,
            image=image,
            bubble_mask=bubble_mask,
        )
        
        replaced_count = len(main_textlines) + len(yolo_textlines) - len(combined_textlines)
        detector.logger.info(f"Hybrid detection: primary detector={len(main_textlines)}, YOLO OBB={len(yolo_textlines)}, replaced/removed={replaced_count}, total={len(combined_textlines)}")
        
        # Build the debug image (when verbose=True)
        debug_img = None
        if verbose:
            debug_img = draw_detection_debug_image(image, main_textlines, yolo_textlines, yolo_obb_overlap_threshold)
            detector.logger.info("Hybrid detection debug image generated")
        
        return combined_textlines, mask, debug_img if debug_img is not None else raw_image
    
    except Exception as e:
        detector.logger.error(f"YOLO OBB auxiliary detection failed: {e}")
        # On failure, return the result of the main detector
        return main_textlines, mask, raw_image


def get_detector_instance(key: str, detector_class):
    """获取或创建检测器实例（用于辅助检测器）"""
    if key not in detector_cache:
        detector_cache[key] = detector_class()
    return detector_cache[key]

def _get_box_label(box: Quadrilateral) -> Optional[str]:
    label = getattr(box, 'det_label', None)
    if label is None:
        label = getattr(box, 'yolo_label', None)
    if isinstance(label, str):
        label = label.strip().lower()
        if label:
            return label
    return None

def _apply_yolo_label_infection(main_boxes: List[Quadrilateral], yolo_boxes: List[Quadrilateral], min_overlap_ratio: float = 0.05) -> None:
    """
    让每个 YOLO 框按照重叠率“感染”一个主检测器框标签（最多感染一个）。
    只写入标签，不改动框几何。
    """
    if not main_boxes or not yolo_boxes:
        return

    for yolo_box in yolo_boxes:
        yolo_label = _get_box_label(yolo_box)
        if not yolo_label:
            continue
        # "other" only helps with enclosure and takes no part in label spreading
        if yolo_label == 'other':
            continue

        yolo_min_x = np.min(yolo_box.pts[:, 0])
        yolo_max_x = np.max(yolo_box.pts[:, 0])
        yolo_min_y = np.min(yolo_box.pts[:, 1])
        yolo_max_y = np.max(yolo_box.pts[:, 1])
        yolo_area = (yolo_max_x - yolo_min_x) * (yolo_max_y - yolo_min_y)
        if yolo_area <= 0:
            continue

        best_idx = -1
        best_overlap = 0.0
        for idx, main_box in enumerate(main_boxes):
            main_min_x = np.min(main_box.pts[:, 0])
            main_max_x = np.max(main_box.pts[:, 0])
            main_min_y = np.min(main_box.pts[:, 1])
            main_max_y = np.max(main_box.pts[:, 1])
            main_area = (main_max_x - main_min_x) * (main_max_y - main_min_y)
            if main_area <= 0:
                continue

            inter_min_x = max(yolo_min_x, main_min_x)
            inter_max_x = min(yolo_max_x, main_max_x)
            inter_min_y = max(yolo_min_y, main_min_y)
            inter_max_y = min(yolo_max_y, main_max_y)
            inter_w = inter_max_x - inter_min_x
            inter_h = inter_max_y - inter_min_y
            if inter_w <= 0 or inter_h <= 0:
                continue

            inter_area = inter_w * inter_h
            overlap_ratio = inter_area / min(yolo_area, main_area)
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_idx = idx

        if best_idx >= 0 and best_overlap >= min_overlap_ratio:
            infected_box = main_boxes[best_idx]
            infected_box.det_label = yolo_label
            infected_box.yolo_label = yolo_label
            infected_box.yolo_infected = True


def _box_aabb(box: Quadrilateral):
    min_x = float(np.min(box.pts[:, 0]))
    max_x = float(np.max(box.pts[:, 0]))
    min_y = float(np.min(box.pts[:, 1]))
    max_y = float(np.max(box.pts[:, 1]))
    return min_x, max_x, min_y, max_y


def _contains_interval(outer_min: float, outer_max: float, inner_min: float, inner_max: float, eps: float = 2.0) -> bool:
    return outer_min <= inner_min + eps and outer_max >= inner_max - eps


def _aabb_contains(outer_aabb, inner_aabb, eps: float = 2.0) -> bool:
    ox1, ox2, oy1, oy2 = outer_aabb
    ix1, ix2, iy1, iy2 = inner_aabb
    return (
        _contains_interval(ox1, ox2, ix1, ix2, eps=eps)
        and _contains_interval(oy1, oy2, iy1, iy2, eps=eps)
    )


def _box_direction(box: Quadrilateral) -> Optional[str]:
    direction = getattr(box, 'assigned_direction', None) or getattr(box, 'direction', None)
    if isinstance(direction, str):
        direction = direction.strip().lower()
    return direction if direction in ('h', 'v') else None


def _is_axis_wrapped_pair(box_a: Quadrilateral, box_b: Quadrilateral, eps: float = 2.0) -> bool:
    """
    主轴包裹判定（按用户规则）：
    - 竖排(v)：看 X 区间是否一方包含另一方
    - 横排(h)：看 Y 区间是否一方包含另一方
    """
    dir_a = _box_direction(box_a)
    dir_b = _box_direction(box_b)
    if dir_a is None or dir_b is None or dir_a != dir_b:
        return False

    a_min_x, a_max_x, a_min_y, a_max_y = _box_aabb(box_a)
    b_min_x, b_max_x, b_min_y, b_max_y = _box_aabb(box_b)

    if dir_a == 'v':
        return (
            _contains_interval(a_min_x, a_max_x, b_min_x, b_max_x, eps=eps)
            or _contains_interval(b_min_x, b_max_x, a_min_x, a_max_x, eps=eps)
        )

    # dir_a == 'h'
    return (
        _contains_interval(a_min_y, a_max_y, b_min_y, b_max_y, eps=eps)
        or _contains_interval(b_min_y, b_max_y, a_min_y, a_max_y, eps=eps)
    )


def _aabb_overlap_ratio(box_a: Quadrilateral, box_b: Quadrilateral) -> float:
    """返回两个 AABB 交集占较小框面积的比例。"""
    a_min_x, a_max_x, a_min_y, a_max_y = _box_aabb(box_a)
    b_min_x, b_max_x, b_min_y, b_max_y = _box_aabb(box_b)
    a_area = max(0.0, a_max_x - a_min_x) * max(0.0, a_max_y - a_min_y)
    b_area = max(0.0, b_max_x - b_min_x) * max(0.0, b_max_y - b_min_y)
    min_area = min(a_area, b_area)
    if min_area <= 0:
        return 0.0

    inter_w = max(0.0, min(a_max_x, b_max_x) - max(a_min_x, b_min_x))
    inter_h = max(0.0, min(a_max_y, b_max_y) - max(a_min_y, b_min_y))
    return (inter_w * inter_h) / min_area


def _detect_sfx_bubble_mask(image: np.ndarray) -> Optional[np.ndarray]:
    """用 MangaLens 生成全图气泡掩码；失败时返回 None，此时不做气泡豁免。"""
    try:
        result = detect_bubbles_with_mangalens(image, return_annotated=False, verbose=False)
        return build_bubble_mask_from_mangalens_result(result, image.shape[:2])
    except Exception as exc:
        get_logger('sfx_filter').warning(
            f'MangaLens bubble detection failed, no bubble exemption for this image: {exc}')
        return None


def _get_sfx_filtered_main_indices(
    main_boxes: List[Quadrilateral],
    yolo_boxes: List[Quadrilateral],
    overlap_threshold: float,
    wrap_eps: float = 2.0,
    image: Optional[np.ndarray] = None,
    model_bubble_overlap_threshold: float = 0.1,
    sfx_filter_include_bubble_text: bool = False,
    bubble_mask: Optional[np.ndarray] = None,
) -> set[int]:
    """
    找出缺少 YOLO 支持的主检测框：
    - YOLO `other` 必须完整包裹主框；或
    - 任一非 `other` YOLO 框与主框的重叠率达到阈值。
    两项均不满足时，再用 MangaLens 模型掩码判定是否在气泡内；气泡内文本仍保留。
    sfx_filter_include_bubble_text=True 时跳过气泡保护。
    """
    # Even when the user sets the merge threshold to 0, a real intersection is still required, so an arbitrary YOLO box
    # does not let every main detector box on the page pass the filter.
    threshold = max(1e-6, min(1.0, float(overlap_threshold)))
    filtered_indices = set()
    bubble_mask_ready = bubble_mask is not None

    for main_idx, main_box in enumerate(main_boxes):
        main_aabb = _box_aabb(main_box)
        supported = False
        for yolo_box in yolo_boxes:
            yolo_label = _get_box_label(yolo_box)
            if yolo_label == 'other':
                if _aabb_contains(_box_aabb(yolo_box), main_aabb, eps=wrap_eps):
                    supported = True
                    break
                continue

            if _aabb_overlap_ratio(main_box, yolo_box) >= threshold:
                supported = True
                break

        if not supported and not sfx_filter_include_bubble_text and image is not None:
            if not bubble_mask_ready:
                bubble_mask = _detect_sfx_bubble_mask(image)
                bubble_mask_ready = True
            if bubble_mask is not None:
                min_x, max_x, min_y, max_y = main_aabb
                image_h, image_w = image.shape[:2]
                x1 = max(0, int(np.floor(min_x)))
                y1 = max(0, int(np.floor(min_y)))
                x2 = min(image_w, int(np.ceil(max_x)))
                y2 = min(image_h, int(np.ceil(max_y)))
                if x2 > x1 and y2 > y1:
                    supported = calc_bbox_mask_overlap_ratio(
                        (x1, y1, x2 - x1, y2 - y1), bubble_mask
                    ) >= model_bubble_overlap_threshold

        if not supported:
            filtered_indices.add(main_idx)

    return filtered_indices


def merge_detection_boxes(
    yolo_boxes: List[Quadrilateral],
    main_boxes: List[Quadrilateral],
    overlap_threshold: float = 0.1,
    use_sfx_filter: bool = False,
    image: Optional[np.ndarray] = None,
    sfx_filter_include_bubble_text: bool = False,
    bubble_mask: Optional[np.ndarray] = None,
) -> List[Quadrilateral]:
    """
    合并主检测器和YOLO检测器的框，智能替换逻辑：
    0. 高优先级结构替换：
       若同方向主框中存在“主轴包裹”成对关系（竖排看X、横排看Y），且某个YOLO框完整覆盖该对主框，
       则直接用该YOLO框替换这对主框。
    1. 如果YOLO框与主检测器框重叠
    2. 且YOLO框完全包含主检测器框
    3. 且YOLO框面积 >= 主检测器框面积 * 2
    4. 且YOLO框与其他未替换的主检测器框的重叠率 < overlap_threshold
    5. 则删除被包含的主检测器框，使用YOLO框替代
    6. 其他情况：如果重叠率 >= overlap_threshold，删除重叠的YOLO框，保留主检测器框
    7. 不重叠或重叠率 < overlap_threshold 的YOLO框直接添加
    
    Args:
        yolo_boxes: YOLO OBB检测器的检测框
        main_boxes: 主检测器的检测框
        overlap_threshold: 重叠率阈值（0.0-1.0）。重叠率 >= 该值时删除YOLO框。设为1.0则保留所有框。
        use_sfx_filter: 过滤既未被 YOLO other 框包裹、也未与其他 YOLO 框达到重叠阈值的主检测框。
        image: 原图，供 MangaLens 气泡掩码判断未获 YOLO 支持的主框；模型失败时不做气泡豁免。
        sfx_filter_include_bubble_text: 让气泡内文本也参与拟声词过滤。
    
    Returns:
        合并后的检测框列表
    """
    if len(main_boxes) == 0:
        return yolo_boxes
    
    # Spread the labels first: each YOLO box labels at most one main box
    _apply_yolo_label_infection(main_boxes, yolo_boxes, min_overlap_ratio=max(0.01, overlap_threshold * 0.5))
    
    # Indexes of main detector boxes to remove. The sound-effect filter only applies to main detector boxes;
    # the YOLO boxes themselves still follow the replace and de-duplicate rules below.
    main_boxes_to_remove = (
        _get_sfx_filtered_main_indices(
            main_boxes,
            yolo_boxes,
            overlap_threshold,
            image=image,
            sfx_filter_include_bubble_text=sfx_filter_include_bubble_text,
            bubble_mask=bubble_mask,
        )
        if use_sfx_filter
        else set()
    )
    # Indexes of YOLO boxes to remove
    yolo_boxes_to_remove = set()
    # YOLO boxes to add (as replacements)
    yolo_boxes_to_add_set = set()  # A set, to avoid duplicates
    # Tolerance for coordinate tests (pixels)
    axis_eps = 2.0

    # Rule 0: high-priority structural replacement (before the existing area ratio rule)
    # Only applies when a YOLO box fully covers a pair of main boxes and the pair satisfies "main axis enclosure".
    for yolo_idx, yolo_box in enumerate(yolo_boxes):
        yolo_label = _get_box_label(yolo_box)
        # As in the existing logic: "other" takes no part in geometric replacement
        if yolo_label == 'other':
            continue

        yolo_aabb = _box_aabb(yolo_box)
        covered_main_indices = []
        for main_idx, main_box in enumerate(main_boxes):
            if _aabb_contains(yolo_aabb, _box_aabb(main_box), eps=axis_eps):
                covered_main_indices.append(main_idx)

        if len(covered_main_indices) < 2:
            continue

        wrapped_pair_indices = set()
        for i in range(len(covered_main_indices)):
            for j in range(i + 1, len(covered_main_indices)):
                idx_i = covered_main_indices[i]
                idx_j = covered_main_indices[j]
                if _is_axis_wrapped_pair(main_boxes[idx_i], main_boxes[idx_j], eps=axis_eps):
                    wrapped_pair_indices.add(idx_i)
                    wrapped_pair_indices.add(idx_j)

        if len(wrapped_pair_indices) >= 2:
            main_boxes_to_remove.update(wrapped_pair_indices)
            yolo_boxes_to_add_set.add(yolo_idx)
    
    for yolo_idx, yolo_box in enumerate(yolo_boxes):
        # YOLO boxes that rule 0 marked as strong replacements skip the area and overlap tests below
        if yolo_idx in yolo_boxes_to_add_set:
            continue

        yolo_label = _get_box_label(yolo_box)
        # "other" only helps with enclosure: it takes no part in the geometric decision to replace or remove main boxes
        if yolo_label == 'other':
            continue

        # AABB and area of the YOLO box
        yolo_min_x = np.min(yolo_box.pts[:, 0])
        yolo_max_x = np.max(yolo_box.pts[:, 0])
        yolo_min_y = np.min(yolo_box.pts[:, 1])
        yolo_max_y = np.max(yolo_box.pts[:, 1])
        yolo_area = (yolo_max_x - yolo_min_x) * (yolo_max_y - yolo_min_y)
        
        # Check whether this YOLO box meets any replacement condition
        can_replace = False
        max_overlap_ratio_with_others = 0.0  # Largest overlap ratio with the other main boxes that are not replaced
        replaced_main_indices = set()  # Indexes of the main boxes this YOLO box replaces
        contained_main_boxes_total_area = 0.0  # Total area of the fully contained main boxes
        
        for main_idx, main_box in enumerate(main_boxes):
            # AABB and area of the main detector box
            main_min_x = np.min(main_box.pts[:, 0])
            main_max_x = np.max(main_box.pts[:, 0])
            main_min_y = np.min(main_box.pts[:, 1])
            main_max_y = np.max(main_box.pts[:, 1])
            main_area = (main_max_x - main_min_x) * (main_max_y - main_min_y)
            
            # Check for overlap
            if not (yolo_max_x < main_min_x or yolo_min_x > main_max_x or
                    yolo_max_y < main_min_y or yolo_min_y > main_max_y):
                # They overlap: work out the overlap area
                inter_min_x = max(yolo_min_x, main_min_x)
                inter_max_x = min(yolo_max_x, main_max_x)
                inter_min_y = max(yolo_min_y, main_min_y)
                inter_max_y = min(yolo_max_y, main_max_y)
                inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
                
                # Overlap ratio (relative to the smaller box)
                overlap_ratio = inter_area / min(yolo_area, main_area) if min(yolo_area, main_area) > 0 else 0
                
                # Check whether the YOLO box fully contains the main detector box
                contains = (yolo_min_x <= main_min_x and yolo_max_x >= main_max_x and
                           yolo_min_y <= main_min_y and yolo_max_y >= main_max_y)
                
                if contains:
                    # The YOLO box fully contains this main detector box
                    replaced_main_indices.add(main_idx)
                    contained_main_boxes_total_area += main_area
                else:
                    # Not fully contained: record the overlap ratio with the other main boxes
                    max_overlap_ratio_with_others = max(max_overlap_ratio_with_others, overlap_ratio)
        
        # Area condition: YOLO box area >= total area of the contained main boxes x 2
        if len(replaced_main_indices) > 0:
            area_ratio = yolo_area / contained_main_boxes_total_area if contained_main_boxes_total_area > 0 else 0
            if area_ratio >= 2.0:
                can_replace = True
        
        # Decide what happens to this YOLO box
        if len(replaced_main_indices) > 0:
            # The YOLO box contains at least one main detector box
            if can_replace:
                # The replacement condition holds (area >= 2x), but the overlap with the main boxes it does not contain still has to be checked
                # A box that meets the 2x area condition is allowed a higher overlap ratio (threshold + 0.1)
                adjusted_threshold = overlap_threshold + 0.1
                if max_overlap_ratio_with_others >= adjusted_threshold:
                    # The overlap with other main boxes is too high: drop this YOLO box without replacing anything
                    yolo_boxes_to_remove.add(yolo_idx)
                else:
                    # Safe to replace: remove the replaced main boxes and add the YOLO box
                    for main_idx in replaced_main_indices:
                        main_boxes_to_remove.add(main_idx)
                    yolo_boxes_to_add_set.add(yolo_idx)
            else:
                # It contains main boxes but the area condition fails (< 2x): the YOLO box is probably a wrong detection, drop it
                yolo_boxes_to_remove.add(yolo_idx)
        else:
            # The YOLO box does not fully contain any main detector box: handle it with the original logic
            if max_overlap_ratio_with_others >= overlap_threshold:
                # It overlaps with a ratio >= the threshold: drop this YOLO box
                yolo_boxes_to_remove.add(yolo_idx)
            # else: no overlap, or the ratio is below the threshold; it is added as a new box later
    
    # Build the final result
    result = []
    
    # Add the main detector boxes that were not removed
    for idx, main_box in enumerate(main_boxes):
        if idx not in main_boxes_to_remove:
            result.append(main_box)
    
    # Add the YOLO boxes (replacements + new boxes without overlap)
    for idx, yolo_box in enumerate(yolo_boxes):
        # Add it unless it is in the removal list (covers both replacement boxes and new boxes)
        if idx not in yolo_boxes_to_remove:
            if not hasattr(yolo_box, 'det_label'):
                yolo_label = _get_box_label(yolo_box)
                if yolo_label:
                    yolo_box.det_label = yolo_label
            result.append(yolo_box)
    
    return result

def draw_detection_debug_image(image: np.ndarray, main_boxes: List[Quadrilateral], yolo_boxes: List[Quadrilateral], overlap_threshold: float = 0.1) -> np.ndarray:
    """
    绘制检测框调试图片，并标注重叠率
    
    Args:
        image: 原始图像
        main_boxes: 主检测器的检测框
        yolo_boxes: YOLO检测器的检测框
        overlap_threshold: 重叠率阈值
    
    Returns:
        绘制了检测框的调试图片
    """
    # Make a copy of the image
    debug_img = image.copy()
    
    # Draw the boxes of the main detector (green)
    for box in main_boxes:
        pts = box.pts.astype(np.int32)
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
        # Add the label
        cv2.putText(debug_img, "Main", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw the boxes of the YOLO detector (blue) and work out the overlap ratio
    for yolo_idx, yolo_box in enumerate(yolo_boxes):
        yolo_label = _get_box_label(yolo_box) or 'unknown'
        # "other" only helps with enclosure and is not drawn in this debug image
        if yolo_label == 'other':
            continue
        pts = yolo_box.pts.astype(np.int32)
        
        # Largest overlap ratio with the main detector boxes
        yolo_min_x = np.min(yolo_box.pts[:, 0])
        yolo_max_x = np.max(yolo_box.pts[:, 0])
        yolo_min_y = np.min(yolo_box.pts[:, 1])
        yolo_max_y = np.max(yolo_box.pts[:, 1])
        yolo_area = (yolo_max_x - yolo_min_x) * (yolo_max_y - yolo_min_y)
        
        max_overlap_ratio = 0.0
        for main_box in main_boxes:
            main_min_x = np.min(main_box.pts[:, 0])
            main_max_x = np.max(main_box.pts[:, 0])
            main_min_y = np.min(main_box.pts[:, 1])
            main_max_y = np.max(main_box.pts[:, 1])
            main_area = (main_max_x - main_min_x) * (main_max_y - main_min_y)
            
            # Check for overlap
            if not (yolo_max_x < main_min_x or yolo_min_x > main_max_x or
                    yolo_max_y < main_min_y or yolo_min_y > main_max_y):
                # Overlap area
                inter_min_x = max(yolo_min_x, main_min_x)
                inter_max_x = min(yolo_max_x, main_max_x)
                inter_min_y = max(yolo_min_y, main_min_y)
                inter_max_y = min(yolo_max_y, main_max_y)
                inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
                
                # Overlap ratio
                overlap_ratio = inter_area / min(yolo_area, main_area) if min(yolo_area, main_area) > 0 else 0
                max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)
        
        # Choose the colour by overlap ratio (RGB)
        if max_overlap_ratio >= overlap_threshold:
            # Above the threshold: red (it will be removed)
            color = (255, 0, 0)  # Red
            label = f"YOLO({yolo_label}):{max_overlap_ratio:.2f}(X)"
        else:
            # Below the threshold: blue (it will be kept)
            color = (0, 0, 255)  # Blue
            label = f"YOLO({yolo_label}):{max_overlap_ratio:.2f}"
        
        cv2.polylines(debug_img, [pts], True, color, 2)
        cv2.putText(debug_img, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Write the threshold at the top of the image
    info_text = f"Overlap Threshold: {overlap_threshold:.2f} | Green=Main, Blue=YOLO(Keep), Red=YOLO(Removed)"
    cv2.putText(debug_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Note on the image: the YOLO boxes have already been de-duplicated by NMS
    note_text = "Note: YOLO boxes are already NMS-filtered"
    cv2.putText(debug_img, note_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(debug_img, note_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return debug_img

async def unload(detector_key: Detector):
    detector = detector_cache.pop(detector_key, None)
    if isinstance(detector, OfflineDetector):
        await detector.unload()

    # YOLO OBB, as an auxiliary detector, is cached under a string key and released together with the main detector.
    yolo_detector = detector_cache.pop('yolo_obb', None)
    if isinstance(yolo_detector, OfflineDetector):
        await yolo_detector.unload()
