import itertools
import os
from abc import abstractmethod
from collections import Counter
from typing import List, Union

import networkx as nx
import numpy as np

from ..config import OcrConfig
from ..utils import (
    InfererModule,
    ModelWrapper,
    Quadrilateral,
    TextBlock,
    calc_bbox_mask_overlap_ratio,
)
from manga_translator.utils.swallowed import note_ignored_error


class CommonOCR(InfererModule):
    def _generate_text_direction(self, bboxes: List[Union[Quadrilateral, TextBlock]]):
        if len(bboxes) > 0:
            if isinstance(bboxes[0], TextBlock):
                for blk in bboxes:
                    for line_idx in range(len(blk.lines)):
                        yield blk, line_idx
            else:
                from ..utils import quadrilateral_can_merge_region

                G = nx.Graph()
                for i, box in enumerate(bboxes):
                    G.add_node(i, box = box)
                for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2):
                    if quadrilateral_can_merge_region(ubox, vbox, aspect_ratio_tol=1):
                        G.add_edge(u, v)
                for node_set in nx.algorithms.components.connected_components(G):
                    nodes = list(node_set)
                    # majority vote for direction
                    dirs = [box.direction for box in [bboxes[i] for i in nodes]]
                    majority_dir = Counter(dirs).most_common(1)[0][0]
                    # sort
                    if majority_dir == 'h':
                        nodes = sorted(nodes, key = lambda x: bboxes[x].aabb.y + bboxes[x].aabb.h // 2)
                    elif majority_dir == 'v':
                        nodes = sorted(nodes, key = lambda x: -(bboxes[x].aabb.x + bboxes[x].aabb.w))
                    # yield overall bbox and sorted indices
                    for node in nodes:
                        yield bboxes[node], majority_dir

    def _should_ignore_region(self, region_img: np.ndarray, ignore_bubble: float, 
                              full_image: np.ndarray = None, textline: Quadrilateral = None,
                              ocr_config: OcrConfig = None, bubble_mask: np.ndarray = None) -> bool:
        """
        General bubble filter: decides whether a text region should be ignored

        Args:
            region_img: the cropped image of the text region (for the simple method)
            ignore_bubble: threshold for ignoring bubbles (0-1)
            full_image: the full image (optional, for the advanced method)
            textline: the text line object (optional, for its coordinates)
            ocr_config: the OCR configuration (optional, for the model-based bubble filter)
            bubble_mask: the bubble mask in the context of the current original image

        Returns:
            True: should be ignored (not a bubble area)
            False: should be kept (a bubble area)
        """
        from ..utils.bubble import is_ignore

        # Model-based bubble filter (the same stage as ignore_bubble, but based on the detection model)
        use_model_filter = bool(getattr(ocr_config, 'use_model_bubble_filter', False)) if ocr_config is not None else False
        if use_model_filter and full_image is not None and textline is not None:
            bbox = textline.aabb
            text_bbox = (int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h))

            if bubble_mask is not None and np.count_nonzero(bubble_mask) > 0:
                overlap_threshold = float(getattr(ocr_config, 'model_bubble_overlap_threshold', 0.1))
                overlap_threshold = max(0.0, min(1.0, overlap_threshold))
                overlap_ratio = calc_bbox_mask_overlap_ratio(text_bbox, bubble_mask)
                if overlap_ratio < overlap_threshold:
                    self.logger.debug(
                        f"Model bubble filter: overlap={overlap_ratio:.3f} < threshold={overlap_threshold:.3f}, filtering region"
                    )
                    return True
        
        # With the full image and the text line available, use the advanced method
        if full_image is not None and textline is not None:
            # Bounding box of the text line
            bbox = textline.aabb
            x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
            return is_ignore(region_img, ignore_bubble, full_image, [x, y, w, h])
        
        # Otherwise use the simple method
        return is_ignore(region_img, ignore_bubble)

    async def recognize(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False, bubble_mask: np.ndarray = None) -> List[Quadrilateral]:
        '''
        Performs the optical character recognition, using the `textlines` as areas of interests.
        Returns a `textlines` list with the `textline.text` property set to the detected text string.
        '''
        if bool(getattr(config, 'use_model_bubble_filter', False)):
            threshold = float(getattr(config, 'model_bubble_overlap_threshold', 0.1))
            self.logger.info(f"Model bubble filter enabled (overlap_threshold={threshold:.3f})")
        return await self._recognize(image, textlines, config, verbose, bubble_mask=bubble_mask)

    @abstractmethod
    async def _recognize(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False, bubble_mask: np.ndarray = None) -> List[Quadrilateral]:
        pass


class OfflineOCR(CommonOCR, ModelWrapper):
    _MODEL_SUB_DIR = 'ocr'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gpu = False  # Subclasses should set this flag in _load

    async def _recognize(self, *args, **kwargs):
        result = await self.infer(*args, **kwargs)
        return result

    @abstractmethod
    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral], args: OcrConfig, verbose: bool = False, bubble_mask: np.ndarray = None) -> List[Quadrilateral]:
        pass

    def _cleanup_ocr_memory(self, *objects, force_gpu_cleanup: bool = False):
        """
        Shared memory clean-up method of the OCR module

        Args:
            *objects: the objects to delete (variable names or object references)
            force_gpu_cleanup: whether GPU memory is freed by force

        Example:
            # clean up a single object
            self._cleanup_ocr_memory(region)

            # clean up several objects
            self._cleanup_ocr_memory(region, image_tensor, ret)

            # clean up and force GPU clean-up
            self._cleanup_ocr_memory(region, image_tensor, force_gpu_cleanup=True)
        """
#         import gc
        
        # Delete the objects passed in
        for obj in objects:
            try:
                del obj
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/ocr/common.py:OfflineOCR._cleanup_ocr_memory")
                pass
        
        # With a GPU, or when clean-up is forced, free GPU memory
        if force_gpu_cleanup or (hasattr(self, 'use_gpu') and self.use_gpu):
            try:
                import torch
                if torch.cuda.is_available():
                    pass
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/ocr/common.py:OfflineOCR._cleanup_ocr_memory")
                pass
        
        # Light garbage collection (no forced full GC, to avoid a performance cost)
        # The main process runs a full gc.collect() at the end of the batch
        
    def _cleanup_batch_data(self, *data_lists, force_gpu_cleanup: bool = False):
        """
        Clean up batch data (containers such as lists and dictionaries)

        Args:
            *data_lists: the data containers to clean up (list, dict and so on)
            force_gpu_cleanup: whether GPU memory is freed by force

        Example:
            # clean up lists
            self._cleanup_batch_data(region_imgs, quadrilaterals)

            # clean up dictionaries
            self._cleanup_batch_data(out_regions, texts)
        """
#         import gc
        
        for data in data_lists:
            if data is None:
                continue
                
            try:
                if isinstance(data, list):
                    data.clear()
                elif isinstance(data, dict):
                    data.clear()
                del data
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/ocr/common.py:OfflineOCR._cleanup_batch_data")
                pass
        
        # GPU clean-up
        if force_gpu_cleanup or (hasattr(self, 'use_gpu') and self.use_gpu):
            try:
                import torch
                if torch.cuda.is_available():
                    pass
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/ocr/common.py:OfflineOCR._cleanup_batch_data")
                pass

    def _get_ocr_canvas_width(self, valid_widths: List[int], base_align: int = 4, extra_pad: int = 0) -> int:
        """
        Normalize OCR canvas width to reduce dynamic-shape explosion on cuDNN.
        Env vars:
        - MANGA_OCR_FIXED_WIDTH: force a minimum fixed width when > 0
        - MANGA_OCR_WIDTH_BUCKET: round width up to this bucket on GPU (default: 256)
        """
        max_content_width = max(valid_widths) + max(0, int(extra_pad))
        base_align = max(1, int(base_align))
        aligned_width = base_align * ((max_content_width + base_align - 1) // base_align)

        # CPU path keeps the original fine-grained width for lower memory overhead.
        if not (hasattr(self, 'use_gpu') and self.use_gpu):
            return aligned_width

        fixed_width = int(os.environ.get("MANGA_OCR_FIXED_WIDTH", "0") or 0)
        if fixed_width > 0:
            return max(aligned_width, fixed_width)

        bucket = int(os.environ.get("MANGA_OCR_WIDTH_BUCKET", "256") or 1)
        if bucket <= 1:
            return aligned_width
        return bucket * ((aligned_width + bucket - 1) // bucket)

