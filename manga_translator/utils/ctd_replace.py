"""
Detection module dedicated to replace-translation

The same flow as win.py:
1. Call the CTD detector to get the raw mask (skipping the refinement inside the detector)
2. Process the raw mask with ctd_utils/textmask.refine_mask
3. Use REFINEMASK_INPAINT mode (which dilates by 5x5)
"""

from typing import List, Tuple

import cv2
import numpy as np
import torch

from ..detection.ctd import ComicTextDetector
from ..detection.ctd_utils.textmask import REFINEMASK_INPAINT, refine_mask
from ..detection.ctd_utils.utils.db_utils import postprocess_mask
from ..detection.ctd_utils.utils.imgproc_utils import preprocess_img
from ..utils import Quadrilateral, det_rearrange_forward
from manga_translator.utils.swallowed import note_ignored_error


class ReplaceTranslationCTD:
    """
    Wrapper of the CTD detector dedicated to replace-translation

    Exactly the same mask refinement flow as win.py
    """
    
    def __init__(self, detector: ComicTextDetector):
        self.detector = detector
    
    async def detect_with_winpy_refine(
        self,
        image: np.ndarray,
        detect_size: int = 1536,
        text_threshold: float = 0.5,
        box_threshold: float = 0.7,
        unclip_ratio: float = 2.3,
        verbose: bool = False
    ) -> Tuple[List[Quadrilateral], np.ndarray, np.ndarray]:
        """
        Mask refinement flow in the style of win.py

        Returns:
            (textlines, mask_raw, mask_refined)
            - textlines: the detected text lines
            - mask_raw: the raw mask (direct output of the neural network)
            - mask_refined: the refined mask (with REFINEMASK_INPAINT)
        """
        # Get the raw output through the detector's internal method
        im_h, im_w = image.shape[:2]
        
        # Call det_rearrange_forward to get the raw mask
        lines_map, mask = det_rearrange_forward(
            image, 
            self.detector.det_batch_forward_ctd, 
            self.detector.input_size[0], 
            4, 
            self.detector.device, 
            verbose
        )
        
        if lines_map is None:
            img_in, ratio, dw, dh = preprocess_img(
                image, 
                input_size=self.detector.input_size, 
                device=self.detector.device, 
                half=self.detector.half, 
                to_tensor=self.detector.backend=='torch'
            )
            blks, mask, lines_map = self.detector.model(img_in)
            
            if self.detector.backend == 'opencv':
                if mask.shape[1] == 2:
                    tmp = mask
                    mask = lines_map
                    lines_map = tmp
            mask = mask.squeeze()
            mask = mask[..., :mask.shape[0]-dh, :mask.shape[1]-dw]
            lines_map = lines_map[..., :lines_map.shape[2]-dh, :lines_map.shape[3]-dw]
        
        mask = postprocess_mask(mask)
        lines, scores = self.detector.seg_rep(None, lines_map, height=im_h, width=im_w)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)
        lines, scores = lines[0][idx], scores[0][idx]
        
        # Resize the mask to the original image size
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        
        # Create the textlines
        textlines = [Quadrilateral(pts.astype(int), '', score) for pts, score in zip(lines, scores)]
        
        # Key point: refine in REFINEMASK_INPAINT mode (the same as win.py)
        mask_refined = refine_mask(image, mask, textlines, refine_mode=REFINEMASK_INPAINT)
        
        # Free GPU memory
        if self.detector.device.startswith('cuda') or self.detector.device == 'mps':
            try:
                if torch.cuda.is_available():
                    pass
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/utils/ctd_replace.py:ReplaceTranslationCTD.detect_with_winpy_refine")
                pass
        
        # Return the same format as win.py: (textlines, mask_raw, mask_refined)
        return textlines, mask, mask_refined


async def detect_for_replace_translation(
    detector: ComicTextDetector,
    image: np.ndarray,
    detect_size: int = 1536,
    text_threshold: float = 0.5,
    box_threshold: float = 0.7,
    unclip_ratio: float = 2.3,
    verbose: bool = False
) -> Tuple[List[Quadrilateral], np.ndarray, np.ndarray]:
    """
    Detection function dedicated to replace-translation

    Exactly as in win.py: returns (textlines, mask_raw, mask_refined)
    """
    wrapper = ReplaceTranslationCTD(detector)
    return await wrapper.detect_with_winpy_refine(
        image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose
    )

