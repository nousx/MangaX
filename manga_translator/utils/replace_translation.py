"""
Replace-translation module - applies the OCR result of a translated image to the raw image

What it does:
1. Run detection + OCR on the raw image and filter out low-confidence regions
2. Run detection + OCR on the translated image, keeping the sub-box information for line breaking
3. Region matching: align the coordinates by size scaling and compute the overlapping areas
4. Filtering: remove the regions that only the raw image or only the translated image has
5. Merging: the OCR result of the translated image becomes the translation field
6. Optional: template matching alignment - computes the offset between the two images automatically and adjusts the text position

Use cases:
- Different editions of the same manga (such as a restored edition and the original)
- Versions in different resolutions
- Moving a translation over when there is no JSON data

Author: manga-translator-ui
Date: 2026-01-01
"""

import asyncio
import logging
import os
import traceback
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..image_formats import SUPPORTED_IMAGE_EXTENSIONS
from .generic import Context, dump_image, imwrite_unicode, open_pil_image, save_pil_image
from .path_manager import TRANSLATED_IMAGES_SUBDIR, get_work_dir
from .textblock import TextBlock
from manga_translator.utils.swallowed import note_ignored_error

logger = logging.getLogger(__name__)


def get_text_to_img_solid_ink(mask_final, cn_text_img, origin_img, mengban=8, pan=150):
    """
    Text compositing function for a very sharp result.
    Core logic: Lanczos4 interpolation + levels compression, to restore the sharpness of the source image as far as possible.

    Args:
        mask_final: the mask image
        cn_text_img: the translated image (the one the text is taken from)
        origin_img: the raw image (the base)
        mengban: size of the mask dilation kernel
        pan: offset of the white point threshold

    Returns:
        The composited image
    """
    # --- 1. Basic checks ---
    if mask_final is None or cn_text_img is None or origin_img is None:
        raise ValueError('Input image is empty')

    h, w = origin_img.shape[:2]

    # --- 2. Scaling ---
    # The mask uses nearest-neighbour interpolation, to keep its edges sharp
    if mask_final.shape[:2] != (h, w):
        mask_final = cv2.resize(mask_final, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # The text uses linear interpolation, for softer and more natural edges
    if cn_text_img.shape[:2] != (h, w):
        cn_text_img = cv2.resize(cn_text_img, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # --- 3. Prepare the mask area ---
    if len(mask_final.shape) == 3:
        mask_gray = cv2.cvtColor(mask_final, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_final
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Safe area (limits where text may go) - the original mask is used as it is, without extra dilation
    mask_safe_zone = mask_binary

    # --- 4. Levels enhancement of the text image ---
    # Like the Levels tool of Photoshop: set a black point and a white point and stretch linearly in between
    if len(cn_text_img.shape) == 3:
        text_gray = cv2.cvtColor(cn_text_img, cv2.COLOR_BGR2GRAY)
    else:
        text_gray = cn_text_img

    # Settings - lower contrast, keeping more of the grey transitions
    in_black = 50   # Black point threshold: lowered to keep the grey at the edges
    in_white = max(in_black + 80, pan + 50) # White point threshold: a wider range for softer transitions
    if in_white > 255: 
        in_white = 255
    
    # Build a lookup table (LUT) for fast levels mapping
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i <= in_black:
            val = 0 # pure black
        elif i >= in_white:
            val = 255 # pure white
        else:
            # Stretch the greys in between linearly
            val = int((i - in_black) / (in_white - in_black) * 255)
        lut[i] = val
        
    # Apply the levels adjustment
    text_enhanced_gray = cv2.LUT(text_gray, lut)
    
    # A slight Gaussian blur, to soften the edges further
    text_enhanced_gray = cv2.GaussianBlur(text_enhanced_gray, (3, 3), 0.5)
    
    # Convert back to BGR for merging
    text_enhanced_bgr = cv2.cvtColor(text_enhanced_gray, cv2.COLOR_GRAY2BGR)

    # --- 5. Composite ---
    result_img = origin_img.copy()
    
    # 5.1 Multiply (bitwise AND) directly inside the safe area
    # origin_img (img_inpainted) has already been erased by inpainting, so no white fill is needed
    roi_bg = result_img[mask_safe_zone == 255]
    roi_text = text_enhanced_bgr[mask_safe_zone == 255]
    
    # Blend: white background (255) & text -> text
    fused = cv2.bitwise_and(roi_bg, roi_text)
    
    result_img[mask_safe_zone == 255] = fused

    return result_img


async def translate_batch_replace_translation(translator, images_with_configs: List[tuple], save_info: dict = None, global_offset: int = 0, global_total: int = None) -> List[Context]:
    """
    Replace-translation mode: extract the OCR result from the translated image and apply it to the raw image

    Flow:
    1. Run detection + OCR on the raw image and filter out low-confidence regions
    2. Find the matching translated image and run detection + OCR on it
    3. Match the regions (taking size scaling into account)
    4. Inpaint and render with the matched regions

    Args:
        translator: the MangaTranslator instance
        images_with_configs: List of (image, config) tuples
        save_info: the save settings
        global_offset: global offset
        global_total: global total number of images
    """
    logger.info(f"Starting replace translation mode with {len(images_with_configs)} images")
    results = []
    
    display_total = global_total if global_total is not None else len(images_with_configs)
    failed_count = 0
    
    for idx, (image, config) in enumerate(images_with_configs):
        # ✅ Check the stop flag
        await asyncio.sleep(0)
        translator._check_cancelled()

        raw_ctx = None
        translated_ctx = None
        translated_image = None
        image_name = image if isinstance(image, str) else (image.name if hasattr(image, 'name') else f"image_{idx}")
        global_idx = global_offset + idx + 1
        loaded_source_image = False

        if isinstance(image, str):
            try:
                with open(image, 'rb') as f:
                    loaded_image = open_pil_image(f, eager=True)
                loaded_image.name = image
                image = loaded_image
                loaded_source_image = True
            except Exception as e:
                logger.error(f"  Failed to load original image: {image_name} - {e}")
                ctx = Context()
                ctx.image_name = image_name
                ctx.text_regions = []
                ctx.success = False
                ctx.translation_error = str(e)
                results.append(ctx)
                failed_count += 1
                await translator._report_progress(f"batch:{global_idx}:{global_idx}:{display_total}:{failed_count}")
                continue
        
        # Force AI line breaking and strict box mode on
        if config and hasattr(config, 'render'):
            config.render.disable_auto_wrap = True
            config.render.layout_mode = 'strict'
            
            # Mark as replace-translation mode, so the renderer recognises it and applies its special logic (such as forcing a single line without wrapping)
            if hasattr(config, 'cli'):
                config.cli.replace_translation = True
            
            logger.info("Replace translation mode: Forced disable_auto_wrap=True, layout_mode='strict', replace_translation=True")
        
        logger.info(f"[{global_idx}/{display_total}] Processing: {os.path.basename(image_name)}")
        
        try:
            translator._set_image_context(config, image)
            
            # === Step 1: find the translated image ===
            translated_path = find_translated_image(image_name)
            if not translated_path:
                logger.warning(f"  [Skip] No corresponding translated image found: {os.path.basename(image_name)}")
                ctx = Context()
                ctx.input = image
                ctx.image_name = image_name
                ctx.text_regions = []
                ctx.success = False
                results.append(ctx)
                failed_count += 1
                await translator._report_progress(f"batch:{global_idx}:{global_idx}:{display_total}:{failed_count}")
                continue
            
            logger.info(f"  Found translated image: {os.path.basename(translated_path)}")
            
            # === Step 2: run detection + OCR on the raw image ===
            logger.info("  [1/4] Detecting text and running OCR on the original image...")
            # ✅ Check the stop flag
            await asyncio.sleep(0)
            translator._check_cancelled()
            
            raw_ctx = await translator._translate_until_translation(image, config)
            raw_ctx.image_name = image_name
            # Keep the original image size (for saving the JSON)
            if hasattr(image, 'size'):
                raw_ctx.original_size = image.size

            if not raw_ctx.text_regions:
                logger.warning("  [Skip] No text regions detected in the original image, outputting it unchanged")
                # Set result to the original image
                raw_ctx.result = image
                raw_ctx.text_regions = []
                # Jump to the save step (without continue)
                skip_to_save = True
            else:
                skip_to_save = False
            
            if not skip_to_save:
                # Filter out low-confidence regions
                min_prob = config.ocr.prob if hasattr(config.ocr, 'prob') and config.ocr.prob else 0.1
                raw_regions_filtered = [r for r in raw_ctx.text_regions if getattr(r, 'prob', 1.0) >= min_prob]
                logger.info(f"    Original image regions: {len(raw_ctx.text_regions)} -> after filtering: {len(raw_regions_filtered)}")
                
                if not raw_regions_filtered:
                    logger.warning("  [Skip] No valid regions remain after filtering, outputting the original image unchanged")
                    # Set result to the original image
                    raw_ctx.result = image
                    raw_ctx.text_regions = []
                    skip_to_save = True
            
            if not skip_to_save:
                # Record the size of the raw image
                raw_size = (raw_ctx.img_rgb.shape[1], raw_ctx.img_rgb.shape[0]) if raw_ctx.img_rgb is not None else (image.width, image.height)
                
                # === Step 3: run detection + OCR on the translated image ===
                logger.info("  [2/4] Detecting text and running OCR on the translated image...")
                # ✅ Check the stop flag
                await asyncio.sleep(0)
                translator._check_cancelled()
                
                translated_image = open_pil_image(translated_path, eager=False)
                translated_image.name = translated_path
                
                translated_ctx = await translator._translate_until_translation(translated_image, config)
                translated_ctx.image_name = translated_path
                
                if not translated_ctx.text_regions:
                    logger.warning("  [Skip] No text regions detected in the translated image, outputting the original image unchanged")
                    # Set result to the original image
                    raw_ctx.result = image
                    raw_ctx.text_regions = []
                    skip_to_save = True
            
            if not skip_to_save:
                # Filter out low-confidence regions
                trans_regions_filtered = [r for r in translated_ctx.text_regions if getattr(r, 'prob', 1.0) >= min_prob]
                logger.info(f"    Translated image regions: {len(translated_ctx.text_regions)} -> after filtering: {len(trans_regions_filtered)}")
                
                # Record the size of the translated image
                trans_size = (translated_ctx.img_rgb.shape[1], translated_ctx.img_rgb.shape[0]) if translated_ctx.img_rgb is not None else (translated_image.width, translated_image.height)
                
                # === Step 4: region matching ===
                logger.info("  [3/4] Matching regions...")
                # ✅ Check the stop flag
                await asyncio.sleep(0)
                translator._check_cancelled()
                
                logger.info(f"    Original image size: {raw_size[0]}x{raw_size[1]}")
                logger.info(f"    Translated image size: {trans_size[0]}x{trans_size[1]}")
                logger.info(f"    Scale factor: x={raw_size[0]/trans_size[0]:.3f}, y={raw_size[1]/trans_size[1]:.3f}")
                
                # Scale the regions of the translated image to the size of the raw image
                scaled_trans_regions = scale_regions_to_target(trans_regions_filtered, trans_size, raw_size)
                
                # Match (the overlap ratio is relative to the smaller box)
                matches = match_regions(raw_regions_filtered, scaled_trans_regions, iou_threshold=0.3)
                logger.info(f"    Matching result: {len(matches)} region pairs (overlap ratio >= 0.3, relative to the smaller box)")
                
                # Create the matched regions (the translated boxes are used directly for rendering)
                matched_regions, matched_raw_indices = create_matched_regions(
                    raw_regions_filtered, scaled_trans_regions, matches
                )
                
                # Regions to inpaint: only the raw boxes that correspond to translated boxes (without duplicates)
                # This avoids inpainting regions that exist in the raw image but not in the translated one
                inpaint_raw_indices = set()
                for raw_idx, trans_idx, overlap in matches:
                    inpaint_raw_indices.add(raw_idx)
                inpaint_regions = [raw_regions_filtered[i] for i in sorted(inpaint_raw_indices)]
                
                # Find the raw regions without a match (these should not be inpainted)
                all_raw_indices = set(range(len(raw_regions_filtered)))
                unmatched_raw_indices = all_raw_indices - inpaint_raw_indices
                if unmatched_raw_indices:
                    logger.info(f"    [Unmatched] {len(unmatched_raw_indices)} unmatched original image regions will not be inpainted: {sorted(unmatched_raw_indices)}")
                
                logger.info(f"    Final regions: {len(matched_regions)} for rendering, {len(inpaint_regions)} for inpainting")

                # === DEBUG: build the matching debug image ===
                if translator.verbose:
                    try:
                        # Copy the raw image as the canvas
                        debug_img = raw_ctx.img_rgb.copy()
                        if len(debug_img.shape) == 2: # Greyscale to RGB
                            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
                        elif debug_img.shape[2] == 4: # RGBA to RGB
                            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGBA2BGR)
                        else:
                            debug_img = debug_img.copy() # BGR/RGB
                        
                        logger.info(f"    [DEBUG] Original image boxes: {len(raw_regions_filtered)}, translated image boxes: {len(scaled_trans_regions)}, matching pairs: {len(matches)}")
                        
                        # 1. Draw the raw boxes (red) - each sub-box separately
                        for i, region in enumerate(raw_regions_filtered):
                            # lines holds several sub-boxes of 4 points each, which have to be drawn one by one
                            # Reshape lines to (n_boxes, 4, 2)
                            lines_reshaped = region.lines.reshape(-1, 4, 2)
                            for box in lines_reshaped:
                                pts = box.reshape((-1, 1, 2)).astype(np.int32)
                                cv2.polylines(debug_img, [pts], True, (0, 0, 255), 2)
                            # Use the center attribute of the TextBlock (the centre of the whole region)
                            center = region.center.astype(int)
                            cv2.putText(debug_img, f"R{i}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        # 2. Draw the translated boxes (green) - each sub-box separately
                        for i, region in enumerate(scaled_trans_regions):
                            # Reshape lines to (n_boxes, 4, 2)
                            lines_reshaped = region.lines.reshape(-1, 4, 2)
                            for box in lines_reshaped:
                                pts = box.reshape((-1, 1, 2)).astype(np.int32)
                                cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
                            # Use the center attribute of the TextBlock, shifted slightly to avoid overlap
                            center = region.center.astype(int)
                            center[1] += 20  # Y offset
                            cv2.putText(debug_img, f"T{i}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # 3. Draw the match lines and overlap ratios (yellow) - from the region centres
                        for raw_idx, trans_idx, overlap in matches:
                            raw_center = raw_regions_filtered[raw_idx].center.astype(int)
                            trans_center = scaled_trans_regions[trans_idx].center.astype(int)
                            
                            cv2.line(debug_img, tuple(raw_center), tuple(trans_center), (0, 255, 255), 2)
                            
                            mid_point = ((raw_center + trans_center) / 2).astype(int)
                            # Show the overlap ratio (relative to the smaller box)
                            cv2.putText(debug_img, f"{overlap:.2f}", tuple(mid_point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Save the debug image
                        debug_path = translator._result_path('replace_debug_match.jpg')
                        imwrite_unicode(debug_path, debug_img, logger)
                        logger.info(f"    [DEBUG] Saved match debug image to: {debug_path}")
                        
                    except Exception as e:
                        logger.warning(f"    [DEBUG] Failed to generate debug image: {e}")
                        traceback.print_exc()
                
                # === Step 5: inpaint the raw image ===
                logger.info("  [4/4] Inpainting the original image...")
                # ✅ Check the stop flag
                await asyncio.sleep(0)
                translator._check_cancelled()
                
                # Check whether there are regions to inpaint
                if not inpaint_regions:
                    logger.warning("  [Skip] No regions require inpainting, saving the original image")
                    # Set result to the original image instead of marking a failure
                    raw_ctx.result = image
                    raw_ctx.text_regions = []
                    skip_to_save = True

            if not skip_to_save:
                # Temporarily replace text_regions with the regions to inpaint
                original_regions = raw_ctx.text_regions
                raw_ctx.text_regions = inpaint_regions
                
                # Check whether the inpainting model is none
                inpainter_model = config.inpainter.inpainter if hasattr(config, 'inpainter') and hasattr(config.inpainter, 'inpainter') else None
                logger.info(f"    [Debug] Inpainting model configuration: {inpainter_model} (type: {type(inpainter_model)})")
                
                # Whether it is none (only when explicitly set to 'none'; None does not count)
                is_none_inpainter = (inpainter_model == 'none' or 
                                    (hasattr(inpainter_model, 'value') and inpainter_model.value == 'none') or
                                    (inpainter_model is not None and str(inpainter_model) == 'none'))
                
                if is_none_inpainter:
                    # The inpainting model is none: use the detection module dedicated to replace-translation
                    # Get the raw mask again and refine it with REFINEMASK_INPAINT (the same as win.py)
                    logger.info("    [Inpainting model=none] Using the detection module for translation replacement...")
                    
                    try:
                        from ..config import Detector
                        from ..detection import get_detector
                        from .ctd_replace import detect_for_replace_translation
                        
                        # Get the CTD detector instance
                        detector = get_detector(Detector.ctd)
                        if not detector.is_loaded():
                            await detector.load(translator.device)
                        
                        # Detect again with the new module, to get the raw mask and the mask refined by REFINEMASK_INPAINT
                        _, mask_raw, mask_refined = await detect_for_replace_translation(
                            detector,
                            raw_ctx.img_rgb,
                            detect_size=config.detector.detection_size if hasattr(config.detector, 'detection_size') else 1536,
                            verbose=translator.verbose
                        )
                        
                        raw_ctx.mask_raw = mask_raw  # Update to the real raw mask
                        raw_ctx.mask = mask_refined   # Use the mask refined by REFINEMASK_INPAINT
                        logger.info(f"    Detection completed, original mask pixels: {np.count_nonzero(mask_raw)}, refined mask pixels: {np.count_nonzero(mask_refined) if mask_refined is not None else 0}")
                        
                    except Exception as e:
                        logger.warning(f"    [Warning] Detection for translation replacement failed: {e}, falling back to simple dilation")
                        # Fallback: simple dilation
                        if raw_ctx.mask_raw is not None:
                            kernel = np.ones((5, 5), np.uint8)
                            raw_ctx.mask = cv2.dilate(raw_ctx.mask_raw, kernel, iterations=1)
                            raw_ctx.mask[raw_ctx.mask > 0] = 255
                        else:
                            raw_ctx.mask = None
                    
                    raw_ctx.mask_is_refined = True  # Mark as refined
                else:
                    # Normal flow: build the refined mask
                    logger.info("    Generating mask for inpainting...")
                    raw_ctx.mask = await translator._run_mask_refinement(config, raw_ctx)
                    
                    # Keep the refined mask in mask_raw (so refinement is skipped when it is loaded later)
                    raw_ctx.mask_raw = raw_ctx.mask
                    
                    # Mark the mask as refined; saving the JSON then sets mask_is_refined=True
                    raw_ctx.mask_is_refined = True
                
                # Extra dilation after the mask refinement (from the settings)
                if raw_ctx.mask is not None:
                    kernel_size = config.kernel_size if hasattr(config, 'kernel_size') else 5
                    mask_dilation_offset = config.mask_dilation_offset if hasattr(config, 'mask_dilation_offset') else 0
                    
                    if mask_dilation_offset > 0:
                        # Number of iterations from the pixel count: offset / (kernel_size - 1)
                        # For example: offset=10, kernel_size=5 -> iterations=10/4=2.5 -> 3
                        iterations = max(int(mask_dilation_offset / (kernel_size - 1) + 0.5), 1)
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        raw_ctx.mask = cv2.dilate(raw_ctx.mask, kernel, iterations=iterations)
                        logger.info(f"    Additional mask dilation: kernel_size={kernel_size}, offset={mask_dilation_offset} pixels, iterations={iterations}")
                    else:
                        logger.info(f"    Skipping additional mask dilation (offset={mask_dilation_offset})")
                
                # Whether to inpaint depends on the inpainting model setting
                if is_none_inpainter:
                    # The inpainting model is none: use smart white fill (paint the mask white directly)
                    logger.info("    [Inpainting model=none] Using smart white fill, skipping the inpainting model")
                    # Fill the mask area with white directly
                    raw_ctx.img_inpainted = raw_ctx.img_rgb.copy()
                    if raw_ctx.mask is not None:
                        raw_ctx.img_inpainted[raw_ctx.mask > 0] = 255
                else:
                    # Inpaint with the inpainting model
                    logger.info(f"    Inpainting with model: {inpainter_model}")
                    raw_ctx.img_inpainted = await translator._run_inpainting(config, raw_ctx)
                raw_ctx.text_regions = original_regions  # Restore the region list
            
                # Save the debug image after inpainting (when verbose is on)
                if translator.verbose:
                    try:
                        inpainted_path = translator._result_path('inpainted.png')
                        imwrite_unicode(inpainted_path, cv2.cvtColor(raw_ctx.img_inpainted, cv2.COLOR_RGB2BGR), logger)
                        logger.info(f"    [DEBUG] Saved inpainted debug image to: {inpainted_path}")
                    except Exception as e:
                        logger.warning(f"    [DEBUG] Failed to save inpainted debug image: {e}")
            
                # === Step 6: render or paste ===
                # Check whether direct paste mode is on
                if config.render.enable_template_alignment:
                    logger.info("  [5/5] Direct paste mode - using the darken_blend2 compositing algorithm")

                    # Image size
                    h, w = raw_ctx.img_inpainted.shape[:2]

                    # Check whether the translated image has a raw mask
                    if not hasattr(translated_ctx, 'mask_raw') or translated_ctx.mask_raw is None:
                        logger.warning("  [Warning] No original mask in the translated image, using the original image mask")
                        translated_mask = raw_ctx.mask
                    else:
                        logger.info("    Using the translated image mask...")
                        # Make sure the mask is not empty
                        if translated_ctx.mask_raw is not None and translated_ctx.mask_raw.size > 0:
                            translated_mask = translated_ctx.mask_raw.copy()
                        else:
                            logger.warning("  [Warning] Translated image mask is empty, using the original image mask")
                            translated_mask = raw_ctx.mask

                    # Overwrite directly (inside the mask area the translated image covers the inpainted image)
                    result_img = raw_ctx.img_inpainted.copy()
                
                    # Make sure the translated image and the inpainted image have the same size
                    h, w = result_img.shape[:2]
                    trans_img = translated_ctx.img_rgb
                    if trans_img.shape[:2] != (h, w):
                        trans_img = cv2.resize(trans_img, (w, h), interpolation=cv2.INTER_LINEAR)
                
                    # Scale the mask to the target size (when it differs)
                    if translated_mask.shape[:2] != (h, w):
                        translated_mask = cv2.resize(translated_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                    # Make sure the mask has a single channel
                    if len(translated_mask.shape) == 3:
                        translated_mask = cv2.cvtColor(translated_mask, cv2.COLOR_BGR2GRAY)
                
                    # === Process the mask with the configured dilation settings ===
                    # Binarise
                    _, thres = cv2.threshold(translated_mask, 127, 255, cv2.THRESH_BINARY)
                
                    # Dilate (from the settings)
                    dilation_pixels = config.render.paste_mask_dilation_pixels
                    if dilation_pixels > 0:
                        # Number of iterations from the settings: pixels // 3
                        iterations = max(dilation_pixels // 3, 1)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        translated_mask = cv2.dilate(thres, kernel, iterations=iterations)
                        logger.info(f"    Mask processing: binarization + dilation (3x3 elliptical kernel, {iterations} iterations, configured size={dilation_pixels} pixels)")
                    else:
                        translated_mask = thres
                        logger.info("    Mask processing: binarization only (dilation disabled)")
                
                    # === Composite with the darken_blend2 logic ===
                    # 1. Extract the text from the translated image (with the mask)
                    text = cv2.bitwise_and(trans_img, trans_img, mask=translated_mask)
                
                    # 2. Clear the matching area in the inpainted image (to make room for the text)
                    result_img = cv2.bitwise_and(result_img, result_img, mask=cv2.bitwise_not(translated_mask))
                
                    # 3. Merge: lay the extracted text over the inpainted image
                    result_img = cv2.add(result_img, text)
                
                    logger.info("    Using darken_blend2 compositing: extract text -> clear regions -> overlay and merge")
                
                    # Save the debug images (when verbose is on)
                    if translator.verbose:
                        try:
                            # Save the extracted text
                            debug_text_path = translator._result_path('debug_extracted_text.png')
                            imwrite_unicode(debug_text_path, cv2.cvtColor(text, cv2.COLOR_RGB2BGR), logger)
                            logger.info(f"    [DEBUG] Saved extracted text: {debug_text_path}")
                        except Exception as e:
                            logger.warning(f"    [DEBUG] Failed to save debug image: {e}")
                
                    # Convert the result to a PIL Image with dump_image
                    raw_ctx.result = dump_image(
                        raw_ctx.input,
                        result_img,
                        getattr(raw_ctx, 'img_alpha', None),
                        mask=translated_mask,
                    )
                
                else:
                    # Original logic: OCR + render again
                    logger.info("  [5/5] Rendering mode - rendering text again using OCR results")
                
                    # Update text_regions of the context to the matched regions
                    raw_ctx.text_regions = matched_regions
                
                    # Render
                    img_rendered = await translator._run_text_rendering(config, raw_ctx)
                
                    # Convert the rendered numpy array to a PIL Image with dump_image
                    raw_ctx.result = dump_image(
                        raw_ctx.input,
                        img_rendered,
                        getattr(raw_ctx, 'img_alpha', None),
                        mask=getattr(raw_ctx, 'mask', None),
                        render_alpha=getattr(raw_ctx, 'img_render_alpha', None),
                    )
            
            # === Step 6: save the result ===
            if save_info:
                try:
                    # Work out the output path with translator._calculate_output_path
                    final_output_path = translator._calculate_output_path(image_name, save_info)
                    final_output_dir = os.path.dirname(final_output_path)
                    
                    if hasattr(raw_ctx, 'result') and raw_ctx.result is not None:
                        os.makedirs(final_output_dir, exist_ok=True)
                        
                        save_pil_image(
                            raw_ctx.result,
                            final_output_path,
                            source_image=raw_ctx.input,
                            quality=translator.save_quality,
                        )
                        logger.info(f"  -> Saved: {os.path.basename(final_output_path)}")
                        
                        # Mark as successful
                        raw_ctx.success = True
                        
                        # ✅ Clear result after saving to free memory
                        raw_ctx.result = None
                        
                        # The inpainted image and the JSON are only saved outside direct paste mode
                        if not config.render.enable_template_alignment:
                            # Save the inpainted image to the new folder structure
                            # The same as in the normal translation flow
                            if translator.save_text and hasattr(raw_ctx, 'img_inpainted') and raw_ctx.img_inpainted is not None:
                                translator._save_inpainted_image(image_name, raw_ctx.img_inpainted)
                            
                            # Save the translation data as JSON
                            if translator.save_text:
                                translator._save_text_to_file(image_name, raw_ctx, config)
                        else:
                            logger.info("  -> [Direct paste mode] Skipping JSON and inpainted image saving")
                        
                        # Export an editable PSD (when enabled)
                        from .photoshop_export import psd_export_requested
                        if psd_export_requested(config):
                            # No PSD is exported in direct paste mode either (there is no text region data)
                            if not config.render.enable_template_alignment:
                                try:
                                    from .photoshop_export import (
                                        get_psd_output_path,
                                        photoshop_export,
                                        resolve_photoshop_font,
                                    )
                                    psd_path = get_psd_output_path(image_name)
                                    cli_cfg = getattr(config, 'cli', None)
                                    default_font = resolve_photoshop_font(config)
                                    line_spacing = getattr(config.render, 'line_spacing', None) if hasattr(config, 'render') else None
                                    script_only = getattr(cli_cfg, 'psd_script_only', False)
                                    photoshop_export(psd_path, raw_ctx, default_font, image_name, translator.verbose, translator._result_path, line_spacing, script_only)
                                    logger.info(f"  -> ✅ [PSD] Exported editable PSD: {os.path.basename(psd_path)}")
                                except Exception as psd_err:
                                    logger.error(f"  PSD export failed: {psd_err}")
                            else:
                                logger.info("  -> [Direct paste mode] Skipping PSD export")
                        
                except Exception as save_err:
                    logger.error(f"  Save failed: {save_err}")
                    raw_ctx.success = False
            else:
                # Without save_info it is still marked as successful (it may be preview mode)
                raw_ctx.success = True
            
            results.append(raw_ctx)
            if not getattr(raw_ctx, 'success', False):
                failed_count += 1
            await translator._report_progress(f"batch:{global_idx}:{global_idx}:{display_total}:{failed_count}")
            
            # ✅ Free memory right after each image
            translator._cleanup_context_memory(raw_ctx, keep_result=True)
            
            # Clear the context of the translated image too, when there is one
            if translated_ctx:
                translator._cleanup_context_memory(translated_ctx, keep_result=False)
            
        except Exception as e:
            logger.error(f"  Processing failed: {e}")
            traceback.print_exc()
            ctx = Context()
            ctx.input = image
            ctx.image_name = image_name
            ctx.text_regions = []
            ctx.success = False
            results.append(ctx)
            failed_count += 1
            await translator._report_progress(f"batch:{global_idx}:{global_idx}:{display_total}:{failed_count}")
        finally:
            if translated_image is not None and hasattr(translated_image, 'close'):
                try:
                    translated_image.close()
                except Exception as ignored_error:
                    note_ignored_error(ignored_error, "manga_translator/utils/replace_translation.py:translate_batch_replace_translation")
                    pass
            if loaded_source_image and hasattr(image, 'close'):
                try:
                    image.close()
                except Exception as ignored_error:
                    note_ignored_error(ignored_error, "manga_translator/utils/replace_translation.py:translate_batch_replace_translation")
                    pass
    
    logger.info(f"Replace translation completed: {len(results)} images processed")
    return results


class ReplaceTranslationResult:
    """Result of a replace-translation"""
    
    def __init__(self, 
                 success: bool = False,
                 message: str = "",
                 matched_regions: List[TextBlock] = None,
                 raw_regions: List[TextBlock] = None,
                 translated_regions: List[TextBlock] = None,
                 source_path: str = "",
                 target_path: str = "",
                 source_size: Tuple[int, int] = None,
                 target_size: Tuple[int, int] = None):
        self.success = success
        self.message = message
        self.matched_regions = matched_regions or []  # Matched regions (for rendering)
        self.raw_regions = raw_regions or []          # Filtered regions of the raw image (for inpainting)
        self.translated_regions = translated_regions or []  # Regions of the translated image
        self.source_path = source_path  # Path of the translated image
        self.target_path = target_path  # Path of the raw image
        self.source_size = source_size  # Size of the translated image
        self.target_size = target_size  # Size of the raw image
    
    def __repr__(self):
        return f"ReplaceTranslationResult(success={self.success}, message='{self.message}', " \
               f"matched={len(self.matched_regions)}, raw={len(self.raw_regions)})"


def find_translated_image(raw_image_path: str) -> Optional[str]:
    """
    Find the translated image that belongs to a raw image

    Looks for an image with the same name in the manga_translator_work/translated_images/ folder

    Args:
        raw_image_path: path of the raw image

    Returns:
        The path of the translated image, or None when it does not exist
    """
    work_dir = get_work_dir(raw_image_path)
    translated_dir = os.path.join(work_dir, TRANSLATED_IMAGES_SUBDIR)
    
    if not os.path.isdir(translated_dir):
        logger.warning(f"Translated image directory does not exist: {translated_dir}")
        return None
    
    # Base file name of the raw image
    raw_basename = os.path.splitext(os.path.basename(raw_image_path))[0]
    raw_ext = os.path.splitext(raw_image_path)[1].lower()
    
    # Try the same extension first
    same_ext_path = os.path.join(translated_dir, f"{raw_basename}{raw_ext}")
    if os.path.exists(same_ext_path):
        return same_ext_path
    
    # Try the other supported image extensions
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        translated_path = os.path.join(translated_dir, f"{raw_basename}{ext}")
        if os.path.exists(translated_path):
            return translated_path
    
    # List every file in the folder (to help the user find the problem)
    try:
        files_in_dir = os.listdir(translated_dir)
        logger.warning(f"No translated image matching '{raw_basename}.*' found; files in directory: {files_in_dir[:5]}{'...' if len(files_in_dir) > 5 else ''}")
    except Exception as e:
        logger.error(f"Failed to list directory contents: {e}")
    
    return None


def filter_masks(mask_img: np.ndarray, textlines: List[Tuple[int, int, int, int]], keep_threshold: float = 1e-2) -> Tuple[List[np.ndarray], List[int]]:
    """
    Filter the mask, keeping only the connected components related to text lines

    Args:
        mask_img: the mask image (binary)
        textlines: list of text lines [(x, y, w, h), ...]
        keep_threshold: threshold for keeping (overlap ratio)

    Returns:
        (list of the connected components kept, list assigning components to text lines)
    """
    mask_img = mask_img.copy()
    
    # Draw the outlines of the text lines on the mask (1-pixel black lines), to split connected areas
    for (x, y, w, h) in textlines:
        cv2.rectangle(mask_img, (x, y), (x + w, y + h), (0), 1)
    
    if len(textlines) == 0:
        return [], []
    
    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_img)
    
    cc2textline_assignment = []
    result = []
    M = len(textlines)
    ratio_mat = np.zeros(shape=(num_labels, M), dtype=np.float32)
    dist_mat = np.zeros(shape=(num_labels, M), dtype=np.float32)
    
    for i in range(1, num_labels):  # Skip the background (0)
        # Filter out areas that are too small
        if stats[i, cv2.CC_STAT_AREA] <= 9:
            continue
        
        # Extract the current connected component
        cc = np.zeros_like(mask_img)
        cc[labels == i] = 255
        x1, y1, w1, h1 = cv2.boundingRect(cc)
        area1 = w1 * h1
        
        # Overlap ratio and distance to each text line
        for j in range(M):
            x2, y2, w2, h2 = textlines[j]
            area2 = w2 * h2
            
            # Overlap area
            overlapping_area = area_overlap(x1, y1, w1, h1, x2, y2, w2, h2)
            ratio_mat[i, j] = overlapping_area / min(area1, area2) if min(area1, area2) > 0 else 0
            
            # Distance
            dist_mat[i, j] = rect_distance(x1, y1, x1 + w1, y1 + h1, x2, y2, x2 + w2, y2 + h2)
        
        # Find the text line with the largest overlap ratio
        j = np.argmax(ratio_mat[i])
        
        if ratio_mat[i, j] > keep_threshold:
            # The overlap ratio is high enough: keep it
            cc2textline_assignment.append(j)
            result.append(np.copy(cc))
        else:
            # The overlap ratio is too low: check the distance
            j = np.argmin(dist_mat[i])
            x2, y2, w2, h2 = textlines[j]
            area2 = w2 * h2
            unit = min([h1, w1, h2, w2])
            
            # A small mask fragment very close to a text line is kept as well
            if dist_mat[i, j] < 0.5 * unit and area1 < area2:
                cc2textline_assignment.append(j)
                result.append(np.copy(cc))
            # else: drop it
    
    return result, cc2textline_assignment


def area_overlap(x1, y1, w1, h1, x2, y2, w2, h2) -> float:
    """Overlap area of two rectangles"""
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return x_overlap * y_overlap


def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b) -> float:
    """Distance between two rectangles"""
    def dist(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0


def get_bounding_rect(region: TextBlock) -> Tuple[float, float, float, float]:
    """
    Get the minimum bounding rectangle (x, y, w, h) of a TextBlock
    """
    if region.lines is None or len(region.lines) == 0:
        return (0, 0, 0, 0)
    
    all_points = region.lines.reshape(-1, 2)
    x_min = np.min(all_points[:, 0])
    y_min = np.min(all_points[:, 1])
    x_max = np.max(all_points[:, 0])
    y_max = np.max(all_points[:, 1])
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def calculate_iou(rect1: Tuple[float, float, float, float], 
                  rect2: Tuple[float, float, float, float]) -> float:
    """
    Overlap ratio of two rectangles (relative to the smaller box)

    Args:
        rect1, rect2: rectangles in (x, y, w, h) form

    Returns:
        The overlap ratio (0-1), computed as intersection area / min(area1, area2),
        which shows better whether a small box is contained in a large one
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # Intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Areas of the two rectangles
    area1 = w1 * h1
    area2 = w2 * h2
    
    # Overlap ratio relative to the smaller box
    min_area = min(area1, area2)
    
    if min_area <= 0:
        return 0.0
    
    return inter_area / min_area


def scale_regions_to_target(regions: List[TextBlock], 
                            source_size: Tuple[int, int], 
                            target_size: Tuple[int, int]) -> List[TextBlock]:
    """
    Scale region coordinates from the size of the source image to the size of the target image

    Args:
        regions: the original list of regions
        source_size: size of the source image (width, height)
        target_size: size of the target image (width, height)

    Returns:
        The list of scaled regions (new objects)
    """
    if source_size == target_size:
        return regions
    
    scale_x = target_size[0] / source_size[0] if source_size[0] > 0 else 1.0
    scale_y = target_size[1] / source_size[1] if source_size[1] > 0 else 1.0
    
    scaled_regions = []
    for region in regions:
        import copy
        new_region = copy.deepcopy(region)
        
        # Scale the coordinates of lines
        if new_region.lines is not None:
            new_region.lines[:, :, 0] *= scale_x
            new_region.lines[:, :, 1] *= scale_y
        
        # Clear the cached attributes to force a recalculation (lines has changed)
        # cached_property caches its result in __dict__
        for attr in ['xyxy', 'xywh', 'center', 'unrotated_polygons', 'unrotated_min_rect', 'min_rect']:
            if attr in new_region.__dict__:
                delattr(new_region, attr)
        
        # Scale the font size
        if new_region.font_size > 0:
            avg_scale = (scale_x + scale_y) / 2
            new_region.font_size = int(new_region.font_size * avg_scale)
        
        scaled_regions.append(new_region)
    
    return scaled_regions


def match_regions(raw_regions: List[TextBlock], 
                  translated_regions: List[TextBlock],
                  iou_threshold: float = 0.3) -> List[Tuple[int, int, float]]:
    """
    Match the regions of the raw image and the translated image - simplified version

    Logic:
    1. Compute the overlap ratio of every raw box with every translated box
    2. Whenever the overlap ratio >= the threshold, the translated box is kept
    3. One translated box may be matched by several raw boxes (many to one)

    Args:
        raw_regions: list of regions of the raw image
        translated_regions: list of regions of the translated image (already scaled to the size of the raw image)
        iou_threshold: overlap ratio threshold (relative to the smaller box)

    Returns:
        The list of matches [(raw_idx, trans_idx, overlap_ratio), ...]
    """
    # Bounding rectangles of all regions
    raw_rects = [get_bounding_rect(r) for r in raw_regions]
    trans_rects = [get_bounding_rect(r) for r in translated_regions]
    
    # Work out every possible overlap
    matches = []
    for trans_idx, trans_rect in enumerate(trans_rects):
        matched_raws = []
        for raw_idx, raw_rect in enumerate(raw_rects):
            overlap = calculate_iou(raw_rect, trans_rect)
            if overlap >= iou_threshold:
                matched_raws.append((raw_idx, overlap))
        
        # When this translated box has matching raw boxes, keep all the matches
        if matched_raws:
            # Create a match record for every matching raw box,
            # so that every matching raw box gets inpainted
            for raw_idx, overlap in matched_raws:
                matches.append((raw_idx, trans_idx, overlap))
            
            # Log it when several raw boxes match this translated box
            if len(matched_raws) > 1:
                raw_indices = [r for r, _ in matched_raws]
                trans_text = translated_regions[trans_idx].text if hasattr(translated_regions[trans_idx], 'text') else ''
                logger.info(f"    [Many-to-one] T{trans_idx} (text=\"{trans_text[:20] if trans_text else ''}...\") matched {len(matched_raws)} original image boxes: {raw_indices}; all will be inpainted")
    
    # Count the regions without a match
    matched_trans = set(t for _, t, _ in matches)
    unmatched_trans = set(range(len(translated_regions))) - matched_trans
    
    if unmatched_trans:
        logger.warning(f"    [Warning] No matches found for {len(unmatched_trans)} translation regions:")
        for trans_idx in sorted(unmatched_trans):
            trans_rect = trans_rects[trans_idx]
            trans_text = translated_regions[trans_idx].text if hasattr(translated_regions[trans_idx], 'text') else ''
            logger.warning(f"      T{trans_idx}: position=({trans_rect[0]:.0f},{trans_rect[1]:.0f}), size={trans_rect[2]:.0f}x{trans_rect[3]:.0f}, text=\"{trans_text[:20] if trans_text else ''}...\"")
    
    logger.info(f"    Matching result: retained {len(matches)} translation regions (overlap ratio >= {iou_threshold}, relative to the smaller box)")
    
    return matches


def merge_rects(rect1: Tuple[float, float, float, float],
                rect2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Merge two rectangles and return the minimum bounding rectangle that contains both

    Args:
        rect1, rect2: rectangles in (x, y, w, h) form

    Returns:
        The merged rectangle (x, y, w, h)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    min_x = min(x1, x2)
    min_y = min(y1, y2)
    max_x = max(x1 + w1, x2 + w2)
    max_y = max(y1 + h1, y2 + h2)
    
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def create_matched_regions(raw_regions: List[TextBlock],
                           translated_regions: List[TextBlock],
                           matches: List[Tuple[int, int, float]]) -> Tuple[List[TextBlock], set]:
    """
    Create the list of matched regions - simplified version

    All data of the translated box is used directly (box, text, style).
    Note: one translated box may match several raw boxes, but it is added to the render list only once

    Args:
        raw_regions: regions of the raw image (to record which ones were matched)
        translated_regions: regions of the translated image
        matches: the matches [(raw_idx, trans_idx, overlap), ...]

    Returns:
        (list of matched regions, set of indexes of the matched raw regions)
    """
    import copy
    
    matched_regions = []
    matched_raw_indices = set()
    added_trans_indices = set()  # Record the translated boxes already added, to avoid duplicates
    
    for raw_idx, trans_idx, overlap in matches:
        # Record the index of the raw box (for inpainting)
        matched_raw_indices.add(raw_idx)
        
        # Add the translated box only once (to avoid rendering it twice)
        if trans_idx in added_trans_indices:
            continue
        added_trans_indices.add(trans_idx)
        
        # Use the data of the translated box directly
        region = copy.deepcopy(translated_regions[trans_idx])
        
        # Key fix: rebuild the translation field, with [BR], from the texts array,
        # because the renderer draws the translation field and needs [BR] for line breaks
        if region.texts and len(region.texts) > 0:
            # With a texts array, join with [BR] (kept the same even for a single line)
            region.translation = "[BR]".join(region.texts)
        elif region.text:
            # Without a texts array, use the text field
            region.translation = region.text
        else:
            # With neither, use an empty string
            region.translation = ""
        
        matched_regions.append(region)
    
    return matched_regions, matched_raw_indices


def filter_raw_regions_for_inpainting(raw_regions: List[TextBlock],
                                      matched_indices: set) -> List[TextBlock]:
    """
    Get the raw regions to inpaint (only the ones that were matched)

    Args:
        raw_regions: all raw regions
        matched_indices: set of the indexes that were matched

    Returns:
        The list of regions to inpaint
    """
    return [raw_regions[i] for i in sorted(matched_indices)]


def calculate_template_alignment_offset(raw_img: np.ndarray, 
                                        translated_img: np.ndarray,
                                        template_size: int = 440) -> Tuple[int, int]:
    """
    Compute the alignment offset between the two images by template matching

    A template is taken from the centre of the translated image and matched in the raw image,
    which gives the horizontal and vertical offset to move by

    Args:
        raw_img: the raw image (BGR)
        translated_img: the translated image (BGR)
        template_size: template size (pixels); computed automatically when 0

    Returns:
        (horizontal_offset, vertical_offset)
        - horizontal_offset: horizontal offset, >0 moves left, <0 moves right
        - vertical_offset: vertical offset, >0 moves up, <0 moves down
    """
    try:
        # Make sure the image is BGR
        if len(translated_img.shape) == 2:
            translated_img = cv2.cvtColor(translated_img, cv2.COLOR_GRAY2BGR)
        if len(raw_img.shape) == 2:
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        
        zh, zw = translated_img.shape[:2]
        
        # Choose the template size automatically (between 1/3 and 1/2 of the short side of the image)
        if template_size <= 0:
            min_side = min(zh, zw)
            template_size = min(max(min_side // 3, 200), min_side // 2)
            logger.info(f"    [Alignment] Automatically calculated template size: {template_size} pixels (image size: {zw}x{zh})")
        
        # Take the template from the centre of the translated image
        cenx = zw // 2 - template_size // 2
        ceny = zh // 2 - template_size // 2
        
        # Make sure the template stays inside the bounds
        if cenx < 0 or ceny < 0 or cenx + template_size > zw or ceny + template_size > zh:
            # Adjust the template size automatically
            max_template_size = min(zw, zh) - 20  # Leave a 10-pixel margin
            if max_template_size < 100:
                logger.warning(f"Image size {zw}x{zh} is too small for template matching, using the default offset (0, 0)")
                return (0, 0)
            template_size = max_template_size
            cenx = zw // 2 - template_size // 2
            ceny = zh // 2 - template_size // 2
            logger.info(f"    [Alignment] Automatically adjusted template size to {template_size} pixels")
        
        muban = translated_img[ceny:ceny + template_size, cenx:cenx + template_size]
        
        # Match in the raw image
        res = cv2.matchTemplate(raw_img, muban, cv2.TM_CCOEFF)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        
        # Get the match position
        xdist, ydist = max_loc
        
        # Work out the offset
        horizontal_offset = cenx - xdist
        vertical_offset = ceny - ydist
        
        # Fine-tune the offset (the same logic as the original version)
        if horizontal_offset < 0:
            horizontal_offset -= 3
        elif horizontal_offset > 0:
            horizontal_offset += 3
        
        if vertical_offset < 0:
            vertical_offset -= 3
        elif vertical_offset > 0:
            vertical_offset += 3
        
        logger.info(f"    [Alignment] Template matching completed: horizontal offset={horizontal_offset}, vertical offset={vertical_offset}")
        
        return (horizontal_offset, vertical_offset)
        
    except Exception as e:
        logger.error(f"    [Alignment] Template matching failed: {e}")
        traceback.print_exc()
        return (0, 0)


# Exported functions and classes
__all__ = [
    'ReplaceTranslationResult',
    'find_translated_image',
    'get_bounding_rect',
    'calculate_iou',
    'scale_regions_to_target',
    'match_regions',
    'create_matched_regions',
    'filter_raw_regions_for_inpainting',
    'get_text_to_img_solid_ink',
]


# ============================================================================
# Below are the mask refinement functions of the win.py version (for inpainter=none mode)
# ============================================================================

def _refine_mask_winpy(rgbimg, rawmask):
    """
    The refine_mask function of the win.py version.
    Refines the mask edges with DenseCRF
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
        
        if len(rawmask.shape) == 2:
            rawmask = rawmask[:, :, None]
        mask_softmax = np.concatenate([cv2.bitwise_not(rawmask)[:, :, None], rawmask], axis=2)
        mask_softmax = mask_softmax.astype(np.float32) / 255.0
        n_classes = 2
        feat_first = mask_softmax.transpose((2, 0, 1)).reshape((n_classes, -1))
        unary = unary_from_softmax(feat_first)
        unary = np.ascontiguousarray(unary)

        d = dcrf.DenseCRF2D(rgbimg.shape[1], rgbimg.shape[0], n_classes)

        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=1, compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NO_NORMALIZATION)

        d.addPairwiseBilateral(sxy=23, srgb=7, rgbim=rgbimg,
                               compat=20,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NO_NORMALIZATION)
        Q = d.inference(5)
        res = np.argmax(Q, axis=0).reshape((rgbimg.shape[0], rgbimg.shape[1]))
        crf_mask = np.array(res * 255, dtype=np.uint8)
        return crf_mask
    except ImportError:
        logger.warning("pydensecrf is not installed, skipping CRF refinement")
        return rawmask


def _complete_mask_winpy(img_np: np.ndarray, ccs: List[np.ndarray], textlines: List[Tuple[int, int, int, int]], cc2textline_assignment):
    """
    The complete_mask function of the win.py version.
    Completes the mask refinement
    """
    from tqdm import tqdm
    
    if len(ccs) == 0:
        return None
    textline_ccs = [np.zeros_like(ccs[0]) for _ in range(len(textlines))]
    for i, cc in enumerate(ccs):
        txtline = cc2textline_assignment[i]
        textline_ccs[txtline] = cv2.bitwise_or(textline_ccs[txtline], cc)
    final_mask = np.zeros_like(ccs[0])
    img_np = cv2.bilateralFilter(img_np, 17, 80, 80)
    for i, cc in enumerate(tqdm(textline_ccs, '[mask]', disable=True)):
        x1, y1, w1, h1 = cv2.boundingRect(cc)
        text_size = min(w1, h1)
        extend_size = int(text_size * 0.1)
        x1 = max(x1 - extend_size, 0)
        y1 = max(y1 - extend_size, 0)
        w1 += extend_size * 2
        h1 += extend_size * 2
        w1 = min(w1, img_np.shape[1] - x1 - 1)
        h1 = min(h1, img_np.shape[0] - y1 - 1)
        dilate_size = max((int(text_size * 0.3) // 2) * 2 + 1, 3)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        cc_region = np.ascontiguousarray(cc[y1: y1 + h1, x1: x1 + w1])
        if cc_region.size == 0:
            continue
        img_region = np.ascontiguousarray(img_np[y1: y1 + h1, x1: x1 + w1])
        cc_region = _refine_mask_winpy(img_region, cc_region)
        cc[y1: y1 + h1, x1: x1 + w1] = cc_region
        cc = cv2.dilate(cc, kern)
        final_mask = cv2.bitwise_or(final_mask, cc)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(final_mask, kern)
