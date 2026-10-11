
import asyncio
import json
import logging
import os
import time
import traceback
import unicodedata
from typing import Any, List, Optional

import cv2
import langcodes
import matplotlib
import numpy as np
import py3langid as langid
import regex as re
import torch
from PIL import Image, ImageFile

from .config import Colorizer, Config, Inpainter, Renderer, Translator
from .server_paths import normalize_server_resource_path
from .utils import (
    BASE_PATH,
    Context,
    ModelWrapper,
    TextBlock,
    build_bubble_mask_from_mangalens_result,
    build_det_rearrange_plan,
    detect_bubbles_with_mangalens,
    dump_image,
    erode_bubble_mask,
    imwrite_unicode,
    is_valuable_text,
    load_image,
    open_pil_image,
    save_pil_image,
    sort_regions,
    visualize_textblocks,
)
from .utils.batch_skip import BatchInputPlan, input_path, plan_batch_inputs, slice_batch_indices
from .utils.onnx_runtime import set_onnx_gpu_disabled
from .utils.text_filter import match_filter
from manga_translator.utils.swallowed import note_ignored_error

matplotlib.use('Agg')  # Use a non-GUI backend

from .colorization import dispatch as dispatch_colorization
from .colorization import prepare as prepare_colorization
from .colorization import unload as unload_colorization
from .detection import dispatch as dispatch_detection
from .detection import prepare as prepare_detection
from .detection import unload as unload_detection
from .detection.imported_yolo import (
    build_mask_from_textlines,
    load_imported_yolo_textlines,
)
from .inpainting import dispatch as dispatch_inpainting
from .inpainting import prepare as prepare_inpainting
from .inpainting import unload as unload_inpainting
from .inpainting.ballon_fill import (
    MODEL_BUBBLE_SHRINK_RATIO,
    inpaint_regions_per_block,
    solid_fill_pure_bubbles,
)
from .mask_refinement import dispatch as dispatch_mask_refinement
from .ocr import dispatch as dispatch_ocr
from .ocr import prepare as prepare_ocr
from .ocr import unload as unload_ocr
from .rendering import dispatch as dispatch_rendering
from .rendering.rich_text import has_content, plain_text_of
from .textline_merge import dispatch as dispatch_textline_merge
from .translators import (
    dispatch as dispatch_translation,
)
from .translators import (
    prepare as prepare_translation,
)
from .translators import (
    unload as unload_translation,
)
from .translators.common import (
    ISO_639_1_TO_KEEP_LANGUAGES,
    ISO_639_1_TO_VALID_LANGUAGES,
    KEEP_LANGUAGES,
)
from .upscaling import dispatch as dispatch_upscaling
from .upscaling import prepare as prepare_upscaling
from .upscaling import unload as unload_upscaling
from .utils.ai_image_preprocess import normalize_ai_image
from .utils.path_manager import (
    find_inpainted_path,
    find_json_path,
    get_inpainted_path,
    get_json_path,
    get_original_txt_path,
    get_work_image_path,
)
from .utils.translation_text import (
    is_thai_language,
    normalize_thai_punctuation,
    remove_trailing_period_if_needed,
)

# Will be overwritten by __main__.py if module is being run directly (with python -m)
logger = logging.getLogger('manga_translator')


def _translation_plain_text(value) -> str:
    # Thin wrapper: the one rich-text to plain-text implementation is rendering.rich_text.plain_text_of
    return plain_text_of(value)


def _has_translation_text(value) -> bool:
    return has_content(value)

ARCHIVE_EXTRACT_IMAGE_DIRNAME = 'original_images'
ARCHIVE_EXTRACT_META_FILENAME = '.extract_meta.json'
_KEEP_LANG_NONE_VALUES = {'', 'NONE', 'OFF', 'DISABLED'}
_DETECTED_KEEP_LANG_CODES = set(KEEP_LANGUAGES.keys())
_KEEP_LANG_CJK_SHARED = 'CJK_SHARED'
_KEEP_LANG_SHARED_CJK_TARGETS = frozenset({'CHS', 'CHT', 'JPN'})
_ENGLISH_KEEP_FILTER_PUNCTUATION = frozenset(
    ".,!?;:'\"-()[]{}<>/&@#%+*=~_|`$^\\"
    "…“”‘’–—"
)


class FileTranslationFailure(Exception):
    """Abort the current file while allowing the overall batch to continue."""

    def __init__(self, stage: str, error: Exception):
        self.stage = stage
        self.original_error = error
        message = str(error).strip() or repr(error)
        super().__init__(message)


def _resolve_archive_output_dir_from_extracted_image(image_path: str, output_folder: str) -> Optional[str]:
    """
    If image_path points to an image extracted from an archive inside the output folder, return the output folder of that archive.
    For example: <output>/A/B/1/original_images/page.png -> <output>/A/B/1
    """
    if not image_path or not output_folder:
        return None

    image_parent = os.path.normpath(os.path.dirname(image_path))
    if os.path.basename(image_parent) != ARCHIVE_EXTRACT_IMAGE_DIRNAME:
        return None

    meta_path = os.path.join(image_parent, ARCHIVE_EXTRACT_META_FILENAME)
    if not os.path.isfile(meta_path):
        return None

    archive_output_dir = os.path.normpath(os.path.dirname(image_parent))
    output_root_abs = os.path.normcase(os.path.abspath(output_folder))
    archive_output_abs = os.path.normcase(os.path.abspath(archive_output_dir))

    try:
        common = os.path.commonpath([output_root_abs, archive_output_abs])
    except ValueError:
        return None

    if common != output_root_abs:
        return None

    return archive_output_dir


def _is_likely_english_text_for_keep_filter(text: str) -> bool:
    normalized = unicodedata.normalize('NFKC', str(text or '').strip())
    if not normalized:
        return False

    has_latin_letter = False
    for char in normalized:
        if char.isspace() or char.isdigit():
            continue
        if char in _ENGLISH_KEEP_FILTER_PUNCTUATION:
            continue

        category = unicodedata.category(char)
        if category.startswith('L') and 'LATIN' in unicodedata.name(char, ''):
            has_latin_letter = True
            continue

        return False

    return has_latin_letter


def _normalize_detected_keep_language(lang_code: Optional[str]) -> str:
    value = str(lang_code or '').strip()
    if not value:
        return 'UNKNOWN'

    upper_value = value.upper()
    if upper_value in _DETECTED_KEEP_LANG_CODES:
        return upper_value

    mapped = ISO_639_1_TO_KEEP_LANGUAGES.get(value.lower())
    if mapped:
        return mapped.upper()

    return 'UNKNOWN'


def _normalize_keep_filter_script_candidate(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', str(text or '').strip())
    if not normalized:
        return ''

    normalized = re.sub(r'^[\p{P}\p{S}\s]+|[\p{P}\p{S}\s]+$', '', normalized)
    return normalized


def _detect_script_based_keep_language(text: str) -> Optional[str]:
    candidate = _normalize_keep_filter_script_candidate(text)
    if not candidate:
        return None

    meaningful_chars = []
    for char in candidate:
        category = unicodedata.category(char)
        if char.isspace() or char.isdigit() or category.startswith(('P', 'S')):
            continue
        meaningful_chars.append(char)

    if not meaningful_chars:
        return None

    meaningful_text = ''.join(meaningful_chars)
    if re.search(r'[\p{Hiragana}\p{Katakana}]', meaningful_text):
        return 'JPN'
    if re.search(r'\p{Hangul}', meaningful_text):
        return 'KOR'
    if re.fullmatch(r'\p{Han}+', meaningful_text):
        return _KEEP_LANG_CJK_SHARED

    return None


def _detect_region_keep_language(text: str) -> str:
    text = str(text or '').strip()
    if not text:
        return 'UNKNOWN'

    if _is_likely_english_text_for_keep_filter(text):
        return 'ENG'

    detected_by_script = _detect_script_based_keep_language(text)
    if detected_by_script:
        return detected_by_script

    try:
        detected_lang, _ = langid.classify(text)
    except Exception as ignored_error:
        note_ignored_error(ignored_error, "manga_translator/manga_translator.py:_detect_region_keep_language")
        return 'UNKNOWN'

    return _normalize_detected_keep_language(detected_lang)


def _keep_language_matches(detected_lang: str, keep_lang: str) -> bool:
    if detected_lang == keep_lang:
        return True
    if keep_lang in {'CHS', 'CHT'} and detected_lang in {'CHS', 'CHT'}:
        return True
    if detected_lang == _KEEP_LANG_CJK_SHARED and keep_lang in _KEEP_LANG_SHARED_CJK_TARGETS:
        return True
    return False


class TranslationInterrupt(Exception):
    """
    Can be raised from within a progress hook to prematurely terminate
    the translation.
    """
    pass

def load_dictionary(file_path):
    dictionary = []
    if file_path:
        path_to_check = file_path if os.path.isabs(file_path) else os.path.join(BASE_PATH, file_path)
        if os.path.exists(path_to_check):
            with open(path_to_check, 'r', encoding='utf-8') as file:
                for line_number, line in enumerate(file, start=1):
                    # Ignore empty lines and lines starting with '#' or '//'
                    if not line.strip() or line.strip().startswith('#') or line.strip().startswith('//'):
                        continue
                    # Remove comment parts
                    line = line.split('#')[0].strip()
                    line = line.split('//')[0].strip()
                    parts = line.split()
                    if len(parts) == 1:
                        # If there is only the left part, the right part defaults to an empty string, meaning delete the left part
                        pattern = re.compile(parts[0])
                        dictionary.append((pattern, '', line_number))
                    elif len(parts) == 2:
                        # If both left and right parts are present, perform the replacement
                        pattern = re.compile(parts[0])
                        dictionary.append((pattern, parts[1], line_number))
                    else:
                        logger.error(f'Invalid dictionary entry at line {line_number}: {line.strip()}')
    return dictionary

def apply_dictionary(text, dictionary):
    for pattern, value, line_number in dictionary:
        text = pattern.sub(value, text)
    return text

def _parse_skip_font_scaling_flag(value, default: bool = True) -> bool:
    """Normalize skip_font_scaling values loaded from JSON."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    if value is None:
        return default
    return bool(value)

class MangaTranslator:
    verbose: bool
    ignore_errors: bool
    _gpu_limited_memory: bool
    device: Optional[str]
    kernel_size: Optional[int]
    models_ttl: int
    _progress_hooks: list[Any]
    result_sub_folder: str
    batch_size: int

    _CONTEXT_INTERMEDIATE_FIELDS = (
        'img_rgb',
        'bubble_mask',
        'img_colorized',
        'upscaled',
        'img_inpainted',
        'img_rendered',
        'img_render_alpha',
        'img_alpha',
        'mask',
        'mask_raw',
        'textlines',
    )

    def __init__(self, params: dict = {}):
        self.pre_dict = params.get('pre_dict', None)
        self.post_dict = params.get('post_dict', None)
        self.font_family = None
        self.kernel_size = None
        self.device = None
        self.text_output_file = params.get('save_text_file', None)
        self._gpu_limited_memory = False
        self.ignore_errors = False
        self.verbose = False
        self.models_ttl = 0
        self.batch_size = 1  # No batching by default
        self.disable_onnx_gpu = False

        self._progress_hooks = []
        self._add_logger_hook()
        
        # Callback that checks for cancellation (for the web server and similar hosts)
        self._cancel_check_callback = None

        params = params or {}
        
        self._batch_contexts = []  # Contexts of the batch being processed
        self._batch_configs = []   # Configurations of the batch being processed
        # In-memory payloads: image_name -> load_text data dict (editor export and similar paths skip the disk round trip)
        self._preloaded_load_text_payloads = {}
        # batch_concurrent: four-stage concurrent mode (off by default, can be switched on in the settings)
        self.batch_concurrent = params.get('batch_concurrent', False)
        
        # Flag that records whether the models are loaded
        self._models_loaded = False
        
        self.parse_init_params(params)
        self.result_sub_folder = ''

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

        self._model_usage_timestamps = {}
        self._detector_cleanup_task = None
        self.context_size = params.get('context_size', 0)
        self.all_page_translations = []
        self._original_page_texts = []  # Original text of each page, used as context in concurrent mode
        self._resume_context_pages = []
        self._resume_context_cursor = 0
        self._resume_context_order = {}
        self._colorizer_history_images = []  # Recently colorized pages, used as history reference for AI colorization

        # Attributes for managing debug images
        self._current_image_context = None  # Context information of the image being processed
        self._saved_image_contexts = {}     # Context information of each image in the batch
        
        # The log file is managed by the UI layer now; it is no longer created here
        self._log_file_path = None
        
        # Filter list switch (on by default)
        self.filter_text_enabled = params.get('filter_text_enabled', True)
        
        # Make sure the filter list file exists
        try:
            from .runtime_files import ensure_runtime_files
            ensure_runtime_files()
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator.__init__")
            pass

    def parse_init_params(self, params: dict):
        self.verbose = params.get('verbose', False)
        self.font_family = params.get('font_family', None)
        self.models_ttl = params.get('models_ttl', 0)
        self.batch_size = params.get('batch_size', 3)  # Batch size (translation batch)
        disable_onnx_gpu = params.get('disable_onnx_gpu', False)
        if isinstance(disable_onnx_gpu, str):
            disable_onnx_gpu = disable_onnx_gpu.strip().lower() in ('1', 'true', 'yes', 'on')
        self.disable_onnx_gpu = bool(disable_onnx_gpu)
        set_onnx_gpu_disabled(self.disable_onnx_gpu)
        if self.disable_onnx_gpu:
            logger.info("ONNX GPU acceleration disabled; ONNX Runtime will use CPUExecutionProvider.")
        
        # batch_concurrent: four-stage concurrent pipeline (optional)
        # When on, detection, OCR, inpainting and translation run concurrently, which is faster
            
        self.ignore_errors = params.get('ignore_errors', False)
        # check mps for apple silicon or cuda for nvidia
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'
        self.device = device if params.get('use_gpu', False) else 'cpu'
        if self.using_gpu and ( not torch.cuda.is_available() and not torch.backends.mps.is_available()):
            # Fall back to the CPU when no GPU is available instead of raising
            logger.warning(
                'CUDA or Metal compatible device could not be found in torch whilst --use-gpu was set. '
                'Automatically falling back to CPU mode.'
            )
            self.device = 'cpu'
        if params.get('model_dir'):
            ModelWrapper._MODEL_DIR = params.get('model_dir')
        #todo: fix why is kernel size loaded in the constructor
        self.kernel_size=int(params.get('kernel_size', 3))
        # Set input files
        self.input_files = params.get('input', [])
        # Set save_text
        self.save_text = params.get('save_text', False)
        # Set load_text
        self.load_text = params.get('load_text', False)
        self.translate_json_only = params.get('translate_json_only', False)
        self.save_mask = not params.get('no_save_mask', False)
        self.template = params.get('template', False)
        self.attempts = params.get('attempts', -1)
        self._attempts_override_provided = 'attempts' in params
        self.save_quality = params.get('save_quality', 100)
        self.skip_no_text = params.get('skip_no_text', False)
        self.generate_and_export = params.get('generate_and_export', False)
        self.export_from_local_json = params.get('export_from_local_json', False)
        self.colorize_only = params.get('colorize_only', False)
        self.upscale_only = params.get('upscale_only', False)
        self.inpaint_only = params.get('inpaint_only', False)
        
        # Replace-translation mode (copies translation data from translated images to raw images)
        self.replace_translation = params.get('replace_translation', False)
        
        
        # batch_concurrent was set and validated during initialisation
        

        
    def _set_image_context(self, config: Config, image=None):
        """Set the context information of the image being processed, used to name the subfolder for debug images"""
        from .utils.generic import get_image_md5

        # Millisecond timestamp, so the name is unique
        timestamp = str(int(time.time() * 1000))
        detection_size = str(getattr(config.detector, 'detection_size', 1024))
        target_lang = getattr(config.translator, 'target_lang', 'unknown')
        translator = getattr(config.translator, 'translator', 'unknown')

        # MD5 hash of the image
        if image is not None:
            file_md5 = get_image_md5(image)
        else:
            file_md5 = "unknown"

        # Subfolder name: {timestamp}-{file_md5}-{detection_size}-{target_lang}-{translator}
        subfolder_name = f"{timestamp}-{file_md5}-{detection_size}-{target_lang}-{translator}"

        self._current_image_context = {
            'subfolder': subfolder_name,
            'file_md5': file_md5,
            'config': config
        }
        
    def _get_image_subfolder(self) -> str:
        """Get the name of the debug subfolder of the current image"""
        if self._current_image_context:
            return self._current_image_context['subfolder']
        return ''
    
    def _save_current_image_context(self, image_md5: str):
        """Save the current image context, to keep it consistent during batch processing"""
        if self._current_image_context:
            self._saved_image_contexts[image_md5] = self._current_image_context.copy()

    def _restore_image_context(self, image_md5: str):
        """Restore a saved image context"""
        if image_md5 in self._saved_image_contexts:
            self._current_image_context = self._saved_image_contexts[image_md5].copy()
            return True
        return False

    @property
    def using_gpu(self):
        return self.device.startswith('cuda') or self.device == 'mps'

    async def translate(self, image: Image.Image, config: Config, image_name: str = None, skip_context_save: bool = False, save_info: dict = None) -> Context:
        """
        Translates a single image by calling translate_batch with batch_size=1.
        
        This is a compatibility wrapper. All translation logic is now unified in translate_batch().

        :param image: Input image.
        :param config: Translation config.
        :param image_name: Image file name for saving results.
        :param save_info: Save configuration (output_folder, format, etc.)
        :return: Translation context.
        """
        # Attach image_name to image object for batch processing
        if image_name and not hasattr(image, 'name'):
            image.name = image_name
        
        # Call unified batch translation with single image
        results = await self.translate_batch(
            images_with_configs=[(image, config)],
            batch_size=1,
            save_info=save_info
        )
        
        # Return the single result
        return results[0] if results else Context()

    def _apply_runtime_cli_overrides(self, config: Config) -> Config:
        if (
            config is not None
            and self._attempts_override_provided
            and hasattr(config, 'cli')
            and hasattr(config.cli, 'attempts')
        ):
            config.cli.attempts = self.attempts
        return config


    def _calculate_output_path(self, image_path: str, save_info: dict) -> str:
        """
        Work out the full path of the output file

        Args:
            image_path: path of the input image
            save_info: dictionary with the output settings:
                - output_folder: the output folder
                - input_folders: the set of input folders
                - format: the output format (optional)
                - save_to_source_dir: whether to write to the manga_translator_work/result subfolder next to the source image

        Returns:
            str: the full path of the output file
        """
        output_folder = save_info.get('output_folder')
        input_folders = save_info.get('input_folders', set())
        output_format = save_info.get('format')
        save_to_source_dir = save_info.get('save_to_source_dir', False)
        
        file_path = image_path
        parent_dir = os.path.normpath(os.path.dirname(file_path))
        
        # Check whether "save next to the source image" is on
        if save_to_source_dir:
            # Write to the manga_translator_work/result subfolder next to the source image
            final_output_dir = os.path.join(parent_dir, 'manga_translator_work', 'result')
        else:
            # Original behaviour: use the configured output folder
            final_output_dir = output_folder

            # Handle the archive extraction folder first: <...>/original_images/<image>
            archive_output_dir = _resolve_archive_output_dir_from_extracted_image(file_path, output_folder)
            if archive_output_dir:
                final_output_dir = archive_output_dir
            else:
                # Work out the relative path so the folder structure is kept
                for folder in input_folders:
                    if parent_dir.startswith(folder):
                        relative_path = os.path.relpath(parent_dir, folder)
                        # Normalize path and avoid adding '.' as a directory component
                        if relative_path == '.':
                            final_output_dir = os.path.join(output_folder, os.path.basename(folder))
                        else:
                            final_output_dir = os.path.join(output_folder, os.path.basename(folder), relative_path)
                        # Normalize to use consistent separators
                        final_output_dir = os.path.normpath(final_output_dir)
                        break
        
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Work out the output file name and format
        base_filename, _ = os.path.splitext(os.path.basename(file_path))
        if output_format and output_format.strip() and output_format.lower() != 'none':
            output_filename = f"{base_filename}.{output_format}"
        else:
            output_filename = os.path.basename(file_path)
        
        final_output_path = os.path.join(final_output_dir, output_filename)
        return final_output_path

    def _save_translated_image(
        self,
        image: Image.Image,
        output_path: str,
        image_path: str,
        overwrite: bool = True,
        mode_label: str = "BATCH",
        source_image: Optional[Image.Image] = None,
    ) -> Optional[bool]:
        """
        Save the translated image to the given path

        Args:
            image: the PIL image object to save
            output_path: path of the output file
            image_path: path of the original image (used to update the translation map)
            overwrite: whether an existing file is overwritten
            mode_label: mode label (for the log)

        Returns:
            True when saved, None when a post-plan race created the output, otherwise False.
        """
        if not overwrite and os.path.exists(output_path):
            logger.info(f"  -> ⚠️ [{mode_label}] Skipping existing file: {os.path.basename(output_path)}")
            return None
        
        try:
            save_pil_image(
                image,
                output_path,
                source_image=source_image,
                quality=self.save_quality,
            )
            logger.info(f"  -> ✅ [{mode_label}] Saved successfully: {os.path.basename(output_path)}")
            
            # Update the translation map
            self._update_translation_map(image_path, output_path)
            return True
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            return False
    
    def _save_and_cleanup_context(self, ctx: Context, save_info: dict, config: Config = None, mode_label: str = "BATCH") -> bool:
        """
        Shared save-and-clean-up method: save the translation result, export the PSD and free memory

        Args:
            ctx: the Context object
            save_info: dictionary with the save information
            config: the Config object (for the PSD export)
            mode_label: mode label (for the log)

        Returns:
            bool: whether saving succeeded
        """
        if not save_info or not ctx.result:
            return False
        
        try:
            overwrite = save_info.get('overwrite', True)
            final_output_path = self._calculate_output_path(ctx.image_name, save_info)
            # Remember this output folder: _save_text_to_file writes it to the JSON so a later editor export goes to the same place
            try:
                ctx.final_output_dir = os.path.dirname(final_output_path)
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._save_and_cleanup_context")
                pass
            success = self._save_translated_image(
                ctx.result,
                final_output_path,
                ctx.image_name,
                overwrite,
                mode_label,
                source_image=ctx.input,
            )
            ctx.output_path = final_output_path
            
            if success is None:
                ctx.success = True
                ctx.skipped = True
                ctx.skip_reason = 'existing_output_race'
                ctx.skip_message = f"Output file appeared during processing: {os.path.basename(final_output_path)}"
            elif success:
                ctx.success = True
            else:
                self._mark_context_failure(
                    ctx,
                    RuntimeError(f"Failed to save output file: {os.path.basename(final_output_path)}"),
                    stage='saving',
                )
            
            # Export an editable PSD (when enabled)
            from .utils.photoshop_export import psd_export_requested
            if success and psd_export_requested(config):
                try:
                    from .utils.photoshop_export import (
                        get_psd_output_path,
                        photoshop_export,
                        resolve_photoshop_font,
                    )
                    psd_path = get_psd_output_path(ctx.image_name)
                    cli_cfg = getattr(config, 'cli', None)
                    default_font = resolve_photoshop_font(config)
                    line_spacing = getattr(config.render, 'line_spacing', None) if hasattr(config, 'render') else None
                    script_only = getattr(cli_cfg, 'psd_script_only', False)
                    photoshop_export(psd_path, ctx, default_font, ctx.image_name, self.verbose, self._result_path, line_spacing, script_only)
                    logger.info(f"  -> ✅ [PSD] Exported editable PSD: {os.path.basename(psd_path)}")
                except Exception as psd_err:
                    logger.error(f"Error exporting PSD for {os.path.basename(ctx.image_name)}: {psd_err}")
            
            # ✅ Clear the result right after saving to free memory
            ctx.result = None
            
            return bool(ctx.success)
        except Exception as e:
            logger.error(f"Error in _save_and_cleanup_context: {e}")
            self._mark_context_failure(ctx, e, stage='saving')
            return False

    @staticmethod
    def _force_direction_and_alignment(regions_data: list, config: Config) -> None:
        """Apply the text direction and alignment the settings force to every region about to be saved."""
        try:
            # Override the direction
            if hasattr(config.render, 'direction'):
                dir_val = config.render.direction
                if hasattr(dir_val, 'value'): dir_val = dir_val.value

                forced_direction = None
                if dir_val == 'vertical': forced_direction = 'v'
                elif dir_val == 'horizontal': forced_direction = 'h'

                if forced_direction:
                    for region in regions_data:
                        region['direction'] = forced_direction

            # Override the alignment
            if hasattr(config.render, 'alignment'):
                align_val = config.render.alignment
                if hasattr(align_val, 'value'): align_val = align_val.value

                if align_val in ('left', 'center', 'right'):
                    for region in regions_data:
                        region['alignment'] = align_val

        except Exception as e:
            logger.warning(f"Failed to override region settings from config: {e}")

    @staticmethod
    def _preserved_skip_font_scaling(ctx: Context, text_output_file: str):
        """The skip_font_scaling flag to keep: the one on the context, else the one already in the file."""
        preserved_skip_font_scaling = getattr(ctx, 'skip_font_scaling', None)
        if preserved_skip_font_scaling is None and os.path.exists(text_output_file):
            try:
                with open(text_output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if existing_data and len(existing_data.values()) > 0:
                    existing_image_data = next(iter(existing_data.values()))
                    if isinstance(existing_image_data, dict) and 'skip_font_scaling' in existing_image_data:
                        preserved_skip_font_scaling = _parse_skip_font_scaling_flag(
                            existing_image_data.get('skip_font_scaling'),
                            default=True,
                        )
            except Exception as e:
                logger.warning(f"Failed to preserve skip_font_scaling from existing JSON {text_output_file}: {e}")
        return preserved_skip_font_scaling

    def _add_mask_to_saved_data(self, ctx: Context, data_to_save: dict) -> None:
        """Store the refined mask of a page with its saved text, unless the mode leaves masks out."""
        skip_mask_export = (
            getattr(ctx, 'used_imported_yolo_labels', False) and
            ((self.template and self.save_text) or self.generate_and_export)
        )

        # Save the refined mask (ctx.mask), not the raw mask (ctx.mask_raw)
        # so that it can be used directly after loading, without refining it again
        mask_to_save = None
        mask_is_refined = False
        if ctx.mask is not None:
            mask_to_save = ctx.mask
            mask_is_refined = True

        if skip_mask_export:
            logger.info("Import YOLO labels enabled in export mode: skipping mask save in JSON")
        elif self.save_mask and mask_to_save is not None:
            try:
                import base64

                import cv2
                _, buffer = cv2.imencode('.png', mask_to_save)
                mask_base64 = base64.b64encode(buffer).decode('utf-8')
                data_to_save['mask_raw'] = mask_base64
                data_to_save['mask_is_refined'] = mask_is_refined
            except Exception as e:
                logger.error(f"Failed to encode mask to base64: {e}")

    @staticmethod
    def _keep_existing_overlays(text_output_file: str, data_to_save: dict) -> None:
        """Carry over the paint, stamp and paste layers the editor stored in the existing file."""
        try:
            with open(text_output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if existing_data and len(existing_data.values()) > 0:
                existing_image_data = next(iter(existing_data.values()))
                if isinstance(existing_image_data, dict):
                    for overlay_key in ('paint_overlay', 'stamp_overlay', 'paste_overlay'):
                        overlay_value = existing_image_data.get(overlay_key)
                        if isinstance(overlay_value, str) and overlay_value:
                            data_to_save[overlay_key] = overlay_value
                    # Keep the list of editable paste overlays as it is, so a backend write-back does not lose data
                    paste_overlays_value = existing_image_data.get('paste_overlays')
                    if isinstance(paste_overlays_value, list):
                        data_to_save['paste_overlays'] = list(paste_overlays_value)
        except Exception as e:
            logger.debug(f"Failed to preserve overlay layers from existing JSON {text_output_file}: {e}")

    @staticmethod
    def _record_last_export_dir(ctx: Context, text_output_file: str, data_to_save: dict) -> None:
        """Record where the result was written, keeping the stored folder when nothing was written this time."""
        final_output_dir = getattr(ctx, 'final_output_dir', None)
        if final_output_dir:
            data_to_save['last_export_dir'] = final_output_dir
        elif os.path.exists(text_output_file):
            # Nothing was written this time (for example in JSON-only export); keep the stored last_export_dir
            try:
                with open(text_output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if existing_data and len(existing_data.values()) > 0:
                    existing_image_data = next(iter(existing_data.values()))
                    if isinstance(existing_image_data, dict):
                        preserved_dir = existing_image_data.get('last_export_dir')
                        if preserved_dir:
                            data_to_save['last_export_dir'] = preserved_dir
            except Exception as e:
                logger.debug(f"Failed to preserve last_export_dir from existing JSON {text_output_file}: {e}")

    def _save_text_to_file(self, image_path: str, ctx: Context, config: Config = None) -> bool:
        """Save or write back the text regions to JSON (with post-render fields such as translation and font_size), using the new folder structure"""
        text_output_file = self.text_output_file
        if not text_output_file:
            # Build the JSON path with the new path manager
            text_output_file = get_json_path(image_path, create_dir=True)

        data = {}
        image_key = os.path.abspath(image_path)

        # Prepare data for JSON serialization
        regions_data = [region.to_dict() for region in ctx.text_regions]

        global_font_family = ''
        if config and hasattr(config, 'render'):
            global_font_family = getattr(config.render, 'font_family', None) or ''
        if not global_font_family:
            global_font_family = self.font_family or ''

        for region in regions_data:
            if not region.get('font_family') and global_font_family:
                region['font_family'] = global_font_family
            region.pop('font_path', None)

        # Force the text direction and alignment from Config (when set)
        # so that, even when textline_merge detection used auto,
        # what is saved reflects the user's forced setting (for example, a whole book set horizontally)
        if config and hasattr(config, 'render'):
            self._force_direction_and_alignment(regions_data, config)

        # Image size (prefer the saved size; works in concurrent mode too)
        if hasattr(ctx, 'original_size') and ctx.original_size:
            original_width, original_height = ctx.original_size
        elif ctx.input and hasattr(ctx.input, 'size'):
            original_width, original_height = ctx.input.size
        else:
            # If neither exists, use the defaults or read them from the image file
            logger.warning("Cannot determine image dimensions; using defaults")
            original_width, original_height = 0, 0
        
        data_to_save = {
            'regions': regions_data,
            'original_width': original_width,
            'original_height': original_height
        }

        preserved_skip_font_scaling = self._preserved_skip_font_scaling(ctx, text_output_file)

        # Export original text / translate only (JSON): a later import should run smart layout again and not inherit the old font size.
        if (self.template and self.save_text) or self.translate_json_only:
            data_to_save['skip_font_scaling'] = False
        # Export-translation mode keeps the fixed font size, so the generated result can be replayed.
        elif self.generate_and_export:
            data_to_save['skip_font_scaling'] = True
        elif preserved_skip_font_scaling is not None:
            data_to_save['skip_font_scaling'] = bool(preserved_skip_font_scaling)

        # For a rendered ctx, region.translation has been replaced in place by prepare_text_replacements_for_layout
        # with the final text (translation_raw keeps the text from before); once flagged, a load_text re-render
        # does not replace it a second time. For exports without rendering (translate_json_only / export original text) the translation is still raw,
        # so the default False is kept and the replacement rules are applied when it is imported and rendered.
        if getattr(ctx, 'img_rendered', None) is not None:
            data_to_save['skip_text_replacements'] = True
        
        # Record the upscaling and colorization settings
        if config:
            if config.upscale and config.upscale.upscale_ratio:
                data_to_save['upscale_ratio'] = config.upscale.upscale_ratio
                if config.upscale.upscaler:
                    data_to_save['upscaler'] = config.upscale.upscaler
                logger.info(f"Recording upscaling information in JSON: ratio={config.upscale.upscale_ratio}, upscaler={config.upscale.upscaler}")
            
            if config.colorizer and config.colorizer.colorizer and config.colorizer.colorizer != 'none':
                data_to_save['colorizer'] = config.colorizer.colorizer
                logger.info(f"Recording colorization information in JSON: colorizer={config.colorizer.colorizer}")

        # Export modes that import YOLO boxes do not save the mask; load_text generates it later when it is missing
        self._add_mask_to_saved_data(ctx, data_to_save)

        # Keep the paint and stamp layers written by the editor (the backend write-back does not produce these keys, so they must not be lost)
        if os.path.exists(text_output_file):
            self._keep_existing_overlays(text_output_file, data_to_save)

        # Record the output folder of this main translation run; a later editor export writes back to it
        self._record_last_export_dir(ctx, text_output_file, data_to_save)

        data[image_key] = data_to_save

        try:
            # Use a custom encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)

            json_string = json.dumps(data, ensure_ascii=False, indent=4, cls=NumpyEncoder)

            with open(text_output_file, 'wb') as f:
                f.write(json_string.encode('utf-8'))
            logger.info(f"JSON saved to: {text_output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to write translation file to {text_output_file}: {e}")
            return False

    def _prepare_loaded_regions(self, loaded_regions: List[TextBlock], use_text_as_translation: bool = False) -> None:
        for region in loaded_regions:
            if not hasattr(region, 'font_size') or not region.font_size:
                try:
                    if region.lines.ndim == 3 and region.lines.shape[1] >= 4 and region.lines.shape[2] >= 2:
                        box_height = np.max(region.lines[:, :, 1]) - np.min(region.lines[:, :, 1])
                        region.font_size = min(int(box_height * 0.8), 128)
                    else:
                        logger.warning(f"Invalid lines shape {region.lines.shape}, using default font_size=24")
                        region.font_size = 24
                except Exception as e:
                    logger.warning(f"Error calculating font_size from lines: {e}, using default font_size=24")
                    region.font_size = 24

            if use_text_as_translation and not region.translation:
                region.translation = region.text
                logger.debug(f"Region translation is empty, using original text: {region.text[:50]}...")

    def _apply_pre_dictionary_to_regions(self, ctx: Context) -> None:
        pre_dict = load_dictionary(self.pre_dict)
        pre_replacements = []
        for region in ctx.text_regions:
            if region.text is None:
                continue
            original = region.text
            region.text = apply_dictionary(region.text, pre_dict)
            if original != region.text:
                pre_replacements.append(f"{original} => {region.text}")

        if pre_replacements:
            logger.info("Pre-translation replacements:")
            for replacement in pre_replacements:
                logger.info(replacement)
        else:
            logger.info("No pre-translation replacements made.")

    def _delete_original_txt_after_json_translation(self, image_path: str) -> None:
        original_txt_path = get_original_txt_path(image_path, create_dir=False)
        if not os.path.exists(original_txt_path):
            return

        try:
            os.remove(original_txt_path)
            logger.info(f"Deleted original text file after JSON translation: {original_txt_path}")
        except Exception as e:
            logger.warning(f"Failed to delete original text file {original_txt_path}: {e}")

    async def _export_text_from_local_json(
        self,
        images_with_configs: List[tuple],
        global_offset: int = 0,
        global_total: int = None,
        skipped_count: int = 0,
    ) -> List[Context]:
        """Export original or translated text without touching the project JSON."""
        from desktop_qt_ui.services import workflow_service

        generator = (
            workflow_service.generate_translated_text
            if self.generate_and_export
            else workflow_service.generate_original_text
        )
        export_label = "translated" if self.generate_and_export else "original"
        template_path = workflow_service.get_template_path_from_config()
        display_total = global_total if global_total is not None else len(images_with_configs)
        results: List[Context] = []

        logger.info(
            "Local JSON text export enabled: reading existing project JSON only; "
            "detection, OCR, translation, and JSON write-back are skipped."
        )
        for image_or_path, config in images_with_configs:
            await asyncio.sleep(0)
            self._check_cancelled()
            image_name = (
                os.fspath(image_or_path)
                if isinstance(image_or_path, (str, os.PathLike))
                else getattr(image_or_path, 'name', None)
            )
            ctx = Context()
            ctx.text_regions = []
            if image_name:
                ctx.image_name = image_name
            if config is not None:
                ctx.config = config

            try:
                if not image_name:
                    raise ValueError("Input image path is unavailable")
                json_path = find_json_path(image_name)
                if not json_path:
                    raise FileNotFoundError(
                        f"Local project JSON not found for {os.path.basename(image_name)}"
                    )
                export_result = generator(json_path, template_path)
                if export_result.startswith("Error"):
                    raise OSError(export_result)
                ctx.success = True
                ctx.output_path = export_result
                logger.info(
                    f"Local {export_label} text exported for "
                    f"{os.path.basename(image_name)}: {export_result}"
                )
            except Exception as exc:
                logger.error(
                    f"Failed to export local {export_label} text for "
                    f"{os.path.basename(image_name) if image_name else 'unknown input'}: {exc}"
                )
                self._mark_context_failure(ctx, exc, stage='export')

            results.append(ctx)
            completed = min(global_offset + len(results), display_total)
            failed_count = sum(
                1 for result in results if getattr(result, 'translation_error', None)
            )
            await self._report_progress(
                f"batch:{completed}:{completed}:{display_total}:{failed_count}:{skipped_count}"
            )

        return results

    async def _handle_template_export(
        self,
        ctx: Context,
        config: Config,
        ensure_json_with_empty_regions: bool,
        *,
        imported_yolo_mode_label: str,
        mask_error_mode_label: str,
        empty_regions_mode_label: str,
        generator_name: str,
        export_log_label: str,
        export_error_label: str,
    ) -> None:
        """Shared template export workflow for original/translated text modes."""
        image_name = getattr(ctx, 'image_name', None)
        if not image_name:
            return

        has_regions = hasattr(ctx, 'text_regions') and ctx.text_regions is not None
        has_non_empty_regions = has_regions and bool(ctx.text_regions)

        # Export modes that import YOLO boxes do not save the mask
        if getattr(ctx, 'used_imported_yolo_labels', False):
            logger.info(f"Import YOLO labels enabled in {imported_yolo_mode_label}: skipping mask refinement and mask save")
            ctx.mask = None
            ctx.mask_raw = None
        # Export mode: always refine the mask (inpainting is skipped)
        elif has_non_empty_regions and ctx.mask is None and ctx.mask_raw is not None:
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
            except Exception:
                logger.error(f"Error during mask-generation in {mask_error_mode_label}:\n{traceback.format_exc()}")
                ctx.mask = ctx.mask_raw  # Fall back to the raw mask
        elif not has_non_empty_regions and ctx.mask_raw is not None:
            logger.info(
                f"{empty_regions_mode_label}: no text regions for {os.path.basename(image_name)}, "
                "skipping mask refinement and exporting empty JSON/TXT only"
            )

        should_export = has_regions and (has_non_empty_regions or ensure_json_with_empty_regions)
        if not should_export:
            return

        self._save_text_to_file(image_name, ctx, config)

        try:
            json_path = find_json_path(image_name)
            if json_path and os.path.exists(json_path):
                from desktop_qt_ui.services import workflow_service
                template_path = workflow_service.get_template_path_from_config()
                if template_path and os.path.exists(template_path):
                    export_result = getattr(workflow_service, generator_name)(json_path, template_path)
                    ctx.output_path = export_result
                    logger.info(f"{export_log_label} for {os.path.basename(image_name)}: {export_result}")
                else:
                    logger.warning(f"Template file not found for {os.path.basename(image_name)}: {template_path}")
            else:
                logger.warning(f"JSON file not found for {os.path.basename(image_name)}")
        except Exception as e:
            logger.error(f"{export_error_label} for {os.path.basename(image_name)}: {e}")

    async def _handle_generate_and_export(
        self,
        ctx: Context,
        config: Config,
        ensure_json_with_empty_regions: bool = False
    ) -> None:
        """
        Shared generate_and_export workflow:
        1) refine mask if available
        2) save JSON
        3) export translated TXT via template
        """
        await self._handle_template_export(
            ctx,
            config,
            ensure_json_with_empty_regions,
            imported_yolo_mode_label="generate_and_export mode",
            mask_error_mode_label="generate_and_export mode",
            empty_regions_mode_label="Generate-and-export mode",
            generator_name="generate_translated_text",
            export_log_label="Translated text export",
            export_error_label="Failed to export clean text",
        )

    async def _handle_template_and_save_text(
        self,
        ctx: Context,
        config: Config,
        ensure_json_with_empty_regions: bool = True
    ) -> None:
        """
        Shared template+save_text workflow:
        1) refine mask if available
        2) save JSON
        3) export original TXT via template
        """
        await self._handle_template_export(
            ctx,
            config,
            ensure_json_with_empty_regions,
            imported_yolo_mode_label="template mode",
            mask_error_mode_label="template mode",
            empty_regions_mode_label="Template mode",
            generator_name="generate_original_text",
            export_log_label="Original text export",
            export_error_label="Failed to export original text",
        )

    def _save_inpainted_image(
        self,
        image_path: str,
        inpainted_img: np.ndarray,
    ) -> Optional[str]:
        if inpainted_img is None:
            return None
        inpainted_path = get_inpainted_path(image_path, create_dir=True)
        return self._save_image_to_path(
            inpainted_path,
            inpainted_img,
            "Inpainted image",
            source_image_path=image_path,
        )

    def _save_work_image(self, image_path: str, image_data, label: str = "Work image") -> Optional[str]:
        """Save the colorized or upscaled base image for the editor to the editor_base folder."""
        work_image_path = get_work_image_path(image_path, create_dir=True)
        return self._save_image_to_path(work_image_path, image_data, label, source_image_path=image_path)

    def _save_editor_base_if_needed(self, ctx, config, image_data=None) -> Optional[str]:
        """Save the base image the editor uses when colorization or upscaling was run."""
        input_image = getattr(ctx, 'input', None)
        image_path = getattr(input_image, 'name', None)
        if not image_path:
            return None

        has_colorized = config.colorizer.colorizer != Colorizer.none
        has_upscaled = bool(config.upscale.upscale_ratio)
        if not has_colorized and not has_upscaled:
            return None

        if image_data is None:
            image_data = getattr(ctx, 'upscaled', None) or getattr(ctx, 'img_colorized', None)
        if image_data is None:
            return None

        return self._save_work_image(image_path, image_data, "Processed base image")

    def _save_image_to_path(
        self,
        target_path: str,
        image_data,
        label: str,
        source_image_path: Optional[str] = None,
    ) -> Optional[str]:
        """Save an image to the given path."""
        try:
            if image_data is None:
                return None

            source_image = None
            if isinstance(image_data, Image.Image):
                image_to_save = image_data.copy()
            elif isinstance(image_data, np.ndarray):
                image_to_save = Image.fromarray(image_data)
            else:
                raise TypeError(f"Unsupported work image type: {type(image_data)}")

            try:
                if source_image_path and os.path.exists(source_image_path):
                    try:
                        source_image = open_pil_image(source_image_path, eager=True)
                    except Exception as exc:
                        logger.warning(
                            f"Failed to read source image metadata for {label.lower()}, saving without ICC: "
                            f"{source_image_path}, error={exc}"
                        )

                save_pil_image(
                    image_to_save,
                    target_path,
                    source_image=source_image,
                    quality=self.save_quality,
                )
                if self.verbose:
                    logger.debug(f"{label} saved to: {target_path}")
                return target_path
            finally:
                if source_image is not None:
                    source_image.close()
                image_to_save.close()
        except Exception as e:
            logger.error(f"Failed to save {label.lower()}: {e}")
            return None

    def _preprocess_load_text_mode(self, images_with_configs: List[tuple]):
        """
        Preprocessing for load_text mode: import translations from TXT files into JSON automatically.
        This method runs once before translation starts, so both the CLI and the UI can use it
        """
        try:
            from manga_translator.utils.path_manager import (
                find_json_path,
                find_txt_files,
            )
            
            # Path of the default template
            template_path = self._get_default_template_path()
            if not template_path or not os.path.exists(template_path):
                logger.warning("Template file not found, skipping TXT to JSON import")
                return
            
            # Collect the image paths to process
            image_paths = []
            for image, config in images_with_configs:
                image_path = None
                if hasattr(image, 'name') and image.name:
                    image_path = image.name
                elif isinstance(image, (str, os.PathLike)):
                    image_path = os.fspath(image)

                if image_path:
                    image_paths.append(image_path)
                else:
                    continue
            
            if not image_paths:
                return
            
            # Import the TXT files as a batch
            success_count = 0
            skip_count = 0
            
            for image_path in image_paths:
                try:
                    # Look for the JSON and TXT files
                    json_path = find_json_path(image_path)
                    original_txt_path, translated_txt_path = find_txt_files(image_path)
                    
                    # Skip when there is no JSON file (an error is reported later)
                    if not json_path:
                        skip_count += 1
                        continue
                    
                    # Prefer the original-text TXT, then the translation TXT
                    txt_path = original_txt_path if original_txt_path else translated_txt_path
                    
                    if not txt_path:
                        skip_count += 1
                        continue
                    
                    # Import the TXT into the JSON
                    from desktop_qt_ui.services.workflow_service import (
                        safe_update_large_json_from_text,
                    )
                    result = safe_update_large_json_from_text(txt_path, json_path, template_path)
                    
                    if not result.startswith("Error"):
                        success_count += 1
                        logger.debug(f"Imported TXT to JSON: {os.path.basename(image_path)}")
                    
                except Exception as e:
                    logger.debug(f"Failed to import TXT for {os.path.basename(image_path)}: {e}")
                    continue
            
            if success_count > 0:
                logger.info(f"TXT to JSON import completed: {success_count} successful, {skip_count} skipped")
            elif skip_count > 0:
                logger.debug(f"No TXT files found for import ({skip_count} images)")
                
        except ImportError as e:
            logger.warning(f"Cannot import workflow_service, skipping TXT to JSON import: {e}")
        except Exception as e:
            logger.warning(f"Error during TXT to JSON import: {e}")

    
    def _get_default_template_path(self) -> Optional[str]:
        """Get the path of the default template file"""
        try:
            from manga_translator.runtime_paths import get_config_path

            possible_paths = [get_config_path('translation_template.json')]
            
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    return abs_path
            
            # If neither exists, try to create the default template
            default_path = os.path.abspath(possible_paths[0])
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            
            default_content = '''翻译模板文件

原文: <original>
译文: <translated>

'''
            with open(default_path, 'w', encoding='utf-8') as f:
                f.write(default_content)
            
            logger.info(f"Created default template at: {default_path}")
            return default_path
            
        except Exception as e:
            logger.warning(f"Failed to get/create default template: {e}")
            return None
    
    def set_preloaded_load_text_payload(self, image_name: str, payload: Optional[dict]) -> None:
        """Register an in-memory load_text payload (the editor export channel).

        A registered payload counts as an editor export (ctx.editor_export=True): content and layout are
        the final version authorised by the editor, and the backend skips text replacement and the project JSON write-back.

        ``editor_export_base_kind`` distinguishes source, paired and
        backend_inpaint explicitly. Only paired carries ``inpainted_rgb`` and renders without inpainting;
        backend_inpaint must run inpainting and never reads a sidecar from disk or takes the AI renderer's
        fallback to the original image. Both mask_raw and the inpainted image may be carried as ndarrays.
        The payload is consumed in place while it is parsed and has to be registered again before each translate.
        """
        if not image_name:
            return
        if payload is None:
            self._preloaded_load_text_payloads.pop(image_name, None)
        else:
            self._preloaded_load_text_payloads[image_name] = payload

    def _extract_render_overlays(self, image_data: dict):
        """Extract the paint layer and the stamp layer (RGBA) from the JSON or the in-memory payload, returned as a list in compositing order.

        Two forms are accepted: an ndarray passed in memory, and a base64 PNG string in the JSON.
        """
        import base64

        overlays = []
        for key in ('paint_overlay', 'stamp_overlay', 'paste_overlay'):
            value = image_data.get(key)
            arr = None
            if isinstance(value, np.ndarray):
                arr = value
            elif isinstance(value, str) and value:
                try:
                    buf = np.frombuffer(base64.b64decode(value), dtype=np.uint8)
                    bgra = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                    if bgra is not None and bgra.ndim == 3 and bgra.shape[2] == 4:
                        arr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
                except Exception as e:
                    logger.warning(f"Failed to decode {key} from JSON: {e}")
            if arr is not None and arr.ndim == 3 and arr.shape[2] == 4 and np.any(arr[..., 3]):
                overlays.append(arr.astype(np.uint8, copy=False))
        return overlays or None

    def _compose_render_overlays_on_inpainted(self, ctx):
        """Alpha-composite the paint, stamp and paste layers onto the base image in order (called before rendering).

        The base image is ctx.img_inpainted when available; without an inpainted image (for example a page with
        "paste overlays only, no text boxes or mask") it falls back to ctx.upscaled. The list of editable paste overlays
        loaded from disk (paste_overlays) is materialised here and appended to the end of the overlay layers (above paint and stamp, as in the editor).
        """
        pending = getattr(self, '_pending_paste_overlays', None)
        overlays = getattr(self, '_loaded_render_overlays', None) or []
        if not overlays and not pending:
            return

        base = getattr(ctx, 'img_inpainted', None)
        base_holder = 'inpainted'
        if base is None:
            base = getattr(ctx, 'upscaled', None)
            base_holder = 'upscaled'
        if base is None:
            return
        was_pil = hasattr(base, 'resize') and hasattr(base, 'mode')
        try:
            base_arr = np.asarray(base)
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._compose_render_overlays_on_inpainted")
            return
        if base_arr.ndim != 3 or base_arr.shape[2] < 3:
            return
        h, w = base_arr.shape[:2]

        if pending:
            try:
                # Materialise the editable paste overlays stored on disk; when parsing or compositing fails they are skipped and the other layers are unaffected
                from desktop_qt_ui.editor.paste_overlay_state import (
                    compose_paste_overlays,
                    parse_page_paste_overlays,
                )

                parsed = parse_page_paste_overlays({'paste_overlays': pending})
                composed = compose_paste_overlays(parsed, (w, h))
                if composed is not None:
                    overlays = list(overlays) + [composed]
                self._pending_paste_overlays = None
            except Exception as materialize_error:
                logger.warning(
                    f"Failed to materialize paste_overlays on disk load: {materialize_error}"
                )
                self._pending_paste_overlays = None

        if not overlays:
            return
        has_alpha = base_arr.shape[2] >= 4
        result = base_arr.copy()
        if has_alpha:
            # RGBA source-over: keep the transparency of the source image and let the overlay cover transparent areas
            composed_rgb = base_arr[..., :3].astype(np.float32)
            composed_alpha = base_arr[..., 3:4].astype(np.float32)
            for overlay in overlays:
                if overlay.shape[:2] != (h, w):
                    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_NEAREST)
                overlay_alpha = overlay[..., 3:4].astype(np.float32) / 255.0
                base_coverage = composed_alpha / 255.0
                out_alpha = overlay_alpha + base_coverage * (1.0 - overlay_alpha)
                safe = np.maximum(out_alpha, 1e-6)
                composed_rgb = (
                    overlay[..., :3].astype(np.float32) * overlay_alpha
                    + composed_rgb * base_coverage * (1.0 - overlay_alpha)
                ) / safe
                composed_alpha = out_alpha * 255.0
            result[..., :3] = np.clip(composed_rgb, 0, 255).astype(np.uint8)
            result[..., 3:4] = np.clip(composed_alpha, 0, 255).astype(np.uint8)
        else:
            composed = base_arr[..., :3].astype(np.float32)
            for overlay in overlays:
                if overlay.shape[:2] != (h, w):
                    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_NEAREST)
                alpha = overlay[..., 3:4].astype(np.float32) / 255.0
                composed = composed * (1.0 - alpha) + overlay[..., :3].astype(np.float32) * alpha
            result[..., :3] = np.clip(composed, 0, 255).astype(np.uint8)
        if base_holder == 'inpainted':
            ctx.img_inpainted = result
        elif was_pil:
            # When the base image is a PIL image (the fallback without text boxes uses ctx.upscaled), write a PIL image back in the same mode,
            # so that numpy does not break PIL semantics downstream, such as resize or "if ctx.result";
            # an RGBA source keeps its transparency in the output
            from PIL import Image as _PillowImage

            if has_alpha:
                ctx.upscaled = _PillowImage.fromarray(result)
            else:
                ctx.upscaled = _PillowImage.fromarray(result[..., :3])
        else:
            ctx.upscaled = result
        logger.info(
            f"Composited {len(overlays)} paint/stamp/paste overlay layer(s) onto render base"
        )

    @staticmethod
    def _normalise_saved_region_fields(region_data: dict, config: Config) -> None:
        """Rename and convert the fields of a saved region to what TextBlock expects."""
        # Convert literal '\\n' to newline characters for the rendering engine
        if 'text' in region_data and isinstance(region_data['text'], str):
            region_data['text'] = region_data['text'].replace('\\n', '\n')
        if 'translation' in region_data and isinstance(region_data['translation'], str):
            region_data['translation'] = region_data['translation'].replace('\\n', '\n')

        # If target_lang is missing or empty, set it from the config.
        if not region_data.get('target_lang'):
            if config and config.translator and config.translator.target_lang:
                region_data['target_lang'] = config.translator.target_lang
                logger.debug(f"Region target_lang missing in JSON, falling back to config's target_lang: {config.translator.target_lang}")

        # Convert hex font_color from editor to an 'fg_color' tuple for TextBlock
        if 'font_color' in region_data and isinstance(region_data['font_color'], str):
            hex_color = region_data.pop('font_color') # Use pop to remove the old key
            if hex_color.startswith('#') and len(hex_color) == 7:
                try:
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    region_data['fg_color'] = (r, g, b)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse font_color '{hex_color}': {e}")

        # Map 'fg_colors' (list) to 'fg_color' (tuple) if present
        if 'fg_colors' in region_data:
            fg_val = region_data.pop('fg_colors')
            if isinstance(fg_val, list):
                region_data['fg_color'] = tuple(fg_val)

        # Map 'bg_colors' or 'text_stroke_color' to 'bg_color'
        if 'bg_colors' in region_data:
            bg_val = region_data.pop('bg_colors')
            if isinstance(bg_val, list):
                region_data['bg_color'] = tuple(bg_val)
        elif 'text_stroke_color' in region_data: # Handle UI specific name
            bg_val = region_data.pop('text_stroke_color')
            if isinstance(bg_val, list): # List RGB
                 region_data['bg_color'] = tuple(bg_val)
            elif isinstance(bg_val, str) and bg_val.startswith('#'): # Hex string
                try:
                    r = int(bg_val[1:3], 16)
                    g = int(bg_val[3:5], 16)
                    b = int(bg_val[5:7], 16)
                    region_data['bg_color'] = (r, g, b)
                except (ValueError, TypeError):
                     pass


        # Stroke width: stroke_width takes precedence over default_stroke_width
        # A stroke_width the user set in the editor should override the original default_stroke_width
        if 'stroke_width' in region_data:
            region_data['default_stroke_width'] = region_data.pop('stroke_width')

    @staticmethod
    def _text_block_from_saved_region(region_data: dict, text_file_path: str):
        """Build the TextBlock of a saved region, giving up its rich styling when that is what fails."""
        try:
            region = TextBlock(**region_data)
        except Exception as construct_err:
            # Safety fuse: a parse failure must not swallow the whole region (otherwise the JSON write-back loses the region
            # for good, together with its original text and coordinates). Strip translation_rich and retry once in degraded form
            # - losing the styling is acceptable, losing the region is not; only if that also fails is it counted and skipped.
            if isinstance(region_data, dict) and 'translation_rich' in region_data:
                degraded_data = {k: v for k, v in region_data.items() if k != 'translation_rich'}
                region = TextBlock(**degraded_data)
                logger.warning(
                    f"Region in {text_file_path} failed to load with translation_rich, "
                    f"discarded rich styling and kept the region: {construct_err}"
                )
            else:
                raise
        return region

    @staticmethod
    def _decode_saved_mask(mask_raw_data):
        """Turn the mask stored with a page (array, base64 PNG or list) into an array, or None."""
        mask_raw = None
        if isinstance(mask_raw_data, np.ndarray):
            # The in-memory payload carries the mask as an ndarray, so base64/PNG encoding is skipped
            mask_raw = mask_raw_data.astype(np.uint8, copy=False)
        elif isinstance(mask_raw_data, str):
            try:
                import base64

                import cv2
                img_bytes = base64.b64decode(mask_raw_data)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                mask_raw = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            except Exception as e:
                logger.error(f"Failed to decode base64 mask: {e}")
        elif isinstance(mask_raw_data, list):
            mask_raw = np.array(mask_raw_data, dtype=np.uint8)
        return mask_raw

    def _load_text_and_regions_from_file(self, image_path: str, config: Config):
        """Load the translation data; supports the new folder structure and stays backward compatible"""
        self._loaded_render_overlays = None
        self._pending_paste_overlays = None
        if not image_path:
            return None, None, False, True, False, 0

        preloaded = self._preloaded_load_text_payloads.get(image_path)
        if preloaded is not None:
            # In-memory path: the editor export injects the data directly, skipping the JSON file lookup and parsing
            text_file_path = f"<preloaded:{os.path.basename(image_path)}>"
            image_data = preloaded
        else:
            # Find the JSON file with path_manager (the new location is preferred)
            text_file_path = find_json_path(image_path)

            if not text_file_path:
                # Check for the old TXT format
                base_path, _ = os.path.splitext(image_path)
                text_file_path_txt = base_path + '_translations.txt'
                if os.path.exists(text_file_path_txt):
                    # If the old format is found, load from it
                    regions = self._load_text_and_regions_from_txt_file(image_path)
                    # Since old format doesn't have mask, we return None for mask and refined status
                    return regions, None, False, True, False, 0
                else:
                    logger.info(f"Translation file not found for: {image_path}")
                    return None, None, False, True, False, 0

            try:
                # Force UTF-8 encoding to handle potential file encoding issues
                with open(text_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read or parse translation file {text_file_path}: {e}")
                return None, None, False, True, False, 0

            # Don't check the image key. Assume the user knows what they are doing
            # and that the first entry in the JSON is the one they want to load.
            if not data or len(data.values()) == 0:
                logger.warning(f"JSON file {text_file_path} is empty or invalid.")
                return None, None, False, True, False, 0

            # Get the first value from the dictionary, regardless of the key.
            image_data = next(iter(data.values()))
        mask_is_refined = False
        skip_font_scaling = True
        skip_text_replacements = False

        # Handle both old and new JSON formats
        if isinstance(image_data, list):
            # Old format: value is a list of regions
            regions_data = image_data
            mask_raw_data = None
        elif isinstance(image_data, dict):
            # New format: value is a dict with 'regions' and 'mask_raw'
            regions_data = image_data.get('regions', [])
            mask_raw_data = image_data.get('mask_raw', None)
            mask_is_refined = image_data.get('mask_is_refined', False)
            skip_font_scaling = _parse_skip_font_scaling_flag(
                image_data.get('skip_font_scaling', True),
                default=True,
            )
            skip_text_replacements = _parse_skip_font_scaling_flag(
                image_data.get('skip_text_replacements', False),
                default=False,
            )
            # Loaded from disk: the editable paste overlays are materialised at compositing time (an in-memory preload only carries the already composited single paste_overlay)
            paste_overlays_value = image_data.get('paste_overlays')
            if (
                image_data.get('paste_overlay') is None
                and isinstance(paste_overlays_value, list)
            ):
                self._pending_paste_overlays = paste_overlays_value
            self._loaded_render_overlays = self._extract_render_overlays(image_data)
        else:
            logger.warning(f"Invalid data format in JSON file {text_file_path}.")
            return None, None, False, True, False, 0

        if preloaded is not None:
            # Editor export: text replacement already took effect while editing, so it is always skipped
            skip_text_replacements = True

        regions = []
        parse_failure_count = 0
        for region_data in regions_data:
            try:
                self._normalise_saved_region_fields(region_data, config)
                
                # Make sure line_spacing and default_stroke_width are passed on correctly
                # These values are already in region_data and are received by the TextBlock constructor

                # Recreate the TextBlock object by unpacking the dictionary
                # This restores all saved attributes
                if 'lines' in region_data and isinstance(region_data['lines'], list):
                    # Fix: Use np.float64 to match TextBlock expectation, not np.int32
                    lines_arr = np.array(region_data['lines'], dtype=np.float64)
                    # Check and correct the shape so that it is (N, 4, 2)
                    if lines_arr.ndim == 2 and lines_arr.shape == (4, 2):
                        lines_arr = lines_arr.reshape(1, 4, 2)
                    elif lines_arr.ndim != 3 or lines_arr.shape[1] != 4 or lines_arr.shape[2] != 2:
                        logger.warning(f"[Load JSON] Invalid lines shape: {lines_arr.shape}; skipping this region")
                        parse_failure_count += 1
                        continue
                    region_data['lines'] = lines_arr

                # Import-translation mode: the user has confirmed the colours, so the stroke colour is not adjusted automatically
                region_data['adjust_bg_color'] = False
                region = self._text_block_from_saved_region(region_data, text_file_path)
                regions.append(region)
            except Exception as e:
                parse_failure_count += 1
                logger.error(f"Failed to parse a region in {text_file_path}: {e}")
                continue
        
        mask_raw = self._decode_saved_mask(mask_raw_data)
        
        logger.info(f"Loaded {len(regions)} regions from {text_file_path}")
        if parse_failure_count:
            logger.error(
                f"{parse_failure_count} region(s) in {text_file_path} could not be parsed and were skipped; "
                "JSON write-back will be disabled for this image to protect the project file"
            )
        if mask_raw is not None:
            logger.info(f"Loaded mask_raw from {text_file_path}")

        if getattr(getattr(config, 'render', None), 'recompute_line_breaks', False):
            # Saved files normally keep their stored layout. Recomputing line breaks
            # needs the full layout step, which also refits the font size.
            skip_font_scaling = False
        return regions, mask_raw, mask_is_refined, skip_font_scaling, skip_text_replacements, parse_failure_count

    def _load_text_and_regions_from_txt_file(self, image_path: str) -> Optional[List[TextBlock]]:
        """
        Loader for the old TXT format (deprecated).
        Only the JSON format is supported now; this method is kept for backward compatibility but is no longer implemented
        """
        logger.warning("TXT format is deprecated and no longer supported. Please use JSON format instead.")
        return None

    # If `revert_upscaling` is True, revert to input size
    # Else leave `ctx` as-is
    async def _revert_upscale(self, config: Config, ctx: Context):
        if config.upscale.revert_upscaling:
            await self._report_progress('downscaling')
            ctx.result = ctx.result.resize(ctx.input.size)

        # In verbose mode, save final.png to the debug folder
        if ctx.result and self.verbose:
            try:
                final_img = np.array(ctx.result)
                if len(final_img.shape) == 3:  # Colour image: convert to BGR order
                    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
                final_path = self._result_path('final.png')
                imwrite_unicode(final_path, final_img, logger)
            except Exception as e:
                logger.error(f"Error saving final.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        return ctx

    def _get_ai_colorizer_history_pages(self, config: Config) -> int:
        colorizer_config = getattr(config, 'colorizer', None)
        try:
            return max(int(getattr(colorizer_config, 'ai_colorizer_history_pages', 0) or 0), 0)
        except (TypeError, ValueError):
            return 0

    def _should_use_ai_colorizer_history(self, config: Config) -> bool:
        colorizer_config = getattr(config, 'colorizer', None)
        colorizer_type = getattr(colorizer_config, 'colorizer', None)
        return (
            self._get_ai_colorizer_history_pages(config) > 0
            and colorizer_type in {
                Colorizer.openai_colorizer,
                Colorizer.gemini_colorizer,
            }
        )

    def _get_colorizer_history_images(self, config: Config) -> list[Image.Image]:
        if not self._should_use_ai_colorizer_history(config):
            return []

        history_pages = self._get_ai_colorizer_history_pages(config)
        if history_pages <= 0 or not self._colorizer_history_images:
            return []
        return list(self._colorizer_history_images[-history_pages:])

    def _append_colorizer_history_image(self, config: Config, image) -> None:
        if not self._should_use_ai_colorizer_history(config) or image is None:
            return

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype(np.uint8))

        history_image = normalize_ai_image(image).copy()
        self._colorizer_history_images.append(history_image)

        history_pages = self._get_ai_colorizer_history_pages(config)
        if history_pages <= 0 or len(self._colorizer_history_images) <= history_pages:
            return

        stale_images = self._colorizer_history_images[:-history_pages]
        self._colorizer_history_images = self._colorizer_history_images[-history_pages:]
        for stale_image in stale_images:
            if hasattr(stale_image, 'close'):
                try:
                    stale_image.close()
                except Exception as ignored_error:
                    note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._append_colorizer_history_image")
                    pass

    def _clear_colorizer_history(self) -> None:
        for history_image in self._colorizer_history_images:
            if hasattr(history_image, 'close'):
                try:
                    history_image.close()
                except Exception as ignored_error:
                    note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._clear_colorizer_history")
                    pass
        self._colorizer_history_images = []

    async def _run_colorizer(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("colorizer", config.colorizer.colorizer)] = current_time

        result = await dispatch_colorization(
            config.colorizer.colorizer,
            colorization_size=config.colorizer.colorization_size,
            denoise_sigma=config.colorizer.denoise_sigma,
            device=self.device,
            image=ctx.input,
            config=config,
            colorizer_history_images=self._get_colorizer_history_images(config),
        )
        self._append_colorizer_history_image(config, result)
        return result

    async def _run_upscaling(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("upscaling", config.upscale.upscaler)] = current_time
        
        # Prepare kwargs for Real-CUGAN (NCNN version) and MangaJaNai
        upscaler_kwargs = {}
        if config.upscale.upscaler == 'realcugan':
            realcugan_model = getattr(config.upscale, 'realcugan_model', None)
            if realcugan_model:
                upscaler_kwargs['model_name'] = realcugan_model
            # tile_size: None=use upscaler default, 0=no tiling, >0=manual tile size
            tile_size = getattr(config.upscale, 'tile_size', None)
            if tile_size is not None:
                upscaler_kwargs['tile_size'] = tile_size
        elif config.upscale.upscaler == 'mangajanai':
            # For mangajanai, upscale_ratio can be a string (x2, x4, DAT2 x4) or a number
            ratio = config.upscale.upscale_ratio
            if isinstance(ratio, str):
                upscaler_kwargs['model_name'] = ratio
                # Parse the actual ratio from the string
                if 'x2' in ratio.lower():
                    actual_ratio = 2
                else:
                    actual_ratio = 4
            elif ratio == 2:
                upscaler_kwargs['model_name'] = 'x2'
                actual_ratio = 2
            else:
                upscaler_kwargs['model_name'] = 'x4'
                actual_ratio = 4

            tile_size = getattr(config.upscale, 'tile_size', None)
            if tile_size is not None:
                upscaler_kwargs['tile_size'] = tile_size
        
        # Get the actual numeric ratio
        if config.upscale.upscaler == 'mangajanai':
            upscale_ratio_num = actual_ratio
        else:
            upscale_ratio_num = config.upscale.upscale_ratio
        
        result = (await dispatch_upscaling(
            config.upscale.upscaler, 
            [ctx.img_colorized], 
            upscale_ratio_num, 
            self.device,
            **upscaler_kwargs
        ))[0]
        
        if self.models_ttl > 0:
            logger.info(f"Upscaling model {config.upscale.upscaler} will be unloaded after {self.models_ttl}s of inactivity")
        else:
            logger.debug(f"Keeping upscaling model {config.upscale.upscaler} loaded because models_ttl=0")
        
        return result

    def _save_detection_debug_images(self, config: Config, ctx: Context, result):
        """Write the debug images a detector returned and drop them from its result."""
        third_elem = result[2]
        # Check whether it is a tuple (holding three images)
        if isinstance(third_elem, tuple) and len(third_elem) == 3:
            try:
                logger.info(f'[DEBUG] Processing 3-element tuple: {[type(x) for x in third_elem]}')
                bbox_img, binary_mask_img, raw_mask_mask = third_elem
                # Save the bounding box debug image
                bbox_debug_path = self._result_path('bboxes_with_scores.png')
                imwrite_unicode(bbox_debug_path, bbox_img, logger)
                logger.info(f'Saved bbox debug image to {bbox_debug_path}')
                # Save the binary mask
                binary_mask_path = self._result_path('mask_binary.png')
                imwrite_unicode(binary_mask_path, binary_mask_img, logger)
                logger.info(f'Saved binary mask to {binary_mask_path}')
                # Keep raw_mask_mask on ctx so a comparison image can be generated later
                ctx.raw_mask_mask = raw_mask_mask
                logger.info(f'[DEBUG] Stored raw_mask_mask for later comparison (shape: {raw_mask_mask.shape})')
                result = (result[0], result[1], None)
            except Exception as e:
                logger.error(f'Failed to save bbox debug images: {e}')
        # Handle the case of two images
        elif isinstance(third_elem, tuple) and len(third_elem) == 2:
            try:
                bbox_img, binary_mask_img = third_elem
                bbox_debug_path = self._result_path('bboxes_with_scores.png')
                imwrite_unicode(bbox_debug_path, bbox_img, logger)
                logger.info(f'Saved bbox debug image to {bbox_debug_path}')
                binary_mask_path = self._result_path('mask_binary.png')
                imwrite_unicode(binary_mask_path, binary_mask_img, logger)
                logger.info(f'Saved binary mask to {binary_mask_path}')
                result = (result[0], result[1], None)
            except Exception as e:
                logger.error(f'Failed to save bbox debug images: {e}')
        # Handle the case of one image (including the hybrid detection debug image)
        elif isinstance(third_elem, np.ndarray) and len(third_elem.shape) == 3:
            try:
                # When YOLO-assisted detection is on, save it as the hybrid detection debug image
                if config.detector.use_yolo_obb:
                    # Save the hybrid detection debug image
                    hybrid_debug_path = self._result_path('hybrid_detection_boxes.png')
                    imwrite_unicode(hybrid_debug_path, cv2.cvtColor(third_elem, cv2.COLOR_RGB2BGR), logger)
                    logger.info(f'✅ Saved hybrid detection debug image: {hybrid_debug_path}')
                else:
                    # Save the ordinary bbox debug image
                    bbox_debug_path = self._result_path('bboxes_with_scores.png')
                    imwrite_unicode(bbox_debug_path, third_elem, logger)
                    logger.info(f'Saved bbox debug image to {bbox_debug_path}')
                result = (result[0], result[1], None)
            except Exception as e:
                logger.error(f'Failed to save bbox debug image: {e}')
        return result

    @staticmethod
    def _remove_duplicate_textlines(result):
        """Non-maximum suppression: of text lines that overlap almost completely, keep the most probable."""
        try:
            from shapely.geometry import Polygon

            def calculate_iou(box_1, box_2):
                poly_1 = Polygon(box_1.pts)
                poly_2 = Polygon(box_2.pts)
                if not poly_1.is_valid or not poly_2.is_valid:
                    return 0.0
                intersection_area = poly_1.intersection(poly_2).area
                union_area = poly_1.union(poly_2).area
                if union_area == 0:
                    return 0.0
                return intersection_area / union_area

            textlines = result[0][:] # Work on a copy
            textlines.sort(key=lambda x: x.prob, reverse=True)

            kept_textlines = []
            while textlines:
                current_box = textlines.pop(0)
                kept_textlines.append(current_box)
                remaining_textlines = []
                for box in textlines:
                    iou = calculate_iou(current_box, box)
                    if iou < 0.9: # IoU threshold, 0.9 means very high overlap
                        remaining_textlines.append(box)
                textlines = remaining_textlines

            if len(result[0]) != len(kept_textlines):
                logger.info(f"Removed {len(result[0]) - len(kept_textlines)} duplicate lines via NMS.")
                result = (kept_textlines, result[1], result[2])

        except Exception as e:
            logger.error(f"An error occurred during Non-Maximum Suppression: {e}")
            pass
        return result

    @staticmethod
    def _split_off_other_textlines(ctx: Context, result):
        """Keep text lines labelled "other" out of OCR, but remember them on the context for merging."""
        all_textlines = result[0]
        forward_textlines = []
        other_textlines = []
        for txtln in all_textlines:
            det_label = getattr(txtln, 'det_label', None) or getattr(txtln, 'yolo_label', None)
            if isinstance(det_label, str) and det_label.strip().lower() == 'other':
                other_textlines.append(txtln)
            else:
                forward_textlines.append(txtln)
        ctx.model_assisted_other_textlines = other_textlines
        ctx.all_detected_textlines = all_textlines
        if other_textlines:
            logger.info(
                f"Detection split: total={len(all_textlines)}, "
                f"forward={len(forward_textlines)}, other_for_model_assisted_merge={len(other_textlines)}"
            )
        result = (forward_textlines, result[1], result[2])
        return result

    async def _run_detection(self, config: Config, ctx: Context):
        # ✅ Check the stop flag
        await asyncio.sleep(0)
        self._check_cancelled()
        self._prime_bubble_detection_cache(config, ctx)
        
        current_time = time.time()
        self._model_usage_timestamps[("detection", config.detector.detector)] = current_time
        import_yolo_labels = bool(getattr(config.detector, 'import_yolo_labels', False))
        ctx.used_imported_yolo_labels = False
        imported_textlines = []
        imported_mask_raw = None
        if import_yolo_labels:
            image_name = getattr(ctx, 'image_name', None)
            input_name = getattr(getattr(ctx, 'input', None), 'name', None)
            image_name_has_dir = bool(image_name and (os.path.isabs(image_name) or os.path.dirname(image_name)))
            input_name_has_dir = bool(input_name and (os.path.isabs(input_name) or os.path.dirname(input_name)))
            if input_name_has_dir and not image_name_has_dir:
                label_lookup_path = input_name
            else:
                label_lookup_path = image_name or input_name
            imported_textlines = load_imported_yolo_textlines(
                getattr(ctx, 'img_rgb', None),
                label_lookup_path,
                logger=logger,
            )
            if imported_textlines:
                imported_mask_raw = build_mask_from_textlines(ctx.img_rgb.shape, imported_textlines)

        should_use_imported_labels_only = (
            import_yolo_labels and
            bool(imported_textlines) and
            self.template and
            self.save_text
        )

        if should_use_imported_labels_only:
            logger.info(
                "Import YOLO labels enabled in template mode: skip detector boxes and use imported labels directly"
            )
            ctx.used_imported_yolo_labels = True
            result = (imported_textlines, imported_mask_raw, None)
        else:
            use_yolo_obb = config.detector.use_yolo_obb
            if self.load_text:
                use_yolo_obb = False
            result = await dispatch_detection(
                config.detector.detector,
                ctx.img_rgb,
                config.detector.detection_size,
                config.detector.text_threshold,
                config.detector.box_threshold,
                config.detector.unclip_ratio,
                self.device,
                self.verbose,
                use_yolo_obb,
                config.detector.yolo_obb_conf,
                config.detector.yolo_obb_overlap_threshold,
                config.detector.min_box_area_ratio,
                self._result_path,
                config.detector.det_rearrange_min_effective_short_side,
                use_sfx_filter=bool(getattr(config.detector, 'use_sfx_filter', False)),
                sfx_filter_include_bubble_text=bool(getattr(config.detector, 'sfx_filter_include_bubble_text', False)),
                bubble_mask=ctx.bubble_mask,
            )
        
            # Handle the bbox debug image (when the detector returned one)
            if self.verbose and result and len(result) == 3 and result[2] is not None:
                result = self._save_detection_debug_images(config, ctx, result)

            if import_yolo_labels and imported_textlines and not self.load_text:
                detector_box_count = len(result[0]) if result and result[0] else 0
                raw_mask = result[1] if result and len(result) > 1 else None
                if raw_mask is None:
                    raw_mask = imported_mask_raw
                ctx.used_imported_yolo_labels = True
                result = (imported_textlines, raw_mask, result[2] if result and len(result) > 2 else None)
                logger.info(
                    f"Import YOLO labels enabled: replace detector boxes with imported boxes "
                    f"(detector_boxes={detector_box_count}, imported_boxes={len(imported_textlines)})"
                )
        
        # --- BEGIN NON-MAXIMUM SUPPRESSION (NMS) FOR DE-DUPLICATION ---
        if result and result[0]:
            result = self._remove_duplicate_textlines(result)
        # --- END NON-MAXIMUM SUPPRESSION (NMS) ---

        # Split the detected boxes:
        # - the forward flow (OCR/translation) does not include "other"
        # - "other" is kept for the model-assisted merge stage of textline_merge
        if result and result[0]:
            result = self._split_off_other_textlines(ctx, result)

        return result

    def _should_prime_bubble_cache(self, config: Config) -> bool:
        render_cfg = getattr(config, 'render', None)
        ocr_cfg = getattr(config, 'ocr', None)
        detector_cfg = getattr(config, 'detector', None)
        inpainter_cfg = getattr(config, 'inpainter', None)
        return any(
            (
                getattr(render_cfg, 'layout_mode', None) == 'balloon_fill',
                bool(getattr(render_cfg, 'center_text_in_bubble', False)),
                bool(getattr(ocr_cfg, 'use_model_bubble_filter', False)),
                bool(getattr(ocr_cfg, 'use_model_bubble_repair_intersection', False)),
                bool(getattr(ocr_cfg, 'limit_mask_dilation_to_bubble_mask', False)),
                bool(getattr(inpainter_cfg, 'solid_fill_pure_bubbles', False)),
                (
                    bool(getattr(detector_cfg, 'use_yolo_obb', False))
                    and bool(getattr(detector_cfg, 'use_sfx_filter', False))
                    and not bool(getattr(detector_cfg, 'sfx_filter_include_bubble_text', False))
                    and not self.load_text
                ),
            )
        )

    def _prime_bubble_detection_cache(self, config: Config, ctx: Context) -> None:
        # Keep the mask beside img_rgb in the existing per-image context.
        if ctx.bubble_mask is not None:
            return
        image = getattr(ctx, 'img_rgb', None)
        if image is None or getattr(image, 'size', 0) == 0:
            return
        if not self._should_prime_bubble_cache(config):
            return
        try:
            result = detect_bubbles_with_mangalens(image, return_annotated=False, verbose=False)
            ctx.bubble_mask = build_bubble_mask_from_mangalens_result(result, image.shape[:2])
            detected = len(result.detections) if result is not None else 0
            logger.info(f"Bubble mask prepared in image context: detections={detected}")
        except Exception as exc:
            ctx.bubble_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            logger.warning(f"Bubble mask preparation failed: {exc}")

    def _save_labeled_textline_debug_image(self, img_rgb: np.ndarray, textlines: List, filename: str = 'bboxes_unfiltered_labeled.png'):
        """
        Draw a debug image of the labelled detection boxes on the original image (called in verbose mode only).
        """
        if img_rgb is None or textlines is None:
            return
        if len(textlines) == 0:
            return

        # BGR colour map (OpenCV)
        label_colors = {
            'balloon': (255, 255, 0),       # cyan
            'qipao': (0, 255, 0),           # green
            'other': (0, 255, 255),         # yellow
            'changfangtiao': (255, 0, 255), # magenta
            'fangkuai': (255, 128, 0),      # orange
            'kuangwai': (128, 0, 255),      # purple
            'hengxie': (255, 128, 0),       # orange
            'shuqing': (128, 0, 255),       # purple
            'unlabeled': (200, 200, 200),   # grey
        }

        try:
            canvas_bgr = cv2.cvtColor(np.copy(img_rgb), cv2.COLOR_RGB2BGR)
            label_stats = {}
            for idx, txtln in enumerate(textlines):
                pts = getattr(txtln, 'pts', None)
                if pts is None:
                    continue
                pts = np.asarray(pts, dtype=np.int32)
                if pts.size == 0:
                    continue

                label = getattr(txtln, 'det_label', None) or getattr(txtln, 'yolo_label', None) or 'unlabeled'
                label = str(label).strip().lower() if label is not None else 'unlabeled'
                if not label:
                    label = 'unlabeled'
                color = label_colors.get(label, (80, 80, 255))  # unknown label: red

                cv2.polylines(canvas_bgr, [pts], True, color=color, thickness=2)

                x = int(np.min(pts[:, 0]))
                y = int(np.min(pts[:, 1])) - 6
                if y < 12:
                    y = int(np.max(pts[:, 1])) + 14
                caption = f'{idx}:{label}'
                cv2.putText(canvas_bgr, caption, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas_bgr, caption, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                label_stats[label] = label_stats.get(label, 0) + 1

            out_path = self._result_path(filename)
            imwrite_unicode(out_path, canvas_bgr, logger)
            logger.info(f'Saved labeled textline debug image to {out_path}')
            logger.info(f'Textline label statistics: {label_stats}')
        except Exception as e:
            logger.error(f'Failed to save labeled textline debug image: {e}')

    async def _unload_model(self, tool: str, model: str, **kwargs):
        logger.info(f"Unloading {tool} model: {model}")
        match tool:
            case 'colorizer':
                await unload_colorization(model)
            case 'detection':
                await unload_detection(model)
            case 'inpainting':
                await unload_inpainting(model)
            case 'ocr':
                await unload_ocr(model)
            case 'upscaling':
                await unload_upscaling(model, **kwargs)
            case 'translation':
                await unload_translation(model)
            case 'textline_merge':
                # textline_merge needs no unloading (it has no model)
                logger.debug("textline_merge does not require unloading")
            case 'rendering':
                # rendering needs no unloading (it has no model)
                logger.debug("rendering does not require unloading")
            case _:
                logger.warning(f"Unknown tool type for unloading: {tool}")
        
        # Free Python memory (works for both CPU and GPU)
        import gc
        gc.collect()
        
        # After unloading, reclaim the CUDA cache and synchronise GPU work
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info(f"Model {tool}/{model} unloaded; memory released")

    # The cost of gc.collect() depends on the number of objects in the process (about a hundred milliseconds in a torch process), not on the memory to free;
    # large arrays are freed at once by reference counting, and Python's generational GC covers reference cycles,
    # so the explicit full-heap scan is rate-limited by time (covering the usual interval of interactive exports)
    _GC_MIN_INTERVAL_S = 60.0
    _last_gc_collect_ts = 0.0

    def _cleanup_gpu_memory(self, aggressive: bool = False):
        """Helper that frees GPU memory.

        Args:
            aggressive: whether to clean up aggressively (empty_cache/ipc_collect).
                        True is recommended at batch boundaries and after unloading models.
        """
        now = time.monotonic()
        if now - MangaTranslator._last_gc_collect_ts >= MangaTranslator._GC_MIN_INTERVAL_S:
            import gc
            gc.collect()
            MangaTranslator._last_gc_collect_ts = now

        # Clear GPU memory only on CUDA devices; MPS has no equivalent of empty_cache yet.
        device_str = str(getattr(self, 'device', ''))
        if device_str.startswith('cuda'):
            try:
                import torch
                if torch.cuda.is_available():
                    if aggressive:
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, 'ipc_collect'):
                            torch.cuda.ipc_collect()
                    torch.cuda.synchronize()
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._cleanup_gpu_memory")
                pass

    def _get_cuda_memory_snapshot(self) -> Optional[dict]:
        """Get a snapshot of the current CUDA memory. Returns None on devices other than CUDA."""
        device_str = str(getattr(self, 'device', ''))
        if not device_str.startswith('cuda'):
            return None
        try:
            if not torch.cuda.is_available():
                return None
            device = torch.device(device_str)
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            return {
                'device': device_index,
                'allocated_mb': torch.cuda.memory_allocated(device_index) / (1024 ** 2),
                'reserved_mb': torch.cuda.memory_reserved(device_index) / (1024 ** 2),
                'peak_allocated_mb': torch.cuda.max_memory_allocated(device_index) / (1024 ** 2),
                'peak_reserved_mb': torch.cuda.max_memory_reserved(device_index) / (1024 ** 2),
                'free_mb': free_bytes / (1024 ** 2),
                'total_mb': total_bytes / (1024 ** 2),
            }
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._get_cuda_memory_snapshot")
            return None

    def _log_cuda_memory_snapshot(self, stage: str, include_peak: bool = True):
        """Print a snapshot of CUDA memory use, to help locate memory growth between stages."""
        if not self.verbose:
            return
        snapshot = self._get_cuda_memory_snapshot()
        if snapshot is None:
            return
        peak_suffix = ""
        if include_peak:
            peak_suffix = (
                f", peak_allocated={snapshot['peak_allocated_mb']:.1f}MB"
                f", peak_reserved={snapshot['peak_reserved_mb']:.1f}MB"
            )
        logger.debug(
            f"[VRAM] {stage}: cuda:{snapshot['device']}, allocated={snapshot['allocated_mb']:.1f}MB, reserved={snapshot['reserved_mb']:.1f}MB{peak_suffix}, free={snapshot['free_mb']:.1f}MB, total={snapshot['total_mb']:.1f}MB"
        )
    
    def _clear_context_intermediate_fields(self, ctx, include_result=False):
        for field_name in self._CONTEXT_INTERMEDIATE_FIELDS:
            if hasattr(ctx, field_name) and getattr(ctx, field_name) is not None:
                delattr(ctx, field_name)
                setattr(ctx, field_name, None)

        if include_result and hasattr(ctx, 'result') and ctx.result is not None:
            del ctx.result
            ctx.result = None

    def _cleanup_context_memory(self, ctx, keep_result=True):
        """
        Clear the intermediate data of a single context (for the special modes)

        Args:
            ctx: the Context object
            keep_result: bool - whether ctx.result is kept
        """
        # Clear the input images
        if hasattr(ctx, 'input') and ctx.input is not None:
            if keep_result and hasattr(ctx, 'result') and ctx.input is ctx.result:
                pass
            else:
                if hasattr(ctx.input, 'close'):
                    try:
                        ctx.input.close()
                    except Exception as ignored_error:
                        note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._cleanup_context_memory")
                        pass
            del ctx.input
            ctx.input = None
        
        # Clear the intermediate images
        self._clear_context_intermediate_fields(ctx, include_result=not keep_result)
        
        # Force garbage collection and clear GPU memory
        self._cleanup_gpu_memory()
        logger.debug('[MEMORY] Context cleanup completed')

    @staticmethod
    def _detach_context_result(ctx):
        """Make ctx.result independent from input images and lazy file handles."""
        result = getattr(ctx, 'result', None)
        if result is None:
            return
        aliased = (
            result is getattr(ctx, 'input', None)
            or result is getattr(ctx, 'upscaled', None)
            or result is getattr(ctx, 'img_colorized', None)
        )
        # dump_image produces a new in-memory image; only an alias of the input or an image that still holds a file handle needs copying
        file_backed = isinstance(result, ImageFile.ImageFile) or getattr(result, 'fp', None) is not None
        if not aliased and not file_backed:
            return
        result.load()
        ctx.result = result.copy()

    @staticmethod
    def _align_preloaded_inpainted(inpainted, img_rgb):
        """Normalise the inpainted image of an in-memory payload to an RGB uint8 array of the same size as the working image."""
        if inpainted is None or img_rgb is None:
            return None
        arr = np.asarray(inpainted)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        elif arr.ndim != 3:
            logger.warning(f"[load_text] preloaded inpainted shape invalid: {arr.shape}, ignored")
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        target_h, target_w = img_rgb.shape[:2]
        if arr.shape[0] != target_h or arr.shape[1] != target_w:
            arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return np.ascontiguousarray(arr)

    
    def _cleanup_batch_memory(self, current_batch_images=None, preprocessed_contexts=None, translated_contexts=None, keep_results=True):
        """
        Shared memory clean-up for a batch

        Args:
            current_batch_images: List[(image, config)] - the original images of the current batch
            preprocessed_contexts: List[(ctx, config)] - the contexts after preprocessing
            translated_contexts: List[(ctx, config)] - the contexts after translation
            keep_results: bool - whether ctx.result is kept (for returning results)
        """
#         import gc
        
        # 1. Clear the original images
        if current_batch_images:
            for image, _ in current_batch_images:
                is_result = False
                if keep_results and translated_contexts:
                    for t_ctx, _ in translated_contexts:
                        if hasattr(t_ctx, 'result') and t_ctx.result is image:
                            is_result = True
                            break
                if not is_result:
                    if hasattr(image, 'close'):
                        try:
                            image.close()
                        except Exception as ignored_error:
                            note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._cleanup_batch_memory")
                            pass
            current_batch_images.clear()
        
        # 2. Clear the input images in the preprocessing contexts
        if preprocessed_contexts:
            for ctx, _ in preprocessed_contexts:
                if hasattr(ctx, 'input') and ctx.input is not None:
                    is_result = False
                    if keep_results and translated_contexts:
                        for t_ctx, _ in translated_contexts:
                            if hasattr(t_ctx, 'result') and t_ctx.result is ctx.input:
                                is_result = True
                                break
                    if not is_result:
                        # Close first, then delete
                        if hasattr(ctx.input, 'close'):
                            try:
                                ctx.input.close()
                            except Exception as ignored_error:
                                note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._cleanup_batch_memory")
                                pass
                    del ctx.input
                    ctx.input = None
            preprocessed_contexts.clear()
        
        # 3. Clear the intermediate images in the translation contexts
        if translated_contexts:
            for ctx, _ in translated_contexts:
                # Clear the intermediate images (deleted explicitly with del)
                self._clear_context_intermediate_fields(ctx, include_result=not keep_results)
            
            translated_contexts.clear()
        
        # 4. Force garbage collection and clear GPU memory
        # Reclaim GPU memory aggressively at batch boundaries, so it does not keep growing in a long batch.
        self._cleanup_gpu_memory(aggressive=True)
        
        # 5. Windows only: force the physical memory to be released
        try:
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            logger.debug('[MEMORY] Windows working set trimmed')
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._cleanup_batch_memory")
            pass  # Ignored on systems other than Windows
        
        logger.debug('[MEMORY] Batch cleanup completed')

    def _build_image_load_error_context(self, image_name: str, error: Exception, config: Config = None) -> Context:
        ctx = Context()
        ctx.image_name = image_name
        ctx.text_regions = []
        ctx.success = False
        ctx.translation_error = str(error)
        ctx.error = ctx.translation_error
        if config is not None:
            ctx.config = config
        return ctx

    def _format_pipeline_error_message(self, stage: str, error: Exception) -> str:
        stage_labels = {
            "preprocessing": "Preprocessing",
            "ocr": "OCR",
            "colorizing": "Colorization",
            "upscaling": "Upscaling",
            "detection": "Detection",
            "textline_merge": "Text line merging",
            "translation": "Translation",
            "mask-generation": "Mask generation",
            "inpainting": "Inpainting",
            "rendering": "Rendering",
            "saving": "Saving",
        }
        raw_message = str(error).strip() or repr(error)
        stage_label = stage_labels.get(stage, stage or "Processing")
        return f"{stage_label} failed: {raw_message}"

    def _resolve_pipeline_error(self, stage: str, error: Exception) -> tuple[str, Exception]:
        if isinstance(error, FileTranslationFailure):
            stage = error.stage or stage
            error = error.original_error
        return stage, error

    def _build_stage_error_context(
        self,
        image,
        error: Exception,
        config: Config = None,
        stage: str = "",
    ) -> Context:
        stage, error = self._resolve_pipeline_error(stage, error)
        image_name = getattr(image, "name", None) if image is not None else None
        ctx = Context()
        if image is not None:
            ctx.input = image
        if image_name:
            ctx.image_name = image_name
        ctx.text_regions = []
        ctx.success = False
        ctx.translation_error = self._format_pipeline_error_message(stage, error)
        ctx.error = ctx.translation_error
        if config is not None:
            ctx.config = config
        return ctx

    def _mark_context_failure(self, ctx: Context, error: Exception, stage: str = "") -> Context:
        stage, error = self._resolve_pipeline_error(stage, error)
        if ctx is None:
            ctx = Context()
        if ctx.text_regions is None:
            ctx.text_regions = []
        if not getattr(ctx, "image_name", None):
            input_image = getattr(ctx, "input", None)
            input_name = getattr(input_image, "name", None) if input_image is not None else None
            if input_name:
                ctx.image_name = input_name
        ctx.success = False
        ctx.translation_error = self._format_pipeline_error_message(stage, error)
        ctx.error = ctx.translation_error
        return ctx

    def _materialize_batch_inputs(self, batch_items: List[tuple]) -> tuple[list[tuple], list[Context]]:
        """
        Lazily load path-based items so batching stays under backend control.
        """
        loaded_items: list[tuple] = []
        load_errors: list[Context] = []

        for image_or_path, config in batch_items:
            if isinstance(image_or_path, str):
                image_path = image_or_path
                try:
                    with open(image_path, 'rb') as f:
                        image = open_pil_image(f, eager=True)
                    image.name = image_path
                    loaded_items.append((image, config))
                except Exception as exc:
                    logger.error(f"Failed to load image {image_path}: {exc}")
                    load_errors.append(self._build_image_load_error_context(image_path, exc, config))
                continue

            loaded_items.append((image_or_path, config))

        return loaded_items, load_errors

    # Background models cleanup job.
    async def _detector_cleanup_job(self):
        logger.info(f"Model cleanup job started with models_ttl={self.models_ttl} seconds")
        while True:
            if self.models_ttl == 0:
                await asyncio.sleep(1)
                continue
            now = time.time()
            for (tool, model), last_used in list(self._model_usage_timestamps.items()):
                time_since_last_use = now - last_used
                if time_since_last_use > self.models_ttl:
                    logger.info(f"Model {tool}/{model} has been idle for {time_since_last_use:.1f}s (TTL: {self.models_ttl}s), unloading...")
                    await self._unload_model(tool, model)
                    del self._model_usage_timestamps[(tool, model)]
            await asyncio.sleep(1)

    def _resolve_ocr_prob_threshold(self, config: Config) -> float:
        return config.ocr.prob if config.ocr.prob is not None else 0.1

    @staticmethod
    def _get_textline_text(textline) -> str:
        return str(getattr(textline, 'text', '') or '')

    @staticmethod
    def _get_textline_prob(textline) -> float:
        try:
            return float(getattr(textline, 'prob', 1.0))
        except (TypeError, ValueError):
            return 1.0

    def _textline_needs_secondary_ocr(self, textline, prob_threshold: float) -> bool:
        return (
            not self._get_textline_text(textline).strip()
            or self._get_textline_prob(textline) < prob_threshold
        )

    def _filter_ocr_textlines(self, config: Config, textlines, prob_threshold: float):
        filtered_textlines = []
        filter_list_count = 0
        low_confidence_count = 0

        for textline in textlines:
            text = self._get_textline_text(textline)
            if not text.strip():
                continue

            textline_prob = self._get_textline_prob(textline)
            if textline_prob < prob_threshold:
                low_confidence_count += 1
                logger.info(
                    f'OCR filtered low-confidence text line: prob={textline_prob:.4f} < threshold={prob_threshold:.4f}, text="{text}"'
                )
                continue

            if self.filter_text_enabled:
                match_result = match_filter(text)
                if match_result:
                    matched_word, match_type = match_result
                    match_label = {"精确": "exact", "包含": "substring"}.get(match_type, match_type)
                    filter_list_count += 1
                    logger.info(f'OCR filtered text line ({match_label} match): "{text}" -> matched: "{matched_word}"')
                    continue

            if config.render.font_color_fg:
                textline.fg_r, textline.fg_g, textline.fg_b = config.render.font_color_fg
            if config.render.font_color_bg:
                textline.bg_r, textline.bg_g, textline.bg_b = config.render.font_color_bg
            filtered_textlines.append(textline)

        if filter_list_count > 0:
            logger.info(f"OCR filter list: removed {filter_list_count} text lines")
        if low_confidence_count > 0:
            logger.info(f"OCR confidence filter: removed {low_confidence_count} text lines")

        return filtered_textlines

    async def _run_ocr(self, config: Config, ctx: Context):
        # ✅ Check the stop flag
        await asyncio.sleep(0)
        self._check_cancelled()
        
        current_time = time.time()
        self._model_usage_timestamps[("ocr", config.ocr.ocr)] = current_time
        
        # Create a subfolder for OCR (verbose mode only)
        if self.verbose:
            image_subfolder = self._get_image_subfolder()
            if image_subfolder:
                if self.result_sub_folder:
                    ocr_result_dir = os.path.join(BASE_PATH, 'result', self.result_sub_folder, image_subfolder, 'ocrs')
                else:
                    ocr_result_dir = os.path.join(BASE_PATH, 'result', image_subfolder, 'ocrs')
                os.makedirs(ocr_result_dir, exist_ok=True)
            else:
                ocr_result_dir = os.path.join(BASE_PATH, 'result', self.result_sub_folder, 'ocrs')
                os.makedirs(ocr_result_dir, exist_ok=True)
        else:
            # Outside verbose mode, use a temporary folder or do not create the OCR result folder
            ocr_result_dir = None
        
        # Set an environment variable temporarily for the OCR module
        old_ocr_dir = os.environ.get('MANGA_OCR_RESULT_DIR', None)
        if ocr_result_dir:
            os.environ['MANGA_OCR_RESULT_DIR'] = ocr_result_dir
        
        ocr_prob_threshold = self._resolve_ocr_prob_threshold(config)

        try:
            # --- Primary OCR run ---
            primary_ocr_engine = config.ocr.ocr
            ocr_name = primary_ocr_engine.value if hasattr(primary_ocr_engine, 'value') else primary_ocr_engine
            logger.info(f"Running primary OCR with: {ocr_name}")
            textlines = await dispatch_ocr(
                primary_ocr_engine,
                ctx.img_rgb,
                ctx.textlines,
                config.ocr,
                self.device,
                self.verbose,
                runtime_config=config,
                bubble_mask=ctx.bubble_mask,
            )

            # --- BEGIN: HYBRID OCR LOGIC ---
            if config.ocr.use_hybrid_ocr:
                # Identify textlines that failed recognition or have low confidence
                # Failure condition: the text is empty or the confidence is below the threshold
                failed_indices = [
                    i for i, tl in enumerate(textlines) 
                    if self._textline_needs_secondary_ocr(tl, ocr_prob_threshold)
                ]
                
                if failed_indices:
                    # Use textlines[i] instead of ctx.textlines[i] because OCR may have changed the order
                    failed_textlines = [textlines[i] for i in failed_indices]
                    logger.info(f"{len(failed_textlines)} textlines failed or have low confidence (< {ocr_prob_threshold}) with primary OCR. Trying secondary OCR...")
                    
                    secondary_ocr_engine = config.ocr.secondary_ocr
                    # We can reuse the same config object, just switching the engine
                    secondary_config = config.ocr
                    
                    secondary_ocr_name = secondary_ocr_engine.value if hasattr(secondary_ocr_engine, 'value') else secondary_ocr_engine
                    logger.info(f"Running secondary OCR with: {secondary_ocr_name}")
                    secondary_results = await dispatch_ocr(
                        secondary_ocr_engine,
                        ctx.img_rgb,
                        failed_textlines,
                        secondary_config,
                        self.device,
                        self.verbose,
                        runtime_config=config,
                        bubble_mask=ctx.bubble_mask,
                    )
                    
                    # Merge the results back into the original list
                    for i, result_tl in zip(failed_indices, secondary_results):
                        textlines[i] = result_tl # Replace the failed textline with the new result
                    
                    logger.info("Secondary OCR processing finished.")
                    
                    # ✅ Clean up after hybrid OCR, so the two OCR calls do not accumulate
#                     import gc
                    if 'secondary_results' in locals():
                        del secondary_results
                    if 'failed_textlines' in locals():
                        del failed_textlines
                    if 'failed_indices' in locals():
                        del failed_indices
                    self._cleanup_gpu_memory()
            # --- END: HYBRID OCR LOGIC ---

        finally:
            # Restore the environment variable
            if old_ocr_dir is not None:
                os.environ['MANGA_OCR_RESULT_DIR'] = old_ocr_dir
            elif 'MANGA_OCR_RESULT_DIR' in os.environ:
                del os.environ['MANGA_OCR_RESULT_DIR']

        return self._filter_ocr_textlines(config, textlines, ocr_prob_threshold)

    @staticmethod
    def _drop_textlines_in_skipped_languages(config: Config, ctx: Context) -> None:
        """Remove the text lines whose detected language is listed in translator.skip_lang."""
        skip_langs = [lang.strip().upper() for lang in config.translator.skip_lang.split(',')]
        filtered_textlines = []  
        for txtln in ctx.textlines:  
            try:  
                detected_lang, confidence = langid.classify(txtln.text)
                source_language = ISO_639_1_TO_VALID_LANGUAGES.get(detected_lang, 'UNKNOWN')
                if source_language != 'UNKNOWN':
                    source_language = source_language.upper()
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._run_textline_merge")
                source_language = 'UNKNOWN'  

            # Print detected source_language and whether it's in skip_langs  
            # logger.info(f'Detected source language: {source_language}, in skip_langs: {source_language in skip_langs}, text: "{txtln.text}"')  

            if source_language in skip_langs:  
                logger.info(f'Filtered out: {txtln.text}')  
                logger.info(f'Reason: Detected language {source_language} is in skip_langs')  
                continue  # Skip this region  
            filtered_textlines.append(txtln)  
        ctx.textlines = filtered_textlines  

    def _filter_small_single_box_regions(self, config: Config, ctx: Context, text_regions):
        """Drop merged regions that consist of one detected box smaller than detector.min_box_area_ratio."""
        img_h, img_w = ctx.img_rgb.shape[:2]
        img_total_pixels = img_h * img_w

        rearrange_plan = build_det_rearrange_plan(
            ctx.img_rgb,
            tgt_size=config.detector.detection_size,
            min_effective_short_side=config.detector.det_rearrange_min_effective_short_side,
        )
        require_rearrange = rearrange_plan is not None

        if require_rearrange:
            h = int(rearrange_plan['h'])
            w = int(rearrange_plan['w'])
            patch_size = int(rearrange_plan['patch_size'])
            asp_ratio = h / w

            # Limit the aspect ratio of the reference tile for area filtering (no more than 3:1).
            # Detection tiles may be taller to fill the long side of the model, but the filter threshold must not grow with them.
            max_patch_aspect_ratio = 3.0
            patch_aspect_ratio = patch_size / w
            if patch_aspect_ratio > max_patch_aspect_ratio:
                adjusted_ph = max_patch_aspect_ratio * w
                tile_pixels = adjusted_ph * w
                logger.info(f"Extreme image aspect ratio detected ({asp_ratio:.2f}); limiting area-filter reference aspect ratio: actual tile={patch_size}x{w} (ratio={patch_aspect_ratio:.2f}), reference tile={adjusted_ph:.0f}x{w} (ratio={max_patch_aspect_ratio:.2f}), area={tile_pixels:.0f} pixels")
            else:
                tile_pixels = patch_size * w
                logger.info(f"Extreme image aspect ratio detected ({asp_ratio:.2f}); using tile area ({patch_size}x{w}={tile_pixels} pixels) for filtering")
        else:
            tile_pixels = img_total_pixels  # No tiling: use the whole image

        before_filter_count = len(text_regions)
        filtered_out_regions = []
        filtered_in_regions = []

        for region in text_regions:
            # Count the detected boxes that were merged
            num_textlines = len(region.lines)

            # With several detected boxes it is a real text region, so keep it
            if num_textlines > 1:
                filtered_in_regions.append(region)
                continue

            # Only regions with a single detected box are filtered by area
            region_area = region.real_area
            # Use the tile area (when tiled) or the whole image area
            area_ratio = region_area / tile_pixels

            if region_area <= 16 or area_ratio <= config.detector.min_box_area_ratio:
                filtered_out_regions.append((region, area_ratio, num_textlines, require_rearrange))
            else:
                filtered_in_regions.append(region)

        text_regions = filtered_in_regions
        after_filter_count = len(text_regions)

        if filtered_out_regions:
            reference_desc = f'tile({patch_size}x{w})' if require_rearrange else f'full image({img_w}x{img_h})'
            filter_ratio = len(filtered_out_regions) / before_filter_count * 100 if before_filter_count > 0 else 0
            # Info level: show the summary only
            logger.info(f"Post-merge area filter: reference={reference_desc}, minimum area ratio={config.detector.min_box_area_ratio:.4f} ({config.detector.min_box_area_ratio*100:.2f}%), before={before_filter_count}, after={after_filter_count}, removed={len(filtered_out_regions)} ({filter_ratio:.1f}%, single-box regions only)")
            # Verbose mode: show the details
            if self.verbose:
                for idx, (region, ratio, num_lines, was_rearranged) in enumerate(filtered_out_regions):
                    # Width and height of the box
                    x1, y1, x2, y2 = region.xyxy
                    width = x2 - x1
                    height = y2 - y1
                    logger.debug(f'  Removing single-box region [{idx+1}]: size={width:.0f}x{height:.0f}, area={region.real_area:.1f} pixels, ratio={ratio*100:.3f}%, text="{region.text[:20]}"')
        return text_regions

    @staticmethod
    def _repair_unpaired_brackets(stripped_text: str) -> str:
        """Remove brackets that have no partner and correct closing brackets of the wrong kind."""
        bracket_pairs = {  
            '(': ')', '（': '）', '[': ']', '【': '】', '{': '}', '〔': '〕', '〈': '〉', '「': '」',  
            '"': '"', '＂': '＂', "'": "'", "“": "”", '《': '》', '『': '』', '〝': '〞', '﹁': '﹂', '﹃': '﹄',  
            '⸂': '⸃', '⸄': '⸅', '⸉': '⸊', '⸌': '⸍', '⸜': '⸝', '⸠': '⸡', '‹': '›', '«': '»', '＜': '＞', '<': '>'  
        }   
        left_symbols = set(bracket_pairs.keys())  
        right_symbols = set(bracket_pairs.values())  

        has_brackets = any(s in stripped_text for s in left_symbols) or any(s in stripped_text for s in right_symbols)  

        if has_brackets:  
            result_chars = []  
            stack = []  
            to_skip = []    

            # First pass: mark the matching brackets
            # First traversal: mark matching brackets
            for i, char in enumerate(stripped_text):  
                if char in left_symbols:  
                    stack.append((i, char))  
                elif char in right_symbols:  
                    if stack:  
                        # There is a matching left bracket: pop it
                        # There is a corresponding left bracket, pop the stack
                        stack.pop()  
                    else:  
                        # No matching left bracket: mark for removal
                        # No corresponding left parenthesis, marked for deletion
                        to_skip.append(i)  

            # Mark unmatched left brackets for removal
            # Mark unmatched left brackets as delete  
            for pos, _ in stack:  
                to_skip.append(pos)  

            has_removed_symbols = len(to_skip) > 0  

            # Second pass: handle brackets that pair up but are of different kinds
            # Second pass: Process matching but mismatched brackets
            stack = []  
            for i, char in enumerate(stripped_text):  
                if i in to_skip:  
                    # Skip the lone brackets
                    # Skip isolated parentheses
                    continue  

                if char in left_symbols:  
                    stack.append(char)  
                    result_chars.append(char)  
                elif char in right_symbols:  
                    if stack:  
                        left_bracket = stack.pop()  
                        expected_right = bracket_pairs.get(left_bracket)  

                        if char != expected_right:  
                            # Replace a right bracket of the wrong kind with the one that matches its left bracket
                            # Replace mismatched right brackets with the correct right brackets corresponding to the left brackets
                            result_chars.append(expected_right)  
                            logger.info(f'Fixed mismatched bracket: replaced "{char}" with "{expected_right}"')  
                        else:  
                            result_chars.append(char)  
                else:  
                    result_chars.append(char)  

            new_stripped_text = ''.join(result_chars)  

            if has_removed_symbols:  
                logger.info(f'Removed unpaired bracket from "{stripped_text}"')  

            if new_stripped_text != stripped_text and not has_removed_symbols:  
                logger.info(f'Fixed brackets: "{stripped_text}" → "{new_stripped_text}"')  

            stripped_text = new_stripped_text  
        return stripped_text

    async def _run_textline_merge(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("textline_merge", "textline_merge")] = current_time
        # Filter out languages to skip  
        if config.translator.skip_lang is not None:  
            self._drop_textlines_in_skipped_languages(config, ctx)
    
        merge_input_textlines = list(ctx.textlines)
        enable_model_assisted_merge = bool(getattr(config.ocr, 'merge_special_require_full_wrap', True))
        model_assisted_other_textlines = []
        if enable_model_assisted_merge:
            model_assisted_other_textlines = getattr(ctx, 'model_assisted_other_textlines', None) or []
            if model_assisted_other_textlines:
                logger.info(
                    f"Model-assisted merge uses auxiliary 'other' boxes only: "
                    f"ocr_textlines={len(merge_input_textlines)}, "
                    f"other_aux={len(model_assisted_other_textlines)}"
                )

        text_regions = await dispatch_textline_merge(
            merge_input_textlines,
            ctx.img_rgb.shape[1],
            ctx.img_rgb.shape[0],
            config,
            verbose=self.verbose,
            model_assisted_other_textlines=(
                model_assisted_other_textlines if enable_model_assisted_merge else None
            )
        )
        for region in text_regions:
            if not hasattr(region, "text_raw"):
                region.text_raw = region.text      # Save initial OCR results for downstream processing.

        # Apply the area filter after merging (based on the merged box)
        # Only small regions with a single detected box are filtered; merged regions with several boxes are kept
        if config.detector.min_box_area_ratio > 0:
            text_regions = self._filter_small_single_box_regions(config, ctx, text_regions)

        keep_lang = str(getattr(config.translator, 'keep_lang', 'none') or 'none').strip().upper()
        keep_lang_enabled = keep_lang not in _KEEP_LANG_NONE_VALUES
        keep_lang_filtered_count = 0

        new_text_regions = []
        for region in text_regions:
            # Skip regions whose text is None
            if region.text is None:
                logger.warning("Skipping region with text=None")
                continue
                
            # Remove leading spaces after pre-translation dictionary replacement                
            original_text = region.text  
            stripped_text = original_text.strip()  
            
            # Record removed leading characters  
            removed_start_chars = original_text[:len(original_text) - len(stripped_text)]  
            if removed_start_chars:  
                logger.info(f'Removed leading characters: "{removed_start_chars}" from "{original_text}"')  
            
            # Modified filtering condition: handle incomplete parentheses  
            stripped_text = self._repair_unpaired_brackets(stripped_text)
              
            region.text = stripped_text.strip()

            if keep_lang_enabled and region.text:
                detected_keep_lang = _detect_region_keep_language(region.text)
                if not _keep_language_matches(detected_keep_lang, keep_lang):
                    keep_lang_filtered_count += 1
                    logger.info(f'Filtered out: {region.text}')
                    logger.info(
                        f'Reason: Detected source language {detected_keep_lang} '
                        f'does not match keep_lang={keep_lang}.'
                    )
                    continue

            # Filter out empty, too short and worthless text
            if not region.text \
                    or len(region.text) < config.ocr.min_text_length \
                    or not is_valuable_text(region.text) \
                    or (not config.translator.no_text_lang_skip and langcodes.tag_distance(region.source_lang, config.translator.target_lang) == 0):
                if region.text and region.text.strip():
                    logger.info(f'Filtered out: {region.text}')
                    if len(region.text) < config.ocr.min_text_length:
                        logger.info('Reason: Text length is less than the minimum required length.')
                    elif not is_valuable_text(region.text):
                        logger.info('Reason: Text is not considered valuable.')
                    elif langcodes.tag_distance(region.source_lang, config.translator.target_lang) == 0:
                        logger.info('Reason: Text language matches the target language and no_text_lang_skip is False.')
                elif not region.text:
                    logger.info('Filtered out: Empty text region')
            else:
                if config.render.font_color_fg or config.render.font_color_bg:
                    if config.render.font_color_bg:
                        region.adjust_bg_color = False
                new_text_regions.append(region)
        if keep_lang_enabled and keep_lang_filtered_count > 0:
            logger.info(
                f"Post-merge language filter: keep_lang={keep_lang}, removed {keep_lang_filtered_count} text regions"
            )
        text_regions = new_text_regions
        text_regions = sort_regions(
            text_regions,
            right_to_left=config.render.rtl,
            img=ctx.img_rgb,
            force_simple_sort=config.force_simple_sort
        )   
        
        
        
        return text_regions

    def _prune_context_history(self):
        """
        Prune translation history to prevent memory leaks in large batch tasks.
        Keeps only the most recent pages needed for context.
        """
        # Minimum history to keep (context_size + buffer)
        # If context_size is 0, keep a small buffer (e.g., 5) just in case
        keep_size = max(self.context_size, 1) + 5
        
        if len(self.all_page_translations) > keep_size:
            # Remove oldest entries
            trim_count = len(self.all_page_translations) - keep_size
            self.all_page_translations = self.all_page_translations[trim_count:]
            if len(self._original_page_texts) >= trim_count:
                self._original_page_texts = self._original_page_texts[trim_count:]
            # Also clean up saved image contexts if they are too old (simple heuristic)
            if len(self._saved_image_contexts) > keep_size * 2:
                # Keep only the last N keys
                keys = list(self._saved_image_contexts.keys())
                keys_to_remove = keys[:-keep_size*2]
                for k in keys_to_remove:
                    del self._saved_image_contexts[k]

    @staticmethod
    def _get_context_region_count(region: Any) -> int:
        lines = getattr(region, 'lines', None)
        if lines is None:
            return 1

        try:
            return max(int(len(lines)), 1)
        except TypeError:
            return 1

    def _build_page_context_entries(self, ctx: Context) -> List[dict]:
        entries = []
        if not getattr(ctx, 'text_regions', None):
            return entries

        for region in ctx.text_regions:
            original_text = getattr(region, 'text', None)
            translated_text = getattr(region, 'translation', None)
            if original_text is None or not translated_text:
                continue

            entries.append({
                "text": original_text,
                "translation": translated_text,
                "original_region_count": self._get_context_region_count(region),
            })

        return entries

    def _normalize_context_page_entries(self, page: Any) -> List[dict]:
        """
        A history context page may have either of two structures:
        1. Old structure: {original_text: translation}
        2. New structure: [{"text": ..., "translation": ..., "original_region_count": ...}, ...]
        """
        normalized_entries: List[dict] = []

        if isinstance(page, dict):
            for original_text, translated_text in page.items():
                normalized_entries.append({
                    "text": original_text,
                    "translation": translated_text,
                })
            return normalized_entries

        if not isinstance(page, list):
            return normalized_entries

        for entry in page:
            if not isinstance(entry, dict):
                continue

            original_text = entry.get("text")
            translated_text = entry.get("translation")
            if original_text is None or translated_text is None:
                continue

            normalized_entry = {
                "text": original_text,
                "translation": translated_text,
            }

            region_count = entry.get("original_region_count")
            if isinstance(region_count, (int, float)):
                normalized_entry["original_region_count"] = max(int(region_count), 1)

            normalized_entries.append(normalized_entry)

        return normalized_entries

    def _page_has_context_entries(self, page: Any) -> bool:
        for entry in self._normalize_context_page_entries(page):
            original_text = str(entry.get("text") or "").strip()
            translated_text = str(entry.get("translation") or "").strip()
            if original_text and translated_text:
                return True
        return False

    def _build_prev_context(self, use_original_text=False, current_page_index=None, batch_index=None, batch_original_texts=None):
        """
        Skip pages with no sentences, take the most recent context_size non-empty pages and build a multi-turn history:
        - user: the text request sent to the AI earlier (without images)
        - assistant: the single-line JSON result the AI returned earlier

        Returns a JSON array string; an empty string when there is no non-empty page.

        Args:
            use_original_text: whether the original text is used as context instead of the translation (currently unused)
            current_page_index: index of the current page, used to decide the context range
            batch_index: index of the current page in the batch (currently unused)
            batch_original_texts: original text data of the current batch (currently unused)
        """
        if self.context_size <= 0:
            return ""

        # Use the pages before the given page index as context
        if current_page_index is not None:
            available_pages = self.all_page_translations[:current_page_index] if self.all_page_translations else []
        else:
            # Use all completed pages
            available_pages = self.all_page_translations or []

        if not available_pages:
            return ""

        # Keep the pages that have sentences
        non_empty_pages = [
            page for page in available_pages
            if self._page_has_context_entries(page)
        ]
        # The number of pages actually used
        pages_used = min(self.context_size, len(non_empty_pages))
        if pages_used == 0:
            return ""
        tail = non_empty_pages[-pages_used:]

        # Build the history as user / assistant message turns
        history_turns = []
        for page in tail:
            page_entries = []
            for entry in self._normalize_context_page_entries(page):
                original_text = entry.get("text")
                translated_text = entry.get("translation")
                original_clean = (original_text or "").replace('\n', ' ').replace('\ufffd', '').strip()
                translated_clean = (translated_text or "").strip()
                if original_clean and translated_clean:
                    page_entry = {
                        "text": original_clean,
                        "translation": translated_clean,
                    }
                    region_count = entry.get("original_region_count")
                    if isinstance(region_count, int) and region_count > 0:
                        page_entry["original_region_count"] = region_count
                    page_entries.append(page_entry)

            if not page_entries:
                continue

            input_data = [
                {
                    "id": index + 1,
                    "text": entry["text"],
                    **({"original_region_count": entry["original_region_count"]} if "original_region_count" in entry else {}),
                }
                for index, entry in enumerate(page_entries)
            ]
            output_data = {
                "translations": [
                    {"id": index + 1, "translation": entry["translation"]}
                    for index, entry in enumerate(page_entries)
                ]
            }
            user_prompt = (
                "Please translate the following manga text regions:\n\n"
                "All texts to translate (JSON Array):\n"
                + json.dumps(input_data, ensure_ascii=False, separators=(',', ':'))
                + "\n\nCRITICAL: Provide translations in the exact same order as the input array. "
                + "Follow the OUTPUT FORMAT specified in the System Prompt."
            )
            assistant_response = json.dumps(output_data, ensure_ascii=False, separators=(',', ':'))
            history_turns.append({
                "user": user_prompt,
                "assistant": assistant_response,
            })

        if not history_turns:
            return ""

        return json.dumps(history_turns, ensure_ascii=False, separators=(',', ':'))

    async def _load_and_prepare_prompts(self, config: Config, ctx: Context):
        """Loads custom HQ and line break prompts into the context object."""
        from .translators.prompt_loader import (
            load_custom_prompt,
            load_line_break_prompt,
        )
        
        # Load custom high-quality prompt from file if specified (supports .yaml/.json)
        ctx.custom_prompt_json = None
        if config.translator.high_quality_prompt_path:
            try:
                prompt_path = normalize_server_resource_path(config.translator.high_quality_prompt_path)
                if not os.path.isabs(prompt_path):
                    prompt_path = os.path.join(BASE_PATH, prompt_path)
                
                ctx.custom_prompt_json = load_custom_prompt(prompt_path)
                if ctx.custom_prompt_json:
                    logger.info(f"Successfully loaded custom HQ prompt from: {prompt_path}")
                    # Log the parsed content for user verification
                    from .translators.common import _flatten_prompt_data
                    _flatten_prompt_data(ctx.custom_prompt_json)
                    from .rendering.auto_linebreak import set_thai_protected_words
                    from .translators.manga_context import glossary_entries
                    set_thai_protected_words(
                        entry.get('translation') for entry in glossary_entries(ctx.custom_prompt_json)
                        if isinstance(entry, dict)
                    )
                    # logger.info(f"--- Parsed Custom Prompt Content ---\n{parsed_content}\n------------------------------------")
                else:
                    logger.warning(f"Custom HQ prompt file not found or invalid: {prompt_path}")
            except Exception as e:
                logger.error(f"Error loading custom HQ prompt: {e}")

        # Load AI line break prompt if enabled (supports .yaml/.json)
        ctx.line_break_prompt_json = None
        if config.render.disable_auto_wrap: # This is the "AI line breaking" switch
            try:
                dict_dir = os.path.join(BASE_PATH, 'dict')
                ctx.line_break_prompt_json = load_line_break_prompt(dict_dir)
                if ctx.line_break_prompt_json:
                    logger.info("AI line breaking is enabled. Loaded line break prompt.")
                else:
                    logger.warning("AI line breaking is enabled, but line break prompt file not found.")
            except Exception as e:
                logger.error(f"Failed to load line break prompt: {e}")
        return ctx

    async def _run_mask_refinement(self, config: Config, ctx: Context):
        # ✅ Check the stop flag
        await asyncio.sleep(0)
        self._check_cancelled()
        
        return await dispatch_mask_refinement(
            ctx.text_regions,
            ctx.img_rgb,
            ctx.mask_raw,
            method='fit_text',
            dilation_offset=config.mask_dilation_offset,
            verbose=self.verbose,
            kernel_size=self.kernel_size,
            use_model_bubble_repair_intersection=bool(getattr(config.ocr, 'use_model_bubble_repair_intersection', False)),
            limit_mask_dilation_to_bubble_mask=bool(getattr(config.ocr, 'limit_mask_dilation_to_bubble_mask', False)),
            debug_path_fn=self._result_path if self.verbose else None,
            bubble_mask=ctx.bubble_mask,
        )

    async def _run_inpainting(self, config: Config, ctx: Context):
        # ✅ Check the stop flag
        await asyncio.sleep(0)
        self._check_cancelled()

        img_shape = tuple(ctx.img_rgb.shape[:2]) if getattr(ctx, 'img_rgb', None) is not None else None
        mask_shape = tuple(ctx.mask.shape[:2]) if getattr(ctx, 'mask', None) is not None else None
        logger.info(
            f"[Inpainting] inpainter={config.inpainter.inpainter}, precision={getattr(config.inpainter, 'inpainting_precision', 'n/a')}, inpainting_size={config.inpainter.inpainting_size}, image_shape={img_shape}, mask_shape={mask_shape}"
        )
        snapshot = self._get_cuda_memory_snapshot()
        if snapshot is not None:
            try:
                torch.cuda.reset_peak_memory_stats(snapshot['device'])
            except Exception as ignored_error:
                note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._run_inpainting")
                pass
            self._log_cuda_memory_snapshot("inpainting/before_dispatch", include_peak=False)
        
        # BT-style inpainting, two independent switches:
        # solid_fill_pure_bubbles - fill plain-colour bubbles with the background colour and skip the model
        # per_block_inpainting - inpaint each isolated block of the refined mask in its own crop; after squaring, the long-image tiling flow is not entered
        img_for_inpaint = ctx.img_rgb
        mask_for_inpaint = ctx.mask
        solid_fill = getattr(config.inpainter, 'solid_fill_pure_bubbles', False)
        per_block = getattr(config.inpainter, 'per_block_inpainting', False)
        text_regions = getattr(ctx, 'text_regions', None) or []
        if (solid_fill and text_regions) or per_block:
            try:
                filled_img = ctx.img_rgb
                remaining_mask = ctx.mask
                if solid_fill and text_regions:
                    # The inpainting mask decides the area that can be filled; the dilated raw mask is only used to cut the text out of the bubble and to sample the background colour.
                    mask_tight = getattr(ctx, 'mask_raw', None)
                    if mask_tight is not None:
                        if mask_tight.shape[:2] != ctx.mask.shape[:2]:
                            mask_tight = cv2.resize(mask_tight, ctx.mask.shape[:2][::-1],
                                                    interpolation=cv2.INTER_LINEAR)
                        # Equivalent of BT REFINEMASK_INPAINT: grow the stroke mask by 2px
                        # to cover the anti-aliased pixels at the text edges; otherwise the grey fringe ruins the background purity sample
                        mask_tight = cv2.dilate(
                            np.where(mask_tight >= 127, 255, 0).astype(np.uint8), None, iterations=2)
                        try:
                            self._prime_bubble_detection_cache(config, ctx)
                            bubble_mask = erode_bubble_mask(
                                ctx.bubble_mask,
                                erode_ratio=MODEL_BUBBLE_SHRINK_RATIO,
                            )
                        except Exception as bubble_exc:
                            logger.warning(f"[Inpainting] Bubble detection failed; skipping solid-color filling: {bubble_exc}")
                            bubble_mask = np.zeros(ctx.img_rgb.shape[:2], dtype=np.uint8)
                        filled_img, remaining_mask, filled_count = solid_fill_pure_bubbles(
                            ctx.img_rgb, ctx.mask, text_regions, mask_tight, bubble_mask,
                            config.ocr.model_bubble_overlap_threshold)
                        logger.info(
                            f"[Inpainting] Filled solid-color bubbles directly: {filled_count}/{len(text_regions)} text regions skipped the inpainting model")

                if per_block:
                    if remaining_mask is ctx.mask:
                        remaining_mask = ctx.mask.copy()

                    async def _inpaint_block(crop, msk):
                        self._check_cancelled()
                        return await dispatch_inpainting(
                            config.inpainter.inpainter, crop, msk, config.inpainter,
                            config.inpainter.inpainting_size, self.device, self.verbose)

                    result, block_count = await inpaint_regions_per_block(
                        filled_img, remaining_mask, _inpaint_block)
                    logger.info(f"[Inpainting] Per-block inpainting completed: {block_count} isolated masks")
                    return result

                img_for_inpaint, mask_for_inpaint = filled_img, remaining_mask
                if not np.any(mask_for_inpaint):
                    logger.info("[Inpainting] Remaining mask is empty; skipping inpainting model")
                    return img_for_inpaint
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"BT-style inpainting failed; falling back to full-page inpainting: {e}")
                img_for_inpaint, mask_for_inpaint = ctx.img_rgb, ctx.mask

        current_time = time.time()
        self._model_usage_timestamps[("inpainting", config.inpainter.inpainter)] = current_time
        try:
            result = await dispatch_inpainting(
                config.inpainter.inpainter,
                img_for_inpaint,
                mask_for_inpaint,
                config.inpainter,
                config.inpainter.inpainting_size,
                self.device,
                self.verbose,
            )
            return result
        finally:
            self._log_cuda_memory_snapshot("inpainting/after_dispatch")

    def _should_skip_inpainting_for_ai_renderer(self, config: Config) -> bool:
        return config.render.renderer in (
            Renderer.openai_renderer,
            Renderer.gemini_renderer,
        )

    async def _run_text_rendering(
        self,
        config: Config,
        ctx: Context,
        skip_font_scaling: bool = False,
        skip_text_replacements: bool = False,
    ):
        # ✅ Check the stop flag
        await asyncio.sleep(0)
        self._check_cancelled()

        current_time = time.time()
        self._model_usage_timestamps[("rendering", config.render.renderer)] = current_time

        fallback_font_family = config.render.font_family or self.font_family or ''
        if ctx.text_regions:
            for region in ctx.text_regions:
                if not getattr(region, 'font_family', ''):
                    region.font_family = fallback_font_family

        render_base_img = (
            ctx.img_inpainted
            if getattr(ctx, 'editor_export', False)
            else (
                ctx.img_rgb
                if self._should_skip_inpainting_for_ai_renderer(config)
                else ctx.img_inpainted
            )
        )
        if render_base_img is None:
            raise RuntimeError("Rendering requires a final inpainted image")

        ctx.img_render_alpha = None
        if config.render.renderer == Renderer.none:
            output = render_base_img
        else:
            # Request debug image for balloon_fill mode when verbose
            need_debug_img = self.verbose and config.render.layout_mode == 'balloon_fill'
            render_alpha = None
            if getattr(ctx, 'img_alpha', None) is not None and render_base_img is not None:
                render_alpha = np.zeros(render_base_img.shape[:2], dtype=np.uint8)
            result = await dispatch_rendering(
                render_base_img,
                ctx.text_regions,
                config,
                ctx.img_rgb,
                return_debug_img=need_debug_img,
                skip_font_scaling=skip_font_scaling,
                skip_text_replacements=skip_text_replacements or bool(getattr(ctx, 'skip_text_replacements', False)),
                render_alpha=render_alpha,
                bubble_mask=ctx.bubble_mask,
            )
            
            # Handle debug image if returned
            if need_debug_img and isinstance(result, tuple):
                output, debug_img = result
                # Save balloon_fill debug image
                if debug_img is not None:
                    try:
                        debug_path = self._result_path('balloon_fill_boxes.png')
                        # debug_img is already BGR after the rendering stage, so it is saved as it is
                        imwrite_unicode(debug_path, debug_img, logger)
                        logger.info(f"📸 Balloon fill debug image saved: {debug_path}")
                    except Exception as e:
                        logger.error(f"Failed to save balloon_fill debug image: {e}")
                semantic_records = getattr(config, '_chinese_linebreak_debug_records', None)
                if isinstance(semantic_records, list) and semantic_records:
                    try:
                        semantic_debug_path = self._result_path('chinese_linebreak_debug.json')
                        semantic_debug_data = {
                            "version": 1,
                            "type": "chinese_linebreak_debug",
                            "records": semantic_records,
                        }
                        with open(semantic_debug_path, 'w', encoding='utf-8') as f:
                            json.dump(semantic_debug_data, f, ensure_ascii=False, indent=2)
                        logger.info(f"Chinese linebreak debug JSON saved: {semantic_debug_path}")
                    except Exception as e:
                        logger.error(f"Failed to save Chinese linebreak debug JSON: {e}")
            else:
                output = result

            if render_alpha is not None and np.any(render_alpha):
                ctx.img_render_alpha = render_alpha
        
        # ✅ Clear the image data that is no longer needed right after rendering
        if hasattr(ctx, 'img_rgb') and ctx.img_rgb is not None:
            del ctx.img_rgb
            ctx.img_rgb = None
        ctx.bubble_mask = None
        if hasattr(ctx, 'img_inpainted') and ctx.img_inpainted is not None:
            del ctx.img_inpainted
            ctx.img_inpainted = None
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        return output

    def _create_confidence_heatmap(self, mask: np.ndarray, vmin: float = 0.0, vmax: float = 1.0, equalize: bool = True) -> np.ndarray:
        """
        Turn a greyscale mask into a confidence heat map with a colour bar

        Args:
            mask: greyscale mask array (0-255)
            vmin: minimum of the colour map (0-1); lower values are shown in the lowest colour
            vmax: maximum of the colour map (0-1); higher values are shown in the highest colour
            equalize: whether histogram equalisation is applied to raise the contrast

        Returns:
            A BGR image with a colour bar
        """
        # Convert a multi-channel image to a single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Histogram equalisation to raise the contrast
        if equalize:
            mask = cv2.equalizeHist(mask)
        
        # Normalise the mask to the range 0-1
        if mask.max() > 0:
            mask_normalized = mask.astype(np.float32) / 255.0
        else:
            mask_normalized = mask.astype(np.float32)
        
        # Rescale: map [vmin, vmax] to [0, 1]
        if vmin != 0.0 or vmax != 1.0:
            mask_normalized = np.clip((mask_normalized - vmin) / (vmax - vmin), 0, 1)
        
        # Apply the colour map (the jet colormap)
        colormap = matplotlib.colormaps['jet']
        colored_mask = colormap(mask_normalized)
        
        # Convert to BGR (matplotlib returns RGBA)
        colored_mask_bgr = (colored_mask[:, :, :3] * 255).astype(np.uint8)
        colored_mask_bgr = cv2.cvtColor(colored_mask_bgr, cv2.COLOR_RGB2BGR)
        
        # Build the image with a colour bar
        h, w = mask.shape
        # Create the colour bar (10% of the image width, at least 50 pixels)
        colorbar_width = max(50, int(w * 0.1))
        colorbar_height = h
        
        # Generate the colour bar
        colorbar = np.linspace(1, 0, colorbar_height).reshape(-1, 1)
        colorbar = np.tile(colorbar, (1, colorbar_width))
        colored_colorbar = colormap(colorbar)
        colored_colorbar_bgr = (colored_colorbar[:, :, :3] * 255).astype(np.uint8)
        colored_colorbar_bgr = cv2.cvtColor(colored_colorbar_bgr, cv2.COLOR_RGB2BGR)
        
        # Create the colour bar with text labels
        # Add a white border and a background for the text
        colorbar_with_labels = np.ones((colorbar_height, colorbar_width + 100, 3), dtype=np.uint8) * 255
        colorbar_with_labels[:, :colorbar_width] = colored_colorbar_bgr
        
        # Add the ticks and the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        num_ticks = 11  # Show 11 ticks
        
        for i in range(num_ticks):
            # Map to the actual value range [vmax, vmin]
            normalized_value = 1.0 - i / (num_ticks - 1)  # From 1.0 down to 0.0
            actual_value = vmin + normalized_value * (vmax - vmin)  # Map to [vmin, vmax]
            y_pos = int(i * (colorbar_height - 1) / (num_ticks - 1))
            
            # Draw the tick line
            cv2.line(colorbar_with_labels, 
                    (colorbar_width, y_pos), 
                    (colorbar_width + 10, y_pos), 
                    (0, 0, 0), 1)
            
            # Draw the text
            text = f'{actual_value:.2f}'
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_y = y_pos + text_size[1] // 2
            cv2.putText(colorbar_with_labels, text, 
                       (colorbar_width + 15, text_y), 
                       font, font_scale, (0, 0, 0), font_thickness)
        
        # Add the title
        title = 'Confidence'
        title_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        title_x = colorbar_width + (100 - title_size[0]) // 2
        cv2.putText(colorbar_with_labels, title, 
                   (title_x, 20), 
                   font, font_scale, (0, 0, 0), font_thickness)
        
        # Join the image and the colour bar
        result_image = np.hstack([colored_mask_bgr, colorbar_with_labels])
        
        return result_image

    def _result_path(self, path: str) -> str:
        """
        Returns path to result folder where intermediate images are saved when using verbose flag
        or web mode input/result images are cached.
        """
        # Use a per-image subfolder only in verbose mode
        if self.verbose:
            image_subfolder = self._get_image_subfolder()
            if image_subfolder:
                if self.result_sub_folder:
                    result_path = os.path.join(BASE_PATH, 'result', self.result_sub_folder, image_subfolder, path)
                else:
                    result_path = os.path.join(BASE_PATH, 'result', image_subfolder, path)
                # Make sure the folder exists
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                return result_path
        
        # In server/web mode (result_sub_folder is empty) and outside verbose mode
        # a subfolder has to be created to save final.png
        if not self.result_sub_folder:
            # When no subfolder is specified (like in desktop-ui mode),
            # the 'path' parameter is expected to be an absolute path to the output file.
            # Therefore, we don't join it with BASE_PATH.
            base_dir = os.path.join(BASE_PATH, 'result')
            result_path = os.path.join(base_dir, path)
        else:
            result_path = os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)
        
        # Make sure the folder exists
        dir_to_create = os.path.dirname(result_path)
        if dir_to_create:
            os.makedirs(dir_to_create, exist_ok=True)
        return result_path

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)
    
    def set_cancel_check_callback(self, callback):
        """Set the callback that checks for cancellation"""
        self._cancel_check_callback = callback
    
    def _check_cancelled(self):
        """Check whether the task was cancelled"""
        if self._cancel_check_callback and self._cancel_check_callback():
            logger.warning("[Stage] Task cancelled")
            raise asyncio.CancelledError("Task cancelled")

    async def _report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            await ph(state, finished)

    def _add_logger_hook(self):
        # TODO: Pass ctx to logger hook
        LOG_MESSAGES = {
            'upscaling': 'Running upscaling',
            'detection': 'Running text detection',
            'ocr': 'Running ocr',
            'mask-generation': 'Running mask refinement',
            'inpainting': 'Running inpainting',
            'translating': 'Running text translation',
            'rendering': 'Running rendering',
            'colorizing': 'Running colorization',
            'downscaling': 'Running downscaling',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-regions': 'No text regions! - Skipping',
            'skip-no-text': 'No text regions with text! - Skipping',
            'error-translating': 'Text translator returned empty queries',
            'cancelled': 'Image translation cancelled',
        }
        LOG_MESSAGES_ERROR = {
            # 'error-lang':           'Target language not supported by chosen translator',
        }

        async def ph(state, finished):
            if state in LOG_MESSAGES:
                logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                logger.warning(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)

    @staticmethod
    def _close_batch_images(batch_images, keep_ids=frozenset()):
        """Close the input images of a batch, except the ones a result still refers to."""
        for image, _ in batch_images:
            if id(image) in keep_ids:
                continue
            if hasattr(image, 'close'):
                try:
                    image.close()
                except Exception as ignored_error:
                    note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._close_batch_images")

    @staticmethod
    def _load_text_failure_context(image, error):
        """Build the result of a page whose saved text could not be rendered: the error plus the original image."""
        ctx = Context()
        ctx.input = image
        ctx.text_regions = []
        if hasattr(image, 'name'):
            ctx.image_name = image.name
        ctx.translation_error = str(error)
        fallback_result = None
        try:
            fallback_result = image.copy()
        except Exception as copy_error:
            logger.warning(f"Failed to copy fallback image for load_text error: {copy_error}")
            image_name = getattr(image, 'name', None)
            if image_name:
                try:
                    reopened_image = open_pil_image(image_name, eager=True)
                    fallback_result = reopened_image.copy()
                    reopened_image.close()
                except Exception as reopen_error:
                    logger.warning(f"Failed to reopen fallback image for load_text error: {reopen_error}")
        ctx.result = fallback_result
        return ctx

    async def _prepare_load_text_masks(
        self,
        *,
        config,
        ctx,
        editor_export_kind,
        image_name,
        import_yolo_labels,
        loaded_mask,
        mask_is_refined,
    ):
        """load_text mode: choose the mask of a page and make it match the size of the image."""
        # Handle the mask
        if editor_export_kind == 'source':
            ctx.mask = np.zeros_like(ctx.img_rgb[:, :, 0])
        elif loaded_mask is not None:
            if mask_is_refined:
                ctx.mask = loaded_mask
            else:
                ctx.mask_raw = loaded_mask
        else:
            if import_yolo_labels:
                try:
                    mask_ctx = Context()
                    # load_text uses the original image already; share its
                    # array so the bubble mask stays with the same context.
                    mask_ctx.img_rgb = ctx.img_rgb
                    mask_ctx.bubble_mask = ctx.bubble_mask
                    mask_ctx.image_name = image_name
                    _, generated_mask_raw, generated_mask = await self._run_detection(config, mask_ctx)
                    ctx.bubble_mask = mask_ctx.bubble_mask
                    if generated_mask_raw is not None:
                        ctx.mask_raw = generated_mask_raw
                    if generated_mask is not None:
                        ctx.mask = generated_mask
                    if ctx.mask_raw is not None or ctx.mask is not None:
                        logger.info("Load text mode: generated mask from detection because JSON has no mask")
                except Exception as e:
                    logger.warning(f"Load text mode: detection-based mask generation failed, fallback to region mask ({e})")

            if ctx.mask_raw is None and ctx.mask is None:
                mask = np.zeros_like(ctx.img_rgb[:, :, 0])
                # fillPoly only accepts integer points; saved outlines are floats.
                polygons = [
                    np.round(p).astype(np.int32).reshape((-1, 1, 2))
                    for r in ctx.text_regions for p in r.lines
                ]
                cv2.fillPoly(mask, polygons, 255)
                ctx.mask_raw = mask

        # In load_text mode, always align the mask to the current image size, upscaled or not,
        # so that ONNX inpainting does not fail on an image/mask size mismatch.
        target_h, target_w = ctx.img_rgb.shape[:2]
        for mask_attr in ('mask_raw', 'mask'):
            mask_val = getattr(ctx, mask_attr, None)
            if mask_val is None:
                continue

            mask_arr = np.asarray(mask_val)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            elif mask_arr.ndim != 2:
                squeezed = np.squeeze(mask_arr)
                if squeezed.ndim == 2:
                    mask_arr = squeezed
                else:
                    logger.warning(
                        f"[load_text] {mask_attr} shape invalid ({mask_arr.shape}), fallback to zero mask {target_h}x{target_w}"
                    )
                    mask_arr = np.zeros((target_h, target_w), dtype=np.uint8)

            if mask_arr.shape[0] != target_h or mask_arr.shape[1] != target_w:
                logger.warning(
                    f"[load_text] Resizing {mask_attr} from {mask_arr.shape[:2]} to {(target_h, target_w)}"
                )
                mask_arr = cv2.resize(mask_arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            if mask_arr.dtype != np.uint8:
                mask_arr = mask_arr.astype(np.uint8, copy=False)

            setattr(ctx, mask_attr, mask_arr)

    async def _finish_saved_page_without_text(
        self,
        *,
        config,
        ctx,
        editor_export_kind,
        existing_inpainted_path,
        image_name,
        loaded_mask,
        preloaded_inpainted,
    ):
        """load_text mode: finish a page that has no text regions, inpainting it when it has a mask."""
        mask_for_inpainting = ctx.mask if ctx.mask is not None else ctx.mask_raw
        has_mask_for_inpainting = False
        if mask_for_inpainting is not None:
            try:
                has_mask_for_inpainting = np.count_nonzero(mask_for_inpainting) > 0
            except Exception as mask_count_err:
                logger.warning(
                    f"Load text mode: failed to inspect imported mask for {os.path.basename(image_name)} "
                    f"({mask_count_err}), falling back to original image"
                )

        if has_mask_for_inpainting:
            logger.info(
                f"No text regions found in JSON for {os.path.basename(image_name)}, "
                "using imported mask for inpaint-only output"
            )
            mask_injected_from_raw = ctx.mask is None
            if ctx.mask is None:
                ctx.mask = np.asarray(mask_for_inpainting, dtype=np.uint8)

            generated_inpainted_in_load_text = False
            if preloaded_inpainted is not None:
                ctx.img_inpainted = preloaded_inpainted
                logger.info("Load text mode: using editor-provided paired inpainted image for mask-only import.")
            elif editor_export_kind == 'paired':
                raise RuntimeError("Paired export is missing its in-memory inpainted image")
            elif existing_inpainted_path and loaded_mask is not None:
                try:
                    existing_inpainted_image = open_pil_image(existing_inpainted_path, eager=False)
                    existing_inpainted_rgb, _ = load_image(existing_inpainted_image)
                    ctx.img_inpainted = existing_inpainted_rgb
                    logger.info("Load text mode: Using existing inpainted image for mask-only import.")
                except Exception as existing_inpaint_err:
                    logger.warning(
                        f"Load text mode: failed to load existing inpainted image for mask-only import, "
                        f"rerunning inpainting ({existing_inpaint_err})"
                    )
                    await self._report_progress('inpainting')
                    ctx.img_inpainted = await self._run_inpainting(config, ctx)
                    generated_inpainted_in_load_text = True
            else:
                await self._report_progress('inpainting')
                ctx.img_inpainted = await self._run_inpainting(config, ctx)
                generated_inpainted_in_load_text = True

            if ctx.img_inpainted is None:
                raise RuntimeError("Inpainting completed without an image")
            if editor_export_kind == 'backend_inpaint':
                if not generated_inpainted_in_load_text:
                    raise RuntimeError("Backend inpaint export did not run inpainting")
                ctx.editor_export_generated_inpainted = np.array(
                    ctx.img_inpainted, dtype=np.uint8, copy=True
                )
            ctx.inpainted_regenerated = generated_inpainted_in_load_text
            if (
                generated_inpainted_in_load_text
                and image_name
                and ctx.img_inpainted is not None
                and self.save_text
            ):
                self._save_inpainted_image(
                    image_name,
                    ctx.img_inpainted,
                )

            # Composite the paint and stamp layers (after saving the inpainted image, so the layers are not baked into that file)
            self._compose_render_overlays_on_inpainted(ctx)

            await self._report_progress('finished', True)
            ctx.result = dump_image(ctx.input, ctx.img_inpainted, ctx.img_alpha, mask=ctx.mask)
            if mask_injected_from_raw:
                ctx.mask = None
            ctx = await self._revert_upscale(config, ctx)
        else:
            logger.info(
                f"No text regions or usable mask found in JSON for {os.path.basename(image_name)}, "
                "returning original image"
            )
            # The page may only have paste or paint layers: composite before returning, so the overlays are not dropped
            self._compose_render_overlays_on_inpainted(ctx)
            await self._report_progress('finished', True)
            ctx.result = ctx.upscaled  # Return the upscaled original image
            ctx = await self._revert_upscale(config, ctx)
        return ctx

    async def _finish_saved_page_with_text(
        self,
        *,
        config,
        ctx,
        editor_export_kind,
        existing_inpainted_path,
        image_name,
        loaded_mask,
        preloaded_inpainted,
        skip_font_scaling,
        skip_text_replacements,
    ):
        """load_text mode: refine the mask, inpaint and render the saved text of a page."""
        # Mask refinement
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            ctx.mask = await self._run_mask_refinement(config, ctx)

        # Inpainting
        generated_inpainted_in_load_text = False
        if editor_export_kind == 'source':
            ctx.img_inpainted = ctx.img_rgb
        elif preloaded_inpainted is not None:
            ctx.img_inpainted = preloaded_inpainted
            logger.info("Load text mode: using editor-provided paired inpainted image, skipping inpainting.")
        elif editor_export_kind == 'paired':
            raise RuntimeError("Paired export is missing its in-memory inpainted image")
        elif editor_export_kind == 'backend_inpaint':
            await self._report_progress('inpainting')
            ctx.img_inpainted = await self._run_inpainting(config, ctx)
            generated_inpainted_in_load_text = True
        elif self._should_skip_inpainting_for_ai_renderer(config):
            logger.info("AI renderer selected: skipping inpainting outside strict editor export.")
            ctx.img_inpainted = ctx.img_rgb
        elif existing_inpainted_path and loaded_mask is not None:
            try:
                existing_inpainted_image = open_pil_image(existing_inpainted_path, eager=False)
                existing_inpainted_rgb, _ = load_image(existing_inpainted_image)
                ctx.img_inpainted = existing_inpainted_rgb
                logger.info("Load text mode: Using existing inpainted image, skipping inpainting.")
            except Exception as existing_inpaint_err:
                logger.warning(
                    f"Load text mode: failed to load existing inpainted image, rerunning inpainting ({existing_inpaint_err})"
                )
                await self._report_progress('inpainting')
                ctx.img_inpainted = await self._run_inpainting(config, ctx)
                generated_inpainted_in_load_text = True
        else:
            await self._report_progress('inpainting')
            ctx.img_inpainted = await self._run_inpainting(config, ctx)
            generated_inpainted_in_load_text = True

        if ctx.img_inpainted is None:
            raise RuntimeError("Inpainting completed without an image")
        if editor_export_kind == 'backend_inpaint':
            if not generated_inpainted_in_load_text:
                raise RuntimeError("Backend inpaint export did not run inpainting")
            ctx.editor_export_generated_inpainted = np.array(
                ctx.img_inpainted, dtype=np.uint8, copy=True
            )

        ctx.inpainted_regenerated = generated_inpainted_in_load_text
        if (
            generated_inpainted_in_load_text
            and image_name
            and ctx.img_inpainted is not None
            and self.save_text
        ):
            self._save_inpainted_image(
                image_name,
                ctx.img_inpainted,
            )

        # Composite the paint and stamp layers (after saving the inpainted image, so the layers are not baked into that file)
        self._compose_render_overlays_on_inpainted(ctx)

        # Rendering - in load_text mode skip_font_scaling from the JSON decides: True = skip font scaling, False = scale the font
        await self._report_progress('rendering')
        ctx.img_rendered = await self._run_text_rendering(
            config,
            ctx,
            skip_font_scaling=skip_font_scaling,
            skip_text_replacements=skip_text_replacements,
        )

        await self._report_progress('finished', True)
        ctx.result = dump_image(
            ctx.input,
            ctx.img_rendered,
            ctx.img_alpha,
            mask=ctx.mask,
            render_alpha=getattr(ctx, 'img_render_alpha', None),
        )
        ctx = await self._revert_upscale(config, ctx)
        return ctx

    def _write_back_load_text_json(self, ctx, config):
        """load_text mode: store the regions as rendered, unless that would lose data."""
        # load_text mode: write the JSON back after rendering (syncs the latest regions, including translation, font_size and so on)
        # Skipped for an editor export (the editor has already stored the project JSON itself)
        if (
            hasattr(ctx, 'text_regions') and ctx.text_regions is not None
            and hasattr(ctx, 'image_name') and ctx.image_name
            and not getattr(ctx, 'editor_export', False)
        ):
            parse_failures = getattr(ctx, 'load_text_parse_failures', 0)
            if parse_failures:
                # Safety fuse: when a region failed to parse, skip the overwriting write-back; otherwise those regions would
                # vanish from the project JSON for good, with their original text and coordinates (there is no backup).
                logger.error(
                    f"{parse_failures} region(s) failed to parse for "
                    f"{os.path.basename(ctx.image_name)}; skipped JSON write-back to protect the project file"
                )
            else:
                try:
                    self._save_text_to_file(ctx.image_name, ctx, config)
                except Exception as save_json_err:
                    logger.error(f"Error updating JSON in load_text mode for {os.path.basename(ctx.image_name)}: {save_json_err}")

    async def _render_page_from_saved_text(self, image, config):
        """load_text mode: render one page from its saved JSON. Returns None when the image cannot be read."""
        self._set_image_context(config, image)
        image_name = image.name if hasattr(image, 'name') else None

        # Handle load_text mode directly, without calling translate(), to avoid an endless loop
        ctx = Context()
        ctx.input = image
        ctx.image_name = image_name
        ctx.verbose = self.verbose
        ctx.save_quality = self.save_quality
        ctx.config = config
        ctx.inpainted_regenerated = False

        # One flag: a registered in-memory payload == an editor export.
        # The backend treats it as the authorised final version and only renders (no JSON write-back, no text replacement,
        # the mask is already refined and the inpainted image is reused as it is).
        preloaded_payload = self._preloaded_load_text_payloads.get(image_name) if image_name else None
        ctx.editor_export = preloaded_payload is not None
        editor_export_kind = (
            preloaded_payload.get('editor_export_base_kind')
            if preloaded_payload is not None
            else None
        )
        if ctx.editor_export and editor_export_kind not in {
            'source', 'paired', 'backend_inpaint'
        }:
            raise ValueError(
                f"Invalid editor export base kind: {editor_export_kind!r}"
            )

        # Load the translation data
        loaded_regions, loaded_mask, mask_is_refined, skip_font_scaling, skip_text_replacements, region_parse_failures = self._load_text_and_regions_from_file(image_name, config)
        if loaded_regions is None:
            json_path = os.path.splitext(image_name)[0] + '_translations.json' if image_name else 'unknown'
            raise FileNotFoundError(f"Translation file not found or invalid: {json_path}")

        # When regions is an empty list, log it and carry on (the original image is rendered)
        if not loaded_regions:
            logger.info(f"No text regions found in JSON for {os.path.basename(image_name)}, will render original image")

        self._prepare_loaded_regions(loaded_regions, use_text_as_translation=True)

        ctx.text_regions = loaded_regions
        ctx.skip_font_scaling = skip_font_scaling
        ctx.skip_text_replacements = skip_text_replacements
        # No JSON write-back when a region failed to parse, so missing regions are not written over the project file
        ctx.load_text_parse_failures = region_parse_failures

        preloaded_inpainted_raw = preloaded_payload.get('inpainted_rgb') if preloaded_payload else None
        # Strict editor export never consults an unverified historical sidecar.
        existing_inpainted_path = None
        if preloaded_payload is None and image_name:
            existing_inpainted_path = find_inpainted_path(image_name)

        # load_text always works from the original image: no colorization or upscaling, and an existing inpainted image is not put into img_colorized/upscaled
        ctx.img_colorized = ctx.input
        ctx.upscaled = ctx.input

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)
        ctx.bubble_mask = None

        # Check the loaded image
        if ctx.img_rgb is None or ctx.img_rgb.size == 0:
            logger.error("[Batch] Failed to load image: img_rgb is empty or invalid")
            return None

        if len(ctx.img_rgb.shape) < 2 or ctx.img_rgb.shape[0] == 0 or ctx.img_rgb.shape[1] == 0:
            logger.error(f"[Batch] Invalid loaded image dimensions: {ctx.img_rgb.shape}")
            return None

        import_yolo_labels = bool(getattr(config.detector, 'import_yolo_labels', False))
        if loaded_mask is not None or not import_yolo_labels:
            # load_text does not run OCR; with skip_font_scaling (a layout authorised by the editor) the
            # center_box anchor is always used, the bubble mask takes no part in placement and the renderer does not use the bubble cache either.
            # Only with automatic layout (balloon_fill / centring in the bubble), or when the mask still needs refining and
            # bubble-limited refinement is on, is the cache primed.
            # When YOLO boxes are imported and detection has to run again to build the mask, _run_detection primes it later itself.
            mask_refinement_will_run = not (loaded_mask is not None and mask_is_refined)
            render_needs_bubble_cache = (
                not skip_font_scaling
                and (
                    getattr(config.render, 'layout_mode', None) == 'balloon_fill'
                    or bool(getattr(config.render, 'center_text_in_bubble', False))
                )
            )
            needs_bubble_cache = render_needs_bubble_cache or (
                (
                    bool(getattr(config.ocr, 'limit_mask_dilation_to_bubble_mask', False))
                    or bool(getattr(config.ocr, 'use_model_bubble_repair_intersection', False))
                )
                and mask_refinement_will_run
            )
            if needs_bubble_cache:
                self._prime_bubble_detection_cache(config, ctx)

        await self._prepare_load_text_masks(
            config=config,
            ctx=ctx,
            editor_export_kind=editor_export_kind,
            image_name=image_name,
            import_yolo_labels=import_yolo_labels,
            loaded_mask=loaded_mask,
            mask_is_refined=mask_is_refined,
        )

        # Align the editor's inpainted image to the working image size, so the branches below can reuse it directly
        preloaded_inpainted = self._align_preloaded_inpainted(preloaded_inpainted_raw, ctx.img_rgb)

        # load_text supports an "inpaint only" mode: even without text regions, inpainting runs when the JSON has a usable mask.
        if not ctx.text_regions:
            ctx = await self._finish_saved_page_without_text(
                config=config,
                ctx=ctx,
                editor_export_kind=editor_export_kind,
                existing_inpainted_path=existing_inpainted_path,
                image_name=image_name,
                loaded_mask=loaded_mask,
                preloaded_inpainted=preloaded_inpainted,
            )
        else:
            ctx = await self._finish_saved_page_with_text(
                config=config,
                ctx=ctx,
                editor_export_kind=editor_export_kind,
                existing_inpainted_path=existing_inpainted_path,
                image_name=image_name,
                loaded_mask=loaded_mask,
                preloaded_inpainted=preloaded_inpainted,
                skip_font_scaling=skip_font_scaling,
                skip_text_replacements=skip_text_replacements,
            )

        self._write_back_load_text_json(ctx, config)
        return ctx

    def _load_page_for_json_translation(self, image, config):
        """translate_json_only mode: build the context of one page from the original text in its JSON."""
        self._set_image_context(config, image)
        image_name = image.name if hasattr(image, 'name') else None

        ctx = Context()
        ctx.input = image
        ctx.image_name = image_name
        ctx.verbose = self.verbose
        ctx.save_quality = self.save_quality
        ctx.config = config
        ctx.from_lang = 'auto'

        loaded_regions, loaded_mask, mask_is_refined, skip_font_scaling, _skip_text_replacements, region_parse_failures = self._load_text_and_regions_from_file(image_name, config)
        if loaded_regions is None:
            json_path = find_json_path(image_name) if image_name else None
            if not json_path and image_name:
                json_path = get_json_path(image_name, create_dir=False)
            raise FileNotFoundError(f"JSON translation data not found or invalid: {json_path}")

        self._prepare_loaded_regions(loaded_regions, use_text_as_translation=False)
        ctx.text_regions = loaded_regions
        ctx.skip_font_scaling = skip_font_scaling
        # No JSON write-back when a region failed to parse, so missing regions are not written over the project file
        ctx.load_text_parse_failures = region_parse_failures

        if loaded_mask is not None:
            if mask_is_refined:
                ctx.mask = loaded_mask
            else:
                ctx.mask_raw = loaded_mask

        self._apply_pre_dictionary_to_regions(ctx)
        return ctx

    def _save_translated_json(self, ctx, config):
        """translate_json_only mode: write the translated regions of one page back to its JSON."""
        parse_failures = getattr(ctx, 'load_text_parse_failures', 0)
        if parse_failures:
            # Safety fuse: the write-back rebuilds the whole JSON from the current regions, so regions that failed to parse
            # would be deleted for good; fail explicitly here instead and keep the original file.
            raise IOError(
                f"{parse_failures} region(s) failed to parse from JSON; "
                "skipped saving to protect the project file"
            )
        save_success = self._save_text_to_file(ctx.image_name, ctx, config)
        if not save_success:
            raise IOError(f"Failed to save JSON for {os.path.basename(ctx.image_name)}")
        self._delete_original_txt_after_json_translation(ctx.image_name)
        ctx.success = True
        ctx.output_path = get_json_path(ctx.image_name, create_dir=False)

    async def translate_batch(self, images_with_configs: List[tuple], batch_size: int = None, image_names: List[str] = None, save_info: dict = None, global_offset: int = 0, global_total: int = None) -> List[Context]:
        """Translate a complete ordered input list and return processed and skipped results."""
        if self.filter_text_enabled:
            from .utils.text_filter import load_filter_list
            load_filter_list(force_reload=True)

        source_items = [
            (image, self._apply_runtime_cli_overrides(config))
            for image, config in images_with_configs
        ]
        plan = plan_batch_inputs(self, source_items, save_info)
        images_with_configs = plan.pending_items
        skipped_count = plan.skipped_count
        self.all_page_translations.clear()
        self._original_page_texts.clear()
        display_total = global_total if global_total is not None else global_offset + len(source_items)
        processing_offset = global_offset + skipped_count
        self._prepare_resume_context(plan)

        if skipped_count:
            completed = min(processing_offset, display_total)
            await self._report_progress(
                f"batch:{completed}:{completed}:{display_total}:0:{skipped_count}"
            )
        if not images_with_configs:
            return plan.merge_results([])

        batch_size = batch_size or self.batch_size
        is_text_export_mode = self.generate_and_export or (self.template and self.save_text)
        if self.export_from_local_json and is_text_export_mode:
            contexts = await self._export_text_from_local_json(
                images_with_configs,
                global_offset=processing_offset,
                global_total=display_total,
                skipped_count=skipped_count,
            )
            return plan.merge_results(contexts)

        if self.load_text:
            logger.info("Load text mode detected: Auto-importing translations from TXT to JSON...")
            self._preprocess_load_text_mode(images_with_configs)

        if self.replace_translation:
            logger.info("Replace translation mode detected: Will extract translations from translated images")
            contexts = await self._translate_batch_replace_translation(
                images_with_configs,
                save_info,
                processing_offset,
                display_total,
            )
            return plan.merge_results(contexts)

        is_template_save_mode = self.template and self.save_text
        has_incompatible_mode = (
            self.load_text
            or self.translate_json_only
            or is_template_save_mode
            or self.generate_and_export
            or self.colorize_only
            or self.upscale_only
            or self.inpaint_only
            or self.replace_translation
        )
        effective_batch_concurrent = self.batch_concurrent and not has_incompatible_mode

        is_hq_translator = False
        first_config = images_with_configs[0][1]
        if first_config and hasattr(first_config.translator, 'translator'):
            translator_type = first_config.translator.translator
            is_hq_translator = translator_type in [Translator.openai_hq, Translator.gemini_hq]
            is_import_export_mode = self.load_text or self.template or self.translate_json_only
            if is_hq_translator and not is_import_export_mode and not effective_batch_concurrent:
                logger.info(f"Detected high-quality translator {translator_type}; enabling high-quality translation mode")
                contexts = await self._translate_batch_high_quality(
                    images_with_configs,
                    save_info,
                    global_offset=processing_offset,
                    global_total=display_total,
                    batch_size=batch_size,
                    skipped_count=skipped_count,
                )
                return plan.merge_results(contexts)
            if is_hq_translator and is_import_export_mode:
                logger.warning("Translation import/export mode detected; skipping high-quality translation and using the standard rendering pipeline.")

        if is_template_save_mode:
            logger.info("Template+SaveText mode detected. Using one-item backend batches.")
            batch_size = 1
        elif batch_size <= 1 and not effective_batch_concurrent:
            logger.debug('Batch size <= 1, using one-item backend batches')
            batch_size = 1

        if self.batch_concurrent and has_incompatible_mode:
            incompatible_modes = []
            if self.load_text:
                incompatible_modes.append("load translation")
            if self.translate_json_only:
                incompatible_modes.append("translate JSON only")
            if is_template_save_mode:
                incompatible_modes.append("export original text")
            if self.generate_and_export:
                incompatible_modes.append("export translation")
            if self.colorize_only:
                incompatible_modes.append("colorize only")
            if self.upscale_only:
                incompatible_modes.append("upscale only")
            if self.inpaint_only:
                incompatible_modes.append("inpaint only")
            if self.replace_translation:
                incompatible_modes.append("replace translation")
            logger.info(f"⚠️  Concurrent pipeline disabled: current modes [{', '.join(incompatible_modes)}] do not support concurrent processing")

        if effective_batch_concurrent:
            mode_desc = "high-quality translation" if is_hq_translator else "standard translation"
            logger.info(
                f"🚀 Concurrent pipeline enabled ({mode_desc}): {len(images_with_configs)} images, translation batch size: {batch_size}"
            )
            from .utils.concurrent_pipeline import ConcurrentPipeline

            self._current_save_info = save_info
            pipeline = ConcurrentPipeline(self, batch_size)
            file_paths = [input_path(item) for item in images_with_configs]
            configs = [item[1] for item in images_with_configs]
            contexts = await pipeline.process_batch(
                file_paths,
                configs,
                progress_offset=processing_offset,
                progress_total=display_total,
                skipped_count=skipped_count,
            )
            self._prune_context_history()
            return plan.merge_results(contexts)

        logger.info(f'Starting batch translation: {len(images_with_configs)} images, batch size: {batch_size}')
        logger.info("[Stage] Batch translation task started")
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())

        results = []

        async def report_completed_image_progress():
            if display_total <= 0:
                return
            completed = min(processing_offset + len(results), display_total)
            failed_count = sum(1 for ctx in results if getattr(ctx, 'translation_error', None))
            current_skipped = skipped_count + sum(1 for ctx in results if getattr(ctx, 'skipped', False))
            await self._report_progress(
                f"batch:{completed}:{completed}:{display_total}:{failed_count}:{current_skipped}"
            )

        batch_slices = slice_batch_indices(
            images_with_configs,
            batch_size,
            self._resume_context_pages,
            self._resume_context_order,
        )
        total_batches = len(batch_slices)

        # Process all images in batches
        for batch_num, (batch_start, batch_end) in enumerate(batch_slices, start=1):
            current_batch_images = []
            preprocessed_contexts = []
            translated_contexts = []
            
            try:
                await asyncio.sleep(0)  # Check whether the task was cancelled
                self._check_cancelled()  # Check the cancel flag

                current_batch_items = images_with_configs[batch_start:batch_end]
                if current_batch_items:
                    self._append_resume_context_before(input_path(current_batch_items[0]))

                # Global image number (takes the offset of the frontend's batched loading into account)
                global_batch_start = processing_offset + batch_start + 1
                global_batch_end = processing_offset + batch_end
                progress_state = f"batch:{global_batch_start}:{global_batch_end}:{display_total}:0:{skipped_count}"
                
                logger.info(f"Processing rolling batch {batch_num}/{total_batches} (images {global_batch_start}-{global_batch_end})")
                logger.info(f"[Stage] Processing batch {batch_num}/{total_batches}")

                current_batch_images, load_error_contexts = self._materialize_batch_inputs(current_batch_items)
                if load_error_contexts:
                    results.extend(load_error_contexts)
                    await report_completed_image_progress()
                if not current_batch_images:
                    await self._report_progress(progress_state)
                    continue

                # --- Stage 1: preprocessing (detection, OCR, text line merging) ---
                
                # Special case: load_text mode (translations are loaded from JSON)
                if self.load_text:
                    logger.info("Load text mode: Loading translations from JSON and skipping text detection/OCR/translation")
                    for i, (image, config) in enumerate(current_batch_images):
                        await asyncio.sleep(0)
                        self._check_cancelled()  # Check the cancel flag
                        try:
                            ctx = await self._render_page_from_saved_text(image, config)
                            if ctx is None:
                                continue
                            
                            preprocessed_contexts.append((ctx, config))

                            # A load_text result sometimes reuses the input image (for example without text regions,
                            # ctx.result = ctx.upscaled), or still holds a lazy handle on the input file.
                            # The memory clean-up that follows closes the input image, so a fully independent result has to be made first;
                            # otherwise the caller's copy/save after translate() returns fails with:
                            # ValueError: Operation on closed image
                            self._detach_context_result(ctx)

                            # ✅ Free memory right after each image (the result is kept)
                            self._cleanup_context_memory(ctx, keep_result=True)
                            
                        except Exception as e:
                            logger.error(f"Error loading text for image {i+1} in batch: {e}")
                            ctx = self._load_text_failure_context(image, e)
                            preprocessed_contexts.append((ctx, config))
                    
                    # In load_text mode everything is already done (rendering included); save and return
                    for ctx, config in preprocessed_contexts:
                        if save_info and ctx.result:
                            try:
                                # Use the shared save-and-clean-up method (includes the PSD export)
                                self._save_and_cleanup_context(ctx, save_info, config, "LOAD_TEXT")
                            except Exception as save_err:
                                logger.error(f"Error saving load_text result for {os.path.basename(ctx.image_name)}: {save_err}")
                        
                        results.append(ctx)
                        await report_completed_image_progress()
                    
                    # ✅ load_text mode: clear the batch data once the batch is done (the images were cleared inside the loop)
                    # Note: an image object that ctx.result still refers to must not be closed
                    if current_batch_images:
                        # Collect every image object a result still refers to
                        result_ids = {id(ctx.result) for ctx, _ in preprocessed_contexts if ctx.result is not None}
                        self._close_batch_images(current_batch_images, keep_ids=result_ids)
                    
                    # Empty the lists, so that _cleanup_batch_memory in finally does not close these images again
                    current_batch_images.clear()
                    preprocessed_contexts.clear()
                    
                    # load_text mode is finished; go on to the next batch
                    continue

                if self.translate_json_only:
                    logger.info("Translate JSON only mode: Loading original text from JSON and skipping detection/OCR/rendering")
                    for i, (image, config) in enumerate(current_batch_images):
                        await asyncio.sleep(0)
                        self._check_cancelled()
                        try:
                            ctx = self._load_page_for_json_translation(image, config)
                            preprocessed_contexts.append((ctx, config))
                        except Exception as e:
                            logger.error(f"Error loading JSON translation data for image {i+1} in batch: {e}")
                            ctx = Context()
                            ctx.input = image
                            ctx.text_regions = []
                            if hasattr(image, 'name'):
                                ctx.image_name = image.name
                            ctx.translation_error = str(e)
                            preprocessed_contexts.append((ctx, config))

                    logger.info("[Stage] Original text loaded from JSON; starting translation")
                    try:
                        translated_contexts = await self._batch_translate_contexts(preprocessed_contexts, batch_size)
                    except Exception as e:
                        logger.error(f"Error during JSON-only batch translation stage: {e}")
                        raise

                    for ctx, config in translated_contexts:
                        if getattr(ctx, 'translation_error', None):
                            results.append(ctx)
                            await report_completed_image_progress()
                            continue

                        try:
                            self._save_translated_json(ctx, config)
                        except Exception as save_err:
                            logger.error(f"Error saving translated JSON for {os.path.basename(ctx.image_name)}: {save_err}")
                            ctx = self._mark_context_failure(ctx, save_err, stage='saving')

                        results.append(ctx)
                        await report_completed_image_progress()
                        self._cleanup_context_memory(ctx, keep_result=True)

                    if current_batch_images:
                        self._close_batch_images(current_batch_images)

                    continue

                # Standard mode: run detection, OCR and the other preprocessing
                logger.info("[Stage] Starting preprocessing (detection, OCR)")
                for i, (image, config) in enumerate(current_batch_images):
                    # Check whether the task was cancelled
                    await asyncio.sleep(0)
                    self._check_cancelled()  # Check the cancel flag
                    try:
                        self._set_image_context(config, image)
                        # ✅ Save the context so the rendering stage can reuse it and no second folder is created
                        from .utils.generic import get_image_md5
                        image_md5 = get_image_md5(image)
                        self._save_current_image_context(image_md5)
                        ctx = await self._translate_until_translation(image, config)
                        if hasattr(image, 'name'):
                            ctx.image_name = image.name
                        preprocessed_contexts.append((ctx, config))
                    except Exception as e:
                        logger.error(f"Error pre-processing image {i+1} in batch: {e}", exc_info=True)
                        ctx = self._build_stage_error_context(image, e, config, stage='preprocessing')
                        preprocessed_contexts.append((ctx, config))

                # --- Stage 2: translation ---
                logger.info("[Stage] Preprocessing completed; starting translation")
                if self.colorize_only or self.upscale_only or self.inpaint_only:
                    # Special case: colorize-only / upscale-only / inpaint-only mode skips translation
                    mode_name = "Colorize Only" if self.colorize_only else ("Upscale Only" if self.upscale_only else "Inpaint Only")
                    logger.info(f"{mode_name} mode: Skipping translation and rendering stages.")
                    translated_contexts = preprocessed_contexts
                elif is_template_save_mode:
                    # Special case: export-original-text mode skips translation
                    logger.info("Template+SaveText mode: Skipping translation, will export original text only.")
                    translated_contexts = preprocessed_contexts
                else:
                    # Standard translation flow
                    try:
                        translated_contexts = await self._batch_translate_contexts(preprocessed_contexts, batch_size)
                    except Exception as e:
                        logger.error(f"Error during batch translation stage: {e}")
                        raise

                # --- Stage 3: rendering and saving ---
                # Special case: export-original-text mode (skips rendering; only saves the JSON and exports the original text)
                if is_template_save_mode:
                    logger.info("Template+SaveText mode: Skipping rendering, exporting original text only.")
                    for ctx, config in translated_contexts:
                        if getattr(ctx, 'translation_error', None):
                            results.append(ctx)
                            await report_completed_image_progress()
                            continue
                        await self._handle_template_and_save_text(ctx, config)
                        # ✅ Mark as successful (the original text was exported)
                        ctx.success = True
                        results.append(ctx)
                        await report_completed_image_progress()
                        
                        # ✅ Free memory right after each image (the result is kept)
                        self._cleanup_context_memory(ctx, keep_result=True)
                    
                    # ✅ Clear the batch data once the batch is done (the images were cleared inside the loop)
                    if current_batch_images:
                        self._close_batch_images(current_batch_images)
                    
                    continue  # Skip rendering and go on to the next batch
                
                # Special case: generate-and-export mode (skips rendering)
                if self.generate_and_export:
                    logger.info("'Generate and Export' mode enabled. Skipping rendering.")
                    for ctx, config in translated_contexts:
                        if getattr(ctx, 'translation_error', None):
                            results.append(ctx)
                            await report_completed_image_progress()
                            continue
                        await self._handle_generate_and_export(ctx, config)
                        # ✅ Mark as successful (the translation was exported)
                        ctx.success = True
                        results.append(ctx)
                        await report_completed_image_progress()
                        
                        # ✅ Free memory right after each image (the result is kept)
                        self._cleanup_context_memory(ctx, keep_result=True)
                    
                    # ✅ Clear the batch data once the batch is done (the images were cleared inside the loop)
                    if current_batch_images:
                        self._close_batch_images(current_batch_images)
                    
                    continue  # Skip rendering and go on to the next batch

                # Standard flow: render and save
                logger.info("[Stage] Translation completed; starting rendering")
                for idx, (ctx, config) in enumerate(translated_contexts):
                    await asyncio.sleep(0)  # Check whether the task was cancelled
                    self._check_cancelled()  # Check the cancel flag
                    if getattr(ctx, 'translation_error', None):
                        results.append(ctx)
                        await report_completed_image_progress()
                        continue
                    try:
                        if hasattr(ctx, 'input'):
                            from .utils.generic import get_image_md5
                            image_md5 = get_image_md5(ctx.input)
                            if not self._restore_image_context(image_md5):
                                self._set_image_context(config, ctx.input)
                        
                        # Colorize/Upscale/Inpaint Only Mode: Skip rendering pipeline
                        if not self.colorize_only and not self.upscale_only and not self.inpaint_only:
                            ctx = await self._complete_translation_pipeline(ctx, config)
                        if save_info and ctx.result:
                            try:
                                # Use the shared save-and-clean-up method (includes the PSD export)
                                self._save_and_cleanup_context(ctx, save_info, config, "BATCH")
                            except Exception as save_err:
                                logger.error(f"Error saving standard batch result for {os.path.basename(ctx.image_name)}: {save_err}")

                        # Save the JSON only when save_text or text_output_file is on (an empty text_regions is saved too)
                        if not getattr(ctx, 'skipped', False) and (self.save_text or self.text_output_file) and hasattr(ctx, 'text_regions') and ctx.text_regions is not None and hasattr(ctx, 'image_name') and ctx.image_name:
                            # Use the config from the loop variable, not the one on ctx
                            self._save_text_to_file(ctx.image_name, ctx, config)

                        results.append(ctx)
                        await report_completed_image_progress()

                        # ✅ Clear this image's intermediate data as soon as it is rendered (without waiting for the whole batch)
                        # (the gc inside is already rate-limited by time, so no periodic forced collection is needed)
                        self._cleanup_context_memory(ctx, keep_result=True)

                    except Exception as e:
                        logger.error(f"Error rendering image in batch: {e}", exc_info=True)
                        ctx = self._mark_context_failure(ctx, e, stage='rendering')
                        results.append(ctx)
                        await report_completed_image_progress()
            
            finally:
                # ✅ Free memory as soon as the batch is done (whether it succeeded or failed)
                logger.debug(f"[Stage] Batch {batch_start//batch_size + 1} completed; cleaning up memory")
                self._cleanup_batch_memory(
                    current_batch_images=current_batch_images,
                    preprocessed_contexts=preprocessed_contexts,
                    translated_contexts=translated_contexts,
                    keep_results=True
                )
                logger.debug(f'[MEMORY] Batch {batch_start//batch_size + 1} cleanup completed')

        logger.info(f"Batch translation completed: processed {len(results)} images")
        return plan.merge_results(results)

    async def _preload_models(self, config: Config) -> None:
        """Load every model the configured pipeline needs before the first page."""
        logger.info('Loading models')

        # ✅ Check the stop flag
        await asyncio.sleep(0)
        self._check_cancelled()

        if config.upscale.upscale_ratio:
            # Pass on the upscaling settings
            upscaler_kwargs = {}
            if config.upscale.upscaler == 'realcugan':
                if config.upscale.realcugan_model:
                    upscaler_kwargs['model_name'] = config.upscale.realcugan_model
                if config.upscale.tile_size is not None:
                    upscaler_kwargs['tile_size'] = config.upscale.tile_size
            elif config.upscale.upscaler == 'mangajanai':
                # For mangajanai, upscale_ratio can be a string (x2, x4, DAT2 x4) or a number
                ratio = config.upscale.upscale_ratio
                if isinstance(ratio, str):
                    upscaler_kwargs['model_name'] = ratio
                elif ratio == 2:
                    upscaler_kwargs['model_name'] = 'x2'
                else:
                    upscaler_kwargs['model_name'] = 'x4'
                if config.upscale.tile_size is not None:
                    upscaler_kwargs['tile_size'] = config.upscale.tile_size
            await prepare_upscaling(config.upscale.upscaler, **upscaler_kwargs)

        await prepare_detection(config.detector.detector)

        await prepare_ocr(config.ocr.ocr, self.device)

        await prepare_inpainting(config.inpainter.inpainter, self.device)

        await prepare_translation(config.translator.translator_gen)

        if config.colorizer.colorizer != Colorizer.none:
            await prepare_colorization(config.colorizer.colorizer)

        self._models_loaded = True  # Mark the models as loaded

    async def _run_inpaint_only_pipeline(self, config: Config, ctx: Context) -> Context:
        """inpaint_only mode: detect the text of a page, build its mask and erase the text."""
        logger.info("=== Inpaint Only Mode ===")
        logger.info("Pipeline: Detection → Fill Text → Textline Merge → Mask Refinement → Inpainting")

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)
        ctx.bubble_mask = None

        # Check the loaded image
        if ctx.img_rgb is None or ctx.img_rgb.size == 0:
            logger.error("[Batch] Failed to load image: img_rgb is empty or invalid")
            raise Exception("Failed to load image: img_rgb is empty or invalid")

        if len(ctx.img_rgb.shape) < 2 or ctx.img_rgb.shape[0] == 0 or ctx.img_rgb.shape[1] == 0:
            logger.error(f"[Batch] Invalid loaded image dimensions: {ctx.img_rgb.shape}")
            raise Exception(f"Invalid loaded image dimensions: {ctx.img_rgb.shape}")

        # Step 1: detection - get the textlines (detected boxes) and mask_raw (raw mask)
        await self._report_progress('detection')
        try:
            ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
            logger.info(f"✓ Step 1 - Detection: Found {len(ctx.textlines) if ctx.textlines else 0} textlines")
            logger.info(f"  - mask_raw: {ctx.mask_raw.shape if ctx.mask_raw is not None else 'None'}")
            if ctx.mask_raw is not None:
                logger.info(f"  - mask_raw non-zero pixels: {np.count_nonzero(ctx.mask_raw)}")
        except Exception as e:
            logger.error(f"Error during detection:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            raise FileTranslationFailure("detection", e) from e

        if not ctx.textlines or ctx.mask_raw is None:
            logger.warning("No textlines or mask_raw detected, skipping inpainting.")
            ctx.img_inpainted = ctx.img_rgb
            ctx.result = ctx.img_inpainted
            ctx.text_regions = []
            await self._report_progress('inpaint-only-complete', True)
            # Not cleaned up here; the caller cleans up after saving the JSON
            return ctx

        # Step 2: fill in text - skip OCR and give each textline placeholder text
        for textline in ctx.textlines:
            textline.text = "TEXT"
        logger.info(f"✓ Step 2 - Fill Text: Filled {len(ctx.textlines)} textlines with placeholder 'TEXT'")

        # Step 3: Textline Merge - merge the textlines into text_regions (large boxes)
        try:
            ctx.text_regions = await dispatch_textline_merge(
                ctx.textlines,
                ctx.img_rgb.shape[1],
                ctx.img_rgb.shape[0],
                config,
                verbose=self.verbose,
                model_assisted_other_textlines=(
                    getattr(ctx, 'model_assisted_other_textlines', None)
                    if bool(getattr(config.ocr, 'merge_special_require_full_wrap', True))
                    else None
                )
            )
            logger.info(f"✓ Step 3 - Textline Merge: Merged {len(ctx.textlines)} textlines into {len(ctx.text_regions)} text_regions")
        except Exception:
            logger.error(f"Error during textline merge:\n{traceback.format_exc()}")
            # Fallback: create a simple TextBlock for each textline
            logger.warning("Falling back to simple text_regions (1 textline = 1 region)")
            ctx.text_regions = []
            fallback_line_spacing = 1.0
            fallback_letter_spacing = 1.0
            if hasattr(config, 'render'):
                line_spacing_val = getattr(config.render, 'line_spacing', None)
                if line_spacing_val is not None:
                    fallback_line_spacing = float(line_spacing_val)
                letter_spacing_val = getattr(config.render, 'letter_spacing', None)
                if letter_spacing_val is not None:
                    fallback_letter_spacing = float(letter_spacing_val)

            for textline in ctx.textlines:
                region = TextBlock(
                    lines=[textline.pts],
                    texts=["TEXT"],
                    font_size=int(textline.font_size) if hasattr(textline, 'font_size') else 20,
                    angle=0,
                    prob=textline.prob if hasattr(textline, 'prob') else 1.0,
                    fg_color=(0, 0, 0),
                    bg_color=(255, 255, 255),
                    line_spacing=fallback_line_spacing,
                    letter_spacing=fallback_letter_spacing
                )
                ctx.text_regions.append(region)
            logger.info(f"Created {len(ctx.text_regions)} simple text_regions")

        if not ctx.text_regions:
            logger.warning("No text_regions created, skipping mask refinement and inpainting.")
            ctx.img_inpainted = ctx.img_rgb
            ctx.result = ctx.img_inpainted
            await self._report_progress('inpaint-only-complete', True)
            # Not cleaned up here; the caller cleans up after saving the JSON
            return ctx

        # Step 4: Mask Refinement - refine the mask with text_regions and mask_raw
        skip_mask_refinement_for_imported_export = (
            getattr(ctx, 'used_imported_yolo_labels', False) and
            ((self.template and self.save_text) or self.generate_and_export)
        )
        if skip_mask_refinement_for_imported_export:
            logger.info("Import YOLO labels enabled in export mode: skipping mask refinement stage")
            ctx.mask = None
        else:
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
                mask_pixels = np.count_nonzero(ctx.mask) if ctx.mask is not None else 0
                logger.info(f"✓ Step 4 - Mask Refinement: Generated mask with {mask_pixels} non-zero pixels")
            except Exception:
                logger.error(f"Error during mask refinement:\n{traceback.format_exc()}")
                # Fall back to simple dilation
                logger.warning("Falling back to simple mask dilation")
                kernel = np.ones((config.kernel_size, config.kernel_size), np.uint8)
                ctx.mask = cv2.dilate(ctx.mask_raw, kernel, iterations=config.mask_dilation_offset // config.kernel_size)
                mask_pixels = np.count_nonzero(ctx.mask) if ctx.mask is not None else 0
                logger.info(f"Simple dilated mask has {mask_pixels} non-zero pixels")

        # Step 5: Inpainting - inpaint with the refined mask
        if self._should_skip_inpainting_for_ai_renderer(config):
            logger.info("AI renderer selected: skipping inpainting and using original work image as render base.")
            ctx.img_inpainted = ctx.img_rgb
        elif ctx.mask is None or np.count_nonzero(ctx.mask) == 0:
            logger.warning("Mask is empty! Skipping inpainting.")
            ctx.img_inpainted = ctx.img_rgb
        else:
            await self._report_progress('inpainting')
            try:
                ctx.img_inpainted = await self._run_inpainting(config, ctx)
                logger.info("✓ Step 5 - Inpainting: Completed successfully")
            except Exception as e:
                logger.error(f"Error during inpainting:\n{traceback.format_exc()}")
                if not self.ignore_errors:
                    raise
                raise FileTranslationFailure("inpainting", e) from e

        # Set the result - convert to a PIL Image (the save function needs PIL)
        from PIL import Image
        if isinstance(ctx.img_inpainted, np.ndarray):
            ctx.result = Image.fromarray(ctx.img_inpainted)
        else:
            ctx.result = ctx.img_inpainted

        ctx.text_regions = []

        # Set the flag that tells _complete_translation_pipeline to skip its work
        ctx.inpaint_only_complete = True

        logger.info("=== Inpaint Only Mode Complete ===")
        await self._report_progress('inpaint-only-complete', True)
        # Not cleaned up here; the caller cleans up after saving the JSON
        return ctx

    def _save_unfiltered_textline_debug_images(self, config: Config, ctx: Context) -> None:
        """Write the debug images that show the detected text lines before OCR."""
        img_bbox_raw = np.copy(ctx.img_rgb)
        for txtln in ctx.textlines:
            det_label = getattr(txtln, 'det_label', None) or getattr(txtln, 'yolo_label', None)
            if isinstance(det_label, str) and det_label.strip().lower() == 'other':
                continue
            cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
        imwrite_unicode(self._result_path('bboxes_unfiltered.png'), cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR), logger)
        # Write the label debug image only when model-assisted merging is on, so the file is not created when the switch is off.
        # The debug image prefers the full raw detection set (including "other"), for checking the label routing.
        if bool(getattr(config.ocr, 'merge_special_require_full_wrap', True)):
            labeled_debug_textlines = getattr(ctx, 'all_detected_textlines', None) or ctx.textlines
            self._save_labeled_textline_debug_image(
                ctx.img_rgb,
                labeled_debug_textlines,
                'bboxes_unfiltered_labeled.png'
            )

    async def _translate_until_translation(self, image: Image.Image, config: Config) -> Context:
        """
        Run every step before translation (colorization, upscaling, detection, OCR, text line merging)
        """
        
        # ✅ Check the stop flag
        await asyncio.sleep(0)
        self._check_cancelled()
        
        ctx = Context()
        ctx.input = image
        ctx.result = None
        
        # Save the original input image for debugging
        if self.verbose:
            try:
                input_img = np.array(image)
                if len(input_img.shape) == 3:  # Colour image: convert to BGR order
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                result_path = self._result_path('input.png')
                imwrite_unicode(result_path, input_img, logger)
            except Exception as e:
                logger.error(f"Error saving input.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # preload and download models (not strictly necessary, remove to lazy load)
        if self.models_ttl == 0 and not self._models_loaded:
            await self._preload_models(config)

        # Start the background cleanup job once if not already started.
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())

        # ✅ Check the stop flag
        await asyncio.sleep(0)
        self._check_cancelled()

        # -- Colorization
        if config.colorizer.colorizer != Colorizer.none:
            await self._report_progress('colorizing')
            try:
                ctx.img_colorized = await self._run_colorizer(config, ctx)
            except Exception as e:
                logger.error(f"Error during colorizing:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                raise FileTranslationFailure("colorizing", e) from e
        else:
            ctx.img_colorized = ctx.input

        # --- Colorize Only Mode Check (for batch processing) ---
        if self.colorize_only:
            logger.info("Colorize Only mode (batch): Running colorization only, skipping detection, OCR, translation and rendering.")
            ctx.result = ctx.img_colorized
            ctx.text_regions = []  # Empty text regions
            self._save_editor_base_if_needed(ctx, config, ctx.img_colorized)
            await self._report_progress('colorize-only-complete', True)
            # Not cleaned up here; the caller cleans up after saving the JSON
            return ctx

        # -- Upscaling
        if config.upscale.upscale_ratio:
            # ✅ Check the stop flag
            await asyncio.sleep(0)
            self._check_cancelled()
            
            await self._report_progress('upscaling')
            try:
                ctx.upscaled = await self._run_upscaling(config, ctx)
            except Exception as e:
                logger.error(f"Error during upscaling:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                raise FileTranslationFailure("upscaling", e) from e
        else:
            ctx.upscaled = ctx.img_colorized

        if (
            hasattr(ctx.input, 'name') and ctx.input.name and
            (config.colorizer.colorizer != Colorizer.none or config.upscale.upscale_ratio)
        ):
            self._save_editor_base_if_needed(ctx, config)

        # --- Upscale Only Mode Check (for batch processing) ---
        if self.upscale_only:
            logger.info("Upscale Only mode (batch): Running upscaling only, skipping detection, OCR, translation and rendering.")
            ctx.result = ctx.upscaled
            ctx.text_regions = []  # Empty text regions
            await self._report_progress('upscale-only-complete', True)
            # Not cleaned up here; the caller cleans up after saving the JSON
            return ctx

        # --- Inpaint Only Mode Check (for batch processing) ---
        if self.inpaint_only:
            return await self._run_inpaint_only_pipeline(config, ctx)

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)
        ctx.bubble_mask = None
        
        # Check the loaded image
        if ctx.img_rgb is None or ctx.img_rgb.size == 0:
            logger.error("Failed to load image: img_rgb is empty or invalid")
            if not self.ignore_errors:
                raise Exception("Failed to load image: img_rgb is empty or invalid")
            raise FileTranslationFailure("preprocessing", RuntimeError("Failed to load image: img_rgb is empty or invalid"))
        
        if len(ctx.img_rgb.shape) < 2 or ctx.img_rgb.shape[0] == 0 or ctx.img_rgb.shape[1] == 0:
            logger.error(f"Invalid loaded image dimensions: {ctx.img_rgb.shape}")
            if not self.ignore_errors:
                raise Exception(f"Invalid loaded image dimensions: {ctx.img_rgb.shape}")
            raise FileTranslationFailure("preprocessing", RuntimeError(f"Invalid loaded image dimensions: {ctx.img_rgb.shape}"))

        # -- Detection
        await self._report_progress('detection')
        try:
            ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
        except Exception as e:
            logger.error(f"Error during detection:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            raise FileTranslationFailure("detection", e) from e

        if self.verbose and ctx.mask_raw is not None:
            # Build a heat map with a confidence colour map and a colour bar
            logger.info(f"Generating confidence heatmap for mask_raw (shape: {ctx.mask_raw.shape}, dtype: {ctx.mask_raw.dtype})")
            heatmap = self._create_confidence_heatmap(ctx.mask_raw, equalize=False)
            logger.info(f"Heatmap generated (shape: {heatmap.shape}), saving to mask_raw.png")
            imwrite_unicode(self._result_path('mask_raw.png'), heatmap, logger)

        if not ctx.textlines:
            await self._report_progress('skip-no-regions', True)
            ctx.result = ctx.upscaled
            ctx.text_regions = []  # Set to an empty list, so that an empty JSON is saved
            return await self._revert_upscale(config, ctx)

        if self.verbose:
            self._save_unfiltered_textline_debug_images(config, ctx)

        # -- OCR
        await self._report_progress('ocr')
        try:
            ctx.textlines = await self._run_ocr(config, ctx)
        except Exception as e:
            logger.error(f"Error during ocr:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            raise FileTranslationFailure("ocr", e) from e

        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            ctx.result = ctx.upscaled
            ctx.text_regions = []  # Set to an empty list, so that an empty JSON is saved
            return await self._revert_upscale(config, ctx)

        # -- Textline merge
        await self._report_progress('textline_merge')
        try:
            ctx.text_regions = await self._run_textline_merge(config, ctx)
        except Exception as e:
            logger.error(f"Error during textline_merge:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            raise FileTranslationFailure("textline_merge", e) from e

        if self.verbose and ctx.text_regions:
            show_panels = not config.force_simple_sort  # Show the panels when simple sorting is not used
            bboxes = visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions, 
                                        show_panels=show_panels, img_rgb=ctx.img_rgb, right_to_left=config.render.rtl)
            imwrite_unicode(self._result_path('bboxes.png'), bboxes, logger)

        # Apply pre-dictionary after textline merge
        self._apply_pre_dictionary_to_regions(ctx)

        # Save the current image context on ctx, for path handling during concurrent translation
        if self._current_image_context:
            ctx.image_context = self._current_image_context.copy()

        return ctx

    @staticmethod
    def _attach_high_quality_batch_data(batch: List[tuple], merged_ctx: Context) -> None:
        """Give the high-quality translators the page images and regions of the batch."""
        hq_batch_data = []
        global_text_index = 1  # Global text numbers start at 1 (the same numbering as in the prompt)
        for ctx, _ in batch:
            if ctx.text_regions:
                num_regions = len(ctx.text_regions)
                # Number the texts of this image continuously across the batch
                text_order = list(range(global_text_index, global_text_index + num_regions))
                global_text_index += num_regions

                upscaled_size = None
                # Use the size of the upscaled image (when upscaled), otherwise the size of the colorized image
                # Note: both PIL Image and numpy array have to be handled
                from PIL import Image as PILImage
                if hasattr(ctx, 'upscaled') and ctx.upscaled is not None:
                    if isinstance(ctx.upscaled, PILImage.Image):
                        w, h = ctx.upscaled.size
                        upscaled_size = (h, w)  # Convert to (height, width)
                    else:
                        upscaled_size = ctx.upscaled.shape[:2]  # numpy: (height, width)
                elif hasattr(ctx, 'img_colorized') and ctx.img_colorized is not None:
                    if isinstance(ctx.img_colorized, PILImage.Image):
                        w, h = ctx.img_colorized.size
                        upscaled_size = (h, w)
                    else:
                        upscaled_size = ctx.img_colorized.shape[:2]
                elif hasattr(ctx, 'img_rgb') and ctx.img_rgb is not None:
                    if isinstance(ctx.img_rgb, PILImage.Image):
                        w, h = ctx.img_rgb.size
                        upscaled_size = (h, w)
                    else:
                        upscaled_size = ctx.img_rgb.shape[:2]

                img_data = {
                    'image': ctx.input if hasattr(ctx, 'input') else None,
                    'text_regions': ctx.text_regions,
                    'original_texts': [region.text for region in ctx.text_regions if region.text is not None],
                    'text_order': text_order,
                    'upscaled_size': upscaled_size
                }
                hq_batch_data.append(img_data)

        if hq_batch_data:
            merged_ctx.high_quality_batch_data = hq_batch_data
            logger.debug(f"[Batch] Prepared high_quality_batch_data for {len(hq_batch_data)} images")

    async def _check_batch_target_language(self, batch: List[tuple]) -> None:
        """Check that the batch came back in the target language and translate it again when it did not."""
        # Collect the filtered regions of every page in the batch
        all_batch_regions = []
        for ctx, config in batch:
            if ctx.text_regions:
                all_batch_regions.extend(ctx.text_regions)

        # Run the target-language check for the whole batch
        batch_lang_check_result = True
        if all_batch_regions and len(all_batch_regions) > 10:
            sample_config = batch[0][1]
            logger.info(f"Starting batch-level target language check with {len(all_batch_regions)} regions...")
            batch_lang_check_result = await self._check_target_language_ratio(
                all_batch_regions,
                sample_config.translator.target_lang,
                min_ratio=0.5
            )

            if not batch_lang_check_result:
                logger.warning("Batch-level target language ratio check failed")

                # Retranslate the batch
                max_batch_retry = sample_config.translator.post_check_max_retry_attempts
                batch_retry_count = 0

                while batch_retry_count < max_batch_retry and not batch_lang_check_result:
                    batch_retry_count += 1
                    logger.warning(f"Starting batch retry {batch_retry_count}/{max_batch_retry}")

                    # Translate every region of the batch again
                    all_original_texts = []
                    region_mapping = []  # Records which ctx each text belongs to

                    for ctx_idx, (ctx, config) in enumerate(batch):
                        if ctx.text_regions:
                            for region in ctx.text_regions:
                                if hasattr(region, 'text') and region.text:
                                    all_original_texts.append(region.text)
                                    region_mapping.append((ctx_idx, region))

                    if all_original_texts:
                        try:
                            # Translate as a batch again
                            logger.info(f"Retrying translation for {len(all_original_texts)} regions...")
                            new_translations = await self._batch_translate_texts(all_original_texts, sample_config, batch[0][0])

                            # Write the new translations to their regions
                            for i, (ctx_idx, region) in enumerate(region_mapping):
                                if i < len(new_translations) and new_translations[i]:
                                    old_translation = region.translation
                                    region.translation = new_translations[i]
                                    logger.debug(f"Region {i+1} translation updated: '{old_translation}' -> '{new_translations[i]}'")

                            # Collect all regions again and check the target-language ratio
                            all_batch_regions = []
                            for ctx, config in batch:
                                if ctx.text_regions:
                                    all_batch_regions.extend(ctx.text_regions)

                            logger.info(f"Re-checking batch-level target language ratio after batch retry {batch_retry_count}...")
                            batch_lang_check_result = await self._check_target_language_ratio(
                                all_batch_regions,
                                sample_config.translator.target_lang,
                                min_ratio=0.5
                            )

                            if batch_lang_check_result:
                                logger.info("Batch-level target language check passed")
                                break
                            else:
                                logger.warning("Batch-level target language check still failed")

                        except Exception as e:
                            logger.error(f"Error during batch retry {batch_retry_count}: {e}")
                            break
                    else:
                        logger.warning("No text found for batch retry")
                        break

                if not batch_lang_check_result:
                    logger.error(f"Batch-level target language check failed after all {max_batch_retry} batch retries")
        else:
            logger.info(f"Skipping batch-level target language check: only {len(all_batch_regions)} regions (threshold: 10)")

        # One success message for all cases
        if batch_lang_check_result:
            logger.info("All translation regions passed post-translation check.")
        else:
            logger.warning("Some translation regions failed post-translation check.")

    def _filter_translated_regions(self, batch: List[tuple]) -> None:
        """Drop regions whose translation is empty, only a number or the same as the original."""
        for ctx, config in batch:
            if ctx.text_regions:
                new_text_regions = []
                for region in ctx.text_regions:
                    should_filter = False
                    filter_reason = ""
                    translation_text = _translation_plain_text(region.translation)

                    if not translation_text.strip():
                        should_filter = True
                        filter_reason = "Translation contain blank areas"
                    elif config.translator.translator != Translator.none:
                        if translation_text.isnumeric():
                            should_filter = True
                            filter_reason = "Numeric translation"
                        elif not config.translator.translator == Translator.original:
                            if self._should_filter_identical_translation(config, region):
                                should_filter = True
                                filter_reason = "Translation identical to original"

                    if should_filter:
                        if translation_text.strip():
                            logger.info(f'Filtered out: {translation_text}')
                            logger.info(f'Reason: {filter_reason}')
                    else:
                        new_text_regions.append(region)
                ctx.text_regions = new_text_regions

    async def _batch_translate_contexts(self, contexts_with_configs: List[tuple], batch_size: int) -> List[tuple]:
        """
        Run the translation step in batches, to keep memory use down
        """
        results = []
        total_contexts = len(contexts_with_configs)
        
        # Process in batches to keep memory use down
        for i in range(0, total_contexts, batch_size):
            await asyncio.sleep(0)  # Check whether the task was cancelled
            self._check_cancelled()  # Check the cancel flag
            batch = contexts_with_configs[i:i + batch_size]
            logger.info(f'Processing translation batch {i//batch_size + 1}/{(total_contexts + batch_size - 1)//batch_size}')
            
            # Collect all texts of the current batch
            all_texts = []
            batch_text_mapping = []  # Records which context and region each text belongs to
            
            for ctx_idx, (ctx, config) in enumerate(batch):
                if not ctx.text_regions:
                    continue
                    
                region_start_idx = len(all_texts)
                for region_idx, region in enumerate(ctx.text_regions):
                    # Skip None values, so later processing does not fail
                    if region.text is not None:
                        all_texts.append(region.text)
                        batch_text_mapping.append((ctx_idx, region_idx))
                
            if not all_texts:
                # The current batch has no text to translate
                results.extend(batch)
                continue
                
            # Translate as a batch
            merged_ctx = None
            try:
                await self._report_progress('translating')
                # Translate with the first configuration (the batch is assumed to share one)
                sample_config = batch[0][1] if batch else None
                if sample_config:
                    # ✅ Merge the text_regions of every image in the batch (for AI line breaking)
                    # Create a temporary ctx, so the original ones are not affected
                    merged_ctx = Context()
                    merged_ctx.config = sample_config  # Copy the configuration
                    
                    all_regions = []
                    for ctx, _ in batch:
                        if ctx.text_regions:
                            all_regions.extend(ctx.text_regions)
                    merged_ctx.text_regions = all_regions
                    
                    # Copy the other attributes needed from the first ctx
                    first_ctx = batch[0][0]
                    if hasattr(first_ctx, 'from_lang'):
                        merged_ctx.from_lang = first_ctx.from_lang
                    
                    # ✅ Load the AI line-breaking prompt and the custom HQ prompt
                    merged_ctx = await self._load_and_prepare_prompts(sample_config, merged_ctx)
                    
                    logger.debug(f"[Batch] Merged {len(all_regions)} text regions from {len(batch)} images for AI line breaking")
                    
                    # Batch translation - pass the merged context (used for AI line breaking only)
                    batch_contexts = [ctx for ctx, config in batch]
                    
                    # History already contains every completed non-empty page before this batch.
                    page_index = len(self.all_page_translations)
                    
                    # Prepare batch_original_texts (context for concurrent mode)
                    batch_original_texts = []
                    for ctx, _ in batch:
                        if ctx.text_regions:
                            image_data = {
                                'original_texts': [region.text for region in ctx.text_regions if region.text is not None]
                            }
                            batch_original_texts.append(image_data)
                    
                    # ✅ Prepare high_quality_batch_data for the HQ translators (images and text_regions)
                    # The HQ translators need it to enter high-quality batch mode, and the AI line-breaking check depends on it
                    if sample_config.translator.translator in [Translator.openai_hq, Translator.gemini_hq]:
                        self._attach_high_quality_batch_data(batch, merged_ctx)
                    
                    translated_texts = await self._batch_translate_texts(
                        all_texts, 
                        sample_config, 
                        merged_ctx, 
                        batch_contexts,
                        page_index=page_index,
                        batch_index=0,  # In batch processing the batch index of the first image is 0
                        batch_original_texts=batch_original_texts
                    )
                else:
                    translated_texts = all_texts  # Keep the original text when it cannot be translated
                    
                # Hand the translations back to their contexts
                text_idx = 0
                for ctx_idx, (ctx, config) in enumerate(batch):
                    if not ctx.text_regions:  # Check whether text_regions is None or empty
                        continue
                    for region_idx, region in enumerate(ctx.text_regions):
                        if text_idx < len(translated_texts):
                            region.translation = translated_texts[text_idx]
                            region.target_lang = config.translator.target_lang
                            region._alignment = config.render.alignment
                            region._direction = config.render.direction
                            text_idx += 1
                        
                # Apply the post-processing (bracket correction, filtering and so on)
                for ctx, config in batch:
                    if ctx.text_regions:
                        ctx.text_regions = await self._apply_post_translation_processing(ctx, config)
                
                # ✅ Save this batch's translations to all_page_translations right away, as context for the next batch
                for ctx, config in batch:
                    if ctx.text_regions:
                        page_entries = self._build_page_context_entries(ctx)
                        self.all_page_translations.append(page_entries)
                        logger.debug(f"[Batch Context] Saved {len(page_entries)} translations for next batch context")
                        
                # Prune history to prevent memory leak
                self._prune_context_history()
                        
                # Target-language check for the whole batch
                if batch and batch[0][1].translator.enable_post_translation_check:
                    await self._check_batch_target_language(batch)
                        
                # Filtering (simplified; the main filter conditions are kept)
                self._filter_translated_regions(batch)
                        
                results.extend(batch)
                
            except Exception as e:
                # Get the exception text safely
                try:
                    error_msg = str(e)
                except Exception as ignored_error:
                    note_ignored_error(ignored_error, "manga_translator/manga_translator.py:MangaTranslator._batch_translate_contexts")
                    error_msg = f"Unable to retrieve exception details (exception type: {type(e).__name__})"
                
                logger.error(f"Error in batch translation: {error_msg}")
                logger.error(traceback.format_exc())
                if not self.ignore_errors:
                    raise
                # On error, mark the files of this batch as failed; the original text is no longer used as a fallback
                for ctx, config in batch:
                    self._mark_context_failure(ctx, FileTranslationFailure("translation", e), stage='translation')
                    ctx.text_regions = []
                results.extend(batch)
            finally:
                # Free memory once the translation batch is done
                # Clear the temporary data in merged_ctx and batch
                if merged_ctx:
                    merged_ctx.text_regions = None
                    merged_ctx = None
                batch = None
                self._cleanup_gpu_memory(aggressive=True)

        return results

    async def _batch_translate_texts(self, texts: List[str], config: Config, ctx: Context, batch_contexts: List[Context] = None, page_index: int = None, batch_index: int = None, batch_original_texts: List[dict] = None) -> List[str]:
        """
        Translate a list of texts as a batch, through the existing translator interface

        Args:
            texts: the list of texts to translate
            config: the configuration object
            ctx: the context object
            batch_contexts: list of the batch contexts
            page_index: index of the current page, used to compute the context in concurrent mode
            batch_index: index of the current page in the batch
            batch_original_texts: original text data of the current batch
        """
        if config.translator.translator == Translator.none:
            return ["" for _ in texts]

        if config.translator.translator in (Translator.codex, Translator.claude):
            # Both run a signed-in command line tool instead of an API client.
            if config.translator.translator == Translator.claude:
                from .translators.claude_cli import ClaudeCLITranslator
                translator = ClaudeCLITranslator()
            else:
                from .translators.codex_cli import CodexCLITranslator
                translator = CodexCLITranslator()
            translator.parse_args(config)
            translator.set_cancel_check_callback(self._cancel_check_callback)
            ctx.config = config
            return await translator.translate(
                getattr(ctx, 'from_lang', None) or 'auto',
                config.translator.target_lang, texts, ctx=ctx,
            )



        # The OpenAI, Gemini and high-quality translators need the context handled
        if config.translator.translator in [Translator.openai, Translator.gemini, Translator.openai_hq, Translator.gemini_hq]:
            if config.translator.translator == Translator.openai:
                from .translators.openai import OpenAITranslator
                translator = OpenAITranslator()
            elif config.translator.translator == Translator.gemini:
                from .translators.gemini import GeminiTranslator
                translator = GeminiTranslator()
            elif config.translator.translator == Translator.openai_hq:
                from .translators.openai_hq import OpenAIHighQualityTranslator
                translator = OpenAIHighQualityTranslator()
            elif config.translator.translator == Translator.gemini_hq:
                from .translators.gemini_hq import GeminiHighQualityTranslator
                translator = GeminiHighQualityTranslator()

            translator.parse_args(config)
            # Note: -1 means retry without limit and is a valid value
            # The retry count always comes from cli.attempts
            
            # Pass the cancel-check callback to the translator
            if self._cancel_check_callback:
                translator.set_cancel_check_callback(self._cancel_check_callback)

            # Build and set the text context for every translator (HQ translators included)
            done_pages = self.all_page_translations
            if self.context_size > 0 and done_pages:
                pages_expected = min(self.context_size, len(done_pages))
                non_empty_pages = [
                    page for page in done_pages
                    if self._page_has_context_entries(page)
                ]
                pages_used = min(self.context_size, len(non_empty_pages))
                skipped = pages_expected - pages_used
            else:
                pages_used = skipped = 0

            if self.context_size > 0:
                logger.info(f"Context-aware translation enabled with {self.context_size} pages of history")

            # Build the context (simplified; context within the batch is not used)
            prev_ctx = self._build_prev_context(
                use_original_text=False,  # Always use the translations as context
                current_page_index=page_index,
                batch_index=None,  # Context within the batch is not used
                batch_original_texts=None
            )
            translator.set_prev_context(prev_ctx)

            if pages_used > 0:
                context_count = prev_ctx.count('"translation"')
                logger.info(f"Carrying {pages_used} pages of context, {context_count} sentences as translation reference")
            if skipped > 0:
                logger.warning(f"Skipped {skipped} pages with no sentences")


            # Attach config to ctx for the translator (for example for AI line breaking)
            ctx.config = config
            
            # openai_hq, gemini_hq and similar need the ctx argument
            if config.translator.translator in [Translator.openai_hq, Translator.gemini_hq]:
                # Every translator that needs context gets ctx here
                return await translator._translate(
                    ctx.from_lang,
                    config.translator.target_lang,
                    texts,
                    ctx
                )
            else:
                # Plain OpenAI and Gemini need the ctx argument (for AI line breaking)
                return await translator._translate(
                    ctx.from_lang,
                    config.translator.target_lang,
                    texts,
                    ctx
                )

        else:
            # Use the general translation dispatcher
            return await dispatch_translation(
                config.translator.translator_gen,
                texts,
                config,
                False,  # use_mtpe removed
                ctx,
                'cpu' if self._gpu_limited_memory else self.device
            )
    
    def _should_filter_identical_translation(self, config: Config, region) -> bool:
        """Keep identical text when no_text_lang_skip is enabled."""
        if getattr(config.translator, 'no_text_lang_skip', False):
            return False
        return str(region.text or '').lower().strip() == _translation_plain_text(region.translation).lower().strip()
            
    async def _apply_post_translation_processing(self, ctx: Context, config: Config) -> List:
        """
        Apply the post-translation processing (bracket correction, filtering and so on)
        """
        # Check whether text_regions is None or empty
        if not ctx.text_regions:
            return []
            
        check_items = [
            # Round brackets
            ["(", "（", "「", "【"],
            ["（", "(", "「", "【"],
            [")", "）", "」", "】"],
            ["）", ")", "」", "】"],
            
            # Square brackets
            ["[", "［", "【", "「"],
            ["［", "[", "【", "「"],
            ["]", "］", "】", "」"],
            ["］", "]", "】", "」"],
            
            # Quotation marks
            ["「", "“", "‘", "『", "【"],
            ["」", "”", "’", "』", "】"],
            ["『", "“", "‘", "「", "【"],
            ["』", "”", "’", "」", "】"],
            
            # Added: lenticular brackets
            ["【", "(", "（", "「", "『", "["],
            ["】", ")", "）", "」", "』", "]"],
        ]

        # Unified rendering: quotes and brackets are not replaced after translation; the rendering layer handles them.

        for region in ctx.text_regions:
            # The translation property always returns str, so no isinstance guard is needed
            if region.text and region.translation:
                # Quotation mark handling
                if '『' in region.text and '』' in region.text:
                    quote_type = '『』'
                elif '「' in region.text and '」' in region.text:
                    quote_type = '「」'
                elif '【' in region.text and '】' in region.text: 
                    quote_type = '【】'
                else:
                    quote_type = None
                
                if quote_type:
                    src_quote_count = region.text.count(quote_type[0])
                    dst_dquote_count = region.translation.count('"')
                    dst_fwquote_count = region.translation.count('＂')
                    
                    if (src_quote_count > 0 and
                        (src_quote_count == dst_dquote_count or src_quote_count == dst_fwquote_count) and
                        not region.translation.isascii()):
                        
                        if quote_type == '「」':
                            region.translation = re.sub(r'"([^"]*)"', r'「\1」', region.translation)
                        elif quote_type == '『』':
                            region.translation = re.sub(r'"([^"]*)"', r'『\1』', region.translation)
                        elif quote_type == '【】':  
                            region.translation = re.sub(r'"([^"]*)"', r'【\1】', region.translation)

                # Bracket correction
                for v in check_items:
                    num_src_std = region.text.count(v[0])
                    num_src_var = sum(region.text.count(t) for t in v[1:])
                    num_dst_std = region.translation.count(v[0])
                    num_dst_var = sum(region.translation.count(t) for t in v[1:])
                    
                    if (num_src_std > 0 and
                        num_src_std != num_src_var and
                        num_src_std == num_dst_std + num_dst_var):
                        for t in v[1:]:
                            region.translation = region.translation.replace(t, v[0])

                # Unified rendering: nothing is replaced here.

        # Note: saving the translation moved to the end of the translate method, so the final result is what gets saved

        # Simplified/Traditional Chinese conversion (OpenCC)
        if config.translator.convert_to_traditional or config.translator.convert_to_simplified:
            try:
                import opencc
                if config.translator.convert_to_traditional:
                    _opencc_converter = opencc.OpenCC('s2twp')
                    logger.info("Applying OpenCC: Simplified → Traditional (s2twp)")
                else:
                    _opencc_converter = opencc.OpenCC('t2s')
                    logger.info("Applying OpenCC: Traditional → Simplified (t2s)")
                for region in ctx.text_regions:
                    if region.translation:
                        original = region.translation
                        region.translation = _opencc_converter.convert(region.translation)
                        if original != region.translation:
                            logger.debug(f"OpenCC: {original} => {region.translation}")
            except ImportError:
                logger.warning("opencc-python-reimplemented not installed, skipping Chinese conversion")

        # Apply the post-dictionary
        post_dict = load_dictionary(self.post_dict)
        post_replacements = []  
        for region in ctx.text_regions:
            original = region.translation
            region.translation = apply_dictionary(region.translation, post_dict)
            if original != region.translation:  
                post_replacements.append(f"{original} => {region.translation}")  

        if post_replacements:
            logger.info("Post-translation replacements:")
            for replacement in post_replacements:
                logger.info(replacement)
        else:
            logger.info("No post-translation replacements made.")

        # Note: the text replacement rules (text_replacements.yaml) now run inside dispatch() of the rendering module.

        # Hallucination check for a single region
        failed_regions = []
        if config.translator.enable_post_translation_check:
            logger.info("Starting post-translation check...")
            
            # Hallucination check at the level of a single region
            for region in ctx.text_regions:
                if _has_translation_text(region.translation):
                    # Only repeated-content hallucinations are checked
                    if await self._check_repetition_hallucination(
                        _translation_plain_text(region.translation),
                        config.translator.post_check_repetition_threshold,
                        silent=False
                    ):
                        failed_regions.append(region)
            
            # Retry the regions that failed
            if failed_regions:
                logger.warning(f"Found {len(failed_regions)} regions that failed repetition check, starting retry...")
                for region in failed_regions:
                    try:
                        logger.info(f"Retrying translation for region with text: '{region.text}'")
                        new_translation = await self._retry_translation_with_validation(region, config, ctx)
                        if new_translation:
                            old_translation = region.translation
                            region.translation = new_translation
                            logger.info(f"Region retry successful: '{old_translation}' -> '{new_translation}'")
                        else:
                            logger.warning(f"Region retry failed, keeping original translation: '{region.translation}'")
                            break
                    except Exception as e:
                        logger.error(f"Error during region retry: {e}")
                        break

        for region in ctx.text_regions:
            if region.translation:
                region.translation = remove_trailing_period_if_needed(
                    region.text,
                    region.translation,
                    bool(getattr(config.translator, 'remove_trailing_period', False)),
                )
                region.translation = normalize_thai_punctuation(
                    region.translation,
                    bool(getattr(config.translator, 'normalize_thai_punctuation', False))
                    and is_thai_language(getattr(config.translator, 'target_lang', '')),
                )
        
        return ctx.text_regions

    async def _complete_translation_pipeline(self, ctx: Context, config: Config) -> Context:
        """
        Finish the steps after translation (mask refinement, inpainting, rendering)
        """
        await self._report_progress('after-translating')

        # Inpaint Only Mode: Skip pipeline, ctx.result already set
        if hasattr(ctx, 'inpaint_only_complete') and ctx.inpaint_only_complete:
            logger.info("Skipping _complete_translation_pipeline (inpaint only mode already complete)")
            return ctx

        # Colorize Only Mode: Skip validation, ctx.result should already be set
        if self.colorize_only:
            return ctx

        if not ctx.text_regions:
            await self._report_progress('error-translating', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)
        elif ctx.text_regions == 'cancel':
            await self._report_progress('cancelled', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- Mask refinement
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
            except Exception as e:
                logger.error(f"Error during mask-generation:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise 
                raise FileTranslationFailure("mask-generation", e) from e

        if self.verbose and ctx.mask is not None:
            try:
                inpaint_input_img = await dispatch_inpainting(Inpainter.none, ctx.img_rgb, ctx.mask, config.inpainter,config.inpainter.inpainting_size,
                                                              self.device, self.verbose)
                
                # Save inpaint_input.png
                inpaint_input_path = self._result_path('inpaint_input.png')
                imwrite_unicode(inpaint_input_path, cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR), logger)
                
                # Save mask_final.png
                mask_final_path = self._result_path('mask_final.png')
                imwrite_unicode(mask_final_path, ctx.mask, logger)
            except Exception as e:
                logger.error(f"Error saving debug images (inpaint_input.png, mask_final.png): {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # -- Inpainting
        generated_inpainted_for_save = False
        if self._should_skip_inpainting_for_ai_renderer(config):
            logger.info("AI renderer selected: skipping inpainting and using original work image as render base.")
            ctx.img_inpainted = ctx.img_rgb
        else:
            await self._report_progress('inpainting')
            try:
                ctx.img_inpainted = await self._run_inpainting(config, ctx)
                generated_inpainted_for_save = True
                
                # ✅ Force GC and GPU clean-up after inpainting
                self._cleanup_gpu_memory()

            except Exception as e:
                logger.error(f"Error during inpainting:\n{traceback.format_exc()}")
                if not self.ignore_errors:
                    raise
                raise FileTranslationFailure("inpainting", e) from e

        if self.verbose:
            try:
                inpainted_path = self._result_path('inpainted.png')
                imwrite_unicode(inpainted_path, cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), logger)
            except Exception as e:
                logger.error(f"Error saving inpainted.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
        # Save the inpainted image to the new folder structure (for the editable image feature)
        # Saved only when "editable image" is on
        if (
            generated_inpainted_for_save
            and self.save_text
            and hasattr(ctx, 'image_name')
            and ctx.image_name
            and ctx.img_inpainted is not None
        ):
            self._save_inpainted_image(
                ctx.image_name,
                ctx.img_inpainted,
            )

        # -- Rendering
        await self._report_progress('rendering')

        # Send the folder information right after the rendering state, so the frontend can check final.png precisely
        if hasattr(self, '_progress_hooks') and self._current_image_context:
            folder_name = self._current_image_context['subfolder']
            # Send a specially formatted message the frontend can parse
            await self._report_progress(f'rendering_folder:{folder_name}')

        try:
            ctx.img_rendered = await self._run_text_rendering(config, ctx)
        except Exception as e:
            logger.error(f"Error during rendering:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            raise FileTranslationFailure("rendering", e) from e

        await self._report_progress('finished', True)
        ctx.result = dump_image(
            ctx.input,
            ctx.img_rendered,
            ctx.img_alpha,
            mask=ctx.mask,
            render_alpha=getattr(ctx, 'img_render_alpha', None),
        )
        
        # Keep the debug folder information on the Context (for cache access in web mode)
        if self.verbose:
            ctx.debug_folder = self._get_image_subfolder()

        return await self._revert_upscale(config, ctx)
    
    async def _check_repetition_hallucination(self, text: str, threshold: int = 5, silent: bool = False) -> bool:
        """
        Check if the text contains repetitive content (model hallucination)
        """
        if not text or len(text.strip()) < threshold:
            return False
            
        # Check for repetition at character level
        consecutive_count = 1
        prev_char = None
        
        for char in text:
            if char == prev_char:
                consecutive_count += 1
                if consecutive_count >= threshold:
                    if not silent:
                        logger.warning(f'Detected character repetition hallucination: "{text}" - repeated character: "{char}", consecutive count: {consecutive_count}')
                    return True
            else:
                consecutive_count = 1
            prev_char = char
        
        # Check for repetition at word level (Chinese is split by character, other languages by spaces)
        segments = re.findall(r'[\u4e00-\u9fff]|\S+', text)
        
        if len(segments) >= threshold:
            consecutive_segments = 1
            prev_segment = None
            
            for segment in segments:
                if segment == prev_segment:
                    consecutive_segments += 1
                    if consecutive_segments >= threshold:
                        if not silent:
                            logger.warning(f'Detected word repetition hallucination: "{text}" - repeated segment: "{segment}", consecutive count: {consecutive_segments}')
                        return True
                else:
                    consecutive_segments = 1
                prev_segment = segment
        
        # Check for repetition at phrase level
        words = text.split()
        if len(words) >= threshold * 2:
            for i in range(len(words) - threshold + 1):
                phrase = ' '.join(words[i:i + threshold//2])
                remaining_text = ' '.join(words[i + threshold//2:])
                if phrase in remaining_text:
                    phrase_count = text.count(phrase)
                    if phrase_count >= 3:  # Lower threshold for phrase repetition
                        if not silent:
                            logger.warning(f'Detected phrase repetition hallucination: "{text}" - repeated phrase: "{phrase}", occurrence count: {phrase_count}')
                        return True
                        
        return False

    async def _check_target_language_ratio(self, text_regions: List, target_lang: str, min_ratio: float = 0.5) -> bool:
        """
        Check if the target language ratio meets the requirement by detecting the merged translation text.
        The language is detected with py3langid.

        Args:
            text_regions: list of text regions
            target_lang: target language code
            min_ratio: minimum target language ratio (unused by the new logic; kept for compatibility)

        Returns:
            bool: True when the check passes, False when it does not
        """
        if not text_regions or len(text_regions) <= 10:
            # Skip this check when there are 10 regions or fewer
            return True
            
        # Join all translated texts
        all_translations = []
        for region in text_regions:
            translation = _translation_plain_text(getattr(region, 'translation', ''))
            if translation.strip():
                all_translations.append(translation.strip())
        
        if not all_translations:
            logger.debug('No valid translation texts for language ratio check')
            return True
            
        # Join all translations into one text for the detection
        merged_text = ''.join(all_translations)
        
        # logger.info(f'Target language check - Merged text preview (first 200 chars): "{merged_text[:200]}"')
        # logger.info(f'Target language check - Total merged text length: {len(merged_text)} characters')
        # logger.info(f'Target language check - Number of regions: {len(all_translations)}')
        
        # Detect the language with py3langid
        try:
            detected_lang, confidence = langid.classify(merged_text)
            detected_language = ISO_639_1_TO_VALID_LANGUAGES.get(detected_lang, 'UNKNOWN')
            if detected_language != 'UNKNOWN':
                detected_language = detected_language.upper()
            
            # logger.info(f'Target language check - py3langid result: "{detected_lang}" -> "{detected_language}" (confidence: {confidence:.3f})')
        except Exception as e:
            logger.debug(f'py3langid failed for merged text: {e}')
            detected_language = 'UNKNOWN'
        
        # Check whether the detected language is the target language
        is_target_lang = (detected_language == target_lang.upper())
        
        # logger.info(f'Target language check: Detected language "{detected_language}" using py3langid (confidence: {confidence:.3f})')
        # logger.info(f'Target language check: Target is "{target_lang.upper()}"')
        # logger.info(f'Target language check result: {"PASSED" if is_target_lang else "FAILED"}')
        
        return is_target_lang

    async def _validate_translation(self, original_text: str, translation: str, target_lang: str, config, ctx: Context = None, silent: bool = False, page_lang_check_result: bool = None) -> bool:
        """
        Validate translation quality (includes target language ratio check and hallucination detection)

        Args:
            page_lang_check_result: result of the page-level target language check; when None the check is run, otherwise the given result is used
        """
        if not config.translator.enable_post_translation_check:
            return True
            
        translation = _translation_plain_text(translation)
        if not translation.strip():
            return True
        
        # 1. Target-language ratio check (page level)
        if page_lang_check_result is None and ctx and ctx.text_regions and len(ctx.text_regions) > 10:
            # Run the page-level target-language check
            page_lang_check_result = await self._check_target_language_ratio(
                ctx.text_regions,
                target_lang,
                min_ratio=0.5
            )
            
        # When the page-level check fails, return the failure at once
        if page_lang_check_result is False:
            if not silent:
                logger.debug("Target language ratio check failed for this region")
            return False
        
        # 2. Check for repeated-content hallucinations (region level)
        if await self._check_repetition_hallucination(
            translation, 
            config.translator.post_check_repetition_threshold,
            silent
        ):
            return False
                
        return True

    async def _retry_translation_with_validation(self, region, config: Config, ctx: Context) -> str:
        """
        Retry translation with validation
        """
        original_translation = region.translation
        max_attempts = config.translator.post_check_max_retry_attempts
        
        for attempt in range(max_attempts):
            # Validate the current translation - during retries only the single region is checked (hallucination), not the page
            is_valid = await self._validate_translation(
                region.text, 
                region.translation, 
                config.translator.target_lang,
                config,
                ctx=None,  # ctx is not passed, to avoid the page-level check
                silent=True,  # Log output is off during retries
                page_lang_check_result=True  # Pass True to skip the page-level check and only run the region-level one
            )
            
            if is_valid:
                if attempt > 0:
                    logger.info(f'Post-translation check passed (Attempt {attempt + 1}/{max_attempts}): "{region.translation}"')
                return region.translation
            
            # Translate again unless this was the last attempt
            if attempt < max_attempts - 1:
                logger.warning(f'Post-translation check failed (Attempt {attempt + 1}/{max_attempts}), re-translating: "{region.text}"')
                
                try:
                    # Translate this text region again on its own
                    if config.translator.translator != Translator.none:
                        from .translators import dispatch
                        retranslated = await dispatch(
                            config.translator.translator_gen,
                            [region.text],
                            config.translator,
                            False,  # use_mtpe removed
                            ctx,
                            'cpu' if self._gpu_limited_memory else self.device
                        )
                        if retranslated:
                            region.translation = retranslated[0]
                            
                            # Apply the formatting
                            if config.render.uppercase:
                                region.translation = region.translation.upper()
                            elif config.render.lowercase:
                                region.translation = region.translation.lower()
                                
                            logger.info(f'Re-translation finished: "{region.text}" -> "{region.translation}"')
                        else:
                            logger.warning(f'Re-translation failed, keeping original translation: "{original_translation}"')
                            region.translation = original_translation
                            break
                    else:
                        logger.warning('Translator is none, cannot re-translate.')
                        break
                        
                except Exception as e:
                    logger.error(f'Error during re-translation: {e}')
                    region.translation = original_translation
                    break
            else:
                logger.warning(f'Post-translation check failed, maximum retry attempts ({max_attempts}) reached, keeping original translation: "{original_translation}"')
                region.translation = original_translation
        
        return region.translation

    def _update_translation_map(self, source_path: str, translated_path: str):
        """Create or update translation_map.json in the output folder"""
        try:
            output_dir = os.path.dirname(translated_path)
            map_path = os.path.join(output_dir, 'translation_map.json')
            
            # Normalise the path for consistency
            source_path_norm = os.path.normpath(source_path)
            translated_path_norm = os.path.normpath(translated_path)

            translation_map = {}
            if os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    try:
                        translation_map = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode {map_path}, creating a new one.")
            
            # Use the path of the translated image as the key, so it is unique
            translation_map[translated_path_norm] = source_path_norm
            
            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump(translation_map, f, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.error(f"Failed to update translation map: {e}")

    async def _translate_batch_replace_translation(self, images_with_configs: List[tuple], save_info: dict = None, global_offset: int = 0, global_total: int = None) -> List[Context]:
        """
        Replace-translation mode: extract the OCR result from the translated image and apply it to the raw image

        Flow:
        1. Run detection + OCR on the raw image and filter out low-confidence regions
        2. Find the matching translated image and run detection + OCR on it
        3. Match the regions (taking size scaling into account)
        4. Inpaint and render with the matched regions

        Args:
            images_with_configs: List of (image, config) tuples
            save_info: the save settings
            global_offset: global offset
            global_total: global total number of images
        """
        from .utils.replace_translation import translate_batch_replace_translation
        return await translate_batch_replace_translation(self, images_with_configs, save_info, global_offset, global_total)

    async def _translate_batch_high_quality(
        self,
        images_with_configs: List[tuple],
        save_info: dict = None,
        global_offset: int = 0,
        global_total: int = None,
        batch_size: int = None,
        skipped_count: int = 0,
    ) -> List[Context]:
        """
        High-quality translation mode: rolling batches, each of which runs the whole flow of preprocessing, translation and rendering on its own.
        When save_info is given, each batch is saved right after it is processed.

        Args:
            images_with_configs: List of (image, config) tuples
            save_info: the save settings
            global_offset: global offset, used to show the correct image number
            global_total: global total number of images, used to show the correct total number of batches
            batch_size: batch size
        """
        # batch_size=1 is a valid request to translate HQ pages one at a time.
        resolved_batch_size = max(1, batch_size if batch_size is not None else self.batch_size)
        logger.info(f"Starting high quality translation in rolling batch mode with batch size: {resolved_batch_size}")
        results = []
        
        # Use the global total, when given, to work out the number of batches; otherwise the image count of this batch
        display_total = global_total if global_total is not None else len(images_with_configs)
        

        async def report_completed_image_progress():
            if display_total <= 0:
                return
            completed = min(global_offset + len(results), display_total)
            failed_count = sum(1 for ctx in results if getattr(ctx, 'translation_error', None))
            current_skipped = skipped_count + sum(1 for ctx in results if getattr(ctx, 'skipped', False))
            await self._report_progress(
                f"batch:{completed}:{completed}:{display_total}:{failed_count}:{current_skipped}"
            )

        batch_slices = slice_batch_indices(
            images_with_configs,
            resolved_batch_size,
            self._resume_context_pages,
            self._resume_context_order,
        )
        total_batches = len(batch_slices)

        for batch_num, (batch_start, batch_end) in enumerate(batch_slices, start=1):
            # Check whether the task was cancelled
            await asyncio.sleep(0)
            self._check_cancelled()  # Check the cancel flag

            current_batch_items = images_with_configs[batch_start:batch_end]
            if current_batch_items:
                self._append_resume_context_before(input_path(current_batch_items[0]))

            # Global image number (takes the offset of the frontend's batched loading into account)
            global_batch_start = global_offset + batch_start + 1
            global_batch_end = global_offset + batch_end
            progress_state = f"batch:{global_batch_start}:{global_batch_end}:{display_total}:0:{skipped_count}"
            
            logger.info(f"Processing rolling batch {batch_num}/{total_batches} (images {global_batch_start}-{global_batch_end})")

            current_batch_images, load_error_contexts = self._materialize_batch_inputs(current_batch_items)
            if load_error_contexts:
                results.extend(load_error_contexts)
                await report_completed_image_progress()
            if not current_batch_images:
                await self._report_progress(progress_state)
                continue

            # Stage one: preprocess the current batch
            preprocessed_contexts = []
            for i, (image, config) in enumerate(current_batch_images):
                # Check whether the task was cancelled
                await asyncio.sleep(0)
                self._check_cancelled()  # Check the cancel flag
                try:
                    self._set_image_context(config, image)
                    # ✅ Save the context so the rendering stage can reuse it and no second folder is created
                    from .utils.generic import get_image_md5
                    image_md5 = get_image_md5(image)
                    self._save_current_image_context(image_md5)
                    ctx = await self._translate_until_translation(image, config)
                    if hasattr(image, 'name'):
                        ctx.image_name = image.name
                    preprocessed_contexts.append((ctx, config))
                except Exception as e:
                    logger.error(f"Error pre-processing image {i+1} in batch: {e}", exc_info=True)
                    ctx = self._build_stage_error_context(image, e, config, stage='preprocessing')
                    preprocessed_contexts.append((ctx, config))

            # Stage two: translate the current batch
            batch_data = []
            global_text_index = 1  # Global text numbers start at 1 (the same numbering as in the prompt)
            for ctx, config in preprocessed_contexts:
                num_regions = len(ctx.text_regions) if ctx.text_regions else 0
                # Number the texts of this image continuously across the batch
                text_order = list(range(global_text_index, global_text_index + num_regions))
                global_text_index += num_regions
                
                # Size after upscaling (for coordinate conversion)
                upscaled_size = None
                if hasattr(ctx, 'img_rgb') and ctx.img_rgb is not None:
                    upscaled_size = ctx.img_rgb.shape[:2]  # (height, width)
                
                image_data = {
                    'image': ctx.input,
                    'text_regions': ctx.text_regions if ctx.text_regions else [],
                    'original_texts': [region.text for region in ctx.text_regions if region.text is not None] if ctx.text_regions else [],
                    'text_order': text_order,
                    'upscaled_size': upscaled_size
                }
                batch_data.append(image_data)

            if any(data['original_texts'] for data in batch_data):
                try:
                    sample_config = preprocessed_contexts[0][1] if preprocessed_contexts else None
                    if sample_config:
                        # ✅ Create a new Context for enhanced_ctx, so the context of the first image is not polluted
                        enhanced_ctx = Context()
                        # Copy the attributes needed from the first image
                        if preprocessed_contexts:
                            first_ctx = preprocessed_contexts[0][0]
                            if hasattr(first_ctx, 'input'):
                                enhanced_ctx.input = first_ctx.input
                            if hasattr(first_ctx, 'img_rgb'):
                                enhanced_ctx.img_rgb = first_ctx.img_rgb
                        
                        enhanced_ctx.high_quality_batch_data = batch_data

                        # ✅ Merge the text_regions of every page into enhanced_ctx (for AI line breaking)
                        all_regions = []
                        for ctx, _ in preprocessed_contexts:
                            if ctx.text_regions:
                                all_regions.extend(ctx.text_regions)
                        enhanced_ctx.text_regions = all_regions
                        logger.debug(f"[HQ Batch] Merged {len(all_regions)} text regions from {len(preprocessed_contexts)} pages")

                        # Centralized prompt loading logic
                        enhanced_ctx = await self._load_and_prepare_prompts(sample_config, enhanced_ctx)
                        
                        all_texts = [text for data in batch_data for text in data['original_texts']]
                        text_mapping = [(img_idx, region_idx) for img_idx, data in enumerate(batch_data) for region_idx, _ in enumerate(data['original_texts'])]
                        
                        logger.info(f"Sending batch data with {len(preprocessed_contexts)} images, {len(all_texts)} text regions to high quality translator")
                        
                        # History already contains every completed non-empty page before this batch.
                        page_index = len(self.all_page_translations)
                        
                        # High-quality translation mode: translate as a batch
                        translated_texts = await self._batch_translate_texts(
                            all_texts, 
                            sample_config, 
                            enhanced_ctx,
                            page_index=page_index,
                            batch_index=None,  # Context within the batch is not used
                            batch_original_texts=None
                        )
                        
                        for text_idx, (img_idx, region_idx) in enumerate(text_mapping):
                            if text_idx < len(translated_texts):
                                ctx, config = preprocessed_contexts[img_idx]
                                if ctx.text_regions and region_idx < len(ctx.text_regions):
                                    region = ctx.text_regions[region_idx]
                                    region.translation = translated_texts[text_idx]
                                    region.target_lang = config.translator.target_lang
                                    region._alignment = config.render.alignment
                                    region._direction = config.render.direction
                        
                        for ctx, config in preprocessed_contexts:
                            if ctx.text_regions:
                                ctx.text_regions = await self._apply_post_translation_processing(ctx, config)
                        
                        # ✅ Save this batch's translations to all_page_translations right away, as context for the next batch
                        for ctx, config in preprocessed_contexts:
                            if ctx.text_regions:
                                # Keep the translations and the original region count, for the history context and AI line breaking
                                page_entries = self._build_page_context_entries(ctx)
                                self.all_page_translations.append(page_entries)
                                logger.debug(f"[HQ Batch Context] Saved {len(page_entries)} translations for next batch context")
                                
                                # Keep the original text (context for concurrent mode)
                                page_original_texts = {i: (r.text_raw if hasattr(r, "text_raw") else r.text)
                                                      for i, r in enumerate(ctx.text_regions)}
                                self._original_page_texts.append(page_original_texts)
                                logger.debug(f"[HQ Batch Context] Saved {len(page_original_texts)} original texts for next batch context")
                                
                                # Prune history to prevent memory leak
                                self._prune_context_history()
                                
                except Exception as e:
                    logger.error(f"Error in high quality batch translation: {e}")
                    if not self.ignore_errors:
                        raise
                    for ctx, config in preprocessed_contexts:
                        self._mark_context_failure(ctx, FileTranslationFailure("translation", e), stage='translation')
                        ctx.text_regions = []
            # --- NEW: Handle Generate and Export for High-Quality Mode ---
            if self.generate_and_export:
                logger.info("'Generate and Export' mode enabled. Skipping rendering.")
                for ctx, config in preprocessed_contexts:
                    if getattr(ctx, 'translation_error', None):
                        results.append(ctx)
                        await report_completed_image_progress()
                        continue
                    await self._handle_generate_and_export(ctx, config)

                    # ✅ Mark as successful (the translation was exported)
                    ctx.success = True
                    results.append(ctx)
                    await report_completed_image_progress()
                
                # ✅ Free memory as soon as the batch is done
                self._cleanup_batch_memory(
                    current_batch_images=current_batch_images,
                    preprocessed_contexts=preprocessed_contexts,
                    keep_results=True
                )
                await self._report_progress(progress_state)
                
                continue # BUG FIX: Continue to the next batch instead of returning

            # Stage three: render and save the current batch
            for ctx, config in preprocessed_contexts:
                # Check whether the task was cancelled
                await asyncio.sleep(0)
                self._check_cancelled()  # Check the cancel flag
                if getattr(ctx, 'translation_error', None):
                    results.append(ctx)
                    await report_completed_image_progress()
                    continue
                try:
                    if hasattr(ctx, 'input'):
                        from .utils.generic import get_image_md5
                        image_md5 = get_image_md5(ctx.input)
                        if not self._restore_image_context(image_md5):
                            self._set_image_context(config, ctx.input)
                    
                    # Colorize/Upscale/Inpaint Only Mode: Skip rendering pipeline
                    if not self.colorize_only and not self.upscale_only and not self.inpaint_only:
                        ctx = await self._complete_translation_pipeline(ctx, config)
                    
                    # --- BEGIN SAVE LOGIC ---
                    if save_info and ctx.result:
                        try:
                            self._save_and_cleanup_context(ctx, save_info, config, "HQ")
                        except Exception as save_err:
                            logger.error(f"Error saving high-quality result for {os.path.basename(ctx.image_name)}: {save_err}")
                            import traceback
                            logger.error(traceback.format_exc())
                    # --- END SAVE LOGIC ---

                    # Save the JSON only when save_text or text_output_file is on (an empty text_regions is saved too)
                    if not getattr(ctx, 'skipped', False) and (self.save_text or self.text_output_file) and hasattr(ctx, 'text_regions') and ctx.text_regions is not None and hasattr(ctx, 'image_name') and ctx.image_name:
                        # Use the config from the loop variable, not the one on ctx
                        self._save_text_to_file(ctx.image_name, ctx, config)

                    # ✅ Mark as successful
                    if not save_info:
                        ctx.success = True
                    # ✅ Clear the intermediate images (metadata such as text_regions is kept)
                    self._cleanup_context_memory(ctx, keep_result=True)

                    results.append(ctx)
                    await report_completed_image_progress()
                except Exception as e:
                    logger.error(f"Error rendering image: {e}")
                    if not self.ignore_errors:
                        raise RuntimeError(f"Rendering failed for {os.path.basename(ctx.image_name) if hasattr(ctx, 'image_name') else 'Unknown'}: {e}") from e
                    ctx = self._mark_context_failure(ctx, e, stage='rendering')
                    results.append(ctx)
                    await report_completed_image_progress()
            
            # ✅ Free memory as soon as the batch is done (the translation history is kept for the next batch)
#             import gc
            # 1. Clear the image references in batch_data
            for data in batch_data:
                if 'image' in data:
                    data['image'] = None
            batch_data.clear()
            
            # 2. Use the shared clean-up method
            self._cleanup_batch_memory(
                preprocessed_contexts=preprocessed_contexts,
                keep_results=True
            )
            
            logger.debug(f'[MEMORY] Batch {batch_num} cleanup completed (kept translation history for context)')
            await self._report_progress(progress_state)

        logger.info(f"High quality translation completed: processed {len(results)} images")
        return results

    def _prepare_resume_context(self, plan: BatchInputPlan) -> None:
        """Install one complete backend batch plan's ordered resume history."""
        self._resume_context_pages = list(plan.resume_pages)
        self._resume_context_cursor = 0
        self._resume_context_order = dict(plan.resume_order)

    def _append_resume_context_before(self, image_path: str) -> None:
        """Append skipped pages that occur before the current source page."""
        if not self._resume_context_pages or not image_path:
            return
        current_path = os.path.abspath(os.path.normpath(str(image_path)))
        current_order = self._resume_context_order.get(current_path)
        if current_order is None:
            return

        while self._resume_context_cursor < len(self._resume_context_pages):
            order, _source_path, entries = self._resume_context_pages[self._resume_context_cursor]
            if order >= current_order:
                break
            self.all_page_translations.append(entries)
            self._resume_context_cursor += 1
        self._prune_context_history()
