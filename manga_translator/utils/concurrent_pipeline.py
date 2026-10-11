"""
Concurrent pipeline module - a truly parallel architecture.
Pipeline concurrency: the four steps detection+OCR, translation, inpainting and rendering run in separate threads.
Each thread has its own event loop and does not block the others
"""
import asyncio
import contextlib
import logging
import os
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import List

import numpy as np

from . import Context, load_image, open_pil_image
from .batch_skip import slice_batch_indices
from manga_translator.utils.swallowed import note_ignored_error

# Use the main logger of manga_translator, so the UI can capture the log
logger = logging.getLogger('manga_translator')


class PipelineAbortError(asyncio.CancelledError):
    """Internal stop signal: used to stop the other worker threads; it should not be treated as a cancellation by the user."""



class ConcurrentPipeline:
    """
    Concurrent pipeline processor - a truly parallel architecture

    4 separate threads, each with its own event loop, none blocking another:
    1. Detection+OCR thread → puts finished items in the translation queue and the inpainting queue
    2. Translation thread → handles the translation queue in batches (HTTP requests are not blocked by GPU work)
    3. Inpainting thread → handles the inpainting queue (GPU inference does not block translation)
    4. Rendering thread → renders the output once translation and inpainting are done

    batch_size controls the translation batch size (how many images are translated at once)
    and also limits the length of the queue waiting for translation, so detection/OCR does not pile up without bound when the API is slow.

    queue.Queue and threading.Lock are used for communication and synchronisation between the threads.
    """
    
    def __init__(self, translator_instance, batch_size: int = 3, max_workers: int = 4):
        """
        Initialise the concurrent pipeline

        Args:
            translator_instance: the MangaTranslator instance
            batch_size: batch size (how many images are translated at once)
            max_workers: thread pool size of each step
        """
        self.translator = translator_instance
        self.batch_size = batch_size
        
        # ✅ A separate thread pool for each step, for real parallel processing
        # Each thread has its own event loop and does not block the others
        self._detection_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='DetectionThread')
        self._translation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='TranslationThread')
        self._inpaint_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='InpaintThread')
        self._render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='RenderThread')
        
        # Thread-safe queues
        self.translation_queue = queue.Queue(maxsize=max(1, batch_size))  # Translation queue (with back pressure)
        self.inpaint_queue = queue.Queue()      # Inpainting queue
        self.render_queue = queue.Queue()       # Rendering queue
        
        # Result storage {image_name: ctx}
        # A thread lock protects the shared data
        self._lock = threading.Lock()
        self.translation_done = {}  # ctx objects whose translation is done (with the translated text_regions)
        self.inpaint_done = {}      # ctx objects whose inpainting is done (with img_inpainted)
        self.pending_redo = set()   # Names of images that need inpainting again after the translation filter (to time when they enter the rendering queue)
        
        # The base ctx objects (results of detection + OCR), for translation and inpainting
        self.base_contexts = {}     # {image_name: ctx}
        
        # Control flags
        self.stop_workers = False
        self.detection_ocr_done = False  # Whether detection + OCR is completely done
        self.translation_thread_done = False  # Whether the translation thread has ended (no more redo inpainting tasks will be posted)
        self.has_critical_error = False  # Whether a critical error happened
        self.critical_error_msg = None   # The critical error message
        self.critical_error_exception = None  # The original exception object
        
        # Statistics
        self.start_time = None
        self.total_images = 0
        self.stats = {
            'detection_ocr': 0,
            'translation': 0,
            'inpaint': 0,
            'rendering': 0
        }
        
        # Result list (thread-safe)
        self._results = []
        self._results_lock = threading.Lock()
        
        # ✅ Thread-safe status message queue (for reporting key log lines to the main thread)
        self._status_queue = queue.Queue()
        self.failed_images = set()
    
    def _emit_status(self, message: str):
        """Send a status message to the main thread (thread-safe)"""
        self._status_queue.put(message)
    
    def _flush_status_to_logger(self):
        """Write the status messages in the queue to the logger (called on the main thread)"""
        while not self._status_queue.empty():
            try:
                msg = self._status_queue.get_nowait()
                logger.info(msg)
            except queue.Empty:
                break

    def _record_failed_image(self, image_name: str | None):
        """Record the number of failed files, so the same file is not counted again at several stages."""
        normalized_name = str(image_name or "").strip()
        if not normalized_name:
            return
        with self._lock:
            self.failed_images.add(normalized_name)
    def _get_failed_count(self) -> int:
        with self._lock:
            return len(self.failed_images)

    def _get_runtime_skipped_count(self) -> int:
        """Read the number of run-time skips from the finished results."""
        with self._results_lock:
            return sum(1 for ctx in self._results if getattr(ctx, "skipped", False))


    def _pop_translation_task(self, timeout: float):
        """Take one task from the translation queue."""
        image_name, config = self.translation_queue.get(timeout=timeout)
        with self._lock:
            ctx = self.base_contexts.get(image_name)
        if not ctx:
            logger.error(f"[Translation] Base context not found for {image_name}")
            return None
        return ctx, config

    def _should_translate_batch(self, batch: List[tuple]):
        """Decide from the number of images whether the current batch should be translated right away."""
        if not batch:
            return False, ""

        if len(batch) >= self.batch_size:
            return True, f'Batch is full ({len(batch)}/{self.batch_size} images)'

        if self.detection_ocr_done:
            return True, f'OCR completed, translating the remaining {len(batch)} images'

        return False, ""

    def _enqueue_translation_task(self, image_name: str, config):
        """
        Submit a task to the translation queue.
        When the translation API is too slow, back pressure builds up here and keeps detection/OCR from running ahead without bound.
        """
        waited = False
        while not self.stop_workers:
            try:
                self.translation_queue.put((image_name, config), timeout=0.1)
                if waited:
                    logger.info(
                        f"[Detection+OCR] Translation queue has capacity again, resuming processing: {image_name} (queue: {self.translation_queue.qsize()}/{self.translation_queue.maxsize})"
                    )
                return
            except queue.Full:
                if not waited:
                    waited = True
                    logger.info(
                        f"[Detection+OCR] Translation queue is full, waiting for the translation thread to consume tasks ({self.translation_queue.qsize()}/{self.translation_queue.maxsize})"
                    )
                self._check_cancelled_or_raise('Detection+OCR', f'Waiting for translation queue capacity: {os.path.basename(image_name)}')

        raise RuntimeError('Concurrent pipeline has stopped; cannot enqueue translation tasks')

    def _check_cancelled_or_raise(self, stage: str, detail: str = ""):
        """Single cancellation check: tells a cancellation by the user from an internal stop."""
        if self.has_critical_error:
            raise PipelineAbortError(self.critical_error_msg or 'Critical error in concurrent pipeline')

        try:
            self.translator._check_cancelled()
        except PipelineAbortError:
            self.stop_workers = True
            raise
        except asyncio.CancelledError:
            self.stop_workers = True
            message = f'[{stage}] Cancelled by user'
            if detail:
                message = f'{message}, {detail}'
            logger.warning(message)
            raise
    
    def _run_async_in_thread(self, coro):
        """Create an event loop in the current thread and run the coroutine"""
        loop = self._create_worker_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Before closing the event loop, cancel and collect every pending task, to avoid "Task was destroyed but it is pending!"
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if pending:
                for task in pending:
                    task.cancel()
                with contextlib.suppress(Exception):
                    loop.run_until_complete(asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=1.0
                    ))

            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())

            if hasattr(loop, "shutdown_default_executor"):
                with contextlib.suppress(Exception):
                    loop.run_until_complete(loop.shutdown_default_executor())

            asyncio.set_event_loop(None)
            loop.close()

    def _create_worker_event_loop(self):
        """
        Create an event loop for a worker thread.
        On Windows SelectorEventLoop is preferred, to avoid the compatibility problems of Proactor in threads.
        """
        if os.name == 'nt' and hasattr(asyncio, 'SelectorEventLoop'):
            try:
                return asyncio.SelectorEventLoop()
            except Exception as e:
                logger.warning(f"[Concurrent pipeline] Failed to create SelectorEventLoop, falling back to the default event loop: {e}")
        return asyncio.new_event_loop()
    
    def _detection_ocr_thread(self, file_paths: List[str], configs: List):
        """
        Detection+OCR worker (runs in its own thread).
        Puts the finished context in the translation queue and the inpainting queue
        """
        self._emit_status("[Detection+OCR] Thread started")
        try:
            self._run_async_in_thread(self._detection_ocr_async(file_paths, configs))
        finally:
            self._emit_status(f"[Detection+OCR] Thread completed ({self.stats['detection_ocr']}/{self.total_images})")
    
    async def _detection_ocr_async(self, file_paths: List[str], configs: List):
        """Asynchronous implementation of detection+OCR"""
        self._check_cancelled_or_raise('Detection+OCR')
        
        logger.info(f"[Detection+OCR thread] Processing {len(file_paths)} images (loading in batches)")
        
        try:
            for idx, (file_path, config) in enumerate(zip(file_paths, configs)):
                self._check_cancelled_or_raise('Detection+OCR', f'Processed {idx}/{len(file_paths)} images')
                
                # Check whether to stop (another thread failed)
                if self.stop_workers:
                    logger.warning(f"[Detection+OCR] Stop signal received, processed {idx}/{len(file_paths)} images")
                    break
                
                image = None
                ctx = None
                current_stage = 'preprocessing'
                try:
                    # Load in batches: an image is only loaded when it is needed
                    current_stage = 'preprocessing'
                    logger.debug(f"[Detection+OCR] Loading image: {file_path}")
                    with open(file_path, 'rb') as f:
                        image = open_pil_image(f, eager=True)
                    image.name = file_path
                    
                    # Create the context
                    ctx = Context()
                    ctx.input = image
                    ctx.image_name = file_path
                    ctx.verbose = self.translator.verbose
                    ctx.save_quality = self.translator.save_quality
                    ctx.config = config
                    
                    logger.info(f"[Detection+OCR] Processing {idx+1}/{self.total_images}: {ctx.image_name}")
                    
                    # Check for cancellation
                    self._check_cancelled_or_raise('Detection+OCR', f'Processed {idx}/{len(file_paths)} images')
                    
                    # Preprocessing: colorization, upscaling
                    if config.colorizer.colorizer.value != 'none':
                        current_stage = 'colorizing'
                        ctx.img_colorized = await self.translator._run_colorizer(config, ctx)
                    else:
                        ctx.img_colorized = ctx.input

                    # Check for cancellation
                    self._check_cancelled_or_raise('Detection+OCR', f'Processed {idx}/{len(file_paths)} images')

                    if config.upscale.upscale_ratio:
                        current_stage = 'upscaling'
                        ctx.upscaled = await self.translator._run_upscaling(config, ctx)
                    else:
                        ctx.upscaled = ctx.img_colorized

                    current_stage = 'preprocessing'
                    self.translator._save_editor_base_if_needed(ctx, config)

                    # Always convert to numpy
                    ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)
                    ctx.bubble_mask = None
                    
                    # Check for cancellation
                    self._check_cancelled_or_raise('Detection+OCR', f'Processed {idx}/{len(file_paths)} images')
                    
                    # Detection
                    current_stage = 'detection'
                    ctx.textlines, ctx.mask_raw, ctx.mask = await self.translator._run_detection(config, ctx)
                    
                    # Check for cancellation
                    self._check_cancelled_or_raise('Detection+OCR', f'Processed {idx}/{len(file_paths)} images')
                    
                    # OCR
                    current_stage = 'ocr'
                    ctx.textlines = await self.translator._run_ocr(config, ctx)
                    
                    # Check for cancellation
                    self._check_cancelled_or_raise('Detection+OCR', f'Processed {idx}/{len(file_paths)} images')
                    
                    # Text line merging
                    if ctx.textlines:
                        current_stage = 'textline_merge'
                        ctx.text_regions = await self.translator._run_textline_merge(config, ctx)
                    
                    self.stats['detection_ocr'] += 1
                    # ✅ Send a status log line (after each image)
                    text_count = len(ctx.text_regions) if ctx.text_regions else 0
                    self._emit_status(f"[Detection+OCR] Completed {idx+1}/{self.total_images}: {os.path.basename(file_path)} ({text_count} text blocks)")
                    
                    # Keep the image size
                    if hasattr(image, 'size'):
                        ctx.original_size = image.size
                    
                    ctx.input = image
                    
                    # Keep the base ctx
                    with self._lock:
                        self.base_contexts[ctx.image_name] = ctx
                    
                    # Put it in the translation queue and the inpainting queue
                    if ctx.text_regions:
                        # Keep the set of original region references, for detecting differences after the translation filter
                        ctx._initial_region_ids = {id(r) for r in ctx.text_regions}
                        self._enqueue_translation_task(ctx.image_name, config)
                        self.inpaint_queue.put((ctx.image_name, config, False))
                        logger.info(f"[Detection+OCR] Queued {ctx.image_name} for translation and inpainting (translation queue size: {self.translation_queue.qsize()})")
                    else:
                        # No text: mark as done and put it in the rendering queue directly
                        with self._lock:
                            self.translation_done[ctx.image_name] = []
                            self.inpaint_done[ctx.image_name] = True
                        ctx.text_regions = []
                        self.render_queue.put((ctx, config))
                        logger.debug(f"[Detection+OCR] No text in {ctx.image_name}, queuing directly for rendering")
                    
                except Exception as e:
                    try:
                        error_msg = str(e)
                    except Exception as ignored_error:
                        note_ignored_error(ignored_error, "manga_translator/utils/concurrent_pipeline.py:ConcurrentPipeline._detection_ocr_async")
                        error_msg = f'Unable to retrieve exception details (exception type: {type(e).__name__})'
                    
                    logger.error(f"[Detection+OCR] Failed: {error_msg}")
                    logger.error(traceback.format_exc())
                    if not self.translator.ignore_errors:
                        self.has_critical_error = True
                        self.critical_error_msg = f'Detection+OCR failed: {error_msg}'
                        self.critical_error_exception = e
                        self.stop_workers = True
                        break

                    failed_ctx = ctx or Context()
                    if image is not None and getattr(failed_ctx, 'input', None) is None:
                        failed_ctx.input = image
                    failed_ctx.image_name = getattr(failed_ctx, 'image_name', None) or file_path
                    failed_ctx.config = config
                    failed_ctx.text_regions = []
                    failed_ctx = self.translator._mark_context_failure(failed_ctx, e, stage=current_stage)
                    self._record_failed_image(failed_ctx.image_name)

                    with self._lock:
                        self.base_contexts[failed_ctx.image_name] = failed_ctx
                        self.translation_done[failed_ctx.image_name] = []
                        self.inpaint_done[failed_ctx.image_name] = True

                    self.stats['detection_ocr'] += 1
                    self._emit_status(f"[Detection+OCR] Skipping failed file {idx+1}/{self.total_images}: {os.path.basename(file_path)}")
                    self.render_queue.put((failed_ctx, config))
                    continue
                except PipelineAbortError:
                    logger.info(f"[Detection+OCR] Stopped due to an internal stop signal: {os.path.basename(file_path)}")
                    break
        except PipelineAbortError:
            self.stop_workers = True
        except asyncio.CancelledError:
            self.stop_workers = True
            raise
        finally:
            # Mark detection + OCR as completely done
            self.detection_ocr_done = True
            logger.info("[Detection+OCR thread] Processing completed")
    
    def _translation_thread(self):
        """Translation worker (runs in its own thread)"""
        self._emit_status("[Translation] Thread started")
        try:
            self._run_async_in_thread(self._translation_async())
        finally:
            logger.info(f"[Translation thread] Thread completed ({self.stats['translation']}/{self.total_images})")
            self._emit_status(f"[Translation] Thread completed ({self.stats['translation']}/{self.total_images})")
    
    async def _translation_async(self):
        """Asynchronous implementation of translation"""
        batch = []
        try:
            self._check_cancelled_or_raise('Translation')
            logger.info(f"[Translation thread] Started, batch size: {self.batch_size}")
            
            while not self.stop_workers:
                try:
                    self._check_cancelled_or_raise('Translation', f"Completed {self.stats['translation']}/{self.total_images}")

                    if self.has_critical_error:
                        logger.warning(f"[Translation] Critical error detected, stopping translation (completed {self.stats['translation']}/{self.total_images})")
                        break
                    
                    # Get a task from the queue (non-blocking)
                    try:
                        task = self._pop_translation_task(timeout=0.1)
                        if task:
                            ctx, config = task
                            batch.append((ctx, config))
                    except queue.Empty:
                        if not batch:
                            if self.detection_ocr_done and self.translation_queue.empty():
                                break
                            if self.has_critical_error:
                                logger.warning("[Translation] Critical error detected, stopping wait")
                                break
                            continue
                    
                    # Collect more images until batch_size is reached
                    while len(batch) < self.batch_size:
                        try:
                            task = self._pop_translation_task(timeout=0.05)
                            if task:
                                ctx, config = task
                                batch.append((ctx, config))
                        except queue.Empty:
                            break
                    
                    # Decide whether the current batch should be translated
                    should_translate, reason = self._should_translate_batch(batch)

                    if should_translate:
                        logger.info(f"[Translation] {reason}, starting translation ({len(batch)} images)")
                        await self._process_translation_batch(batch)
                        batch = []
                    
                except PipelineAbortError:
                    logger.info("[Translation] Stopped due to an internal stop signal")
                    break
                except asyncio.CancelledError:
                    self.stop_workers = True
                    raise
                except Exception as e:
                    try:
                        error_msg = str(e)
                    except Exception as ignored_error:
                        note_ignored_error(ignored_error, "manga_translator/utils/concurrent_pipeline.py:ConcurrentPipeline._translation_async")
                        error_msg = f'Unable to retrieve exception details (exception type: {type(e).__name__})'
                    
                    logger.error(f"[Translation thread] Error: {error_msg}")
                    logger.error(traceback.format_exc())
                    self.has_critical_error = True
                    self.critical_error_msg = f'Translation thread error: {error_msg}'
                    self.critical_error_exception = e
                    self.stop_workers = True
                    break
            
            # Handle the remaining batch
            if batch and not self.stop_workers:
                logger.info(f"[Translation] Translating the remaining {len(batch)} images")
                await self._process_translation_batch(batch)
            
            if self.stats['translation'] >= self.total_images:
                logger.info(f"[Translation thread] All images translated ({self.stats['translation']}/{self.total_images})")
        except PipelineAbortError:
            self.stop_workers = True
        finally:
            self.translation_thread_done = True
            logger.info("[Translation thread] Stopped")
    
    async def _process_translation_batch(self, batch: List[tuple]):
        """Handle one translation batch"""
        if not batch:
            return
        batch_items = [(ctx.image_name, config) for ctx, config in batch]
        batch_slices = slice_batch_indices(
            batch_items,
            len(batch),
            self.translator._resume_context_pages,
            self.translator._resume_context_order,
        )
        if len(batch_slices) > 1:
            for batch_start, batch_end in batch_slices:
                await self._process_translation_batch(batch[batch_start:batch_end])
            return

        self.translator._append_resume_context_before(batch[0][0].image_name)
        
        logger.info(f"[Translation] Translating a batch of {len(batch)} images")
        
        try:
            self._check_cancelled_or_raise('Translation', f'Translating a batch of {len(batch)} images')
            # Call the translation directly (already inside the event loop of its own thread)
            translated_batch = await self.translator._batch_translate_contexts(batch, len(batch))
            self._check_cancelled_or_raise('Translation', f'Translating a batch of {len(batch)} images')
            
            self.stats['translation'] += len(batch)
            # ✅ Send a status log line
            self._emit_status(f"[Translation] Batch completed ({self.stats['translation']}/{self.total_images})")
            
            ready_to_render = 0
            redo_tasks = []  # Push outside the lock, so queue.put does not block inside it
            for ctx, config in translated_batch:
                # Work out whether the translation filter removed regions (only for ctx objects translated successfully)
                has_filtered = False
                filtered_count = 0
                if not getattr(ctx, 'translation_error', None):
                    initial_ids = getattr(ctx, '_initial_region_ids', None)
                    if initial_ids:
                        final_ids = {id(r) for r in (ctx.text_regions or [])}
                        filtered_ids = initial_ids - final_ids
                        has_filtered = bool(filtered_ids)
                        filtered_count = len(filtered_ids)

                with self._lock:
                    self.translation_done[ctx.image_name] = ctx.text_regions
                    if ctx.image_name in self.base_contexts:
                        self.base_contexts[ctx.image_name].text_regions = ctx.text_regions

                    if has_filtered:
                        # The translation filter removed regions: mark for redo. The first run may or may not be finished yet.
                        # The first-run branch of the inpainting thread skips entering the rendering queue because pending_redo is set;
                        # it enters the rendering queue when the redo task is handled.
                        self.pending_redo.add(ctx.image_name)
                        redo_tasks.append((ctx.image_name, config))
                        logger.info(f"[Translation] {ctx.image_name}: filtered out {filtered_count} regions; inpainting will run again")
                    elif ctx.image_name in self.inpaint_done:
                        # No difference + the first inpainting run is done -> enter the rendering queue at once
                        self.render_queue.put((ctx, config))
                        ready_to_render += 1
                        logger.info(f"[Translation] Translation and inpainting completed for {ctx.image_name}, immediately queuing for rendering")

            # Push the redo task to the inpainting queue outside the lock
            for image_name, config in redo_tasks:
                self.inpaint_queue.put((image_name, config, True))

            if ready_to_render > 0:
                logger.info(f"[Translation] Immediately queued {ready_to_render}/{len(batch)} images in the batch for rendering")
            if redo_tasks:
                logger.info(f"[Translation] {len(redo_tasks)}/{len(batch)} images in the batch require inpainting again")
            if ready_to_render == 0 and not redo_tasks:
                logger.debug(f"[Translation] Inpainting completed for 0/{len(batch)} images in the batch; waiting for inpainting to finish before queuing for rendering")
            
        except PipelineAbortError:
            self.stop_workers = True
            raise
        except asyncio.CancelledError:
            self.stop_workers = True
            raise
        except Exception as e:
            try:
                error_msg = str(e)
            except Exception as str_error:
                error_msg = f'Unable to retrieve exception details (string conversion error: {type(str_error).__name__})'
                logger.error(f"[Translation] Failed to convert exception to string: {str_error}")
            
            logger.error(f"[Translation] Batch failed: {error_msg}")
            logger.error(f"[Translation] Exception type: {type(e).__name__}")
            logger.error(traceback.format_exc())

            self.stats['translation'] += len(batch)
            self._emit_status(f"[Translation] Skipping failed batch ({self.stats['translation']}/{self.total_images})")

            for ctx, config in batch:
                self.translator._mark_context_failure(ctx, e, stage='translation')
                self._record_failed_image(ctx.image_name)
                with self._lock:
                    self.translation_done[ctx.image_name] = []
                    if ctx.image_name in self.base_contexts:
                        self.base_contexts[ctx.image_name].text_regions = []
                    if self.translator.ignore_errors and ctx.image_name in self.inpaint_done:
                        self.render_queue.put((ctx, config))
                ctx.text_regions = []

            if not self.translator.ignore_errors:
                self.has_critical_error = True
                self.critical_error_msg = f'Translation batch failed: {error_msg}'
                self.critical_error_exception = e
                self.stop_workers = True
    
    def _inpaint_thread(self):
        """Inpainting worker (runs in its own thread)"""
        self._emit_status("[Inpainting] Thread started")
        try:
            self._run_async_in_thread(self._inpaint_async())
        finally:
            self._emit_status(f"[Inpainting] Thread completed ({self.stats['inpaint']}/{self.total_images})")
    
    async def _inpaint_async(self):
        """Asynchronous implementation of inpainting"""
        self._check_cancelled_or_raise('Inpainting')
        
        logger.info("[Inpainting thread] Started")
        
        inpaint_count = 0
        
        try:
            while not self.stop_workers:
                current_stage = 'inpainting'
                image_name = None
                config = None
                ctx = None
                is_redo = False
                try:
                    self._check_cancelled_or_raise('Inpainting', f'Completed {inpaint_count}/{self.total_images}')

                    if self.has_critical_error:
                        logger.warning(f"[Inpainting] Critical error detected, stopping inpainting (completed {inpaint_count}/{self.total_images})")
                        break
                    
                    # Check whether all tasks are done.
                    # The translation thread may only find out that regions were filtered after the first inpainting queue has emptied,
                    # and then post redo inpainting tasks; exiting is only allowed after the translation thread has ended.
                    if (
                        self.detection_ocr_done
                        and self.translation_thread_done
                        and self.inpaint_queue.empty()
                    ):
                        await asyncio.sleep(0.5)
                        self._check_cancelled_or_raise('Inpainting', f'Completed {inpaint_count}/{self.total_images}')
                        if self.translation_thread_done and self.inpaint_queue.empty():
                            logger.info(f"[Inpainting thread] All tasks completed ({inpaint_count}/{self.total_images})")
                            break
                    
                    # Try to get a task
                    try:
                        image_name, config, is_redo = self.inpaint_queue.get(timeout=1.0)
                    except queue.Empty:
                        if self.has_critical_error:
                            logger.warning("[Inpainting] Critical error detected, stopping wait")
                            break
                        continue

                    with self._lock:
                        ctx = self.base_contexts.get(image_name)
                    if not ctx:
                        logger.error(f"[Inpainting] Base context not found for {image_name}")
                        continue

                    if is_redo:
                        logger.info(f"[Inpainting] Running again after filtering: {ctx.image_name} (remaining regions: {len(ctx.text_regions) if ctx.text_regions else 0})")
                        # Clear the old mask, so _run_mask_refinement rebuilds it from the filtered regions
                        ctx.mask = None
                    else:
                        logger.info(f"[Inpainting] Processing: {ctx.image_name}")

                    if getattr(ctx, 'translation_error', None):
                        self._record_failed_image(ctx.image_name)
                        with self._lock:
                            self.inpaint_done[ctx.image_name] = True
                            self.pending_redo.discard(ctx.image_name)
                            if ctx.image_name in self.translation_done:
                                self.render_queue.put((ctx, config))
                        if not is_redo:
                            self.stats['inpaint'] += 1
                            inpaint_count += 1
                            self._emit_status(f"[Inpainting] Skipping failed file {inpaint_count}/{self.total_images}: {os.path.basename(ctx.image_name)}")
                        continue

                    # Mask refinement
                    if ctx.mask is None and ctx.text_regions:
                        current_stage = 'mask-generation'
                        self._check_cancelled_or_raise('Inpainting', f'Processing {os.path.basename(ctx.image_name)}')
                        ctx.mask = await self.translator._run_mask_refinement(config, ctx)
                        self._check_cancelled_or_raise('Inpainting', f'Processing {os.path.basename(ctx.image_name)}')

                    # Inpainting
                    if ctx.text_regions:
                        current_stage = 'inpainting'
                        self._check_cancelled_or_raise('Inpainting', f'Processing {os.path.basename(ctx.image_name)}')
                        ctx.img_inpainted = await self.translator._run_inpainting(config, ctx)
                        self._check_cancelled_or_raise('Inpainting', f'Processing {os.path.basename(ctx.image_name)}')

                    if not is_redo:
                        self.stats['inpaint'] += 1
                        inpaint_count += 1
                        self._emit_status(f"[Inpainting] Completed {inpaint_count}/{self.total_images}: {os.path.basename(ctx.image_name)}")
                    else:
                        self._emit_status(f"[Inpainting] Repeat run completed: {os.path.basename(ctx.image_name)}")

                    # Mark inpainting as done
                    with self._lock:
                        self.inpaint_done[ctx.image_name] = True

                        if is_redo:
                            # After a redo the translation is certainly done and the difference confirmed. Enter the rendering queue unconditionally.
                            self.pending_redo.discard(ctx.image_name)
                            render_ctx = self.base_contexts.get(ctx.image_name)
                            if render_ctx:
                                # text_regions is already the version after the translation filter (the translation thread wrote it back)
                                self.render_queue.put((render_ctx, config))
                                logger.info(f"[Inpainting] Repeat run completed for {ctx.image_name}, queuing for rendering")
                            else:
                                logger.error(f"[Inpainting] Base context not found for {ctx.image_name}")
                        elif ctx.image_name in self.pending_redo:
                            # The translation confirmed a filter and the redo task is pending; the finished first run does not enter the queue
                            logger.info(f"[Inpainting] First run completed for {ctx.image_name}, waiting to run again")
                        elif ctx.image_name in self.translation_done:
                            # The translation is done and no region was filtered: add to the rendering queue
                            render_ctx = self.base_contexts.get(ctx.image_name)
                            if render_ctx:
                                translated_regions = self.translation_done.get(ctx.image_name)
                                if isinstance(translated_regions, (list, tuple)):
                                    render_ctx.text_regions = translated_regions
                                elif translated_regions:
                                    logger.warning(f"[Inpainting] Unexpected translation result type for {ctx.image_name}: {type(translated_regions)}, using an empty list")
                                    render_ctx.text_regions = []
                                else:
                                    render_ctx.text_regions = []
                                self.render_queue.put((render_ctx, config))
                                logger.info(f"[Inpainting] Translation and inpainting completed for {ctx.image_name}, queuing for rendering")
                            else:
                                logger.error(f"[Inpainting] Base context not found for {ctx.image_name}")
                    
                except Exception as e:
                    try:
                        error_msg = str(e)
                    except Exception as ignored_error:
                        note_ignored_error(ignored_error, "manga_translator/utils/concurrent_pipeline.py:ConcurrentPipeline._inpaint_async")
                        error_msg = f'Unable to retrieve exception details (exception type: {type(e).__name__})'
                    
                    logger.error(f"[Inpainting thread] Error: {error_msg}")
                    logger.error(traceback.format_exc())
                    if not self.translator.ignore_errors:
                        self.has_critical_error = True
                        self.critical_error_msg = f'Inpainting thread error: {error_msg}'
                        self.critical_error_exception = e
                        self.stop_workers = True
                        break

                    if ctx is None:
                        ctx = Context()
                        ctx.image_name = image_name
                        ctx.config = config
                    ctx = self.translator._mark_context_failure(ctx, e, stage=current_stage)
                    self._record_failed_image(ctx.image_name)
                    ctx.text_regions = []

                    with self._lock:
                        self.inpaint_done[ctx.image_name] = True
                        if ctx.image_name in self.base_contexts:
                            self.base_contexts[ctx.image_name] = ctx
                        if is_redo:
                            # The redo failed: clear pending_redo and push to the rendering queue directly
                            self.pending_redo.discard(ctx.image_name)
                            self.render_queue.put((ctx, config))
                        else:
                            # The first run failed: when the translation has marked it for redo, let the redo task push to rendering
                            if (ctx.image_name not in self.pending_redo
                                    and ctx.image_name in self.translation_done):
                                self.render_queue.put((ctx, config))

                    if not is_redo:
                        self.stats['inpaint'] += 1
                        inpaint_count += 1
                        self._emit_status(f"[Inpainting] Skipping failed file {inpaint_count}/{self.total_images}: {os.path.basename(ctx.image_name)}")
                    else:
                        self._emit_status(f"[Inpainting] Repeat run failed: {os.path.basename(ctx.image_name)}")
                    continue
                except PipelineAbortError:
                    logger.info("[Inpainting] Stopped due to an internal stop signal")
                    break
        except PipelineAbortError:
            self.stop_workers = True
        except asyncio.CancelledError:
            self.stop_workers = True
            raise
        finally:
            logger.info("[Inpainting thread] Stopped")
    
    def _render_thread(self):
        """Rendering worker (runs in its own thread)"""
        self._emit_status("[Rendering] Thread started")
        try:
            self._run_async_in_thread(self._render_async())
        finally:
            self._emit_status(f"[Rendering] Thread completed ({self.stats['rendering']}/{self.total_images})")
    
    async def _render_async(self):
        """Asynchronous implementation of rendering"""
        self._check_cancelled_or_raise('Rendering')
        
        logger.info("[Rendering thread] Started")
        
        rendered_count = 0
        
        try:
            while not self.stop_workers or rendered_count < self.total_images:
                ctx = None
                config = None
                try:
                    self._check_cancelled_or_raise('Rendering', f'Completed {rendered_count}/{self.total_images}')

                    if self.has_critical_error:
                        logger.warning(f"[Rendering] Critical error detected, stopping rendering (completed {rendered_count}/{self.total_images})")
                        break
                    
                    # Try to get a task
                    try:
                        ctx, config = self.render_queue.get(timeout=1.0)
                    except queue.Empty:
                        # Check whether to exit
                        if self.stop_workers:
                            logger.info(f"[Rendering] Stop signal received, rendered {rendered_count}/{self.total_images} images")
                            break
                        if rendered_count >= self.total_images:
                            break
                        if self.has_critical_error:
                            logger.warning("[Rendering] Critical error detected, stopping wait")
                            break
                        continue
                    
                    logger.info(f"[Rendering] Retrieved task from queue: {ctx.image_name} (remaining in queue: {self.render_queue.qsize()})")
                    
                    # Check the ctx
                    with self._lock:
                        verified_ctx = self.base_contexts.get(ctx.image_name)
                    if not verified_ctx:
                        logger.error(f"[Rendering] Base context not found for {ctx.image_name}, skipping")
                        continue
                    
                    ctx = verified_ctx
                    logger.info(f"[Rendering] Processing: {ctx.image_name}")

                    if getattr(ctx, 'translation_error', None):
                        self._record_failed_image(ctx.image_name)
                        self.stats['rendering'] += 1
                        rendered_count += 1
                        self._emit_status(f"[Rendering] Skipping failed file {rendered_count}/{self.total_images}: {os.path.basename(ctx.image_name)}")

                        with self._results_lock:
                            self._results.append(ctx)
                        self.translator._cleanup_context_memory(ctx, keep_result=True)
                        with self._lock:
                            if ctx.image_name in self.base_contexts:
                                del self.base_contexts[ctx.image_name]
                        continue
                    
                    # Check whether the data needed for rendering is complete
                    if not hasattr(ctx, 'img_rgb') or ctx.img_rgb is None:
                        logger.error("[Rendering] ctx.img_rgb is None, cannot render! Skipping this image")
                        ctx = self.translator._mark_context_failure(ctx, RuntimeError('Missing original image data'), stage='rendering')
                        self._record_failed_image(ctx.image_name)
                        self.stats['rendering'] += 1
                        rendered_count += 1
                        self._emit_status(f"[Rendering] Skipping failed file {rendered_count}/{self.total_images}: {os.path.basename(ctx.image_name)}")
                        with self._results_lock:
                            self._results.append(ctx)
                        self.translator._cleanup_context_memory(ctx, keep_result=True)
                        with self._lock:
                            if ctx.image_name in self.base_contexts:
                                del self.base_contexts[ctx.image_name]
                        continue
                    
                    # Rendering may release context arrays; retain the generated sidecar image.
                    inpainted_snapshot = None
                    if (
                        (self.translator.save_text or self.translator.text_output_file)
                        and getattr(ctx, 'img_inpainted', None) is not None
                        and getattr(ctx, 'mask', None) is not None
                        and np.any(ctx.mask)
                    ):
                        inpainted_snapshot = np.copy(ctx.img_inpainted)
                        logger.debug("[Rendering] Backed up the inpainted image for saving")
                    
                    if not ctx.text_regions:
                        from .generic import dump_image
                        ctx.result = dump_image(ctx.input, ctx.img_rgb, ctx.img_alpha)
                    else:
                        self._check_cancelled_or_raise('Rendering', f'Processing {os.path.basename(ctx.image_name)}')
                        ctx.img_rendered = await self.translator._run_text_rendering(config, ctx)
                        self._check_cancelled_or_raise('Rendering', f'Processing {os.path.basename(ctx.image_name)}')
                        from .generic import dump_image
                        ctx.result = dump_image(
                            ctx.input,
                            ctx.img_rendered,
                            ctx.img_alpha,
                            mask=ctx.mask,
                            render_alpha=getattr(ctx, 'img_render_alpha', None),
                        )
                    
                    self.stats['rendering'] += 1
                    rendered_count += 1
                    
                    # ✅ Send a status log line (after each image)
                    self._emit_status(f"[Rendering] Completed {rendered_count}/{self.total_images}: {os.path.basename(ctx.image_name)}")
                    
                    # Save
                    if ctx.result is not None:
                        logger.info(f"[Rendering] ctx.result set, type: {type(ctx.result)}")
                        
                        try:
                            if hasattr(self.translator, '_current_save_info') and self.translator._current_save_info:
                                save_info = self.translator._current_save_info
                                
                                if inpainted_snapshot is not None:
                                    try:
                                        saved_inpainted_path = self.translator._save_inpainted_image(
                                            ctx.image_name,
                                            inpainted_snapshot,
                                        )
                                        if saved_inpainted_path is None:
                                            raise OSError(
                                                f"Failed to write inpainted image: {ctx.image_name}"
                                            )
                                    finally:
                                        inpainted_snapshot = None
                                
                                # Save the translation result and export the PSD
                                self.translator._save_and_cleanup_context(ctx, save_info, config, "CONCURRENT")
                                
                                if (self.translator.save_text or self.translator.text_output_file) and ctx.text_regions is not None:
                                    self.translator._save_text_to_file(ctx.image_name, ctx, config)
                            else:
                                logger.warning("[Rendering] No save_info, skipping save")
                            
                            ctx.success = True
                                    
                        except Exception as save_err:
                            logger.error(f"[Rendering] Failed to save {os.path.basename(ctx.image_name)}: {save_err}")
                            logger.error(traceback.format_exc())
                            ctx = self.translator._mark_context_failure(ctx, save_err, stage='saving')
                            self._record_failed_image(ctx.image_name)
                    else:
                        logger.error("[Rendering] ctx.result is None!")
                    
                    # Add to the result list
                    with self._results_lock:
                        self._results.append(ctx)

                    # Free memory - call the shared clean-up function
                    logger.debug(f"[Rendering] Releasing memory: {ctx.image_name}")
                    self.translator._cleanup_context_memory(ctx, keep_result=True)

                    # Clear base_contexts
                    with self._lock:
                        if ctx.image_name in self.base_contexts:
                            del self.base_contexts[ctx.image_name]
                            logger.debug(f"[Rendering] Cleared base context for {ctx.image_name}")
                    
                except Exception as e:
                    try:
                        error_msg = str(e)
                    except Exception as ignored_error:
                        note_ignored_error(ignored_error, "manga_translator/utils/concurrent_pipeline.py:ConcurrentPipeline._render_async")
                        error_msg = f'Unable to retrieve exception details (exception type: {type(e).__name__})'
                    
                    logger.error(f"[Rendering thread] Error: {error_msg}")
                    logger.error(traceback.format_exc())
                    if not self.translator.ignore_errors:
                        self.has_critical_error = True
                        self.critical_error_msg = f'Rendering thread error: {error_msg}'
                        self.critical_error_exception = e
                        self.stop_workers = True
                        break

                    if ctx is not None:
                        ctx = self.translator._mark_context_failure(ctx, e, stage='rendering')
                        self._record_failed_image(ctx.image_name)
                        self.stats['rendering'] += 1
                        rendered_count += 1
                        self._emit_status(f"[Rendering] Skipping failed file {rendered_count}/{self.total_images}: {os.path.basename(ctx.image_name)}")
                        with self._results_lock:
                            self._results.append(ctx)
                        self.translator._cleanup_context_memory(ctx, keep_result=True)
                        with self._lock:
                            if ctx.image_name in self.base_contexts:
                                del self.base_contexts[ctx.image_name]
                        continue

                    self.has_critical_error = True
                    self.critical_error_msg = f'Rendering thread error: {error_msg}'
                    self.critical_error_exception = e
                    self.stop_workers = True
                    break
                except PipelineAbortError:
                    logger.info("[Rendering] Stopped due to an internal stop signal")
                    break
        except PipelineAbortError:
            self.stop_workers = True
        except asyncio.CancelledError:
            self.stop_workers = True
            raise
        finally:
            logger.info("[Rendering thread] Stopped")
    
    async def process_batch(
        self,
        file_paths: List[str],
        configs: List,
        *,
        progress_offset: int = 0,
        progress_total: int | None = None,
        skipped_count: int = 0,
    ) -> List[Context]:
        """Run the concurrent pipeline for the backend-planned pending inputs."""
        self.total_images = len(file_paths)
        self.start_time = datetime.now(timezone.utc)
        
        logger.info(f"[Concurrent pipeline] Processing {self.total_images} images")
        logger.info("[Concurrent pipeline] Parallel mode: 4 independent threads (detection+OCR / translation / inpainting / rendering)")
        
        # Reset the statistics
        for key in self.stats:
            self.stats[key] = 0
        self.translation_done.clear()
        self.inpaint_done.clear()
        self.pending_redo.clear()
        self.base_contexts.clear()
        self.failed_images.clear()
        self.detection_ocr_done = False
        self.translation_thread_done = False
        self.stop_workers = False
        self.has_critical_error = False
        self.critical_error_msg = None
        self.critical_error_exception = None
        self._results = []
        
        # Include stop_workers in the single cancel callback, so in-flight API calls respond to a stop quickly too
        original_cancel_callback = getattr(self.translator, "_cancel_check_callback", None)
        if hasattr(self.translator, "set_cancel_check_callback"):
            def _pipeline_cancel_check():
                if self.has_critical_error:
                    raise PipelineAbortError(self.critical_error_msg or 'Critical error in concurrent pipeline')
                if original_cancel_callback:
                    try:
                        if bool(original_cancel_callback()):
                            return True
                    except Exception as e:
                        logger.debug(f"[Concurrent pipeline] External cancellation callback raised an exception (safe to ignore): {e}")
                if self.stop_workers:
                    raise PipelineAbortError('Concurrent pipeline stopped')
                return False
            self.translator.set_cancel_check_callback(_pipeline_cancel_check)
        
        # Submit the 4 separate thread tasks
        futures = [
            self._detection_executor.submit(self._detection_ocr_thread, file_paths, configs),
            self._translation_executor.submit(self._translation_thread),
            self._inpaint_executor.submit(self._inpaint_thread),
            self._render_executor.submit(self._render_thread),
        ]
        
        try:
            # Wait for all threads to finish (checked in an outer loop, to respond to cancellation)
            last_rendered = 0
            while True:
                done, not_done = wait(futures, timeout=0.5)
                
                # ✅ Flush the status log lines of the worker threads to the main thread
                self._flush_status_to_logger()
                self._check_cancelled_or_raise('Concurrent pipeline')
                
                # ✅ Report progress (when the number of rendered images changed)
                current_rendered = self.stats['rendering']
                if current_rendered > last_rendered:
                    try:
                        current_failed = self._get_failed_count()
                        runtime_skipped = skipped_count + self._get_runtime_skipped_count()
                        completed = progress_offset + current_rendered
                        total = progress_total if progress_total is not None else progress_offset + self.total_images
                        await self.translator._report_progress(
                            f"batch:1:{completed}:{total}:{current_failed}:{runtime_skipped}"
                        )
                    except Exception as ignored_error:
                        note_ignored_error(ignored_error, "manga_translator/utils/concurrent_pipeline.py:ConcurrentPipeline.process_batch")
                        pass
                    last_rendered = current_rendered
                
                if len(not_done) == 0:
                    break
                # Check for exceptions
                for f in done:
                    if f.exception():
                        raise f.exception()
                # Yield control and check for cancellation
                await asyncio.sleep(0)
                
        except PipelineAbortError as e:
            logger.info(f"[Concurrent pipeline] Stopped waiting due to an internal stop signal: {e}")
            self.stop_workers = True
            done, not_done = wait(futures, timeout=10.0)
            self._flush_status_to_logger()
            if not_done:
                thread_names = []
                for i, future in enumerate(futures):
                    if future in not_done:
                        names = ['Detection+OCR', 'Translation', 'Inpainting', 'Rendering']
                        thread_names.append(names[i])
                logger.warning(f"[Concurrent pipeline] {len(not_done)} threads failed to stop within 10 seconds: {', '.join(thread_names)}")
            else:
                logger.info("[Concurrent pipeline] All threads stopped")
        except asyncio.CancelledError:
            # The user cancelled the task
            logger.info("[Concurrent pipeline] Cancellation signal received")
            self.stop_workers = True
            # Wait for all threads to stop (at most 10 seconds)
            logger.info("[Concurrent pipeline] Waiting for all threads to stop...")
            done, not_done = wait(futures, timeout=10.0)
            self._flush_status_to_logger()
            if not_done:
                # Show which threads did not stop
                thread_names = []
                for i, future in enumerate(futures):
                    if future in not_done:
                        names = ['Detection+OCR', 'Translation', 'Inpainting', 'Rendering']
                        thread_names.append(names[i])
                logger.warning(f"[Concurrent pipeline] {len(not_done)} threads failed to stop within 10 seconds: {', '.join(thread_names)}")
            else:
                logger.info("[Concurrent pipeline] All threads stopped")
            raise
        except Exception as e:
            logger.error(f"[Concurrent pipeline] Error: {e}")
            logger.error(traceback.format_exc())
            self.stop_workers = True
            raise
        finally:
            self.stop_workers = True
            if hasattr(self.translator, "set_cancel_check_callback"):
                self.translator.set_cancel_check_callback(original_cancel_callback)
            # Shut down all thread pools
            for executor in [self._detection_executor, self._translation_executor, 
                           self._inpaint_executor, self._render_executor]:
                if executor:
                    executor.shutdown(wait=False)
        
        # Check for a critical error
        if self.has_critical_error:
            error_msg = self.critical_error_msg or 'Unknown error'
            logger.error(f"[Concurrent pipeline] Processing failed: {error_msg}")
            if self.critical_error_exception:
                raise self.critical_error_exception
            else:
                raise RuntimeError(f'Concurrent pipeline processing failed: {error_msg}')
        
        # Statistics
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        logger.info("[Concurrent pipeline] Completed!")
        logger.info(f"  Total time: {elapsed:.2f}s")
        logger.info(f"  Average time: {elapsed/self.total_images:.2f}s/image")
        logger.info(f"  Processing statistics: detection+OCR={self.stats['detection_ocr']}, translation={self.stats['translation']}, inpainting={self.stats['inpaint']}, rendering={self.stats['rendering']}")
        
        return self._results
