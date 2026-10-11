# -*- coding: utf-8 -*-
import asyncio
import builtins
import io
import json
import logging
import os
import re
from base64 import b64decode
from contextlib import asynccontextmanager
from typing import Union

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

from manga_translator import Config
from manga_translator.image_formats import (
    RGB_PIL_FORMATS,
    resolve_output_image_format,
)
from manga_translator.utils import normalize_pil_image, open_pil_image
from manga_translator.utils.image_modes import normalize_rgb_image

logger = logging.getLogger('manga_translator.server')


class TaskLogHandler(logging.Handler):
    """Log handler of a single task, which sends the logs to the task queue"""
    
    def __init__(self, task_id: str, session_id: str = None):
        super().__init__()
        self.task_id = task_id
        self.session_id = session_id
        self.ignored_loggers = {'uvicorn.access', 'uvicorn.error', 'httpcore', 'httpx'}
    
    def emit(self, record):
        try:
            if record.name in self.ignored_loggers:
                return
            
            from manga_translator.server.core.logging_manager import add_log
            msg = self.format(record)
            level = record.levelname
            # Root logger already prints the original record to the console.
            # Only mirror it into the task log queue here to avoid duplicate output.
            add_log(msg, level, self.task_id, self.session_id, skip_print=True)
        except Exception:
            self.handleError(record)


def _create_task_log_handler(task_id: str, session_id: str = None) -> TaskLogHandler:
    """Create the log handler of a task and add it to the relevant loggers"""
    handler = TaskLogHandler(task_id, session_id)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    
    # Add to the manga_translator logger (the namespace with an underscore)
    mt_logger = logging.getLogger('manga_translator')
    mt_logger.addHandler(handler)
    
    # Add to the manga-translator logger (the namespace with a hyphen, used by the translators)
    mt_hyphen_logger = logging.getLogger('manga-translator')
    mt_hyphen_logger.addHandler(handler)
    
    return handler


def _remove_task_log_handler(handler: TaskLogHandler):
    """Remove the log handler of a task"""
    if handler is None:
        return
    
    try:
        mt_logger = logging.getLogger('manga_translator')
        mt_logger.removeHandler(handler)
    except Exception:
        pass
    
    try:
        mt_hyphen_logger = logging.getLogger('manga-translator')
        mt_hyphen_logger.removeHandler(handler)
    except Exception:
        pass


@asynccontextmanager
async def with_user_env_vars(config: Config):
    """
    Compatibility wrapper kept for the existing call sites.

    User-specific API settings are now carried on the request Config object and
    resolved explicitly by each backend, so we no longer mutate process-wide
    environment variables here.
    """
    del config
    yield

class TranslateRequest(BaseModel):
    """This request can be a multipart or a json request"""
    image: bytes|str
    config: Config = Config()

class BatchTranslateRequest(BaseModel):
    """Batch translation request"""
    images: list[bytes|str]
    config: dict | Config = {}
    batch_size: int = 4
    filenames: list[str] = []  # List of original file names (optional)
    
    class Config:
        arbitrary_types_allowed = True

async def to_pil_image(image: Union[str, bytes, Image.Image]) -> Image.Image:
    try:
        if isinstance(image, Image.Image):
            return normalize_pil_image(image, eager=False)
        elif isinstance(image, builtins.bytes):
            image = open_pil_image(io.BytesIO(image), eager=False)
            return image
        else:
            if re.match(r'^data:image/.+;base64,', image):
                value = image.split(',', 1)[1]
                image_data = b64decode(value, validate=True)
                image = open_pil_image(io.BytesIO(image_data), eager=False)
                return image
            raise HTTPException(
                status_code=422,
                detail="Image must be uploaded as bytes or a base64 data URI",
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid image data")


def _run_translate_sync(pil_image, config: Config, task_id: str = None, cancel_check_callback=None):
    """
    Helper function that runs a translation synchronously.
    It runs in the thread pool, so the FastAPI event loop is not blocked.
    The global translator instance is used, reusing the loaded models.

    Args:
        pil_image: the PIL image
        config: the translation configuration
        task_id: the task ID (used to update the thread information)
        cancel_check_callback: callback that checks for cancellation
    """
    import threading

    #     import gc
    from manga_translator.server.core.task_manager import (
        get_global_translator,
        update_task_thread_id,
    )
    
    # Update the thread ID of the task
    if task_id:
        update_task_thread_id(task_id, threading.current_thread().ident)
    
    # Get the global translator instance (the models are reused)
    translator = get_global_translator()
    
    # Set the cancel-check callback
    if cancel_check_callback:
        translator.set_cancel_check_callback(cancel_check_callback)
    
    # Create an event loop in the new thread to run the asynchronous translation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(translator.translate(pil_image, config))
        return result
    finally:
        try:
            # Clear the cancel callback, so it does not affect the next task
            if cancel_check_callback:
                translator.set_cancel_check_callback(None)
            
            # Before closing the event loop, cancel every pending task
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Wait until all tasks are cancelled
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Close the event loop
            loop.close()
            
            # Clear the thread-local reference to the event loop (essential in a Docker environment)
            asyncio.set_event_loop(None)
            
            # Per-request memory clean-up: clear the internal state of the translator, keep the models
            from manga_translator.server.core.task_manager import cleanup_after_request
            cleanup_after_request()
            
        except Exception as e:
            logger.warning(f"Error during thread cleanup: {e}")


def _run_translate_batch_sync(images_with_configs: list, batch_size: int, task_id: str = None, cancel_check_callback=None):
    """
    Helper function that runs a batch translation synchronously.
    It runs in the thread pool, so the FastAPI event loop is not blocked.
    The global translator instance is used, reusing the loaded models.

    Args:
        images_with_configs: list of images with their configurations
        batch_size: the batch size
        task_id: the task ID (used to update the thread information)
        cancel_check_callback: callback that checks for cancellation
    """
    import threading

    #     import gc
    from manga_translator.server.core.task_manager import (
        get_global_translator,
        update_task_thread_id,
    )
    
    # Update the thread ID of the task
    if task_id:
        update_task_thread_id(task_id, threading.current_thread().ident)
    
    # Get the global translator instance (the models are reused)
    translator = get_global_translator()
    
    # Set the cancel-check callback
    if cancel_check_callback:
        translator.set_cancel_check_callback(cancel_check_callback)
    
    # Create an event loop in the new thread to run the asynchronous translation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(translator.translate_batch(images_with_configs, batch_size))
        return result
    finally:
        try:
            # Clear the cancel callback
            if cancel_check_callback:
                translator.set_cancel_check_callback(None)
            
            # Before closing the event loop, cancel every pending task
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Wait until all tasks are cancelled
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Close the event loop
            loop.close()
            
            # Clear the thread-local reference to the event loop (essential in a Docker environment)
            asyncio.set_event_loop(None)
            
            # Per-request memory clean-up: clear the internal state of the translator, keep the models
            from manga_translator.server.core.task_manager import cleanup_after_request
            cleanup_after_request()
            
        except Exception as e:
            logger.warning(f"Error during thread cleanup: {e}")


def prepare_translator_params(config: Config, workflow: str = "normal") -> dict:
    """Prepare translator parameters based on workflow."""
    translator_params = {}
    
    if hasattr(config, 'cli'):
        if hasattr(config.cli, 'load_text'):
            config.cli.load_text = False
        if hasattr(config.cli, 'translate_json_only'):
            config.cli.translate_json_only = False
        if hasattr(config.cli, 'template'):
            config.cli.template = False
        if hasattr(config.cli, 'generate_and_export'):
            config.cli.generate_and_export = False
        if hasattr(config.cli, 'upscale_only'):
            config.cli.upscale_only = False
        if hasattr(config.cli, 'colorize_only'):
            config.cli.colorize_only = False
        if hasattr(config.cli, 'inpaint_only'):
            config.cli.inpaint_only = False
        # Replace-translation can only be used in the Qt UI; the web UI may not use it
        if hasattr(config.cli, 'replace_translation'):
            config.cli.replace_translation = False
        if hasattr(config.cli, 'use_gpu'):
            config.cli.use_gpu = False
        if hasattr(config.cli, 'attempts'):
            attempts = config.cli.attempts
            # -1 means retry without limit; 0 means no retry (only the first request)
            if attempts is not None and (attempts >= 0 or attempts == -1):
                translator_params['attempts'] = attempts
    
    if hasattr(config, 'render') and getattr(config.render, 'font_family', None):
        translator_params['font_family'] = config.render.font_family
        logger.debug(f"Using font family: {config.render.font_family}")

    # Direct paste mode can only be used in the replace-translation mode of the Qt UI; the web UI may not use it
    if hasattr(config, 'render') and hasattr(config.render, 'enable_template_alignment'):
        config.render.enable_template_alignment = False
    
    # Prompt path - the relative path is passed on as it is; the translation program joins it with BASE_PATH itself
    # (high_quality_prompt_path is in config.translator and is read by the translation program directly)
    
    if workflow == "export_original":
        translator_params['template'] = True
        translator_params['save_text'] = True
    elif workflow == "save_json":
        translator_params['save_text'] = True
        translator_params['generate_and_export'] = True
    elif workflow == "load_text":
        translator_params['load_text'] = True
    elif workflow == "upscale_only":
        translator_params['upscale_only'] = True
    elif workflow == "colorize_only":
        translator_params['colorize_only'] = True
    
    return translator_params


async def get_ctx(req: Request, config: Config, image: str|bytes, workflow: str = "normal"):
    """Translate single image. The global translator instance is used, reusing the loaded models."""
    from manga_translator.server.core.logging_manager import add_log
    from manga_translator.server.core.task_manager import get_semaphore
    
    # Get the semaphore dynamically (supports hot reloading)
    translation_semaphore = get_semaphore()
    
    pil_image = await to_pil_image(image)
    
    try:
        # Prepare the workflow parameters (they affect how translation behaves, but the translator does not need rebuilding)
        prepare_translator_params(config, workflow)
        
        async with with_user_env_vars(config):
            # Wait for a translation slot
            if translation_semaphore:
                try:
                    waiters_count = len(translation_semaphore._waiters) if hasattr(translation_semaphore, '_waiters') and translation_semaphore._waiters else 0
                except Exception:
                    waiters_count = 0
                
                if waiters_count > 0:
                    add_log(f"Waiting for a translation slot... ({waiters_count} tasks queued)", "INFO")
                
                async with translation_semaphore:
                    add_log("Translation slot acquired; starting translation", "INFO")
                    # Run in the translation thread pool, reusing the global translator
                    from manga_translator.server.core.task_manager import (
                        run_in_translator_thread,
                    )
                    ctx = await run_in_translator_thread(_run_translate_sync, pil_image, config)
            else:
                # Without a semaphore, run directly
                from manga_translator.server.core.task_manager import (
                    run_in_translator_thread,
                )
                ctx = await run_in_translator_thread(_run_translate_sync, pil_image, config)
        
        result = {
            'success': ctx.success if hasattr(ctx, 'success') else (ctx.result is not None),
            'workflow': workflow
        }
        
        if ctx.result:
            result['has_image'] = True
        
        if hasattr(ctx, 'text_regions') and ctx.text_regions:
            result['text_regions'] = []
            for region in ctx.text_regions:
                region_data = {
                    'text': region.text if hasattr(region, 'text') else '',
                    'translation': region.translation if hasattr(region, 'translation') else '',
                }
                result['text_regions'].append(region_data)
        
        ctx._workflow_result = result
        return ctx
    
    finally:
        # Close the input image
        try:
            pil_image.close()
        except Exception:
            pass
        
        # Note: cleaning up ctx and reclaiming memory is done by cleanup_after_request in _run_translate_sync;
        # only the local resources created by this function are handled here


async def while_streaming(req: Request, transform, config: Config, image: bytes | str, workflow: str = "normal", original_filename: str = None):
    """Streaming translation with concurrency control."""
    from manga_translator.server.core.config_manager import (
        reload_admin_settings_if_changed,
    )
    from manga_translator.server.core.logging_manager import (
        add_log,
        generate_task_id,
        set_session_id,
        set_task_id,
    )
    from manga_translator.server.core.task_manager import (
        get_semaphore,
        is_task_cancelled,
        register_active_task,
        unregister_active_task,
        update_task_status,
    )
    
    # Check whether the configuration changed (hot reloading)
    reload_admin_settings_if_changed()
    
    # Get the semaphore dynamically (supports hot reloading)
    translation_semaphore = get_semaphore()
    
    task_id = generate_task_id()
    set_task_id(task_id)
    
    # Keep the original file name in config
    if original_filename:
        config._original_filename = original_filename
    
    session_id = getattr(config, '_session_id', None)
    if session_id:
        set_session_id(session_id)
    
    username = getattr(config, '_username', 'unknown')
    translator_name = "unknown"
    if hasattr(config, 'translator') and hasattr(config.translator, 'translator'):
        translator_name = config.translator.translator
    
    current_task = None
    try:
        current_task = asyncio.current_task()
    except RuntimeError:
        pass
    register_active_task(task_id, current_task, username, translator_name)
    
    # Create a log handler dedicated to the task (as the Qt UI does)
    task_log_handler = None
    try:
        task_log_handler = _create_task_log_handler(task_id, session_id)
    except Exception as e:
        add_log(f"Failed to create task log handler: {e}", "WARNING")
    
    async def generate():
        # Get the semaphore again inside the generator (to be sure the latest one is used)
        nonlocal translation_semaphore
        translation_semaphore = get_semaphore()
        print(f"[DEBUG] generate() started, semaphore={translation_semaphore}, task_id={task_id}")
        
        # Check the per-user concurrency limit first (before acquiring the semaphore)
        from manga_translator.server.core.middleware import (
            check_concurrent_limit,
            decrement_task_count,
            increment_task_count,
        )
        
        # Increase the task count of the user
        increment_task_count(username)
        
        try:
            # Check whether the per-user concurrency limit is exceeded
            check_concurrent_limit(username)
            
            if translation_semaphore is None:
                print("[DEBUG] semaphore is None, trying to initialise it")
                from manga_translator.server.core.task_manager import init_semaphore
                init_semaphore()
                translation_semaphore = get_semaphore()
                print(f"[DEBUG] after initialisation semaphore={translation_semaphore}")
            
            if translation_semaphore:
                # Check the current waiting queue
                try:
                    waiters_count = len(translation_semaphore._waiters) if hasattr(translation_semaphore, '_waiters') and translation_semaphore._waiters else 0
                except Exception:
                    waiters_count = 0
                
                if waiters_count > 0:
                    add_log(f"Waiting for a translation slot... ({waiters_count} tasks queued)", "INFO")
                    # Send the queued status to the frontend
                    yield pack_message(1, json.dumps({
                        "stage": "queued", 
                        "message": f"Queued... ({waiters_count} tasks ahead)",
                        "queue_position": waiters_count + 1
                    }, ensure_ascii=False).encode('utf-8'))
                
                # Wait for the semaphore (the real queueing happens here)
                print(f"[DEBUG] about to acquire semaphore, task_id={task_id}, waiters={waiters_count}")
                async with translation_semaphore:
                    # With a slot acquired, update the status to running
                    print(f"[DEBUG] semaphore acquired! task_id={task_id}, state updated to running")
                    update_task_status(task_id, "running")
                    add_log("✓ Translation slot acquired; starting translation", "INFO")
                    # Send the notification that a slot was acquired
                    yield pack_message(1, json.dumps({
                        "stage": "slot_acquired", 
                        "message": "Translation slot acquired, starting..."
                    }, ensure_ascii=False).encode('utf-8'))
                    
                    async for chunk in _do_translation():
                        yield chunk
            else:
                async for chunk in _do_translation():
                    yield chunk
        finally:
            # Decrease the task count of the user
            decrement_task_count(username)
    
    async def _do_translation():
        try:
            yield pack_message(1, json.dumps({"stage": "task_id", "task_id": task_id}, ensure_ascii=False).encode('utf-8'))
            
            add_log("Starting translation task", "INFO")
            yield pack_message(1, json.dumps({"stage": "start", "message": "Starting..."}, ensure_ascii=False).encode('utf-8'))
            
            add_log("Loading image", "INFO")
            yield pack_message(1, json.dumps({"stage": "image_loading", "message": "Loading image..."}, ensure_ascii=False).encode('utf-8'))
            pil_image = await to_pil_image(image)
            
            add_log("Preparing translation parameters", "INFO")
            prepare_translator_params(config, workflow)
            
            if is_task_cancelled(task_id):
                add_log("Task cancelled", "WARNING")
                raise asyncio.CancelledError("The task was cancelled by an administrator")
            
            async with with_user_env_vars(config):
                add_log("Using the global translator (reusing models)", "INFO")
                yield pack_message(1, json.dumps({"stage": "translator_init", "message": "Initialising the translator..."}, ensure_ascii=False).encode('utf-8'))
                
                if is_task_cancelled(task_id):
                    raise asyncio.CancelledError("The task was cancelled by an administrator")
                
                add_log("Running translation", "INFO")
                yield pack_message(1, json.dumps({"stage": "translating", "message": "Translating..."}, ensure_ascii=False).encode('utf-8'))
                
                if is_task_cancelled(task_id):
                    raise asyncio.CancelledError("The task was cancelled by an administrator")
                
                try:
                    add_log("Calling translator", "INFO")
                    # Run in the translation thread pool, reusing the global translator
                    from manga_translator.server.core.task_manager import (
                        run_in_translator_thread,
                    )
                    def cancel_callback():
                        return is_task_cancelled(task_id)

                    ctx = await run_in_translator_thread(_run_translate_sync, pil_image, config, task_id, cancel_callback)
                    add_log(f"Translation complete; result available: {ctx.result is not None if hasattr(ctx, 'result') else False}", "INFO")
                    
                    result = {
                        'success': ctx.success if hasattr(ctx, 'success') else (ctx.result is not None),
                        'workflow': workflow
                    }
                    
                    if ctx.result:
                        result['has_image'] = True
                    
                    if hasattr(ctx, 'text_regions') and ctx.text_regions:
                        result['text_regions'] = []
                        for region in ctx.text_regions:
                            region_data = {
                                'text': region.text if hasattr(region, 'text') else '',
                                'translation': region.translation if hasattr(region, 'translation') else '',
                            }
                            result['text_regions'].append(region_data)
                    
                    ctx._workflow_result = result
                    
                    yield pack_message(1, json.dumps({"stage": "translate_done", "message": "Processing result..."}, ensure_ascii=False).encode('utf-8'))
                except Exception as translate_error:
                    print(f"[STREAMING ERROR] Translation failed: {type(translate_error).__name__}")
                    yield pack_message(2, json.dumps({"error": "Translation failed", "stage": "translate"}, ensure_ascii=False).encode('utf-8'))
                    return
            
            has_result = ctx.result is not None if hasattr(ctx, 'result') else False
            has_text_regions = hasattr(ctx, 'text_regions') and ctx.text_regions
            text_region_count = len(ctx.text_regions) if has_text_regions else 0
            
            if has_text_regions:
                yield pack_message(1, json.dumps({
                    "stage": "processing", 
                    "message": f"Found {text_region_count} text regions"
                }, ensure_ascii=False).encode('utf-8'))
            
            if not has_result:
                error_msg = "Translation failed: no result image"
                print(f"[STREAMING ERROR] {error_msg}")
                yield pack_message(2, json.dumps({"error": error_msg, "stage": "no_result"}, ensure_ascii=False).encode('utf-8'))
                return
            
            try:
                yield pack_message(1, json.dumps({"stage": "transforming", "message": "Converting..."}, ensure_ascii=False).encode('utf-8'))
                result_data = transform(ctx)
                
                # Save the translation result to the history
                try:
                    original_filename = getattr(config, '_original_filename', None)
                    await save_translation_to_history(ctx, username, task_id, workflow, original_filename, config)
                    add_log("Translation saved to history", "INFO")
                except Exception as save_error:
                    add_log(f"Failed to save history: {save_error}", "WARNING")
                
                yield pack_message(1, json.dumps({"stage": "sending", "message": "Sending..."}, ensure_ascii=False).encode('utf-8'))
                yield pack_message(0, result_data)
                
                yield pack_message(1, json.dumps({"stage": "complete", "message": "Done!"}, ensure_ascii=False).encode('utf-8'))
            except Exception as transform_error:
                print(f"[STREAMING ERROR] Transform failed: {type(transform_error).__name__}")
                yield pack_message(2, json.dumps({"error": "Transform failed", "stage": "transform"}, ensure_ascii=False).encode('utf-8'))
                return
            
        except asyncio.CancelledError:
            add_log("Task cancelled", "WARNING")
            try:
                yield pack_message(2, json.dumps({"error": "Task cancelled by admin", "stage": "cancelled"}, ensure_ascii=False).encode('utf-8'))
            except Exception:
                pass
        except Exception as e:
            print(f"[STREAMING ERROR] Translation failed: {type(e).__name__}")
            try:
                yield pack_message(2, json.dumps({"error": "Translation failed", "stage": "unknown"}, ensure_ascii=False).encode('utf-8'))
            except Exception:
                pass
        finally:
            add_log("Cleaning up", "DEBUG")
            try:
                # Use the shared Context clean-up function
                if 'ctx' in locals() and ctx:
                    from manga_translator.server.core.task_manager import (
                        cleanup_context,
                    )
                    cleanup_context(ctx)
                
                # Close the input image
                if 'pil_image' in locals() and pil_image:
                    try:
                        pil_image.close()
                    except Exception:
                        pass
                
                # Note: do not call translator.unload_models(),
                # because the global translator is used and the models should be kept for reuse;
                # memory clean-up is done by cleanup_after_request in _run_translate_sync
                
                add_log("Cleanup done", "DEBUG")
            except Exception as cleanup_error:
                add_log(f"Cleanup failed: {cleanup_error}", "WARNING")
            
            # Remove the log handler dedicated to the task
            _remove_task_log_handler(task_log_handler)
            
            unregister_active_task(task_id)
    
    print(f"[DEBUG] while_streaming returns StreamingResponse, task_id={task_id}")
    return StreamingResponse(generate(), media_type="application/octet-stream")


def pack_message(status: int, data: bytes) -> bytes:
    """Pack streaming message: 1 byte status + 4 bytes size + data"""
    return status.to_bytes(1, 'big') + len(data).to_bytes(4, 'big') + data



async def get_batch_ctx(req: Request, config: Config, images: list[str|bytes], batch_size: int = 4, workflow: str = "normal", task_id: str = None):
    """Batch translation (with the logic of the UI layer)

    Args:
        task_id: the task ID, used to check the cancellation state
    """
    from manga_translator.server.core.logging_manager import add_log
    from manga_translator.server.core.task_manager import (
        get_semaphore,
        is_task_cancelled,
    )
    
    # Get the semaphore dynamically (supports hot reloading)
    translation_semaphore = get_semaphore()
    
    pil_images = []
    contexts = []
    
    try:
        # Check whether it was cancelled
        if task_id and is_task_cancelled(task_id):
            raise Exception("Task cancelled")
        
        # Convert images to PIL Image objects
        for img in images:
            # Check the cancel state before converting each image
            if task_id and is_task_cancelled(task_id):
                raise Exception("Task cancelled")
            pil_img = await to_pil_image(img)
            pil_images.append(pil_img)
        
        # Prepare the translator parameters (they affect the workflow behaviour)
        prepare_translator_params(config, workflow)
        
        # Prepare the batch data
        images_with_configs = [(img, config) for img in pil_images]
        
        # Use the shared wrapper for managing environment variables
        async with with_user_env_vars(config):
            # Check the cancel state once more before translating
            if task_id and is_task_cancelled(task_id):
                raise Exception("Task cancelled")
            
            # Wait for a translation slot (the same as the streaming endpoint)
            if translation_semaphore:
                from manga_translator.server.core.task_manager import update_task_status
                try:
                    waiters_count = len(translation_semaphore._waiters) if hasattr(translation_semaphore, '_waiters') and translation_semaphore._waiters else 0
                except Exception:
                    waiters_count = 0
                
                print(f"[DEBUG] get_batch_ctx about to acquire semaphore, task_id={task_id}, waiters={waiters_count}")
                if waiters_count > 0:
                    add_log(f"Batch translation waiting for a slot... ({waiters_count} tasks queued)", "INFO")
                
                async with translation_semaphore:
                    print(f"[DEBUG] get_batch_ctx semaphore acquired! task_id={task_id}")
                    if task_id:
                        update_task_status(task_id, "running")
                    add_log("Batch translation slot acquired; starting execution", "INFO")
                    
                    # Run in the translation thread pool, reusing the global translator
                    from manga_translator.server.core.task_manager import (
                        run_in_translator_thread,
                    )
                    cancel_callback = (lambda: is_task_cancelled(task_id)) if task_id else None
                    contexts = await run_in_translator_thread(
                        _run_translate_batch_sync, images_with_configs, batch_size, task_id, cancel_callback
                    )
            else:
                # Without a semaphore, run directly
                from manga_translator.server.core.task_manager import (
                    run_in_translator_thread,
                )
                cancel_callback = (lambda: is_task_cancelled(task_id)) if task_id else None
                contexts = await run_in_translator_thread(
                    _run_translate_batch_sync, images_with_configs, batch_size, task_id, cancel_callback
                )
            
            # Check the cancel state after translating
            if task_id and is_task_cancelled(task_id):
                raise Exception("Task cancelled")
            
            # Add the workflow result to each context
            for ctx in contexts:
                if ctx:
                    result = {
                        'success': ctx.success if hasattr(ctx, 'success') else (ctx.result is not None),
                        'workflow': workflow
                    }
                    if ctx.result:
                        result['has_image'] = True
                    if hasattr(ctx, 'text_regions') and ctx.text_regions:
                        result['text_regions'] = []
                        for region in ctx.text_regions:
                            region_data = {
                                'text': region.text if hasattr(region, 'text') else '',
                                'translation': region.translation if hasattr(region, 'translation') else '',
                            }
                            result['text_regions'].append(region_data)
                    ctx._workflow_result = result
        
        # Copy the result image before returning, so the clean-up in finally does not affect it
        for ctx in contexts:
            if ctx and hasattr(ctx, 'result') and ctx.result is not None:
                try:
                    ctx.result = ctx.result.copy()
                except Exception:
                    pass  # When the copy fails, keep the original reference
        
        return contexts
    
    finally:
        # Clean up the resources
        try:
            # Clean up the PIL images (the original input images)
            for pil_img in pil_images:
                try:
                    pil_img.close()
                except Exception:
                    pass
            
            # Note: cleaning up the contexts and reclaiming memory is done by cleanup_after_request in _run_translate_batch_sync
            # Do not call translator.unload_models(): the global translator is used and the models should be kept for reuse
            
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up batch translation resources: {cleanup_error}")


async def save_translation_to_history(ctx, username: str, task_id: str, workflow: str, original_filename: str = None, config = None) -> None:
    """
    Save a translation result to the history

    Args:
        ctx: the translation context, with the result image
        username: the user name
        task_id: the task ID
        workflow: the workflow type
        original_filename: the original file name (optional)
        config: the configuration object (optional, used to get the output format)
    """
    import shutil
    import tempfile
    from datetime import datetime, timezone

    from manga_translator.server.core.logging_manager import add_log
    
    add_log(f"Saving translation to history for user: {username}, task: {task_id[:8]}", "DEBUG")
    
    # Check whether ctx is valid
    if not ctx:
        add_log("Cannot save history: ctx is None", "WARNING")
        return
    
    if not hasattr(ctx, 'result') or ctx.result is None:
        add_log("Cannot save history: ctx.result is None", "WARNING")
        return
    
    try:
        # Get the history service
        from manga_translator.server.routes.history import get_history_service
        history_service = get_history_service()
        add_log("History service obtained successfully", "DEBUG")
    except Exception as e:
        add_log(f"History service not available: {e}", "WARNING")
        return
    
    # Save the result image to a temporary file
    temp_dir = None
    temp_files = []
    try:
        # Create a temporary folder
        temp_dir = tempfile.mkdtemp()
        
        # Get the output format from the configuration
        output_format = None
        if config and hasattr(config, 'cli') and hasattr(config.cli, 'format'):
            fmt = config.cli.format
            if fmt and fmt != '不指定':
                output_format = fmt.lower()
        
        # Sanitise the file name, to prevent path traversal attacks
        def sanitize_filename(filename: str) -> str:
            if not filename:
                return None
            # Keep only the file name part, dropping the path
            filename = os.path.basename(filename)
            # Remove dangerous characters
            dangerous_chars = ['..', '/', '\\', '\x00', '<', '>', ':', '"', '|', '?', '*']
            for char in dangerous_chars:
                filename = filename.replace(char, '_')
            # Limit the length
            if len(filename) > 200:
                base, ext = os.path.splitext(filename)
                filename = base[:200-len(ext)] + ext
            return filename if filename else None
        
        # Decide the file name and the save format
        safe_filename = sanitize_filename(original_filename) if original_filename else None
        
        if safe_filename:
            base_name = os.path.splitext(safe_filename)[0]
            save_format, ext = resolve_output_image_format(
                output_format,
                original_path=safe_filename,
            )
            result_filename = f"{base_name}{ext}"
        else:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            save_format, ext = resolve_output_image_format(output_format)
            result_filename = f"translated_{timestamp}{ext}"
        
        result_path = os.path.join(temp_dir, result_filename)
        
        # Save the PIL Image
        if hasattr(ctx.result, 'save'):
            # Copy the image, to avoid the "Operation on closed image" error
            try:
                img_to_save = ctx.result.copy()
            except Exception:
                # When the copy fails, try the original image directly
                img_to_save = ctx.result
            
            if save_format in RGB_PIL_FORMATS and img_to_save.mode not in ('RGB', 'L'):
                img_to_save = normalize_rgb_image(img_to_save)
                add_log(f"Converted {ctx.result.mode} to RGB for {save_format} format", "DEBUG")
            
            img_to_save.save(result_path, save_format)
            temp_files.append(result_path)
            add_log(f"Saved result image to temp: {result_path} (format: {save_format})", "DEBUG")
        else:
            add_log(f"ctx.result does not have save method, type: {type(ctx.result)}", "WARNING")
            return
        
        if not temp_files:
            add_log("No temp files to save", "WARNING")
            return
        
        # Build the metadata
        metadata = {
            'workflow': workflow,
            'task_id': task_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add the text region information
        if hasattr(ctx, 'text_regions') and ctx.text_regions:
            text_data = []
            for region in ctx.text_regions:
                text_data.append({
                    'original': region.text if hasattr(region, 'text') else '',
                    'translated': region.translation if hasattr(region, 'translation') else ''
                })
            metadata['text_regions'] = text_data
            add_log(f"Added {len(text_data)} text regions to metadata", "DEBUG")
        
        # Save to the history
        _result = history_service.save_translation_result(
            user_id=username,
            session_token=task_id,
            files=temp_files,
            metadata=metadata
        )
        add_log(f"Translation saved to history successfully, session: {task_id[:8]}", "INFO")
        
    except Exception as e:
        add_log(f"Failed to save translation to history ({type(e).__name__})", "ERROR")
    finally:
        # Remove the temporary folder
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
