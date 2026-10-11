"""
Task management module

Responsible for concurrency control, tracking active tasks and task cancellation.
A ThreadPoolExecutor manages the translation threads; maximum concurrency = maximum number of threads.
One MangaTranslator instance is reused globally, to avoid loading the models repeatedly.
"""

import asyncio
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from manga_translator.server.core.logging_manager import add_log

logger = logging.getLogger('manga_translator.server')


# Translation thread pool (created dynamically from max_concurrent_tasks)
translation_executor: Optional[ThreadPoolExecutor] = None

# Semaphore for concurrency control (limits the number of translation tasks running at once)
translation_semaphore: Optional[asyncio.Semaphore] = None

# Global translator instance (reuses the models, to avoid loading them again)
_global_translator = None
_translator_lock = threading.Lock()
_translator_params_hash = None  # Hash of the current translator's parameters, to decide whether it has to be rebuilt

# Global server configuration (set from the start-up arguments)
server_config = {
    'use_gpu': False,
    'verbose': False,
    'models_ttl': 0,
    'retry_attempts': None,
    'admin_password': None,
    'max_concurrent_tasks': 3,
}

# Tracking of active tasks
active_tasks = {}
active_tasks_lock = threading.Lock()


def init_semaphore():
    """Initialise the concurrency control semaphore and the thread pool"""
    global translation_semaphore, translation_executor
    
    max_concurrent = server_config.get('max_concurrent_tasks', 3)
    logger.info(f"[init_semaphore] Read max_concurrent_tasks = {max_concurrent} from server_config")
    
    # Shut down the old thread pool (when there is one)
    if translation_executor is not None:
        translation_executor.shutdown(wait=False)
    
    # Create a new thread pool: maximum threads = maximum concurrent tasks
    translation_executor = ThreadPoolExecutor(
        max_workers=max_concurrent,
        thread_name_prefix="translator_"
    )
    
    # Create the semaphore for asynchronous waiting
    translation_semaphore = asyncio.Semaphore(max_concurrent)
    
    logger.info(f"Translation thread pool initialized: maximum threads = {max_concurrent}")


def get_semaphore() -> Optional[asyncio.Semaphore]:
    """Get the concurrency control semaphore"""
    return translation_semaphore


def get_executor() -> Optional[ThreadPoolExecutor]:
    """Get the translation thread pool"""
    return translation_executor


async def run_in_translator_thread(func: Callable, *args, **kwargs) -> Any:
    """
    Run a function in the translation thread pool, without blocking the event loop.
    """
    if translation_executor is None:
        init_semaphore()
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        translation_executor,
        lambda: func(*args, **kwargs)
    )


def register_active_task(
    task_id: str, 
    task: Optional[asyncio.Task] = None,
    username: Optional[str] = None,
    translator: Optional[str] = None,
    future: Optional[Future] = None,
    status: str = "queued"
):
    """Register an active task; the default state is queued"""
    with active_tasks_lock:
        active_tasks[task_id] = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "cancel_requested": False,
            "task": task,
            "future": future,
            "username": username or "unknown",
            "translator": translator or "unknown",
            "thread_id": None
        }


def update_task_status(task_id: str, status: str):
    """Update the state of a task (queued -> running -> completed)"""
    with active_tasks_lock:
        if task_id in active_tasks:
            active_tasks[task_id]["status"] = status


def update_task_thread_id(task_id: str, thread_id: int):
    """Update the thread ID of a task"""
    with active_tasks_lock:
        if task_id in active_tasks:
            active_tasks[task_id]["thread_id"] = thread_id


def unregister_active_task(task_id: str):
    """Unregister an active task"""
    with active_tasks_lock:
        if task_id in active_tasks:
            del active_tasks[task_id]


def get_active_tasks() -> list:
    """Get all active tasks"""
    with active_tasks_lock:
        tasks = []
        for task_id, info in active_tasks.items():
            start_time = info["start_time"]
            if isinstance(start_time, str):
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            else:
                start_dt = start_time
            
            now = datetime.now(timezone.utc)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            
            task_data = {
                "task_id": task_id,
                "start_time": info["start_time"],
                "status": info["status"],
                "duration": (now - start_dt).total_seconds(),
                "username": info.get("username", "unknown"),
                "translator": info.get("translator", "unknown"),
                "thread_id": info.get("thread_id")
            }
            tasks.append(task_data)
        return tasks


def is_task_cancelled(task_id: str) -> bool:
    """Check whether a task was cancelled"""
    with active_tasks_lock:
        if task_id in active_tasks:
            return active_tasks[task_id].get("cancel_requested", False)
        return False


def cancel_task(task_id: str, force: bool = False) -> dict:
    """Cancel the given translation task"""
    with active_tasks_lock:
        if task_id in active_tasks:
            active_tasks[task_id]["cancel_requested"] = True
            
            if force:
                task = active_tasks[task_id].get("task")
                future = active_tasks[task_id].get("future")
                cancelled = False
                
                if task and not task.done():
                    task.cancel()
                    cancelled = True
                
                if future and not future.done():
                    future.cancel()
                    cancelled = True
                
                if cancelled:
                    add_log(f"Administrator forcibly cancelled task: {task_id[:8]}", "WARNING")
                    return {"success": True, "message": "任务已强制终止"}
                else:
                    add_log(f"Administrator requested forced task cancellation, but the task has already completed: {task_id[:8]}", "INFO")
                    return {"success": True, "message": "任务已完成，无需取消"}
            else:
                add_log(f"Administrator requested task cancellation: {task_id[:8]}", "WARNING")
                return {"success": True, "message": "取消请求已发送（协作式取消）"}
        else:
            return {"success": False, "message": "任务不存在或已完成"}


def update_server_config(config: dict):
    """Update the server configuration"""
    global _global_translator, _translator_params_hash
    
    # Check whether the translator has to be rebuilt
    rebuild_translator = False
    key_params = ['use_gpu', 'verbose', 'models_ttl']
    for key in key_params:
        if key in config and config[key] != server_config.get(key):
            rebuild_translator = True
            break
    
    if 'max_concurrent_tasks' in config:
        old_value = server_config.get('max_concurrent_tasks', 3)
        new_value = config['max_concurrent_tasks']
        server_config['max_concurrent_tasks'] = new_value
        
        if old_value != new_value:
            init_semaphore()
            logger.info(f"Concurrency updated: {old_value} -> {new_value}")
    
    for key in ['use_gpu', 'verbose', 'models_ttl', 'retry_attempts', 'admin_password']:
        if key in config:
            server_config[key] = config[key]
    
    # When key parameters changed, reset the global translator
    if rebuild_translator and _global_translator is not None:
        with _translator_lock:
            logger.info("Server configuration changed; resetting the global translator...")
            _global_translator = None
            _translator_params_hash = None


def get_server_config() -> dict:
    """Get the server configuration"""
    return server_config.copy()


def get_thread_pool_status() -> dict:
    """Get the status of the thread pool"""
    if translation_executor is None:
        return {"initialized": False, "max_workers": 0, "active_threads": 0}
    
    with active_tasks_lock:
        active_count = len(active_tasks)
    
    return {
        "initialized": True,
        "max_workers": server_config.get('max_concurrent_tasks', 3),
        "active_tasks": active_count,
        "translator_loaded": _global_translator is not None
    }


def shutdown_executor():
    """Shut down the thread pool and the translator (called when the server shuts down)"""
    global translation_executor, _global_translator
    
    if translation_executor is not None:
        logger.info("Shutting down the translation thread pool...")
        translation_executor.shutdown(wait=True)
        translation_executor = None
    
    if _global_translator is not None:
        logger.info("Unloading the global translator...")
        with _translator_lock:
            _global_translator = None
    
    logger.info("Resource cleanup complete")



# ============================================================================
# Management of the global translator instance (reuses the models, to avoid loading them again)
# ============================================================================

def _get_params_hash(params: dict) -> str:
    """Compute the hash of the parameters, used to decide whether the translator has to be rebuilt"""
    key_params = ['use_gpu', 'verbose', 'models_ttl']
    values = tuple(params.get(k) for k in key_params)
    return str(values)


def get_global_translator(params: dict = None):
    """
    Get the global translator instance, reusing the models to avoid loading them repeatedly.

    How the models are reused:
    1. MangaTranslator caches the loaded models internally (OCR, detector, inpainter and so on)
    2. by reusing the same MangaTranslator instance, the models only have to be loaded once
    3. a different Config is passed in for each translation, and the translator chooses the matching models from the configuration
    4. the models_ttl parameter controls how long the models stay in memory

    Args:
        params: the translator parameters (use_gpu, verbose, models_ttl and so on)

    Returns:
        The MangaTranslator instance
    """
    global _global_translator, _translator_params_hash
    
    from manga_translator import MangaTranslator
    
    # Without arguments, use the server configuration
    if params is None:
        params = {
            'use_gpu': server_config.get('use_gpu', False),
            'verbose': server_config.get('verbose', False),
            'models_ttl': server_config.get('models_ttl', 0),
        }
        retry_attempts = server_config.get('retry_attempts')
        if retry_attempts is not None:
            params['attempts'] = retry_attempts
    
    params_hash = _get_params_hash(params)
    
    with _translator_lock:
        # Check whether the translator has to be rebuilt
        if _global_translator is None or _translator_params_hash != params_hash:
            if _global_translator is not None:
                logger.info("Translator parameters changed; rebuilding the instance...")
            else:
                logger.info(f"Creating global translator instance (GPU={params.get('use_gpu')}, models_ttl={params.get('models_ttl')}s)...")
            
            _global_translator = MangaTranslator(params=params)
            _translator_params_hash = params_hash
            logger.info("Global translator instance created; models will be loaded on demand and cached")
        
        return _global_translator


def reset_global_translator():
    """
    Reset the global translator (for an administrator to free memory by hand)
    """
    global _global_translator, _translator_params_hash
    
    with _translator_lock:
        if _global_translator is not None:
            logger.info("Resetting the global translator...")
            try:
                if hasattr(_global_translator, 'unload_models'):
                    _global_translator.unload_models()
            except Exception as e:
                logger.warning(f"Error unloading models: {e}")
            
            _global_translator = None
            _translator_params_hash = None
            
            # Force garbage collection
            import gc
            gc.collect()
            # Free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU memory cleared")
            except Exception:
                pass
            
            logger.info("Global translator reset")
            return {"success": True, "message": "翻译器已重置，模型已卸载"}
        else:
            return {"success": True, "message": "翻译器未初始化，无需重置"}


def get_translator_status() -> dict:
    """
    Get the status of the translator
    """
    with _translator_lock:
        if _global_translator is None:
            return {
                "initialized": False,
                "models_loaded": []
            }
        
        # Get the information of the loaded models
        models_loaded = []
        if hasattr(_global_translator, '_model_usage_timestamps'):
            for (tool, model), timestamp in _global_translator._model_usage_timestamps.items():
                models_loaded.append({
                    "tool": tool,
                    "model": model,
                    "last_used": timestamp
                })
        
        return {
            "initialized": True,
            "models_loaded": models_loaded,
            "models_ttl": server_config.get('models_ttl', 0),
            "use_gpu": server_config.get('use_gpu', False)
        }


def cleanup_after_request():
    """
    Request-level memory clean-up (called after each translation request ends)

    Clears the intermediate state inside the translator but keeps the loaded models.
    This is the core function for the problem of the Web UI not releasing memory.

    What is cleared:
    - the batch context cache of the translator
    - the image contexts (MD5 cache and so on)
    - the page translation history
    - other intermediate state
    """
    import gc
    
    with _translator_lock:
        if _global_translator is not None:
            logger.debug("[MEMORY] Starting request-level memory cleanup...")
            
            try:
                # 1. Clear the batch contexts
                if hasattr(_global_translator, '_batch_contexts'):
                    _global_translator._batch_contexts.clear()
                if hasattr(_global_translator, '_batch_configs'):
                    _global_translator._batch_configs.clear()
                
                # 2. Clear the image context cache
                if hasattr(_global_translator, '_current_image_context'):
                    _global_translator._current_image_context = None
                if hasattr(_global_translator, '_saved_image_contexts'):
                    _global_translator._saved_image_contexts.clear()
                
                # 3. Clear the page translation history
                if hasattr(_global_translator, 'all_page_translations'):
                    _global_translator.all_page_translations.clear()
                if hasattr(_global_translator, '_original_page_texts'):
                    _global_translator._original_page_texts.clear()
                if hasattr(_global_translator, '_clear_colorizer_history'):
                    _global_translator._clear_colorizer_history()
                
                # 4. Clear the cancel callback
                if hasattr(_global_translator, '_cancel_check_callback'):
                    _global_translator._cancel_check_callback = None
                
                logger.debug("[MEMORY] Translator internal state cleared")
                
            except Exception as e:
                logger.warning(f"[MEMORY] Error clearing translator state: {e}")
    
    # 5. Force garbage collection
    gc.collect()
    # 6. Free GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    # 7. Windows only: force the physical memory to be released
    try:
        import ctypes
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except Exception:
        pass
    
    logger.debug("[MEMORY] Request-level memory cleanup complete")


def cleanup_context(ctx):
    """
    Clear all resources in a Context object thoroughly

    Args:
        ctx: the translation context object
        keep_result: whether result is kept (to be returned to the front end)
    """
    if ctx is None:
        return
    
    import gc
    
    # List of all attributes to clear
    attrs_to_clear = [
        # Image data (the largest memory use)
        'input', 'img_rgb', 'img_alpha', 'img_colorized', 'upscaled',
        'img_inpainted', 'img_rendered', 'mask', 'mask_raw', 'bubble_mask',
        # Data of high-quality translation
        'high_quality_batch_data', 'annotated_image',
        # Other objects that may be large
        'textlines', 'text_regions',
        # Workflow results (can be cleared once the frontend has taken them)
        '_workflow_result',
    ]
    
    for attr in attrs_to_clear:
        if hasattr(ctx, attr):
            obj = getattr(ctx, attr)
            if obj is not None:
                # A PIL Image is closed first
                if hasattr(obj, 'close'):
                    try:
                        obj.close()
                    except Exception:
                        pass
                # A list is emptied
                elif isinstance(obj, list):
                    obj.clear()
                # A dictionary is emptied
                elif isinstance(obj, dict):
                    obj.clear()
                # Delete the reference
                try:
                    delattr(ctx, attr)
                except Exception:
                    setattr(ctx, attr, None)
    
    # result is handled separately (it usually has to be kept for the frontend)
    # The caller is responsible for calling this function to clean up after it has used result
    
    gc.collect()
