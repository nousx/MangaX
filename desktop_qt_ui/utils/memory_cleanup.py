"""
Memory clean-up module

One place for memory clean-up, used to free the memory the models hold once translation is done.
Tuned for CPU mode in particular, where the model weights take up RAM directly.
"""
import logging

logger = logging.getLogger(__name__)


def cleanup_all_model_caches(unload_models: bool = False) -> int:
    """
    Clear the model caches of all modules, and optionally unload the model instances explicitly

    Args:
        unload_models: whether the model instances are unloaded explicitly (False by default)
                      - True: unload the models and clear the caches (frees more memory, but they have to be loaded again next time)
                      - False: only clear the cache dictionaries (the model instances are kept, so the next use is faster)

    Returns:
        The number of caches cleared
    """
    cleanup_count = 0
    
    # Clear the translator cache
    try:
        from manga_translator.translators import translator_cache
        if translator_cache:
            if unload_models:
                # Unload all model instances first
                for model_instance in list(translator_cache.values()):
                    try:
                        if hasattr(model_instance, '_unload'):
                            model_instance._unload()
                        elif hasattr(model_instance, 'unload'):
                            model_instance.unload()
                        # Delete the model reference
                        del model_instance
                    except Exception as e:
                        logger.debug(f"Error unloading translator model: {e}")
            
            cleanup_count += len(translator_cache)
            translator_cache.clear()
    except Exception as e:
        logger.debug(f"Error clearing translator cache: {e}")
    
    # Clear the OCR cache
    try:
        from manga_translator.ocr import ocr_cache
        if ocr_cache:
            if unload_models:
                # Unload all model instances first
                for model_instance in list(ocr_cache.values()):
                    try:
                        if hasattr(model_instance, '_unload'):
                            model_instance._unload()
                        elif hasattr(model_instance, 'unload'):
                            model_instance.unload()
                        del model_instance
                    except Exception as e:
                        logger.debug(f"Error unloading OCR model: {e}")
            
            cleanup_count += len(ocr_cache)
            ocr_cache.clear()
    except Exception as e:
        logger.debug(f"Error clearing OCR cache: {e}")
    
    # Clear the detector cache
    try:
        from manga_translator.detection import detector_cache
        if detector_cache:
            if unload_models:
                # Unload all model instances first
                for model_instance in list(detector_cache.values()):
                    try:
                        if hasattr(model_instance, '_unload'):
                            model_instance._unload()
                        elif hasattr(model_instance, 'unload'):
                            model_instance.unload()
                        del model_instance
                    except Exception as e:
                        logger.debug(f"Error unloading detector model: {e}")
            
            cleanup_count += len(detector_cache)
            detector_cache.clear()
    except Exception as e:
        logger.debug(f"Error clearing detector cache: {e}")
    
    # Clear the inpainter cache
    try:
        from manga_translator.inpainting import inpainter_cache
        if inpainter_cache:
            if unload_models:
                # Unload all model instances first
                for model_instance in list(inpainter_cache.values()):
                    try:
                        if hasattr(model_instance, '_unload'):
                            model_instance._unload()
                        elif hasattr(model_instance, 'unload'):
                            model_instance.unload()
                        del model_instance
                    except Exception as e:
                        logger.debug(f"Error unloading inpainter model: {e}")
            
            cleanup_count += len(inpainter_cache)
            inpainter_cache.clear()
    except Exception as e:
        logger.debug(f"Error clearing inpainter cache: {e}")
    
    # Clear the upscaler cache
    try:
        from manga_translator.upscaling import upscaler_cache
        if upscaler_cache:
            if unload_models:
                # Unload all model instances first
                for model_instance in list(upscaler_cache.values()):
                    try:
                        if hasattr(model_instance, '_unload'):
                            model_instance._unload()
                        elif hasattr(model_instance, 'unload'):
                            model_instance.unload()
                        del model_instance
                    except Exception as e:
                        logger.debug(f"Error unloading upscaler model: {e}")
            
            cleanup_count += len(upscaler_cache)
            upscaler_cache.clear()
    except Exception as e:
        logger.debug(f"Error clearing upscaler cache: {e}")
    
    # Clear the colorizer cache
    try:
        from manga_translator.colorization import colorizer_cache
        if colorizer_cache:
            if unload_models:
                # Unload all model instances first
                for model_instance in list(colorizer_cache.values()):
                    try:
                        if hasattr(model_instance, '_unload'):
                            model_instance._unload()
                        elif hasattr(model_instance, 'unload'):
                            model_instance.unload()
                        del model_instance
                    except Exception as e:
                        logger.debug(f"Error unloading colorizer model: {e}")
            
            cleanup_count += len(colorizer_cache)
            colorizer_cache.clear()
    except Exception as e:
        logger.debug(f"Error clearing colorizer cache: {e}")
    
    return cleanup_count


def cleanup_gpu_memory():
    """
    Free GPU memory and the internal caches of PyTorch
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Empty the CUDA cache
            pass
            pass
            # Clear the CUDA memory pool
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except Exception:
                pass
            
            return True
    except Exception:
        pass
    return False


def cleanup_physical_memory():
    """
    Release physical memory (Windows only)

    On Windows EmptyWorkingSet is called first to force the physical memory to be released,
    then it falls back to SetProcessWorkingSetSize.
    Especially important in CPU mode.
    """
    try:
        import ctypes
        import os

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        process_handle = kernel32.OpenProcess(0x0400 | 0x0100, False, os.getpid())
        if not process_handle:
            return False

        try:
            if hasattr(psapi, "EmptyWorkingSet") and psapi.EmptyWorkingSet(process_handle):
                return True
            if kernel32.SetProcessWorkingSetSize(process_handle, -1, -1):
                return True
        finally:
            kernel32.CloseHandle(process_handle)
    except Exception:
        pass  # Ignored on systems other than Windows
    return False


def full_memory_cleanup(log_callback=None, unload_models: bool = False):
    """
    Run the full memory clean-up

    Args:
        log_callback: log callback that receives string messages (deprecated; the parameter is kept for compatibility)
        unload_models: whether the model instances are unloaded explicitly (False by default)
                      - True: unload the models and clear the caches (frees more memory, but they have to be loaded again next time)
                      - False: only clear the cache dictionaries (the model instances are kept, so the next use is faster)

    Returns:
        dict: a dictionary with the clean-up results
    """
    result = {
        'caches_cleared': 0,
        'gpu_cleared': False,
        'physical_memory_released': False,
        'models_unloaded': unload_models
    }
    
    # 1. Clear the model caches
    result['caches_cleared'] = cleanup_all_model_caches(unload_models=unload_models)
    
    # 2. Force garbage collection (run several times, to be thorough)
    pass
    pass
    pass
    # 3. Free GPU memory
    result['gpu_cleared'] = cleanup_gpu_memory()
    
    # 4. Release physical memory
    result['physical_memory_released'] = cleanup_physical_memory()
    
    return result


