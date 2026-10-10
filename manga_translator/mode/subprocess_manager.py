#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Subprocess manager - memory management and resuming
"""
# import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, 'frozen', False)
    else Path(__file__).resolve().parent.parent.parent
)

# Memory thresholds
DEFAULT_MEMORY_THRESHOLD_MB = 0  # no absolute memory limit by default
DEFAULT_MEMORY_THRESHOLD_PERCENT = 80  # restart at 80% of system memory by default
DEFAULT_BATCH_SIZE_PER_RESTART = 50


def get_memory_usage_mb() -> float:
    """Memory used by this process, in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0


def get_total_memory_mb() -> float:
    """Total system memory, in MB."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 / 1024
    except ImportError:
        return 0


def get_system_memory_percent() -> float:
    """System-wide memory use as a percentage (all processes)."""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return 0





def worker_translate_batch(
    file_paths: List[str],
    output_dir: str,
    verbose: bool,
    overwrite: bool,
    config_dict: dict,
    memory_limit_mb: int,
    memory_limit_percent: int,
    result_queue: multiprocessing.Queue
):
    """
    Worker function run in the child process: translate one batch of images.
    """
    import asyncio
    
    async def _do_translate():
        # Make the project importable
        sys.path.insert(0, str(ROOT_DIR))
        sys.path.insert(0, str(ROOT_DIR / 'desktop_qt_ui'))
        
        import logging

        from desktop_qt_ui.services.translation_setup import build_backend_config, build_save_info
        from manga_translator import MangaTranslator
        from manga_translator.utils import init_logging, set_log_level
        
        init_logging()
        set_log_level(logging.DEBUG if verbose else logging.INFO)
        
        # Apply command-line arguments
        cli_config = config_dict.get('cli', {})
        cli_config['verbose'] = verbose
        cli_config['overwrite'] = overwrite
        config_dict['cli'] = cli_config
        
        font_family = config_dict.get('render', {}).get('font_family')
        if font_family:
            config_dict['font_family'] = font_family
        # Create the translator
        translator_params = cli_config.copy()
        translator_params.update(config_dict)
        translator = MangaTranslator(params=translator_params)
        
        # Same configuration and save options the desktop window builds,
        # including the 'cli' section and the "save beside the source" setting.
        manga_config = build_backend_config(config_dict, str(ROOT_DIR))
        save_info = build_save_info(config_dict, output_dir, [], overwrite=overwrite)
        
        completed = []
        skipped = []
        failed = []
        items = [(file_path, manga_config) for file_path in file_paths]
        contexts = await translator.translate_batch(items, save_info=save_info)

        for ctx in contexts:
            image_name = getattr(ctx, 'image_name', '') or ''
            file_name = os.path.basename(image_name) or 'unknown image'
            if getattr(ctx, 'skipped', False):
                skipped.append(image_name)
                reason = getattr(ctx, 'skip_message', None) or 'The backend skipped this file'
                print(f"⏭️  Skipped: {file_name} - {reason}")
            elif getattr(ctx, 'translation_error', None):
                failed.append(image_name)
                print(f"❌ Failed: {file_name} - {ctx.translation_error}")
            elif getattr(ctx, 'success', False) or getattr(ctx, 'result', None):
                completed.append(image_name)
                print(f"✅ Done: {file_name}")
            else:
                failed.append(image_name)
                print(f"❌ Failed: {file_name} - no result returned")

        return completed, skipped, failed
    
    try:
        completed, skipped, failed = asyncio.run(_do_translate())
        result_queue.put({
            'status': 'success',
            'completed': completed,
            'skipped': skipped,
            'failed': failed,
        })
    except Exception as e:
        import traceback
        print(f"\n❌ Worker process error: {e}")
        result_queue.put({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'completed': [],
            'skipped': [],
            'failed': []
        })


async def translate_with_subprocess(
    all_files: List[str],
    output_dir: str,
    config_dict: dict,
    verbose: bool,
    overwrite: bool,
    memory_limit_mb: int = DEFAULT_MEMORY_THRESHOLD_MB,
    memory_limit_percent: int = DEFAULT_MEMORY_THRESHOLD_PERCENT,
    batch_per_restart: int = DEFAULT_BATCH_SIZE_PER_RESTART,
    resume: bool = False
) -> Tuple[int, int, int]:
    """
    Translate in subprocess mode, with memory management.
    
    Args:
        memory_limit_mb: memory limit in MB, 0 means no limit
        memory_limit_percent: restart the worker when system memory use passes this percentage
    
    Returns:
        (success_count, skipped_count, failed_count)
    """
    completed_files = set()
    total_files = len(all_files)
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Total system memory, for display
    total_mem = get_total_memory_mb()
    
    print(f"\n{'='*60}")
    print("🚀 Subprocess translation mode")
    print(f"📊 Total files: {total_files}")
    # Show the absolute limit when one is set, otherwise the percentage limit
    if memory_limit_mb > 0:
        print(f"📊 Memory limit: {memory_limit_mb} MB")
    elif memory_limit_percent > 0:
        limit_mb = total_mem * memory_limit_percent / 100
        print(f"📊 Memory limit: {memory_limit_percent}% (about {limit_mb:.0f} MB)")
    if batch_per_restart > 0:
        print(f"📊 Images per batch: {batch_per_restart}")
    print(f"{'='*60}\n")
    
    restart_count = 0
    
    while True:
        # At the start of each round, drop the files already finished
        pending_files = [f for f in all_files if f not in completed_files]
        
        if not pending_files:
            break
        
        # Take one batch (0 means no limit: everything at once)
        if batch_per_restart > 0:
            batch_files = pending_files[:batch_per_restart]
        else:
            batch_files = pending_files
        
        print(f"\n{'='*40}")
        print(f"🔄 Batch {restart_count + 1}: {len(batch_files)} files")
        print(f"📊 Progress: {len(completed_files)}/{total_files}")
        print(f"{'='*40}")
        
        result_queue = multiprocessing.Queue()
        
        process = multiprocessing.Process(
            target=worker_translate_batch,
            args=(
                batch_files,
                output_dir,
                verbose,
                overwrite,
                config_dict,
                memory_limit_mb,
                memory_limit_percent,
                result_queue
            )
        )
        
        process.start()
        
        try:
            # Read the result from the queue first (the worker exits after sending it)
            timeout = len(batch_files) * 600
            try:
                result = result_queue.get(timeout=timeout)
                
                if result['status'] == 'success':
                    batch_completed = result.get('completed', [])
                    batch_skipped = result.get('skipped', [])
                    batch_failed = result.get('failed', [])

                    success_count += len(batch_completed)
                    skipped_count += len(batch_skipped)
                    failed_count += len(batch_failed)
                    completed_files.update(batch_completed)
                    completed_files.update(batch_skipped)
                    completed_files.update(batch_failed)

                    print(
                        f"\n📊 Batch finished: {len(batch_completed)} succeeded, "
                        f"{len(batch_skipped)} skipped, {len(batch_failed)} failed"
                    )
                else:
                    print(f"\n❌ Batch error: {result.get('error', 'unknown error')}")
                    if verbose and 'traceback' in result:
                        print(result['traceback'])
                    failed_count += len(batch_files)
                    completed_files.update(batch_files)
                    
            except Exception as e:
                print(f"\n⚠️ Could not read the worker's result: {e}")
                # No result: mark this batch as failed
                failed_count += len(batch_files)
                completed_files.update(batch_files)
            
            # Wait for the worker to exit
            process.join(timeout=30)
            if process.is_alive():
                print("⚠️ The worker did not exit; terminating it")
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by the user")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            raise
        
        main_mem = get_memory_usage_mb()
        if main_mem > 0:
            print(f"📊 Main process memory: {main_mem:.0f} MB")
        
        restart_count += 1
    
    if failed_count == 0:
        print("\n✅ All files processed")
    else:
        print(f"\n⚠️ {failed_count} files failed")
    
    return success_count, skipped_count, failed_count
