#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line translation tool - uses the same translation logic as the desktop window
Supports a subprocess mode for memory management and resuming
"""
import argparse
import asyncio
import multiprocessing
import os
import sys
from pathlib import Path

# Put the project root on the Python path
ROOT_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, 'frozen', False)
    else Path(__file__).resolve().parent.parent.parent
)
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'desktop_qt_ui'))

# Memory management defaults
DEFAULT_MEMORY_THRESHOLD_MB = 8000  # 8 GB by default
DEFAULT_BATCH_SIZE_PER_RESTART = 50  # check after this many images


def parse_args():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Manga translation command-line tool - same translation logic as the desktop window',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate one image
  python -m manga_translator local -i manga.jpg
  
  # Translate a folder
  python -m manga_translator local -i ./manga_folder/ -o ./output/
  
  # Use a custom settings file
  python -m manga_translator local -i manga.jpg --config my_config.json
  
  # Subprocess mode (restarts the worker every 50 images to release memory)
  python -m manga_translator local -i ./manga_folder/ --subprocess
  
  # Custom memory management (restart every 20 images)
  python -m manga_translator local -i ./manga_folder/ --subprocess --batch-per-restart 20
  
  # Continue from where the last run stopped (needs --subprocess)
  python -m manga_translator local -i ./manga_folder/ --subprocess --resume
  
  # Detailed logs
  python -m manga_translator local -i manga.jpg -v
        """
    )
    
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='Input image or folder paths')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder (default: the input folder name with a -translated suffix)')
    parser.add_argument('--config', default=None,
                        help='Settings file path (default: config/config.json)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed logs')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip pages whose output already exists (continue an unfinished run)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    
    # Memory management
    parser.add_argument('--subprocess', action='store_true',
                        help='Run in subprocess mode (adds memory management and resuming)')
    parser.add_argument('--memory-limit', type=int, default=DEFAULT_MEMORY_THRESHOLD_MB,
                        help=f'Memory limit in MB; the worker restarts when it is exceeded (default: {DEFAULT_MEMORY_THRESHOLD_MB}, 0 means no limit)')
    parser.add_argument('--memory-percent', type=int, default=80,
                        help='Memory limit as a percentage of system memory; the worker restarts when it is exceeded (default: 80)')
    parser.add_argument('--batch-per-restart', type=int, default=DEFAULT_BATCH_SIZE_PER_RESTART,
                        help=f'Restart the worker after this many images to release memory (default: {DEFAULT_BATCH_SIZE_PER_RESTART})')
    parser.add_argument('--resume', action='store_true',
                        help='Continue from where the last run stopped (needs --subprocess)')
    
    # Concurrent mode
    parser.add_argument('--concurrent', action='store_true',
                        help='Run detection, OCR, translation and rendering as a concurrent pipeline')
    
    return parser.parse_args()




async def translate_files(input_paths, output_dir, config_service, verbose=False, overwrite=False, args=None):
    """Translate files with the same logic as the desktop window."""
    
    # Imported late so --help does not load every module
    import logging
    import logging.handlers

    from desktop_qt_ui.services.file_service import FileService
    from desktop_qt_ui.services.translation_setup import build_backend_config, build_save_info
    from manga_translator import MangaTranslator
    from manga_translator.utils import (
        get_logger,
        init_logging,
        set_log_level,
    )
    
    init_logging()
    if verbose:
        set_log_level(logging.DEBUG)
    else:
        set_log_level(logging.INFO)
    
    # Send manga_translator logs to the console as well
    manga_logger = logging.getLogger('manga_translator')
    manga_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Add a console handler if there is none yet
    if not any(isinstance(h, logging.StreamHandler) for h in manga_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter('[%(name)s] %(message)s')
        console_handler.setFormatter(formatter)
        manga_logger.addHandler(console_handler)
    
    # Add a log file (same place and format as the desktop window)
    from datetime import datetime
    
    log_dir = ROOT_DIR / 'result'
    log_dir.mkdir(exist_ok=True)
    
    # Time-stamped log file name (same format as the desktop window)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = log_dir / f'log_{timestamp}.txt'
    
    # Check whether a file handler was already added
    has_file_handler = any(
        isinstance(h, logging.FileHandler)
        for h in logging.root.handlers
    )
    
    if not has_file_handler:
        file_handler = logging.FileHandler(
            str(log_file),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)
        print(f"📝 Log file: {log_file}")
    
    logger = get_logger('local')
    
    # Read the settings
    config = config_service.get_config()
    config_dict = config.model_dump()
    
    # CLI settings come from the settings file; command-line arguments override them
    cli_config = config_dict.get('cli', {})
    
    # Apply command-line arguments (they override the settings file)
    if verbose:
        cli_config['verbose'] = True
    else:
        verbose = cli_config.get('verbose', False)
    
    # overwrite: the command line wins, otherwise the settings file
    if getattr(args, 'skip_existing', False):
        overwrite = False
        cli_config['overwrite'] = False
    elif overwrite:
        cli_config['overwrite'] = True
    else:
        overwrite = cli_config.get('overwrite', False)
    
    # use_gpu: the command line wins
    if hasattr(args, 'use_gpu') and args.use_gpu is not None:
        cli_config['use_gpu'] = args.use_gpu

    # disable_onnx_gpu: the command line wins
    if hasattr(args, 'disable_onnx_gpu') and args.disable_onnx_gpu is not None:
        cli_config['disable_onnx_gpu'] = args.disable_onnx_gpu
    
    # format: the command line wins
    if hasattr(args, 'format') and args.format is not None:
        cli_config['format'] = args.format
    
    # batch_size: the command line wins
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        cli_config['batch_size'] = args.batch_size
    
    # attempts: the command line wins
    if hasattr(args, 'attempts') and args.attempts is not None:
        cli_config['attempts'] = args.attempts
    
    # concurrent: the command line wins, otherwise the settings file
    if hasattr(args, 'concurrent') and args.concurrent:
        cli_config['batch_concurrent'] = True
    # Not given on the command line: keep batch_concurrent from the settings file (already in cli_config)
    
    config_dict['cli'] = cli_config
    
    
    print(f"\n{'='*60}")
    print(f"Translator: {config_dict['translator']['translator']}")
    print(f"Target language: {config_dict['translator']['target_lang']}")
    print(f"Use GPU: {cli_config.get('use_gpu', True)}")
    print(f"ONNX GPU disabled: {cli_config.get('disable_onnx_gpu', False)}")
    print(f"Batch size: {cli_config.get('batch_size', 1)}")
    print(f"Concurrent mode: {'on' if cli_config.get('batch_concurrent', False) else 'off'}")
    print(f"Overwrite existing files: {overwrite}")
    print(f"Output format: {cli_config.get('format') or 'keep original'}")
    print(f"Save quality: {cli_config.get('save_quality', 95)}")
    print(f"{'='*60}\n")
    
    # Collect every image file
    file_service = FileService()
    all_files = []
    
    # Separate files from folders
    folders = []
    individual_files = []
    
    for input_path in input_paths:
        input_path = os.path.abspath(input_path)
        if os.path.isfile(input_path):
            individual_files.append(input_path)
        elif os.path.isdir(input_path):
            folders.append(input_path)
    
    # Natural-sort the folders (same as the desktop window)
    folders.sort(key=file_service._natural_sort_key)
    
    # Handle the files folder by folder
    for folder in folders:
        # Every image in the folder, recursively (already natural-sorted)
        folder_files = file_service.get_image_files_from_folder(folder, recursive=True)
        all_files.extend(folder_files)
    
    # Files given one by one (natural-sorted)
    individual_files.sort(key=file_service._natural_sort_key)
    all_files.extend(individual_files)
    
    if not all_files:
        print("❌ No image files found")
        return
    
    print(f"📁 Found {len(all_files)} image files\n")
    
    # Decide the output folder
    if output_dir:
        final_output_dir = os.path.abspath(output_dir)
    else:
        # Use the output folder from the settings file, or the default rule
        if config_dict.get('app', {}).get('last_output_path'):
            final_output_dir = config_dict['app']['last_output_path']
        else:
            # Default: a -translated folder beside the first input path
            first_input = input_paths[0]
            if os.path.isdir(first_input):
                final_output_dir = first_input.rstrip('/\\') + '-translated'
            else:
                final_output_dir = os.path.dirname(first_input)
    
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"📤 Output folder: {final_output_dir}\n")
    
    # Prepare translation parameters (as the desktop window does)
    translator_params = config_dict.get('cli', {}).copy()
    # Keep key cli values so they are not overwritten
    cli_attempts = translator_params.get('attempts', -1)
    translator_params.update(config_dict)
    # Restore the cli value (when config_dict has no attempts)
    if 'attempts' not in config_dict:
        translator_params['attempts'] = cli_attempts
    
    font_family = config_dict.get('render', {}).get('font_family')
    if font_family:
        translator_params['font_family'] = font_family
    # Create the translator
    print("🔧 Starting the translator...")
    translator = MangaTranslator(params=translator_params)
    print("✅ Translator ready")
    
    # Same configuration the desktop window builds, including the 'cli' section.
    manga_config = build_backend_config(config_dict, str(ROOT_DIR), warn=logger.warning)
    
    # Input folders, used to keep the folder structure
    input_folders = set()
    for input_path in input_paths:
        if os.path.isdir(input_path):
            input_folders.add(os.path.normpath(os.path.abspath(input_path)))
    
    print("\n📁 Preparing the image list...")
    # Only paths are kept; image data is not loaded here
    file_paths_with_configs = []
    for file_path in all_files:
        # Only check the file can be read; do not load it
        try:
            if os.path.exists(file_path) and os.path.isfile(file_path):
                file_paths_with_configs.append((file_path, manga_config))
            else:
                print(f"❌ File does not exist: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Cannot access: {os.path.basename(file_path)} - {e}")
    
    if not file_paths_with_configs:
        print("No images to translate")
        return
    
    save_info = build_save_info(config_dict, final_output_dir, input_folders, overwrite=overwrite)
    output_format = save_info['format']
    
    # Make sure the output folder exists
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir, exist_ok=True)
        print(f"✅ Created output folder: {final_output_dir}")
    

    batch_size = cli_config.get('batch_size', 3)
    total_images = len(file_paths_with_configs)
    total_batches = (total_images + batch_size - 1) // batch_size if batch_size > 0 else 1
    
    print(f"\n📊 Batch mode: {total_images} images in {total_batches} batches")
    print("📋 Save settings:")
    print(f"   Output folder: {final_output_dir}")
    print(f"   Output format: {output_format or 'keep original'}")
    print(f"   Overwrite: {overwrite}")
    print(f"   Save quality: {cli_config.get('save_quality', 95)}")
    print(f"   Batch size: {batch_size} images per batch")
    if verbose and input_folders:
        print("   Input folders:")
        for folder in input_folders:
            print(f"      - {folder}")
    print()
    
    try:
        print("🚀 Translating...")
        contexts = await translator.translate_batch(
            file_paths_with_configs,
            save_info=save_info,
        )

        success_count = 0
        skipped_count = 0
        failed_count = 0
        print("\n📊 Translation finished, checking results...\n")
        logger.info(f"Received {len(contexts)} translation results")

        for ctx in contexts:
            if not ctx:
                failed_count += 1
                print("❌ Translation failed: unknown image")
                continue

            image_name = getattr(ctx, 'image_name', '') or ''
            file_name = os.path.basename(image_name) or 'unknown image'
            if getattr(ctx, 'skipped', False):
                skipped_count += 1
                reason = getattr(ctx, 'skip_message', None) or 'The backend skipped this file'
                print(f"⏭️  Skipped: {file_name} - {reason}")
            elif getattr(ctx, 'translation_error', None):
                failed_count += 1
                print(f"❌ Translation failed: {file_name}")
                if verbose:
                    print(f"   Error: {ctx.translation_error}")
            elif getattr(ctx, 'success', False) or getattr(ctx, 'result', None):
                success_count += 1
                output_path = getattr(ctx, 'output_path', None)
                print(f"✅ Done: {file_name}" + (f" -> {output_path}" if output_path else ""))
            else:
                failed_count += 1
                print(f"❌ Translation failed: {file_name} - the result is empty")

        print(
            f"\n📊 Batch finished: {success_count} succeeded, "
            f"{skipped_count} skipped, {failed_count} failed."
        )
        print(f"💾 Files saved to: {final_output_dir}")
    except Exception as e:
        print(f"\n❌ Batch translation error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        success_count = 0
        skipped_count = 0
        failed_count = total_images

    print(f"\n{'='*60}")
    print(f"✅ Succeeded: {success_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📊 Total: {len(all_files)}")
    print(f"{'='*60}")
    
    # Check the output folder
    if os.path.exists(final_output_dir):
        output_files = [f for f in os.listdir(final_output_dir) if os.path.isfile(os.path.join(final_output_dir, f))]
        print(f"\n📁 Output folder: {final_output_dir}")
        print(f"   Contains {len(output_files)} files")
        if verbose and output_files:
            for f in output_files[:10]:  # show the first 10 only
                file_path = os.path.join(final_output_dir, f)
                file_size = os.path.getsize(file_path) / 1024
                print(f"   - {f} ({file_size:.1f} KB)")
            if len(output_files) > 10:
                print(f"   ... and {len(output_files) - 10} more files")
    else:
        print(f"\n⚠️  Output folder does not exist: {final_output_dir}")
    print()


async def run_local_mode(args):
    """Entry point of local mode."""
    # Import the settings service late
    from desktop_qt_ui.services.config_service import ConfigService
    from desktop_qt_ui.services.file_service import FileService
    
    # Create the settings service
    config_service = ConfigService(str(ROOT_DIR))
    
    # Load the settings file when one is given
    config_path = getattr(args, 'config', None)
    if config_path:
        if not config_service.load_config_file(config_path):
            print(f"❌ Could not load the settings file: {config_path}")
            sys.exit(1)
    
    # Is subprocess mode requested?
    use_subprocess = getattr(args, 'subprocess', False)
    verbose = getattr(args, 'verbose', False)
    overwrite = getattr(args, 'overwrite', False)
    
    if use_subprocess:
        # Subprocess mode
        print("\n🔧 Subprocess mode on (with memory management)")
        
        # Collect the files
        file_service = FileService()
        all_files = []
        input_paths = args.input
        
        folders = []
        individual_files = []
        
        for input_path in input_paths:
            input_path = os.path.abspath(input_path)
            if os.path.isfile(input_path):
                individual_files.append(input_path)
            elif os.path.isdir(input_path):
                folders.append(input_path)
        
        folders.sort(key=file_service._natural_sort_key)
        for folder in folders:
            folder_files = file_service.get_image_files_from_folder(folder, recursive=True)
            all_files.extend(folder_files)
        
        individual_files.sort(key=file_service._natural_sort_key)
        all_files.extend(individual_files)
        
        if not all_files:
            print("❌ No image files found")
            sys.exit(1)
        
        print(f"📁 Found {len(all_files)} image files")
        
        # Decide the output folder
        output_dir = getattr(args, 'output', None)
        if not output_dir:
            config = config_service.get_config()
            if config.app.last_output_path:
                output_dir = config.app.last_output_path
            else:
                first_input = input_paths[0]
                if os.path.isdir(first_input):
                    output_dir = first_input.rstrip('/\\') + '-translated'
                else:
                    output_dir = os.path.dirname(first_input)
        
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"📤 Output folder: {output_dir}")
        

        # Import the subprocess manager
        from .subprocess_manager import translate_with_subprocess
        
        try:
            config_dict = config_service.get_config().model_dump()
            cli_config = config_dict.get('cli', {})
            if hasattr(args, 'use_gpu') and args.use_gpu is not None:
                cli_config['use_gpu'] = args.use_gpu
            if hasattr(args, 'disable_onnx_gpu') and args.disable_onnx_gpu is not None:
                cli_config['disable_onnx_gpu'] = args.disable_onnx_gpu
            config_dict['cli'] = cli_config
            
            success_count, skipped_count, failed_count = await translate_with_subprocess(
                all_files=all_files,
                output_dir=output_dir,
                config_dict=config_dict,
                verbose=verbose,
                overwrite=overwrite,
                memory_limit_mb=getattr(args, 'memory_limit', DEFAULT_MEMORY_THRESHOLD_MB),
                memory_limit_percent=getattr(args, 'memory_percent', 80),
                batch_per_restart=getattr(args, 'batch_per_restart', DEFAULT_BATCH_SIZE_PER_RESTART)
            )
            
            print(f"\n{'='*60}")
            print(f"✅ Succeeded: {success_count}")
            print(f"⏭️  Skipped: {skipped_count}")
            print(f"❌ Failed: {failed_count}")
            print(f"📊 Total: {len(all_files)}")
            print(f"💾 Output folder: {output_dir}")
            print(f"{'='*60}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled by the user")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        # Direct mode
        try:
            await translate_files(
                args.input,
                args.output if hasattr(args, 'output') else None,
                config_service,
                verbose=verbose,
                overwrite=overwrite,
                args=args
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled by the user")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def main():
    """Entry point when run directly."""
    # Needed on Windows for child processes
    multiprocessing.freeze_support()
    
    args = parse_args()
    asyncio.run(run_local_mode(args))


if __name__ == '__main__':
    main()
