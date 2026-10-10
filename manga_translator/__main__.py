#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manga Translator - command-line entry point
Supports several run modes: web, local, ws, shared
"""
import asyncio
import logging
import os
import sys
import warnings

# Hide third-party warnings; set before importing torch so the pynvml notice from torch.cuda is caught.
warnings.filterwarnings('ignore', message='.*Triton.*')
warnings.filterwarnings('ignore', message='.*triton.*')
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*pynvml package is deprecated.*', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='ctranslate2')

# Tune GPU memory before PyTorch starts so shared memory can be used
# expandable_segments reduces GPU memory fragmentation and out-of-memory errors
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# Load PyTorch before PyQt6, whose Qt DLL path otherwise breaks loading c10.dll
# The rendering module (text_render.py) needs PyQt6 and would trigger the DLL conflict
# See: https://github.com/pytorch/pytorch/issues/166628
try:
    import torch  # noqa: F401
except ImportError:
    pass

def main():
    """Entry point."""
    from manga_translator.args import parse_args
    
    # Parse arguments
    args = parse_args()

    # Export the ONNX GPU switch to the environment so every run mode honours it
    if getattr(args, 'disable_onnx_gpu', False):
        os.environ['MT_DISABLE_ONNX_GPU'] = '1'
    
    # Import logging helpers late to avoid loading large libraries early
    from manga_translator.utils.log import get_logger, init_logging, set_log_level
    
    # Set up logging
    init_logging()
    set_log_level(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = get_logger(args.mode)

    # Before dispatching, every CLI mode writes out the bundled settings tables and AI prompt tables.
    from manga_translator.runtime_files import ensure_runtime_files
    ensure_runtime_files(logger)
    
    # Dispatch by mode
    if args.mode == 'web':
        # Web server mode (API + web interface)
        logger.info('[web] Starting Web server')
        from manga_translator.server import run_server
        run_server(args)
    
    elif args.mode == 'local':
        # Local mode (command-line translation)
        logger.info('Running in local mode')
        from manga_translator.mode.local import run_local_mode
        asyncio.run(run_local_mode(args))
    
    elif args.mode == 'ws':
        # WebSocket mode
        logger.info('Running in WebSocket mode')
        from manga_translator.mode.ws import MangaTranslatorWS
        translator = MangaTranslatorWS(vars(args))
        asyncio.run(translator.listen(vars(args)))
    
    elif args.mode == 'shared':
        # Shared/API mode
        logger.info('Running in shared/API mode')
        from manga_translator.mode.share import MangaShare
        translator = MangaShare(vars(args))
        asyncio.run(translator.listen(vars(args)))
    
    else:
        logger.error(f'Unknown mode: {args.mode}')
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nTranslation cancelled by user.')
        sys.exit(0)
    except asyncio.CancelledError:
        print('\nTranslation cancelled by user.')
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f'\n{e.__class__.__name__}: {e}')
        traceback.print_exc()
        sys.exit(1)
