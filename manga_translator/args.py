import argparse
import os
import sys

from .image_formats import OUTPUT_IMAGE_FORMATS


def _env_true(name: str) -> bool:
    return os.getenv(name, '').strip().lower() in ('true', '1', 'yes', 'on')


def create_parser():
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Manga Translator - translate manga pages',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Sub-commands
    subparsers = parser.add_subparsers(dest='mode', help='Run mode')
    
    # ===== Web mode (web server: API + web interface) =====
    web_parser = subparsers.add_parser('web', help='Web server mode (API + web interface)')
    web_parser.add_argument('--host', 
                           default=os.getenv('MT_WEB_HOST', '127.0.0.1'),
                           help='Server host (default: 127.0.0.1, reachable from this machine only; '
                                'pass --host 0.0.0.0 explicitly for LAN or container access; environment variable: MT_WEB_HOST)')
    web_parser.add_argument('--cors-origins',
                           default=os.getenv('MT_WEB_CORS_ORIGINS'),
                           help='Comma-separated origins allowed to call the API across origins (default: local origins only; '
                                '"*" allows every origin; environment variable: MT_WEB_CORS_ORIGINS)')
    web_parser.add_argument('--port', 
                           default=int(os.getenv('MT_WEB_PORT', '8000')), 
                           type=int,
                           help='Server port (default: 8000; environment variable: MT_WEB_PORT)')
    web_parser.add_argument('--use-gpu', 
                           action='store_true',
                           default=_env_true('MT_USE_GPU'),
                           help='Use the GPU (environment variable: MT_USE_GPU=true)')
    web_parser.add_argument('--disable-onnx-gpu',
                           action='store_true',
                           default=_env_true('MT_DISABLE_ONNX_GPU'),
                           help='Disable ONNX Runtime GPU acceleration (environment variable: MT_DISABLE_ONNX_GPU=true)')
    web_parser.add_argument('--models-ttl', 
                           default=int(os.getenv('MT_MODELS_TTL', '0')), 
                           type=int,
                           help='Seconds to keep models in memory after their last use (0 means forever; environment variable: MT_MODELS_TTL)')
    web_parser.add_argument('--retry-attempts', 
                           default=int(os.getenv('MT_RETRY_ATTEMPTS', '-1')) if os.getenv('MT_RETRY_ATTEMPTS') else None, 
                           type=int,
                           help='Retries after a failed translation (-1 retries forever, None uses the value sent through the API; environment variable: MT_RETRY_ATTEMPTS)')
    web_parser.add_argument('-v', '--verbose', 
                           action='store_true',
                           default=os.getenv('MT_VERBOSE', '').lower() in ('true', '1', 'yes'),
                           help='Show detailed logs (environment variable: MT_VERBOSE=true)')
    
    # ===== Local mode (default) =====
    local_parser = subparsers.add_parser('local', help='Command-line translation mode')
    local_parser.add_argument('-i', '--input', required=True, nargs='+',
                             help='Input image or folder paths')
    local_parser.add_argument('-o', '--output', default=None,
                             help='Output folder (default: the input folder name with a -translated suffix)')
    local_parser.add_argument('--config', default=None,
                             help='Settings file path (default: config/config.json)')
    local_parser.add_argument('-v', '--verbose', action='store_true',
                             help='Show detailed logs')
    local_parser.add_argument('--skip-existing', action='store_true',
                              help='Skip pages whose output already exists (continue an unfinished run)')
    local_parser.add_argument('--overwrite', action='store_true',
                             help='Overwrite existing files')
    local_parser.add_argument('--use-gpu', action='store_true', default=None,
                             help='Use GPU acceleration (overrides the settings file)')
    local_parser.add_argument('--disable-onnx-gpu', action='store_true', default=None,
                             help='Disable ONNX Runtime GPU acceleration (overrides the settings file)')
    local_parser.add_argument('--format', default=None,
                             help=f"Output format: {'/'.join(OUTPUT_IMAGE_FORMATS)} (overrides the settings file)")
    local_parser.add_argument('--batch-size', type=int, default=None,
                             help='Batch size (overrides the settings file)')
    local_parser.add_argument('--attempts', type=int, default=None,
                             help='Retries after a failed translation, -1 retries forever (overrides the settings file)')
    # Memory management (subprocess mode)
    local_parser.add_argument('--subprocess', action='store_true',
                             help='Run in subprocess mode (adds memory management and resuming)')
    local_parser.add_argument('--memory-limit', type=int, default=0,
                             help='Memory limit in MB; the worker restarts when it is exceeded, 0 means no limit (default: 0)')
    local_parser.add_argument('--memory-percent', type=int, default=0,
                             help='Memory limit as a percentage of system memory; the worker restarts when it is exceeded, 0 means no limit (default: 0)')
    local_parser.add_argument('--batch-per-restart', type=int, default=0,
                             help='Restart the worker after this many images to release memory, 0 means never (default: 0)')
    
    # ===== WebSocket mode =====
    ws_parser = subparsers.add_parser('ws', help='WebSocket mode')
    ws_parser.add_argument('--host', default='127.0.0.1',
                          help='WebSocket service host (default: 127.0.0.1)')
    ws_parser.add_argument('--port', default=5003, type=int,
                          help='WebSocket service port (default: 5003)')
    ws_parser.add_argument('--nonce', default=None,
                          help='Nonce that protects internal WebSocket traffic')
    ws_parser.add_argument('--ws-url', default='ws://localhost:5000',
                          help='Server URL for WebSocket mode (default: ws://localhost:5000)')
    ws_parser.add_argument('--models-ttl', default=0, type=int,
                          help='Seconds to keep models in memory after their last use (0 means forever)')
    ws_parser.add_argument('--retry-attempts', default=None, type=int,
                          help='Retries after a failed translation (-1 retries forever, None uses the value sent through the API)')
    ws_parser.add_argument('-v', '--verbose', action='store_true',
                          help='Show detailed logs')
    ws_parser.add_argument('--use-gpu', action='store_true',
                          help='Use the GPU')
    ws_parser.add_argument('--disable-onnx-gpu', action='store_true',
                          default=_env_true('MT_DISABLE_ONNX_GPU'),
                          help='Disable ONNX Runtime GPU acceleration')
    
    # ===== Shared mode (API instance) =====
    shared_parser = subparsers.add_parser('shared', help='API mode')
    shared_parser.add_argument('--host', default='127.0.0.1',
                              help='API service host (default: 127.0.0.1)')
    shared_parser.add_argument('--port', default=5003, type=int,
                              help='API service port (default: 5003)')
    shared_parser.add_argument('--nonce', default=None,
                              help='Nonce that protects internal API server traffic')
    shared_parser.add_argument('--models-ttl', default=0, type=int,
                              help='Seconds models stay in memory (0 means forever)')
    shared_parser.add_argument('--retry-attempts', default=None, type=int,
                              help='Retries after a failed translation (-1 retries forever, None uses the value sent through the API)')
    shared_parser.add_argument('-v', '--verbose', action='store_true',
                              help='Show detailed logs')
    shared_parser.add_argument('--use-gpu', action='store_true',
                              help='Use the GPU')
    shared_parser.add_argument('--disable-onnx-gpu', action='store_true',
                              default=_env_true('MT_DISABLE_ONNX_GPU'),
                              help='Disable ONNX Runtime GPU acceleration')
    
    return parser


def parse_args():
    """Parse the command-line arguments."""
    parser = create_parser()
    
    # Default to local mode when the first argument is not a mode
    if len(sys.argv) > 1 and sys.argv[1] not in ['web', 'local', 'ws', 'shared']:
        # Look for -i, which local mode requires
        if '-i' in sys.argv or '--input' in sys.argv:
            # Insert 'local' before the first argument
            sys.argv.insert(1, 'local')
    
    args = parser.parse_args()
    
    # Still no mode: show the help text
    if args.mode is None:
        parser.print_help()
        sys.exit(1)
    
    return args
