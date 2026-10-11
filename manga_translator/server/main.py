import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import warnings
from argparse import Namespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Hide warnings of third-party libraries; it has to be set before torch is imported, to catch the pynvml notice of torch.cuda.
warnings.filterwarnings('ignore', message='.*Triton.*')
warnings.filterwarnings('ignore', message='.*triton.*')
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*pynvml package is deprecated.*', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='ctranslate2')

# Load PyTorch before PyQt6, so the Qt DLL path of PyQt6 does not interfere with loading c10.dll
# The rendering module (text_render.py) depends on PyQt6, which triggers the DLL conflict
# See: https://github.com/pytorch/pytorch/issues/166628
try:
    import torch  # noqa: F401
except ImportError:
    pass

import logging

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from manga_translator.server_paths import (
    SERVER_DATA_RELATIVE_DIR,
    USER_RESOURCES_RELATIVE_DIR,
    ensure_server_data_layout,
)

# Import core modules
from manga_translator.server.core import config_manager, logging_manager, task_manager
from manga_translator.server.core.setup_guard import (
    CORS_ORIGINS_ENV,
    build_cors_options,
    initial_setup_hint,
    parse_cors_origins,
)
from manga_translator.server.instance import ExecutorInstance, executor_instances

# Initialise the server configuration file (copied from the template when it does not exist)
config_manager.init_server_config_file()
ensure_server_data_layout()


def _ensure_web_startup_files() -> None:
    """Create the same runtime tables used by CLI and desktop startup."""
    from manga_translator.runtime_files import ensure_runtime_files

    ensure_runtime_files(logger)

# Import route modules
# Import sessions_router
from manga_translator.server.routes import (
    admin_router,
    audit_router,
    auth_router,
    config_management_router,
    config_router,
    files_router,
    groups_router,
    history_router,
    init_auth_services,
    init_history_routes,
    init_quota_routes,
    init_resource_routes,
    logs_router,
    quota_router,
    resources_router,
    sessions_router,
    translation_router,
    users_router,
    web_router,
)

logger = logging.getLogger('manga_translator.server')

# Set the web server flag, so the translator does not reload .env and overwrite the user's environment variables
os.environ['MANGA_TRANSLATOR_WEB_SERVER'] = 'true'

# Load the .env file at start-up
from manga_translator.utils.dotenv_utils import APP_DOTENV_PATH_ENV, load_app_dotenv
from manga_translator.runtime_paths import get_application_dir

env_path = os.path.join(get_application_dir(), '.env')
os.environ[APP_DOTENV_PATH_ENV] = env_path
if os.path.exists(env_path):
    load_app_dotenv(env_path, override=False)
    print(f"[INFO] Loaded environment variables from: {env_path}")
    # Print the API keys that were loaded (without their values)
    loaded_keys = [k for k in os.environ.keys() if 'API' in k or 'KEY' in k or 'TOKEN' in k]
    if loaded_keys:
        print(f"[INFO] Loaded API keys: {', '.join(loaded_keys)}")
else:
    print(f"[WARNING] .env file not found at: {env_path}")

app = FastAPI()
nonce = None

# Initialize logging manager
logging_manager.setup_log_handler()

# Note: config_manager and task_manager are initialized when imported
# task_manager.init_semaphore() will be called in run_server()

# Global service instances (will be initialized on startup)
_account_service = None
_session_service = None
_permission_service = None
_audit_service = None
_system_initializer = None
# Bind address of the running server (set by run_server); used for the
# first-run setup hint.
_bind_address = None


def _log_initial_setup_hint() -> None:
    """Explain how to create the first admin when no accounts exist yet."""
    if _account_service is None or _bind_address is None:
        return
    if _account_service.list_users():
        return
    from manga_translator.server.core.logging_manager import add_log

    host, port = _bind_address
    for line in initial_setup_hint(host, port):
        logger.warning(line)
        print(f"[SETUP] {line}")
        add_log(line, "WARNING")


@app.on_event("startup")
async def startup_event():
    """Initialize services on server startup"""
    global _account_service, _session_service, _permission_service, _audit_service, _system_initializer
    
    from manga_translator.server.core import (
        AccountService,
        AuditService,
        PermissionService,
        SessionService,
        init_middleware_services,
        init_system,
    )
    from manga_translator.server.core.logging_manager import add_log
    from manga_translator.server.routes.translation_auth import init_translation_auth
    
    # Add the start-up log
    add_log("Server is starting...", "INFO")
    logger.info("Server starting up...")
    _ensure_web_startup_files()
    from manga_translator.server.core.permission_integration import (
        IntegratedPermissionService,
    )
    from manga_translator.server.core.permission_service_v2 import (
        EnhancedPermissionService,
    )
    from manga_translator.server.core.resource_service import ResourceManagementService
    from manga_translator.server.repositories.permission_repository import (
        PermissionRepository,
    )
    from manga_translator.server.repositories.resource_repository import (
        ResourceRepository,
    )
    
    # Initialize services - all data files live in the manga_translator/server/data folder
    DATA_DIR = SERVER_DATA_RELATIVE_DIR
    _account_service = AccountService(accounts_file=f"{DATA_DIR}/accounts.json")
    _session_service = SessionService(
        sessions_file=f"{DATA_DIR}/sessions.json",
        session_timeout_minutes=60,
        enable_persistence=True
    )
    _permission_service = PermissionService(_account_service)
    _audit_service = AuditService(audit_log_file=f"{DATA_DIR}/audit.log")
    
    # Initialize middleware services
    init_middleware_services(_account_service, _session_service, _permission_service)
    
    # Initialize auth services
    init_auth_services(_account_service, _session_service, _audit_service)
    
    # Initialize translation authentication
    init_translation_auth(_audit_service)
    
    # Initialize resource management services
    prompts_repo = ResourceRepository(f"{USER_RESOURCES_RELATIVE_DIR}/prompts/index.json")
    fonts_repo = ResourceRepository(f"{USER_RESOURCES_RELATIVE_DIR}/fonts/index.json")
    resource_service = ResourceManagementService(prompts_repo, fonts_repo)
    
    # Initialize enhanced permission service for resource routes
    permission_repo = PermissionRepository("manga_translator/server/data/permissions.json")
    enhanced_permission_service = EnhancedPermissionService(permission_repo)
    
    # Initialize integrated permission service (with user group support)
    integrated_permission_service = IntegratedPermissionService(_account_service, enhanced_permission_service)
    
    # Initialize resource routes
    init_resource_routes(resource_service, integrated_permission_service)
    
    # Initialize history management services
    from manga_translator.server.core.history_service import HistoryManagementService
    from manga_translator.server.core.search_service import SearchService
    from manga_translator.server.repositories.translation_repository import (
        TranslationRepository,
    )
    
    translation_repo = TranslationRepository("manga_translator/server/data/translation_history.json")
    history_service = HistoryManagementService(
        result_directory="manga_translator/server/data/results",
        translation_repo=translation_repo
    )
    search_service = SearchService()
    
    # Initialize history routes
    init_history_routes(history_service, integrated_permission_service, search_service)
    
    # Initialize quota management services
    from manga_translator.server.core.group_service import GroupService
    from manga_translator.server.core.quota_service import QuotaManagementService
    from manga_translator.server.repositories.quota_repository import QuotaRepository
    
    quota_repo = QuotaRepository("manga_translator/server/data/quotas.json")
    group_service = GroupService()
    quota_service = QuotaManagementService(quota_repo, permission_repo, group_service)
    
    # Initialize quota routes
    init_quota_routes(quota_service)
    
    # Initialize system (includes creating default admin, starting background tasks)
    _system_initializer = init_system(_account_service, _session_service, _audit_service)
    await _system_initializer.initialize()
    
    # Start cleanup service
    from manga_translator.server.core.cleanup_service import get_cleanup_service
    cleanup_service = get_cleanup_service()
    cleanup_service.start()
    
    logger.info("Services initialized successfully")
    add_log("Server startup complete; all services initialized", "INFO")
    _log_initial_setup_hint()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    global _system_initializer
    
    # Stop cleanup service
    from manga_translator.server.core.cleanup_service import get_cleanup_service
    cleanup_service = get_cleanup_service()
    cleanup_service.stop()
    
    if _system_initializer:
        await _system_initializer.shutdown()
    
    logger.info("Server shutdown completed")

# Configure middleware
# CORS: by default only loopback origins may call the API cross-origin (the
# bundled web UI is same-origin and unaffected).  Override with the
# MT_WEB_CORS_ORIGINS environment variable or `web --cors-origins`.
def configure_cors(origins=None) -> dict:
    """(Re)install the CORS middleware. Must be called before the app starts serving."""
    options = build_cors_options(origins)
    app.user_middleware = [m for m in app.user_middleware if m.cls is not CORSMiddleware]
    app.add_middleware(CORSMiddleware, **options)
    return options


configure_cors(parse_cors_origins(os.environ.get(CORS_ORIGINS_ENV)))

# Add validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误，返回详细的错误信息"""
    error_details = []
    for error in exc.errors():
        error_details.append({
            'loc': error['loc'],
            'msg': error['msg'],
            'type': error['type']
        })
    
    print(f"[ERROR] Request validation failed for {request.url.path}")
    print(f"[ERROR] Validation errors: {json.dumps(error_details, indent=2)}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": error_details,
            "body": str(exc.body) if hasattr(exc, 'body') else None
        }
    )

# Mount static files
static_dir = os.path.join(get_application_dir(), "manga_translator", "server", "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Favicon route
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(static_dir, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404)


# Mount Qt UI locales for i18n (shared translation files)
locales_dir = os.path.join(get_application_dir(), "desktop_qt_ui", "locales")
if os.path.exists(locales_dir):
    app.mount("/locales", StaticFiles(directory=locales_dir), name="locales")

# Register route modules
app.include_router(translation_router)
app.include_router(admin_router)
app.include_router(config_router)
app.include_router(files_router)
app.include_router(web_router)
app.include_router(users_router)
app.include_router(sessions_router)
app.include_router(audit_router)
app.include_router(auth_router)
app.include_router(groups_router)
app.include_router(resources_router)
app.include_router(history_router)
app.include_router(quota_router)
app.include_router(config_management_router)
app.include_router(logs_router)

# Internal API endpoint for instance registration
@app.post("/register", response_description="no response", tags=["internal-api"])
async def register_instance(instance: ExecutorInstance, req: Request, req_nonce: str = Header(alias="X-Nonce")):
    # Fail closed when no nonce is configured and compare in constant time:
    # a registered executor is trusted to return pickled results.
    if not nonce or not secrets.compare_digest(req_nonce.encode('utf-8'), nonce.encode('utf-8')):
        raise HTTPException(401, detail="Invalid nonce")
    instance.ip = req.client.host
    executor_instances.register(instance)

def generate_nonce():
    return secrets.token_hex(16)

def start_translator_client_proc(host: str, port: int, nonce: str, params: Namespace):
    cmds = [
        sys.executable,
        '-m', 'manga_translator',
        'shared',
        '--host', host,
        '--port', str(port),
        '--nonce', nonce,
    ]
    if params.use_gpu:
        cmds.append('--use-gpu')
    if getattr(params, 'disable_onnx_gpu', False):
        cmds.append('--disable-onnx-gpu')
    if params.ignore_errors:
        cmds.append('--ignore-errors')
    if params.verbose:
        cmds.append('--verbose')
    if params.models_ttl:
        cmds.append('--models-ttl=%s' % params.models_ttl)
    if getattr(params, 'pre_dict', None):
        cmds.extend(['--pre-dict', params.pre_dict])
    if getattr(params, 'post_dict', None):
        cmds.extend(['--post-dict', params.post_dict])       
    proc = subprocess.Popen(cmds, cwd=get_application_dir())
    executor_instances.register(ExecutorInstance(ip=host, port=port))

    def handle_exit_signals(signal, frame):
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit_signals)
    signal.signal(signal.SIGTERM, handle_exit_signals)

    return proc

def prepare(args):
    global nonce
    
    # web mode has no nonce argument; getattr avoids an AttributeError
    args_nonce = getattr(args, 'nonce', None)
    if args_nonce is None:
        nonce = os.getenv('MT_WEB_NONCE', generate_nonce())
    else:
        nonce = args_nonce
    
    # start_instance may not exist in some modes either
    if getattr(args, 'start_instance', False):
        return start_translator_client_proc(args.host, args.port + 1, nonce, args)
    
    folder_name= "upload-cache"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

def init_translator(use_gpu=False, verbose=False):
    """初始化翻译器（预留函数）"""
    # This function is for initialisation such as preloading models
    # For now the translator is only initialised on the first request
    pass

def run_server(args):
    """启动 Web API 服务器（纯API模式，不带界面）"""
    import uvicorn

    global _bind_address
    _bind_address = (args.host, args.port)

    cors_origins = parse_cors_origins(getattr(args, 'cors_origins', None))
    if cors_origins:
        cors_options = configure_cors(cors_origins)
        if cors_options['allow_origins'] == ['*']:
            logger.warning("CORS: every origin is allowed to call this API (--cors-origins '*').")

    if getattr(args, 'disable_onnx_gpu', False):
        os.environ['MT_DISABLE_ONNX_GPU'] = '1'
    
    # Set the server configuration (before prepare)
    task_manager.server_config['use_gpu'] = getattr(args, 'use_gpu', False)
    task_manager.server_config['verbose'] = getattr(args, 'verbose', False)
    task_manager.server_config['models_ttl'] = getattr(args, 'models_ttl', 0)
    task_manager.server_config['retry_attempts'] = getattr(args, 'retry_attempts', None)
    
    # Load the administrator password and the concurrency setting from admin_settings
    task_manager.server_config['admin_password'] = config_manager.admin_settings.get('admin_password')
    if config_manager.admin_settings.get('max_concurrent_tasks'):
        task_manager.server_config['max_concurrent_tasks'] = config_manager.admin_settings['max_concurrent_tasks']
    
    print(f"[SERVER CONFIG] use_gpu={task_manager.server_config['use_gpu']}, verbose={task_manager.server_config['verbose']}, models_ttl={task_manager.server_config['models_ttl']}, retry_attempts={task_manager.server_config['retry_attempts']}, max_concurrent_tasks={task_manager.server_config['max_concurrent_tasks']}")
    
    # Initialise concurrency control
    task_manager.init_semaphore()
    
    # web mode does not start a separate translation instance (the same as old versions)
    args.start_instance = False
    proc = prepare(args)
    print("Nonce: "+nonce)
    try:
        # A longer timeout, to support batch translation (30 minutes)
        uvicorn.run(
            app, 
            host=args.host, 
            port=args.port,
            timeout_keep_alive=1800,  # Keep the connection for 30 minutes
            timeout_graceful_shutdown=30  # Graceful shutdown timeout: 30 seconds
        )
    except Exception:
        if proc:
            proc.terminate()

def main(args):
    """启动 Web UI 服务器（带界面模式）"""
    # ui mode and web mode use the same implementation
    run_server(args)

if __name__ == '__main__':
    from manga_translator.args import parse_arguments
    args = parse_arguments()
    main(args)
