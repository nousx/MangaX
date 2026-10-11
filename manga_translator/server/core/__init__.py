"""
核心模块

提供配置管理、身份验证、日志管理、任务管理、响应工具、数据模型和持久化功能。
"""

# Data models
# Services
from manga_translator.server.core.account_service import AccountService
from manga_translator.server.core.audit_service import AuditService

# Configuration management
from manga_translator.server.core.config_manager import (
    ADMIN_CONFIG_PATH,
    AVAILABLE_WORKFLOWS,
    DEFAULT_ADMIN_SETTINGS,
    SERVER_CONFIG_PATH,
    get_available_workflows,
    init_server_config_file,
    load_admin_settings,
    load_default_config,
    load_default_config_dict,
    parse_config,
    save_admin_settings,
    temp_env_vars,
)
from manga_translator.server.core.env_service import EnvService

# Log management
from manga_translator.server.core.logging_manager import (
    WebLogHandler,
    add_log,
    export_logs,
    generate_task_id,
    get_logs,
    get_task_id,
    global_log_queue,
    set_task_id,
    setup_log_handler,
    task_logs,
)

# Authentication and authorisation middleware (new version)
from manga_translator.server.core.middleware import (
    check_concurrent_limit,
    check_daily_quota,
    check_parameter_permission,
    check_translator_permission,
    create_error_response,
    decrement_task_count,
    get_services,
    increment_daily_usage,
    increment_task_count,
    init_middleware_services,
    require_admin,
    require_auth,
)
from manga_translator.server.core.models import (
    AuditEvent,
    Session,
    UserAccount,
    UserPermissions,
)
from manga_translator.server.core.permission_service import PermissionService

# Persistence tools
from manga_translator.server.core.persistence import (
    atomic_write_json,
    cleanup_old_backups,
    create_backup,
    ensure_directory,
    load_json,
)

# Response tools
from manga_translator.server.core.response_utils import (
    apply_user_env_vars,
    transform_to_bytes,
    transform_to_image,
    transform_to_json,
)
from manga_translator.server.core.session_service import SessionService

# System initialisation
from manga_translator.server.core.system_init import (
    SystemInitializer,
    get_system_initializer,
    init_system,
)

# Task management
from manga_translator.server.core.task_manager import (
    active_tasks,
    cancel_task,
    cleanup_after_request,
    cleanup_context,
    get_active_tasks,
    get_semaphore,
    get_server_config,
    init_semaphore,
    is_task_cancelled,
    register_active_task,
    server_config,
    translation_semaphore,
    unregister_active_task,
    update_server_config,
    update_task_status,
)

__all__ = [
    # Data models
    'UserPermissions',
    'UserAccount',
    'Session',
    'AuditEvent',
    # Persistence tools
    'atomic_write_json',
    'load_json',
    'create_backup',
    'cleanup_old_backups',
    'ensure_directory',
    # Services
    'AccountService',
    'SessionService',
    'PermissionService',
    'AuditService',
    'EnvService',
    # System initialisation
    'SystemInitializer',
    'init_system',
    'get_system_initializer',
    # Configuration management
    'ADMIN_CONFIG_PATH',
    'SERVER_CONFIG_PATH',
    'DEFAULT_ADMIN_SETTINGS',
    'AVAILABLE_WORKFLOWS',
    'load_admin_settings',
    'save_admin_settings',
    'load_default_config_dict',
    'load_default_config',
    'parse_config',
    'get_available_workflows',
    'temp_env_vars',
    'init_server_config_file',
    # Authentication and authorisation middleware (new version)
    'init_middleware_services',
    'get_services',
    'create_error_response',
    'require_auth',
    'require_admin',
    'check_translator_permission',
    'check_parameter_permission',
    'check_concurrent_limit',
    'check_daily_quota',
    'increment_task_count',
    'decrement_task_count',
    'increment_daily_usage',
    # Log management
    'task_logs',
    'global_log_queue',
    'generate_task_id',
    'set_task_id',
    'get_task_id',
    'add_log',
    'get_logs',
    'export_logs',
    'WebLogHandler',
    'setup_log_handler',
    # Task management
    'translation_semaphore',
    'server_config',
    'active_tasks',
    'init_semaphore',
    'get_semaphore',
    'register_active_task',
    'unregister_active_task',
    'update_task_status',
    'get_active_tasks',
    'is_task_cancelled',
    'cancel_task',
    'update_server_config',
    'get_server_config',
    'cleanup_after_request',
    'cleanup_context',
    # Response tools
    'transform_to_image',
    'transform_to_json',
    'transform_to_bytes',
    'apply_user_env_vars',
]
