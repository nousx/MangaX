"""
Translation endpoint authentication and authorization helpers.

This module provides helper functions for integrating authentication and
permission checks into translation endpoints.
"""

import logging
from enum import Enum
from typing import Optional

from fastapi import HTTPException, Request

from manga_translator import Config
from manga_translator.server.core.audit_service import AuditService
from manga_translator.server.core.group_management_service import (
    get_group_management_service,
)
from manga_translator.server.core.middleware import (
    check_concurrent_limit,
    check_daily_quota,
    decrement_task_count,
    get_services,
    increment_daily_usage,
    increment_task_count,
)

logger = logging.getLogger(__name__)


def filter_disabled_parameters(config: Config, username: str, permission_service) -> None:
    """
    Filter out the parameters the user may not change, using the defaults set by the administrator

    The configurations of the user group and of the user are merged at execution time:
    - the allow/deny lists of the user group
    - the allow/deny lists of the user

    Priority (high to low):
    1. the user's deny list (highest priority; it disables even what the allow list of the user group permits)
    2. the user's allow list (it can unlock the deny list of the user group)
    3. the deny list of the user group
    4. the allow list of the user group

    Finally disabled = the user's deny list + (the deny list of the user group - the user's allow list)

    Args:
        config: the translation configuration object
        username: the user name
        permission_service: the permission service
    """
    try:
        # Get the services
        account_service, _, _ = get_services()
        group_service = get_group_management_service()
        
        # Get the user information
        user_account = account_service.get_user(username)
        if not user_account:
            return
        
        group_id = user_account.group if hasattr(user_account, 'group') else 'default'
        
        # Get the parameter configuration of the user group
        group = group_service.get_group(group_id)
        group_param_config = group.get('parameter_config', {}) if group else {}
        
        # Get the permission configuration of the user
        user_permissions = user_account.permissions if hasattr(user_account, 'permissions') else None
        
        # The user's whitelist and blacklist
        user_allowed_params = set()  # User whitelist
        user_denied_params = set()   # User blacklist
        if user_permissions:
            allowed = getattr(user_permissions, 'allowed_parameters', ['*'])
            denied = getattr(user_permissions, 'denied_parameters', [])
            # Only a non-wildcard entry is a real whitelist
            if '*' not in allowed:
                user_allowed_params = set(allowed)
            user_denied_params = set(denied)
        
        # Group blacklist (parameters with disabled=True)
        # Note: the disable configuration may be nested in parameter_config.parameter_config
        group_disabled = {}
        
        # Check for a nested parameter_config (new format)
        nested_param_config = group_param_config.get('parameter_config', {})
        if nested_param_config:
            logger.debug(f"Found nested parameter_config for user {username}: {list(nested_param_config.keys())}")
            for full_key, settings in nested_param_config.items():
                if isinstance(settings, dict) and settings.get('disabled', False):
                    group_disabled[full_key] = settings
                    logger.debug(f"Parameter {full_key} is disabled for group {group_id}, default: {settings.get('default_value')}")
        
        # Check the old format too (disable configuration directly in parameter_config)
        for full_key, settings in group_param_config.items():
            if full_key == 'parameter_config':
                continue  # Skip the nested configuration
            if isinstance(settings, dict) and settings.get('disabled', False):
                group_disabled[full_key] = settings
        
        logger.debug(f"Total disabled parameters for user {username}: {list(group_disabled.keys())}")
        
        # Get the user-level parameter configuration (for default values)
        user_param_config = {}
        user_disabled = {}  # The user's disable configuration
        if hasattr(user_account, 'parameter_config') and user_account.parameter_config:
            user_param_config = user_account.parameter_config
            # Check whether the user configuration has a nested parameter_config (disable configuration)
            nested_user_param = user_param_config.get('parameter_config', {})
            if nested_user_param:
                for full_key, settings in nested_user_param.items():
                    if isinstance(settings, dict) and settings.get('disabled', False):
                        user_disabled[full_key] = settings
        
        # Work out the parameters that end up disabled
        # Finally disabled = user blacklist + (group blacklist - user whitelist)
        final_disabled = {}
        
        # 1. User blacklist (highest priority)
        for param in user_denied_params:
            final_disabled[param] = {'disabled': True, 'source': 'user'}
        
        # 2. Group blacklist, which the user whitelist can unlock
        for full_key, settings in group_disabled.items():
            # When the user whitelist contains this parameter, it is unlocked (not disabled)
            if full_key in user_allowed_params:
                continue
            # When it is already in the user blacklist, the user blacklist setting stays
            if full_key not in final_disabled:
                final_disabled[full_key] = {**settings, 'source': 'group'}
        
        if not final_disabled:
            return
        
        # Go through the disabled parameters and overwrite the values the user submitted with the defaults
        # Priority of default values: user configuration > group configuration > server default
        for full_key, settings in final_disabled.items():
            # Parse the parameter path, such as "translator.translator" -> section="translator", key="translator"
            parts = full_key.split('.')
            if len(parts) != 2:
                continue
            
            section, key = parts
            
            # Get the default value (by priority)
            default_value = None
            
            # 1. Prefer the default in the user's disable configuration
            if full_key in user_disabled:
                user_setting = user_disabled[full_key]
                if isinstance(user_setting, dict) and 'default_value' in user_setting:
                    default_value = user_setting['default_value']
            
            # 2. Then the value from the user configuration (format: {"section": {"key": value}})
            if default_value is None and section in user_param_config:
                user_section = user_param_config[section]
                if isinstance(user_section, dict) and key in user_section:
                    default_value = user_section[key]
            
            # 3. Then the default in the group's disable configuration
            if default_value is None and isinstance(settings, dict):
                default_value = settings.get('default_value')
            
            # 4. Finally try the default from the group's parameter configuration (the part that is not disable configuration)
            if default_value is None:
                # Get the value of section.key from group_param_config
                section_config = group_param_config.get(section, {})
                if isinstance(section_config, dict) and key in section_config:
                    section_value = section_config[key]
                    # A simple value (not a disable configuration object) is used directly
                    if not isinstance(section_value, dict) or 'disabled' not in section_value:
                        default_value = section_value
            
            # AppSettings has a cli attribute, so cli.attempts can be set directly
            # Find the sub-object of config for the section
            if hasattr(config, section):
                section_obj = getattr(config, section)
                if hasattr(section_obj, key) and default_value is not None:
                    # Get the type annotation of the target attribute
                    field_type = None
                    if hasattr(section_obj.__class__, '__annotations__'):
                        field_type = section_obj.__class__.__annotations__.get(key)
                    
                    # When the target type is an enum and the current value is a string, convert it to the enum
                    if field_type and isinstance(field_type, type) and issubclass(field_type, Enum):
                        if isinstance(default_value, str):
                            # Try to find the enum member by its string value
                            try:
                                default_value = field_type(default_value)
                            except (ValueError, KeyError):
                                logger.warning(f"Failed to convert '{default_value}' to {field_type.__name__}, using as-is")
                    
                    setattr(section_obj, key, default_value)
    
    except Exception as e:
        logger.warning(f"Failed to filter disabled parameters for user {username}: {e}")

# Global audit service instance (initialized on server startup)
_audit_service: Optional[AuditService] = None


def init_translation_auth(audit_service: AuditService) -> None:
    """
    Initialize translation authentication module
    
    Args:
        audit_service: Audit service instance
    """
    global _audit_service
    _audit_service = audit_service
    logger.info("Translation authentication module initialized")


def get_audit_service() -> AuditService:
    """Get audit service instance"""
    if not _audit_service:
        raise RuntimeError("Translation authentication module not initialized")
    return _audit_service


async def verify_translation_auth(
    request: Request,
    config: Config,
    translator: Optional[str] = None
) -> tuple[str, str]:
    """
    Verify authentication and permissions for translation request
    
    This function:
    1. Extracts and validates session token from request headers
    2. Checks translator permission
    3. Filters config parameters based on user permissions
    4. Checks concurrent task limit
    5. Checks daily quota
    
    Args:
        request: FastAPI request object
        config: Translation configuration
        translator: Translator name (if None, extracted from config)
    
    Returns:
        tuple[str, str]: (username, ip_address)
    
    Raises:
        HTTPException: If authentication or authorization fails
    """
    # Extract session token from headers
    session_token = request.headers.get("X-Session-Token")
    
    if not session_token:
        logger.warning("Translation request without session token")
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "NO_TOKEN",
                    "message": "未提供会话令牌，请先登录"
                }
            }
        )
    
    # Verify session token
    _, session_service, permission_service = get_services()
    session = session_service.verify_token(session_token)
    
    if not session:
        logger.warning("Translation request with invalid token")
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "INVALID_TOKEN",
                    "message": "会话令牌无效或已过期，请重新登录"
                }
            }
        )

    if not session_service.update_activity(session_token):
        logger.warning("Translation request with expired token during activity refresh")
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "INVALID_TOKEN",
                    "message": "会话令牌无效或已过期，请重新登录"
                }
            }
        )
    
    username = session.username
    ip_address = request.client.host if request.client else "unknown"
    
    # Store the session ID in the configuration, for log tracing
    config._session_id = session_token
    
    # [Important] Apply the defaults of disabled parameters first, then check the permissions,
    # so that when the translator parameter is disabled, the default translator set by the administrator is used
    filter_disabled_parameters(config, username, permission_service)
    
    # Extract translator from config (after filter_disabled_parameters applied defaults)
    if translator is None:
        if hasattr(config, 'translator') and hasattr(config.translator, 'translator'):
            translator = config.translator.translator
        else:
            translator = "unknown"
    
    permissions = permission_service.get_user_permissions(username)

    def _feature_value(value) -> str:
        if value is None:
            return ''
        return str(getattr(value, 'value', value) or '')

    def _assert_feature_permission(
        feature_type: str,
        feature_name: str,
        checker,
        allowed_attr: str,
        allowed_details_key: str,
        code: str,
        label: str,
    ) -> None:
        if not checker(username, feature_name):
            allowed_values = getattr(permissions, allowed_attr, []) if permissions else []

            logger.warning(
                f"Permission denied: User '{username}' attempted to use "
                f"unauthorized {feature_type} '{feature_name}'"
            )

            audit_service = get_audit_service()
            audit_service.log_event(
                event_type="permission_denied",
                username=username,
                ip_address=ip_address,
                details={
                    feature_type: feature_name,
                    "reason": f"{feature_type}_not_allowed",
                    allowed_details_key: allowed_values,
                },
                result="failure",
            )

            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "code": code,
                        "message": f"您没有权限使用{label} '{feature_name}'",
                        "details": {
                            feature_type: feature_name,
                            allowed_details_key: allowed_values,
                        },
                    }
                },
            )

    _assert_feature_permission(
        'translator',
        translator,
        permission_service.check_translator_permission,
        'allowed_translators',
        'allowed_translators',
        'TRANSLATOR_PERMISSION_DENIED',
        '翻译器',
    )

    if hasattr(config, 'ocr'):
        primary_ocr = _feature_value(getattr(config.ocr, 'ocr', ''))
        if primary_ocr:
            _assert_feature_permission(
                'ocr',
                primary_ocr,
                permission_service.check_ocr_permission,
                'allowed_ocr',
                'allowed_ocr',
                'OCR_PERMISSION_DENIED',
                'OCR',
            )

        secondary_ocr = _feature_value(getattr(config.ocr, 'secondary_ocr', ''))
        if secondary_ocr and secondary_ocr != primary_ocr:
            _assert_feature_permission(
                'ocr',
                secondary_ocr,
                permission_service.check_ocr_permission,
                'allowed_ocr',
                'allowed_ocr',
                'OCR_PERMISSION_DENIED',
                'OCR',
            )

    if hasattr(config, 'colorizer'):
        colorizer = _feature_value(getattr(config.colorizer, 'colorizer', ''))
        if colorizer:
            _assert_feature_permission(
                'colorizer',
                colorizer,
                permission_service.check_colorizer_permission,
                'allowed_colorizers',
                'allowed_colorizers',
                'COLORIZER_PERMISSION_DENIED',
                '上色器',
            )

    if hasattr(config, 'render'):
        renderer = _feature_value(getattr(config.render, 'renderer', ''))
        if renderer:
            _assert_feature_permission(
                'renderer',
                renderer,
                permission_service.check_renderer_permission,
                'allowed_renderers',
                'allowed_renderers',
                'RENDERER_PERMISSION_DENIED',
                '渲染器',
            )
    
    # Note: checking the concurrency limit and increasing the count is done by track_task_start/track_task_end in the route layer;
    # only authentication and permission checks happen here, and the counters are not changed
    
    logger.info(
        f"Translation auth verified: user='{username}', translator='{translator}'"
    )
    
    return username, ip_address


def log_translation_task_created(
    username: str,
    ip_address: str,
    translator: str,
    config: Config,
    task_id: Optional[str] = None
) -> None:
    """
    Log translation task creation audit event
    
    Args:
        username: Username
        ip_address: IP address
        translator: Translator name
        config: Translation configuration
        task_id: Task ID (optional)
    """
    audit_service = get_audit_service()
    
    # Extract relevant config details
    details = {
        "translator": translator,
        "task_id": task_id or "unknown"
    }
    
    # Add target language if available
    if hasattr(config, 'translator') and hasattr(config.translator, 'target_lang'):
        details["target_lang"] = config.translator.target_lang
    
    # Log audit event
    audit_service.log_event(
        event_type="create_task",
        username=username,
        ip_address=ip_address,
        details=details,
        result="success"
    )
    
    logger.debug(f"Logged translation task creation: user='{username}', translator='{translator}'")


def track_task_start(username: str) -> None:
    """
    Track task start (increment counters and check limits)
    
    This function:
    1. Increments concurrent task count
    2. Checks concurrent limit (raises HTTPException if exceeded)
    3. Checks daily quota (raises HTTPException if exceeded)
    4. Increments daily usage
    
    If any check fails, the concurrent count is rolled back.
    
    Args:
        username: Username
    
    Raises:
        HTTPException: If concurrent limit or daily quota exceeded
    """
    # Increase the concurrency count first
    increment_task_count(username)
    
    # Get the current count for the log
    _, _, permission_service = get_services()
    current_count = permission_service.get_active_task_count(username)
    # Use the effective concurrency limit (from the user group first)
    max_tasks = permission_service.get_effective_max_concurrent(username)
    print(f"[并发检查] 用户 '{username}': 当前任务数={current_count}, 最大允许={max_tasks}")
    
    try:
        # Check the concurrency limit
        check_concurrent_limit(username)
        # Check the daily quota
        check_daily_quota(username)
        # Increase the daily usage
        increment_daily_usage(username)
        print(f"[并发检查] 用户 '{username}': 检查通过，任务开始")
    except Exception as e:
        # The check failed: roll back the concurrency count
        print(f"[并发检查] 用户 '{username}': 检查失败，回滚计数 - {e}")
        decrement_task_count(username)
        raise


def track_task_end(username: str) -> None:
    """
    Track task end (decrement counters)
    
    Args:
        username: Username
    """
    decrement_task_count(username)
