"""
Permission management service (PermissionService)

Checks user permissions, filters configuration data, and manages concurrency limits and quotas.
"""

import logging
from collections import defaultdict
from datetime import date
from typing import Any, Dict, Optional

from manga_translator.server.core.account_service import AccountService
from manga_translator.server.core.models import UserPermissions

logger = logging.getLogger(__name__)


class PermissionService:
    """Permission management service"""

    FEATURE_PERMISSION_FIELDS = {
        'translator': ('allowed_translators', 'denied_translators'),
        'ocr': ('allowed_ocr', 'denied_ocr'),
        'colorizer': ('allowed_colorizers', 'denied_colorizers'),
        'renderer': ('allowed_renderers', 'denied_renderers'),
        'workflow': ('allowed_workflows', 'denied_workflows'),
    }
    
    def __init__(self, account_service: AccountService):
        """
        Initialise the permission management service

        Args:
            account_service: the account management service instance
        """
        self.account_service = account_service
        
        # Tracks the number of active tasks of each user: username -> count
        self.active_tasks: Dict[str, int] = defaultdict(int)
        
        # Tracks the daily quota use of each user: (username, date) -> count
        self.daily_usage: Dict[tuple, int] = defaultdict(int)
    
    def _resolve_feature_permission_fields(self, feature_type: str) -> tuple[str, str]:
        fields = self.FEATURE_PERMISSION_FIELDS.get(feature_type)
        if not fields:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        return fields

    def _get_group_feature_permissions(self, account, allowed_field: str, denied_field: str) -> tuple[set, set]:
        group_allowed = set()
        group_denied = set()

        try:
            from manga_translator.server.core.group_management_service import (
                get_group_management_service,
            )
            group_service = get_group_management_service()
            group = group_service.get_group(account.group)

            if group:
                group_allowed = set(group.get(allowed_field, []))
                group_denied = set(group.get(denied_field, []))
        except Exception as e:
            logger.warning(f"Failed to get group permissions for {allowed_field}: {e}")

        return group_allowed, group_denied

    def check_feature_permission(self, username: str, feature_type: str, feature_name: str) -> bool:
        """
        Check the permission for the given capability.

        Priority (high to low):
        1. the user's deny list
        2. the user's allow list (it can unlock the deny list of the user group)
        3. the deny list of the user group
        4. the allow list of the user group
        """
        account = self.account_service.get_user(username)
        if not account:
            logger.warning(f"User not found: {username}")
            return False

        allowed_field, denied_field = self._resolve_feature_permission_fields(feature_type)
        permissions = account.permissions

        user_allowed = set(getattr(permissions, allowed_field, []) or [])
        user_denied = set(getattr(permissions, denied_field, []) or [])

        if feature_name in user_denied:
            return False

        if "*" in user_allowed or feature_name in user_allowed:
            return True

        group_allowed, group_denied = self._get_group_feature_permissions(
            account,
            allowed_field,
            denied_field,
        )

        if feature_name in group_denied:
            return False

        if not group_allowed or "*" in group_allowed or feature_name in group_allowed:
            return True

        return False

    def filter_allowed_options(self, username: str, feature_type: str, options: list[str]) -> list[str]:
        """Filter a list of options by the permissions of a user."""
        return [
            option for option in options
            if self.check_feature_permission(username, feature_type, option)
        ]

    def check_translator_permission(self, username: str, translator: str) -> bool:
        """Check the translator permission."""
        return self.check_feature_permission(username, 'translator', translator)

    def check_ocr_permission(self, username: str, ocr: str) -> bool:
        """Check the OCR permission."""
        return self.check_feature_permission(username, 'ocr', ocr)

    def check_colorizer_permission(self, username: str, colorizer: str) -> bool:
        """Check the colorizer permission."""
        return self.check_feature_permission(username, 'colorizer', colorizer)

    def check_renderer_permission(self, username: str, renderer: str) -> bool:
        """Check the renderer permission."""
        return self.check_feature_permission(username, 'renderer', renderer)

    def check_workflow_permission(self, username: str, workflow: str) -> bool:
        """Check the workflow permission."""
        return self.check_feature_permission(username, 'workflow', workflow)
    
    def check_parameter_permission(self, username: str, parameter: str) -> bool:
        """
        Check a parameter permission

        Args:
            username: the user name
            parameter: name of the parameter (such as "translator.target_lang")

        Returns:
            bool: whether the user may adjust that parameter
        """
        account = self.account_service.get_user(username)
        if not account:
            logger.warning(f"User not found: {username}")
            return False
        
        permissions = account.permissions
        
        # Check for a wildcard permission
        if "*" in permissions.allowed_parameters:
            return True
        
        # Check whether it is in the allowed list
        return parameter in permissions.allowed_parameters
    
    def filter_parameters(self, username: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter parameters, keeping only those the user has permission for

        Args:
            username: the user name
            parameters: the original parameter dictionary

        Returns:
            Dict[str, Any]: the filtered parameter dictionary
        """
        account = self.account_service.get_user(username)
        if not account:
            logger.warning(f"User not found: {username}")
            return {}
        
        permissions = account.permissions
        
        # With a wildcard permission, return all parameters
        if "*" in permissions.allowed_parameters:
            return parameters
        
        # Filter the parameters
        filtered = {}
        for key, value in parameters.items():
            if self.check_parameter_permission(username, key):
                filtered[key] = value
            else:
                logger.debug(f"Filtered parameter '{key}' for user '{username}'")
        
        return filtered
    
    def check_offline_translation_permission(self, username: str) -> bool:
        """
        Check whether a user has the offline translation permission (tasks keep running after the user goes offline)

        Args:
            username: the user name

        Returns:
            bool: whether offline translation is allowed
        """
        account = self.account_service.get_user(username)
        if not account:
            logger.warning(f"User not found: {username}")
            return False
        
        # Check the user's permissions
        if account.permissions.allow_offline_translation:
            return True
        
        # Check the group's permissions
        try:
            from manga_translator.server.core.group_management_service import (
                get_group_management_service,
            )
            group_service = get_group_management_service()
            group = group_service.get_group(account.group)
            
            if group:
                return group.get('allow_offline_translation', False)
        except Exception as e:
            logger.warning(f"Failed to get group offline translation permission: {e}")
        
        return False
    
    def check_concurrent_limit(self, username: str) -> bool:
        """
        Check the concurrency limit

        Note: this function should be called after increment_task_count,
        so the condition checked is current_tasks <= max (not <)

        Priority: the user group configuration > the user configuration

        Args:
            username: the user name

        Returns:
            bool: whether the user may create a new task (the concurrency limit is not exceeded)
        """
        account = self.account_service.get_user(username)
        if not account:
            logger.warning(f"User not found: {username}")
            return False
        
        # Get the effective concurrency limit (from the user group first)
        max_concurrent = self.get_effective_max_concurrent(username)
        current_tasks = self.active_tasks.get(username, 0)
        
        # Check whether the limit is exceeded (the count was increased first, hence <=)
        can_create = current_tasks <= max_concurrent
        
        if not can_create:
            logger.info(
                f"User '{username}' reached concurrent task limit "
                f"({current_tasks}/{max_concurrent})"
            )
        
        return can_create
    
    def get_effective_max_concurrent(self, username: str) -> int:
        """
        Get the effective maximum concurrency of a user (taken from the user group first)

        Args:
            username: the user name

        Returns:
            int: the maximum number of concurrent tasks
        """
        account = self.account_service.get_user(username)
        if not account:
            return 1  # The default limit is 1
        
        # Take the concurrency limit from the user group first
        try:
            from manga_translator.server.core.group_management_service import (
                get_group_management_service,
            )
            group_service = get_group_management_service()
            group = group_service.get_group(account.group)
            if group:
                param_config = group.get('parameter_config', {})
                quota_config = param_config.get('quota', {})
                group_max = quota_config.get('max_concurrent_tasks')
                if group_max is not None and group_max > 0:
                    return group_max
        except Exception as e:
            logger.warning(f"Failed to get group concurrent limit: {e}")
        
        # When the group has none set, use the user-level configuration
        return account.permissions.max_concurrent_tasks
    
    def check_daily_quota(self, username: str) -> bool:
        """
        Check the daily quota

        Args:
            username: the user name

        Returns:
            bool: whether the user has quota left
        """
        account = self.account_service.get_user(username)
        if not account:
            logger.warning(f"User not found: {username}")
            return False
        
        # Get the quota settings of the user group
        daily_quota = -1  # No limit by default
        try:
            from manga_translator.server.core.group_management_service import (
                get_group_management_service,
            )
            group_service = get_group_management_service()
            group = group_service.get_group(account.group)
            if group:
                param_config = group.get('parameter_config', {})
                quota_config = param_config.get('quota', {})
                # Use the daily_image_limit of the user group
                group_quota = quota_config.get('daily_image_limit', -1)
                if group_quota is not None and group_quota > 0:
                    daily_quota = group_quota
        except Exception as e:
            logger.warning(f"Failed to get group quota: {e}")
        
        # When the group has none set, use the user-level quota
        if daily_quota == -1:
            daily_quota = account.permissions.daily_quota
        
        # -1 means no limit
        if daily_quota == -1:
            return True
        
        # Get today's usage
        today = date.today()
        usage_key = (username, today)
        current_usage = self.daily_usage.get(usage_key, 0)
        
        # Check whether the quota is exceeded
        can_create = current_usage < daily_quota
        
        if not can_create:
            logger.info(
                f"User '{username}' reached daily quota "
                f"({current_usage}/{daily_quota})"
            )
        
        return can_create
    
    def increment_task_count(self, username: str) -> None:
        """
        Increase the active task count of a user

        Args:
            username: the user name
        """
        self.active_tasks[username] = self.active_tasks.get(username, 0) + 1
        logger.debug(f"User '{username}' active tasks: {self.active_tasks[username]}")
    
    def decrement_task_count(self, username: str) -> None:
        """
        Decrease the active task count of a user

        Args:
            username: the user name
        """
        if username in self.active_tasks:
            self.active_tasks[username] = max(0, self.active_tasks[username] - 1)
            logger.debug(f"User '{username}' active tasks: {self.active_tasks[username]}")
    
    def increment_daily_usage(self, username: str) -> None:
        """
        Increase the daily usage of a user

        Args:
            username: the user name
        """
        today = date.today()
        usage_key = (username, today)
        self.daily_usage[usage_key] = self.daily_usage.get(usage_key, 0) + 1
        logger.debug(f"User '{username}' daily usage: {self.daily_usage[usage_key]}")
    
    def get_user_permissions(self, username: str) -> Optional[UserPermissions]:
        """
        Get the permissions of a user

        Args:
            username: the user name

        Returns:
            Optional[UserPermissions]: the user permissions object, or None when the user does not exist
        """
        account = self.account_service.get_user(username)
        if not account:
            return None
        
        return account.permissions
    
    def update_user_permissions(
        self,
        username: str,
        permissions: UserPermissions
    ) -> bool:
        """
        Update the permissions of a user (takes effect at once)

        Args:
            username: the user name
            permissions: the new permissions object

        Returns:
            bool: whether the update succeeded
        """
        try:
            success = self.account_service.update_user(
                username,
                {'permissions': permissions}
            )
            
            if success:
                logger.info(f"Updated permissions for user: {username}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to update permissions for user '{username}': {e}")
            return False
    
    def filter_config_for_user(
        self,
        username: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter configuration data for a user

        Args:
            username: the user name
            config: the original configuration dictionary

        Returns:
            Dict[str, Any]: the filtered configuration dictionary
        """
        account = self.account_service.get_user(username)
        if not account:
            logger.warning(f"User not found: {username}")
            return {}
        
        permissions = account.permissions
        filtered_config = config.copy()
        
        # Filter the translator list
        if 'translators' in filtered_config:
            filtered_config['translators'] = self.filter_allowed_options(
                username,
                'translator',
                filtered_config['translators'],
            )
        
        # Filter the parameters
        if 'parameters' in filtered_config:
            filtered_config['parameters'] = self.filter_parameters(
                username,
                filtered_config['parameters']
            )
        
        # Add the user's permission information
        filtered_config['user_permissions'] = {
            'allowed_translators': permissions.allowed_translators,
            'allowed_ocr': permissions.allowed_ocr,
            'allowed_colorizers': permissions.allowed_colorizers,
            'allowed_renderers': permissions.allowed_renderers,
            'allowed_workflows': permissions.allowed_workflows,
            'allowed_parameters': permissions.allowed_parameters,
            'max_concurrent_tasks': permissions.max_concurrent_tasks,
            'daily_quota': permissions.daily_quota,
            'can_upload_files': permissions.can_upload_files,
            'can_delete_files': permissions.can_delete_files
        }
        
        return filtered_config
    
    def get_active_task_count(self, username: str) -> int:
        """
        Get the number of active tasks of a user

        Args:
            username: the user name

        Returns:
            int: the number of active tasks
        """
        return self.active_tasks.get(username, 0)
    
    def get_daily_usage(self, username: str) -> int:
        """
        Get today's usage of a user

        Args:
            username: the user name

        Returns:
            int: today's usage
        """
        today = date.today()
        usage_key = (username, today)
        return self.daily_usage.get(usage_key, 0)
    
    def get_effective_daily_quota(self, username: str) -> int:
        """
        Get the effective daily quota of a user (taken from the user group first)

        Args:
            username: the user name

        Returns:
            int: the daily quota; -1 means unlimited
        """
        account = self.account_service.get_user(username)
        if not account:
            return -1
        
        # Take the quota from the user group first
        try:
            from manga_translator.server.core.group_management_service import (
                get_group_management_service,
            )
            group_service = get_group_management_service()
            group = group_service.get_group(account.group)
            if group:
                param_config = group.get('parameter_config', {})
                quota_config = param_config.get('quota', {})
                group_quota = quota_config.get('daily_image_limit', -1)
                if group_quota is not None and group_quota > 0:
                    return group_quota
        except Exception as e:
            logger.warning(f"Failed to get group quota: {e}")
        
        # When the group has none set, use the user-level quota
        return account.permissions.daily_quota
    
    def cleanup_old_usage_data(self) -> None:
        """
        Remove old usage data (the last 7 days are kept)
        """
        today = date.today()
        keys_to_remove = []
        
        for (username, usage_date) in self.daily_usage.keys():
            days_old = (today - usage_date).days
            if days_old > 7:
                keys_to_remove.append((username, usage_date))
        
        for key in keys_to_remove:
            del self.daily_usage[key]
        
        if keys_to_remove:
            logger.info(f"Cleaned up {len(keys_to_remove)} old usage records")
    
    def get_effective_file_permissions(self, username: str) -> dict:
        """
        Get the effective file operation permissions of a user (taken from the user group first)

        Args:
            username: the user name

        Returns:
            dict: dictionary of the file operation permissions
        """
        account = self.account_service.get_user(username)
        if not account:
            return {
                'can_upload_fonts': False,
                'can_delete_fonts': False,
                'can_upload_prompts': False,
                'can_delete_prompts': False,
            }
        
        # User-level permissions are used by default
        result = {
            'can_upload_fonts': account.permissions.can_upload_files,
            'can_delete_fonts': account.permissions.can_delete_files,
            'can_upload_prompts': account.permissions.can_upload_files,
            'can_delete_prompts': account.permissions.can_delete_files,
        }
        
        # Take the permissions from the user group first
        try:
            from manga_translator.server.core.group_management_service import (
                get_group_management_service,
            )
            group_service = get_group_management_service()
            group = group_service.get_group(account.group)
            if group:
                param_config = group.get('parameter_config', {})
                perm_config = param_config.get('permissions', {})
                
                # When the group has a configuration, use it
                if 'can_upload_fonts' in perm_config:
                    result['can_upload_fonts'] = perm_config['can_upload_fonts']
                if 'can_delete_fonts' in perm_config:
                    result['can_delete_fonts'] = perm_config['can_delete_fonts']
                if 'can_upload_prompts' in perm_config:
                    result['can_upload_prompts'] = perm_config['can_upload_prompts']
                if 'can_delete_prompts' in perm_config:
                    result['can_delete_prompts'] = perm_config['can_delete_prompts']
        except Exception as e:
            logger.warning(f"Failed to get group file permissions: {e}")
        
        return result
    
    def can_upload_fonts(self, username: str) -> bool:
        """Check whether a user may upload fonts"""
        return self.get_effective_file_permissions(username).get('can_upload_fonts', False)
    
    def can_delete_fonts(self, username: str) -> bool:
        """Check whether a user may delete fonts"""
        return self.get_effective_file_permissions(username).get('can_delete_fonts', False)
    
    def can_upload_prompts(self, username: str) -> bool:
        """Check whether a user may upload prompts"""
        return self.get_effective_file_permissions(username).get('can_upload_prompts', False)
    
    def can_delete_prompts(self, username: str) -> bool:
        """Check whether a user may delete prompts"""
        return self.get_effective_file_permissions(username).get('can_delete_prompts', False)
