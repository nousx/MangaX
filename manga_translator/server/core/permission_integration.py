"""
Permission system integration module

Integrates the new permission system with the existing account system.
"""

import logging
from typing import Any, Dict, Optional

from manga_translator.server.core.account_service import AccountService
from manga_translator.server.core.permission_service_v2 import (
    EnhancedPermissionService,
    get_enhanced_permission_service,
)

logger = logging.getLogger(__name__)


class IntegratedPermissionService:
    """Integrated permission service"""
    
    def __init__(
        self,
        account_service: AccountService,
        permission_service: Optional[EnhancedPermissionService] = None
    ):
        """
        Initialise the integrated permission service

        Args:
            account_service: the account service instance
            permission_service: the enhanced permission service instance (optional)
        """
        self.account_service = account_service
        
        if permission_service is None:
            permission_service = get_enhanced_permission_service()
        
        self.permission_service = permission_service
    
    def _get_user_group(self, username: str) -> Optional[str]:
        """
        Get the user group a user belongs to

        Args:
            username: the user name

        Returns:
            The user group ID, or None when the user does not exist
        """
        account = self.account_service.get_user(username)
        if account:
            return account.group
        return None
    
    def check_upload_prompt_permission(self, username: str) -> bool:
        """
        Check whether a user may upload prompts

        Args:
            username: the user name

        Returns:
            bool: whether the user has the permission
        """
        group_id = self._get_user_group(username)
        return self.permission_service.check_upload_prompt_permission(username, group_id)
    
    def check_upload_font_permission(self, username: str) -> bool:
        """
        Check whether a user may upload fonts

        Args:
            username: the user name

        Returns:
            bool: whether the user has the permission
        """
        group_id = self._get_user_group(username)
        result = self.permission_service.check_upload_font_permission(username, group_id)
        logger.info(f"[DEBUG] check_upload_font_permission: user={username}, group={group_id}, result={result}")
        return result
    
    def check_delete_own_files_permission(self, username: str) -> bool:
        """
        Check whether a user may delete their own files

        Args:
            username: the user name

        Returns:
            bool: whether the user has the permission
        """
        group_id = self._get_user_group(username)
        return self.permission_service.check_delete_own_files_permission(username, group_id)
    
    def check_delete_all_files_permission(self, username: str) -> bool:
        """
        Check whether a user may delete all files

        Args:
            username: the user name

        Returns:
            bool: whether the user has the permission
        """
        group_id = self._get_user_group(username)
        return self.permission_service.check_delete_all_files_permission(username, group_id)
    
    def check_delete_file_permission(self, username: str, file_owner: str) -> bool:
        """
        Check whether a user may delete the given file

        Args:
            username: the user name
            file_owner: user name of the file owner

        Returns:
            bool: whether the user has the permission
        """
        # For one's own file, check can_delete_own_files
        if username == file_owner:
            return self.check_delete_own_files_permission(username)
        
        # For someone else's file, check can_delete_all_files
        return self.check_delete_all_files_permission(username)
    
    def check_view_permission(self, username: str) -> str:
        """
        Get the view permission level of a user

        Args:
            username: the user name

        Returns:
            str: the permission level ("own", "none", "all")
        """
        group_id = self._get_user_group(username)
        return self.permission_service.check_view_permission(username, group_id)
    
    def get_view_history_permission(self, username: str) -> str:
        """
        Get the history view permission level of a user

        Args:
            username: the user name

        Returns:
            str: the permission level ("own", "none", "all")
        """
        group_id = self._get_user_group(username)
        return self.permission_service.get_view_history_permission(username, group_id)
    
    def check_save_enabled(self, username: str) -> bool:
        """
        Check whether saving translation results is enabled for a user

        Args:
            username: the user name

        Returns:
            bool: whether saving is enabled
        """
        group_id = self._get_user_group(username)
        return self.permission_service.check_save_enabled(username, group_id)
    
    def check_edit_own_env_permission(self, username: str) -> bool:
        """
        Check whether a user may edit their own .env configuration

        Args:
            username: the user name

        Returns:
            bool: whether the user has the permission
        """
        group_id = self._get_user_group(username)
        return self.permission_service.check_edit_own_env_permission(username, group_id)
    
    def check_edit_server_env_permission(self, username: str) -> bool:
        """
        Check whether a user may edit the server .env configuration

        Args:
            username: the user name

        Returns:
            bool: whether the user has the permission
        """
        group_id = self._get_user_group(username)
        return self.permission_service.check_edit_server_env_permission(username, group_id)
    
    def check_view_own_logs_permission(self, username: str) -> bool:
        """
        Check whether a user may view their own logs

        Args:
            username: the user name

        Returns:
            bool: whether the user has the permission
        """
        group_id = self._get_user_group(username)
        return self.permission_service.check_view_own_logs_permission(username, group_id)
    
    def check_view_all_logs_permission(self, username: str) -> bool:
        """
        Check whether a user may view the logs of all users

        Args:
            username: the user name

        Returns:
            bool: whether the user has the permission
        """
        group_id = self._get_user_group(username)
        return self.permission_service.check_view_all_logs_permission(username, group_id)
    
    def check_view_system_logs_permission(self, username: str) -> bool:
        """
        Check whether a user may view the system logs

        Args:
            username: the user name

        Returns:
            bool: whether the user has the permission
        """
        group_id = self._get_user_group(username)
        return self.permission_service.check_view_system_logs_permission(username, group_id)
    
    def check_view_logs_permission(self, username: str, log_owner: Optional[str] = None) -> bool:
        """
        Check whether a user may view the given logs

        Args:
            username: the user name
            log_owner: user name of the log owner (None means the system logs)

        Returns:
            bool: whether the user has the permission
        """
        # System logs
        if log_owner is None:
            return self.check_view_system_logs_permission(username)
        
        # One's own logs
        if username == log_owner:
            return self.check_view_own_logs_permission(username)
        
        # Other people's logs
        return self.check_view_all_logs_permission(username)
    
    def get_effective_permissions(self, username: str) -> Dict[str, Any]:
        """
        Get the effective permissions of a user (with the inheritance rules applied)

        Args:
            username: the user name

        Returns:
            Dict[str, Any]: dictionary of the effective permissions
        """
        group_id = self._get_user_group(username)
        return self.permission_service.get_effective_permissions(username, group_id)
    
    def get_permission_summary(self, username: str) -> Dict[str, Any]:
        """
        Get a summary of the permissions of a user (for display)

        Args:
            username: the user name

        Returns:
            Dict[str, Any]: the permission summary
        """
        group_id = self._get_user_group(username)
        return self.permission_service.get_permission_summary(username, group_id)
    
    def set_user_permissions(
        self,
        username: str,
        permissions: Dict[str, Any],
        updated_by: str
    ) -> bool:
        """
        Set the permissions of a user

        Args:
            username: the user name
            permissions: the permission dictionary
            updated_by: user name of whoever makes the update

        Returns:
            bool: whether it succeeded
        """
        return self.permission_service.set_user_permissions(username, permissions, updated_by)
    
    def set_group_permissions(
        self,
        group_id: str,
        permissions: Dict[str, Any]
    ) -> bool:
        """
        Set the permissions of a user group

        Args:
            group_id: the user group ID
            permissions: the permission dictionary

        Returns:
            bool: whether it succeeded
        """
        return self.permission_service.set_group_permissions(group_id, permissions)
    
    def set_global_permissions(self, permissions: Dict[str, Any]) -> bool:
        """
        Set the global default permissions

        Args:
            permissions: the permission dictionary

        Returns:
            bool: whether it succeeded
        """
        return self.permission_service.set_global_permissions(permissions)
    
    def delete_user_permissions(self, username: str) -> bool:
        """
        Delete the permissions of a user (falling back to the user group / global permissions)

        Args:
            username: the user name

        Returns:
            bool: whether it succeeded
        """
        return self.permission_service.delete_user_permissions(username)
    
    def delete_group_permissions(self, group_id: str) -> bool:
        """
        Delete the permissions of a user group (falling back to the global permissions)

        Args:
            group_id: the user group ID

        Returns:
            bool: whether it succeeded
        """
        return self.permission_service.delete_group_permissions(group_id)


# Global service instance
_integrated_permission_service: Optional[IntegratedPermissionService] = None


def get_integrated_permission_service(
    account_service: Optional[AccountService] = None
) -> IntegratedPermissionService:
    """
    Get the integrated permission service instance

    Args:
        account_service: the account service instance (optional)

    Returns:
        IntegratedPermissionService: the service instance
    """
    global _integrated_permission_service
    
    if _integrated_permission_service is None:
        if account_service is None:
            # Import and get the default account service
            from .account_service import get_account_service
            account_service = get_account_service()
        
        _integrated_permission_service = IntegratedPermissionService(account_service)
    
    return _integrated_permission_service
