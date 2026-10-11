"""
Enhanced Permission Service

Implements a permission system based on inheritance: global → user group → user.
Supports fine-grained permissions (can_upload_prompt, can_upload_font and so on)
"""

import logging
from typing import Any, Dict, Optional

from manga_translator.server.models.permission_models import UserPermission
from manga_translator.server.repositories.permission_repository import (
    PermissionRepository,
)

logger = logging.getLogger(__name__)


class EnhancedPermissionService:
    """Enhanced permission management service"""
    
    def __init__(self, permission_repo: PermissionRepository):
        """
        Initialise the permission service

        Args:
            permission_repo: the permission repository instance
        """
        self.permission_repo = permission_repo
    
    def check_upload_prompt_permission(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """
        Check whether a user may upload prompts

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            bool: whether the user has the permission
        """
        perms = self.permission_repo.get_effective_permissions(user_id, group_id)
        return perms.get("can_upload_prompt", False)
    
    def check_upload_font_permission(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """
        Check whether a user may upload fonts

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            bool: whether the user has the permission
        """
        perms = self.permission_repo.get_effective_permissions(user_id, group_id)
        result = perms.get("can_upload_font", False)
        logger.info(f"[DEBUG] EnhancedPermissionService.check_upload_font_permission: user={user_id}, group={group_id}, perms={perms}, result={result}")
        return result
    
    def check_delete_own_files_permission(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """
        Check whether a user may delete their own files

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            bool: whether the user has the permission
        """
        perms = self.permission_repo.get_effective_permissions(user_id, group_id)
        return perms.get("can_delete_own_files", True)  # Allowed by default
    
    def check_delete_all_files_permission(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """
        Check whether a user may delete all files

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            bool: whether the user has the permission
        """
        perms = self.permission_repo.get_effective_permissions(user_id, group_id)
        return perms.get("can_delete_all_files", False)
    
    def check_view_permission(self, user_id: str, group_id: Optional[str] = None) -> str:
        """
        Get the view permission level of a user

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            str: the permission level ("own", "none", "all")
        """
        perms = self.permission_repo.get_effective_permissions(user_id, group_id)
        return perms.get("view_permission", "own")
    
    def check_save_enabled(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """
        Check whether saving translation results is enabled for a user

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            bool: whether saving is enabled
        """
        perms = self.permission_repo.get_effective_permissions(user_id, group_id)
        return perms.get("save_enabled", True)
    
    def check_edit_own_env_permission(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """
        Check whether a user may edit their own .env configuration

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            bool: whether the user has the permission
        """
        perms = self.permission_repo.get_effective_permissions(user_id, group_id)
        return perms.get("can_edit_own_env", False)
    
    def check_edit_server_env_permission(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """
        Check whether a user may edit the server .env configuration

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            bool: whether the user has the permission
        """
        perms = self.permission_repo.get_effective_permissions(user_id, group_id)
        return perms.get("can_edit_server_env", False)
    
    def check_view_own_logs_permission(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """
        Check whether a user may view their own logs

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            bool: whether the user has the permission
        """
        perms = self.permission_repo.get_effective_permissions(user_id, group_id)
        return perms.get("can_view_own_logs", True)
    
    def check_view_all_logs_permission(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """
        Check whether a user may view the logs of all users

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            bool: whether the user has the permission
        """
        perms = self.permission_repo.get_effective_permissions(user_id, group_id)
        return perms.get("can_view_all_logs", False)
    
    def check_view_system_logs_permission(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """
        Check whether a user may view the system logs

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            bool: whether the user has the permission
        """
        perms = self.permission_repo.get_effective_permissions(user_id, group_id)
        return perms.get("can_view_system_logs", False)
    
    def get_view_history_permission(self, user_id: str, group_id: Optional[str] = None) -> str:
        """
        Get the history view permission level of a user

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            str: the permission level ("own", "none", "all")
        """
        # Use the view_permission field
        return self.check_view_permission(user_id, group_id)
    
    def is_admin(self, user_id: str) -> bool:
        """
        Check whether a user is an administrator

        Args:
            user_id: the user ID

        Returns:
            bool: whether the user is an administrator
        """
        perms = self.permission_repo.get_effective_permissions(user_id, None)
        # An administrator usually has the can_delete_all_files and can_view_all_logs permissions
        return (perms.get("can_delete_all_files", False) and 
                perms.get("can_view_all_logs", False))
    
    def get_effective_permissions(self, user_id: str, group_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the effective permissions of a user (with the inheritance rules applied)

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            Dict[str, Any]: dictionary of the effective permissions
        """
        return self.permission_repo.get_effective_permissions(user_id, group_id)
    
    def set_user_permissions(
        self,
        user_id: str,
        permissions: Dict[str, Any],
        updated_by: str
    ) -> bool:
        """
        Set the permissions of a user

        Args:
            user_id: the user ID
            permissions: the permission dictionary
            updated_by: ID of whoever makes the update

        Returns:
            bool: whether it succeeded
        """
        try:
            # Create the UserPermission object
            user_perm = UserPermission.create(
                user_id=user_id,
                updated_by=updated_by,
                **permissions
            )
            
            # Save to the repository
            self.permission_repo.set_user_permissions(user_id, user_perm)
            self.permission_repo.update_last_modified()
            
            logger.info(f"Set permissions for user '{user_id}' by '{updated_by}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set permissions for user '{user_id}': {e}")
            return False
    
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
        try:
            self.permission_repo.set_group_permissions(group_id, permissions)
            self.permission_repo.update_last_modified()
            
            logger.info(f"Set permissions for group '{group_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set permissions for group '{group_id}': {e}")
            return False
    
    def set_global_permissions(self, permissions: Dict[str, Any]) -> bool:
        """
        Set the global default permissions

        Args:
            permissions: the permission dictionary

        Returns:
            bool: whether it succeeded
        """
        try:
            self.permission_repo.set_global_permissions(permissions)
            self.permission_repo.update_last_modified()
            
            logger.info("Set global permissions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set global permissions: {e}")
            return False
    
    def delete_user_permissions(self, user_id: str) -> bool:
        """
        Delete the permissions of a user (falling back to the user group / global permissions)

        Args:
            user_id: the user ID

        Returns:
            bool: whether it succeeded
        """
        try:
            success = self.permission_repo.delete_user_permissions(user_id)
            if success:
                self.permission_repo.update_last_modified()
                logger.info(f"Deleted permissions for user '{user_id}'")
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete permissions for user '{user_id}': {e}")
            return False
    
    def delete_group_permissions(self, group_id: str) -> bool:
        """
        Delete the permissions of a user group (falling back to the global permissions)

        Args:
            group_id: the user group ID

        Returns:
            bool: whether it succeeded
        """
        try:
            success = self.permission_repo.delete_group_permissions(group_id)
            if success:
                self.permission_repo.update_last_modified()
                logger.info(f"Deleted permissions for group '{group_id}'")
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete permissions for group '{group_id}': {e}")
            return False
    
    def get_permission_summary(self, user_id: str, group_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of the permissions of a user (for display)

        Args:
            user_id: the user ID
            group_id: the user group ID (optional)

        Returns:
            Dict[str, Any]: the permission summary
        """
        effective_perms = self.get_effective_permissions(user_id, group_id)
        
        return {
            "user_id": user_id,
            "group_id": group_id,
            "upload": {
                "can_upload_prompt": effective_perms.get("can_upload_prompt", False),
                "can_upload_font": effective_perms.get("can_upload_font", False)
            },
            "delete": {
                "can_delete_own_files": effective_perms.get("can_delete_own_files", True),
                "can_delete_all_files": effective_perms.get("can_delete_all_files", False)
            },
            "view": {
                "view_permission": effective_perms.get("view_permission", "own"),
                "can_view_own_logs": effective_perms.get("can_view_own_logs", True),
                "can_view_all_logs": effective_perms.get("can_view_all_logs", False),
                "can_view_system_logs": effective_perms.get("can_view_system_logs", False)
            },
            "config": {
                "can_edit_own_env": effective_perms.get("can_edit_own_env", False),
                "can_edit_server_env": effective_perms.get("can_edit_server_env", False)
            },
            "save_enabled": effective_perms.get("save_enabled", True)
        }


# Global service instance
_enhanced_permission_service: Optional[EnhancedPermissionService] = None


def get_enhanced_permission_service(
    permission_repo: Optional[PermissionRepository] = None
) -> EnhancedPermissionService:
    """
    Get the enhanced permission service instance

    Args:
        permission_repo: the permission repository instance (optional)

    Returns:
        EnhancedPermissionService: the service instance
    """
    global _enhanced_permission_service
    
    if _enhanced_permission_service is None:
        if permission_repo is None:
            # Create the default repository
            permission_repo = PermissionRepository("manga_translator/server/data/permissions.json")
        
        _enhanced_permission_service = EnhancedPermissionService(permission_repo)
    
    return _enhanced_permission_service
