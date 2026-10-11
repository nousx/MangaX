"""
User Group Management Service

Implements creating, renaming and deleting user groups and managing their configuration.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from manga_translator.server.models.group_models import UserGroup
from manga_translator.server.repositories.group_repository import GroupRepository

logger = logging.getLogger(__name__)


class GroupManagementService:
    """User group management service"""
    
    def __init__(self, group_repo: GroupRepository, accounts_file: str):
        """
        Initialise the user group management service

        Args:
            group_repo: the user group repository instance
            accounts_file: path of the accounts file
        """
        self.group_repo = group_repo
        self.accounts_file = accounts_file
    
    def create_group(
        self,
        group_id: str,
        name: str,
        description: str,
        admin_id: str,
        permissions: Optional[Dict[str, Any]] = None,
        quota_limits: Optional[Dict[str, Any]] = None,
        visible_presets: Optional[List[str]] = None,
        parameter_config: Optional[Dict[str, Any]] = None
    ) -> Optional[UserGroup]:
        """
        Create a new user group

        Args:
            group_id: the user group ID
            name: name of the user group
            description: the description
            admin_id: ID of the administrator who creates it
            permissions: the permission configuration
            quota_limits: the quota limits
            visible_presets: list of visible presets
            parameter_config: the parameter configuration

        Returns:
            UserGroup: the user group object that was created, or None on failure
        """
        try:
            # Check whether the user group already exists
            if self.group_repo.group_exists(group_id):
                logger.error(f"Group '{group_id}' already exists")
                return None
            
            # Create the group data
            group_data = {
                "name": name,
                "description": description,
                "parameter_config": parameter_config or {}
            }
            
            # Create the user group
            success = self.group_repo.create_group(group_id, group_data)
            
            if not success:
                logger.error(f"Failed to create group '{group_id}'")
                return None
            
            # Write the audit log
            self._log_audit(admin_id, "create_group", {
                "group_id": group_id,
                "name": name
            })
            
            logger.info(f"Created group '{group_id}' by admin '{admin_id}'")
            
            # Return the group object
            return UserGroup(
                id=group_id,
                name=name,
                description=description,
                permissions=permissions or {},
                quota_limits=quota_limits or {},
                visible_presets=visible_presets or [],
                created_at=datetime.now(timezone.utc).isoformat(),
                created_by=admin_id,
                is_system=False
            )
            
        except Exception as e:
            logger.error(f"Error creating group '{group_id}': {e}")
            return None
    
    def rename_group(
        self,
        old_group_id: str,
        new_group_id: str,
        new_name: str,
        admin_id: str
    ) -> bool:
        """
        Rename a user group

        Args:
            old_group_id: the current user group ID
            new_group_id: the new user group ID
            new_name: the new user group name
            admin_id: the administrator ID

        Returns:
            bool: whether it succeeded
        """
        try:
            # Check whether it is a system group
            if self.group_repo.is_system_group(old_group_id):
                logger.error(f"Cannot rename system group '{old_group_id}'")
                return False
            
            # Check whether the old group exists
            if not self.group_repo.group_exists(old_group_id):
                logger.error(f"Group '{old_group_id}' does not exist")
                return False
            
            # Check whether the new group ID already exists
            if self.group_repo.group_exists(new_group_id):
                logger.error(f"Group '{new_group_id}' already exists")
                return False
            
            # Rename the user group
            success = self.group_repo.rename_group(old_group_id, new_group_id, new_name)
            
            if not success:
                logger.error(f"Failed to rename group '{old_group_id}' to '{new_group_id}'")
                return False
            
            # Update the group link of every user that belongs to the group
            self._update_user_group_associations(old_group_id, new_group_id)
            
            # Write the audit log
            self._log_audit(admin_id, "rename_group", {
                "old_group_id": old_group_id,
                "new_group_id": new_group_id,
                "new_name": new_name
            })
            
            logger.info(f"Renamed group '{old_group_id}' to '{new_group_id}' by admin '{admin_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Error renaming group '{old_group_id}': {e}")
            return False
    
    def delete_group(self, group_id: str, admin_id: str) -> bool:
        """
        Delete a user group

        Args:
            group_id: the user group ID
            admin_id: the administrator ID

        Returns:
            bool: whether it succeeded
        """
        try:
            # Check whether it is a system group
            if self.group_repo.is_system_group(group_id):
                logger.error(f"Cannot delete system group '{group_id}'")
                return False
            
            # Check whether the group exists
            if not self.group_repo.group_exists(group_id):
                logger.error(f"Group '{group_id}' does not exist")
                return False
            
            # Move all users of the group to the default group
            moved_count = self._move_users_to_default_group(group_id)
            
            # Delete the user group
            success = self.group_repo.delete_group(group_id)
            
            if not success:
                logger.error(f"Failed to delete group '{group_id}'")
                return False
            
            # Write the audit log
            self._log_audit(admin_id, "delete_group", {
                "group_id": group_id,
                "moved_users": moved_count
            })
            
            logger.info(f"Deleted group '{group_id}' by admin '{admin_id}', moved {moved_count} users to default")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting group '{group_id}': {e}")
            return False
    
    def get_all_groups(self) -> List[Dict[str, Any]]:
        """
        Get all user groups

        Returns:
            List[Dict]: list of the user groups
        """
        try:
            groups = self.group_repo.get_all_groups()
            
            # Convert to list format
            result = []
            for group_id, group_data in groups.items():
                result.append({
                    "id": group_id,
                    "name": group_data.get("name", group_id),
                    "description": group_data.get("description", ""),
                    "parameter_config": group_data.get("parameter_config", {}),
                    "allowed_translators": group_data.get("allowed_translators", ["*"]),
                    "denied_translators": group_data.get("denied_translators", []),
                    "allowed_ocr": group_data.get("allowed_ocr", ["*"]),
                    "denied_ocr": group_data.get("denied_ocr", []),
                    "allowed_colorizers": group_data.get("allowed_colorizers", ["*"]),
                    "denied_colorizers": group_data.get("denied_colorizers", []),
                    "allowed_renderers": group_data.get("allowed_renderers", ["*"]),
                    "denied_renderers": group_data.get("denied_renderers", []),
                    "allowed_workflows": group_data.get("allowed_workflows", ["*"]),
                    "denied_workflows": group_data.get("denied_workflows", []),
                    "default_preset_id": group_data.get("default_preset_id"),
                    "visible_presets": group_data.get("visible_presets", []),
                    "is_system": self.group_repo.is_system_group(group_id)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting all groups: {e}")
            return []
    
    def get_group(self, group_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single user group

        Args:
            group_id: the user group ID

        Returns:
            Dict: the user group information, or None when it does not exist
        """
        try:
            group_data = self.group_repo.get_group(group_id)
            
            if not group_data:
                return None
            
            return {
                "id": group_id,
                "name": group_data.get("name", group_id),
                "description": group_data.get("description", ""),
                "parameter_config": group_data.get("parameter_config", {}),
                "allowed_translators": group_data.get("allowed_translators", ["*"]),
                "denied_translators": group_data.get("denied_translators", []),
                "allowed_ocr": group_data.get("allowed_ocr", ["*"]),
                "denied_ocr": group_data.get("denied_ocr", []),
                "allowed_colorizers": group_data.get("allowed_colorizers", ["*"]),
                "denied_colorizers": group_data.get("denied_colorizers", []),
                "allowed_renderers": group_data.get("allowed_renderers", ["*"]),
                "denied_renderers": group_data.get("denied_renderers", []),
                "allowed_workflows": group_data.get("allowed_workflows", ["*"]),
                "denied_workflows": group_data.get("denied_workflows", []),
                "allowed_languages": group_data.get("allowed_languages", ["*"]),
                "denied_languages": group_data.get("denied_languages", []),
                "default_preset_id": group_data.get("default_preset_id"),
                "visible_presets": group_data.get("visible_presets", []),
                "allow_offline_translation": group_data.get("allow_offline_translation", False),
                "is_system": self.group_repo.is_system_group(group_id)
            }
            
        except Exception as e:
            logger.error(f"Error getting group '{group_id}': {e}")
            return None
    
    def update_group_config(
        self,
        group_id: str,
        config: Dict[str, Any],
        admin_id: str
    ) -> bool:
        """
        Update the configuration of a user group

        Args:
            group_id: the user group ID
            config: the new configuration
            admin_id: the administrator ID

        Returns:
            bool: whether it succeeded
        """
        try:
            # Check whether the group exists
            if not self.group_repo.group_exists(group_id):
                logger.error(f"Group '{group_id}' does not exist")
                return False
            
            # Update the configuration
            success = self.group_repo.update_group_config(group_id, config)
            
            if not success:
                logger.error(f"Failed to update config for group '{group_id}'")
                return False
            
            # Write the audit log
            self._log_audit(admin_id, "update_group_config", {
                "group_id": group_id
            })
            
            logger.info(f"Updated config for group '{group_id}' by admin '{admin_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Error updating config for group '{group_id}': {e}")
            return False
    
    def _update_user_group_associations(self, old_group_id: str, new_group_id: str) -> int:
        """
        Update the group link of all users

        Args:
            old_group_id: the old user group ID
            new_group_id: the new user group ID

        Returns:
            int: number of users updated
        """
        try:
            # Read the accounts file
            with open(self.accounts_file, 'r', encoding='utf-8') as f:
                accounts_data = json.load(f)
            
            updated_count = 0
            
            # Update every user that belongs to the old group
            for account in accounts_data.get("accounts", []):
                if account.get("group") == old_group_id:
                    account["group"] = new_group_id
                    updated_count += 1
            
            # Write the file back
            if updated_count > 0:
                with open(self.accounts_file, 'w', encoding='utf-8') as f:
                    json.dump(accounts_data, f, indent=2, ensure_ascii=False)
            
            return updated_count
            
        except Exception as e:
            logger.error(f"Error updating user group associations: {e}")
            return 0
    
    def _move_users_to_default_group(self, group_id: str) -> int:
        """
        Move all users of a user group to the default group

        Args:
            group_id: ID of the user group being deleted

        Returns:
            int: number of users moved
        """
        return self._update_user_group_associations(group_id, "default")
    
    def _log_audit(self, admin_id: str, action: str, details: Dict[str, Any]) -> None:
        """
        Write an audit log record

        Args:
            admin_id: the administrator ID
            action: the kind of operation
            details: details of the operation
        """
        try:
            audit_file = "manga_translator/server/data/audit.log"
            
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "admin_id": admin_id,
                "action": action,
                "details": details
            }
            
            with open(audit_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Error writing audit log: {e}")


# Global service instance
_group_management_service: Optional[GroupManagementService] = None


def get_group_management_service(
    group_repo: Optional[GroupRepository] = None,
    accounts_file: Optional[str] = None
) -> GroupManagementService:
    """
    Get the user group management service instance

    Args:
        group_repo: the user group repository instance (optional)
        accounts_file: path of the accounts file (optional)

    Returns:
        GroupManagementService: the service instance
    """
    global _group_management_service
    
    if _group_management_service is None:
        if group_repo is None:
            group_repo = GroupRepository("manga_translator/server/data/group_config.json")
        
        if accounts_file is None:
            accounts_file = "manga_translator/server/data/accounts.json"
        
        _group_management_service = GroupManagementService(group_repo, accounts_file)
    
    return _group_management_service
