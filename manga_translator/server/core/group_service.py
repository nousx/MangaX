"""
User group configuration management service

Responsible for the parameter configuration of user groups, including the visibility, the read-only state and the default value of parameters.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GroupService:
    """User group configuration management service"""
    
    def __init__(self, config_file: str = "manga_translator/server/data/group_config.json"):
        """
        Initialise the user group service

        Args:
            config_file: path of the user group configuration file
        """
        self.config_file = config_file
        self.groups: Dict[str, Dict[str, Any]] = {}
        self._load_groups()
    
    def _load_groups(self) -> None:
        """Load the user group configuration from the file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.groups = data.get('groups', {})
                logger.info(f"Loaded {len(self.groups)} group(s) from {self.config_file}")
            else:
                logger.warning(f"Group config file not found: {self.config_file}")
                self._create_default_groups()
        except Exception as e:
            logger.error(f"Failed to load group config: {e}")
            self._create_default_groups()
    
    def _create_default_groups(self) -> None:
        """Create the default user group configuration"""
        self.groups = {
            "admin": {
                "name": "Administrators",
                "description": "Administrator user group with all permissions",
                "parameter_config": {}
            },
            "default": {
                "name": "Default users",
                "description": "Default user group of new users",
                "parameter_config": {
                    "target_lang": {
                        "visible": True,
                        "readonly": False,
                        "default_value": "CHS"
                    }
                }
            }
        }
        self._save_groups()
    
    def _save_groups(self) -> None:
        """Save the user group configuration to the file"""
        try:
            data = {
                "version": "1.0",
                "groups": self.groups
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved group config to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save group config: {e}")
    
    def get_group(self, group_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the configuration of a user group

        Args:
            group_name: name of the user group

        Returns:
            The configuration dictionary of the user group, or None when it does not exist
        """
        return self.groups.get(group_name)
    
    def get_parameter_config(self, group_name: str, parameter: str) -> Optional[Dict[str, Any]]:
        """
        Get the configuration of a specific parameter in a user group

        Args:
            group_name: name of the user group
            parameter: name of the parameter

        Returns:
            The configuration dictionary of the parameter, or None when it does not exist
        """
        group = self.get_group(group_name)
        if not group:
            return None
        
        param_config = group.get('parameter_config', {})
        return param_config.get(parameter)
    
    def get_all_groups(self) -> Dict[str, Dict[str, Any]]:
        """Get the configuration of all user groups"""
        return self.groups
    
    def update_group(self, group_name: str, group_data: Dict[str, Any]) -> bool:
        """
        Update the configuration of a user group

        Args:
            group_name: name of the user group
            group_data: the user group data

        Returns:
            Whether it succeeded
        """
        try:
            self.groups[group_name] = group_data
            self._save_groups()
            return True
        except Exception as e:
            logger.error(f"Failed to update group {group_name}: {e}")
            return False
    
    def delete_group(self, group_name: str) -> bool:
        """
        Delete a user group

        Args:
            group_name: name of the user group

        Returns:
            Whether it succeeded
        """
        if group_name in ['admin', 'default']:
            logger.warning(f"Cannot delete system group: {group_name}")
            return False
        
        try:
            if group_name in self.groups:
                del self.groups[group_name]
                self._save_groups()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete group {group_name}: {e}")
            return False


# Global user group service instance
_group_service: Optional[GroupService] = None


def get_group_service() -> GroupService:
    """Get the user group service instance"""
    global _group_service
    if _group_service is None:
        _group_service = GroupService()
    return _group_service
