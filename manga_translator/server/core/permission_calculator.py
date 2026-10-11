"""
Permission calculation service

Implements the permission calculation from the user group and the user's own settings.

Rule:
final permissions = (parameters visible to the user group + the user's allowed list) - the user's denied list
Priority: the user's own settings > the user group settings
"""

import logging
from typing import Any, Dict, List, Optional, Set

from manga_translator.server.core.group_service import get_group_service
from manga_translator.server.core.models import UserAccount

logger = logging.getLogger(__name__)


class PermissionCalculator:
    """Permission calculator"""
    
    def __init__(self):
        self.group_service = get_group_service()
    
    def calculate_allowed_translators(self, user: UserAccount) -> List[str]:
        """
        Compute the final list of translators a user may use

        Args:
            user: the user account object

        Returns:
            The list of translators the user may use
        """
        # An administrator is allowed everything
        if user.role == 'admin':
            return ['*']
        
        # Get the group configuration
        self.group_service.get_group(user.group)
        
        # Base set: from the user group (groups do not limit translators for now, so all by default)
        base_translators: Set[str] = {'*'}
        
        # The user's own allowed list
        user_allowed = set(user.permissions.allowed_translators)
        
        # When the user's allowed list contains '*', everything is allowed
        if '*' in user_allowed:
            allowed = base_translators
        else:
            # Otherwise, base set + the user's allowed list
            allowed = base_translators.union(user_allowed)
        
        # Subtract the user's denied list
        denied = set(user.permissions.denied_translators)
        final = allowed - denied
        
        # When the result contains '*' together with other items, keep only '*'
        if '*' in final and len(final) > 1:
            return ['*']
        
        return list(final)
    
    def calculate_allowed_parameters(self, user: UserAccount) -> List[str]:
        """
        Compute the final list of parameters a user may adjust

        Args:
            user: the user account object

        Returns:
            The list of parameters the user may adjust
        """
        # An administrator is allowed everything
        if user.role == 'admin':
            return ['*']
        
        # Get the group configuration
        group_config = self.group_service.get_group(user.group)
        
        # Base set: the visible parameters from the user group
        base_parameters: Set[str] = set()
        if group_config:
            param_config = group_config.get('parameter_config', {})
            for param_name, param_settings in param_config.items():
                if param_settings.get('visible', False):
                    base_parameters.add(param_name)
        
        # When the group has no configuration, everything is allowed by default
        if not base_parameters:
            base_parameters = {'*'}
        
        # The user's own allowed list
        user_allowed = set(user.permissions.allowed_parameters)
        
        # When the user's allowed list contains '*', everything is allowed
        if '*' in user_allowed:
            allowed = base_parameters if '*' not in base_parameters else {'*'}
        else:
            # Otherwise, base set + the user's allowed list
            allowed = base_parameters.union(user_allowed)
        
        # Subtract the user's denied list
        denied = set(user.permissions.denied_parameters)
        final = allowed - denied
        
        # When the result contains '*' together with other items, keep only '*'
        if '*' in final and len(final) > 1:
            return ['*']
        
        return list(final)
    
    def get_parameter_config(self, user: UserAccount, parameter: str) -> Optional[Dict[str, Any]]:
        """
        Get the configuration of a specific parameter for a user (visibility, read-only, default value)

        Args:
            user: the user account object
            parameter: name of the parameter

        Returns:
            The configuration dictionary of the parameter, or None when it is not accessible
        """
        # Check whether the user may access this parameter
        allowed_params = self.calculate_allowed_parameters(user)
        
        if '*' not in allowed_params and parameter not in allowed_params:
            return None
        
        # Get the parameter configuration from the user group
        param_config = self.group_service.get_parameter_config(user.group, parameter)
        
        if not param_config:
            # When the group has no configuration, return the default configuration
            return {
                'visible': True,
                'readonly': False,
                'default_value': None
            }
        
        return param_config
    
    def check_translator_permission(self, user: UserAccount, translator: str) -> bool:
        """
        Check whether a user may use the given translator

        Args:
            user: the user account object
            translator: name of the translator

        Returns:
            Whether the user has the permission
        """
        allowed = self.calculate_allowed_translators(user)
        return '*' in allowed or translator in allowed
    
    def check_parameter_permission(self, user: UserAccount, parameter: str) -> bool:
        """
        Check whether a user may adjust the given parameter

        Args:
            user: the user account object
            parameter: name of the parameter

        Returns:
            Whether the user has the permission
        """
        allowed = self.calculate_allowed_parameters(user)
        return '*' in allowed or parameter in allowed


# Global permission calculator instance
_permission_calculator: Optional[PermissionCalculator] = None


def get_permission_calculator() -> PermissionCalculator:
    """Get the permission calculator instance"""
    global _permission_calculator
    if _permission_calculator is None:
        _permission_calculator = PermissionCalculator()
    return _permission_calculator
