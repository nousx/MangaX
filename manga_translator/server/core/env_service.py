"""
Environment variable service (EnvService)

Manages loading, parsing, updating and hot reloading of the .env file.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from manga_translator.utils.dotenv_utils import (
    normalize_env_value,
    read_dotenv_file,
    validate_env_key,
    write_dotenv_file,
)

logger = logging.getLogger(__name__)


class EnvService:
    """Environment variable service"""
    
    def __init__(self, env_file: str = ".env"):
        """
        Initialise the environment variable service

        Args:
            env_file: path of the .env file (relative to the workspace root)
        """
        self.env_file = env_file
        self.env_vars: Dict[str, str] = {}
        self._load_env_file()
    
    def load_env_file(self, path: Optional[str] = None) -> Dict[str, str]:
        """
        Load the .env file

        Args:
            path: path of the .env file (the path given at initialisation is used when None)

        Returns:
            Dict[str, str]: dictionary of the loaded environment variables
        """
        if path:
            self.env_file = path
        
        return self._load_env_file()
    
    def save_env_file(self, path: Optional[str] = None, env_vars: Optional[Dict[str, str]] = None) -> bool:
        """
        Save the .env file

        Args:
            path: path of the .env file (the current path is used when None)
            env_vars: the environment variables to save (the current environment variables are used when None)

        Returns:
            bool: whether the save succeeded
        """
        if path:
            self.env_file = path
        
        if env_vars is None:
            env_vars = self.env_vars
        
        try:
            write_dotenv_file(self.env_file, env_vars)
            
            logger.info(f"Saved {len(env_vars)} environment variable(s) to {self.env_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save .env file: {e}")
            return False
    
    def reload_env(self) -> bool:
        """
        Reload the environment variables

        Returns:
            bool: whether the reload succeeded
        """
        try:
            self._load_env_file()
            logger.info("Environment variables reloaded")
            return True
        except Exception as e:
            logger.error(f"Failed to reload environment variables: {e}")
            return False
    
    def get_env_vars(self, show_values: bool = False) -> Dict[str, str]:
        """
        Get the environment variables

        Args:
            show_values: whether the real values are shown (sensitive information is hidden when False)

        Returns:
            Dict[str, str]: dictionary of the environment variables
        """
        if show_values:
            return self.env_vars.copy()
        else:
            # Hide sensitive values
            return {
                key: self._mask_value(value)
                for key, value in self.env_vars.items()
            }
    
    def get_env_var(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a single environment variable

        Args:
            key: name of the environment variable
            default: the default value

        Returns:
            Optional[str]: value of the environment variable
        """
        return self.env_vars.get(key, default)
    
    def update_env_var(self, key: str, value: str) -> bool:
        """
        Update a single environment variable

        Args:
            key: name of the environment variable
            value: value of the environment variable

        Returns:
            bool: whether the update succeeded
        """
        try:
            key = validate_env_key(key)
            value = normalize_env_value(value)

            # Update the value in memory
            self.env_vars[key] = value
            
            # Update the system environment variable as well
            os.environ[key] = value
            
            # Save to the file
            success = self.save_env_file()
            
            if success:
                logger.info(f"Updated environment variable: {key}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to update environment variable {key}: {e}")
            return False
    
    def delete_env_var(self, key: str) -> bool:
        """
        Delete an environment variable

        Args:
            key: name of the environment variable

        Returns:
            bool: whether the deletion succeeded
        """
        try:
            if key in self.env_vars:
                del self.env_vars[key]
                
                # Remove it from the system environment variables as well
                if key in os.environ:
                    del os.environ[key]
                
                # Save to the file
                success = self.save_env_file()
                
                if success:
                    logger.info(f"Deleted environment variable: {key}")
                
                return success
            else:
                logger.warning(f"Environment variable {key} does not exist")
                return False
        except Exception as e:
            logger.error(f"Failed to delete environment variable {key}: {e}")
            return False
    
    def _load_env_file(self) -> Dict[str, str]:
        """
        Load the environment variables from the .env file

        Returns:
            Dict[str, str]: dictionary of the loaded environment variables
        """
        self.env_vars = {}
        
        try:
            env_path = Path(self.env_file)
            
            # Check whether the file exists
            if not env_path.exists():
                logger.warning(f".env file not found at {self.env_file}")
                return self.env_vars
            
            self.env_vars = read_dotenv_file(env_path)
            for key, value in self.env_vars.items():
                os.environ[key] = value
            
            logger.info(f"Loaded {len(self.env_vars)} environment variable(s) from {self.env_file}")
            return self.env_vars
        
        except Exception as e:
            logger.error(f"Failed to load .env file: {e}")
            return self.env_vars
    
    def _mask_value(self, value: str) -> str:
        """
        Hide a sensitive value

        Args:
            value: the original value

        Returns:
            str: the hidden value
        """
        if len(value) <= 4:
            return '*' * len(value)
        else:
            # Show the first 2 and the last 2 characters
            return value[:2] + '*' * (len(value) - 4) + value[-2:]
