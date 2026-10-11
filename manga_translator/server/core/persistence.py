"""
Persistent storage utilities

Provides atomic writes, backups and data loading.
"""

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def atomic_write_json(file_path: str, data: Dict[str, Any], create_backup: bool = True) -> bool:
    """
    Write a JSON file atomically

    A temporary file and a rename make the write atomic, so an interruption cannot corrupt the data.

    Args:
        file_path: path of the target file
        data: the data to write
        create_backup: whether a backup is made before writing

    Returns:
        bool: whether the write succeeded
    """
    try:
        file_path = Path(file_path)
        
        # Make sure the folder exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # When the file exists and a backup is wanted, create the backup
        if create_backup and file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            try:
                shutil.copy2(file_path, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        
        # Write to a temporary file
        temp_fd, temp_path = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp"
        )
        
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Make sure the data reaches the disk
            
            # Rename the temporary file to the target file atomically
            # On Windows, an existing target file has to be deleted first
            if os.name == 'nt' and file_path.exists():
                file_path.unlink()
            
            os.rename(temp_path, file_path)
            logger.debug(f"Successfully wrote to {file_path}")
            return True
            
        except Exception as e:
            # Remove the temporary file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            raise e
            
    except Exception as e:
        logger.error(f"Failed to write {file_path}: {e}")
        return False


def load_json(file_path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a JSON file

    When the file does not exist or is damaged, a restore from the backup is tried.

    Args:
        file_path: the file path
        default: the default value returned when the file does not exist

    Returns:
        Dict[str, Any]: the loaded data
    """
    file_path = Path(file_path)
    
    # Try to load the main file
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Successfully loaded {file_path}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            
            # Try to restore from the backup
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            if backup_path.exists():
                logger.info(f"Attempting to restore from backup: {backup_path}")
                try:
                    with open(backup_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        logger.info("Successfully restored from backup")
                        # Restore the main file
                        atomic_write_json(str(file_path), data, create_backup=False)
                        return data
                except Exception as backup_error:
                    logger.error(f"Failed to restore from backup: {backup_error}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    # Return the default value
    if default is not None:
        logger.info(f"Using default value for {file_path}")
        return default
    
    logger.warning(f"File {file_path} not found and no default provided")
    return {}


def create_backup(file_path: str, backup_dir: Optional[str] = None) -> Optional[str]:
    """
    Create a timestamped backup of a file

    Args:
        file_path: path of the file to back up
        backup_dir: the backup folder (the folder of the file is used when None)

    Returns:
        Optional[str]: path of the backup file, or None on failure
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist, cannot create backup")
            return None
        
        # Decide the backup folder
        if backup_dir:
            backup_dir_path = Path(backup_dir)
            backup_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            backup_dir_path = file_path.parent
        
        # Build a backup file name with a timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir_path / backup_name
        
        # Copy the file
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return str(backup_path)
        
    except Exception as e:
        logger.error(f"Failed to create backup of {file_path}: {e}")
        return None


def cleanup_old_backups(backup_dir: str, pattern: str, keep_count: int = 5) -> int:
    """
    Remove old backup files, keeping only the newest few

    Args:
        backup_dir: the backup folder
        pattern: the file name pattern (glob format)
        keep_count: number of backups kept

    Returns:
        int: number of files deleted
    """
    try:
        backup_dir_path = Path(backup_dir)
        
        if not backup_dir_path.exists():
            return 0
        
        # Get all matching backup files
        backup_files = sorted(
            backup_dir_path.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Delete the surplus backups
        deleted_count = 0
        for backup_file in backup_files[keep_count:]:
            try:
                backup_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted old backup: {backup_file}")
            except Exception as e:
                logger.warning(f"Failed to delete {backup_file}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old backup(s)")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Failed to cleanup backups: {e}")
        return 0


def ensure_directory(dir_path: str) -> bool:
    """
    Make sure a folder exists

    Args:
        dir_path: the folder path

    Returns:
        bool: whether it succeeded
    """
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        return False
