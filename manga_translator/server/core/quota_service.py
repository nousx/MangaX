"""
Quota management service (QuotaManagementService)

Manages the upload limits, the limit on the number of sessions and the daily translation quota of users.
Supports checking, counting and resetting quotas.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

from manga_translator.server.core.group_service import GroupService
from manga_translator.server.models.quota_models import QuotaLimit, QuotaStats
from manga_translator.server.repositories.permission_repository import (
    PermissionRepository,
)
from manga_translator.server.repositories.quota_repository import QuotaRepository

logger = logging.getLogger(__name__)


class QuotaManagementService:
    """Quota management service"""
    
    # Default quota limits
    DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    DEFAULT_MAX_FILES_PER_UPLOAD = 10
    DEFAULT_MAX_SESSIONS = 5
    DEFAULT_DAILY_QUOTA = -1  # -1 means unlimited
    
    def __init__(
        self,
        quota_repo: QuotaRepository,
        permission_repo: PermissionRepository,
        group_service: GroupService,
        data_path: str = "manga_translator/server/data"
    ):
        """
        Initialise the quota management service

        Args:
            quota_repo: the quota data repository
            permission_repo: the permission data repository
            group_service: the user group service
            data_path: the data storage path
        """
        self.quota_repo = quota_repo
        self.permission_repo = permission_repo
        self.group_service = group_service
        self.data_path = Path(data_path)
        
        # Tracking of active sessions (in memory)
        self._active_sessions: Dict[str, List[str]] = {}  # user_id -> [session_tokens]
        
        logger.info("QuotaManagementService initialized")
    
    def _get_user_quota_limit(self, user_id: str) -> QuotaLimit:
        """
        Get the quota limits of a user (with inheritance)

        Priority: user level > user group level > global default

        Args:
            user_id: the user ID

        Returns:
            QuotaLimit: the quota limits of the user
        """
        # 1. Try the user-level quota
        user_quota_data = self.quota_repo.get_user_quota(user_id)
        if user_quota_data:
            return QuotaLimit.from_dict(user_quota_data)
        
        # 2. Try the quota of the user group
        user_info = self.permission_repo.get_user_permissions(user_id)
        if user_info and 'group' in user_info:
            group_name = user_info['group']
            group_config = self.group_service.get_group_config(group_name)
            
            if group_config and 'quota_limits' in group_config:
                limits = group_config['quota_limits']
                quota = QuotaLimit(
                    user_id=user_id,
                    max_file_size=limits.get('max_file_size', self.DEFAULT_MAX_FILE_SIZE),
                    max_files_per_upload=limits.get('max_files_per_upload', self.DEFAULT_MAX_FILES_PER_UPLOAD),
                    max_sessions=limits.get('max_sessions', self.DEFAULT_MAX_SESSIONS),
                    daily_quota=limits.get('daily_quota', self.DEFAULT_DAILY_QUOTA),
                    current_usage=0,
                    last_reset=datetime.now(UTC).isoformat()
                )
                # Save at user level for faster access later
                self.quota_repo.set_user_quota(user_id, quota)
                return quota
        
        # 3. Use the global defaults
        quota = QuotaLimit(
            user_id=user_id,
            max_file_size=self.DEFAULT_MAX_FILE_SIZE,
            max_files_per_upload=self.DEFAULT_MAX_FILES_PER_UPLOAD,
            max_sessions=self.DEFAULT_MAX_SESSIONS,
            daily_quota=self.DEFAULT_DAILY_QUOTA,
            current_usage=0,
            last_reset=datetime.now(UTC).isoformat()
        )
        # Save at user level
        self.quota_repo.set_user_quota(user_id, quota)
        return quota
    
    def check_upload_limit(self, user_id: str, file_size: int, file_count: int) -> tuple[bool, Optional[str]]:
        """
        Check the upload limits

        Args:
            user_id: the user ID
            file_size: size of a single file (bytes)
            file_count: number of files

        Returns:
            tuple[bool, Optional[str]]: (whether it is allowed, error message)
        """
        try:
            quota = self._get_user_quota_limit(user_id)
            
            # Check the file size limit
            if file_size > quota.max_file_size:
                max_mb = quota.max_file_size / (1024 * 1024)
                current_mb = file_size / (1024 * 1024)
                return False, f"File size {current_mb:.2f}MB exceeds the limit of {max_mb:.2f}MB"
            
            # Check the file count limit
            if file_count > quota.max_files_per_upload:
                return False, f"File count {file_count} exceeds the limit of {quota.max_files_per_upload}"
            
            logger.info(f"Upload limit check passed for user {user_id}: {file_count} files, {file_size} bytes")
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking upload limit for user {user_id}: {e}")
            return False, f"Error while checking the upload limits: {str(e)}"
    
    def check_session_limit(self, user_id: str) -> tuple[bool, Optional[str]]:
        """
        Check the limit on the number of sessions

        Args:
            user_id: the user ID

        Returns:
            tuple[bool, Optional[str]]: (whether it is allowed, error message)
        """
        try:
            quota = self._get_user_quota_limit(user_id)
            
            # Get the current number of active sessions
            active_count = len(self._active_sessions.get(user_id, []))
            
            # Check whether the limit is exceeded
            if active_count >= quota.max_sessions:
                return False, f"Number of active sessions {active_count} has reached the limit of {quota.max_sessions}"
            
            logger.info(f"Session limit check passed for user {user_id}: {active_count}/{quota.max_sessions}")
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking session limit for user {user_id}: {e}")
            return False, f"Error while checking the session limit: {str(e)}"
    
    def check_daily_quota(self, user_id: str, image_count: int = 1) -> tuple[bool, Optional[str]]:
        """
        Check the daily translation quota

        Args:
            user_id: the user ID
            image_count: number of images to translate

        Returns:
            tuple[bool, Optional[str]]: (whether it is allowed, error message)
        """
        try:
            quota = self._get_user_quota_limit(user_id)
            
            # -1 means an unlimited quota
            if quota.daily_quota == -1:
                logger.info(f"Daily quota check passed for user {user_id}: unlimited quota")
                return True, None
            
            # Check whether the quota has to be reset
            self._check_and_reset_daily_quota(user_id, quota)
            
            # Get the quota again (it may have been reset)
            quota = self._get_user_quota_limit(user_id)
            
            # Check the remaining quota
            remaining = quota.daily_quota - quota.current_usage
            if remaining < image_count:
                logger.warning(f"Daily quota exceeded for user {user_id}: remaining {remaining}, requested {image_count}")
                return False, f"Daily quota insufficient: remaining {remaining}, needed {image_count}"
            
            logger.info(f"Daily quota check passed for user {user_id}: {quota.current_usage}/{quota.daily_quota}")
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking daily quota for user {user_id}: {e}")
            return False, f"Error while checking the daily quota: {str(e)}"
    
    def _check_and_reset_daily_quota(self, user_id: str, quota: QuotaLimit) -> None:
        """
        Check the daily quota and reset it if needed

        Args:
            user_id: the user ID
            quota: the current quota
        """
        if not quota.last_reset:
            # When it was never reset, reset it now
            self.reset_daily_quota(user_id)
            return
        
        try:
            last_reset = datetime.fromisoformat(quota.last_reset)
            now = datetime.now(UTC)
            
            # When the last reset was on a different date, reset
            if last_reset.date() < now.date():
                logger.info(f"Resetting daily quota for user {user_id} (last reset: {last_reset.date()})")
                self.reset_daily_quota(user_id)
        except Exception as e:
            logger.error(f"Error parsing last_reset date for user {user_id}: {e}")
            # When parsing fails, reset the quota
            self.reset_daily_quota(user_id)
    
    def increment_quota_usage(self, user_id: str, image_count: int) -> bool:
        """
        Increase the quota usage (called only after a successful translation)

        Args:
            user_id: the user ID
            image_count: number of images translated successfully

        Returns:
            bool: whether it succeeded
        """
        try:
            success = self.quota_repo.increment_usage(user_id, image_count)
            if success:
                logger.info(f"Incremented quota usage for user {user_id} by {image_count}")
            else:
                logger.warning(f"Failed to increment quota usage for user {user_id}")
            return success
        except Exception as e:
            logger.error(f"Error incrementing quota usage for user {user_id}: {e}")
            return False
    
    def reset_daily_quota(self, user_id: Optional[str] = None) -> bool:
        """
        Reset the daily quota

        Args:
            user_id: the user ID; all users are reset when None

        Returns:
            bool: whether it succeeded
        """
        try:
            if user_id:
                # Reset a single user
                success = self.quota_repo.reset_daily_usage(user_id)
                if success:
                    logger.info(f"Reset daily quota for user {user_id}")
                return success
            else:
                # Reset all users
                all_quotas = self.quota_repo.get_all_quotas()
                for uid in all_quotas.keys():
                    self.quota_repo.reset_daily_usage(uid)
                logger.info(f"Reset daily quota for all users ({len(all_quotas)} users)")
                return True
        except Exception as e:
            logger.error(f"Error resetting daily quota: {e}")
            return False
    
    def get_quota_stats(self, user_id: str) -> Optional[QuotaStats]:
        """
        Get the quota statistics of a user

        Args:
            user_id: the user ID

        Returns:
            QuotaStats: the quota statistics
        """
        try:
            quota = self._get_user_quota_limit(user_id)
            
            # Work out the remaining quota
            remaining = -1 if quota.daily_quota == -1 else (quota.daily_quota - quota.current_usage)
            
            # Get the number of active sessions
            active_sessions = len(self._active_sessions.get(user_id, []))
            
            stats = QuotaStats(
                user_id=user_id,
                daily_limit=quota.daily_quota,
                used_today=quota.current_usage,
                remaining=remaining,
                active_sessions=active_sessions,
                total_uploaded=quota.current_usage  # A simplified implementation; it may need tracking separately in practice
            )
            
            logger.debug(f"Retrieved quota stats for user {user_id}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting quota stats for user {user_id}: {e}")
            return None
    
    def get_all_quota_stats(self) -> Dict[str, QuotaStats]:
        """
        Get the quota statistics of all users (administrator function)

        Returns:
            Dict[str, QuotaStats]: mapping from user ID to quota statistics
        """
        try:
            all_quotas = self.quota_repo.get_all_quotas()
            stats_dict = {}
            
            for user_id in all_quotas.keys():
                stats = self.get_quota_stats(user_id)
                if stats:
                    stats_dict[user_id] = stats
            
            logger.info(f"Retrieved quota stats for {len(stats_dict)} users")
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error getting all quota stats: {e}")
            return {}
    
    def register_session(self, user_id: str, session_token: str) -> bool:
        """
        Register an active session

        Args:
            user_id: the user ID
            session_token: the session token

        Returns:
            bool: whether it succeeded
        """
        try:
            if user_id not in self._active_sessions:
                self._active_sessions[user_id] = []
            
            if session_token not in self._active_sessions[user_id]:
                self._active_sessions[user_id].append(session_token)
                logger.info(f"Registered session {session_token} for user {user_id}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error registering session for user {user_id}: {e}")
            return False
    
    def unregister_session(self, user_id: str, session_token: str) -> bool:
        """
        Unregister an active session

        Args:
            user_id: the user ID
            session_token: the session token

        Returns:
            bool: whether it succeeded
        """
        try:
            if user_id in self._active_sessions:
                if session_token in self._active_sessions[user_id]:
                    self._active_sessions[user_id].remove(session_token)
                    logger.info(f"Unregistered session {session_token} for user {user_id}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error unregistering session for user {user_id}: {e}")
            return False
    
    def get_active_sessions(self, user_id: str) -> List[str]:
        """
        Get the list of active sessions of a user

        Args:
            user_id: the user ID

        Returns:
            List[str]: list of session tokens
        """
        return self._active_sessions.get(user_id, []).copy()
    
    def set_user_quota_limits(
        self,
        user_id: str,
        max_file_size: Optional[int] = None,
        max_files_per_upload: Optional[int] = None,
        max_sessions: Optional[int] = None,
        daily_quota: Optional[int] = None
    ) -> bool:
        """
        Set the quota limits of a user (administrator function)

        Args:
            user_id: the user ID
            max_file_size: maximum file size
            max_files_per_upload: maximum number of files per upload
            max_sessions: maximum number of sessions
            daily_quota: the daily quota

        Returns:
            bool: whether it succeeded
        """
        try:
            # Get the existing quota or create a new one
            quota = self._get_user_quota_limit(user_id)
            
            # Update the given limits
            if max_file_size is not None:
                quota.max_file_size = max_file_size
            if max_files_per_upload is not None:
                quota.max_files_per_upload = max_files_per_upload
            if max_sessions is not None:
                quota.max_sessions = max_sessions
            if daily_quota is not None:
                quota.daily_quota = daily_quota
            
            # Save the updated quota
            self.quota_repo.set_user_quota(user_id, quota)
            logger.info(f"Updated quota limits for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting quota limits for user {user_id}: {e}")
            return False
