"""
Translation flow integration module

Integrates permission checks, quota management, history and logging into the translation flow.

Requirements: 1.2, 3.1, 27.2, 31.2
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TranslationIntegrationService:
    """
    Translation flow integration service

    Coordinates the services when a translation starts, progresses and completes:
    - permission checks
    - quota checks and updates
    - saving the history
    - logging

    Requirements: 1.2, 3.1, 27.2, 31.2
    """
    
    def __init__(
        self,
        permission_service=None,
        quota_service=None,
        history_service=None,
        log_service=None
    ):
        """
        Initialise the translation integration service

        Args:
            permission_service: the permission service
            quota_service: the quota service
            history_service: the history service
            log_service: the log service
        """
        self.permission_service = permission_service
        self.quota_service = quota_service
        self.history_service = history_service
        self.log_service = log_service
        
        logger.info("TranslationIntegrationService initialized")
    
    def check_translation_permission(
        self,
        username: str,
        translator: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check the translation permission

        Args:
            username: the user name
            translator: name of the translator

        Returns:
            Tuple[bool, Optional[str]]: (whether it is allowed, error message)

        Requirements: 1.2
        """
        if not self.permission_service:
            logger.warning("Permission service not available, allowing by default")
            return True, None
        
        try:
            has_permission = self.permission_service.check_translator_permission(
                username, translator
            )
            
            if not has_permission:
                permissions = self.permission_service.get_user_permissions(username)
                allowed = permissions.allowed_translators if permissions else []
                error_msg = f"您没有权限使用翻译器 '{translator}'。允许的翻译器: {allowed}"
                logger.warning(f"Permission denied for user {username}: translator '{translator}' is not allowed. Allowed translators: {allowed}")
                return False, error_msg
            
            logger.debug(f"Permission check passed for user {username}, translator {translator}")
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False, f"权限检查失败: {str(e)}"
    
    def check_quota_before_translation(
        self,
        username: str,
        image_count: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """
        Check the quota before translating

        Args:
            username: the user name
            image_count: number of images to translate

        Returns:
            Tuple[bool, Optional[str]]: (whether it is allowed, error message)

        Requirements: 27.2
        """
        if not self.quota_service:
            logger.warning("Quota service not available, allowing by default")
            return True, None
        
        try:
            # Check the daily quota
            allowed, error_msg = self.quota_service.check_daily_quota(username, image_count)
            
            if not allowed:
                return False, error_msg
            
            logger.debug(f"Quota check passed for user {username}, count {image_count}")
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking quota: {e}")
            return False, f"配额检查失败: {str(e)}"
    
    def on_translation_start(
        self,
        session_token: str,
        username: str,
        translator: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Handling when a translation starts

        Args:
            session_token: the session token
            username: the user name
            translator: name of the translator
            config: the translation configuration

        Returns:
            bool: whether it succeeded

        Requirements: 31.2
        """
        try:
            # Log the start of the translation
            if self.log_service:
                self.log_service.log_translation_event(
                    session_token=session_token,
                    user_id=username,
                    event_type='translation_start',
                    message=f'Starting translation with translator: {translator}',
                    level='info',
                    details={
                        'translator': translator,
                        'config': self._sanitize_config(config) if config else None,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                )
            
            logger.info(f"Translation started: user={username}, session={session_token}, translator={translator}")
            return True
            
        except Exception as e:
            logger.error(f"Error in on_translation_start: {e}")
            return False
    
    def on_translation_progress(
        self,
        session_token: str,
        username: str,
        progress: float,
        message: str = ""
    ) -> bool:
        """
        Handling while a translation is in progress

        Args:
            session_token: the session token
            username: the user name
            progress: the progress (0-100)
            message: the progress message

        Returns:
            bool: whether it succeeded

        Requirements: 31.2
        """
        try:
            # Log the progress of the translation
            if self.log_service:
                self.log_service.log_translation_event(
                    session_token=session_token,
                    user_id=username,
                    event_type='translation_progress',
                    message=message or f'Translation progress: {progress:.1f}%',
                    level='info',
                    details={
                        'progress': progress,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                )
            
            logger.debug(f"Translation progress: user={username}, session={session_token}, progress={progress}%")
            return True
            
        except Exception as e:
            logger.error(f"Error in on_translation_progress: {e}")
            return False
    
    def on_translation_complete(
        self,
        session_token: str,
        username: str,
        result_files: list,
        image_count: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Handling when a translation completes

        Args:
            session_token: the session token
            username: the user name
            result_files: list of result files
            image_count: number of images translated successfully
            metadata: the metadata

        Returns:
            bool: whether it succeeded

        Requirements: 3.1, 27.2, 31.2
        """
        try:
            # 1. Update the quota (on success only)
            if self.quota_service and image_count > 0:
                self.quota_service.increment_quota_usage(username, image_count)
                logger.info(f"Quota updated for user {username}: +{image_count}")
            
            # 2. Save the translation result to the history
            if self.history_service and result_files:
                try:
                    self.history_service.save_translation_result(
                        user_id=username,
                        session_token=session_token,
                        files=result_files,
                        metadata=metadata
                    )
                    logger.info(f"Translation result saved: session={session_token}, files={len(result_files)}")
                except Exception as e:
                    logger.error(f"Failed to save translation result: {e}")
            
            # 3. Log the completion of the translation
            if self.log_service:
                self.log_service.log_translation_event(
                    session_token=session_token,
                    user_id=username,
                    event_type='translation_complete',
                    message=f'Translation complete: {image_count} images',
                    level='info',
                    details={
                        'image_count': image_count,
                        'file_count': len(result_files) if result_files else 0,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                )
            
            logger.info(f"Translation completed: user={username}, session={session_token}, images={image_count}")
            return True
            
        except Exception as e:
            logger.error(f"Error in on_translation_complete: {e}")
            return False
    
    def on_translation_error(
        self,
        session_token: str,
        username: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Handling when a translation fails

        Args:
            session_token: the session token
            username: the user name
            error_message: the error message
            error_details: details of the error

        Returns:
            bool: whether it succeeded

        Requirements: 31.2
        """
        try:
            # Log the translation error
            if self.log_service:
                self.log_service.log_translation_event(
                    session_token=session_token,
                    user_id=username,
                    event_type='translation_error',
                    message=error_message,
                    level='error',
                    details={
                        'error': error_message,
                        'details': error_details,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                )
            
            logger.error(f"Translation error: user={username}, session={session_token}, error={error_message}")
            return True
            
        except Exception as e:
            logger.error(f"Error in on_translation_error: {e}")
            return False
    
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a configuration by removing sensitive information

        Args:
            config: the original configuration

        Returns:
            Dict[str, Any]: the cleaned configuration
        """
        if not config:
            return {}
        
        # Copy the configuration
        sanitized = dict(config)
        
        # Remove the sensitive fields
        sensitive_keys = ['api_key', 'api_secret', 'password', 'token', 'key']
        
        def remove_sensitive(d):
            if isinstance(d, dict):
                return {
                    k: '***' if any(s in k.lower() for s in sensitive_keys) else remove_sensitive(v)
                    for k, v in d.items()
                }
            elif isinstance(d, list):
                return [remove_sensitive(item) for item in d]
            return d
        
        return remove_sensitive(sanitized)


# Global instance
_integration_service: Optional[TranslationIntegrationService] = None


def init_translation_integration(
    permission_service=None,
    quota_service=None,
    history_service=None,
    log_service=None
) -> TranslationIntegrationService:
    """
    Initialise the translation integration service

    Args:
        permission_service: the permission service
        quota_service: the quota service
        history_service: the history service
        log_service: the log service

    Returns:
        TranslationIntegrationService: the integration service instance
    """
    global _integration_service
    _integration_service = TranslationIntegrationService(
        permission_service=permission_service,
        quota_service=quota_service,
        history_service=history_service,
        log_service=log_service
    )
    logger.info("Translation integration service initialized")
    return _integration_service


def get_translation_integration() -> Optional[TranslationIntegrationService]:
    """
    Get the translation integration service instance

    Returns:
        Optional[TranslationIntegrationService]: the integration service instance
    """
    return _integration_service
