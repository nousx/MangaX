"""
Session management service (SessionService)

Manages user login sessions and the generation and verification of tokens.
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import uuid4

from manga_translator.server.core.models import Session
from manga_translator.server.core.persistence import atomic_write_json, load_json

logger = logging.getLogger(__name__)


class SessionService:
    """Session management service"""
    
    def __init__(
        self,
        sessions_file: Optional[str] = None,
        session_timeout_minutes: int = 60,
        enable_persistence: bool = False
    ):
        """
        Initialise the session management service

        Args:
            sessions_file: path of the session storage file (optional, for persistence)
            session_timeout_minutes: the session timeout (minutes)
            enable_persistence: whether session persistence is enabled
        """
        self.sessions_file = sessions_file
        self.session_timeout_minutes = session_timeout_minutes
        self.enable_persistence = enable_persistence
        
        # In-memory session storage: token -> Session
        self.sessions_by_token: Dict[str, Session] = {}
        # Index by session ID: session_id -> Session
        self.sessions_by_id: Dict[str, Session] = {}
        # Index by user name: username -> List[Session]
        self.sessions_by_username: Dict[str, List[Session]] = {}
        
        # When persistence is on, load the sessions
        if self.enable_persistence and self.sessions_file:
            self._load_sessions()
    
    def create_session(
        self,
        username: str,
        role: str,
        ip_address: str,
        user_agent: str
    ) -> Session:
        """
        Create a new session

        Args:
            username: the user name
            role: the user role
            ip_address: the IP address
            user_agent: the user agent string

        Returns:
            Session: the session object that was created
        """
        # Generate a unique session ID and token
        session_id = str(uuid4())
        token = secrets.token_urlsafe(32)
        
        # Create the session object
        from datetime import timezone
        now = datetime.now(timezone.utc)
        session = Session(
            session_id=session_id,
            username=username,
            role=role,
            token=token,
            created_at=now,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
            is_active=True
        )
        
        # Keep in the in-memory indexes
        self.sessions_by_token[token] = session
        self.sessions_by_id[session_id] = session
        
        if username not in self.sessions_by_username:
            self.sessions_by_username[username] = []
        self.sessions_by_username[username].append(session)
        
        # Persist (when enabled)
        if self.enable_persistence:
            self._save_sessions()
        
        logger.info(f"Created session for user: {username} (session_id: {session_id})")
        return session
    
    def get_session(self, token: str) -> Optional[Session]:
        """
        Get a session by token

        Args:
            token: the session token

        Returns:
            Optional[Session]: the session object, or None when it does not exist or has expired
        """
        session = self.sessions_by_token.get(token)
        
        if not session:
            return None
        
        # Check whether the session has expired
        if self._is_session_expired(session):
            logger.debug(f"Session expired: {session.session_id}")
            self._deactivate_session(session)
            return None
        
        return session
    
    def get_session_by_id(self, session_id: str) -> Optional[Session]:
        """
        Get a session by session ID

        Args:
            session_id: the session ID

        Returns:
            Optional[Session]: the session object, or None when it does not exist
        """
        return self.sessions_by_id.get(session_id)
    
    def list_sessions(self, username: Optional[str] = None) -> List[Session]:
        """
        List the active sessions

        Args:
            username: optional; only the sessions of the given user are listed

        Returns:
            List[Session]: list of active sessions
        """
        if username:
            # Return the active sessions of the given user
            user_sessions = self.sessions_by_username.get(username, [])
            return [s for s in user_sessions if s.is_active and not self._is_session_expired(s)]
        else:
            # Return all active sessions
            return [
                s for s in self.sessions_by_token.values()
                if s.is_active and not self._is_session_expired(s)
            ]
    
    def update_activity(self, token: str) -> bool:
        """
        Update the activity time of a session

        Args:
            token: the session token

        Returns:
            bool: whether the update succeeded
        """
        session = self.sessions_by_token.get(token)
        
        if not session or not session.is_active:
            return False
        
        # Check whether the session has expired
        if self._is_session_expired(session):
            self._deactivate_session(session)
            return False
        
        # Update the last activity time
        from datetime import timezone
        session.last_activity = datetime.now(timezone.utc)
        
        # Persist (when enabled)
        if self.enable_persistence:
            self._save_sessions()
        
        return True
    
    def terminate_session(self, session_id: str) -> bool:
        """
        Terminate the given session

        Args:
            session_id: the session ID

        Returns:
            bool: whether the termination succeeded
        """
        session = self.sessions_by_id.get(session_id)
        
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        # Mark as inactive
        self._deactivate_session(session)
        
        logger.info(f"Terminated session: {session_id} (user: {session.username})")
        return True
    
    def terminate_user_sessions(self, username: str) -> int:
        """
        Terminate all sessions of a user

        Args:
            username: the user name

        Returns:
            int: number of sessions terminated
        """
        user_sessions = self.sessions_by_username.get(username, [])
        terminated_count = 0
        
        for session in user_sessions:
            if session.is_active:
                self._deactivate_session(session)
                terminated_count += 1
        
        logger.info(f"Terminated {terminated_count} session(s) for user: {username}")
        return terminated_count
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions

        Returns:
            int: number of sessions removed
        """
        expired_sessions = []
        
        for session in self.sessions_by_token.values():
            if session.is_active and self._is_session_expired(session):
                expired_sessions.append(session)
        
        for session in expired_sessions:
            self._deactivate_session(session)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired session(s)")
        
        return len(expired_sessions)
    
    def verify_token(self, token: str) -> Optional[Session]:
        """
        Verify a token and return the session

        Args:
            token: the session token

        Returns:
            Optional[Session]: the valid session object, or None when the token is invalid
        """
        session = self.get_session(token)
        
        if session:
            # Update the activity time
            self.update_activity(token)
        
        return session
    
    def clear_all_sessions(self) -> int:
        """
        Clear all sessions (for a system restart)

        Returns:
            int: number of sessions cleared
        """
        count = len([s for s in self.sessions_by_token.values() if s.is_active])
        
        self.sessions_by_token.clear()
        self.sessions_by_id.clear()
        self.sessions_by_username.clear()
        
        # Persist (when enabled)
        if self.enable_persistence:
            self._save_sessions()
        
        logger.info(f"Cleared all sessions (count: {count})")
        return count
    
    def _is_session_expired(self, session: Session) -> bool:
        """
        Check whether a session has expired

        Args:
            session: the session object

        Returns:
            bool: whether the session has expired
        """
        if not session.is_active:
            return True
        
        timeout = timedelta(minutes=self.session_timeout_minutes)
        from datetime import timezone
        return datetime.now(timezone.utc) - session.last_activity > timeout
    
    def _deactivate_session(self, session: Session) -> None:
        """
        Deactivate a session

        Args:
            session: the session object
        """
        session.is_active = False
        
        # Remove from the token index
        if session.token in self.sessions_by_token:
            del self.sessions_by_token[session.token]
        
        # Persist (when enabled)
        if self.enable_persistence:
            self._save_sessions()
    
    def _load_sessions(self) -> None:
        """Load the sessions from persistent storage"""
        if not self.sessions_file:
            return
        
        try:
            data = load_json(self.sessions_file, default={'version': '1.0', 'sessions': []})
            
            sessions_data = data.get('sessions', [])
            
            for session_data in sessions_data:
                try:
                    session = Session.from_dict(session_data)
                    
                    # Only active sessions that have not expired are loaded
                    if session.is_active and not self._is_session_expired(session):
                        self.sessions_by_token[session.token] = session
                        self.sessions_by_id[session.session_id] = session
                        
                        if session.username not in self.sessions_by_username:
                            self.sessions_by_username[session.username] = []
                        self.sessions_by_username[session.username].append(session)
                except Exception as e:
                    logger.error(f"Failed to load session: {e}")
            
            active_count = len(self.sessions_by_token)
            logger.info(f"Loaded {active_count} active session(s)")
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
    
    def _save_sessions(self) -> None:
        """Save the sessions to persistent storage"""
        if not self.sessions_file:
            return
        
        try:
            # Only active sessions are saved
            active_sessions = [
                s for s in self.sessions_by_id.values()
                if s.is_active
            ]
            
            data = {
                'version': '1.0',
                'sessions': [session.to_dict() for session in active_sessions]
            }
            
            success = atomic_write_json(self.sessions_file, data, create_backup=False)
            if success:
                logger.debug(f"Saved {len(active_sessions)} session(s)")
            else:
                logger.error("Failed to save sessions")
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
