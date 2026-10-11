"""
Repository for translation history management.
Optimisation: storage is split per user, for better performance with many users.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from manga_translator.server.core.download_ticket_service import resolve_path_within
from manga_translator.server.models import TranslationResult


class TranslationRepository:
    """
    Repository for managing translation history.
    Storage is split per user, with a separate JSON file for each user.
    An index file is kept as well, for looking up a session_token quickly.
    """
    
    def __init__(self, base_path: str):
        """
        Initialise the repository.

        Args:
            base_path: the original single-file path, which is converted to a folder path
        """
        # Turn the former file path into a folder
        self.base_dir = Path(base_path).parent / 'history'
        self.index_file = self.base_dir / '_index.json'
        self._lock = threading.RLock()
        self._ensure_dirs()
        self._migrate_old_data(base_path)
    
    def _ensure_dirs(self) -> None:
        """Make sure the folders exist"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _migrate_old_data(self, old_file: str) -> None:
        """Migrate the old single-file data to the new split structure"""
        old_path = Path(old_file)
        if not old_path.exists():
            return
        
        try:
            with open(old_path, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            
            sessions = old_data.get('sessions', [])
            if not sessions:
                return
            
            # Migrate grouped by user
            for session in sessions:
                user_id = session.get('user_id', 'unknown')
                self._add_to_user_file(user_id, session)
            
            # Back up and delete the old file
            backup_path = old_path.with_suffix('.json.migrated')
            old_path.rename(backup_path)
            
        except Exception as e:
            print(f"Migration warning: {e}")
    
    def _get_user_file(self, user_id: str) -> Path:
        """Get the path of the history file of a user"""
        # Use a safe file name
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in user_id)
        return resolve_path_within(self.base_dir, self.base_dir / f'{safe_name}.json')
    
    def _read_user_data(self, user_id: str) -> Dict[str, Any]:
        """Read the data of a user"""
        user_file = self._get_user_file(user_id)
        with self._lock:
            if not user_file.exists():
                return {'sessions': [], 'last_updated': None}
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {'sessions': [], 'last_updated': None}
    
    def _write_user_data(self, user_id: str, data: Dict[str, Any]) -> None:
        """Write the data of a user"""
        user_file = self._get_user_file(user_id)
        data['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        with self._lock:
            temp_path = user_file.with_suffix('.tmp')
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(temp_path, user_file)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise
    
    def _add_to_user_file(self, user_id: str, session: dict) -> None:
        """Add a session to the file of a user"""
        data = self._read_user_data(user_id)
        data['sessions'].append(session)
        self._write_user_data(user_id, data)
        self._update_index(session['session_token'], user_id)
    
    def _read_index(self) -> Dict[str, str]:
        """Read the index file (session_token -> user_id)"""
        with self._lock:
            if not self.index_file.exists():
                return {}
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
    
    def _write_index(self, index: Dict[str, str]) -> None:
        """Write the index file"""
        with self._lock:
            temp_path = self.index_file.with_suffix('.tmp')
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(index, f, ensure_ascii=False)
                os.replace(temp_path, self.index_file)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise
    
    def _update_index(self, session_token: str, user_id: str) -> None:
        """Update the index"""
        index = self._read_index()
        index[session_token] = user_id
        self._write_index(index)
    
    def _remove_from_index(self, session_token: str) -> None:
        """Remove from the index"""
        index = self._read_index()
        if session_token in index:
            del index[session_token]
            self._write_index(index)
    
    def add_session(self, result: TranslationResult) -> None:
        """Add a translation session to the history"""
        self._add_to_user_file(result.user_id, result.to_dict())
    
    def get_user_sessions(self, user_id: str) -> List[dict]:
        """Get all sessions of the given user"""
        data = self._read_user_data(user_id)
        return data.get('sessions', [])
    
    def get_session_by_token(self, session_token: str) -> Optional[dict]:
        """Get a session by token"""
        # Look in the index first
        index = self._read_index()
        user_id = index.get(session_token)
        
        if user_id:
            data = self._read_user_data(user_id)
            for session in data.get('sessions', []):
                if session.get('session_token') == session_token:
                    return session
        
        # Not in the index: go through all user files (for old data)
        for user_file in self.base_dir.glob('*.json'):
            if user_file.name.startswith('_'):
                continue
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for session in data.get('sessions', []):
                    if session.get('session_token') == session_token:
                        # Update the index
                        self._update_index(session_token, session.get('user_id', 'unknown'))
                        return session
            except Exception:
                continue
        
        return None
    
    def get_all_sessions(self) -> List[dict]:
        """Get all sessions (for administrators)"""
        all_sessions = []
        for user_file in self.base_dir.glob('*.json'):
            if user_file.name.startswith('_'):
                continue
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_sessions.extend(data.get('sessions', []))
            except Exception:
                continue
        return all_sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        # Go through all user files to find and delete it
        for user_file in self.base_dir.glob('*.json'):
            if user_file.name.startswith('_'):
                continue
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                original_len = len(data.get('sessions', []))
                sessions = [s for s in data.get('sessions', []) if s.get('id') != session_id]
                
                if len(sessions) < original_len:
                    # Found and deleted
                    deleted_session = next(
                        (s for s in data.get('sessions', []) if s.get('id') == session_id), 
                        None
                    )
                    data['sessions'] = sessions
                    user_id = user_file.stem
                    self._write_user_data(user_id, data)
                    
                    if deleted_session:
                        self._remove_from_index(deleted_session.get('session_token', ''))
                    return True
            except Exception:
                continue
        return False
    
    def update_session(self, session_id: str, updates: dict) -> bool:
        """Update a session"""
        for user_file in self.base_dir.glob('*.json'):
            if user_file.name.startswith('_'):
                continue
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for session in data.get('sessions', []):
                    if session.get('id') == session_id:
                        session.update(updates)
                        user_id = user_file.stem
                        self._write_user_data(user_id, data)
                        return True
            except Exception:
                continue
        return False
    
    def search_sessions(self, user_id: Optional[str] = None, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> List[dict]:
        """Search the sessions"""
        def filter_func(session):
            if start_date and session.get('timestamp', '') < start_date:
                return False
            if end_date and session.get('timestamp', '') > end_date:
                return False
            return True
        
        if user_id:
            # Query the given user only
            sessions = self.get_user_sessions(user_id)
            return [s for s in sessions if filter_func(s)]
        else:
            # Query all users
            all_sessions = self.get_all_sessions()
            return [s for s in all_sessions if filter_func(s)]
