"""
Log Management Service

Responsible for the system logs and the session (dialog) logs, including:
- recording translation events
- querying and retrieving logs
- exporting logs
- pushing logs live

Requirements: 31.1-31.6, 32.1-32.8, 33.1-33.8
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from manga_translator.server.models import LogEntry
from manga_translator.server.repositories.log_repository import LogRepository


class LogManagementService:
    """Log management service class"""
    
    def __init__(self, log_repository: LogRepository):
        """
        Initialise the log management service

        Args:
            log_repository: the log data repository
        """
        self.log_repo = log_repository
    
    def log_translation_event(self, session_token: str, user_id: str,
                             event_type: str, message: str, level: str = 'info',
                             details: Optional[Dict[str, Any]] = None) -> LogEntry:
        """
        Record a translation event

        Args:
            session_token: the session token
            user_id: the user ID
            event_type: the event type (translation_start, translation_progress,
                       translation_complete, translation_error, etc.)
            message: the log message
            level: the log level (info, warning, error)
            details: dictionary of details

        Returns:
            The log entry that was created

        Requirements: 31.2, 33.2
        """
        # Validate the log level
        valid_levels = ['info', 'warning', 'error']
        if level not in valid_levels:
            raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")
        
        # Create the log entry
        log_entry = LogEntry.create(
            session_token=session_token,
            user_id=user_id,
            level=level,
            event_type=event_type,
            message=message,
            details=details
        )
        
        # Save to the repository
        self.log_repo.add_log(log_entry)
        
        return log_entry
    
    def get_session_logs(self, session_token: str, user_id: str,
                        is_admin: bool = False) -> List[Dict[str, Any]]:
        """
        Get the logs of a session

        Args:
            session_token: the session token
            user_id: ID of the requesting user
            is_admin: whether the user is an administrator

        Returns:
            The list of logs

        Requirements: 31.1, 32.4, 33.1, 35.3-35.8
        """
        # Get the session logs
        logs = self.log_repo.get_session_logs(session_token)
        
        # For a non-administrator, verify ownership
        if not is_admin and logs:
            # Check the user ID of the first entry (all entries should belong to the same user)
            if logs[0].get('user_id') != user_id:
                raise PermissionError(f"User {user_id} does not have permission to view logs for session {session_token}")
        
        # Sort by timestamp
        logs.sort(key=lambda x: x.get('timestamp', ''))
        
        return logs
    
    def get_user_logs(self, user_id: str, level: Optional[str] = None,
                     start_time: Optional[str] = None,
                     end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all logs of a user

        Args:
            user_id: the user ID
            level: optional filter by log level
            start_time: optional start time
            end_time: optional end time

        Returns:
            The list of logs

        Requirements: 31.3, 34.1
        """
        logs = self.log_repo.search_logs(
            user_id=user_id,
            level=level,
            start_time=start_time,
            end_time=end_time
        )
        
        # Sort by timestamp
        logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return logs
    
    def get_system_logs(self, level: Optional[str] = None,
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None,
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the system logs (administrator)

        Args:
            level: optional filter by log level
            start_time: optional start time
            end_time: optional end time
            limit: optional limit on the number of results

        Returns:
            The list of logs

        Requirements: 31.1-31.6, 32.1
        """
        logs = self.log_repo.search_logs(
            level=level,
            start_time=start_time,
            end_time=end_time
        )
        
        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Apply the limit
        if limit:
            logs = logs[:limit]
        
        return logs
    
    def get_all_sessions_logs(self, user_id: Optional[str] = None,
                             start_time: Optional[str] = None,
                             end_time: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the logs of all sessions (administrator)

        Args:
            user_id: optional filter by user ID
            start_time: optional start time
            end_time: optional end time

        Returns:
            Dictionary of logs grouped by session token

        Requirements: 32.1-32.8
        """
        logs = self.log_repo.search_logs(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Group by session token
        sessions_logs = {}
        for log in logs:
            session_token = log.get('session_token')
            if session_token not in sessions_logs:
                sessions_logs[session_token] = []
            sessions_logs[session_token].append(log)
        
        # Sort the logs of each session by time
        for session_token in sessions_logs:
            sessions_logs[session_token].sort(key=lambda x: x.get('timestamp', ''))
        
        return sessions_logs
    
    def search_logs(self, user_id: Optional[str] = None,
                   session_token: Optional[str] = None,
                   level: Optional[str] = None,
                   event_type: Optional[str] = None,
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search the logs

        Args:
            user_id: filter by user ID
            session_token: filter by session token
            level: filter by log level
            event_type: filter by event type
            start_time: the start time
            end_time: the end time
            keyword: keyword search (in the message)

        Returns:
            The list of matching logs

        Requirements: 31.3, 32.3
        """
        # Basic search
        logs = self.log_repo.search_logs(
            user_id=user_id,
            session_token=session_token,
            level=level,
            start_time=start_time,
            end_time=end_time
        )
        
        # Extra filtering
        if event_type:
            logs = [log for log in logs if log.get('event_type') == event_type]
        
        if keyword:
            keyword_lower = keyword.lower()
            logs = [log for log in logs 
                   if keyword_lower in log.get('message', '').lower()]
        
        # Sort by timestamp
        logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return logs
    
    def export_session_logs(self, session_token: str, user_id: str,
                           is_admin: bool = False, format: str = 'json') -> bytes:
        """
        Export the logs of a single session

        Args:
            session_token: the session token
            user_id: ID of the requesting user
            is_admin: whether the user is an administrator
            format: the export format (json or txt)

        Returns:
            The content of the log file (bytes)

        Requirements: 31.5, 33.8
        """
        # Get the logs (with the permission check)
        logs = self.get_session_logs(session_token, user_id, is_admin)
        
        if format == 'json':
            # JSON format
            content = json.dumps(logs, indent=2, ensure_ascii=False)
            return content.encode('utf-8')
        elif format == 'txt':
            # Text format
            lines = []
            lines.append(f"Session Logs: {session_token}")
            lines.append(f"Exported at: {datetime.now(timezone.utc).isoformat()}")
            lines.append("=" * 80)
            lines.append("")
            
            for log in logs:
                lines.append(f"[{log.get('timestamp')}] [{log.get('level').upper()}] {log.get('event_type')}")
                lines.append(f"  {log.get('message')}")
                if log.get('details'):
                    lines.append(f"  Details: {json.dumps(log.get('details'), ensure_ascii=False)}")
                lines.append("")
            
            content = '\n'.join(lines)
            return content.encode('utf-8')
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def export_multiple_sessions_logs(self, session_tokens: List[str],
                                     user_id: str, is_admin: bool = False,
                                     format: str = 'json') -> bytes:
        """
        Export the logs of several sessions as a batch

        Args:
            session_tokens: list of session tokens
            user_id: ID of the requesting user
            is_admin: whether the user is an administrator
            format: the export format (json or txt)

        Returns:
            The content of the log file (bytes)

        Requirements: 32.8
        """
        all_logs = {}
        
        for session_token in session_tokens:
            try:
                logs = self.get_session_logs(session_token, user_id, is_admin)
                all_logs[session_token] = logs
            except PermissionError:
                # Skip sessions without permission
                continue
        
        if format == 'json':
            content = json.dumps(all_logs, indent=2, ensure_ascii=False)
            return content.encode('utf-8')
        elif format == 'txt':
            lines = []
            lines.append("Multiple Session Logs Export")
            lines.append(f"Exported at: {datetime.now(timezone.utc).isoformat()}")
            lines.append(f"Total sessions: {len(all_logs)}")
            lines.append("=" * 80)
            lines.append("")
            
            for session_token, logs in all_logs.items():
                lines.append(f"\n{'=' * 80}")
                lines.append(f"Session: {session_token}")
                lines.append(f"{'=' * 80}\n")
                
                for log in logs:
                    lines.append(f"[{log.get('timestamp')}] [{log.get('level').upper()}] {log.get('event_type')}")
                    lines.append(f"  {log.get('message')}")
                    if log.get('details'):
                        lines.append(f"  Details: {json.dumps(log.get('details'), ensure_ascii=False)}")
                    lines.append("")
            
            content = '\n'.join(lines)
            return content.encode('utf-8')
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_log_statistics(self, user_id: Optional[str] = None,
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Get log statistics

        Args:
            user_id: optional filter by user ID
            start_time: optional start time
            end_time: optional end time

        Returns:
            Dictionary of statistics

        Requirements: 31.6, 32.2
        """
        logs = self.log_repo.search_logs(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Count the entries of each level
        level_counts = {'info': 0, 'warning': 0, 'error': 0}
        event_type_counts = {}
        session_count = set()
        user_count = set()
        
        for log in logs:
            level = log.get('level', 'info')
            level_counts[level] = level_counts.get(level, 0) + 1
            
            event_type = log.get('event_type', 'unknown')
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            session_count.add(log.get('session_token'))
            user_count.add(log.get('user_id'))
        
        return {
            'total_logs': len(logs),
            'level_counts': level_counts,
            'event_type_counts': event_type_counts,
            'unique_sessions': len(session_count),
            'unique_users': len(user_count),
            'time_range': {
                'start': start_time,
                'end': end_time
            }
        }
    
    def clear_session_logs(self, session_token: str, user_id: str,
                          is_admin: bool = False) -> int:
        """
        Clear the logs of a session

        Args:
            session_token: the session token
            user_id: ID of the requesting user
            is_admin: whether the user is an administrator

        Returns:
            The number of logs deleted

        Requirements: 33.6
        """
        # Verify the permission first
        self.get_session_logs(session_token, user_id, is_admin)
        
        # Delete the logs
        deleted_count = self.log_repo.delete_session_logs(session_token)
        
        return deleted_count
    
    def cleanup_old_logs(self, days: int = 30) -> int:
        """
        Remove old logs

        Args:
            days: number of days kept

        Returns:
            The number of logs deleted
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_timestamp = cutoff_time.isoformat()
        
        deleted_count = self.log_repo.delete_old_logs(cutoff_timestamp)
        
        return deleted_count
