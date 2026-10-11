"""
Audit log service (AuditService)

Records and queries the audit log, with filtering, export and rotation.
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from manga_translator.server.core.models import AuditEvent

logger = logging.getLogger(__name__)


class AuditService:
    """Audit log service"""
    
    def __init__(
        self,
        audit_log_file: str = "manga_translator/server/data/audit.log",
        max_log_size_mb: int = 10,
        max_backup_files: int = 5
    ):
        """
        Initialise the audit log service

        Args:
            audit_log_file: path of the audit log file
            max_log_size_mb: maximum size of the log file (MB); it is rotated automatically beyond that
            max_backup_files: number of backup files kept
        """
        self.audit_log_file = audit_log_file
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.max_backup_files = max_backup_files
        
        # Make sure the folder of the log file exists
        log_dir = Path(audit_log_file).parent
        if log_dir and not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Make sure the log file exists
        if not Path(audit_log_file).exists():
            Path(audit_log_file).touch()
    
    def log_event(
        self,
        event_type: str,
        username: str,
        ip_address: str,
        details: Dict[str, Any],
        result: str
    ) -> AuditEvent:
        """
        Record an audit event

        Args:
            event_type: the event type (such as 'login', 'logout', 'create_task', 'permission_change')
            username: the user name
            ip_address: the IP address
            details: details of the event
            result: the result ('success' or 'failure')

        Returns:
            AuditEvent: the audit event object that was created
        """
        # Create the audit event
        event = AuditEvent(
            event_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            username=username,
            ip_address=ip_address,
            details=details,
            result=result
        )
        
        # Write to the log file
        try:
            with open(self.audit_log_file, 'a', encoding='utf-8') as f:
                f.write(event.to_json_line() + '\n')
            
            # Check whether rotation is needed
            self._check_and_rotate()
            
            logger.debug(
                f"Logged audit event: {event_type} by {username} - {result}"
            )
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
        
        return event
    
    def query_events(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEvent]:
        """
        Query audit events

        Args:
            filters: dictionary of filter conditions; supported keys:
                - username: the user name
                - event_type: the event type
                - result: the result ('success' or 'failure')
                - start_time: the start time (datetime)
                - end_time: the end time (datetime)
            limit: maximum number of events returned
            offset: number of events skipped (for paging)

        Returns:
            List[AuditEvent]: list of the audit events that match
        """
        if filters is None:
            filters = {}
        
        events = []
        
        try:
            with open(self.audit_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse each line
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    event_data = json.loads(line)
                    event = AuditEvent.from_dict(event_data)
                    
                    # Apply the filter conditions
                    if self._matches_filters(event, filters):
                        events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse audit log line: {e}")
                    continue
            
            # Sort by time, descending (newest first)
            events.sort(key=lambda e: e.timestamp, reverse=True)
            
            # Apply paging
            return events[offset:offset + limit]
        
        except FileNotFoundError:
            logger.warning(f"Audit log file not found: {self.audit_log_file}")
            return []
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []
    
    def export_events(
        self,
        filters: Optional[Dict[str, Any]] = None,
        format: str = 'json'
    ) -> str:
        """
        Export audit events

        Args:
            filters: the filter conditions (as for query_events)
            format: the export format ('json' or 'csv')

        Returns:
            str: the exported data as a string
        """
        events = self.query_events(filters=filters, limit=10000)
        
        if format == 'json':
            return self._export_json(events)
        elif format == 'csv':
            return self._export_csv(events)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def rotate_log_file(self) -> bool:
        """
        Rotate the log file by hand

        Returns:
            bool: whether the rotation succeeded
        """
        try:
            if not Path(self.audit_log_file).exists():
                logger.warning("Audit log file does not exist, nothing to rotate")
                return False
            
            # Build the backup file name (with a timestamp)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"{self.audit_log_file}.{timestamp}"
            
            # Move the current log file to the backup
            shutil.move(self.audit_log_file, backup_file)
            
            # Create a new log file
            Path(self.audit_log_file).touch()
            
            logger.info(f"Rotated audit log: {backup_file}")
            
            # Remove old backup files
            self._cleanup_old_backups()
            
            return True
        except Exception as e:
            logger.error(f"Failed to rotate audit log: {e}")
            return False
    
    def _matches_filters(
        self,
        event: AuditEvent,
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check whether an event matches the filter conditions

        Args:
            event: the audit event
            filters: the filter conditions

        Returns:
            bool: whether it matches
        """
        # Filter by user name
        if 'username' in filters:
            if event.username != filters['username']:
                return False
        
        # Filter by event type
        if 'event_type' in filters:
            if event.event_type != filters['event_type']:
                return False
        
        # Filter by result
        if 'result' in filters:
            if event.result != filters['result']:
                return False
        
        # Filter by time range
        if 'start_time' in filters:
            if event.timestamp < filters['start_time']:
                return False
        
        if 'end_time' in filters:
            if event.timestamp > filters['end_time']:
                return False
        
        return True
    
    def _check_and_rotate(self) -> None:
        """Check the size of the log file and rotate it when it is over the limit"""
        try:
            file_size = Path(self.audit_log_file).stat().st_size
            
            if file_size > self.max_log_size_bytes:
                logger.info(
                    f"Audit log size ({file_size} bytes) exceeds limit "
                    f"({self.max_log_size_bytes} bytes), rotating..."
                )
                self.rotate_log_file()
        except Exception as e:
            logger.error(f"Failed to check log file size: {e}")
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backup files, keeping only the newest N"""
        try:
            log_dir = Path(self.audit_log_file).parent
            log_name = Path(self.audit_log_file).name
            
            # Find all backup files
            backup_files = sorted(
                log_dir.glob(f"{log_name}.*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Delete the backup files beyond the limit
            for backup_file in backup_files[self.max_backup_files:]:
                try:
                    backup_file.unlink()
                    logger.info(f"Deleted old backup: {backup_file}")
                except Exception as e:
                    logger.error(f"Failed to delete backup {backup_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
    
    def _export_json(self, events: List[AuditEvent]) -> str:
        """Export in JSON format"""
        data = [event.to_dict() for event in events]
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def _export_csv(self, events: List[AuditEvent]) -> str:
        """Export in CSV format"""
        if not events:
            return ""
        
        # CSV header
        lines = [
            "event_id,timestamp,event_type,username,ip_address,result,details"
        ]
        
        # CSV data rows
        for event in events:
            details_str = json.dumps(event.details, ensure_ascii=False).replace('"', '""')
            line = (
                f'"{event.event_id}",'
                f'"{event.timestamp.isoformat()}",'
                f'"{event.event_type}",'
                f'"{event.username}",'
                f'"{event.ip_address}",'
                f'"{event.result}",'
                f'"{details_str}"'
            )
            lines.append(line)
        
        return '\n'.join(lines)
