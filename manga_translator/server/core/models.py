"""
Data models

Defines the data model classes of user accounts, permissions, sessions and audit events.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class UserPermissions:
    """Data model of user permissions"""
    # Translator permissions (whitelist + blacklist)
    allowed_translators: List[str] = field(default_factory=lambda: ["*"])
    denied_translators: List[str] = field(default_factory=list)

    # OCR permissions (whitelist + blacklist)
    allowed_ocr: List[str] = field(default_factory=list)
    denied_ocr: List[str] = field(default_factory=list)

    # Colorizer permissions (whitelist + blacklist)
    allowed_colorizers: List[str] = field(default_factory=list)
    denied_colorizers: List[str] = field(default_factory=list)

    # Renderer permissions (whitelist + blacklist)
    allowed_renderers: List[str] = field(default_factory=list)
    denied_renderers: List[str] = field(default_factory=list)

    # Workflow permissions (whitelist + blacklist)
    allowed_workflows: List[str] = field(default_factory=list)
    denied_workflows: List[str] = field(default_factory=list)
    
    # Parameter permissions (whitelist + blacklist)
    allowed_parameters: List[str] = field(default_factory=lambda: ["*"])
    denied_parameters: List[str] = field(default_factory=list)
    
    # Quota limits
    max_concurrent_tasks: int = 10
    daily_quota: int = -1  # -1 means no limit
    
    # File operation permissions
    can_upload_files: bool = True
    can_delete_files: bool = True
    
    # Offline translation permission
    allow_offline_translation: bool = False  # Whether offline translation is allowed (tasks keep running after the user goes offline)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPermissions':
        """Create from a dictionary"""
        return cls(
            allowed_translators=data.get('allowed_translators', ["*"]),
            denied_translators=data.get('denied_translators', []),
            allowed_ocr=data.get('allowed_ocr', []),
            denied_ocr=data.get('denied_ocr', []),
            allowed_colorizers=data.get('allowed_colorizers', []),
            denied_colorizers=data.get('denied_colorizers', []),
            allowed_renderers=data.get('allowed_renderers', []),
            denied_renderers=data.get('denied_renderers', []),
            allowed_workflows=data.get('allowed_workflows', []),
            denied_workflows=data.get('denied_workflows', []),
            allowed_parameters=data.get('allowed_parameters', ["*"]),
            denied_parameters=data.get('denied_parameters', []),
            max_concurrent_tasks=data.get('max_concurrent_tasks', 10),
            daily_quota=data.get('daily_quota', -1),
            can_upload_files=data.get('can_upload_files', True),
            can_delete_files=data.get('can_delete_files', True),
            allow_offline_translation=data.get('allow_offline_translation', False)
        )


@dataclass
class UserAccount:
    """Data model of a user account"""
    username: str
    password_hash: str
    role: str  # 'admin' or 'user'
    permissions: UserPermissions
    created_at: datetime
    group: str = "default"  # User group name
    last_login: Optional[datetime] = None
    is_active: bool = True
    must_change_password: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary (for serialisation)"""
        return {
            'username': self.username,
            'password_hash': self.password_hash,
            'role': self.role,
            'group': self.group,
            'permissions': self.permissions.to_dict(),
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'must_change_password': self.must_change_password
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserAccount':
        """Create from a dictionary (for deserialisation)"""
        return cls(
            username=data['username'],
            password_hash=data['password_hash'],
            role=data['role'],
            permissions=UserPermissions.from_dict(data['permissions']),
            created_at=datetime.fromisoformat(data['created_at']),
            group=data.get('group', 'default'),
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None,
            is_active=data.get('is_active', True),
            must_change_password=data.get('must_change_password', False)
        )


@dataclass
class Session:
    """Data model of a session"""
    session_id: str
    username: str
    role: str
    token: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary (for serialisation)"""
        return {
            'session_id': self.session_id,
            'username': self.username,
            'role': self.role,
            'token': self.token,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create from a dictionary (for deserialisation)"""
        return cls(
            session_id=data['session_id'],
            username=data['username'],
            role=data['role'],
            token=data['token'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_activity=datetime.fromisoformat(data['last_activity']),
            ip_address=data['ip_address'],
            user_agent=data['user_agent'],
            is_active=data.get('is_active', True)
        )


@dataclass
class AuditEvent:
    """Data model of an audit event"""
    event_id: str
    timestamp: datetime
    event_type: str  # 'login', 'logout', 'create_task', 'permission_change', etc.
    username: str
    ip_address: str
    details: Dict[str, Any]
    result: str  # 'success' or 'failure'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary (for serialisation)"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'username': self.username,
            'ip_address': self.ip_address,
            'details': self.details,
            'result': self.result
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from a dictionary (for deserialisation)"""
        return cls(
            event_id=data['event_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=data['event_type'],
            username=data['username'],
            ip_address=data['ip_address'],
            details=data['details'],
            result=data['result']
        )
    
    def to_json_line(self) -> str:
        """Convert to a JSON line (for the log file)"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


