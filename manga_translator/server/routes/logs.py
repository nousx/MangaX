"""
Log management API routes

Provides querying, exporting and managing logs.

Requirements: 31.1-31.6, 32.1-32.8, 33.1-33.8
"""

import io
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from manga_translator.server.core.log_management_service import LogManagementService
from manga_translator.server.core.middleware import require_admin, require_auth
from manga_translator.server.core.models import Session
from manga_translator.server.core.session_security_service import SessionSecurityService
from manga_translator.server.repositories.log_repository import LogRepository

# Create the router
logs_router = APIRouter(prefix='/api/logs', tags=['logs'])

# Initialise the services
log_repo = LogRepository('manga_translator/server/data/logs.json')
log_service = LogManagementService(log_repo)
session_security_service = SessionSecurityService()


# Pydantic models
class ExportRequest(BaseModel):
    """Request model of a batch export"""
    session_tokens: List[str]
    format: str = 'json'


class CleanupRequest(BaseModel):
    """Request model of a clean-up"""
    days: int = 30


@logs_router.get('/session/{session_token}')
async def get_session_logs(
    session_token: str,
    format: str = Query('list', description='Return format (json or list)'),
    session: Session = Depends(require_auth)
):
    """
    Get the logs of a session

    Requirements: 31.1, 33.1-33.8, 35.3-35.8
    """
    try:
        user_id = session.username
        is_admin = session.role == 'admin'
        
        # Check session ownership (requirements 35.3, 35.4, 35.5)
        allowed, reason = session_security_service.check_session_ownership(
            session_token,
            user_id,
            "view"
        )
        
        # Record the access attempt (requirement 35.8)
        session_security_service.log_access_attempt(
            session_token,
            user_id,
            "view",
            allowed,
            reason
        )
        
        if not allowed:
            raise HTTPException(
                status_code=403,
                detail=reason or "You do not have permission to access the logs of this session"
            )
        
        # Get the logs
        logs = log_service.get_session_logs(session_token, user_id, is_admin)
        
        if format == 'json':
            return logs
        else:
            return {
                'success': True,
                'message': 'Session logs retrieved successfully',
                'data': {
                    'session_token': session_token,
                    'logs': logs,
                    'count': len(logs)
                }
            }
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to retrieve session logs: {str(e)}')


@logs_router.get('/session/{session_token}/export')
async def export_session_logs(
    session_token: str,
    format: str = Query('json', description='Export format (json or txt)'),
    session: Session = Depends(require_auth)
):
    """
    Export the logs of a single session

    Requirements: 31.5, 33.8, 35.7
    """
    try:
        user_id = session.username
        is_admin = session.role == 'admin'
        
        # Check session ownership (requirements 35.3, 35.4, 35.5, 35.7)
        allowed, reason = session_security_service.check_session_ownership(
            session_token,
            user_id,
            "export"
        )
        
        # Record the access attempt (requirement 35.8)
        session_security_service.log_access_attempt(
            session_token,
            user_id,
            "export",
            allowed,
            reason
        )
        
        if not allowed:
            raise HTTPException(
                status_code=403,
                detail=reason or "You do not have permission to export the logs of this session"
            )
        
        # Export the logs
        log_data = log_service.export_session_logs(
            session_token, user_id, is_admin, format
        )
        
        # Decide the file name and the MIME type
        if format == 'json':
            filename = f'session_{session_token}_logs.json'
            media_type = 'application/json'
        else:
            filename = f'session_{session_token}_logs.txt'
            media_type = 'text/plain'
        
        # Return the file
        return StreamingResponse(
            io.BytesIO(log_data),
            media_type=media_type,
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to export session logs: {str(e)}')


@logs_router.delete('/session/{session_token}/clear')
async def clear_session_logs(
    session_token: str,
    session: Session = Depends(require_auth)
):
    """
    Clear the logs of a session

    Requirements: 33.6, 35.7
    """
    try:
        user_id = session.username
        is_admin = session.role == 'admin'
        
        # Check session ownership (requirements 35.3, 35.4, 35.5, 35.7)
        allowed, reason = session_security_service.check_session_ownership(
            session_token,
            user_id,
            "edit"
        )
        
        # Record the access attempt (requirement 35.8)
        session_security_service.log_access_attempt(
            session_token,
            user_id,
            "edit",
            allowed,
            reason
        )
        
        if not allowed:
            raise HTTPException(
                status_code=403,
                detail=reason or "You do not have permission to clear the logs of this session"
            )
        
        # Clear the logs
        deleted_count = log_service.clear_session_logs(session_token, user_id, is_admin)
        
        return {
            'success': True,
            'message': 'Session logs cleared successfully',
            'data': {
                'session_token': session_token,
                'deleted_count': deleted_count
            }
        }
    
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to clear session logs: {str(e)}')


@logs_router.get('')
async def get_logs(
    task_id: Optional[str] = Query(None, description='Filter by task ID'),
    limit: int = Query(50, description='Limit on the number returned'),
    level: Optional[str] = Query(None, description='Filter by log level'),
    session: Session = Depends(require_auth)
):
    """
    Get the logs (with a filter by task ID)

    Used on the user side to watch the logs of a translation task live
    """
    try:
        from manga_translator.server.core.logging_manager import get_task_logs
        
        if task_id:
            # Get the logs by task ID
            logs = get_task_logs(task_id, limit)
            return logs
        else:
            # Return an empty list
            return []
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to retrieve logs: {str(e)}')


@logs_router.get('/user')
async def get_user_logs(
    level: Optional[str] = Query(None, description='Filter by log level'),
    start_time: Optional[str] = Query(None, description='Start time (ISO format)'),
    end_time: Optional[str] = Query(None, description='End time (ISO format)'),
    session: Session = Depends(require_auth)
):
    """
    Get all logs of the user

    Requirements: 31.3, 34.1
    """
    try:
        user_id = session.username
        
        # Get the logs
        logs = log_service.get_user_logs(user_id, level, start_time, end_time)
        
        return {
            'success': True,
            'message': 'User logs retrieved successfully',
            'data': {
                'user_id': user_id,
                'logs': logs,
                'count': len(logs)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to retrieve user logs: {str(e)}')


@logs_router.get('/search')
async def search_logs(
    session_token: Optional[str] = Query(None, description='Filter by session token'),
    level: Optional[str] = Query(None, description='Filter by log level'),
    event_type: Optional[str] = Query(None, description='Filter by event type'),
    start_time: Optional[str] = Query(None, description='Start time'),
    end_time: Optional[str] = Query(None, description='End time'),
    keyword: Optional[str] = Query(None, description='Keyword search'),
    session: Session = Depends(require_auth)
):
    """
    Search the logs

    Requirements: 31.3, 32.3
    """
    try:
        user_id = session.username
        is_admin = session.role == 'admin'
        
        # Non-administrators can only search their own logs
        search_user_id = user_id if not is_admin else None
        
        # Search the logs
        logs = log_service.search_logs(
            user_id=search_user_id,
            session_token=session_token,
            level=level,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            keyword=keyword
        )
        
        return {
            'success': True,
            'message': 'Logs search completed',
            'data': {
                'logs': logs,
                'count': len(logs),
                'filters': {
                    'session_token': session_token,
                    'level': level,
                    'event_type': event_type,
                    'start_time': start_time,
                    'end_time': end_time,
                    'keyword': keyword
                }
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to search logs: {str(e)}')


@logs_router.get('/admin/system')
async def get_system_logs(
    level: Optional[str] = Query(None, description='Filter by log level'),
    start_time: Optional[str] = Query(None, description='Start time'),
    end_time: Optional[str] = Query(None, description='End time'),
    limit: Optional[int] = Query(None, description='Limit on the number of results'),
    session: Session = Depends(require_admin)
):
    """
    Get the system logs (administrator)

    Requirements: 31.1-31.6, 32.1
    """
    try:
        # Get the system logs
        logs = log_service.get_system_logs(level, start_time, end_time, limit)
        
        return {
            'success': True,
            'message': 'System logs retrieved successfully',
            'data': {
                'logs': logs,
                'count': len(logs)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to retrieve system logs: {str(e)}')


@logs_router.get('/admin/sessions')
async def get_all_sessions_logs(
    user_id: Optional[str] = Query(None, description='Filter by user ID'),
    start_time: Optional[str] = Query(None, description='Start time'),
    end_time: Optional[str] = Query(None, description='End time'),
    session: Session = Depends(require_admin)
):
    """
    Get the logs of all sessions (administrator)

    Requirements: 32.1-32.8
    """
    try:
        # Get the logs of all sessions
        sessions_logs = log_service.get_all_sessions_logs(user_id, start_time, end_time)
        
        return {
            'success': True,
            'message': 'All sessions logs retrieved successfully',
            'data': {
                'sessions': sessions_logs,
                'session_count': len(sessions_logs)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to retrieve sessions logs: {str(e)}')


@logs_router.post('/admin/export')
async def export_multiple_sessions(
    request: ExportRequest,
    session: Session = Depends(require_admin)
):
    """
    Export the logs of several sessions as a batch (administrator)

    Requirements: 32.8
    """
    try:
        if not request.session_tokens:
            raise HTTPException(status_code=400, detail='No session tokens provided')
        
        user_id = session.username
        
        # Export the logs
        log_data = log_service.export_multiple_sessions_logs(
            request.session_tokens, user_id, is_admin=True, format=request.format
        )
        
        # Decide the file name and the MIME type
        if request.format == 'json':
            filename = 'multiple_sessions_logs.json'
            media_type = 'application/json'
        else:
            filename = 'multiple_sessions_logs.txt'
            media_type = 'text/plain'
        
        # Return the file
        return StreamingResponse(
            io.BytesIO(log_data),
            media_type=media_type,
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to export logs: {str(e)}')


@logs_router.get('/admin/statistics')
async def get_log_statistics(
    user_id: Optional[str] = Query(None, description='Filter by user ID'),
    start_time: Optional[str] = Query(None, description='Start time'),
    end_time: Optional[str] = Query(None, description='End time'),
    session: Session = Depends(require_admin)
):
    """
    Get log statistics (administrator)

    Requirements: 31.6, 32.2
    """
    try:
        # Get the statistics
        stats = log_service.get_log_statistics(user_id, start_time, end_time)
        
        return {
            'success': True,
            'message': 'Log statistics retrieved successfully',
            'data': stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to retrieve log statistics: {str(e)}')


@logs_router.post('/admin/cleanup')
async def cleanup_old_logs(
    request: CleanupRequest,
    session: Session = Depends(require_admin)
):
    """
    Remove old logs (administrator)
    """
    try:
        if request.days < 1:
            raise HTTPException(status_code=400, detail='Days must be at least 1')
        
        # Remove old logs
        deleted_count = log_service.cleanup_old_logs(request.days)
        
        return {
            'success': True,
            'message': 'Old logs cleaned up successfully',
            'data': {
                'deleted_count': deleted_count,
                'retention_days': request.days
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to cleanup old logs: {str(e)}')
