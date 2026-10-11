"""
History management routes

Provides the API for querying, searching and downloading the translation history.
"""

import logging
import mimetypes
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from manga_translator.server.core.download_ticket_service import (
    DownloadTicketService,
    resolve_path_within,
)
from manga_translator.server.core.history_service import HistoryManagementService
from manga_translator.server.core.middleware import require_admin, require_auth
from manga_translator.server.core.models import Session
from manga_translator.server.core.permission_integration import (
    IntegratedPermissionService,
)
from manga_translator.server.core.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/history", tags=["history"])


# ============================================================================
# Request Models
# ============================================================================

class BatchDownloadRequest(BaseModel):
    """Request model of a batch download"""
    session_tokens: List[str]
    filename: Optional[str] = None

# Global service instance (initialised when the server starts)
_history_service: HistoryManagementService = None
_search_service: SearchService = None
_permission_service: IntegratedPermissionService = None
_download_ticket_service = DownloadTicketService()


def init_history_routes(
    history_service: HistoryManagementService,
    permission_service: IntegratedPermissionService,
    search_service: Optional[SearchService] = None,
    **kwargs  # Kept for the old way of calling
) -> None:
    """
    Initialise the service instances the history routes use

    Args:
        history_service: the history management service
        permission_service: the permission management service
        search_service: the search service (optional)
    """
    global _history_service, _search_service, _permission_service
    _history_service = history_service
    _search_service = search_service or SearchService()
    _permission_service = permission_service
    logger.info("History routes initialized")


def get_history_service() -> HistoryManagementService:
    """Get the history management service instance"""
    if not _history_service:
        raise RuntimeError("History service not initialized")
    return _history_service


def get_permission_service() -> IntegratedPermissionService:
    """Get the permission management service instance"""
    if not _permission_service:
        raise RuntimeError("Permission service not initialized")
    return _permission_service


def get_search_service() -> SearchService:
    """Get the search service instance"""
    if not _search_service:
        raise RuntimeError("Search service not initialized")
    return _search_service


def _sanitize_history_filename(filename: str) -> str:
    if not filename or '/' in filename or '\\' in filename or filename in {'.', '..'}:
        raise HTTPException(status_code=400, detail="Invalid file name")
    return filename


def _resolve_history_file_path(
    result_directory: str | Path,
    result_path: str,
    filename: str,
) -> Path:
    safe_filename = _sanitize_history_filename(filename)
    try:
        session_dir = resolve_path_within(result_directory, result_path)
    except ValueError:
        raise HTTPException(status_code=404, detail="The file folder of the session does not exist")

    if session_dir == Path(result_directory):
        raise HTTPException(status_code=404, detail="The file folder of the session does not exist")
    if not session_dir.exists() or not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="The file folder of the session does not exist")

    try:
        file_path = resolve_path_within(session_dir, session_dir / safe_filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file name")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="The file does not exist")
    return file_path


def _sanitize_download_filename(filename: Optional[str], default_name: str) -> str:
    sanitized = Path(filename or default_name).name.strip().replace('\r', '').replace('\n', '')
    if not sanitized or sanitized in {'.', '..'}:
        sanitized = Path(default_name).name.strip().replace('\r', '').replace('\n', '')
    if not sanitized or sanitized in {'.', '..'}:
        sanitized = 'download.zip'
    if not sanitized.endswith('.zip'):
        sanitized += '.zip'
    return sanitized


def _get_history_user_id(session: Session) -> Optional[str]:
    return None if session.role == 'admin' else session.username


def _build_ticket_response(ticket, url: str) -> dict:
    expires_in = max(1, int((ticket.expires_at - datetime.now(timezone.utc)).total_seconds()))
    return {
        "url": url,
        "filename": ticket.filename,
        "expires_in": expires_in,
        "expires_at": ticket.expires_at.isoformat(),
    }


def _issue_download_ticket(
    path: str | Path,
    allowed_root: str | Path,
    filename: str,
    media_type: str,
    delete_on_cleanup: bool = False,
) -> dict:
    ticket = _download_ticket_service.issue_ticket(
        path=path,
        allowed_root=allowed_root,
        filename=filename,
        media_type=media_type,
        delete_on_cleanup=delete_on_cleanup,
    )
    return _build_ticket_response(ticket, f"/api/history/downloads/t/{ticket.token}")


@router.api_route("/downloads/t/{ticket}", methods=["GET", "HEAD"])
async def download_by_ticket(ticket: str):
    """Serve a file download with a short-lived download ticket."""
    download_ticket = _download_ticket_service.get_ticket(ticket)
    if download_ticket is None:
        raise HTTPException(status_code=404, detail="The download link is invalid or has expired")

    return FileResponse(
        path=download_ticket.path,
        filename=download_ticket.filename,
        media_type=download_ticket.media_type,
        headers={"Cache-Control": "private, no-store"},
    )


# ============================================================================
# Endpoints for querying the user's history
# ============================================================================

@router.get("", response_model=dict)
async def get_user_history(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    session: Session = Depends(require_auth),
    history_service: HistoryManagementService = Depends(get_history_service),
    permission_service: IntegratedPermissionService = Depends(get_permission_service)
):
    """
    Get the translation history of the user (with filters)

    Requirements: 3.2, 12.2

    Args:
        start_date: the start date
        end_date: the end date
        status: filter by status
        session: the user session
        history_service: the history management service
        permission_service: the permission management service

    Returns:
        dict: contains the list of history records

    Raises:
        HTTPException: when the permission is insufficient or the query fails
    """
    # Check the view permission
    view_permission = permission_service.get_view_history_permission(session.username)
    
    if view_permission == 'none':
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to view the history"
        )
    
    try:
        # Build the filter conditions
        filters = {}
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        if status:
            filters['status'] = status
        
        # Get the user's history
        results = history_service.get_user_history(session.username, filters)
        
        return {
            "success": True,
            "history": [result.to_dict() for result in results],
            "count": len(results)
        }
    
    except Exception as e:
        logger.error(f"Error getting history for user {session.username}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while getting the history")


@router.get("/{session_token}", response_model=dict)
async def get_session_details(
    session_token: str,
    session: Session = Depends(require_auth),
    history_service: HistoryManagementService = Depends(get_history_service),
    permission_service: IntegratedPermissionService = Depends(get_permission_service)
):
    """
    Get the details of a single session

    Args:
        session_token: the session token
        session: the user session
        history_service: the history management service
        permission_service: the permission management service

    Returns:
        dict: the details of the session
    """
    # Check the view permission
    view_permission = permission_service.get_view_history_permission(session.username)
    
    if view_permission == 'none':
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to view the history"
        )
    
    try:
        # Decide the user ID (an administrator can view everything)
        user_id = _get_history_user_id(session)
        
        # Get the session directly through history_service, which checks ownership itself
        result = history_service.get_session_by_token(session_token, user_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail="The session does not exist"
            )
        
        # Get the file list of the session
        files = [
            Path(file_path).name
            for file_path in history_service.get_session_files(session_token, user_id)
        ]
        
        result_dict = result.to_dict()
        result_dict['files'] = files
        
        return {
            "success": True,
            "session": result_dict
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session details: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while getting the session details")


@router.post("/{session_token}/download-ticket")
async def create_session_download_ticket(
    session_token: str,
    filename: Optional[str] = Query(None, description="Custom download file name"),
    session: Session = Depends(require_auth),
    history_service: HistoryManagementService = Depends(get_history_service),
    permission_service: IntegratedPermissionService = Depends(get_permission_service)
):
    """Create a short-lived ticket for the ZIP download of a single session."""
    view_permission = permission_service.get_view_history_permission(session.username)
    if view_permission == 'none':
        raise HTTPException(status_code=403, detail="You do not have permission to download the history")

    user_id = _get_history_user_id(session)
    zip_path = history_service.create_download_archive(session_token, user_id)
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="The session does not exist")

    download_filename = _sanitize_download_filename(
        filename,
        f"history_{session_token[:8]}.zip",
    )
    return _issue_download_ticket(
        path=zip_path,
        allowed_root=tempfile.gettempdir(),
        filename=download_filename,
        media_type="application/zip",
        delete_on_cleanup=True,
    )


# ============================================================================
# Endpoints for the administrator's history queries
# ============================================================================

@router.get("/admin/all", response_model=dict)
async def get_all_history(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, description="Number per page"),
    offset: int = Query(0, description="Offset"),
    session: Session = Depends(require_admin),
    history_service: HistoryManagementService = Depends(get_history_service)
):
    """
    An administrator views the translation history of all users

    Requirements: 5.1-5.5, 12.4

    Args:
        user_id: filter by user ID
        start_date: the start date
        end_date: the end date
        status: filter by status
        limit: number per page
        offset: the offset
        session: the administrator session
        history_service: the history management service

    Returns:
        dict: contains the list of all history records
    """
    try:
        # Build the filter conditions
        filters = {}
        if user_id:
            filters['user_id'] = user_id
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        if status:
            filters['status'] = status
        
        # Get all history
        all_results = history_service.get_all_history(filters)
        total = len(all_results)
        
        # Apply paging
        paginated_results = all_results[offset:offset + limit]
        
        # Convert to the format the frontend expects
        records = []
        for result in paginated_results:
            result_dict = result.to_dict()
            # Map the field names to what the frontend expects
            records.append({
                'id': result_dict.get('session_token', result_dict.get('id', '')),
                'username': result_dict.get('user_id', ''),
                'filename': result_dict.get('metadata', {}).get('files', [''])[0] if result_dict.get('metadata', {}).get('files') else '-',
                'translator': result_dict.get('metadata', {}).get('translator', '-'),
                'status': result_dict.get('status', 'completed'),
                'created_at': result_dict.get('timestamp', ''),
                'file_count': result_dict.get('file_count', 0),
                'total_size': result_dict.get('total_size', 0),
            })
        
        return {
            "success": True,
            "records": records,
            "total": total,
            # Keep the old format for compatibility
            "history": [result.to_dict() for result in paginated_results],
            "count": len(paginated_results)
        }
    
    except Exception as e:
        logger.error(f"Error getting all history: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while getting the history")


# ============================================================================
# Search endpoints
# ============================================================================

@router.get("/search", response_model=dict)
async def search_history(
    q: str = Query(..., description="Search query"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    session: Session = Depends(require_auth),
    search_service: SearchService = Depends(get_search_service),
    permission_service: IntegratedPermissionService = Depends(get_permission_service)
):
    """
    Search the translation history

    Requirements: 13.1-13.5

    Args:
        q: the search query
        start_date: the start date
        end_date: the end date
        status: filter by status
        session: the user session
        search_service: the search service
        permission_service: the permission management service

    Returns:
        dict: the search results

    Raises:
        HTTPException: when the permission is insufficient or the search fails
    """
    # Check the view permission
    view_permission = permission_service.get_view_history_permission(session.username)
    
    if view_permission == 'none':
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to view the history"
        )
    
    try:
        # Build the filter conditions
        filters = {}
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        if status:
            filters['status'] = status
        
        # Decide the search scope
        user_id = _get_history_user_id(session)
        
        # Run the search
        results = search_service.search(q, filters, user_id)
        
        # Get the search statistics
        stats = search_service.get_search_stats(q, filters, user_id)
        
        return {
            "success": True,
            "query": q,
            "results": [result.to_dict() for result in results],
            "count": len(results),
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error searching history: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while searching the history")


# ============================================================================
# Download endpoints
# ============================================================================

@router.get("/{session_token}/download")
async def download_session(
    session_token: str,
    background_tasks: BackgroundTasks,
    filename: Optional[str] = Query(None, description="Custom download file name"),
    session: Session = Depends(require_auth),
    history_service: HistoryManagementService = Depends(get_history_service),
    permission_service: IntegratedPermissionService = Depends(get_permission_service)
):
    """
    Download the translation results of a single session

    Args:
        session_token: the session token
        filename: a custom file name (optional)
        session: the user session
        history_service: the history management service
        permission_service: the permission management service

    Returns:
        FileResponse: the ZIP file
    """
    # Check the view permission
    view_permission = permission_service.get_view_history_permission(session.username)
    
    if view_permission == 'none':
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to download the history"
        )
    
    try:
        # Decide the user ID (an administrator can download everything)
        is_admin = session.role == 'admin'
        user_id = None if is_admin else session.username
        
        # Create the ZIP file (history_service checks ownership itself)
        zip_path = history_service.create_download_archive(session_token, user_id)
        
        if not zip_path or not os.path.exists(zip_path):
            raise HTTPException(
                status_code=404,
                detail="The session does not exist"
            )
        
        # Add a background task that removes the temporary file
        background_tasks.add_task(history_service.cleanup_temp_file, zip_path)
        
        # Use the custom file name or the default one
        download_filename = _sanitize_download_filename(
            filename,
            f"history_{session_token[:8]}.zip",
        )
        
        # Return the file
        return FileResponse(
            path=zip_path,
            filename=download_filename,
            media_type="application/zip"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading session: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while downloading the session")


@router.post("/batch-download-ticket")
async def create_batch_download_ticket(
    request: BatchDownloadRequest,
    session: Session = Depends(require_auth),
    history_service: HistoryManagementService = Depends(get_history_service),
    permission_service: IntegratedPermissionService = Depends(get_permission_service)
):
    """Create a short-lived ticket for a batch ZIP download of history."""
    view_permission = permission_service.get_view_history_permission(session.username)
    if view_permission == 'none':
        raise HTTPException(status_code=403, detail="You do not have permission to download the history")

    if len(request.session_tokens) > 50:
        raise HTTPException(status_code=400, detail="A batch download supports at most 50 sessions")

    user_id = _get_history_user_id(session)
    zip_path = history_service.create_batch_download_archive(request.session_tokens, user_id)
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="The download file could not be created, or you do not have access")

    download_filename = _sanitize_download_filename(request.filename, os.path.basename(zip_path))
    return _issue_download_ticket(
        path=zip_path,
        allowed_root=tempfile.gettempdir(),
        filename=download_filename,
        media_type="application/zip",
        delete_on_cleanup=True,
    )


@router.post("/batch-download")
async def batch_download_sessions(
    request: BatchDownloadRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(require_auth),
    history_service: HistoryManagementService = Depends(get_history_service),
    permission_service: IntegratedPermissionService = Depends(get_permission_service)
):
    """
    Download the translation results of several sessions as a batch

    Args:
        request: the batch download request
        session: the user session
        history_service: the history management service
        permission_service: the permission management service

    Returns:
        FileResponse: the ZIP file
    """
    # Check the view permission
    view_permission = permission_service.get_view_history_permission(session.username)
    
    if view_permission == 'none':
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to download the history"
        )
    
    # Limit the number of batch downloads
    if len(request.session_tokens) > 50:
        raise HTTPException(
            status_code=400,
            detail="A batch download supports at most 50 sessions"
        )
    
    try:
        # Decide the user ID (an administrator can download everything)
        user_id = _get_history_user_id(session)
        
        # Create the batch ZIP file
        zip_path = history_service.create_batch_download_archive(
            request.session_tokens,
            user_id
        )
        
        if not zip_path or not os.path.exists(zip_path):
            raise HTTPException(
                status_code=404,
                detail="The download file could not be created, or you do not have access"
            )
        
        # Add a background task that removes the temporary file
        background_tasks.add_task(history_service.cleanup_temp_file, zip_path)
        
        # Return the file
        return FileResponse(
            path=zip_path,
            filename=os.path.basename(zip_path),
            media_type="application/zip"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error batch downloading sessions: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while downloading the sessions as a batch")


# ============================================================================
# Endpoints for deleting history
# ============================================================================

@router.get("/{session_token}/file/{filename}")
async def get_history_file(
    session_token: str,
    filename: str,
    session: Session = Depends(require_auth),
    history_service: HistoryManagementService = Depends(get_history_service),
    permission_service: IntegratedPermissionService = Depends(get_permission_service)
):
    """
    Get a single file of the history

    Args:
        session_token: the session token
        filename: the file name
        session: the user session

    Returns:
        FileResponse: the file content
    """
    # Check the view permission
    view_permission = permission_service.get_view_history_permission(session.username)
    
    if view_permission == 'none':
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to view the history"
        )
    
    try:
        # Decide the user ID (an administrator can view everything)
        is_admin = session.role == 'admin'
        user_id = None if is_admin else session.username
        
        # Get the session directly through history_service, which checks ownership itself
        result = history_service.get_session_by_token(session_token, user_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="The session does not exist, or you do not have access")
        
        # Build the file path
        file_path = _resolve_history_file_path(
            history_service.result_directory,
            result.result_path,
            filename,
        )
        media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        
        # Return the file
        return FileResponse(
            path=file_path,
            filename=file_path.name,
            media_type=media_type
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history file: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while getting the file")


@router.post("/{session_token}/file/{filename}/download-ticket")
async def create_history_file_download_ticket(
    session_token: str,
    filename: str,
    session: Session = Depends(require_auth),
    history_service: HistoryManagementService = Depends(get_history_service),
    permission_service: IntegratedPermissionService = Depends(get_permission_service)
):
    """Create a short-lived ticket for the download of a single history file."""
    view_permission = permission_service.get_view_history_permission(session.username)
    if view_permission == 'none':
        raise HTTPException(status_code=403, detail="You do not have permission to view the history")

    user_id = _get_history_user_id(session)
    result = history_service.get_session_by_token(session_token, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="The session does not exist, or you do not have access")

    file_path = _resolve_history_file_path(
        history_service.result_directory,
        result.result_path,
        filename,
    )
    media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    return _issue_download_ticket(
        path=file_path,
        allowed_root=history_service.result_directory,
        filename=file_path.name,
        media_type=media_type,
        delete_on_cleanup=False,
    )


@router.delete("/{session_token}", response_model=dict)
async def delete_session(
    session_token: str,
    session: Session = Depends(require_auth),
    history_service: HistoryManagementService = Depends(get_history_service),
    permission_service: IntegratedPermissionService = Depends(get_permission_service)
):
    """
    Delete a translation session

    Args:
        session_token: the session token
        session: the user session
        history_service: the history management service
        permission_service: the permission management service

    Returns:
        dict: the result of the deletion
    """
    # Check the delete permission
    is_admin = session.role == 'admin'
    can_delete_own = permission_service.check_delete_own_files_permission(session.username)
    
    if not is_admin and not can_delete_own:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to delete the history"
        )
    
    try:
        # Delete the session (history_service checks ownership itself)
        user_id = None if is_admin else session.username
        success = history_service.delete_session(session_token, user_id)
        
        if success:
            logger.info(f"User {session.username} deleted session: {session_token}")
            return {
                "success": True,
                "message": "Session deleted"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="The session does not exist"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while deleting the session")

