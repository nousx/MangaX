"""
Audit log routes module.

This module contains all /audit/* endpoints for audit log management.
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from manga_translator.server.core.audit_service import AuditService
from manga_translator.server.core.middleware import require_admin
from manga_translator.server.core.models import Session

logger = logging.getLogger('manga_translator.server')

router = APIRouter(prefix="/audit", tags=["audit"])


# ============================================================================
# Request/Response Models
# ============================================================================

class AuditEventResponse(BaseModel):
    """Audit event response"""
    event_id: str
    timestamp: str
    event_type: str
    username: str
    ip_address: str
    details: dict
    result: str


# ============================================================================
# Audit Log Endpoints
# ============================================================================

@router.get("/events", response_model=List[AuditEventResponse])
async def query_audit_events(
    username: Optional[str] = Query(None, description="Filter by user name"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    result: Optional[str] = Query(None, pattern="^(success|failure)$", description="Filter by result"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events returned"),
    offset: int = Query(0, ge=0, description="Number of events skipped (for paging)"),
    session: Session = Depends(require_admin)
):
    """
    Query audit events (administrator)

    Requires administrator permission. Queries the audit log, with several filter conditions.

    - **username**: filter by user name (optional)
    - **event_type**: filter by event type (optional)
    - **result**: filter by result (success or failure, optional)
    - **start_time**: start time, ISO format (optional)
    - **end_time**: end time, ISO format (optional)
    - **limit**: maximum number of events returned (default 100, maximum 1000)
    - **offset**: number of events skipped, for paging (default 0)
    """
    try:
        # Build the filter conditions
        filters = {}
        
        if username:
            filters['username'] = username
        if event_type:
            filters['event_type'] = event_type
        if result:
            filters['result'] = result
        if start_time:
            try:
                filters['start_time'] = datetime.fromisoformat(start_time)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "code": "INVALID_TIME_FORMAT",
                            "message": "start_time has an invalid format; use the ISO format"
                        }
                    }
                )
        if end_time:
            try:
                filters['end_time'] = datetime.fromisoformat(end_time)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "code": "INVALID_TIME_FORMAT",
                            "message": "end_time has an invalid format; use the ISO format"
                        }
                    }
                )
        
        # Query the audit events
        audit_service = AuditService()
        events = audit_service.query_events(
            filters=filters,
            limit=limit,
            offset=offset
        )
        
        logger.info(
            f"Admin '{session.username}' queried audit events: "
            f"{len(events)} results (filters: {filters})"
        )
        
        return [
            AuditEventResponse(
                event_id=event.event_id,
                timestamp=event.timestamp.isoformat(),
                event_type=event.event_type,
                username=event.username,
                ip_address=event.ip_address,
                details=event.details,
                result=event.result
            )
            for event in events
        ]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying audit events: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An error occurred while querying the audit events"
                }
            }
        )


@router.get("/export")
async def export_audit_events(
    username: Optional[str] = Query(None, description="Filter by user name"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    result: Optional[str] = Query(None, pattern="^(success|failure)$", description="Filter by result"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    format: str = Query("json", pattern="^(json|csv)$", description="Export format (json or csv)"),
    session: Session = Depends(require_admin)
):
    """
    Export the audit log (administrator)

    Requires administrator permission. Exports the audit log in JSON or CSV format.

    - **username**: filter by user name (optional)
    - **event_type**: filter by event type (optional)
    - **result**: filter by result (success or failure, optional)
    - **start_time**: start time, ISO format (optional)
    - **end_time**: end time, ISO format (optional)
    - **format**: export format (json or csv, default json)
    """
    try:
        # Build the filter conditions
        filters = {}
        
        if username:
            filters['username'] = username
        if event_type:
            filters['event_type'] = event_type
        if result:
            filters['result'] = result
        if start_time:
            try:
                filters['start_time'] = datetime.fromisoformat(start_time)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "code": "INVALID_TIME_FORMAT",
                            "message": "start_time has an invalid format; use the ISO format"
                        }
                    }
                )
        if end_time:
            try:
                filters['end_time'] = datetime.fromisoformat(end_time)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "code": "INVALID_TIME_FORMAT",
                            "message": "end_time has an invalid format; use the ISO format"
                        }
                    }
                )
        
        # Export the audit events
        audit_service = AuditService()
        export_data = audit_service.export_events(
            filters=filters,
            format=format
        )
        
        # Write the audit log
        try:
            audit_service.log_event(
                event_type='export_audit_log',
                username=session.username,
                ip_address='',  # TODO: get it from the request
                details={
                    'format': format,
                    'filters': {k: str(v) for k, v in filters.items()}
                },
                result='success'
            )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")
        
        logger.info(
            f"Admin '{session.username}' exported audit log: "
            f"format={format}, filters={filters}"
        )
        
        # Set the response headers
        media_type = "application/json" if format == "json" else "text/csv"
        from datetime import timezone
        filename = f"audit_log_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.{format}"
        
        return Response(
            content=export_data,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting audit events: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An error occurred while exporting the audit log"
                }
            }
        )
