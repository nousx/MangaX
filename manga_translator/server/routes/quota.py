"""
Quota management routes

Provides the API for querying quotas, statistics and management.
"""

import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from manga_translator.server.core.middleware import require_admin, require_auth
from manga_translator.server.core.models import Session
from manga_translator.server.core.quota_service import QuotaManagementService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["quota"])

# Global service instance (initialised when the server starts)
_quota_service: QuotaManagementService = None


def init_quota_routes(quota_service: QuotaManagementService) -> None:
    """
    Initialise the service instance the quota routes use

    Args:
        quota_service: the quota management service
    """
    global _quota_service
    _quota_service = quota_service
    logger.info("Quota routes initialized")


def get_quota_service() -> QuotaManagementService:
    """Get the quota management service instance"""
    if not _quota_service:
        raise RuntimeError("Quota service not initialized")
    return _quota_service


# ============================================================================
# Request and response models
# ============================================================================

class QuotaStatsResponse(BaseModel):
    """Quota statistics response"""
    user_id: str
    daily_limit: int
    used_today: int
    remaining: int
    active_sessions: int
    total_uploaded: int


class AllQuotaStatsResponse(BaseModel):
    """Response with the quota statistics of all users"""
    quotas: Dict[str, QuotaStatsResponse]
    total_users: int


class QuotaResetRequest(BaseModel):
    """Quota reset request"""
    user_id: Optional[str] = None  # None means reset all users


class QuotaResetResponse(BaseModel):
    """Quota reset response"""
    success: bool
    message: str
    users_reset: Optional[int] = None


class SetQuotaLimitsRequest(BaseModel):
    """Request for setting quota limits"""
    user_id: str
    max_file_size: Optional[int] = None
    max_files_per_upload: Optional[int] = None
    max_sessions: Optional[int] = None
    daily_quota: Optional[int] = None


# ============================================================================
# User quota endpoints
# ============================================================================

@router.get("/quota/stats", response_model=QuotaStatsResponse)
async def get_user_quota_stats(
    session: Session = Depends(require_auth),
    quota_service: QuotaManagementService = Depends(get_quota_service)
):
    """
    Get the quota statistics of the current user

    Returns:
        QuotaStatsResponse: the quota statistics
    """
    try:
        stats = quota_service.get_quota_stats(session.username)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Quota information not found")
        
        return QuotaStatsResponse(
            user_id=stats.user_id,
            daily_limit=stats.daily_limit,
            used_today=stats.used_today,
            remaining=stats.remaining,
            active_sessions=stats.active_sessions,
            total_uploaded=stats.total_uploaded
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quota stats for user {session.username}: {e}")
        raise HTTPException(status_code=500, detail=f"Getting the quota statistics failed: {str(e)}")


# ============================================================================
# Administrator quota endpoints
# ============================================================================

@router.get("/admin/quota/stats", response_model=AllQuotaStatsResponse)
async def get_all_quota_stats(
    session: Session = Depends(require_admin),
    quota_service: QuotaManagementService = Depends(get_quota_service)
):
    """
    Get the quota statistics of all users (administrator)

    Returns:
        AllQuotaStatsResponse: the quota statistics of all users
    """
    try:
        all_stats = quota_service.get_all_quota_stats()
        
        # Convert to the response format
        quotas_dict = {}
        for user_id, stats in all_stats.items():
            quotas_dict[user_id] = QuotaStatsResponse(
                user_id=stats.user_id,
                daily_limit=stats.daily_limit,
                used_today=stats.used_today,
                remaining=stats.remaining,
                active_sessions=stats.active_sessions,
                total_uploaded=stats.total_uploaded
            )
        
        return AllQuotaStatsResponse(
            quotas=quotas_dict,
            total_users=len(quotas_dict)
        )
        
    except Exception as e:
        logger.error(f"Error getting all quota stats: {e}")
        raise HTTPException(status_code=500, detail=f"Getting the quota statistics failed: {str(e)}")


@router.post("/admin/quota/reset", response_model=QuotaResetResponse)
async def reset_quota(
    request: QuotaResetRequest,
    session: Session = Depends(require_admin),
    quota_service: QuotaManagementService = Depends(get_quota_service)
):
    """
    Reset quotas by hand (administrator)

    Args:
        request: the reset request, with an optional user_id

    Returns:
        QuotaResetResponse: the result of the reset
    """
    try:
        if request.user_id:
            # Reset a single user
            success = quota_service.reset_daily_quota(request.user_id)
            
            if success:
                logger.info(f"Admin {session.username} reset quota for user {request.user_id}")
                return QuotaResetResponse(
                    success=True,
                    message=f"Quota of user {request.user_id} was reset",
                    users_reset=1
                )
            else:
                return QuotaResetResponse(
                    success=False,
                    message=f"Resetting the quota of user {request.user_id} failed"
                )
        else:
            # Reset all users
            success = quota_service.reset_daily_quota(user_id=None)
            
            if success:
                all_quotas = quota_service.get_all_quota_stats()
                users_count = len(all_quotas)
                logger.info(f"Admin {session.username} reset quota for all users ({users_count} users)")
                return QuotaResetResponse(
                    success=True,
                    message="The quotas of all users were reset",
                    users_reset=users_count
                )
            else:
                return QuotaResetResponse(
                    success=False,
                    message="Resetting the quotas of all users failed"
                )
                
    except Exception as e:
        logger.error(f"Error resetting quota: {e}")
        raise HTTPException(status_code=500, detail=f"Resetting the quota failed: {str(e)}")


@router.post("/admin/quota/set-limits")
async def set_quota_limits(
    request: SetQuotaLimitsRequest,
    session: Session = Depends(require_admin),
    quota_service: QuotaManagementService = Depends(get_quota_service)
):
    """
    Set the quota limits of a user (administrator)

    Args:
        request: the request for setting quota limits

    Returns:
        dict: the result of the operation
    """
    try:
        success = quota_service.set_user_quota_limits(
            user_id=request.user_id,
            max_file_size=request.max_file_size,
            max_files_per_upload=request.max_files_per_upload,
            max_sessions=request.max_sessions,
            daily_quota=request.daily_quota
        )
        
        if success:
            logger.info(f"Admin {session.username} updated quota limits for user {request.user_id}")
            return {
                "success": True,
                "message": f"Quota limits of user {request.user_id} were updated"
            }
        else:
            return {
                "success": False,
                "message": f"Updating the quota limits of user {request.user_id} failed"
            }
            
    except Exception as e:
        logger.error(f"Error setting quota limits: {e}")
        raise HTTPException(status_code=500, detail=f"Setting the quota limits failed: {str(e)}")


@router.get("/admin/quota/user/{user_id}", response_model=QuotaStatsResponse)
async def get_user_quota_stats_admin(
    user_id: str,
    session: Session = Depends(require_admin),
    quota_service: QuotaManagementService = Depends(get_quota_service)
):
    """
    Get the quota statistics of the given user (administrator)

    Args:
        user_id: the user ID

    Returns:
        QuotaStatsResponse: the quota statistics
    """
    try:
        stats = quota_service.get_quota_stats(user_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail=f"Quota information of user {user_id} was not found")
        
        return QuotaStatsResponse(
            user_id=stats.user_id,
            daily_limit=stats.daily_limit,
            used_today=stats.used_today,
            remaining=stats.remaining,
            active_sessions=stats.active_sessions,
            total_uploaded=stats.total_uploaded
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quota stats for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Getting the quota statistics failed: {str(e)}")
