"""
User management routes module.

This module contains all /users/* endpoints for user account management.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from manga_translator.server.core.audit_service import AuditService
from manga_translator.server.core.middleware import get_services, require_admin
from manga_translator.server.core.models import Session, UserPermissions

logger = logging.getLogger('manga_translator.server')

router = APIRouter(prefix="/api/admin/users", tags=["users"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateUserRequest(BaseModel):
    """Request for creating a user"""
    # Character rules are enforced by AccountService.create_user (returns 400).
    username: str = Field(..., min_length=1, max_length=50, description="User name")
    password: str = Field(..., min_length=6, description="Password (at least 6 characters)")
    role: str = Field(..., pattern="^(admin|user)$", description="Role (admin or user)")
    group: str = Field(default="default", description="User group name")
    permissions: Optional[dict] = Field(None, description="User permissions (optional)")


class UpdateUserRequest(BaseModel):
    """Request for updating a user"""
    role: Optional[str] = Field(None, pattern="^(admin|user)$", description="Role")
    group: Optional[str] = Field(None, description="User group name")
    is_active: Optional[bool] = Field(None, description="Whether the account is active")
    must_change_password: Optional[bool] = Field(None, description="Whether the password must be changed")


class UpdatePermissionsRequest(BaseModel):
    """Request for updating permissions"""
    allowed_translators: Optional[List[str]] = Field(None, description="List of translators the user may use (allow list)")
    denied_translators: Optional[List[str]] = Field(None, description="List of translators the user may not use (deny list)")
    allowed_ocr: Optional[List[str]] = Field(None, description="List of OCR engines the user may use (allow list)")
    denied_ocr: Optional[List[str]] = Field(None, description="List of OCR engines the user may not use (deny list)")
    allowed_colorizers: Optional[List[str]] = Field(None, description="List of colorizers the user may use (allow list)")
    denied_colorizers: Optional[List[str]] = Field(None, description="List of colorizers the user may not use (deny list)")
    allowed_renderers: Optional[List[str]] = Field(None, description="List of renderers the user may use (allow list)")
    denied_renderers: Optional[List[str]] = Field(None, description="List of renderers the user may not use (deny list)")
    allowed_workflows: Optional[List[str]] = Field(None, description="List of workflows the user may use (allow list)")
    denied_workflows: Optional[List[str]] = Field(None, description="List of workflows the user may not use (deny list)")
    allowed_parameters: Optional[List[str]] = Field(None, description="List of parameters the user may adjust (allow list)")
    denied_parameters: Optional[List[str]] = Field(None, description="List of parameters the user may not adjust (deny list)")
    max_concurrent_tasks: Optional[int] = Field(None, ge=0, description="Maximum number of concurrent tasks")
    daily_quota: Optional[int] = Field(None, ge=-1, description="Daily translation quota (-1 means unlimited)")
    can_upload_files: Optional[bool] = Field(None, description="Whether the user may upload files")
    can_delete_files: Optional[bool] = Field(None, description="Whether the user may delete files")


class UserResponse(BaseModel):
    """User response"""
    username: str
    role: str
    group: str
    permissions: dict
    created_at: str
    last_login: Optional[str]
    is_active: bool
    must_change_password: bool


# ============================================================================
# User Management Endpoints
# ============================================================================

@router.post("", response_model=UserResponse, status_code=201)
async def create_user(
    request: CreateUserRequest,
    session: Session = Depends(require_admin)
):
    """
    Create a new user (administrator)

    Requires administrator permission. Creates a new user account and sets its initial permissions.

    - **username**: the user name (unique)
    - **password**: the password (at least 6 characters)
    - **role**: the role (admin or user)
    - **permissions**: the user permissions (optional; the default permissions are used when not given)
    """
    account_service, _, _ = get_services()
    
    try:
        # Parse the permissions
        permissions = None
        if request.permissions:
            permissions = UserPermissions.from_dict(request.permissions)
        
        # Create the user
        account = account_service.create_user(
            username=request.username,
            password=request.password,
            role=request.role,
            group=request.group,
            permissions=permissions
        )
        
        # Write the audit log
        try:
            audit_service = AuditService()
            audit_service.log_event(
                event_type='create_user',
                username=session.username,
                ip_address='',  # TODO: get it from the request
                details={
                    'target_user': request.username,
                    'role': request.role
                },
                result='success'
            )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")
        
        logger.info(f"User created by admin '{session.username}': {request.username}")
        
        return UserResponse(
            username=account.username,
            role=account.role,
            group=account.group,
            permissions=account.permissions.to_dict(),
            created_at=account.created_at.isoformat(),
            last_login=account.last_login.isoformat() if account.last_login else None,
            is_active=account.is_active,
            must_change_password=account.must_change_password
        )
    
    except ValueError as e:
        logger.warning(f"Failed to create user: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": str(e)
                }
            }
        )
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An error occurred while creating the user"
                }
            }
        )


@router.get("")
async def list_users(
    session: Session = Depends(require_admin)
):
    """
    List all users (administrator)

    Requires administrator permission. Returns the list of all user accounts, with their quota usage.
    """
    account_service, _, permission_service = get_services()
    
    try:
        accounts = account_service.list_users()
        
        result = []
        for account in accounts:
            # Get the quota use of the user
            daily_used = permission_service.get_daily_usage(account.username)
            daily_limit = permission_service.get_effective_daily_quota(account.username)
            
            user_data = {
                "username": account.username,
                "role": account.role,
                "group": account.group,
                "permissions": account.permissions.to_dict(),
                "created_at": account.created_at.isoformat(),
                "last_login": account.last_login.isoformat() if account.last_login else None,
                "is_active": account.is_active,
                "must_change_password": account.must_change_password,
                "quota": {
                    "daily_used": daily_used,
                    "daily_limit": daily_limit if daily_limit > 0 else 999999,
                    "monthly_used": 0,  # TODO: implement monthly statistics
                    "monthly_limit": 999999
                }
            }
            result.append(user_data)
        
        return result
    
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An error occurred while getting the list of users"
                }
            }
        )


@router.get("/{username}", response_model=UserResponse)
async def get_user(
    username: str,
    session: Session = Depends(require_admin)
):
    """
    Get the information of a user (administrator)

    Requires administrator permission. Returns the details of the given user.

    - **username**: the user name
    """
    account_service, _, _ = get_services()
    
    try:
        account = account_service.get_user(username)
        
        if not account:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "code": "USER_NOT_FOUND",
                        "message": f"User '{username}' does not exist"
                    }
                }
            )
        
        return UserResponse(
            username=account.username,
            role=account.role,
            group=account.group,
            permissions=account.permissions.to_dict(),
            created_at=account.created_at.isoformat(),
            last_login=account.last_login.isoformat() if account.last_login else None,
            is_active=account.is_active,
            must_change_password=account.must_change_password
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An error occurred while getting the user information"
                }
            }
        )


@router.put("/{username}", response_model=UserResponse)
async def update_user(
    username: str,
    request: UpdateUserRequest,
    session: Session = Depends(require_admin)
):
    """
    Update the information of a user (administrator)

    Requires administrator permission. Updates the role, the active state and other information of a user.

    - **username**: the user name
    - **role**: the role (optional)
    - **is_active**: whether the account is active (optional)
    - **must_change_password**: whether the password must be changed (optional)
    """
    account_service, session_service, _ = get_services()
    
    try:
        # Build the update dictionary
        updates = {}
        if request.role is not None:
            updates['role'] = request.role
        if request.group is not None:
            updates['group'] = request.group
        if request.is_active is not None:
            updates['is_active'] = request.is_active
        if request.must_change_password is not None:
            updates['must_change_password'] = request.must_change_password
        
        if not updates:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "NO_UPDATES",
                        "message": "No fields to update were given"
                    }
                }
            )
        
        # Update the user
        account_service.update_user(username, updates)
        
        # When a user is deactivated, end all their sessions
        if request.is_active is False:
            terminated_count = session_service.terminate_user_sessions(username)
            logger.info(f"Terminated {terminated_count} session(s) for deactivated user: {username}")
        
        # Write the audit log
        try:
            audit_service = AuditService()
            audit_service.log_event(
                event_type='update_user',
                username=session.username,
                ip_address='',  # TODO: get it from the request
                details={
                    'target_user': username,
                    'updates': updates
                },
                result='success'
            )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")
        
        # Get the user information after the update
        account = account_service.get_user(username)
        
        logger.info(f"User updated by admin '{session.username}': {username}")
        
        return UserResponse(
            username=account.username,
            role=account.role,
            group=account.group,
            permissions=account.permissions.to_dict(),
            created_at=account.created_at.isoformat(),
            last_login=account.last_login.isoformat() if account.last_login else None,
            is_active=account.is_active,
            must_change_password=account.must_change_password
        )
    
    except ValueError as e:
        logger.warning(f"Failed to update user: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": str(e)
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An error occurred while updating the user information"
                }
            }
        )


@router.delete("/{username}", status_code=204)
async def delete_user(
    username: str,
    session: Session = Depends(require_admin)
):
    """
    Delete a user (administrator)

    Requires administrator permission. Deletes the given user account and terminates all its sessions.

    - **username**: the user name
    """
    account_service, session_service, _ = get_services()
    
    try:
        # Prevent deleting oneself
        if username == session.username:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "CANNOT_DELETE_SELF",
                        "message": "You cannot delete your own account"
                    }
                }
            )
        
        # End all sessions of the user
        terminated_count = session_service.terminate_user_sessions(username)
        logger.info(f"Terminated {terminated_count} session(s) for deleted user: {username}")
        
        # Delete the user
        account_service.delete_user(username)
        
        # Write the audit log
        try:
            audit_service = AuditService()
            audit_service.log_event(
                event_type='delete_user',
                username=session.username,
                ip_address='',  # TODO: get it from the request
                details={
                    'target_user': username
                },
                result='success'
            )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")
        
        logger.info(f"User deleted by admin '{session.username}': {username}")
        
        return None  # 204 No Content
    
    except ValueError as e:
        logger.warning(f"Failed to delete user: {e}")
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "USER_NOT_FOUND",
                    "message": str(e)
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An error occurred while deleting the user"
                }
            }
        )


@router.put("/{username}/permissions", response_model=UserResponse)
async def update_user_permissions(
    username: str,
    request: UpdatePermissionsRequest,
    session: Session = Depends(require_admin)
):
    """
    Update the permissions of a user (administrator)

    Requires administrator permission. Updates the permission configuration of a user.

    - **username**: the user name
    - **allowed_translators**: list of translators the user may use (optional)
    - **allowed_parameters**: list of parameters the user may adjust (optional)
    - **max_concurrent_tasks**: maximum number of concurrent tasks (optional)
    - **daily_quota**: daily translation quota (optional; -1 means unlimited)
    - **can_upload_files**: whether the user may upload files (optional)
    - **can_delete_files**: whether the user may delete files (optional)
    """
    account_service, _, permission_service = get_services()
    
    try:
        # Get the current user
        account = account_service.get_user(username)
        if not account:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "code": "USER_NOT_FOUND",
                        "message": f"User '{username}' does not exist"
                    }
                }
            )
        
        # Build the permission update dictionary
        permissions_dict = account.permissions.to_dict()
        updated_fields = []
        
        if request.allowed_translators is not None:
            permissions_dict['allowed_translators'] = request.allowed_translators
            updated_fields.append('allowed_translators')
        if request.denied_translators is not None:
            permissions_dict['denied_translators'] = request.denied_translators
            updated_fields.append('denied_translators')
        if request.allowed_ocr is not None:
            permissions_dict['allowed_ocr'] = request.allowed_ocr
            updated_fields.append('allowed_ocr')
        if request.denied_ocr is not None:
            permissions_dict['denied_ocr'] = request.denied_ocr
            updated_fields.append('denied_ocr')
        if request.allowed_colorizers is not None:
            permissions_dict['allowed_colorizers'] = request.allowed_colorizers
            updated_fields.append('allowed_colorizers')
        if request.denied_colorizers is not None:
            permissions_dict['denied_colorizers'] = request.denied_colorizers
            updated_fields.append('denied_colorizers')
        if request.allowed_renderers is not None:
            permissions_dict['allowed_renderers'] = request.allowed_renderers
            updated_fields.append('allowed_renderers')
        if request.denied_renderers is not None:
            permissions_dict['denied_renderers'] = request.denied_renderers
            updated_fields.append('denied_renderers')
        if request.allowed_workflows is not None:
            permissions_dict['allowed_workflows'] = request.allowed_workflows
            updated_fields.append('allowed_workflows')
        if request.denied_workflows is not None:
            permissions_dict['denied_workflows'] = request.denied_workflows
            updated_fields.append('denied_workflows')
        if request.allowed_parameters is not None:
            permissions_dict['allowed_parameters'] = request.allowed_parameters
            updated_fields.append('allowed_parameters')
        if request.denied_parameters is not None:
            permissions_dict['denied_parameters'] = request.denied_parameters
            updated_fields.append('denied_parameters')
        if request.max_concurrent_tasks is not None:
            permissions_dict['max_concurrent_tasks'] = request.max_concurrent_tasks
            updated_fields.append('max_concurrent_tasks')
        if request.daily_quota is not None:
            permissions_dict['daily_quota'] = request.daily_quota
            updated_fields.append('daily_quota')
        if request.can_upload_files is not None:
            permissions_dict['can_upload_files'] = request.can_upload_files
            updated_fields.append('can_upload_files')
        if request.can_delete_files is not None:
            permissions_dict['can_delete_files'] = request.can_delete_files
            updated_fields.append('can_delete_files')
        
        if not updated_fields:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "NO_UPDATES",
                        "message": "No permission fields to update were given"
                    }
                }
            )
        
        # Update the permissions
        account_service.update_user(username, {'permissions': permissions_dict})
        
        # Write the audit log
        try:
            audit_service = AuditService()
            audit_service.log_event(
                event_type='update_permissions',
                username=session.username,
                ip_address='',  # TODO: get it from the request
                details={
                    'target_user': username,
                    'updated_fields': updated_fields,
                    'new_permissions': permissions_dict
                },
                result='success'
            )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")
        
        # Get the user information after the update
        account = account_service.get_user(username)
        
        logger.info(f"Permissions updated by admin '{session.username}' for user: {username}")
        
        return UserResponse(
            username=account.username,
            role=account.role,
            group=account.group,
            permissions=account.permissions.to_dict(),
            created_at=account.created_at.isoformat(),
            last_login=account.last_login.isoformat() if account.last_login else None,
            is_active=account.is_active,
            must_change_password=account.must_change_password
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user permissions: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An error occurred while updating the user permissions"
                }
            }
        )
