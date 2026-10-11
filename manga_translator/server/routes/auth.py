"""
Authentication API endpoints

Provides user authentication functionality including login, logout, password change, 
session checking, initial setup, and user registration.
"""

import logging
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from manga_translator.server.core.config_manager import admin_settings
from manga_translator.server.core.request_rate_limiter import SlidingWindowRateLimiter
from manga_translator.server.core.setup_guard import (
    SETUP_TOKEN_ENV,
    SETUP_TOKEN_HEADER,
    SETUP_TOKEN_MIN_LENGTH,
    evaluate_setup_access,
    is_direct_loopback_request,
)
from manga_translator.server.core.username_policy import (
    InvalidUsernameError,
    validate_username,
)

logger = logging.getLogger('manga_translator.server.routes.auth')

router = APIRouter(prefix="/auth", tags=["authentication"])

LOGIN_WINDOW = timedelta(minutes=10)
LOGIN_IP_MAX_ATTEMPTS = 15
LOGIN_USER_MAX_ATTEMPTS = 8
REGISTER_WINDOW = timedelta(minutes=10)
REGISTER_IP_MAX_ATTEMPTS = 5
SETUP_WINDOW = timedelta(minutes=10)
SETUP_IP_MAX_DENIED_ATTEMPTS = 10

_auth_rate_limiter = SlidingWindowRateLimiter()

# Service instances (will be injected by middleware)
_account_service = None
_session_service = None
_audit_service = None


def init_auth_services(account_service, session_service, audit_service):
    """Initialize service instances for authentication routes"""
    global _account_service, _session_service, _audit_service
    _account_service = account_service
    _session_service = session_service
    _audit_service = audit_service


def _client_ip(req: Request) -> str:
    return req.client.host if req.client else "unknown"


def _normalized_username(username: Optional[str]) -> str:
    return (username or "").strip().lower()


def _raise_rate_limit(detail: str, retry_after: int) -> None:
    raise HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(retry_after)},
    )


def _check_login_rate_limit(client_ip: str, username: str) -> None:
    keys = (
        (f"auth:login:ip:{client_ip}", LOGIN_IP_MAX_ATTEMPTS),
        (f"auth:login:user:{_normalized_username(username)}", LOGIN_USER_MAX_ATTEMPTS),
    )
    for key, max_attempts in keys:
        allowed, retry_after = _auth_rate_limiter.check(key, max_attempts, LOGIN_WINDOW)
        if not allowed:
            _raise_rate_limit("Too many login attempts; try again later", retry_after)


def _record_login_failure(client_ip: str, username: str) -> None:
    _auth_rate_limiter.record(f"auth:login:ip:{client_ip}", LOGIN_IP_MAX_ATTEMPTS, LOGIN_WINDOW)
    _auth_rate_limiter.record(
        f"auth:login:user:{_normalized_username(username)}",
        LOGIN_USER_MAX_ATTEMPTS,
        LOGIN_WINDOW,
    )


def _clear_login_failure(username: str) -> None:
    _auth_rate_limiter.reset(f"auth:login:user:{_normalized_username(username)}")


def _consume_registration_attempt(client_ip: str) -> None:
    key = f"auth:register:ip:{client_ip}"
    allowed, retry_after = _auth_rate_limiter.check(key, REGISTER_IP_MAX_ATTEMPTS, REGISTER_WINDOW)
    if not allowed:
        _raise_rate_limit("Too many registration attempts; try again later", retry_after)
    _auth_rate_limiter.record(key, REGISTER_IP_MAX_ATTEMPTS, REGISTER_WINDOW)


def _require_valid_new_username(username: str) -> None:
    """Reject usernames outside the allowlist with a clear 400 (new accounts only)."""
    try:
        validate_username(username)
    except InvalidUsernameError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _require_setup_access(req: Request, body_token: Optional[str]) -> str:
    """
    Allow the initial admin setup only from a direct loopback client or with
    the setup token.  Returns how access was granted ("loopback" or "token").

    The loopback decision never uses X-Forwarded-For style headers: their
    mere presence marks the request as proxied (i.e. not local).
    """
    client_ip = _client_ip(req)
    key = f"auth:setup:denied:ip:{client_ip}"
    allowed, retry_after = _auth_rate_limiter.check(key, SETUP_IP_MAX_DENIED_ATTEMPTS, SETUP_WINDOW)
    if not allowed:
        _raise_rate_limit("Too many initial setup attempts; try again later", retry_after)

    supplied_token = req.headers.get(SETUP_TOKEN_HEADER) or body_token
    granted, reason = evaluate_setup_access(req, supplied_token)
    if granted:
        return reason

    _auth_rate_limiter.record(key, SETUP_IP_MAX_DENIED_ATTEMPTS, SETUP_WINDOW)
    logger.warning(f"Initial setup denied for {client_ip}: {reason}")
    if _audit_service:
        _audit_service.log_event(
            event_type="initial_setup",
            username="",
            ip_address=client_ip,
            details={"reason": reason},
            result="failure"
        )
    if reason == "token_invalid":
        detail = "Invalid setup token"
    elif reason == "token_required":
        detail = (
            "Initial setup is only allowed from the server machine itself (127.0.0.1), "
            f"or with the setup token from {SETUP_TOKEN_ENV}"
        )
    else:
        detail = (
            "Initial setup is only allowed from the server machine itself (127.0.0.1); "
            f"for remote setup, start the server with the environment variable {SETUP_TOKEN_ENV} set "
            f"(at least {SETUP_TOKEN_MIN_LENGTH} characters) and enter it here"
        )
    raise HTTPException(status_code=403, detail=detail)


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    """Change password request model"""
    old_password: str
    new_password: str


class RegisterRequest(BaseModel):
    """User registration request model"""
    username: str = Field(..., min_length=2, max_length=50, description="User name")
    password: str = Field(..., min_length=6, description="Password (at least 6 characters)")


class InitialSetupRequest(BaseModel):
    """Initial admin setup request model"""
    username: str = Field(..., min_length=2, max_length=50, description="Administrator user name")
    password: str = Field(..., min_length=6, description="Administrator password (at least 6 characters)")
    setup_token: Optional[str] = Field(
        None,
        max_length=512,
        description="Setup token (MANGA_TRANSLATOR_SETUP_TOKEN); required for non-loopback clients"
    )


class LoginResponse(BaseModel):
    """Login response model"""
    success: bool
    token: Optional[str] = None
    message: Optional[str] = None
    user: Optional[dict] = None
    must_change_password: bool = False


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, req: Request):
    """
    User login endpoint
    
    Validates username and password, creates a session, and returns a session token.
    
    **Requirements: 3.1, 3.2, 3.3, 3.4**
    """
    if not _account_service or not _session_service or not _audit_service:
        raise HTTPException(500, detail="Services not initialized")
    
    # Get client IP
    client_ip = _client_ip(req)
    user_agent = req.headers.get("user-agent", "unknown")
    _check_login_rate_limit(client_ip, request.username)
    
    # Verify credentials
    if not _account_service.verify_password(request.username, request.password):
        _record_login_failure(client_ip, request.username)
        # Log failed login attempt
        _audit_service.log_event(
            event_type="login",
            username=request.username,
            ip_address=client_ip,
            details={"reason": "invalid_credentials"},
            result="failure"
        )
        
        return LoginResponse(
            success=False,
            message="Wrong user name or password"
        )
    
    # Get user account
    user = _account_service.get_user(request.username)
    if not user:
        _record_login_failure(client_ip, request.username)
        return LoginResponse(
            success=False,
            message="The user does not exist"
        )
    
    if not user.is_active:
        _record_login_failure(client_ip, request.username)
        return LoginResponse(
            success=False,
            message="The account has been disabled"
        )
    
    # Create session
    session = _session_service.create_session(
        username=user.username,
        role=user.role,
        ip_address=client_ip,
        user_agent=user_agent
    )
    
    # Log successful login
    _audit_service.log_event(
        event_type="login",
        username=request.username,
        ip_address=client_ip,
        details={"session_id": session.session_id},
        result="success"
    )
    _clear_login_failure(request.username)
    
    return LoginResponse(
        success=True,
        token=session.token,
        user={
            "username": user.username,
            "role": user.role,
            "permissions": user.permissions.to_dict()
        },
        must_change_password=user.must_change_password
    )


@router.post("/logout")
async def logout(req: Request):
    """
    User logout endpoint
    
    Terminates the current session.
    
    **Requirements: 3.5**
    """
    if not _session_service or not _audit_service:
        raise HTTPException(500, detail="Services not initialized")
    
    # Get token from header
    token = req.headers.get("x-session-token")
    if not token:
        raise HTTPException(401, detail="No session token provided")
    
    # Get session
    session = _session_service.get_session(token)
    if not session:
        raise HTTPException(401, detail="Invalid session token")
    
    # Terminate session
    _session_service.terminate_session(session.session_id)
    
    # Log logout
    _audit_service.log_event(
        event_type="logout",
        username=session.username,
        ip_address=req.client.host if req.client else "unknown",
        details={"session_id": session.session_id},
        result="success"
    )
    
    return {"success": True, "message": "Logged out successfully"}


@router.post("/change-password")
async def change_password(request: ChangePasswordRequest, req: Request):
    """
    Change password endpoint
    
    Allows users to change their password.
    
    **Requirements: 1.6**
    """
    if not _account_service or not _session_service or not _audit_service:
        raise HTTPException(500, detail="Services not initialized")
    
    # Get token from header
    token = req.headers.get("x-session-token")
    if not token:
        raise HTTPException(401, detail="No session token provided")
    
    # Get session
    session = _session_service.get_session(token)
    if not session:
        raise HTTPException(401, detail="Invalid session token")
    
    # Verify old password
    if not _account_service.verify_password(session.username, request.old_password):
        return {"success": False, "message": "The old password is wrong"}
    
    # Change password
    success = _account_service.change_password(session.username, request.new_password)
    
    if success:
        # Log password change
        _audit_service.log_event(
            event_type="password_change",
            username=session.username,
            ip_address=req.client.host if req.client else "unknown",
            details={},
            result="success"
        )
        
        # Clear must_change_password flag if set
        _account_service.update_user(session.username, {"must_change_password": False})
        
        return {"success": True, "message": "Password changed successfully"}
    else:
        return {"success": False, "message": "Changing the password failed"}


@router.get("/check")
async def check_session(req: Request):
    """
    Check session status endpoint
    
    Verifies if the current session is valid.
    
    **Requirements: 3.4, 4.5**
    """
    if not _session_service:
        raise HTTPException(500, detail="Services not initialized")
    
    # Get token from header
    token = req.headers.get("x-session-token")
    if not token:
        return {"valid": False, "message": "No session token provided"}
    
    # Verify token
    session = _session_service.verify_token(token)
    if not session:
        return {"valid": False, "message": "Invalid or expired session"}
    
    # Update activity
    _session_service.update_activity(token)
    
    return {
        "valid": True,
        "user": {
            "username": session.username,
            "role": session.role
        }
    }


@router.get("/status")
async def get_auth_status(req: Request):
    """
    Get the status of the authentication system

    Returns:
    - need_setup: whether the initial setup is needed (there are no users at all)
    - registration_enabled: whether user registration is enabled
    - setup_token_required: whether the current client needs the setup token to complete the initial setup (true for access from another machine)
    """
    if not _account_service:
        raise HTTPException(500, detail="Services not initialized")
    
    # Check whether there is any user
    users = _account_service.list_users()
    need_setup = len(users) == 0
    
    # Get the registration settings
    registration_config = admin_settings.get('registration', {})
    registration_enabled = registration_config.get('enabled', False)
    
    return {
        "need_setup": need_setup,
        "registration_enabled": registration_enabled,
        "setup_token_required": need_setup and not is_direct_loopback_request(req)
    }


@router.post("/setup")
async def initial_setup(request: InitialSetupRequest, req: Request):
    """
    Initial setup endpoint - creates the first administrator account

    This endpoint can only be called while the system has no users at all.
    """
    if not _account_service or not _session_service or not _audit_service:
        raise HTTPException(500, detail="Services not initialized")
    
    # Check whether a user already exists
    users = _account_service.list_users()
    if len(users) > 0:
        raise HTTPException(
            status_code=400,
            detail="The system is already initialised and cannot be set up again"
        )

    # Only the operator (loopback) or a holder of the setup token may create
    # the first administrator.
    setup_access = _require_setup_access(req, request.setup_token)
    _require_valid_new_username(request.username)
    
    # Validate the user name
    if not request.username or len(request.username) < 2:
        raise HTTPException(
            status_code=400,
            detail="The user name needs at least 2 characters"
        )
    
    # Validate the password
    if not request.password or len(request.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="The password needs at least 6 characters"
        )
    
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    
    try:
        # Create the administrator account
        from manga_translator.server.core.models import UserPermissions
        
        admin_permissions = UserPermissions(
            allowed_translators=["*"],
            allowed_parameters=["*"],
            max_concurrent_tasks=10,
            daily_quota=-1,
            can_upload_files=True,
            can_delete_files=True
        )
        
        account = _account_service.create_user(
            username=request.username,
            password=request.password,
            role='admin',
            group='admin',
            permissions=admin_permissions
        )
        
        # Create the session
        session = _session_service.create_session(
            username=account.username,
            role=account.role,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Write the audit log
        _audit_service.log_event(
            event_type="initial_setup",
            username=request.username,
            ip_address=client_ip,
            details={"action": "create_first_admin", "access": setup_access},
            result="success"
        )
        
        logger.info(f"Initial setup completed: created admin user '{request.username}'")
        
        return {
            "success": True,
            "message": "Initial setup completed",
            "token": session.token,
            "user": {
                "username": account.username,
                "role": account.role,
                "permissions": account.permissions.to_dict()
            }
        }
    
    except ValueError as e:
        logger.warning(f"Initial setup failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Initial setup error: {e}")
        raise HTTPException(status_code=500, detail="Initial setup failed")


@router.post("/register")
async def register_user(request: RegisterRequest, req: Request):
    """
    User registration endpoint

    It can only be used when the administrator has enabled registration.
    """
    if not _account_service or not _session_service or not _audit_service:
        raise HTTPException(500, detail="Services not initialized")

    client_ip = _client_ip(req)
    _consume_registration_attempt(client_ip)
    
    # Check whether registration is open
    registration_config = admin_settings.get('registration', {})
    if not registration_config.get('enabled', False):
        raise HTTPException(
            status_code=403,
            detail="Registration is not enabled; contact the administrator"
        )
    
    # Validate the user name
    if not request.username or len(request.username) < 2:
        raise HTTPException(
            status_code=400,
            detail="The user name needs at least 2 characters"
        )
    
    # Validate the password
    if not request.password or len(request.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="The password needs at least 6 characters"
        )
    
    _require_valid_new_username(request.username)

    # Check whether the user name already exists
    existing_user = _account_service.get_user(request.username)
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="The user name already exists"
        )
    
    user_agent = req.headers.get("user-agent", "unknown")
    
    try:
        # Get the default user group
        default_group = registration_config.get('default_group', 'default')
        
        # Create the ordinary user account
        account = _account_service.create_user(
            username=request.username,
            password=request.password,
            role='user',
            group=default_group
        )
        
        # Create the session
        session = _session_service.create_session(
            username=account.username,
            role=account.role,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Write the audit log
        _audit_service.log_event(
            event_type="register",
            username=request.username,
            ip_address=client_ip,
            details={"group": default_group},
            result="success"
        )
        
        logger.info(f"New user registered: '{request.username}' (group: {default_group})")
        
        return {
            "success": True,
            "message": "Registered successfully",
            "token": session.token,
            "user": {
                "username": account.username,
                "role": account.role,
                "permissions": account.permissions.to_dict()
            }
        }
    
    except ValueError as e:
        logger.warning(f"Registration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
