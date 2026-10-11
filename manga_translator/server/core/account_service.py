"""
Account management service (AccountService)

Manages creating, looking up, updating and deleting user accounts.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import bcrypt

from manga_translator.server.core.models import UserAccount, UserPermissions
from manga_translator.server.core.persistence import atomic_write_json, load_json
from manga_translator.server.core.username_policy import validate_username

logger = logging.getLogger(__name__)


class AccountService:
    """Account management service"""
    
    def __init__(self, accounts_file: str = "manga_translator/server/data/accounts.json"):
        """
        Initialise the account management service

        Args:
            accounts_file: path of the account storage file
        """
        self.accounts_file = accounts_file
        self.accounts: Dict[str, UserAccount] = {}
        self._load_accounts()
    
    def create_user(
        self,
        username: str,
        password: str,
        role: str,
        group: str = "default",
        permissions: Optional[UserPermissions] = None
    ) -> UserAccount:
        """
        Create a new user

        Args:
            username: the user name
            password: the password (plain text)
            role: the role ('admin' or 'user')
            group: name of the user group ('default' by default)
            permissions: the user permissions (the default permissions are used when None)

        Returns:
            UserAccount: the user account that was created

        Raises:
            ValueError: when the user name is not valid, the user name already exists, the password is too weak or the role is invalid
        """
        # Allowlist check for new accounts only; existing accounts are loaded
        # as-is by _load_accounts() so legacy usernames can still log in.
        validate_username(username)

        # Check that the user name is unique
        if username in self.accounts:
            raise ValueError(f"User name '{username}' already exists")
        
        # Check the password strength (at least 6 characters)
        if len(password) < 6:
            raise ValueError("The password must be at least 6 characters long")
        
        # Validate the role
        if role not in ['admin', 'user']:
            raise ValueError(f"Invalid role: {role}")
        
        # Use the default permissions (when none are given)
        if permissions is None:
            if role == 'admin':
                permissions = UserPermissions(
                    allowed_translators=["*"],
                    allowed_ocr=["*"],
                    allowed_colorizers=["*"],
                    allowed_renderers=["*"],
                    allowed_workflows=["*"],
                    allowed_parameters=["*"],
                    max_concurrent_tasks=10,
                    daily_quota=-1,
                    can_upload_files=True,
                    can_delete_files=True
                )
            else:
                # Ordinary users inherit the group configuration by default
                # Empty allowed_translators/allowed_parameters means "inherit from the group"
                # User-level settings can override the group (unlock with a whitelist or disable with a blacklist)
                permissions = UserPermissions(
                    allowed_translators=[],  # empty = inherit from the group
                    denied_translators=[],
                    allowed_ocr=[],          # empty = inherit from the group
                    denied_ocr=[],
                    allowed_colorizers=[],   # empty = inherit from the group
                    denied_colorizers=[],
                    allowed_renderers=[],    # empty = inherit from the group
                    denied_renderers=[],
                    allowed_workflows=[],    # empty = inherit from the group
                    denied_workflows=[],
                    allowed_parameters=[],   # empty = inherit from the group
                    denied_parameters=[],
                    max_concurrent_tasks=2,
                    daily_quota=100,
                    can_upload_files=True,
                    can_delete_files=False
                )
        
        # Hash the password
        password_hash = self._hash_password(password)
        
        # Create the user account
        account = UserAccount(
            username=username,
            password_hash=password_hash,
            role=role,
            group=group,
            permissions=permissions,
            created_at=datetime.now(timezone.utc),
            last_login=None,
            is_active=True,
            must_change_password=False
        )
        
        # Keep in memory
        self.accounts[username] = account
        
        # Persist
        self._save_accounts()
        
        logger.info(f"Created user: {username} (role: {role})")
        return account
    
    def get_user(self, username: str) -> Optional[UserAccount]:
        """
        Get the information of a user

        Args:
            username: the user name

        Returns:
            Optional[UserAccount]: the user account, or None when it does not exist
        """
        return self.accounts.get(username)
    
    def list_users(self) -> List[UserAccount]:
        """
        List all users

        Returns:
            List[UserAccount]: list of all user accounts
        """
        return list(self.accounts.values())
    
    def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        """
        Update the information of a user

        Args:
            username: the user name
            updates: dictionary of the fields to update

        Returns:
            bool: whether the update succeeded

        Raises:
            ValueError: when the user does not exist or an update field is invalid
        """
        account = self.accounts.get(username)
        if not account:
            raise ValueError(f"User '{username}' does not exist")
        
        # Fields that may be updated
        allowed_fields = {
            'role', 'group', 'permissions', 'is_active', 'must_change_password',
            'parameter_config', 'default_preset_id'
        }
        
        # Validate the fields to update
        for field in updates.keys():
            if field not in allowed_fields:
                raise ValueError(f"Updating this field is not allowed: {field}")
        
        # Apply the update
        if 'role' in updates:
            if updates['role'] not in ['admin', 'user']:
                raise ValueError(f"Invalid role: {updates['role']}")
            account.role = updates['role']
        
        if 'permissions' in updates:
            if isinstance(updates['permissions'], dict):
                account.permissions = UserPermissions.from_dict(updates['permissions'])
            elif isinstance(updates['permissions'], UserPermissions):
                account.permissions = updates['permissions']
            else:
                raise ValueError("permissions must be a dictionary or a UserPermissions object")
        
        if 'is_active' in updates:
            account.is_active = bool(updates['is_active'])
        
        if 'must_change_password' in updates:
            account.must_change_password = bool(updates['must_change_password'])

        if 'group' in updates:
            account.group = str(updates['group'])

        if 'parameter_config' in updates:
            account.parameter_config = updates['parameter_config']

        if 'default_preset_id' in updates:
            account.default_preset_id = updates['default_preset_id']
        
        # Persist
        self._save_accounts()
        
        logger.info(f"Updated user: {username}")
        return True
    
    def delete_user(self, username: str) -> bool:
        """
        Delete a user

        Args:
            username: the user name

        Returns:
            bool: whether the deletion succeeded

        Raises:
            ValueError: when the user does not exist
        """
        if username not in self.accounts:
            raise ValueError(f"User '{username}' does not exist")
        
        # Remove from memory
        del self.accounts[username]
        
        # Persist
        self._save_accounts()
        
        logger.info(f"Deleted user: {username}")
        return True
    
    def verify_password(self, username: str, password: str) -> bool:
        """
        Verify a password

        Args:
            username: the user name
            password: the password (plain text)

        Returns:
            bool: whether the password is correct
        """
        account = self.accounts.get(username)
        if not account:
            return False
        
        return self._verify_password(password, account.password_hash)
    
    def change_password(self, username: str, new_password: str) -> bool:
        """
        Change a password

        Args:
            username: the user name
            new_password: the new password (plain text)

        Returns:
            bool: whether the change succeeded

        Raises:
            ValueError: when the user does not exist or the password is too weak
        """
        account = self.accounts.get(username)
        if not account:
            raise ValueError(f"User '{username}' does not exist")
        
        # Check the password strength
        if len(new_password) < 6:
            raise ValueError("The password must be at least 6 characters long")
        
        # Hash the new password
        account.password_hash = self._hash_password(new_password)
        account.must_change_password = False
        
        # Persist
        self._save_accounts()
        
        logger.info(f"Changed password for user: {username}")
        return True
    
    def create_default_admin(
        self,
        username: str = "admin",
        password: str = "admin123"
    ) -> Optional[UserAccount]:
        """
        Create the default administrator account

        Args:
            username: the administrator user name
            password: the administrator password

        Returns:
            Optional[UserAccount]: the administrator account that was created, or None when it already exists
        """
        # When a user already exists, nothing is created
        if self.accounts:
            logger.info("Users already exist, skipping default admin creation")
            return None
        
        # Create the default administrator
        try:
            admin = self.create_user(
                username=username,
                password=password,
                role='admin',
                permissions=UserPermissions(
                    allowed_translators=["*"],
                    allowed_ocr=["*"],
                    allowed_colorizers=["*"],
                    allowed_renderers=["*"],
                    allowed_workflows=["*"],
                    allowed_parameters=["*"],
                    max_concurrent_tasks=10,
                    daily_quota=-1,
                    can_upload_files=True,
                    can_delete_files=True
                )
            )
            admin.must_change_password = True
            self._save_accounts()
            
            logger.warning(
                f"Created default admin account - "
                f"Username: {username} "
                f"(PLEASE CHANGE THIS PASSWORD IMMEDIATELY!)"
            )
            return admin
        except Exception as e:
            logger.error(f"Failed to create default admin: {e}")
            return None
    
    def update_last_login(self, username: str) -> bool:
        """
        Update the time of the last login

        Args:
            username: the user name

        Returns:
            bool: whether the update succeeded
        """
        account = self.accounts.get(username)
        if not account:
            return False
        
        account.last_login = datetime.now(timezone.utc)
        self._save_accounts()
        return True
    
    def _hash_password(self, password: str) -> str:
        """
        Hash a password

        Args:
            password: the plain-text password

        Returns:
            str: the hashed password
        """
        salt = bcrypt.gensalt()
        # bcrypt has a 72 byte limit, truncate if necessary
        password_bytes = password.encode('utf-8')[:72]
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify a password

        Args:
            password: the plain-text password
            password_hash: the hashed password

        Returns:
            bool: whether the password matches
        """
        try:
            # bcrypt has a 72 byte limit, truncate if necessary
            password_bytes = password.encode('utf-8')[:72]
            return bcrypt.checkpw(
                password_bytes,
                password_hash.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def _load_accounts(self) -> None:
        """Load the accounts from persistent storage"""
        try:
            data = load_json(self.accounts_file, default={'version': '1.0', 'accounts': []})
            
            accounts_data = data.get('accounts', [])
            self.accounts = {}
            
            for account_data in accounts_data:
                try:
                    account = UserAccount.from_dict(account_data)
                    self.accounts[account.username] = account
                except Exception as e:
                    logger.error(f"Failed to load account: {e}")
            
            logger.info(f"Loaded {len(self.accounts)} account(s)")
        except Exception as e:
            logger.error(f"Failed to load accounts: {e}")
            self.accounts = {}
    
    def _save_accounts(self) -> None:
        """Save the accounts to persistent storage"""
        try:
            data = {
                'version': '1.0',
                'accounts': [account.to_dict() for account in self.accounts.values()]
            }
            
            success = atomic_write_json(self.accounts_file, data, create_backup=True)
            if success:
                logger.debug(f"Saved {len(self.accounts)} account(s)")
            else:
                logger.error("Failed to save accounts")
        except Exception as e:
            logger.error(f"Failed to save accounts: {e}")
