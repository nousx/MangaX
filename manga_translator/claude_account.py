"""Locate the Claude Code CLI and inspect or start its account sign-in.

The same rules as for Codex apply (see ``codex_account``): the executable is
looked up through the account's own environment, never in the app folder,
and a path taken from the shareable settings file has to be approved by the
user before the app runs it.
"""

import json
import os
import subprocess
from pathlib import Path

from . import codex_account
from .codex_account import (
    STATE_ERROR,
    STATE_MISSING,
    STATE_NEEDS_APPROVAL,
    STATE_SIGNED_IN,
    STATE_SIGNED_OUT,
    CodexPathNotApproved,
    CodexStatus,
)

CLAUDE_INSTALL_URL = "https://claude.com/claude-code"

# Only the native executable is run. The npm launcher (claude.cmd) is a batch
# file, and running one with arguments built from manga text is not safe.
_EXECUTABLE_NAMES = ("claude.exe",) if os.name == "nt" else ("claude",)
_APPROVAL_FILE = "approved_claude_cli.txt"

# These decide which account, endpoint and credential store the CLI uses. The
# app loads a shareable .env file into its own environment, so each one is
# taken from the account's real environment instead, or dropped.
_ACCOUNT_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CONFIG_DIR",
)


def _approval_file() -> Path | None:
    return codex_account._approval_file(_APPROVAL_FILE)


def approve_claude_cli(cli_path: str) -> str:
    """Record that this user allows the app to run the configured Claude path."""
    resolved = codex_account._validated_configured_path(cli_path, _EXECUTABLE_NAMES, "Claude")
    return codex_account._record_approval(resolved, _approval_file())


def find_claude_cli(cli_path: str = "") -> str | None:
    """Return the Claude executable, or None when it is not installed.

    Raises FileNotFoundError for an unusable configured path and
    CodexPathNotApproved for a configured path this user has not approved.
    """
    if cli_path:
        resolved = codex_account._validated_configured_path(cli_path, _EXECUTABLE_NAMES, "Claude")
        if not codex_account._is_approved(resolved, _approval_file()):
            raise CodexPathNotApproved(resolved)
        return resolved
    executable = codex_account._search_path(_EXECUTABLE_NAMES)
    if executable:
        return executable
    # Default location of the native installer.
    home = codex_account._trusted_environment().get("USERPROFILE" if os.name == "nt" else "HOME")
    if home and Path(home).is_absolute():
        candidate = Path(home) / ".local" / "bin" / _EXECUTABLE_NAMES[0]
        if candidate.is_file():
            return str(candidate)
    return None


def claude_child_env() -> dict[str, str]:
    """Environment for Claude child processes, with account settings the app cannot override."""
    env = os.environ.copy()
    trusted = codex_account._trusted_environment()
    for name in _ACCOUNT_ENV:
        env.pop(name, None)
        if name in trusted:
            env[name] = trusted[name]
    return env


def claude_status(cli_path: str = "", timeout: float = 20.0) -> CodexStatus:
    """Report whether Claude Code is installed and signed in. Never raises."""
    try:
        executable = find_claude_cli(cli_path)
    except CodexPathNotApproved as exc:
        return CodexStatus(STATE_NEEDS_APPROVAL, exc.path)
    except FileNotFoundError as exc:
        return CodexStatus(STATE_MISSING, str(exc))
    if not executable:
        return CodexStatus(STATE_MISSING)
    try:
        result = subprocess.run(
            [executable, "auth", "status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=claude_child_env(),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CodexStatus(STATE_ERROR, str(exc), executable)
    try:
        info = json.loads(result.stdout)
    except ValueError:
        info = {}
    if result.returncode == 0 and isinstance(info, dict) and info.get("loggedIn") is True:
        # The plan is enough to recognise the account; the address stays private.
        plan = str(info.get("subscriptionType") or info.get("authMethod") or "").strip()
        return CodexStatus(STATE_SIGNED_IN, plan[:60], executable)
    return CodexStatus(STATE_SIGNED_OUT, "", executable)


def start_claude_login(cli_path: str = "") -> bool:
    """Open `claude auth login` in its own console window so the user can sign in."""
    try:
        executable = find_claude_cli(cli_path)
    except (FileNotFoundError, PermissionError):
        return False
    if not executable:
        return False
    try:
        subprocess.Popen(
            [executable, "auth", "login"],
            env=claude_child_env(),
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        )
    except OSError:
        return False
    return True
