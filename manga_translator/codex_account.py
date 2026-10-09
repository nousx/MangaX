"""Locate the Codex CLI and inspect or start its account sign-in.

Kept free of heavy imports so the desktop UI can use it without loading the
translation backends.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

CODEX_INSTALL_URL = "https://github.com/openai/codex#quickstart"

STATE_MISSING = "missing"
STATE_SIGNED_OUT = "signed_out"
STATE_SIGNED_IN = "signed_in"
STATE_ERROR = "error"

# App API settings must not override the user's saved Codex account or
# redirect account authentication to an app's custom API endpoint.
_OVERRIDING_ENV = ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL")


@dataclass(frozen=True)
class CodexStatus:
    state: str
    detail: str = ""
    executable: str = ""


def find_codex_cli(cli_path: str = "") -> str | None:
    """Return the Codex executable, or None when it is not installed.

    A configured path that does not exist raises, so a typo is not silently
    replaced by another installation.
    """
    if cli_path:
        executable = Path(cli_path).expanduser()
        if not executable.is_file():
            raise FileNotFoundError("Codex CLI path does not exist. Check Translation settings.")
        return str(executable.resolve())
    executable = shutil.which("codex.exe" if os.name == "nt" else "codex")
    if executable:
        return executable
    if os.name == "nt":
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/OpenAI/Codex/bin/codex.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def codex_child_env() -> dict[str, str]:
    """Environment for Codex child processes, without API overrides."""
    env = os.environ.copy()
    for name in _OVERRIDING_ENV:
        env.pop(name, None)
    return env


def codex_status(cli_path: str = "", timeout: float = 15.0) -> CodexStatus:
    """Report whether Codex is installed and signed in. Never raises."""
    try:
        executable = find_codex_cli(cli_path)
    except FileNotFoundError as exc:
        return CodexStatus(STATE_MISSING, str(exc))
    if not executable:
        return CodexStatus(STATE_MISSING)
    try:
        result = subprocess.run(
            [executable, "login", "status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=codex_child_env(),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CodexStatus(STATE_ERROR, str(exc), executable)
    # Codex prints the status line on stderr.
    lines = (result.stdout + "\n" + result.stderr).strip().splitlines()
    detail = lines[0].strip() if lines else ""
    state = STATE_SIGNED_IN if result.returncode == 0 else STATE_SIGNED_OUT
    return CodexStatus(state, detail, executable)


def start_codex_login(cli_path: str = "") -> bool:
    """Open `codex login` in its own console window so the user can sign in."""
    try:
        executable = find_codex_cli(cli_path)
    except FileNotFoundError:
        return False
    if not executable:
        return False
    try:
        subprocess.Popen(
            [executable, "login"],
            env=codex_child_env(),
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        )
    except OSError:
        return False
    return True
