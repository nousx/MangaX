"""Locate the Codex CLI and inspect or start its account sign-in.

Kept free of heavy imports so the desktop UI can use it without loading the
translation backends.

The app runs this executable, so where it comes from matters:

* Without a configured path, Codex is looked up in absolute PATH folders and
  its default install location. The app folder is never searched.
* A configured path lives in the settings file, which users share and import.
  It is therefore not trusted on its own: the user has to approve the exact
  file once, and that approval is stored per Windows account, outside the
  shareable settings.
"""

import functools
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

CODEX_INSTALL_URL = "https://github.com/openai/codex#quickstart"

STATE_MISSING = "missing"
STATE_NEEDS_APPROVAL = "needs_approval"
STATE_SIGNED_OUT = "signed_out"
STATE_SIGNED_IN = "signed_in"
STATE_ERROR = "error"

# App API settings must not override the user's saved Codex account or
# redirect account authentication to an app's custom API endpoint.
_OVERRIDING_ENV = ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL")

_EXECUTABLE_NAMES = ("codex.exe",) if os.name == "nt" else ("codex",)


class CodexPathNotApproved(PermissionError):
    """The configured Codex path has not been approved by this user."""

    def __init__(self, path: str):
        super().__init__("The custom Codex CLI path has not been approved. Approve it in Translation settings.")
        self.path = path


@dataclass(frozen=True)
class CodexStatus:
    state: str
    detail: str = ""
    executable: str = ""


@functools.lru_cache(maxsize=1)
def _trusted_environment() -> dict[str, str]:
    """Return this user's environment as the operating system defines it.

    The process environment is not trusted here: the app loads variables from
    its shareable .env file into os.environ, so PATH, LOCALAPPDATA or
    USERPROFILE could point wherever an imported file wants. Windows rebuilds
    the environment from the registry for the account that owns the process;
    other systems get the account's home folder and the standard folders.
    """
    if os.name != "nt":
        import pwd

        home = pwd.getpwuid(os.getuid()).pw_dir
        folders = ["/usr/local/bin", "/usr/bin", "/bin", "/opt/homebrew/bin", f"{home}/.local/bin"]
        return {"PATH": os.pathsep.join(folders), "LOCALAPPDATA": f"{home}/.config"}

    import ctypes
    from ctypes import wintypes

    token_access = 0x0008 | 0x0002 | 0x0004  # TOKEN_QUERY | TOKEN_DUPLICATE | TOKEN_IMPERSONATE
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    userenv = ctypes.WinDLL("userenv", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    userenv.CreateEnvironmentBlock.argtypes = [ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.BOOL]
    userenv.DestroyEnvironmentBlock.argtypes = [ctypes.c_void_p]

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), token_access, ctypes.byref(token)):
        return {}
    try:
        block = ctypes.c_void_p()
        # bInherit=False: do not merge in this process's own environment.
        if not userenv.CreateEnvironmentBlock(ctypes.byref(block), token, False):
            return {}
        try:
            result = {}
            address = block.value
            while True:
                entry = ctypes.wstring_at(address)
                if not entry:
                    break
                address += (len(entry) + 1) * ctypes.sizeof(ctypes.c_wchar)
                name, separator, value = entry.partition("=")
                if name and separator:
                    result[name.upper()] = value
            return result
        finally:
            userenv.DestroyEnvironmentBlock(block)
    finally:
        kernel32.CloseHandle(token)


def _local_app_data() -> Path | None:
    value = _trusted_environment().get("LOCALAPPDATA")
    return Path(value) if value and Path(value).is_absolute() else None


def _approval_file(filename: str = "approved_codex_cli.txt") -> Path | None:
    base = _local_app_data()
    return base / "MangaX" / filename if base else None


def _is_network_path(path: str) -> bool:
    return path.replace("/", "\\").startswith("\\\\")


def _validated_configured_path(
    cli_path: str,
    names: tuple[str, ...] = _EXECUTABLE_NAMES,
    label: str = "Codex",
) -> str:
    """Return the resolved configured executable, or raise FileNotFoundError."""
    if _is_network_path(cli_path):
        raise FileNotFoundError(f"{label} CLI path must be on this computer, not a network location.")
    executable = Path(cli_path).expanduser()
    if not executable.is_absolute():
        raise FileNotFoundError(f"{label} CLI path must be a full path. Check Translation settings.")
    if not executable.is_file():
        raise FileNotFoundError(f"{label} CLI path does not exist. Check Translation settings.")
    if executable.name.lower() not in names:
        raise FileNotFoundError(
            f"{label} CLI path must point to {names[0]}. Check Translation settings."
        )
    resolved = str(executable.resolve())
    # A mapped drive or a link can still lead to a network share.
    if _is_network_path(resolved):
        raise FileNotFoundError(f"{label} CLI path must be on this computer, not a network location.")
    return resolved


def _is_approved(resolved: str, target: Path | None = None) -> bool:
    target = target or _approval_file()
    if target is None:
        return False
    try:
        approved = target.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return bool(approved) and os.path.normcase(approved) == os.path.normcase(resolved)


def _record_approval(resolved: str, target: Path | None) -> str:
    if target is None:
        raise OSError("The per-user data folder is unavailable, so the path cannot be approved.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(resolved, encoding="utf-8")
    return resolved


def approve_codex_cli(cli_path: str) -> str:
    """Record that this user allows the app to run the configured Codex path."""
    return _record_approval(_validated_configured_path(cli_path), _approval_file())


def _search_path(names: tuple[str, ...] = _EXECUTABLE_NAMES) -> str | None:
    """Look for Codex in the folders of the account's own PATH.

    The process PATH is not used because the shareable .env file can replace
    it. shutil.which() is not used because on Windows it also searches the
    current directory, which is the app folder: a planted codex.exe there
    must never be picked up. Network folders are skipped as well.
    """
    here = Path.cwd().resolve()
    for entry in _trusted_environment().get("PATH", "").split(os.pathsep):
        folder = Path(entry.strip('"'))
        if not entry or not folder.is_absolute() or _is_network_path(entry.strip('"')):
            continue
        try:
            resolved = folder.resolve()
        except OSError:
            continue
        if resolved == here or _is_network_path(str(resolved)):
            continue
        for name in names:
            candidate = folder / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def find_codex_cli(cli_path: str = "") -> str | None:
    """Return the Codex executable, or None when it is not installed.

    Raises FileNotFoundError for an unusable configured path, so a typo is not
    silently replaced by another installation, and CodexPathNotApproved for a
    configured path this user has not approved yet.
    """
    if cli_path:
        resolved = _validated_configured_path(cli_path)
        if not _is_approved(resolved):
            raise CodexPathNotApproved(resolved)
        return resolved
    executable = _search_path()
    if executable:
        return executable
    base = _local_app_data() if os.name == "nt" else None
    if base:
        candidate = base / "Programs/OpenAI/Codex/bin/codex.exe"
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
    except CodexPathNotApproved as exc:
        return CodexStatus(STATE_NEEDS_APPROVAL, exc.path)
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
    except (FileNotFoundError, PermissionError):
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
