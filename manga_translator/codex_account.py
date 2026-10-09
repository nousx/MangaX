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


def _local_app_data() -> Path | None:
    """Return the per-user data folder as the operating system reports it.

    Environment variables such as LOCALAPPDATA are deliberately not used: the
    app loads variables from its shareable .env file into the process, so they
    could redirect this to a location someone else controls.
    """
    if os.name != "nt":
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".config"
    import ctypes
    from ctypes import wintypes

    class _Guid(ctypes.Structure):
        _fields_ = [
            ("data1", wintypes.DWORD),
            ("data2", wintypes.WORD),
            ("data3", wintypes.WORD),
            ("data4", ctypes.c_ubyte * 8),
        ]

    # FOLDERID_LocalAppData {F1B32785-6FBA-4FCF-9D55-7B8E7F157091}
    folder_id = _Guid(0xF1B32785, 0x6FBA, 0x4FCF,
                      (ctypes.c_ubyte * 8)(0x9D, 0x55, 0x7B, 0x8E, 0x7F, 0x15, 0x70, 0x91))
    buffer = ctypes.c_void_p()
    shell32 = ctypes.WinDLL("shell32")
    ole32 = ctypes.WinDLL("ole32")
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    result = shell32.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None, ctypes.byref(buffer))
    try:
        if result != 0 or not buffer.value:
            return None
        return Path(ctypes.wstring_at(buffer.value))
    finally:
        ole32.CoTaskMemFree(buffer)


def _approval_file() -> Path | None:
    base = _local_app_data()
    return base / "MangaX" / "approved_codex_cli.txt" if base else None


def _is_network_path(path: str) -> bool:
    return path.replace("/", "\\").startswith("\\\\")


def _validated_configured_path(cli_path: str) -> str:
    """Return the resolved configured executable, or raise FileNotFoundError."""
    if _is_network_path(cli_path):
        raise FileNotFoundError("Codex CLI path must be on this computer, not a network location.")
    executable = Path(cli_path).expanduser()
    if not executable.is_absolute():
        raise FileNotFoundError("Codex CLI path must be a full path. Check Translation settings.")
    if not executable.is_file():
        raise FileNotFoundError("Codex CLI path does not exist. Check Translation settings.")
    if executable.name.lower() not in _EXECUTABLE_NAMES:
        raise FileNotFoundError(
            f"Codex CLI path must point to {_EXECUTABLE_NAMES[0]}. Check Translation settings."
        )
    resolved = str(executable.resolve())
    # A mapped drive or a link can still lead to a network share.
    if _is_network_path(resolved):
        raise FileNotFoundError("Codex CLI path must be on this computer, not a network location.")
    return resolved


def _is_approved(resolved: str) -> bool:
    target = _approval_file()
    if target is None:
        return False
    try:
        approved = target.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return bool(approved) and os.path.normcase(approved) == os.path.normcase(resolved)


def approve_codex_cli(cli_path: str) -> str:
    """Record that this user allows the app to run the configured Codex path."""
    resolved = _validated_configured_path(cli_path)
    target = _approval_file()
    if target is None:
        raise OSError("The per-user data folder is unavailable, so the path cannot be approved.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(resolved, encoding="utf-8")
    return resolved


def _search_path() -> str | None:
    """Look for Codex in absolute PATH folders only.

    shutil.which() on Windows also searches the current directory, which is
    the app folder: a planted codex.exe there must never be picked up.
    """
    here = Path.cwd().resolve()
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        folder = Path(entry.strip('"'))
        # PATH can be overridden from the shareable .env file, so a network
        # folder in it must not be able to supply the executable.
        if not entry or not folder.is_absolute() or _is_network_path(entry.strip('"')):
            continue
        try:
            resolved = folder.resolve()
        except OSError:
            continue
        if resolved == here or _is_network_path(str(resolved)):
            continue
        for name in _EXECUTABLE_NAMES:
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
