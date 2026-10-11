"""Start the desktop app without a console window.

Run this file with pythonw.exe. The app itself still runs under python.exe,
but with a console that has no window: its output streams stay valid, and
the console tools it starts (git, the upscalers) share that hidden console
and do not open windows of their own.

Win-Start-Console.bat starts the app with a visible console instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = Path("desktop_qt_ui") / "main.py"
APP_NAME = "MangaX"

_MB_ICONERROR = 0x10


def console_python(executable: str) -> Path:
    """Return python.exe next to the given interpreter (usually pythonw.exe)."""
    return Path(executable).with_name("python.exe")


def build_environment(root: Path, base: dict[str, str]) -> dict[str, str]:
    """Return the environment Win-Start.bat gives the app."""
    env = dict(base)
    env["PYTHONUTF8"] = "1"
    # A stray torch in the per-user site-packages breaks the bundled runtime
    env["PYTHONNOUSERSITE"] = "1"
    git_dir = root / "PortableGit" / "cmd"
    if (git_dir / "git.exe").is_file():
        paths = env.get("PATH", "").split(os.pathsep)
        if str(git_dir) not in paths:
            env["PATH"] = os.pathsep.join([str(git_dir), *filter(None, paths)])
    return env


def failure_message(exit_code: int, root: Path) -> str:
    """Describe a failed run and where to look next."""
    code = str(exit_code)
    if not 0 <= exit_code <= 255:
        # Windows crash codes are easier to look up in hexadecimal
        code = f"{exit_code} (0x{exit_code & 0xFFFFFFFF:08X})"
    return (
        f"{APP_NAME} stopped with exit code {code}.\n\n"
        f"Log files are in:\n{root / 'result'}\n\n"
        "To see the error, start the program with Win-Start-Console.bat.\n"
        "If it keeps failing, run Win-Install-or-Update.bat and choose [1] Install."
    )


def show_error(text: str) -> None:
    if os.name != "nt":
        return
    import ctypes

    ctypes.windll.user32.MessageBoxW(None, text, APP_NAME, _MB_ICONERROR)


def main() -> int:
    python = console_python(sys.executable)
    if not python.is_file():
        show_error(f"python.exe was not found next to:\n{sys.executable}")
        return 1
    try:
        process = subprocess.Popen(
            [str(python), str(ENTRY)],
            cwd=ROOT,
            env=build_environment(ROOT, dict(os.environ)),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as error:
        show_error(f"{APP_NAME} could not be started:\n{error}")
        return 1
    exit_code = process.wait()
    if exit_code != 0:
        show_error(failure_message(exit_code, ROOT))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
