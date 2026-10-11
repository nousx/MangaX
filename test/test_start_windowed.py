import _bootstrap  # noqa: F401, I001

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_start_windowed():
    # "packaging" is also the name of a PyPI library, so load the file by path
    spec = importlib.util.spec_from_file_location(
        "start_windowed_test", ROOT / "packaging" / "start_windowed.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


start_windowed = _load_start_windowed()


def test_should_pick_python_exe_next_to_pythonw():
    python = start_windowed.console_python(r"C:\MangaX\packaging\python\pythonw.exe")

    assert python == Path(r"C:\MangaX\packaging\python\python.exe")


def test_should_set_the_runtime_flags_of_the_batch_launcher(tmp_path):
    env = start_windowed.build_environment(tmp_path, {"PATH": "base", "PYTHONUTF8": "0"})

    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PATH"] == "base"


def test_should_put_bundled_git_first_on_path_when_present(tmp_path):
    git_dir = tmp_path / "PortableGit" / "cmd"
    git_dir.mkdir(parents=True)
    (git_dir / "git.exe").write_bytes(b"")

    env = start_windowed.build_environment(tmp_path, {"PATH": "base"})

    assert env["PATH"].split(os.pathsep) == [str(git_dir), "base"]


def test_should_not_change_the_given_environment(tmp_path):
    base = {"PATH": "base"}

    start_windowed.build_environment(tmp_path, base)

    assert base == {"PATH": "base"}


def test_should_name_the_log_folder_and_console_launcher_when_run_fails(tmp_path):
    message = start_windowed.failure_message(1, tmp_path)

    assert "exit code 1." in message
    assert str(tmp_path / "result") in message
    assert "Win-Start-Console.bat" in message


def test_should_show_crash_codes_in_hexadecimal(tmp_path):
    message = start_windowed.failure_message(3221225477, tmp_path)

    assert "0xC0000005" in message


class _Process:
    def __init__(self, exit_code):
        self._exit_code = exit_code

    def wait(self):
        return self._exit_code


@pytest.fixture
def launch(monkeypatch, tmp_path):
    python_dir = tmp_path / "python"
    python_dir.mkdir()
    (python_dir / "python.exe").write_bytes(b"")
    state = {"errors": [], "exit_code": 0}

    def fake_popen(args, **kwargs):
        state["args"] = args
        state["kwargs"] = kwargs
        return _Process(state["exit_code"])

    monkeypatch.setattr(start_windowed.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(start_windowed.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(start_windowed, "show_error", state["errors"].append)
    monkeypatch.setattr(sys, "executable", str(python_dir / "pythonw.exe"))
    state["python"] = python_dir / "python.exe"
    return state


def test_should_start_the_app_with_python_exe_and_no_console_window(launch):
    assert start_windowed.main() == 0

    assert launch["args"] == [str(launch["python"]), str(start_windowed.ENTRY)]
    assert launch["kwargs"]["creationflags"] == 0x08000000
    assert launch["kwargs"]["cwd"] == start_windowed.ROOT
    assert launch["errors"] == []


def test_should_report_a_failed_run_and_return_its_exit_code(launch):
    launch["exit_code"] = 3

    assert start_windowed.main() == 3

    assert len(launch["errors"]) == 1
    assert "exit code 3." in launch["errors"][0]


def test_should_report_missing_python_without_starting_anything(launch):
    launch["python"].unlink()

    assert start_windowed.main() == 1

    assert "args" not in launch
    assert len(launch["errors"]) == 1

