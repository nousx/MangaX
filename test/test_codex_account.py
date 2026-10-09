import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from manga_translator import codex_account
from manga_translator.codex_account import (
    STATE_ERROR,
    STATE_MISSING,
    STATE_NEEDS_APPROVAL,
    STATE_SIGNED_IN,
    STATE_SIGNED_OUT,
    CodexPathNotApproved,
    approve_codex_cli,
    codex_status,
    find_codex_cli,
    start_codex_login,
)


def completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(["codex"], returncode, stdout, stderr)


class CodexAccountTests(unittest.TestCase):
    def test_should_report_missing_when_codex_is_not_installed(self):
        with patch.object(codex_account, "find_codex_cli", return_value=None):
            self.assertEqual(codex_status().state, STATE_MISSING)

    def test_should_report_missing_when_configured_path_does_not_exist(self):
        status = codex_status(str(Path(tempfile.gettempdir()) / "no-such-codex.exe"))

        self.assertEqual(status.state, STATE_MISSING)
        self.assertIn("path does not exist", status.detail)

    def test_should_raise_when_configured_path_does_not_exist(self):
        with self.assertRaises(FileNotFoundError):
            find_codex_cli(str(Path(tempfile.gettempdir()) / "no-such-codex.exe"))

    def test_should_reject_configured_path_that_is_not_named_codex(self):
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory) / "other-tool.exe"
            other.write_bytes(b"")

            with self.assertRaises(FileNotFoundError):
                find_codex_cli(str(other))

    def test_should_reject_network_and_relative_configured_paths(self):
        for path in (r"\\server\share\codex.exe", "//server/share/codex.exe", "codex.exe"):
            with self.subTest(path=path), self.assertRaises(FileNotFoundError):
                find_codex_cli(path)

    def test_should_require_approval_before_using_a_configured_path(self):
        name = "codex.exe" if os.name == "nt" else "codex"
        with tempfile.TemporaryDirectory() as directory:
            configured = Path(directory) / name
            configured.write_bytes(b"")
            approval = Path(directory) / "state" / "approved.txt"
            with patch.object(codex_account, "_approval_file", return_value=approval), \
                    patch.object(codex_account.subprocess, "run") as run, \
                    patch.object(codex_account.subprocess, "Popen") as popen:
                with self.assertRaises(CodexPathNotApproved):
                    find_codex_cli(str(configured))
                status = codex_status(str(configured))
                started = start_codex_login(str(configured))

                self.assertEqual(status.state, STATE_NEEDS_APPROVAL)
                self.assertFalse(started)
                run.assert_not_called()
                popen.assert_not_called()

                approve_codex_cli(str(configured))

                self.assertEqual(find_codex_cli(str(configured)), str(configured.resolve()))

    def test_should_not_carry_approval_over_to_a_different_path(self):
        name = "codex.exe" if os.name == "nt" else "codex"
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a" / name
            second = Path(directory) / "b" / name
            for path in (first, second):
                path.parent.mkdir()
                path.write_bytes(b"")
            with patch.object(codex_account, "_approval_file", return_value=Path(directory) / "approved.txt"):
                approve_codex_cli(str(first))

                with self.assertRaises(CodexPathNotApproved):
                    find_codex_cli(str(second))

    def test_should_ignore_codex_in_the_current_directory(self):
        name = "codex.exe" if os.name == "nt" else "codex"
        with tempfile.TemporaryDirectory() as directory:
            planted = Path(directory) / name
            planted.write_bytes(b"")
            planted.chmod(0o755)
            with patch.dict(os.environ, {"PATH": os.pathsep.join([directory, ".", ""])}), \
                    patch.object(codex_account, "_local_app_data", return_value=Path(directory) / "none"), \
                    patch.object(codex_account.Path, "cwd", return_value=Path(directory)):
                self.assertIsNone(find_codex_cli())

    def test_should_ignore_network_folders_in_path(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"PATH": "//server/share"}), \
                patch.object(codex_account, "_local_app_data", return_value=Path(directory)), \
                patch.object(codex_account.os, "access", side_effect=AssertionError("network folder was probed")):
            self.assertIsNone(find_codex_cli())

    def test_should_not_take_the_approval_location_from_the_environment(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"LOCALAPPDATA": directory, "XDG_CONFIG_HOME": directory}):
            self.assertNotEqual(codex_account._approval_file().parent.parent, Path(directory))

    def test_should_find_codex_in_an_absolute_path_folder(self):
        name = "codex.exe" if os.name == "nt" else "codex"
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory) / name
            installed.write_bytes(b"")
            installed.chmod(0o755)
            with patch.dict(os.environ, {"PATH": directory}):
                self.assertEqual(find_codex_cli(), str(installed))

    def test_should_report_signed_in_when_status_command_succeeds(self):
        with patch.object(codex_account, "find_codex_cli", return_value="codex"), \
                patch.object(codex_account.subprocess, "run", return_value=completed(0, stderr="Logged in using ChatGPT\n")):
            status = codex_status()

        self.assertEqual(status.state, STATE_SIGNED_IN)
        self.assertEqual(status.detail, "Logged in using ChatGPT")

    def test_should_report_signed_out_when_status_command_fails(self):
        with patch.object(codex_account, "find_codex_cli", return_value="codex"), \
                patch.object(codex_account.subprocess, "run", return_value=completed(1, stderr="Not logged in\n")):
            self.assertEqual(codex_status().state, STATE_SIGNED_OUT)

    def test_should_report_error_when_status_command_times_out(self):
        with patch.object(codex_account, "find_codex_cli", return_value="codex"), \
                patch.object(codex_account.subprocess, "run", side_effect=subprocess.TimeoutExpired("codex", 1)):
            self.assertEqual(codex_status().state, STATE_ERROR)

    def test_should_not_pass_api_overrides_to_codex(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only", "CODEX_API_KEY": "test-only",
                                     "OPENAI_BASE_URL": "https://example.invalid"}), \
                patch.object(codex_account, "find_codex_cli", return_value="codex"), \
                patch.object(codex_account.subprocess, "run", return_value=completed(0)) as run:
            codex_status()

        env = run.call_args.kwargs["env"]
        for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"):
            self.assertNotIn(name, env)

    def test_should_not_start_login_when_codex_is_missing(self):
        with patch.object(codex_account, "find_codex_cli", return_value=None), \
                patch.object(codex_account.subprocess, "Popen") as popen:
            self.assertFalse(start_codex_login())

        popen.assert_not_called()

    def test_should_start_login_with_the_resolved_executable(self):
        with patch.object(codex_account, "find_codex_cli", return_value="codex-bin"), \
                patch.object(codex_account.subprocess, "Popen") as popen:
            self.assertTrue(start_codex_login())

        self.assertEqual(popen.call_args.args[0], ["codex-bin", "login"])


if __name__ == "__main__":
    unittest.main()
