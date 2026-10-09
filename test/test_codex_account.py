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
    STATE_SIGNED_IN,
    STATE_SIGNED_OUT,
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
