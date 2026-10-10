import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from manga_translator import claude_account, codex_account
from manga_translator.claude_account import (
    approve_claude_cli,
    claude_child_env,
    claude_status,
    find_claude_cli,
    start_claude_login,
)
from manga_translator.codex_account import (
    STATE_MISSING,
    STATE_NEEDS_APPROVAL,
    STATE_SIGNED_IN,
    STATE_SIGNED_OUT,
    CodexPathNotApproved,
)
from manga_translator.config import Config, TranslatorConfig
from manga_translator.translators import get_translator
from manga_translator.translators.claude_cli import ClaudeCLITranslator
from manga_translator.translators.common import InvalidServerResponse

EXECUTABLE = "claude.exe" if os.name == "nt" else "claude"


def completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(["claude"], returncode, stdout, stderr)


class ClaudeAccountTests(unittest.TestCase):
    def test_should_report_missing_when_claude_is_not_installed(self):
        with patch.object(claude_account, "find_claude_cli", return_value=None):
            self.assertEqual(claude_status().state, STATE_MISSING)

    def test_should_report_signed_in_with_plan_but_without_address(self):
        output = json.dumps({"loggedIn": True, "subscriptionType": "pro", "email": "someone@example.invalid"})
        with patch.object(claude_account, "find_claude_cli", return_value="claude"), \
                patch.object(claude_account.subprocess, "run", return_value=completed(0, output)):
            status = claude_status()

        self.assertEqual(status.state, STATE_SIGNED_IN)
        self.assertEqual(status.detail, "pro")

    def test_should_report_signed_out_when_not_logged_in(self):
        for result in (completed(0, json.dumps({"loggedIn": False})), completed(1, ""), completed(0, "not json")):
            with self.subTest(result=result), \
                    patch.object(claude_account, "find_claude_cli", return_value="claude"), \
                    patch.object(claude_account.subprocess, "run", return_value=result):
                self.assertEqual(claude_status().state, STATE_SIGNED_OUT)

    def test_should_require_approval_before_using_a_configured_path(self):
        with tempfile.TemporaryDirectory() as directory:
            configured = Path(directory) / EXECUTABLE
            configured.write_bytes(b"")
            approval = Path(directory) / "state" / "approved.txt"
            with patch.object(claude_account, "_approval_file", return_value=approval), \
                    patch.object(claude_account.subprocess, "run") as run, \
                    patch.object(claude_account.subprocess, "Popen") as popen:
                with self.assertRaises(CodexPathNotApproved):
                    find_claude_cli(str(configured))
                self.assertEqual(claude_status(str(configured)).state, STATE_NEEDS_APPROVAL)
                self.assertFalse(start_claude_login(str(configured)))
                run.assert_not_called()
                popen.assert_not_called()

                approve_claude_cli(str(configured))

                self.assertEqual(find_claude_cli(str(configured)), str(configured.resolve()))

    def test_should_keep_claude_and_codex_approvals_apart(self):
        self.assertNotEqual(claude_account._approval_file(), codex_account._approval_file())

    def test_should_reject_configured_path_that_is_not_the_claude_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("claude.cmd", "other.exe"):
                other = Path(directory) / name
                other.write_bytes(b"")
                with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                    find_claude_cli(str(other))

    def test_should_find_claude_in_the_account_path_and_not_the_process_path(self):
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory) / EXECUTABLE
            installed.write_bytes(b"")
            installed.chmod(0o755)
            with patch.object(codex_account, "_trusted_environment", return_value={"PATH": directory}):
                self.assertEqual(find_claude_cli(), str(installed))
            with patch.dict(os.environ, {"PATH": directory}), \
                    patch.object(codex_account, "_trusted_environment", return_value={"PATH": ""}):
                self.assertIsNone(find_claude_cli())

    def test_should_take_account_settings_from_the_account_environment(self):
        process = {"ANTHROPIC_API_KEY": "test-only", "ANTHROPIC_BASE_URL": "https://example.invalid",
                   "CLAUDE_CONFIG_DIR": "C:/planted"}
        with patch.dict(os.environ, process), \
                patch.object(codex_account, "_trusted_environment", return_value={"CLAUDE_CONFIG_DIR": "C:/real"}):
            env = claude_child_env()

        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("ANTHROPIC_BASE_URL", env)
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], "C:/real")

    def test_should_not_pass_app_environment_to_the_cli(self):
        planted = {"NODE_OPTIONS": "--require planted.js", "HTTPS_PROXY": "http://planted.invalid:1",
                   "NODE_EXTRA_CA_CERTS": "C:/planted.pem", "PATH": "C:/planted"}
        trusted = {"PATH": "C:/real", "USERPROFILE": "C:/Users/real"}
        with patch.dict(os.environ, planted), \
                patch.object(codex_account, "_trusted_environment", return_value=trusted):
            self.assertEqual(claude_child_env(), trusted)
            self.assertEqual(codex_account.codex_child_env(), trusted)

    def test_should_not_start_the_cli_when_the_account_environment_is_unknown(self):
        with patch.object(codex_account, "_trusted_environment", return_value={}), \
                patch.object(claude_account, "find_claude_cli", return_value="claude-bin"), \
                patch.object(claude_account.subprocess, "run") as run, \
                patch.object(claude_account.subprocess, "Popen") as popen:
            with self.assertRaises(OSError):
                claude_child_env()
            self.assertEqual(claude_status().state, codex_account.STATE_ERROR)
            self.assertFalse(start_claude_login())

        run.assert_not_called()
        popen.assert_not_called()

    def test_should_start_login_with_the_resolved_executable(self):
        with patch.object(claude_account, "find_claude_cli", return_value="claude-bin"), \
                patch.object(claude_account.subprocess, "Popen") as popen:
            self.assertTrue(start_claude_login())

        self.assertEqual(popen.call_args.args[0], ["claude-bin", "auth", "login"])


class ClaudeTranslatorTests(unittest.TestCase):
    def test_registration_and_config_roundtrip(self):
        config = Config(translator=TranslatorConfig(translator="claude", claude_model="example-model",
                                                    claude_batch_size=7))
        restored = Config.model_validate_json(config.model_dump_json())
        translator = get_translator(restored.translator.translator)
        translator.parse_args(restored)

        self.assertIsInstance(translator, ClaudeCLITranslator)
        self.assertEqual(translator.model, "example-model")
        self.assertEqual(translator.batch_size, 7)

    def test_should_default_to_the_sonnet_model(self):
        self.assertEqual(TranslatorConfig().claude_model, "sonnet")

    def test_command_disables_tools_settings_and_sessions(self):
        translator = ClaudeCLITranslator()
        command = translator._build_command("claude-bin", "English", "Thai")

        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertEqual(command[command.index("--setting-sources") + 1], "")
        for flag in ("-p", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("--model") + 1], "sonnet")
        self.assertNotIn("--dangerously-skip-permissions", command)

    def test_style_guide_comes_from_the_selected_prompt_file(self):
        ctx = {"custom_prompt_json": {"system_prompt": "Write natural {{{target_lang}}}."}}
        translator = ClaudeCLITranslator()
        translator._style_guide = ClaudeCLITranslator.style_guide(ctx, "Thai")

        self.assertEqual(translator._style_guide, "Write natural Thai.")
        self.assertIn("Write natural Thai.", translator._build_system_prompt("English", "Thai"))
        self.assertEqual(ClaudeCLITranslator.style_guide({"custom_prompt_json": {"system_prompt": ""}}, "Thai"), "")
        self.assertEqual(ClaudeCLITranslator.style_guide(None, "Thai"), "")


class ClaudeProcessTests(unittest.IsolatedAsyncioTestCase):
    async def run_fake_cli(self, mode):
        """Use a real child process to exercise stdin, UTF-8 and cleanup."""
        actual_popen = subprocess.Popen
        processes = []
        translations = {"translations": [{"id": 0, "translation": "สวัสดี"}]}
        outputs = {
            "ok": {"is_error": False, "structured_output": translations},
            "limit": {"is_error": True, "result": "You've hit your usage limit. Try again later."},
            "missing": {"is_error": False, "result": "plain text"},
        }
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake_cli.py"
            script.write_text(
                "import sys\n"
                "data=sys.stdin.buffer.read().decode('utf-8')\n"
                "assert 'こんにちは' in data\n"
                + ("sys.stdout.buffer.write(b'not json')\n" if mode == "invalid" else
                   f"sys.stdout.buffer.write({json.dumps(outputs.get(mode, outputs['ok']), ensure_ascii=False)!r}.encode('utf-8'))\n")
                + ("sys.exit(1)\n" if mode == "limit" else ""),
                encoding="utf-8",
            )

            def start(command, **kwargs):
                self.assertNotIn("shell", kwargs)
                self.assertNotIn("ANTHROPIC_API_KEY", kwargs["env"])
                process = actual_popen([sys.executable, str(script)], **kwargs)
                processes.append(process)
                return process

            with patch("manga_translator.translators.claude_cli.subprocess.Popen", side_effect=start), \
                    patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-only"}):
                try:
                    return await ClaudeCLITranslator()._translate_batch("fake-claude", "Japanese", "Thai", ["こんにちは"])
                finally:
                    self.assertTrue(processes)
                    self.assertIsNotNone(processes[0].poll(), "CLI child process must be reaped")

    async def test_unicode_stdin_and_structured_result(self):
        self.assertEqual(await self.run_fake_cli("ok"), ["สวัสดี"])

    async def test_usage_limit_message_reaches_the_user(self):
        with self.assertRaises(RuntimeError) as raised:
            await self.run_fake_cli("limit")

        self.assertIn("usage limit", str(raised.exception))

    async def test_invalid_or_unstructured_output_is_rejected(self):
        for mode in ("invalid", "missing"):
            with self.subTest(mode=mode), self.assertRaises(InvalidServerResponse):
                await self.run_fake_cli(mode)


if __name__ == "__main__":
    unittest.main()
