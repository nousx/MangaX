import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from manga_translator.config import Config, TranslatorConfig
from manga_translator.translators import get_translator
from manga_translator.translators.codex_cli import CodexCLITranslator
from manga_translator.translators.common import InvalidServerResponse


class CodexResponseTests(unittest.TestCase):
    def test_ids_restore_balloon_order(self):
        self.assertEqual(CodexCLITranslator.validate_response({"translations": [
            {"id": 1, "translation": "สอง"}, {"id": 0, "translation": "หนึ่ง"},
        ]}, 2), ["หนึ่ง", "สอง"])

    def test_invalid_batches_are_rejected(self):
        for items in (
            [], [{"id": 0, "translation": "one"}],
            [{"id": 0, "translation": "one"}, {"id": 0, "translation": "two"}],
            [{"id": True, "translation": "one"}, {"id": 0, "translation": "two"}],
            [{"id": 0, "translation": "one"}, {"id": 2, "translation": "two"}],
            [{"id": 0, "translation": "one"}, {"id": 1, "translation": " "}],
        ):
            with self.subTest(items=items), self.assertRaises(InvalidServerResponse):
                CodexCLITranslator.validate_response({"translations": items}, 2)

    def test_registration_and_config_roundtrip(self):
        config = Config(translator=TranslatorConfig(translator="codex", codex_model="example-model"))
        restored = Config.model_validate_json(config.model_dump_json())
        translator = get_translator(restored.translator.translator)
        translator.parse_args(restored)
        self.assertEqual(translator.model, "example-model")


class CodexProcessTests(unittest.IsolatedAsyncioTestCase):
    async def run_fake_cli(self, mode, translator):
        """Use a real child process to exercise stdin, UTF-8, cancellation and cleanup."""
        actual_popen = subprocess.Popen
        processes = []
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake_cli.py"
            script.write_text(
                "import json,sys,time\nfrom pathlib import Path\n"
                "data=sys.stdin.buffer.read().decode('utf-8')\n"
                "assert 'こんにちは' in data\n"
                + ("time.sleep(30)\n" if mode == "slow" else "")
                + ("sys.exit(9)\n" if mode == "failed" else "")
                + ("sys.stderr.buffer.write(('user: ' + data + chr(10)"
                   " + 'ERROR: You have hit your usage limit. Try again at 12:15 PM.' + chr(10)).encode('utf-8'))\n"
                   "sys.exit(1)\n" if mode == "limit" else "")
                + "out=Path(sys.argv[sys.argv.index('--output-last-message')+1])\n"
                + ("out.write_text('bad json',encoding='utf-8')\n" if mode == "invalid" else
                   "out.write_text(json.dumps({'translations':[{'id':0,'translation':'สวัสดี'}]},ensure_ascii=False),encoding='utf-8')\n"),
                encoding="utf-8",
            )

            def start(command, **kwargs):
                self.assertIn("--ignore-user-config", command)
                self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
                self.assertIn('web_search="disabled"', command)
                for feature in ("shell_tool", "unified_exec", "apps", "hooks", "plugins"):
                    self.assertIn(feature, command)
                self.assertNotIn("shell", kwargs)
                for name in ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
                    self.assertNotIn(name, kwargs["env"])
                process = actual_popen([sys.executable, str(script), *command[1:]], **kwargs)
                processes.append(process)
                return process

            with patch("manga_translator.translators.codex_cli.subprocess.Popen", side_effect=start):
                try:
                    return await translator._translate_batch("fake-codex", "Japanese", "Thai", ["こんにちは"])
                finally:
                    self.assertTrue(processes)
                    self.assertIsNotNone(processes[0].poll(), "CLI child process must be reaped")

    async def test_unicode_stdin_and_result(self):
        with patch.dict(os.environ, {"CODEX_API_KEY": "test-only", "OPENAI_API_KEY": "test-only",
                                    "OPENAI_BASE_URL": "https://example.invalid"}):
            self.assertEqual(await self.run_fake_cli("ok", CodexCLITranslator()), ["สวัสดี"])

    async def test_failure_and_invalid_json(self):
        with self.assertRaises(RuntimeError):
            await self.run_fake_cli("failed", CodexCLITranslator())
        with self.assertRaises(InvalidServerResponse):
            await self.run_fake_cli("invalid", CodexCLITranslator())

    async def test_timeout_reaps_process(self):
        translator = CodexCLITranslator()
        translator.timeout = 0.05
        with self.assertRaises(TimeoutError):
            await self.run_fake_cli("slow", translator)

    async def test_stop_callback_reaps_process(self):
        translator = CodexCLITranslator()
        translator.set_cancel_check_callback(lambda: True)
        with self.assertRaises(asyncio.CancelledError):
            await self.run_fake_cli("slow", translator)

    async def test_should_tell_the_user_why_codex_failed(self):
        with self.assertRaises(RuntimeError) as raised:
            await self.run_fake_cli("limit", CodexCLITranslator())

        message = str(raised.exception)
        self.assertIn("usage limit", message)
        self.assertIn("12:15 PM", message)
        self.assertNotIn("こんにちは", message, "the request text must not be echoed into the error")


class CodexStyleGuideTests(unittest.TestCase):
    def test_should_send_the_prompt_file_style_guide(self):
        translator = CodexCLITranslator()
        translator._style_guide = CodexCLITranslator.style_guide(
            {"custom_prompt_json": {"system_prompt": "Write natural {{{target_lang}}}."}}, "Thai")

        prompt = translator._build_prompt("English", "Thai", ["HELLO"])

        self.assertIn("Write natural Thai.", prompt)
        self.assertLess(prompt.index("Write natural Thai."), prompt.index('"items"'))

    def test_should_send_no_style_guide_when_the_prompt_file_has_none(self):
        translator = CodexCLITranslator()
        translator._style_guide = CodexCLITranslator.style_guide({"custom_prompt_json": {"glossary": {}}}, "Thai")

        self.assertNotIn("Style guide chosen by the user", translator._build_prompt("English", "Thai", ["HELLO"]))

    def test_should_cap_the_style_guide_length(self):
        guide = CodexCLITranslator.style_guide({"custom_prompt_json": {"system_prompt": "x" * 20000}}, "Thai")

        self.assertEqual(len(guide), 8000)

    def test_should_pick_only_diagnostic_lines_from_cli_output(self):
        output = "\n".join(["user: secret manga line", "thinking...", "ERROR: quota exceeded", ""]).encode("utf-8")

        self.assertEqual(CodexCLITranslator.failure_lines(output), "ERROR: quota exceeded")
        self.assertEqual(CodexCLITranslator.failure_lines(b""), "")
        self.assertEqual(CodexCLITranslator.failure_lines(None), "")


if __name__ == "__main__":
    unittest.main()
