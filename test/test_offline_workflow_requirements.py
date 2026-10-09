import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop_qt_ui"))
from core.config_models import AppSettings
from core.workflow_requirements import required_api_sections
from services.config_service import ConfigService


class EmptyEnvironment:
    _has_env_value = staticmethod(ConfigService._has_env_value)

    def load_env_vars(self):
        return {}


class OfflineWorkflowTests(unittest.TestCase):
    def missing(self, config):
        return ConfigService.get_missing_runtime_api_requirements(EmptyEnvironment(), config)

    def test_ocr_original_export_ignores_unused_translator_and_renderer(self):
        config = AppSettings()
        config.cli.template = True
        config.cli.save_text = True
        config.translator.translator = "openai_hq"
        config.render.renderer = "gemini_renderer"
        self.assertEqual(required_api_sections(config), {"ocr", "colorizer"})
        self.assertEqual(self.missing(config), [])

    def test_cloud_ocr_still_requires_its_own_key(self):
        config = AppSettings()
        config.cli.template = True
        config.cli.save_text = True
        config.ocr.ocr = "openai_ocr"
        self.assertEqual([item["section"] for item in self.missing(config)], ["ocr"])

    def test_codex_and_local_ocr_need_no_api_key(self):
        config = AppSettings()
        config.translator.translator = "codex"
        self.assertEqual(self.missing(config), [])

    def test_full_cloud_translation_keeps_validation(self):
        config = AppSettings()
        self.assertEqual([item["section"] for item in self.missing(config)], ["translator"])

    def test_existing_json_export_runs_no_ai_stages(self):
        config = AppSettings()
        config.cli.export_from_local_json = True
        config.cli.generate_and_export = True
        config.ocr.ocr = "openai_ocr"
        self.assertEqual(self.missing(config), [])


if __name__ == "__main__":
    unittest.main()
