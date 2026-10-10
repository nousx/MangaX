"""'Script only' PSD export must work without the 'export editable PSD' switch."""
from types import SimpleNamespace

import pytest

from manga_translator.config import CliConfig, Config
from manga_translator.utils.photoshop_export import psd_export_requested


@pytest.mark.parametrize("export_psd,script_only,expected", [
    (False, False, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
])
def test_should_request_export_when_either_switch_is_on(export_psd, script_only, expected):
    config = Config(cli=CliConfig(export_editable_psd=export_psd, psd_script_only=script_only))

    assert psd_export_requested(config) is expected


def test_should_not_request_export_when_settings_are_missing():
    assert psd_export_requested(None) is False
    assert psd_export_requested(SimpleNamespace()) is False
