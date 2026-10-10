"""The desktop window and the command-line mode must configure a run the same way."""
import os

import pytest

from desktop_qt_ui.services.translation_setup import (
    build_backend_config,
    build_save_info,
    normalize_output_format,
    normalize_upscale_ratio,
)


def settings(**cli):
    return {
        "render": {"direction": "h", "font_family": "Itim"},
        "translator": {"translator": "claude", "target_lang": "THA"},
        "cli": {"export_editable_psd": True, "psd_script_only": True, **cli},
        "app": {"theme": "dark"},
    }


def test_should_carry_psd_export_options_into_the_backend_config(tmp_path):
    config = build_backend_config(settings(), str(tmp_path))

    assert config.cli.export_editable_psd is True
    assert config.cli.psd_script_only is True


def test_should_translate_short_direction_names(tmp_path):
    config = build_backend_config(settings(), str(tmp_path))

    assert config.render.direction.value == "horizontal"


def test_should_ignore_desktop_only_sections(tmp_path):
    config = build_backend_config(settings(), str(tmp_path))

    assert not hasattr(config, "app")


def test_should_resolve_relative_prompt_path_against_the_app_folder(tmp_path):
    prompt = tmp_path / "dict" / "story.yaml"
    prompt.parent.mkdir()
    prompt.write_text("system_prompt: x", encoding="utf-8")
    data = settings()
    data["translator"]["high_quality_prompt_path"] = "dict/story.yaml"

    config = build_backend_config(data, str(tmp_path))

    assert os.path.samefile(config.translator.high_quality_prompt_path, prompt)


def test_should_warn_when_prompt_file_is_missing(tmp_path):
    warnings = []
    data = settings()
    data["translator"]["high_quality_prompt_path"] = "dict/missing.yaml"

    config = build_backend_config(data, str(tmp_path), warn=warnings.append)

    assert config.translator.high_quality_prompt_path == "dict/missing.yaml"
    assert len(warnings) == 1 and "missing.yaml" in warnings[0]


@pytest.mark.parametrize("value,expected", [
    ("不使用", None), (None, None), ("2", 2), (4, 4), ("x2", "x2"), ("DAT2 x4", "DAT2 x4"), ("junk", None),
])
def test_should_normalize_upscale_ratio(value, expected):
    assert normalize_upscale_ratio(value) == expected


@pytest.mark.parametrize("value,expected", [("不指定", None), ("", None), (None, None), ("png", "png")])
def test_should_normalize_output_format(value, expected):
    assert normalize_output_format(value) == expected


def test_should_save_beside_the_source_when_the_setting_is_on():
    info = build_save_info(settings(save_to_source_dir=True, format="不指定"), "out", ["a/b", None, "a/b"])

    assert info["save_to_source_dir"] is True
    assert info["format"] is None
    assert info["input_folders"] == {os.path.normpath("a/b")}
    assert info["output_folder"] == "out"


def test_should_let_the_caller_override_overwrite():
    data = settings(overwrite=True)

    assert build_save_info(data, "out", [])["overwrite"] is True
    assert build_save_info(data, "out", [], overwrite=False)["overwrite"] is False
