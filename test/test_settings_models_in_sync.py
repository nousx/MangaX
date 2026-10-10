"""The desktop settings models and the backend ones describe the same settings twice.

Until they are merged, this test stops them drifting apart unnoticed: a field
added to one side only, or a default changed on one side only, fails here and
has to be either mirrored or recorded below as intended.
"""
import _bootstrap  # noqa: F401, I001

import pytest

from core import config_models as desktop
from manga_translator import config as backend

PAIRS = {
    "CliSettings": "CliConfig",
    "ColorizerSettings": "ColorizerConfig",
    "DetectorSettings": "DetectorConfig",
    "InpainterSettings": "InpainterConfig",
    "OcrSettings": "OcrConfig",
    "RenderSettings": "RenderConfig",
    "TranslatorSettings": "TranslatorConfig",
    "UpscaleSettings": "UpscaleConfig",
}

# Defaults that differ today, as (desktop, backend). The desktop value is what a
# new installation starts with; the backend value applies when a section is
# missing from the settings passed in. Recorded so a new difference stands out.
KNOWN_DEFAULT_DIFFERENCES = {
    "CliSettings": {"format": ("不指定", None), "overwrite": (True, False), "save_text": (True, False)},
    "DetectorSettings": {"box_threshold": (0.5, 0.7), "unclip_ratio": (2.5, 2.3)},
    "InpainterSettings": {"inpainter": ("lama_mpe", "lama_large"), "inpainting_precision": ("fp32", "bf16")},
    "OcrSettings": {"prob": (0.1, None), "secondary_ocr": ("mocr", "48px"), "use_hybrid_ocr": (True, False)},
    "RenderSettings": {
        "disable_auto_wrap": (True, False), "font_family": ("", None), "font_size_minimum": (0, -1),
        "letter_spacing": (1.0, None), "line_spacing": (1.0, None),
    },
    "TranslatorSettings": {
        "high_quality_prompt_path": ("dict/prompt_example.yaml", None), "target_lang": ("CHS", "ENG"),
    },
}
# Fields that exist on one side on purpose.
DESKTOP_ONLY = {
    "CliSettings": {"load_text", "template", "generate_and_export", "colorize_only", "upscale_only", "inpaint_only"},
    "RenderSettings": {"disable_system_fonts"},
}


def _default(field):
    value = field.default
    return getattr(value, "value", value)


def _models(name):
    return getattr(desktop, name), getattr(backend, PAIRS[name])


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_shared_fields_should_have_the_same_default_unless_recorded(name):
    desktop_model, backend_model = _models(name)
    shared = set(desktop_model.model_fields) & set(backend_model.model_fields)

    differences = {
        field: (_default(desktop_model.model_fields[field]), _default(backend_model.model_fields[field]))
        for field in sorted(shared)
        if _default(desktop_model.model_fields[field]) != _default(backend_model.model_fields[field])
    }

    assert differences == KNOWN_DEFAULT_DIFFERENCES.get(name, {})


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_desktop_should_not_gain_a_field_the_backend_ignores(name):
    desktop_model, backend_model = _models(name)

    desktop_only = set(desktop_model.model_fields) - set(backend_model.model_fields)

    assert desktop_only <= DESKTOP_ONLY.get(name, set()), (
        f"{name} has fields the backend {PAIRS[name]} does not know: "
        f"{sorted(desktop_only - DESKTOP_ONLY.get(name, set()))}"
    )


@pytest.mark.parametrize("field", ["normalize_thai_punctuation", "remove_trailing_period"])
def test_translator_switches_should_exist_on_both_sides(field):
    assert field in desktop.TranslatorSettings.model_fields
    assert field in backend.TranslatorConfig.model_fields


def test_render_switches_should_exist_on_both_sides():
    assert "recompute_line_breaks" in desktop.RenderSettings.model_fields
    assert "recompute_line_breaks" in backend.RenderConfig.model_fields
