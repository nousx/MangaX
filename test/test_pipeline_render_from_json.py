"""End-to-end run of the render-from-saved-translations path, without any model.

Loads a saved page, lays the text out again, renders, saves the image and
writes the Photoshop script. It covers how the pieces fit together, which
the unit tests of each piece cannot show.
"""
import _bootstrap  # noqa: F401, I001

import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image

from desktop_qt_ui.services.translation_setup import build_backend_config, build_save_info
from manga_translator.manga_translator import MangaTranslator
from manga_translator.rendering import auto_linebreak
from manga_translator.utils.path_manager import get_json_path

SENTENCE = "อีกไม่นาน ฉันจะเอาการหลอกลวงของเธอไปจบ แล้วส่งเข้าคุกให้ได้!!"
BADLY_WRAPPED = "อีกไม่นาน ฉันจะเอากา[BR]รหลอกลวงของเธอไปจ[BR]บ แล้วส่งเข้าคุกให้ได้!!"


def _settings(**cli):
    return {
        "render": {"layout_mode": "balloon_fill", "recompute_line_breaks": True, "font_family": ""},
        "translator": {"translator": "none", "target_lang": "THA"},
        "detector": {"detector": "none"},
        "inpainter": {"inpainter": "none"},
        "colorizer": {"colorizer": "none"},
        "cli": {
            "load_text": True, "use_gpu": False, "overwrite": True, "save_to_source_dir": True,
            "batch_size": 1, "batch_concurrent": False, "filter_text_enabled": False, **cli,
        },
    }


def _saved_page(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / "001.png"
    Image.new("RGB", (640, 360), "white").save(image)
    region = {
        "lines": [[[80, 100], [560, 100], [560, 260], [80, 260]]],
        "texts": ["SOON, ILL PUT AN END TO YOUR FRAUD AND TAKE YOU STRAIGHT TO PRISON!!"],
        "text": "SOON, ILL PUT AN END TO YOUR FRAUD AND TAKE YOU STRAIGHT TO PRISON!!",
        "translation": BADLY_WRAPPED,
        "font_size": 40,
        "target_lang": "THA",
        "direction": "h",
    }
    json_path = Path(get_json_path(str(image), create_dir=True))
    json_path.write_text(
        json.dumps({str(image): {"regions": [region], "skip_font_scaling": True}}, ensure_ascii=False),
        encoding="utf-8",
    )
    return image


def _run(image: Path, settings: dict, root: Path):
    config = build_backend_config(settings, str(root))
    params = dict(settings["cli"])
    params.update(settings)
    translator = MangaTranslator(params=params)
    save_info = build_save_info(settings, str(root / "unused-output"), [str(image.parent)])
    return asyncio.run(translator.translate_batch([(str(image), config)], batch_size=1, save_info=save_info))


def _saved_region(image: Path) -> dict:
    data = json.loads(Path(get_json_path(str(image), create_dir=False)).read_text(encoding="utf-8"))
    return next(iter(data.values()))["regions"][0]


@pytest.fixture
def thai_segmenter():
    if not auto_linebreak.HAS_PYTHAINLP:
        pytest.skip("pythainlp is not installed")


def test_should_rewrap_a_saved_page_between_words_and_save_beside_the_source(tmp_path, thai_segmenter):
    image = _saved_page(tmp_path / "Chapter 1")

    contexts = _run(image, _settings(), tmp_path)

    assert len(contexts) == 1 and not getattr(contexts[0], "translation_error", None)
    assert (image.parent / "manga_translator_work" / "result" / "001.png").is_file()
    assert not (tmp_path / "unused-output").exists()
    region = _saved_region(image)
    lines = auto_linebreak._BR_RE.split(region["translation"])[::2]
    assert "".join(lines).replace(" ", "") == SENTENCE.replace(" ", "")
    allowed, position = set(), 0
    for token in auto_linebreak._tokenize_thai_words(SENTENCE):
        position += len(token.replace(" ", ""))
        allowed.add(position)
    position = 0
    for line in lines[:-1]:
        position += len(line.replace(" ", ""))
        assert position in allowed, f"line break inside a word: {lines}"
    assert region["translation_unwrapped"] == SENTENCE


def test_should_keep_saved_line_breaks_when_recomputing_is_off(tmp_path):
    image = _saved_page(tmp_path / "Chapter 1")
    settings = _settings()
    settings["render"]["recompute_line_breaks"] = False

    _run(image, settings, tmp_path)

    assert _saved_region(image)["translation"] == BADLY_WRAPPED


def test_should_write_the_photoshop_script_when_only_script_export_is_on(tmp_path):
    image = _saved_page(tmp_path / "Chapter 1")

    _run(image, _settings(psd_script_only=True, export_editable_psd=False), tmp_path)

    scripts = list((image.parent / "manga_translator_work" / "psd").glob("001*.jsx"))
    assert len(scripts) == 1 and scripts[0].stat().st_size > 0


def test_should_not_write_a_photoshop_script_when_export_is_off(tmp_path):
    image = _saved_page(tmp_path / "Chapter 1")

    _run(image, _settings(), tmp_path)

    assert not (image.parent / "manga_translator_work" / "psd").exists()
