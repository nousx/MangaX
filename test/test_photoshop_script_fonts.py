"""The saved Photoshop script must use each region's own font and a clean image that still exists later."""
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from manga_translator.utils import photoshop_export as export
from manga_translator.utils.path_manager import get_inpainted_path
from manga_translator.utils.textblock import TextBlock


def _region(font_family="", translation="สวัสดี"):
    region = _text_block(font_family, translation)
    # Normally set by the render step: where the text box ends up on the page.
    region.dst_points = np.array([[[50, 50], [150, 50], [150, 150], [50, 150]]])
    return region


def _text_block(font_family, translation):
    return TextBlock(
        lines=np.array([[[50, 50], [150, 50], [150, 80], [50, 80]]]),
        texts=["HELLO"],
        font_size=30,
        translation=translation,
        font_family=font_family,
        direction="h",
        alignment="center",
        target_lang="THA",
    )


@pytest.mark.parametrize("value,expected", [
    ("Anuphan::SemiBold", ("Anuphan", "SemiBold")),
    ("Playpen Sans Thai::SemiBold", ("Playpen Sans Thai", "SemiBold")),
    ("Anuphan", ("Anuphan", "")),
    ("  Anuphan :: Regular ", ("Anuphan", "Regular")),
    ("C:/fonts/anuphan-600-normal.ttf", ("anuphan-600-normal", "")),
    ("", ("", "")),
    (None, ("", "")),
])
def test_should_split_a_stored_font_into_family_and_style(value, expected):
    assert export.split_photoshop_font(value) == expected


def test_should_ask_photoshop_for_the_font_of_the_region_not_the_default():
    script = export.generate_text_layer_jsx(0, _region("Anuphan::SemiBold"), "Anuphan::Regular")

    assert "var requestedFont0 = 'Anuphan';" in script
    assert "var requestedStyle0 = 'SemiBold';" in script
    assert "findFontPostScriptName(requestedFont0, requestedStyle0)" in script
    assert "Regular" not in script
    assert "::" not in script


def test_should_fall_back_to_the_default_font_when_the_region_has_none():
    script = export.generate_text_layer_jsx(3, _region(""), "Anuphan::Regular")

    assert "var requestedFont3 = 'Anuphan';" in script
    assert "var requestedStyle3 = 'Regular';" in script


def test_should_use_the_photoshop_default_when_no_font_is_known():
    script = export.generate_text_layer_jsx(0, _region(""), None)

    assert "requestedFont0" not in script
    assert "Using Photoshop default font" in script


def test_should_match_family_and_style_in_the_font_lookup():
    assert "function findFontPostScriptName(fontName, fontStyle)" in export.JSX_TEMPLATE
    assert "normalizeFontToken(fonts[s].style) === wantedStyle" in export.JSX_TEMPLATE


@pytest.fixture
def page(tmp_path):
    image_path = tmp_path / "001.png"
    Image.new("RGB", (200, 200), "white").save(image_path)
    return image_path


def _script_only_export(page, tmp_path, ctx):
    export.photoshop_export(
        str(tmp_path / "out.psd"), ctx, default_font="Anuphan::Regular", image_path=str(page), script_only=True
    )
    return (page.parent / "manga_translator_work" / "psd" / "001_photoshop_script.jsx").read_text(encoding="utf-8")


def test_should_point_a_saved_script_at_the_inpainted_image_in_the_work_folder(page, tmp_path):
    saved = get_inpainted_path(str(page), create_dir=True)
    Image.new("RGB", (200, 200), "gray").save(saved)
    ctx = SimpleNamespace(text_regions=[], img_inpainted=np.zeros((200, 200, 3), dtype=np.uint8))

    script = _script_only_export(page, tmp_path, ctx)

    assert "new File(CHAPTER_DIR + '/manga_translator_work/inpainted/001_inpainted" in script
    assert tempfile.gettempdir().replace("\\", "/") + "/.ps_inpainted.png" not in script


def test_should_keep_a_copy_of_the_inpainted_image_when_none_is_saved(page, tmp_path):
    ctx = SimpleNamespace(text_regions=[], img_inpainted=np.zeros((200, 200, 3), dtype=np.uint8))

    script = _script_only_export(page, tmp_path, ctx)

    kept = page.parent / "manga_translator_work" / "psd" / "001_inpainted.png"
    assert kept.is_file()
    assert "new File(CHAPTER_DIR + '/manga_translator_work/psd/001_inpainted.png')" in script
    assert ".ps_inpainted.png" not in script


def test_should_save_a_script_that_names_no_folder_of_this_machine(page, tmp_path):
    saved = get_inpainted_path(str(page), create_dir=True)
    Image.new("RGB", (200, 200), "gray").save(saved)
    ctx = SimpleNamespace(text_regions=[], img_inpainted=np.zeros((200, 200, 3), dtype=np.uint8))
    psd_path = page.parent / "manga_translator_work" / "psd" / "001.psd"

    export.photoshop_export(
        str(psd_path), ctx, default_font="Anuphan::Regular", image_path=str(page), script_only=True
    )
    script = (psd_path.parent / "001_photoshop_script.jsx").read_text(encoding="utf-8")

    assert str(tmp_path).replace("\\", "/") not in script
    assert tempfile.gettempdir().replace("\\", "/") not in script
    assert "var CHAPTER_DIR = SCRIPT_FILE.parent.parent.parent.fsName" in script
    assert "new File(CHAPTER_DIR + '/001.png')" in script
    assert "new File(CHAPTER_DIR + '/manga_translator_work/psd/001.psd')" in script
    assert "var ERROR_FILE_PATH = Folder.temp.fsName" in script
    assert "'Adding inpainted layer from: ' + CHAPTER_DIR + '/manga_translator_work/inpainted/" in script


def test_should_not_rewrite_translated_text_that_ends_with_a_path_of_the_page(page, tmp_path):
    page_path = str(page).replace("\\", "/")
    ctx = SimpleNamespace(
        text_regions=[_region("Anuphan::SemiBold", translation=f"see {page_path}")],
        img_inpainted=np.zeros((200, 200, 3), dtype=np.uint8),
    )

    script = _script_only_export(page, tmp_path, ctx)

    assert script.count(page_path) == 1
    assert f"see {page_path}" in script
    assert export.TEXT_LAYERS_SLOT not in script
    assert "new File(CHAPTER_DIR + '/001.png')" in script


def test_should_keep_a_path_outside_the_chapter_folder_as_it_is(tmp_path):
    chapter = tmp_path / "Chapter 1"
    outside = tmp_path / "elsewhere" / "001.psd"

    assert export.chapter_relative_expression(str(outside), str(chapter)) is None
    assert export.chapter_relative_expression(str(chapter), str(chapter)) is None
    assert export.chapter_relative_expression("", str(chapter)) is None


def test_should_address_a_path_inside_the_chapter_folder_from_the_script(tmp_path):
    chapter = tmp_path / "Chapter 1"
    inside = chapter / "manga_translator_work" / "inpainted" / "it's 001.webp"

    expression = export.chapter_relative_expression(str(inside), str(chapter))

    assert expression == "CHAPTER_DIR + '/manga_translator_work/inpainted/it\\'s 001.webp'"


def test_should_leave_a_script_alone_when_its_error_file_line_is_not_found():
    script = "var x = 'C:/chapter/001.png';"

    assert export.make_script_relocatable(script, "C:/chapter", ["C:/chapter/001.png"], "C:/temp/.ps_error.txt") == script
