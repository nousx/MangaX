"""The bundled Thai fonts: catalog integrity, plain names and licenses."""
import _bootstrap  # noqa: F401

import hashlib
import json

from fontTools.ttLib import TTFont

ROOT = _bootstrap.ROOT
FONTS = ROOT / "fonts"
CATALOG = json.loads((FONTS / "thai-font-catalog.json").read_text(encoding="utf-8"))


def test_should_list_every_bundled_face_with_a_matching_hash():
    for family in CATALOG["families"]:
        assert (FONTS / family["license"]).is_file(), family["family"]
        for variant in family["variants"]:
            path = FONTS / variant["file"]
            assert path.is_file(), variant["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == variant["sha256"], variant["file"]


def test_should_not_put_the_app_name_in_font_or_file_names():
    for family in CATALOG["families"]:
        assert "MangaX" not in family["family"]
        for variant in family["variants"]:
            assert "MangaX" not in variant["file"]
    assert not list(FONTS.glob("MangaX*"))


def test_should_name_each_font_file_after_its_catalog_family():
    # One face per family keeps this fast; hashes above cover the rest.
    for family in CATALOG["families"]:
        font = TTFont(FONTS / family["variants"][0]["file"], lazy=True)
        try:
            names = [record.toUnicode() for record in font["name"].names]
            assert font["name"].getBestFamilyName() == family["family"]
            assert not any("MangaX" in name for name in names), family["family"]
        finally:
            font.close()


def test_should_include_the_handwriting_and_other_tlwg_families():
    families = {family["family"] for family in CATALOG["families"]}

    assert {"Purisa", "Sawasdee", "Garuda", "Loma", "Umpush", "Waree", "Kinnari", "Norasi",
            "Laksaman", "Itim", "Mali", "Sarabun"} <= families
