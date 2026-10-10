"""Stored automatic line breaks can be dropped so a later layout wraps the text again."""
import numpy as np
import pytest

from manga_translator.config import RenderConfig
from manga_translator.rendering.auto_linebreak import unwrapped_translation
from manga_translator.utils.textblock import TextBlock

ORIGINAL = "อีกไม่นาน ฉันจะเอาการหลอกลวงของเธอไปจบ แล้วส่งเข้าคุกให้ได้!!"
BADLY_WRAPPED = "อีกไม่นาน ฉันจะเอากา[BR]รหลอกลวงของเธอไปจ[BR]บ แล้วส่งเข้าคุกให้ได้!!"


def test_should_return_the_saved_text_when_it_still_matches():
    assert unwrapped_translation(BADLY_WRAPPED, ORIGINAL, "THA") == ORIGINAL


def test_should_restore_a_space_lost_at_a_break_from_the_saved_text():
    wrapped = "อีกไม่นาน[BR]ฉันจะไป"

    assert unwrapped_translation(wrapped, "อีกไม่นาน ฉันจะไป", "THA") == "อีกไม่นาน ฉันจะไป"


def test_should_use_the_current_text_when_it_was_edited_after_wrapping():
    edited = "อีกไม่นาน[BR]ฉันจะกลับมา"

    assert unwrapped_translation(edited, ORIGINAL, "THA") == "อีกไม่นานฉันจะกลับมา"


def test_should_join_thai_without_a_space_when_nothing_was_saved():
    assert unwrapped_translation(BADLY_WRAPPED, "", "THA") == ORIGINAL


@pytest.mark.parametrize("language,expected", [
    ("ENG", "GET BACK HERE"),
    ("KOR", "GET BACK HERE"),
    ("JPN", "GETBACKHERE"),
    ("CHS", "GETBACKHERE"),
])
def test_should_join_with_a_space_only_for_languages_written_with_spaces(language, expected):
    assert unwrapped_translation("GET[BR]BACK [BR] HERE", "", language) == expected


@pytest.mark.parametrize("marker", ["[BR]", "[br]", "<br>", "【BR】"])
def test_should_recognize_every_break_marker(marker):
    assert unwrapped_translation(f"หนึ่ง{marker}สอง", "", "THA") == "หนึ่งสอง"


def test_should_ignore_saved_text_that_itself_contains_breaks():
    assert unwrapped_translation("หนึ่ง[BR]สอง", "หนึ่ง[BR]สอง", "THA") == "หนึ่งสอง"


@pytest.mark.parametrize("value", ["ไม่มีการตัดบรรทัด", "", None, {"type": "doc"}])
def test_should_leave_text_without_breaks_untouched(value):
    assert unwrapped_translation(value, "something else", "THA") == value


def _block(**fields):
    lines = np.array([[[0, 0], [100, 0], [100, 40], [0, 40]]])
    return TextBlock(lines=lines, texts=["HELLO"], **fields)


def test_should_save_and_reload_the_text_before_wrapping():
    block = _block(translation="สวัสดี[BR]ครับ", translation_unwrapped="สวัสดี ครับ")

    saved = block.to_dict()

    assert saved["translation_unwrapped"] == "สวัสดี ครับ"
    assert _block(translation=saved["translation"],
                  translation_unwrapped=saved["translation_unwrapped"]).translation_unwrapped == "สวัสดี ครับ"


def test_should_default_to_empty_for_files_saved_before_the_field_existed():
    assert _block(translation="สวัสดี").translation_unwrapped == ""


def test_should_be_off_by_default():
    assert RenderConfig().recompute_line_breaks is False
