"""Forced line breaks in Thai text must fall between words, never inside one."""
import re

import pytest

from manga_translator.rendering import auto_linebreak
from manga_translator.rendering.auto_linebreak import (
    _insert_br_by_pixel_budget,
    _thai_break_units,
)

SENTENCE = "อีกไม่นาน ฉันจะเอาการหลอกลวงของเธอไปจบ แล้วส่งเข้าคุกให้ได้!!"
WORDS = ["อีกไม่นาน", " ", "ฉัน", "จะ", "เอาการ", "หลอกลวง", "ของ", "เธอ", "ไป", "จบ", " ",
         "แล้ว", "ส่ง", "เข้า", "คุก", "ให้ได้", "!!"]
# A combining vowel or tone mark at the start of a line means a syllable was cut apart.
COMBINING_MARK_AT_START = re.compile(r"^[ะ-ฺๅ็-๎]")


@pytest.fixture(autouse=True)
def fixed_tokenizer(monkeypatch):
    """Use a fixed word list so the test does not depend on the installed dictionary."""
    def tokenize(text):
        return WORDS if text == SENTENCE else re.findall(r"[฀-๿]+|\s+|[^฀-๿\s]+", text)

    monkeypatch.setattr(auto_linebreak, "_tokenize_thai_words", tokenize)
    monkeypatch.setattr(auto_linebreak, "get_string_width", lambda size, text, letter_spacing=1.0: len(text) * size)


@pytest.mark.parametrize("segments", [2, 3, 4, 5])
def test_should_break_only_between_words_when_forcing_thai_lines(segments):
    lines = _insert_br_by_pixel_budget(SENTENCE, segments, 40, True, target_lang="THA").split("[BR]")

    assert len(lines) == segments
    assert "".join(lines).replace(" ", "") == SENTENCE.replace(" ", "")
    boundaries = set()
    position = 0
    for word in WORDS:
        position += len(word.strip())
        boundaries.add(position)
    position = 0
    for line in lines[:-1]:
        position += len(line.replace(" ", ""))
        assert position in boundaries
    assert not any(COMBINING_MARK_AT_START.match(line) for line in lines)


def test_should_keep_punctuation_with_the_word_before_it():
    lines = _insert_br_by_pixel_budget(SENTENCE, 5, 40, True, target_lang="THA").split("[BR]")

    assert lines[-1].endswith("ให้ได้!!")
    assert not any(line.startswith("!") for line in lines)


def test_should_keep_repetition_mark_and_quotes_attached_to_their_word(monkeypatch):
    tokens = ["เขา", "บอก", "ว่า", " ", "'", "ไป", "เร็ว", " ", "ๆ", "'", " ", "นะ"]
    monkeypatch.setattr(auto_linebreak, "_tokenize_thai_words", lambda text: tokens)

    assert _thai_break_units("".join(tokens)) == ["เขา", "บอก", "ว่า ", "'ไป", "เร็ว ๆ' ", "นะ"]


def test_should_leave_text_unbroken_when_it_is_a_single_word(monkeypatch):
    monkeypatch.setattr(auto_linebreak, "_tokenize_thai_words", lambda text: [text])

    assert _insert_br_by_pixel_budget("หลอกลวง", 3, 40, True, target_lang="THA") == "หลอกลวง"


def test_should_still_split_other_languages_by_character():
    assert _insert_br_by_pixel_budget("ABCDEF", 2, 40, True, target_lang="JPN").count("[BR]") == 1


@pytest.mark.parametrize("language", ["THA", "tha", "th", "th_TH", "th-TH", "Thai", " thai "])
def test_should_recognize_every_spelling_of_thai(language):
    assert auto_linebreak._is_thai_lang(language)


@pytest.mark.parametrize("language", ["", None, "ENG", "en_US", "KOR", "JPN", "CHS", "tr_TR"])
def test_should_not_treat_other_languages_as_thai(language):
    assert not auto_linebreak._is_thai_lang(language)


def test_should_wrap_thai_by_word_when_the_language_is_given_as_a_name():
    lines, _ = auto_linebreak._calc_horizontal_layout(40, SENTENCE, 40 * 22, "Thai", True)

    assert "".join(lines).replace(" ", "") == SENTENCE.replace(" ", "")
    assert not any(COMBINING_MARK_AT_START.match(line) for line in lines)
    assert all(line.replace(" ", "") != "" for line in lines)


class TestBreakUnitRules:
    """Rules on top of the segmenter. The tokens are fixed, so no dictionary is needed."""

    @pytest.fixture(autouse=True)
    def fixed_measures(self, monkeypatch):
        monkeypatch.setattr(auto_linebreak, "_is_thai_fragment", lambda token: False)
        monkeypatch.setattr(
            auto_linebreak,
            "_measure_horizontal_line_width",
            lambda size, text, letter_spacing=1.0: len(text) * size,
        )

    def _use_tokens(self, monkeypatch, tokens):
        monkeypatch.setattr(auto_linebreak, "_tokenize_thai_words", lambda text: tokens)
        return "".join(tokens)

    def test_should_not_start_a_wrapped_line_with_punctuation(self, monkeypatch):
        text = self._use_tokens(monkeypatch, ["มา", "หรือ", "เปล่า", "?", "อย่า", "ล้อ", "เล่น", "!"])

        # Ten cells: "มาหรือเปล่า" fills the first line exactly, so "?" would be next.
        lines, _ = auto_linebreak._layout_horizontal_thai(10, text, 110)

        assert "".join(lines) == text
        assert not any(line[0] in "?!" for line in lines)

    def test_should_not_start_a_wrapped_line_with_a_repetition_mark(self, monkeypatch):
        text = self._use_tokens(monkeypatch, ["ขู่", "กัน", "ชัด", "ๆ", " ", "ท่าน", "อัยการ"])

        # Nine cells: "ขู่กันชัด" fills the first line exactly, so "ๆ" would be next.
        lines, _ = auto_linebreak._layout_horizontal_thai(10, text, 90)

        assert len(lines) > 1
        assert not any(line.startswith("ๆ") for line in lines)
        assert any("ชัดๆ" in line for line in lines)

    def test_should_keep_a_number_with_its_classifier(self, monkeypatch):
        text = self._use_tokens(monkeypatch, ["อีก", "5", "ปี", "ก็", "คง"])

        assert _thai_break_units(text) == ["อีก", "5ปี", "ก็", "คง"]

    def test_should_keep_a_number_with_its_classifier_across_a_space(self, monkeypatch):
        text = self._use_tokens(monkeypatch, ["อีก", " ", "5", " ", "ปี", "ก็", "คง"])

        assert _thai_break_units(text) == ["อีก ", "5 ปี", "ก็", "คง"]

    def test_should_not_join_a_number_with_an_ordinary_word(self, monkeypatch):
        text = self._use_tokens(monkeypatch, ["ปี", "2024", "เขา", "ไป"])

        assert _thai_break_units(text) == ["ปี", "2024", "เขา", "ไป"]

    def test_should_carry_a_fragment_to_the_word_after_it(self, monkeypatch):
        monkeypatch.setattr(auto_linebreak, "_is_thai_fragment", lambda token: token == "ค")
        text = self._use_tokens(monkeypatch, ["พวก", "เรา", " ", "ค", "วอน", "กับ", "พัค"])

        assert _thai_break_units(text) == ["พวก", "เรา ", "ควอน", "กับ", "พัค"]

    def test_should_attach_a_final_fragment_to_the_word_before_it(self, monkeypatch):
        monkeypatch.setattr(auto_linebreak, "_is_thai_fragment", lambda token: token == "ค")
        text = self._use_tokens(monkeypatch, ["คุณ", "ค"])

        assert _thai_break_units(text) == ["คุณค"]


class TestProtectedWords:
    """Glossary names must stay whole. These use the real segmenter."""

    NAME_SENTENCE = "ชเว จงฮยอกไปหาโค จองซุกที่ร้านโอเด้ง"

    @pytest.fixture(autouse=True)
    def real_tokenizer(self, monkeypatch):
        if not auto_linebreak.HAS_PYTHAINLP:
            pytest.skip("pythainlp is not installed")
        monkeypatch.undo()
        yield
        auto_linebreak.set_thai_protected_words(())

    def test_should_split_an_unknown_name_without_protection(self):
        auto_linebreak.set_thai_protected_words(())

        assert "จงฮยอก" not in auto_linebreak._tokenize_thai_words(self.NAME_SENTENCE)

    def test_should_keep_registered_names_whole(self):
        auto_linebreak.set_thai_protected_words(["จงฮยอก", "จองซุก", "โอเด้ง"])

        tokens = auto_linebreak._tokenize_thai_words(self.NAME_SENTENCE)

        assert {"จงฮยอก", "จองซุก", "โอเด้ง"} <= set(tokens)
        assert "".join(tokens) == self.NAME_SENTENCE

    def test_should_never_break_a_line_inside_a_registered_name(self):
        auto_linebreak.set_thai_protected_words(["จงฮยอก", "จองซุก", "โอเด้ง"])

        for segments in range(2, 7):
            lines = _insert_br_by_pixel_budget(self.NAME_SENTENCE, segments, 40, True, target_lang="THA").split("[BR]")
            joined = "\n".join(lines)
            assert all(name in joined for name in ("จงฮยอก", "จองซุก", "โอเด้ง")), lines

    def test_should_ignore_entries_that_are_not_thai_words(self):
        auto_linebreak.set_thai_protected_words(["OK", "", None, 42, "ก", "จงฮยอก"])

        assert auto_linebreak._thai_protected_words == frozenset({"จงฮยอก"})
