"""Optional clean-up of sentence-ending marks in Thai translations."""
import pytest

from manga_translator.utils.translation_text import is_thai_language, normalize_thai_punctuation


@pytest.mark.parametrize("text,expected", [
    ("ไม่เหนื่อยเหรอ?", "ไม่เหนื่อยเหรอ"),
    ("นี่ลูกชายเหรอคะ?", "นี่ลูกชายเหรอคะ"),
    ("ไปไหนกันแน่?", "ไปไหนกันแน่?"),
    ("กินข้าวหรือยัง?", "กินข้าวหรือยัง?"),
    ("ทำไม?", "ทำไม"),
    ("ใครน่ะ?", "ใครน่ะ"),
])
def test_should_drop_question_mark_after_a_question_word(text, expected):
    assert normalize_thai_punctuation(text, True) == expected


@pytest.mark.parametrize("text", ["คราวนี้ไปทำอะไรมาอีก?", "จะกินข้าวหรือก๋วยเตี๋ยว?"])
def test_should_keep_question_mark_when_other_words_follow_the_question_word(text):
    """Deliberately conservative: only a question word at the very end is trusted."""
    assert normalize_thai_punctuation(text, True) == text


@pytest.mark.parametrize("text", ["ตัวสำรอง?", "หา?", "จริงดิ?", "เขามาแล้ว?"])
def test_should_keep_question_mark_when_no_question_word_precedes_it(text):
    assert normalize_thai_punctuation(text, True) == text


@pytest.mark.parametrize("text,expected", [
    ("ส่งเข้าคุกให้ได้!!", "ส่งเข้าคุกให้ได้!"),
    ("หยุดนะ!!!", "หยุดนะ!"),
    ("อะไรนะ??", "อะไรนะ"),
    ("ว่าไงนะ?!", "ว่าไงนะ?!"),
    ("ทำไมงั้นเหรอ?!", "ทำไมงั้นเหรอ!"),
    ("พูดอะไร!?", "พูดอะไร!"),
    ("เดี๋ยว⁈", "เดี๋ยว?!"),
    ("ไม่！", "ไม่!"),
])
def test_should_leave_one_mark_at_each_ending(text, expected):
    assert normalize_thai_punctuation(text, True) == expected


@pytest.mark.parametrize("text", ["กลับมานี่นะโว้ย!", "ครับ!", "...", "จ้ะ! พี่ก็เหมือนกันนะคะ!"])
def test_should_never_remove_a_single_exclamation_mark(text):
    assert normalize_thai_punctuation(text, True) == text


def test_should_handle_marks_in_the_middle_of_a_bubble():
    assert normalize_thai_punctuation("ทำไมล่ะ? บอกมาสิ!!", True) == "ทำไมล่ะ บอกมาสิ!"


def test_should_keep_line_break_tags():
    assert normalize_thai_punctuation("ไม่เหนื่อย[BR]เหรอ?", True) == "ไม่เหนื่อย[BR]เหรอ"


@pytest.mark.parametrize("value", ["ไม่เหนื่อยเหรอ?", "", None, {"type": "doc"}])
def test_should_change_nothing_when_disabled_or_not_plain_text(value):
    assert normalize_thai_punctuation(value, False) == value
    if not isinstance(value, str):
        assert normalize_thai_punctuation(value, True) == value


@pytest.mark.parametrize("code,expected", [("THA", True), ("th_TH", True), ("Thai", True), ("ENG", False), ("", False), (None, False)])
def test_should_recognize_thai_language_codes(code, expected):
    assert is_thai_language(code) is expected
