"""The steps that clean up text regions after text lines are merged."""
import _bootstrap  # noqa: F401, I001

from types import SimpleNamespace

import pytest

from manga_translator.manga_translator import MangaTranslator


@pytest.mark.parametrize("text, expected", [
    ("plain text", "plain text"),
    ("(paired)", "(paired)"),
    ("(never closed", "never closed"),
    ("never opened)", "never opened"),
    ("「wrong closer)", "「wrong closer」"),
    ("(a) and b)", "(a) and b"),
    ("", ""),
])
def test_bracket_repair_should_drop_unpaired_and_correct_mismatched_brackets(text, expected):
    assert MangaTranslator._repair_unpaired_brackets(text) == expected


def test_skipped_languages_should_be_removed_from_the_text_lines():
    english = SimpleNamespace(text="This is clearly an English sentence about the weather today.")
    japanese = SimpleNamespace(text="これは日本語の文章です。今日は天気がいいですね。")
    config = SimpleNamespace(translator=SimpleNamespace(skip_lang="ENG"))
    ctx = SimpleNamespace(textlines=[english, japanese])

    MangaTranslator._drop_textlines_in_skipped_languages(config, ctx)

    assert ctx.textlines == [japanese]


def test_skipped_languages_should_keep_every_line_when_none_match():
    english = SimpleNamespace(text="This is clearly an English sentence about the weather today.")
    config = SimpleNamespace(translator=SimpleNamespace(skip_lang="KOR, JPN"))
    ctx = SimpleNamespace(textlines=[english])

    MangaTranslator._drop_textlines_in_skipped_languages(config, ctx)

    assert ctx.textlines == [english]
