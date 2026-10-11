"""Regression test for the automatic BR clean-up of a single region under AI line breaking.

When AI line breaking (disable_auto_wrap) is on, `_validate_br_markers` cleans the surplus [BR]/<br>/【BR】 markers
out of the translation result of a single region (region_count < 2), making it one line;
this clean-up is independent of the "AI line break check" (check_br_and_retry) switch.
"""

from types import MethodType, SimpleNamespace

from manga_translator.translators.common import CommonTranslator


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _make_region(n_lines):
    return SimpleNamespace(lines=[None] * n_lines)


def _make_ctx(ai_break, check_br, regions):
    return SimpleNamespace(
        config=SimpleNamespace(
            render=SimpleNamespace(disable_auto_wrap=ai_break, check_br_and_retry=check_br)
        ),
        text_regions=regions,
    )


def _make_validator():
    validator = SimpleNamespace(logger=_Logger())
    validator._validate_br_markers = MethodType(CommonTranslator._validate_br_markers, validator)
    return validator


def test_single_region_br_cleaned_even_when_check_disabled():
    validator = _make_validator()
    translations = ["你好<br>世界"]
    result = validator._validate_br_markers(
        translations,
        ctx=_make_ctx(ai_break=True, check_br=False, regions=[_make_region(1)]),
    )
    assert result is True
    assert translations[0] == "你好世界"


def test_single_region_all_br_variants_cleaned():
    validator = _make_validator()
    translations = ["第一行 [BR] 第二行【BR】第三行<br/>尾"]
    validator._validate_br_markers(
        translations,
        ctx=_make_ctx(ai_break=True, check_br=False, regions=[_make_region(1)]),
    )
    assert translations[0] == "第一行第二行第三行尾"


def test_multi_line_region_br_preserved():
    validator = _make_validator()
    translations = ["你好[BR]世界"]
    validator._validate_br_markers(
        translations,
        ctx=_make_ctx(ai_break=True, check_br=False, regions=[_make_region(2)]),
    )
    assert translations[0] == "你好[BR]世界"


def test_mixed_single_and_multi_region():
    validator = _make_validator()
    translations = ["单<br>行", "两行没有br", "第二行也没有br", "第<br/>二<br>个"]
    # Both multi-region items lack BR, which is above the tolerance -> a retry is triggered
    result = validator._validate_br_markers(
        translations,
        ctx=_make_ctx(
            ai_break=True,
            check_br=True,
            regions=[_make_region(1), _make_region(2), _make_region(2), _make_region(1)],
        ),
    )
    assert result is False
    assert translations[0] == "单行"
    assert translations[3] == "第二个"


def test_no_cleanup_when_ai_break_disabled():
    validator = _make_validator()
    translations = ["手动<br>换行"]
    validator._validate_br_markers(
        translations,
        ctx=_make_ctx(ai_break=False, check_br=False, regions=[_make_region(1)]),
    )
    assert translations[0] == "手动<br>换行"