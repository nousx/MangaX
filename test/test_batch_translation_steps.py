"""The steps of translating a batch that run after the translator has answered."""
import _bootstrap  # noqa: F401, I001

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from manga_translator.config import Config, Translator, TranslatorConfig
from manga_translator.manga_translator import MangaTranslator


@pytest.fixture(scope="module")
def translator() -> MangaTranslator:
    return MangaTranslator(params={"translator": "none", "use_gpu": False, "filter_text_enabled": False})


def region(text: str, translation: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, translation=translation)


def page(*regions) -> SimpleNamespace:
    return SimpleNamespace(text_regions=list(regions))


def config(**translator_settings) -> Config:
    return Config(translator=TranslatorConfig(translator=Translator.openai, **translator_settings))


def test_filter_should_drop_empty_numeric_and_unchanged_translations(translator):
    kept = region("Hello there", "สวัสดี")
    ctx = page(kept, region("Hello", "   "), region("Page 12", "12"), region("Same", "same"))

    translator._filter_translated_regions([(ctx, config())])

    assert ctx.text_regions == [kept]


def test_filter_should_keep_unchanged_text_when_the_setting_asks_for_it(translator):
    unchanged = region("Same", "same")
    ctx = page(unchanged)

    translator._filter_translated_regions([(ctx, config(no_text_lang_skip=True))])

    assert ctx.text_regions == [unchanged]


def test_filter_should_leave_a_page_without_regions_alone(translator):
    ctx = page()

    translator._filter_translated_regions([(ctx, config())])

    assert ctx.text_regions == []


def test_language_check_should_be_skipped_for_ten_regions_or_fewer(translator, monkeypatch):
    calls = []

    async def ratio(*args, **kwargs):
        calls.append(args)
        return True

    monkeypatch.setattr(translator, "_check_target_language_ratio", ratio)
    ctx = page(*[region(f"line {n}", f"บรรทัด {n}") for n in range(10)])

    asyncio.run(translator._check_batch_target_language([(ctx, config())]))

    assert calls == []


def test_language_check_should_translate_again_until_the_batch_passes(translator, monkeypatch):
    verdicts = iter([False, True])
    retries = []

    async def ratio(regions, target_lang, min_ratio=0.5):
        return next(verdicts)

    async def translate_again(texts, sample_config, ctx):
        retries.append(list(texts))
        return [f"แปลใหม่ {index}" for index in range(len(texts))]

    monkeypatch.setattr(translator, "_check_target_language_ratio", ratio)
    monkeypatch.setattr(translator, "_batch_translate_texts", translate_again)
    ctx = page(*[region(f"line {n}", f"line {n}") for n in range(11)])

    asyncio.run(translator._check_batch_target_language([(ctx, config(post_check_max_retry_attempts=3))]))

    assert len(retries) == 1
    assert [item.translation for item in ctx.text_regions] == [f"แปลใหม่ {index}" for index in range(11)]


def test_language_check_should_stop_after_the_allowed_number_of_retries(translator, monkeypatch):
    retries = []

    async def ratio(regions, target_lang, min_ratio=0.5):
        return False

    async def translate_again(texts, sample_config, ctx):
        retries.append(len(texts))
        return list(texts)

    monkeypatch.setattr(translator, "_check_target_language_ratio", ratio)
    monkeypatch.setattr(translator, "_batch_translate_texts", translate_again)
    ctx = page(*[region(f"line {n}", f"line {n}") for n in range(11)])

    asyncio.run(translator._check_batch_target_language([(ctx, config(post_check_max_retry_attempts=2))]))

    assert retries == [11, 11]


def test_high_quality_data_should_number_the_texts_across_the_pages_of_a_batch():
    first = SimpleNamespace(text_regions=[region("a", ""), region("b", "")], input="first image",
                            upscaled=np.zeros((30, 20, 3), dtype=np.uint8))
    empty = SimpleNamespace(text_regions=[], input="empty image")
    second = SimpleNamespace(text_regions=[region("c", "")], input="second image",
                             img_rgb=np.zeros((50, 40, 3), dtype=np.uint8))
    merged = SimpleNamespace()

    MangaTranslator._attach_high_quality_batch_data([(first, None), (empty, None), (second, None)], merged)

    data = merged.high_quality_batch_data
    assert [item["text_order"] for item in data] == [[1, 2], [3]]
    assert [item["upscaled_size"] for item in data] == [(30, 20), (50, 40)]
    assert [item["original_texts"] for item in data] == [["a", "b"], ["c"]]
    assert [item["image"] for item in data] == ["first image", "second image"]
