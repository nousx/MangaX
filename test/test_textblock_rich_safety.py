"""Safety tests of rich text in TextBlock (F04a/F19/F02/F30).

Covered:
- an invalid translation_rich does not blow up the constructor: the style is dropped and the region is kept (F04a)
- valid rich text is stored normally
- assigning an equal value through the translation setter does not clear translation_rich (F02)
- for the old 'hr' direction, the BR→rich conversion and the string path both keep the Unicode logical order (F30)

Run: uv run --no-sync pytest test/test_textblock_rich_safety.py
"""

import unittest

from manga_translator.rendering.rich_text import RICH_TEXT_FORMAT
from manga_translator.utils import TextBlock

_LINES = [[[0, 0], [100, 0], [100, 80], [0, 80]]]


def _make_block(**kwargs):
    return TextBlock(lines=_LINES, texts=["原文"], **kwargs)


def _valid_document():
    return {
        "format": RICH_TEXT_FORMAT,
        "blocks": [
            {
                "type": "paragraph",
                "inlines": [
                    {"type": "text", "text": "普通", "style": {}},
                    {"type": "text", "text": "红字", "style": {"color": "#ff0000"}},
                ],
            }
        ],
    }


# The shape of the old experimental format: unknown keys such as 'spans'/'source' are rejected by strict parsing (ValueError)
_MALFORMED_DOCUMENTS = [
    {"format": RICH_TEXT_FORMAT, "source": "红字", "blocks": []},
    {
        "format": RICH_TEXT_FORMAT,
        "blocks": [
            {"type": "paragraph", "spans": [{"type": "text", "text": "红", "style": {}}]}
        ],
    },
    {"format": RICH_TEXT_FORMAT},  # blocks is missing
    {"format": RICH_TEXT_FORMAT, "blocks": "oops"},  # blocks is not a list (TypeError)
    {
        "format": RICH_TEXT_FORMAT,
        "blocks": [
            {
                "type": "paragraph",
                "inlines": [
                    # fontPath is an old field that was removed (the protocol only accepts fontFamily now)
                    {"type": "text", "text": "红", "style": {"fontPath": "C:/x.ttf"}}
                ],
            }
        ],
    },
]


class MalformedRichTranslationTest(unittest.TestCase):
    """F04a: an invalid translation_rich only loses the style and never the region."""

    def test_malformed_translation_rich_keeps_region(self):
        for malformed in _MALFORMED_DOCUMENTS:
            with self.subTest(document=malformed):
                with self.assertLogs("manga-translator.textblock", level="WARNING"):
                    region = _make_block(
                        translation="保留的译文",
                        translation_raw="原始译文",
                        translation_rich=malformed,
                    )
                self.assertIsNone(region.translation_rich)
                self.assertEqual(region.translation, "保留的译文")
                self.assertEqual(region.translation_raw, "原始译文")
                self.assertEqual(region.text, "原文")
                self.assertNotIn("translation_rich", region.to_dict())

    def test_malformed_rich_passed_as_translation_keeps_region(self):
        with self.assertLogs("manga-translator.textblock", level="WARNING"):
            region = _make_block(translation=_MALFORMED_DOCUMENTS[0])
        self.assertIsNone(region.translation_rich)
        self.assertEqual(region.translation, "")
        self.assertEqual(region.text, "原文")

    def test_load_style_kwargs_survive_malformed_rich(self):
        """The load_text case: after rich parsing fails, the other fields are constructed as usual."""
        with self.assertLogs("manga-translator.textblock", level="WARNING"):
            region = _make_block(
                translation="译文",
                translation_rich=_MALFORMED_DOCUMENTS[1],
                font_size=32,
                direction="v",
                target_lang="CHS",
            )
        self.assertEqual(region.font_size, 32)
        self.assertEqual(region.direction, "v")
        self.assertEqual(region.target_lang, "CHS")

    def test_removed_apis_are_gone(self):
        """F19: the API with zero callers has been removed."""
        self.assertFalse(hasattr(TextBlock, "clear_translation_rich"))
        region = _make_block(translation="x")
        with self.assertRaises(TypeError):
            region.set_translation_rich(_valid_document(), sync_plain_when_empty=True)


class ValidRichTranslationTest(unittest.TestCase):
    def test_valid_rich_document_is_stored(self):
        document = _valid_document()
        region = _make_block(translation_rich=document)

        self.assertEqual(region.translation_rich, document)
        self.assertEqual(region.translation, "普通红字")
        self.assertEqual(region.translation_raw, "普通红字")
        self.assertEqual(region.get_translation_for_rendering(), document)
        self.assertEqual(region.to_dict()["translation_rich"], document)

    def test_valid_rich_with_plain_translation(self):
        document = _valid_document()
        region = _make_block(translation="纯文本译文", translation_rich=document)

        self.assertEqual(region.translation, "纯文本译文")
        self.assertEqual(region.translation_rich, document)

    def test_rich_document_passed_as_translation(self):
        document = _valid_document()
        region = _make_block(translation=document)

        self.assertEqual(region.translation, "普通红字")
        self.assertEqual(region.translation_rich, document)

    def test_plain_break_document_is_runtime_only(self):
        region = _make_block(translation="第一行[BR]第二行")

        self.assertTrue(region.ensure_translation_rich_from_legacy_breaks())
        self.assertIsNotNone(region.translation_rich)
        self.assertNotIn("translation_rich", region.to_dict())

    def test_non_equivalent_empty_paragraph_is_persisted(self):
        document = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {"type": "paragraph", "inlines": [{"type": "text", "text": "甲", "style": {}}]},
                {"type": "paragraph", "inlines": []},
                {"type": "paragraph", "inlines": [{"type": "text", "text": "乙", "style": {}}]},
            ],
        }
        region = _make_block(translation="甲[BR]乙", translation_rich=document)

        self.assertEqual(region.to_dict()["translation_rich"], document)


class TranslationSetterRichInvalidationTest(unittest.TestCase):
    """F02: assigning an equal value is not an edit and does not clear translation_rich."""

    def test_equal_assignment_keeps_rich(self):
        region = _make_block(translation_rich=_valid_document())
        plain = region.translation

        region.translation = plain  # Written back unchanged, as when OpenCC has no hit or the post-dictionary has no replacement

        self.assertIsNotNone(region.translation_rich)
        self.assertEqual(region.translation, plain)

    def test_changed_assignment_clears_rich(self):
        region = _make_block(translation_rich=_valid_document())
        plain = region.translation

        region.translation = plain + "！"

        self.assertIsNone(region.translation_rich)
        self.assertEqual(region.translation, plain + "！")

    def test_raw_semantics_unchanged(self):
        region = _make_block(translation="旧", translation_raw="原始")
        region.translation = "新"
        self.assertEqual(region.translation_raw, "原始")


class RtlLegacyBreakConversionTest(unittest.TestCase):
    """F30: for the old 'hr' direction, the BR→rich conversion and the string render path both keep the logical order."""

    @staticmethod
    def _paragraph_texts(document):
        return [
            "".join(inline["text"] for inline in block["inlines"])
            for block in document["blocks"]
        ]

    def test_hr_conversion_matches_string_path_logical_order(self):
        source_lines = ["abc123", "مرحبا abc", "123", "ab"]
        region = _make_block(
            translation="[BR]".join(source_lines),
            direction="hr",
            target_lang="ARA",
        )

        self.assertTrue(region.ensure_translation_rich_from_legacy_breaks())
        rich_lines = self._paragraph_texts(region.translation_rich)

        # Compared line by line with the single-line string path: get_translation_for_rendering is the behaviour baseline
        expected = []
        for line in source_lines:
            single = _make_block(translation=line, direction="hr", target_lang="ARA")
            self.assertIsNone(single.translation_rich)
            expected.append(single.get_translation_for_rendering())
        self.assertEqual(rich_lines, expected)

        # Bidirectional layout is left to the renderer; old direction aliases are only normalised to horizontal, and characters are not reversed in advance.
        self.assertEqual(rich_lines, source_lines)
        self.assertEqual(region.direction, "h")

    def test_h_direction_conversion_keeps_order(self):
        region = _make_block(translation="abc123[BR]def", direction="h")

        self.assertTrue(region.ensure_translation_rich_from_legacy_breaks())

        self.assertEqual(
            self._paragraph_texts(region.translation_rich),
            ["abc123", "def"],
        )

    def test_conversion_skipped_when_rich_exists(self):
        document = _valid_document()
        region = _make_block(
            translation="abc[BR]def",
            translation_rich=document,
            direction="hr",
            target_lang="ARA",
        )
        self.assertFalse(region.ensure_translation_rich_from_legacy_breaks())
        self.assertEqual(region.translation_rich, document)


if __name__ == "__main__":
    unittest.main()
