import io
import os
import unittest

from manga_translator.rendering.text_render import (
    _sanitized_font_bytes,
    qt_family_is_ambiguous,
    strip_qt_foundry_brackets,
)
from manga_translator.utils import BASE_PATH

BRACKETED_FONT = os.path.join(BASE_PATH, 'fonts', '[toolbox]书卷楷-简繁(v2.4).ttf')
CLEAN_FONT = os.path.join(BASE_PATH, 'fonts', 'Prompt-Regular.ttf')


class TestQtFamilyAmbiguity(unittest.TestCase):
    def test_leading_bracket_is_ambiguous(self):
        # Qt's parseFontName splits "[X]Y" into an empty family name + the foundry X
        self.assertTrue(qt_family_is_ambiguous('[工具箱]书卷楷-简繁'))
        self.assertTrue(qt_family_is_ambiguous('[toolbox]FangYuan-GBK W7'))
        self.assertTrue(qt_family_is_ambiguous('  [toolbox]QiangDiao-W'))

    def test_normal_names_are_not_ambiguous(self):
        self.assertFalse(qt_family_is_ambiguous('工具箱书卷楷-简繁'))
        self.assertFalse(qt_family_is_ambiguous('Microsoft YaHei UI'))
        # With the foundry written at the end, the family name is not empty and Qt can parse out "Helvetica"
        self.assertFalse(qt_family_is_ambiguous('Helvetica [Cronyx]'))
        self.assertFalse(qt_family_is_ambiguous(''))
        self.assertFalse(qt_family_is_ambiguous('[未闭合'))
        self.assertFalse(qt_family_is_ambiguous('闭合]在前'))

    def test_strip_brackets(self):
        self.assertEqual(strip_qt_foundry_brackets('[工具箱]书卷楷-简繁'), '工具箱书卷楷-简繁')
        self.assertEqual(strip_qt_foundry_brackets(' [toolbox]X '), 'toolboxX')
        self.assertEqual(strip_qt_foundry_brackets(''), '')


@unittest.skipUnless(os.path.exists(BRACKETED_FONT), 'bracketed toolbox font not present')
class TestSanitizedFontBytes(unittest.TestCase):
    def test_bracketed_font_is_rewritten(self):
        from fontTools.ttLib import TTFont

        data, original_names = _sanitized_font_bytes(BRACKETED_FONT)
        self.assertIsNotNone(data)
        self.assertIn('[工具箱]书卷楷-简繁', original_names)

        source = TTFont(BRACKETED_FONT, lazy=True)
        rewritten = TTFont(io.BytesIO(data), lazy=True)
        try:
            for record in rewritten['name'].names:
                if record.nameID in (1, 4, 16):
                    value = record.toUnicode()
                    self.assertNotIn('[', value)
                    self.assertNotIn(']', value)
            # The missing English preferred family name (nameID 16) is filled in; offscreen freetype relies on it to choose the name
            self.assertIsNotNone(rewritten['name'].getName(16, 3, 1, 0x409))
            # Only the name table is touched; the glyph data is unchanged
            self.assertEqual(rewritten['maxp'].numGlyphs, source['maxp'].numGlyphs)
        finally:
            source.close()
            rewritten.close()

    @unittest.skipUnless(os.path.exists(CLEAN_FONT), 'clean control font not present')
    def test_clean_font_needs_no_rewrite(self):
        data, original_names = _sanitized_font_bytes(CLEAN_FONT)
        self.assertIsNone(data)
        self.assertEqual(original_names, [])


if __name__ == '__main__':
    unittest.main()
