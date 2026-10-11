import _bootstrap  # noqa: F401
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

import manga_translator.rendering as rendering_module
from manga_translator.config import Config
from manga_translator.rendering import calc_box_from_font, render, text_render
from manga_translator.rendering.rich_text import (
    RICH_TEXT_FORMAT,
    ensure_rich_text_document,
    legacy_line_breaks_to_document,
)
from manga_translator.rendering.text_render._layout import (
    CJK_Compatibility_Forms_translate,
    _build_tcy_plan,
    _build_vertical_char_plan,
    _vertical_free_rotation_advance,
)
from manga_translator.rendering.text_render._vertical_types import VerticalGlyphBase
from manga_translator.rendering.text_replacement_layout import (
    ReplacementLayoutRecord,
    sync_translation_raw_from_layout,
)
from manga_translator.utils import TextBlock


def _sample_document():
    return {
        "format": RICH_TEXT_FORMAT,
        "blocks": [
            {
                "type": "paragraph",
                "inlines": [
                    {"type": "text", "text": "普通", "style": {}},
                    {
                        "type": "text",
                        "text": "红字",
                        "style": {
                            "color": "#ff0000",
                            "stroke": {"color": "#000000", "width": 2},
                        },
                    },
                    {
                        "type": "ruby",
                        "base": [
                            {"type": "text", "text": "漢字", "style": {}},
                        ],
                        "text": [
                            {"type": "text", "text": "かんじ", "style": {}},
                        ],
                    },
                    {
                        "type": "text",
                        "text": "点",
                        "style": {"emphasis": True},
                    },
                ],
            }
        ],
    }


class RichTextRenderingTest(unittest.TestCase):
    def test_vertical_auto_rotation_character_policy(self):
        # The special path of the engine only rotates the four curly quotes + the four Japanese corner brackets (see _VERTICAL_ROTATE_CHARS);
        # the other characters (round and lenticular brackets included) are not rotated in the rendering layer; ordinary automatic rotation moved to rich_text_rules.yaml.
        for char in "AaZz019,.:;?!-~()\"'ー⸺–—～﹏…⋯●•（）【】︱︲":
            self.assertEqual(CJK_Compatibility_Forms_translate(char, 1)[1], 0, char)
        for char in "“‘”’「『」』":
            self.assertEqual(CJK_Compatibility_Forms_translate(char, 1)[1], 90, char)

    def test_free_rotation_advance_projects_between_axes(self):
        base = VerticalGlyphBase(
            translated="字",
            rot_degree=0,
            bitmap=None,
            advance_y=40,
            ink_x=0.0,
            ink_w=30.0,
            y=0,
            advance_x=30,
            glyph_left=0.0,
            frame_width=30,
        )
        # 0 degrees keeps the vertical advance
        self.assertEqual(_vertical_free_rotation_advance(base, 0), 40)
        # +-90 degrees takes the horizontal advance (in line with BallonsTranslator; it may be smaller than the original vertical advance)
        self.assertEqual(_vertical_free_rotation_advance(base, 90), 30)
        self.assertEqual(_vertical_free_rotation_advance(base, -90), 30)
        # An angle in between is the sum of the projections on the two axes, between the two end values and not below the smaller one
        mid = _vertical_free_rotation_advance(base, 45)
        self.assertGreater(mid, 30)
        self.assertEqual(mid, _vertical_free_rotation_advance(base, -45))

    def setUp(self):
        # Rendering mutates thread-local font state per region.  Reset for every
        # test so font-specific geometry assertions are order independent.
        text_render.set_font("Arial-Unicode-Regular.ttf")
        text_render.set_bold(False)

    def test_legacy_break_markers_become_paragraph_blocks(self):
        document = legacy_line_breaks_to_document(
            "红[BR]蓝<br>绿【BR】紫\n黑"
        ).to_dict()

        self.assertEqual(document["format"], RICH_TEXT_FORMAT)
        self.assertEqual(
            [
                block["inlines"][0]["text"] if block["inlines"] else ""
                for block in document["blocks"]
            ],
            ["红", "蓝", "绿", "紫", "黑"],
        )

    def test_legacy_break_markers_preserve_surrounding_spaces(self):
        document = legacy_line_breaks_to_document(
            "第一行  [BR]  第二行\t<br />\t第三行 【BR】 第四行"
        ).to_dict()

        self.assertEqual(
            [
                block["inlines"][0]["text"] if block["inlines"] else ""
                for block in document["blocks"]
            ],
            ["第一行  ", "  第二行\t", "\t第三行 ", " 第四行"],
        )


    def test_rich_text_schema_rejects_noncanonical_fields(self):
        with self.assertRaises(ValueError):
            ensure_rich_text_document(
                {
                    "format": RICH_TEXT_FORMAT,
                    "document": {"blocks": []},
                }
            )

        with self.assertRaises(ValueError):
            ensure_rich_text_document(
                {
                    "format": RICH_TEXT_FORMAT,
                    "source": "红字",
                    "blocks": [],
                }
            )

        with self.assertRaises(ValueError):
            ensure_rich_text_document(
                {
                    "format": RICH_TEXT_FORMAT,
                    "blocks": [
                        {
                            "type": "paragraph",
                            "spans": [
                                {"type": "text", "text": "红", "style": {}},
                            ],
                        }
                    ],
                }
            )

        family_document = ensure_rich_text_document(
            {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [
                            {
                                "type": "text",
                                "text": "红",
                                "style": {"fontFamily": "Microsoft YaHei UI"},
                            },
                        ],
                    }
                ],
            }
        )
        self.assertEqual(
            family_document.to_dict()["blocks"][0]["inlines"][0]["style"]["fontFamily"],
            "Microsoft YaHei UI",
        )

        with self.assertRaises(ValueError):
            ensure_rich_text_document(
                {
                    "format": RICH_TEXT_FORMAT,
                    "blocks": [
                        {
                            "type": "paragraph",
                            "inlines": [
                                {
                                    "type": "text",
                                    "text": "红",
                                    "style": {"font_size": 32},
                                },
                            ],
                        }
                    ],
                }
            )

    def test_vertical_advance_forces_slot_and_ink_center(self):
        import cv2

        document = ensure_rich_text_document(
            self._single_span_document(
                "字", {"verticalAdvance": "half", "transform": {"rotation": 45}}
            )
        )
        span = document.blocks[0].spans[0]
        self.assertEqual(span.style.vertical_advance, "half")
        self.assertEqual(
            document.to_dict()["blocks"][0]["inlines"][0]["style"]["verticalAdvance"],
            "half",
        )

        with self.assertRaises(ValueError):
            ensure_rich_text_document(
                self._single_span_document("字", {"verticalAdvance": "quarter"})
            )

        for char in "字。︙“":
            self.assertEqual(
                text_render._vertical_base(32, char, 1.0, 0.0, "half").advance_y, 16
            )
        self.assertEqual(
            text_render._vertical_base(32, "字", 1.0, 0.0, "full").advance_y, 32
        )

        half = text_render._vertical_base(32, "字", 1.0, 0.0, "half")
        _x, ink_y, _width, ink_height = cv2.boundingRect(cv2.findNonZero(half.bitmap))
        self.assertAlmostEqual(
            half.y + ink_y,
            (half.advance_y - ink_height) / 2.0,
            delta=1.0,
        )

        punctuation = VerticalGlyphBase("。", 0, None, 32, 2, 8, 0, 20, 1, 32)
        self.assertEqual(
            text_render._vertical_char_bitmap_x(
                0.0, 32.0, punctuation, ink_center=True
            ),
            10.0,
        )

        plan = _build_vertical_char_plan(span, half, 32, (0, 0, 0), None, 0.0)
        self.assertEqual(plan.advance_y, 16)
        self.assertEqual(_build_tcy_plan(span, 32, 0.0, None, 1.0).advance_main, 16)

    def test_textblock_stores_rich_text_as_canonical_dict(self):
        document = _sample_document()
        region = TextBlock(
            lines=[[[0, 0], [100, 0], [100, 80], [0, 80]]],
            texts=["原文"],
            translation_rich=ensure_rich_text_document(document),
        )

        self.assertEqual(region.translation, "普通红字漢字点")
        self.assertIsInstance(region.translation_rich, dict)
        self.assertEqual(region.translation_rich, document)
        self.assertEqual(region.translation_raw, "普通红字漢字点")
        self.assertEqual(region.get_translation_for_rendering(), document)
        self.assertEqual(region.to_dict()["translation"], "普通红字漢字点")
        self.assertEqual(region.to_dict()["translation_rich"], document)

    def test_textblock_normalizes_rich_text_assignment(self):
        region = TextBlock(
            lines=[[[0, 0], [100, 0], [100, 80], [0, 80]]],
            texts=["原文"],
            translation="旧文本",
            translation_raw="原始文本",
        )
        document = ensure_rich_text_document(_sample_document())

        region.set_translation_rich(document, sync_plain=True)

        self.assertEqual(region.translation, "普通红字漢字点")
        self.assertEqual(region.translation_rich, document.to_dict())
        self.assertEqual(region.translation_raw, "原始文本")

    def test_structured_document_measures_as_multiple_lines(self):
        document = legacy_line_breaks_to_document("第一行[BR]第二行").to_dict()

        width, height, line_count, _ = calc_box_from_font(
            32,
            document,
            True,
            line_spacing=1.0,
        )

        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        self.assertEqual(line_count, 2)

    def test_low_level_rich_text_metrics_return_one_entry_per_paragraph(self):
        document = legacy_line_breaks_to_document("第一行[BR]第二行").to_dict()

        horizontal_lines, horizontal_widths = text_render.calc_horizontal(
            32,
            document,
            max_width=9999,
            max_height=9999,
        )
        vertical_lines, vertical_heights, vertical_widths = (
            text_render.calc_vertical_metrics(
                32,
                document,
                max_height=9999,
            )
        )

        self.assertEqual(len(horizontal_lines), 2)
        self.assertEqual(len(horizontal_widths), 2)
        self.assertTrue(all(width > 0 for width in horizontal_widths))
        self.assertEqual(len(vertical_lines), 2)
        self.assertEqual(len(vertical_heights), 2)
        self.assertEqual(len(vertical_widths), 2)
        self.assertTrue(all(height > 0 for height in vertical_heights))

    def test_structured_document_renders_horizontal_and_vertical(self):
        document = _sample_document()

        horizontal = text_render.put_text_horizontal(
            32,
            document,
            420,
            180,
            "left",
            False,
            (255, 255, 255),
            (0, 0, 0),
            line_spacing=1.0,
        )
        vertical = text_render.put_text_vertical(
            32,
            document,
            260,
            "left",
            (255, 255, 255),
            (0, 0, 0),
            line_spacing=1.0,
        )

        self.assertIsNotNone(horizontal)
        self.assertIsNotNone(vertical)
        self.assertEqual(horizontal.shape[2], 4)
        self.assertEqual(vertical.shape[2], 4)
        self.assertGreater(int(horizontal[:, :, 3].max()), 0)
        self.assertGreater(int(vertical[:, :, 3].max()), 0)

    def test_vertical_column_keeps_thickness_separate_from_annotations(self):
        from manga_translator.rendering.text_render._vertical_types import (
            VerticalColumnPlan,
        )

        document = ensure_rich_text_document(
            {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [
                            {"type": "text", "text": "前", "style": {}},
                            {
                                "type": "ruby",
                                "base": [
                                    {
                                        "type": "text",
                                        "text": "中",
                                        "style": {"emphasis": True},
                                    }
                                ],
                                "text": [{"type": "text", "text": "なか", "style": {}}],
                            },
                            {"type": "text", "text": "後", "style": {}},
                        ],
                    }
                ],
            }
        )

        layout = text_render._build_rich_vertical_layout(
            document,
            32,
            0.07,
            (255, 255, 255),
            (0, 0, 0),
            1.0,
        )[0]

        self.assertIsInstance(layout, VerticalColumnPlan)
        self.assertEqual(layout.thickness, 32)
        self.assertGreater(layout.ruby_cross_extent, 0)
        self.assertGreater(layout.annotation_cross_extent, layout.ruby_cross_extent)
        self.assertIsInstance(layout.items, tuple)
        self.assertIsInstance(layout.ruby_plans, tuple)
        self.assertIsInstance(layout.emphasis_plans, tuple)

    def test_vertical_bold_does_not_change_body_column_width(self):
        plain = ensure_rich_text_document(
            {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [
                            {"type": "text", "text": "能", "style": {}},
                            {"type": "text", "text": "写", "style": {}},
                            {"type": "text", "text": "出", "style": {}},
                        ],
                    }
                ],
            }
        )
        bold = ensure_rich_text_document(
            {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [
                            {"type": "text", "text": "能", "style": {}},
                            {"type": "text", "text": "写", "style": {"bold": True}},
                            {"type": "text", "text": "出", "style": {}},
                        ],
                    }
                ],
            }
        )

        plain_layout = text_render._build_rich_vertical_layout(
            plain, 32, 0.07, (255, 255, 255), (0, 0, 0), 1.0
        )[0]
        bold_layout = text_render._build_rich_vertical_layout(
            bold, 32, 0.07, (255, 255, 255), (0, 0, 0), 1.0
        )[0]
        plain_geometry = text_render._rich_vertical_layout_geometry(
            [plain_layout], 32, 1.0
        )
        bold_geometry = text_render._rich_vertical_layout_geometry(
            [bold_layout], 32, 1.0
        )

        self.assertEqual(bold_layout.thickness, plain_layout.thickness)
        self.assertEqual(bold_geometry["layout_width"], plain_geometry["layout_width"])
        self.assertGreater(bold_geometry["paint_width"], plain_geometry["paint_width"])

    def test_vertical_upright_character_centers_by_advance_not_ink(self):
        from manga_translator.rendering.text_render._vertical_types import (
            VerticalGlyphBase,
        )

        base = VerticalGlyphBase("字", 0, None, 32, 2, 8, 0, 20, 3, 32)

        x = text_render._vertical_char_bitmap_x(0.0, 32.0, base)

        # The left edge of the advance box is 6, plus the glyph left bearing of 3.
        self.assertEqual(x, 9.0)
        # The old ink-centred result was (32-8)/2-2 = 10; this makes sure the old logic is not back.
        self.assertNotEqual(x, 10.0)

    def test_vertical_question_mark_centers_by_punctuation_ink(self):
        from manga_translator.rendering.text_render._vertical_types import (
            VerticalGlyphBase,
        )

        base = VerticalGlyphBase("？", 0, None, 32, 2, 8, 0, 20, 1, 32)

        x = text_render._vertical_char_bitmap_x(0.0, 32.0, base)

        self.assertEqual(x, 10.0)

    def test_vertical_rich_layout_uses_fixed_column_thickness(self):
        plain = ensure_rich_text_document(
            {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [{"type": "text", "text": "能写出", "style": {}}],
                    },
                    {
                        "type": "paragraph",
                        "inlines": [{"type": "text", "text": "好歌词", "style": {}}],
                    },
                ],
            }
        )
        large = ensure_rich_text_document(
            {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [
                            {"type": "text", "text": "能", "style": {}},
                            {"type": "text", "text": "写", "style": {"fontSize": 64}},
                            {"type": "text", "text": "出", "style": {}},
                        ],
                    },
                    {
                        "type": "paragraph",
                        "inlines": [{"type": "text", "text": "好歌词", "style": {}}],
                    },
                ],
            }
        )
        tcy = ensure_rich_text_document(
            {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [
                            {"type": "text", "text": "年", "style": {}},
                            {
                                "type": "tcy",
                                "content": [
                                    {"type": "text", "text": "2026", "style": {}}
                                ],
                            },
                            {"type": "text", "text": "版", "style": {}},
                        ],
                    },
                    {
                        "type": "paragraph",
                        "inlines": [{"type": "text", "text": "好歌词", "style": {}}],
                    },
                ],
            }
        )

        plain_layouts = text_render._build_rich_vertical_layout(
            plain, 32, 0.07, (255, 255, 255), (0, 0, 0), 1.0
        )
        large_layouts = text_render._build_rich_vertical_layout(
            large, 32, 0.07, (255, 255, 255), (0, 0, 0), 1.0
        )
        tcy_layouts = text_render._build_rich_vertical_layout(
            tcy, 32, 0.07, (255, 255, 255), (0, 0, 0), 1.0
        )
        plain_geometry = text_render._rich_vertical_layout_geometry(
            plain_layouts, 32, 1.0
        )
        large_geometry = text_render._rich_vertical_layout_geometry(
            large_layouts, 32, 1.0
        )
        tcy_geometry = text_render._rich_vertical_layout_geometry(tcy_layouts, 32, 1.0)

        self.assertTrue(all(layout.thickness == 32 for layout in large_layouts))
        self.assertEqual(large_layouts[0].thickness, plain_layouts[0].thickness)
        self.assertEqual(tcy_layouts[0].thickness, plain_layouts[0].thickness)
        self.assertEqual(large_geometry["layout_width"], plain_geometry["layout_width"])
        self.assertEqual(tcy_geometry["layout_width"], plain_geometry["layout_width"])
        self.assertGreater(large_geometry["paint_width"], plain_geometry["paint_width"])
        self.assertGreater(tcy_geometry["paint_width"], plain_geometry["paint_width"])

        columns = text_render._rich_vertical_column_positions(
            large_layouts, large_geometry
        )
        self.assertAlmostEqual(
            columns[0][2] - columns[1][2], 32 + large_geometry["spacing_x"]
        )

    def test_rich_text_ruby_expands_measured_box(self):
        plain = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [{"type": "text", "text": "漢字", "style": {}}],
                }
            ],
        }
        ruby = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {
                            "type": "ruby",
                            "base": [{"type": "text", "text": "漢字", "style": {}}],
                            "text": [{"type": "text", "text": "かんじ", "style": {}}],
                        }
                    ],
                }
            ],
        }

        plain_v_w, plain_v_h, _, _ = calc_box_from_font(
            32, plain, False, line_spacing=1.0
        )
        ruby_v_w, ruby_v_h, _, _ = calc_box_from_font(32, ruby, False, line_spacing=1.0)
        plain_h_w, plain_h_h, _, _ = calc_box_from_font(
            32, plain, True, line_spacing=1.0
        )
        ruby_h_w, ruby_h_h, _, _ = calc_box_from_font(32, ruby, True, line_spacing=1.0)

        self.assertGreater(ruby_v_w, plain_v_w)
        self.assertEqual(ruby_v_h, plain_v_h)
        self.assertEqual(ruby_h_w, plain_h_w)
        self.assertGreater(ruby_h_h, plain_h_h)

    def test_empty_ruby_does_not_expand_measured_box(self):
        plain = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [{"type": "text", "text": "漢字", "style": {}}],
                }
            ],
        }
        empty_ruby = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {
                            "type": "ruby",
                            "base": [{"type": "text", "text": "漢字", "style": {}}],
                            "text": [{"type": "text", "text": "", "style": {}}],
                        }
                    ],
                }
            ],
        }

        self.assertEqual(
            calc_box_from_font(32, empty_ruby, False, line_spacing=1.0),
            calc_box_from_font(32, plain, False, line_spacing=1.0),
        )
        self.assertEqual(
            calc_box_from_font(32, empty_ruby, True, line_spacing=1.0),
            calc_box_from_font(32, plain, True, line_spacing=1.0),
        )

    def test_vertical_ruby_compact_box_reports_body_center_for_anchoring(self):
        plain = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [{"type": "text", "text": "漢字", "style": {}}],
                }
            ],
        }
        ruby = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {
                            "type": "ruby",
                            "base": [{"type": "text", "text": "漢字", "style": {}}],
                            "text": [{"type": "text", "text": "かんじ", "style": {}}],
                        }
                    ],
                }
            ],
        }

        # Plain text: body centre == exact centre of the render box
        plain_w, plain_h, _, plain_body = calc_box_from_font(
            32, plain, False, line_spacing=1.0
        )
        self.assertAlmostEqual(plain_body[0], plain_w / 2.0, places=3)
        self.assertAlmostEqual(plain_body[1], plain_h / 2.0, places=3)

        # Compact box: the ruby only extends to the right, and the body centre is left of the render box centre
        ruby_w, ruby_h, _, ruby_body = calc_box_from_font(
            32, ruby, False, line_spacing=1.0
        )
        self.assertLess(ruby_body[0], ruby_w / 2.0)
        self.assertAlmostEqual(ruby_body[1], ruby_h / 2.0, places=3)

        # After the caller shifts the whole box by the body anchor (body centres aligned on the same anchor (0,0)):
        # the left edge of the body column coincides with that of plain text, and all the ruby space extends to the right.
        plain_points, plain_body_world = calc_box_from_font(
            32, plain, False, line_spacing=1.0, center=(0, 0)
        )
        ruby_points, ruby_body_world = calc_box_from_font(
            32, ruby, False, line_spacing=1.0, center=(0, 0)
        )
        self.assertAlmostEqual(plain_body_world[0], 0.0, places=3)
        self.assertAlmostEqual(plain_body_world[1], 0.0, places=3)

        plain_left = float(plain_points[0, :, 0].min())
        plain_right = float(plain_points[0, :, 0].max())
        ruby_left = float(ruby_points[0, :, 0].min()) - ruby_body_world[0]
        ruby_right = float(ruby_points[0, :, 0].max()) - ruby_body_world[0]
        self.assertAlmostEqual(ruby_left, plain_left, delta=1.0)
        self.assertGreater(ruby_right, plain_right)

    def test_horizontal_body_center_reflects_ruby_and_emphasis(self):
        ruby = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {
                            "type": "ruby",
                            "base": [{"type": "text", "text": "漢字", "style": {}}],
                            "text": [{"type": "text", "text": "かんじ", "style": {}}],
                        }
                    ],
                }
            ],
        }
        emphasis = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {"type": "text", "text": "着重", "style": {"emphasis": True}}
                    ],
                }
            ],
        }

        # The ruby of the first line takes the top of the box -> the body centre is below the exact centre of the box
        ruby_w, ruby_h, _, ruby_body = calc_box_from_font(
            32, ruby, True, line_spacing=1.0
        )
        self.assertAlmostEqual(ruby_body[0], ruby_w / 2.0, places=3)
        self.assertGreater(ruby_body[1], ruby_h / 2.0)

        # The emphasis marks of the last line take the bottom of the box -> the body centre is above the exact centre of the box
        emp_w, emp_h, _, emp_body = calc_box_from_font(
            32, emphasis, True, line_spacing=1.0
        )
        self.assertAlmostEqual(emp_body[0], emp_w / 2.0, places=3)
        self.assertLess(emp_body[1], emp_h / 2.0)

    def test_plain_inputs_report_box_center_as_body_center(self):
        # For a plain string and for a document without decorations, the body centre is the exact centre of the box.
        # Holds strictly for horizontal text (the stroke pad wraps the line ink symmetrically); for vertical text, since 2026-07-17 the global
        # stroke takes part in the measured geometry (the same source as the rendered surface), and the overflow of the ink+pad of the first and last characters
        # above and below their slots may be asymmetric, so the centre may deviate by <= the stroke pad (round(0.07*32)+1).
        stroke_pad = round(0.07 * 32) + 1
        for value, horizontal in (
            ("第一行[BR]第二行", True),
            ("第一行[BR]第二行", False),
            (legacy_line_breaks_to_document("第一行[BR]第二行").to_dict(), True),
            (legacy_line_breaks_to_document("第一行[BR]第二行").to_dict(), False),
        ):
            w, h, _, body = calc_box_from_font(32, value, horizontal, line_spacing=1.0)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            if horizontal:
                self.assertAlmostEqual(body[0], w / 2.0, places=3)
                self.assertAlmostEqual(body[1], h / 2.0, places=3)
            else:
                self.assertAlmostEqual(body[0], w / 2.0, delta=stroke_pad)
                self.assertAlmostEqual(body[1], h / 2.0, delta=stroke_pad)

    def test_vertical_rich_text_applies_local_layer_effects(self):
        plain = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [{"type": "text", "text": "文字", "style": {}}],
                }
            ],
        }
        styled = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {
                            "type": "text",
                            "text": "文",
                            "style": {
                                "bold": True,
                                "italic": True,
                                "transform": {
                                    "rotation": 18,
                                    "mirrorX": True,
                                    "offsetX": 12,
                                },
                            },
                        },
                        {"type": "text", "text": "字", "style": {}},
                    ],
                }
            ],
        }

        plain_render = text_render.put_text_vertical(
            36,
            plain,
            180,
            "left",
            (255, 255, 255),
            (0, 0, 0),
            line_spacing=1.0,
        )
        styled_render = text_render.put_text_vertical(
            36,
            styled,
            180,
            "left",
            (255, 255, 255),
            (0, 0, 0),
            line_spacing=1.0,
        )

        self.assertIsNotNone(plain_render)
        self.assertIsNotNone(styled_render)
        self.assertNotEqual(plain_render.shape, styled_render.shape)
        self.assertGreater(
            int(styled_render[:, :, 3].sum()), int(plain_render[:, :, 3].sum())
        )

    def test_bold_is_applied_before_stroke_colorization(self):
        normal = text_render._line_surface("太", 48, 3, 0.07, False, 1.0, False)
        bold = text_render._line_surface("太", 48, 3, 0.07, False, 1.0, True)

        self.assertIsNotNone(normal)
        self.assertIsNotNone(bold)
        self.assertGreater(int(bold["text"].sum()), int(normal["text"].sum()))
        self.assertTrue(np.all(bold["border"] >= bold["text"]))

        layer = text_render.add_color(
            bold["text"], (0, 0, 0), bold["border"], (255, 255, 255)
        )
        body_pixels = layer[bold["text"] >= 240]
        self.assertGreater(len(body_pixels), 0)
        self.assertLess(int(body_pixels[:, :3].max()), 16)

    def test_tcy_node_uses_horizontal_block_in_vertical_rendering(self):
        tcy_document = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {"type": "text", "text": "年", "style": {}},
                        {
                            "type": "tcy",
                            "content": [
                                {"type": "text", "text": "2026", "style": {}},
                            ],
                        },
                        {"type": "text", "text": "版", "style": {}},
                    ],
                }
            ],
        }
        plain_document = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {"type": "text", "text": "年2026版", "style": {}},
                    ],
                }
            ],
        }

        tcy_height = text_render.get_string_height(32, tcy_document)
        plain_height = text_render.get_string_height(32, plain_document)
        rendered = text_render.put_text_vertical(
            32,
            tcy_document,
            260,
            "left",
            (255, 255, 255),
            (0, 0, 0),
            line_spacing=1.0,
        )

        self.assertLess(tcy_height, plain_height)
        self.assertIsNotNone(rendered)
        self.assertGreater(int(rendered[:, :, 3].max()), 0)

    def test_vertical_middle_column_style_overflow_does_not_widen_frame(self):
        # Regression of 2026-07-17: the italic shear and stroke growth of characters in a middle column fall in the column gap
        # and must not enlarge the whole box on the left and right (the characters of the side columns did not change, so the box should not).
        def _doc(middle_inlines):
            return {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [{"type": "text", "text": "あいう", "style": {}}],
                    },
                    {"type": "paragraph", "inlines": middle_inlines},
                    {
                        "type": "paragraph",
                        "inlines": [{"type": "text", "text": "さしす", "style": {}}],
                    },
                ],
            }

        plain = _doc([{"type": "text", "text": "かきく", "style": {}}])
        italic = _doc(
            [
                {"type": "text", "text": "か", "style": {}},
                {"type": "text", "text": "き", "style": {"italic": True}},
                {"type": "text", "text": "く", "style": {}},
            ]
        )
        stroke = _doc(
            [
                {"type": "text", "text": "か", "style": {}},
                {
                    "type": "text",
                    "text": "き",
                    "style": {"stroke": {"color": "#ff0000", "width": 0.6}},
                },
                {"type": "text", "text": "く", "style": {}},
            ]
        )

        plain_m = text_render.measure_rich_text_metrics(
            48, plain, False, 1.0, stroke_width=0.0
        )
        italic_m = text_render.measure_rich_text_metrics(
            48, italic, False, 1.0, stroke_width=0.0
        )
        stroke_m = text_render.measure_rich_text_metrics(
            48, stroke, False, 1.0, stroke_width=0.0
        )
        self.assertEqual(italic_m["width"], plain_m["width"])
        self.assertEqual(stroke_m["width"], plain_m["width"])

        # The contract measured box == output surface still holds when the overflow overlaps the area of a neighbouring column
        for document, metrics in ((italic, italic_m), (stroke, stroke_m)):
            surface = text_render.put_text_vertical(
                48, document, 400, "left", (0, 0, 0), None, 1.0, stroke_width=0.0
            )
            self.assertEqual(
                (surface.shape[1], surface.shape[0]),
                (metrics["width"], metrics["height"]),
            )

    def test_tcy_block_compresses_to_base_font_cap(self):
        # Regression of 2026-07-17: when the tate-chu-yoko ink is wider than 1.1 times the base font size, the whole group is compressed horizontally
        # (in line with the reference implementation mtu-json-gui), so the box width no longer grows without limit with the number of digits.
        def _doc(digits, digit_style=None):
            return {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [
                            {"type": "text", "text": "第", "style": {}},
                            {
                                "type": "tcy",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": digits,
                                        "style": digit_style or {},
                                    }
                                ],
                            },
                            {"type": "text", "text": "话", "style": {}},
                        ],
                    }
                ],
            }

        short_m = text_render.measure_rich_text_metrics(
            48, _doc("12"), False, 1.0, stroke_width=0.0
        )
        four_m = text_render.measure_rich_text_metrics(
            48, _doc("1234"), False, 1.0, stroke_width=0.0
        )
        five_m = text_render.measure_rich_text_metrics(
            48, _doc("12345"), False, 1.0, stroke_width=0.0
        )
        cap = int(48 * 1.1) + 3  # Compression limit + the margin from ceil and centred rounding
        self.assertLessEqual(four_m["width"], cap)
        self.assertEqual(four_m["width"], five_m["width"])  # Beyond the limit everything is compressed to the same limit
        self.assertLessEqual(short_m["width"], four_m["width"])  # Below the limit nothing is compressed

        # The compression factor enters the one plan shared by measuring and rendering: with italics + global stroke, output surface == measured box
        styled = _doc("2024", {"italic": True})
        styled_m = text_render.measure_rich_text_metrics(
            48, styled, False, 1.0, stroke_width=0.07
        )
        surface = text_render.put_text_vertical(
            48, styled, 400, "left", (0, 0, 0), (255, 255, 255), 1.0, stroke_width=0.07
        )
        self.assertEqual(
            (surface.shape[1], surface.shape[0]),
            (styled_m["width"], styled_m["height"]),
        )

    def test_add_color_keeps_text_alpha_when_border_layer_is_blank(self):
        # F05 regression: when the stroke colour is not None but the stroke layer is all zero, the output alpha must still contain the body
        # (output alpha = max(stroke alpha, text alpha)); the whole run must not turn transparent.
        text_alpha = np.zeros((8, 8), dtype=np.uint8)
        text_alpha[2:6, 2:6] = 255
        blank_border = np.zeros_like(text_alpha)

        layer = text_render.add_color(text_alpha, (255, 0, 0), blank_border, (0, 0, 0))

        self.assertTrue(
            np.array_equal(layer[:, :, 3], np.maximum(blank_border, text_alpha))
        )
        self.assertEqual(int(layer[:, :, 3].max()), 255)

    def test_horizontal_ruby_visible_when_region_stroke_enabled(self):
        # F05 case 1 regression: with the region stroke on (bg is not None), the ruby layer of horizontal text must not be fully transparent.
        base_document = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [{"type": "text", "text": "漢字", "style": {}}],
                }
            ],
        }
        ruby_document = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {
                            "type": "ruby",
                            "base": [{"type": "text", "text": "漢字", "style": {}}],
                            "text": [{"type": "text", "text": "かんじ", "style": {}}],
                        }
                    ],
                }
            ],
        }

        base_render = text_render.put_text_horizontal(
            32,
            base_document,
            200,
            100,
            "left",
            False,
            (255, 255, 255),
            (0, 0, 0),
            line_spacing=1.0,
        )
        ruby_render = text_render.put_text_horizontal(
            32,
            ruby_document,
            200,
            100,
            "left",
            False,
            (255, 255, 255),
            (0, 0, 0),
            line_spacing=1.0,
        )

        self.assertIsNotNone(base_render)
        self.assertIsNotNone(ruby_render)
        # The ruby layer has ink: the total ink is more than in a render without ruby, and the crop box is taller because of the ruby line
        self.assertGreater(
            int(ruby_render[:, :, 3].sum()), int(base_render[:, :, 3].sum())
        )
        self.assertGreater(ruby_render.shape[0], base_render.shape[0])

    def test_vertical_ruby_is_drawn_once_for_the_complete_base_span(self):
        from unittest import mock

        from manga_translator.rendering.text_render import _render as render_module

        ruby_document = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {
                            "type": "ruby",
                            "base": [{"type": "text", "text": "漢字", "style": {}}],
                            "text": [{"type": "text", "text": "123", "style": {}}],
                        }
                    ],
                }
            ],
        }

        with mock.patch.object(render_module, "_paint_vertical_ruby") as draw_ruby:
            rendered = text_render.put_text_vertical(
                32,
                ruby_document,
                200,
                "center",
                (255, 255, 255),
                (0, 0, 0),
                line_spacing=1.0,
            )

        self.assertIsNotNone(rendered)
        self.assertEqual(draw_ruby.call_count, 1)
        args = draw_ruby.call_args.args
        self.assertEqual("".join(glyph.char for glyph in args[1].glyphs), "123")
        self.assertGreater(args[1].paint_end - args[1].paint_start, 32)

    def test_horizontal_ruby_characters_are_evenly_distributed_over_base(self):
        document = ensure_rich_text_document(
            {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [
                            {
                                "type": "ruby",
                                "base": [{"type": "text", "text": "漢字", "style": {}}],
                                "text": [{"type": "text", "text": "123", "style": {}}],
                            }
                        ],
                    }
                ],
            }
        )

        layout = text_render._build_rich_horizontal_layout(
            document,
            32,
            0.0,
            None,
            False,
            1.0,
        )[0]
        run = layout.runs[0]
        items = run.ruby.glyphs
        self.assertEqual(len(items), 3)

        actual_centers = [item.main_center for item in items]
        slot = run.logical_width / 3.0
        expected_centers = [slot * 0.5, slot * 1.5, slot * 2.5]
        for actual, expected in zip(actual_centers, expected_centers):
            self.assertAlmostEqual(actual, expected, places=4)

    def test_ruby_plan_uses_the_same_slot_contract_for_both_axes(self):
        from manga_translator.rendering.text_render._plans import (
            FlowAxis,
            plan_ruby_glyphs,
        )

        horizontal = plan_ruby_glyphs("123", 0, 90, FlowAxis.HORIZONTAL)
        self.assertEqual(
            [glyph.main_center for glyph in horizontal],
            [15.0, 45.0, 75.0],
        )

        vertical = plan_ruby_glyphs(
            "123456",
            10,
            70,
            FlowAxis.VERTICAL,
            nominal_glyph_extent=20,
        )
        self.assertTrue(all(glyph.main_scale < 1.0 for glyph in vertical))
        paint_start = min(
            glyph.main_center - 10 * glyph.main_scale for glyph in vertical
        )
        paint_end = max(glyph.main_center + 10 * glyph.main_scale for glyph in vertical)
        self.assertAlmostEqual(paint_end - paint_start, 72.0)

    def test_zero_width_span_stroke_keeps_span_visible(self):
        # F05 case 3 regression: stroke.width=0 at span level must not make the whole run vanish as transparent.
        document = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {
                            "type": "text",
                            "text": "零描边",
                            "style": {"stroke": {"color": "#000000", "width": 0}},
                        }
                    ],
                }
            ],
        }

        rendered = text_render.put_text_horizontal(
            32,
            document,
            200,
            100,
            "left",
            False,
            (255, 255, 255),
            (0, 0, 0),
            line_spacing=1.0,
        )

        self.assertIsNotNone(rendered)
        self.assertGreater(int(rendered[:, :, 3].max()), 0)

    def test_measure_horizontal_span_metrics_match_render_path(self):
        # F21: now that horizontal metrics use _line_metrics (no rasterisation), logical_width/ascent/
        # descent must be exactly equal to the values the render path (_rich_span_surface) produces.
        document = ensure_rich_text_document(
            {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [
                            {"type": "text", "text": "漢字Abc", "style": {}},
                            {
                                "type": "text",
                                "text": "大字",
                                "style": {"fontSize": 48, "bold": True},
                            },
                        ],
                    }
                ],
            }
        )

        from manga_translator.rendering.text_render import _layout as layout_module

        for span in document.blocks[0].spans:
            span_font = text_render._style_font_size(32, span.style)
            run = layout_module._build_horizontal_run_plan(
                span, 32, 0.07, (0, 0, 0), False, 1.0
            )
            metrics = text_render._line_metrics(span.text, span_font, 1.0)

            self.assertTrue(run.has_ink)
            self.assertEqual(float(metrics["logical_width"]), run.logical_width)
            self.assertEqual(float(metrics["ascent"]), run.ascent)
            self.assertEqual(float(metrics["descent"]), run.descent)

    def test_vertical_column_plan_is_deterministic_and_geometry_shared(self):
        document = ensure_rich_text_document(
            {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [
                            {"type": "text", "text": "粗", "style": {"bold": True}},
                            {
                                "type": "text",
                                "text": "斜",
                                "style": {
                                    "italic": True,
                                    "transform": {"rotation": 18},
                                },
                            },
                            {
                                "type": "text",
                                "text": "描",
                                "style": {
                                    "stroke": {"color": "#000000", "width": 0.12}
                                },
                            },
                            {
                                "type": "ruby",
                                "base": [
                                    {
                                        "type": "text",
                                        "text": "漢",
                                        "style": {"emphasis": True},
                                    }
                                ],
                                "text": [{"type": "text", "text": "かん", "style": {}}],
                            },
                            {
                                "type": "tcy",
                                "content": [
                                    {"type": "text", "text": "2026", "style": {}}
                                ],
                            },
                        ],
                    },
                    {
                        "type": "paragraph",
                        "inlines": [{"type": "text", "text": "第二列", "style": {}}],
                    },
                ],
            }
        )

        full = text_render._build_rich_vertical_layout(
            document, 32, 0.07, (255, 255, 255), (0, 0, 0), 1.0
        )
        measured = text_render._build_rich_vertical_layout(
            document, 32, 0.07, (255, 255, 255), (0, 0, 0), 1.0
        )

        self.assertEqual(len(full), len(measured))
        for full_layout, measured_layout in zip(full, measured):
            self.assertEqual(measured_layout.thickness, full_layout.thickness)
            self.assertEqual(measured_layout.height, full_layout.height)
            self.assertEqual(
                measured_layout.content_paint_bounds, full_layout.content_paint_bounds
            )
            self.assertEqual(
                measured_layout.ruby_cross_extent, full_layout.ruby_cross_extent
            )
            self.assertEqual(
                measured_layout.annotation_cross_extent,
                full_layout.annotation_cross_extent,
            )
            self.assertEqual(len(measured_layout.items), len(full_layout.items))
            for full_item, measured_item in zip(
                full_layout.items, measured_layout.items
            ):
                self.assertEqual(measured_item, full_item)

        full_geometry = text_render._rich_vertical_layout_geometry(full, 32, 1.0)
        measured_geometry = text_render._rich_vertical_layout_geometry(
            measured, 32, 1.0
        )
        self.assertEqual(measured_geometry, full_geometry)

    def test_paragraph_spans_are_lazily_cached(self):
        # F24: a parsed document is immutable, and spans are cached after the first computation (no full deepcopy on every access any more);
        # the cached span.style is decoupled from inline.style.
        document = ensure_rich_text_document(_sample_document())
        paragraph = document.blocks[0]

        first = paragraph.spans
        second = paragraph.spans

        self.assertIs(first, second)
        self.assertIsNot(first[0].style, paragraph.inlines[0].style)

    def test_high_level_render_uses_legacy_breaks_collected_after_layout(self):
        image = np.zeros((220, 220, 3), dtype=np.uint8)
        region = TextBlock(
            lines=[[[40, 40], [180, 40], [180, 180], [40, 180]]],
            texts=["原文"],
            translation="红[BR]蓝",
            fg_color=(255, 255, 255),
            bg_color=(0, 0, 0),
            direction="h",
            target_lang="CHS",
        )
        region.font_size = 32
        region.font_family = "Arial Unicode MS"
        dst_points = np.array(
            [[[40, 40], [180, 40], [180, 180], [40, 180]]],
            dtype=np.float32,
        )
        sync_translation_raw_from_layout([region], Config())

        rendered = render(
            image,
            region,
            dst_points,
            hyphenate=True,
            line_spacing=1.0,
            disable_font_border=False,
            config=Config(),
        )

        self.assertEqual(rendered.shape, image.shape)
        self.assertGreater(int(rendered.max()), 0)
        self.assertEqual(region.translation, "红[BR]蓝")
        self.assertIsInstance(region.translation_rich, dict)
        self.assertEqual(region.translation_rich["format"], RICH_TEXT_FORMAT)
        self.assertEqual(len(region.translation_rich["blocks"]), 2)

    def test_high_level_render_only_warns_when_fill_layer_is_empty(self):
        image = np.zeros((220, 220, 3), dtype=np.uint8)
        region = TextBlock(
            lines=[[[40, 40], [180, 40], [180, 180], [40, 180]]],
            texts=["原文"],
            translation="文字",
            fg_color=(0, 0, 0),
            bg_color=(255, 255, 255),
            direction="v",
            target_lang="CHS",
        )
        region.font_size = 32
        region.font_family = "Arial Unicode MS"
        dst_points = np.asarray(
            [[[40, 40], [180, 40], [180, 180], [40, 180]]],
            dtype=np.float32,
        )

        with patch.object(
            rendering_module.text_render,
            "put_text_vertical",
            return_value=None,
        ):
            with self.assertNoLogs("manga-translator.render", level="WARNING"):
                render(
                    image.copy(),
                    region,
                    dst_points,
                    hyphenate=True,
                    line_spacing=1.0,
                    disable_font_border=False,
                    config=Config(),
                    paint_part="effects",
                )
            with self.assertLogs("manga-translator.render", level="WARNING") as logs:
                render(
                    image.copy(),
                    region,
                    dst_points,
                    hyphenate=True,
                    line_spacing=1.0,
                    disable_font_border=False,
                    config=Config(),
                    paint_part="fill",
                )

        self.assertTrue(any("[RENDER SKIPPED]" in message for message in logs.output))

    def test_text_render_paint_parts_keep_stroke_outside_fill(self):
        stroke = text_render.put_text_horizontal(
            32,
            "文字",
            200,
            100,
            "left",
            False,
            (0, 0, 0),
            (255, 255, 255),
            stroke_width=0.2,
            paint_part="stroke",
        )
        fill = text_render.put_text_horizontal(
            32,
            "文字",
            200,
            100,
            "left",
            False,
            (0, 0, 0),
            (255, 255, 255),
            stroke_width=0.2,
            paint_part="fill",
        )

        self.assertEqual(stroke.shape, fill.shape)
        self.assertGreater(int(stroke[:, :, 3].sum()), int(fill[:, :, 3].sum()))
        self.assertTrue(np.any((stroke[:, :, 3] > 0) & (fill[:, :, 3] == 0)))

    def test_dispatch_paints_complete_regions_in_input_order(self):
        points = np.asarray(
            [[[40, 40], [180, 40], [180, 180], [40, 180]]], dtype=np.float32
        )
        regions = [
            TextBlock(
                lines=[points[0].tolist()],
                texts=["原文"],
                translation=text,
                fg_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                direction="h",
                target_lang="CHS",
            )
            for text in ("先行文字", "后行文字")
        ]
        calls = []

        def fake_render(image, region, *args, **kwargs):
            calls.append((region.translation, kwargs.get("paint_part")))
            return image

        with (
            patch.object(
                rendering_module,
                "download_chinese_linebreak_models_if_enabled",
                new=AsyncMock(),
            ),
            patch.object(
                rendering_module,
                "resize_regions_to_font_size",
                return_value=[points, points],
            ),
            patch.object(rendering_module, "render", side_effect=fake_render),
        ):
            for ordered_regions in (regions, regions[::-1]):
                with self.subTest(order=[region.translation for region in ordered_regions]):
                    calls.clear()
                    result = asyncio.run(
                        rendering_module.dispatch(
                            np.zeros((220, 220, 3), dtype=np.uint8),
                            ordered_regions,
                            Config(),
                            skip_font_scaling=True,
                        )
                    )

                    self.assertIsNotNone(result)
                    # None renders effects, stroke and fill together for each region.
                    # Reversing the input must also reverse the complete-region order.
                    self.assertEqual(
                        calls,
                        [(region.translation, None) for region in ordered_regions],
                    )

    def test_multiline_plain_and_ruby_documents_render_without_supersampling(self):
        # Multi-line plain text produced by BR and a document with ruby should both be rendered directly by the ordinary Qt path.
        def _make_region(translation):
            region = TextBlock(
                lines=[[[40, 40], [180, 40], [180, 180], [40, 180]]],
                texts=["原文"],
                translation=translation,
                fg_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                direction="h",
                target_lang="CHS",
            )
            region.font_size = 32
            region.font_family = "Arial Unicode MS"
            return region

        dst_points = np.array(
            [[[40, 40], [180, 40], [180, 180], [40, 180]]], dtype=np.float32
        )
        image = np.zeros((220, 220, 3), dtype=np.uint8)

        plain_region = _make_region("红[BR]蓝")
        sync_translation_raw_from_layout([plain_region], Config())
        self.assertEqual(plain_region.translation_rich["format"], RICH_TEXT_FORMAT)

        plain_rendered = render(
            image.copy(),
            plain_region,
            dst_points,
            hyphenate=True,
            line_spacing=1.0,
            disable_font_border=False,
            config=Config(),
        )
        self.assertGreater(int(plain_rendered.max()), 0)

        ruby_region = _make_region("")
        ruby_region.set_translation_rich(
            ensure_rich_text_document(
                {
                    "format": RICH_TEXT_FORMAT,
                    "blocks": [
                        {
                            "type": "paragraph",
                            "inlines": [
                                {
                                    "type": "ruby",
                                    "base": [
                                        {"type": "text", "text": "漢字", "style": {}}
                                    ],
                                    "text": [
                                        {"type": "text", "text": "かんじ", "style": {}}
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ),
            sync_plain=True,
        )

        ruby_rendered = render(
            image.copy(),
            ruby_region,
            dst_points,
            hyphenate=True,
            line_spacing=1.0,
            disable_font_border=False,
            config=Config(),
        )
        self.assertGreater(int(ruby_rendered.max()), 0)

    def test_rich_document_font_size_shrinks_to_fit_region_box(self):
        # F07 regression: outside skip_font_scaling, a rich-text region no longer uses the estimated font size as it is,
        # but shrinks to the largest font size the unrotated bounding box of the region can hold (without automatic wrapping).
        from manga_translator.rendering import resize_regions_to_font_size

        region = TextBlock(
            lines=[[[0, 0], [120, 0], [120, 60], [0, 60]]],
            texts=["原文"],
            direction="h",
            target_lang="CHS",
        )
        region.set_translation_rich(
            ensure_rich_text_document(
                legacy_line_breaks_to_document(
                    "很长的一段译文需要收缩字号[BR]第二行同样很长"
                ).to_dict()
            ),
            sync_plain=True,
        )
        region.font_size = 80  # The estimated font size is far larger than the box
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        config = Config()

        dst_points_list = resize_regions_to_font_size(
            image, [region], config, original_img=None
        )

        self.assertEqual(len(dst_points_list), 1)
        self.assertIsNotNone(dst_points_list[0])
        self.assertLess(region.font_size, 80)
        # Measured the same way, the shrunk font size should fit in the original box
        req_w, req_h, _, _ = calc_box_from_font(
            region.font_size,
            region.get_translation_for_rendering(),
            True,
            line_spacing=1.0,
            config=config,
        )
        self.assertLessEqual(req_w, 120)
        self.assertLessEqual(req_h, 60)

    def test_h_tag_is_not_a_supported_rendering_protocol(self):
        raw_text = "<H>ABC</H>"

        lines, heights, _ = text_render.calc_vertical_metrics(
            32,
            raw_text,
            max_height=99999,
        )
        expected_height = sum(
            text_render.get_char_offset_y(32, char) for char in raw_text
        )

        self.assertEqual(lines, [raw_text])
        self.assertEqual(heights, [expected_height])

    def test_replacement_sync_collects_legacy_breaks_after_layout(self):
        region = TextBlock(
            lines=[[[0, 0], [100, 0], [100, 80], [0, 80]]],
            texts=["原文"],
            translation="红[BR]蓝",
        )
        region._replacement_layout_record = ReplacementLayoutRecord(
            raw_text="红[BR]蓝",
            replaced_text="红[BR]蓝",
        )

        sync_translation_raw_from_layout([region], Config())

        self.assertEqual(region.translation, "红[BR]蓝")
        self.assertEqual(region.translation_raw, "红[BR]蓝")
        self.assertEqual(region.translation_rich["format"], RICH_TEXT_FORMAT)
        self.assertEqual(len(region.translation_rich["blocks"]), 2)
        first_style = region.translation_rich["blocks"][0]["inlines"][0]["style"]
        self.assertEqual(first_style, {})
        self.assertFalse(hasattr(region, "_replacement_layout_record"))

    def test_replacement_sync_projects_space_replaced_by_break_to_raw(self):
        region = TextBlock(
            lines=[[[0, 0], [100, 0], [100, 80], [0, 80]]],
            texts=["source"],
            translation="hello world",
        )
        region._replacement_layout_record = ReplacementLayoutRecord(
            raw_text="hello world",
            replaced_text="hello world",
        )
        region.translation = "hello[BR]world"

        sync_translation_raw_from_layout([region], Config())

        self.assertEqual(region.translation, "hello[BR]world")
        self.assertEqual(region.translation_raw, "hello[BR]world")

    # ------------------------------------------------------------------
    # Italics as an angle + offset envelope (in line with the reference implementation mtu-json-gui)
    # ------------------------------------------------------------------

    @staticmethod
    def _single_span_document(text, style):
        return {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [{"type": "text", "text": text, "style": style}],
                }
            ],
        }

    @staticmethod
    def _ink_rect(surface):
        import cv2

        nz = cv2.findNonZero(surface[:, :, 3])
        return cv2.boundingRect(nz)

    def test_italic_angle_parses_and_legacy_bool_maps_to_default(self):
        # A number = the shear angle; true = the default angle (DEFAULT_ITALIC_ANGLE, measured as 10 degrees in Photoshop); 0 is normalised to False
        document = ensure_rich_text_document(
            self._single_span_document("字", {"italic": 24})
        )
        self.assertEqual(document.blocks[0].spans[0].style.italic, 24.0)
        self.assertEqual(
            document.to_dict()["blocks"][0]["inlines"][0]["style"]["italic"], 24.0
        )

        legacy = ensure_rich_text_document(
            self._single_span_document("字", {"italic": True})
        )
        self.assertIs(legacy.blocks[0].spans[0].style.italic, True)

        zero = ensure_rich_text_document(
            self._single_span_document("字", {"italic": 0})
        )
        self.assertIs(zero.blocks[0].spans[0].style.italic, False)

        with self.assertRaises(ValueError):
            ensure_rich_text_document(
                self._single_span_document("字", {"italic": "斜"})
            )

        plain = text_render.measure_rich_text_metrics(
            32, self._single_span_document("測試文字", {}), True, 1.0, stroke_width=0.0
        )
        angled = text_render.measure_rich_text_metrics(
            32,
            self._single_span_document(
                "測試文字", {"italic": text_render.DEFAULT_ITALIC_ANGLE}
            ),
            True,
            1.0,
            stroke_width=0.0,
        )
        legacy_m = text_render.measure_rich_text_metrics(
            32,
            self._single_span_document("測試文字", {"italic": True}),
            True,
            1.0,
            stroke_width=0.0,
        )
        self.assertGreater(angled["width"], plain["width"])
        self.assertEqual(
            (angled["width"], angled["height"]), (legacy_m["width"], legacy_m["height"])
        )

    def test_transform_stretch_round_trips_and_rejects_invalid_factors(self):
        document = ensure_rich_text_document(
            self._single_span_document(
                "拉伸", {"transform": {"scaleX": 1.8, "scaleY": 0.7}}
            )
        )
        transform = document.blocks[0].spans[0].style.transform

        self.assertEqual((transform.scale_x, transform.scale_y), (1.8, 0.7))
        self.assertEqual(
            document.to_dict()["blocks"][0]["inlines"][0]["style"]["transform"],
            {"scaleX": 1.8, "scaleY": 0.7},
        )
        for invalid in (0, -1, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ensure_rich_text_document(
                    self._single_span_document(
                        "拉伸", {"transform": {"scaleX": invalid}}
                    )
                )

    def test_stretch_changes_each_axis_without_bitmap_resampling(self):
        plain_doc = self._single_span_document("拉伸", {})
        wide_doc = self._single_span_document("拉伸", {"transform": {"scaleX": 1.8}})
        tall_doc = self._single_span_document("拉伸", {"transform": {"scaleY": 1.8}})
        plain = text_render.measure_rich_text_metrics(
            40, plain_doc, True, 1.0, stroke_width=0.0
        )
        wide = text_render.measure_rich_text_metrics(
            40, wide_doc, True, 1.0, stroke_width=0.0
        )
        tall = text_render.measure_rich_text_metrics(
            40, tall_doc, True, 1.0, stroke_width=0.0
        )

        self.assertGreater(wide["width"], plain["width"])
        self.assertGreater(tall["height"], plain["height"])
        with patch("cv2.resize", side_effect=AssertionError("bitmap resampling")):
            wide_surface = text_render.put_text_horizontal(
                40,
                ensure_rich_text_document(wide_doc),
                10,
                10,
                "center",
                False,
                (0, 0, 0),
                None,
                line_spacing=1.0,
                stroke_width=0.0,
            )
            vertical_surface = text_render.put_text_vertical(
                40,
                ensure_rich_text_document(tall_doc),
                10,
                "center",
                (0, 0, 0),
                None,
                1.0,
                stroke_width=0.0,
            )
        self.assertEqual(
            (wide_surface.shape[1], wide_surface.shape[0]),
            (wide["width"], wide["height"]),
        )
        self.assertGreater(vertical_surface.shape[0], 0)

    def test_horizontal_offset_expands_envelope_and_moves_ink(self):
        # A uniform offset is no longer cancelled by tight cropping of the ink: the envelope grows in the direction of the offset, the ink really moves,
        # and the size of the output surface matches the measured box pixel for pixel (strictly equal without a stroke).
        plain_doc = self._single_span_document("測試文字", {})
        offset_doc = self._single_span_document(
            "測試文字", {"transform": {"offsetX": 50}}
        )

        plain_m = text_render.measure_rich_text_metrics(
            32, plain_doc, True, 1.0, stroke_width=0.0
        )
        offset_m = text_render.measure_rich_text_metrics(
            32, offset_doc, True, 1.0, stroke_width=0.0
        )
        self.assertGreater(offset_m["width"], plain_m["width"])

        plain_surface = text_render.put_text_horizontal(
            32,
            ensure_rich_text_document(plain_doc),
            10,
            10,
            "center",
            False,
            (0, 0, 0),
            None,
            line_spacing=1.0,
            stroke_width=0.0,
        )
        offset_surface = text_render.put_text_horizontal(
            32,
            ensure_rich_text_document(offset_doc),
            10,
            10,
            "center",
            False,
            (0, 0, 0),
            None,
            line_spacing=1.0,
            stroke_width=0.0,
        )
        self.assertEqual(
            (plain_surface.shape[1], plain_surface.shape[0]),
            (plain_m["width"], plain_m["height"]),
        )
        self.assertEqual(
            (offset_surface.shape[1], offset_surface.shape[0]),
            (offset_m["width"], offset_m["height"]),
        )
        shift = self._ink_rect(offset_surface)[0] - self._ink_rect(plain_surface)[0]
        self.assertGreaterEqual(shift, 12)
        self.assertLessEqual(shift, 20)

    def test_vertical_offset_y_expands_envelope_and_moves_ink(self):
        plain_doc = self._single_span_document("縦書", {})
        offset_doc = self._single_span_document("縦書", {"transform": {"offsetY": 75}})

        plain_m = text_render.measure_rich_text_metrics(
            32, plain_doc, False, 1.0, stroke_width=0.0
        )
        offset_m = text_render.measure_rich_text_metrics(
            32, offset_doc, False, 1.0, stroke_width=0.0
        )
        self.assertGreaterEqual(offset_m["height"], plain_m["height"] + 20)

        plain_surface = text_render.put_text_vertical(
            32,
            ensure_rich_text_document(plain_doc),
            10,
            "center",
            (0, 0, 0),
            None,
            1.0,
            stroke_width=0.0,
        )
        offset_surface = text_render.put_text_vertical(
            32,
            ensure_rich_text_document(offset_doc),
            10,
            "center",
            (0, 0, 0),
            None,
            1.0,
            stroke_width=0.0,
        )
        self.assertEqual(
            (plain_surface.shape[1], plain_surface.shape[0]),
            (plain_m["width"], plain_m["height"]),
        )
        self.assertEqual(
            (offset_surface.shape[1], offset_surface.shape[0]),
            (offset_m["width"], offset_m["height"]),
        )
        shift = self._ink_rect(offset_surface)[1] - self._ink_rect(plain_surface)[1]
        self.assertGreaterEqual(shift, 20)
        self.assertLessEqual(shift, 28)

    def test_offset_reports_body_center_for_anchoring(self):
        # The body centre of horizontal text is computed from the actual main ink; a transform offset is part of the body geometry
        # and the centre of the old logic is no longer kept on top of it as decoration outside the box.
        doc = {
            "format": RICH_TEXT_FORMAT,
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {"type": "text", "text": "正文", "style": {}},
                        {
                            "type": "text",
                            "text": "上浮",
                            "style": {"transform": {"offsetY": -100}},
                        },
                    ],
                }
            ],
        }
        metrics = text_render.measure_rich_text_metrics(
            32, doc, True, 1.0, stroke_width=0.0
        )
        self.assertAlmostEqual(
            metrics["body_center"][1], metrics["height"] / 2.0, places=3
        )

    def test_horizontal_ink_layout_uses_content_height_not_font_line_box(self):
        metrics = text_render.measure_rich_text_metrics(
            48, "高度受限字号对比", True, 1.0, stroke_width=0.0
        )
        qt_line = text_render._line_metrics("高度受限字号对比", 48, 1.0)
        self.assertLess(
            metrics["height"], round(qt_line["ascent"] + qt_line["descent"])
        )

    def test_horizontal_ink_layout_expands_only_for_actual_vertical_collision(self):
        cjk = ensure_rich_text_document(
            legacy_line_breaks_to_document("第一行\n第二行").to_dict()
        )
        mixed = ensure_rich_text_document(
            legacy_line_breaks_to_document("gypqj\nÁÉÎÔŨ").to_dict()
        )
        cjk_layout = text_render._build_rich_horizontal_layout(
            cjk, 64, 0.0, None, False, 1.0
        )
        mixed_layout = text_render._build_rich_horizontal_layout(
            mixed, 64, 0.0, None, False, 1.0
        )
        cjk_geometry = text_render._rich_horizontal_layout_geometry(cjk_layout, 64, 1.0)
        mixed_geometry = text_render._rich_horizontal_layout_geometry(
            mixed_layout, 64, 1.0
        )
        cjk_advance = cjk_geometry["baselines"][1] - cjk_geometry["baselines"][0]
        mixed_advance = mixed_geometry["baselines"][1] - mixed_geometry["baselines"][0]
        self.assertGreater(mixed_advance, cjk_advance)

    def test_horizontal_global_stroke_is_measured_inside_ink_plan(self):
        metrics = text_render.measure_rich_text_metrics(
            48, "描边文字", True, 1.0, stroke_width=0.07
        )
        surface = text_render.put_text_horizontal(
            48,
            "描边文字",
            10,
            10,
            "center",
            False,
            (0, 0, 0),
            (255, 255, 255),
            line_spacing=1.0,
            stroke_width=0.07,
        )
        self.assertEqual(
            (surface.shape[1], surface.shape[0]), (metrics["width"], metrics["height"])
        )

        config = Config()
        config.render.disable_font_border = True
        no_stroke_box = calc_box_from_font(48, "描边文字", True, 1.0, config=config)
        no_stroke_surface = text_render.put_text_horizontal(
            48,
            "描边文字",
            10,
            10,
            "center",
            False,
            (0, 0, 0),
            None,
            line_spacing=1.0,
            config=config,
        )
        self.assertEqual(
            (no_stroke_surface.shape[1], no_stroke_surface.shape[0]),
            no_stroke_box[:2],
        )

    def test_italic_surface_matches_measured_envelope(self):
        doc = self._single_span_document("斜体測試", {"italic": 15})
        metrics = text_render.measure_rich_text_metrics(
            32, doc, True, 1.0, stroke_width=0.0
        )
        surface = text_render.put_text_horizontal(
            32,
            ensure_rich_text_document(doc),
            10,
            10,
            "center",
            False,
            (0, 0, 0),
            None,
            line_spacing=1.0,
            stroke_width=0.0,
        )
        self.assertEqual(
            (surface.shape[1], surface.shape[0]), (metrics["width"], metrics["height"])
        )

    def test_glow_and_outer_stroke_expand_and_match_measured_envelope(self):
        plain = self._single_span_document("效果", {})
        styled = self._single_span_document(
            "效果",
            {
                "glow": {"color": "#00ffff", "blur": 0.125},
                "outerStroke": {"color": "#ff0000", "width": 0.1875},
            },
        )
        plain_metrics = text_render.measure_rich_text_metrics(
            32, plain, True, 1.0, stroke_width=0.0
        )
        styled_metrics = text_render.measure_rich_text_metrics(
            32, styled, True, 1.0, stroke_width=0.0
        )
        surface = text_render.put_text_horizontal(
            32,
            ensure_rich_text_document(styled),
            10,
            10,
            "center",
            False,
            (0, 0, 0),
            None,
            line_spacing=1.0,
            stroke_width=0.0,
        )
        self.assertGreater(styled_metrics["width"], plain_metrics["width"])
        self.assertGreater(styled_metrics["height"], plain_metrics["height"])
        self.assertEqual(
            (surface.shape[1], surface.shape[0]),
            (styled_metrics["width"], styled_metrics["height"]),
        )

    def test_glow_does_not_change_horizontal_line_spacing(self):
        def document(style):
            return ensure_rich_text_document(
                {
                    "format": RICH_TEXT_FORMAT,
                    "blocks": [
                        {
                            "type": "paragraph",
                            "inlines": [
                                {"type": "text", "text": "第一行", "style": style},
                            ],
                        },
                        {
                            "type": "paragraph",
                            "inlines": [
                                {"type": "text", "text": "第二行", "style": style},
                            ],
                        },
                    ],
                }
            )

        plain_layout = text_render._build_rich_horizontal_layout(
            document({}), 164, 0.0, None, False, 1.0
        )
        glow_layout = text_render._build_rich_horizontal_layout(
            document({"glow": {"color": "#00ffff", "blur": 0.10}}),
            164,
            0.0,
            None,
            False,
            1.0,
        )
        plain_geometry = text_render._rich_horizontal_layout_geometry(
            plain_layout, 164, 0.1
        )
        glow_geometry = text_render._rich_horizontal_layout_geometry(
            glow_layout, 164, 0.1
        )

        plain_advance = plain_geometry["baselines"][1] - plain_geometry["baselines"][0]
        glow_advance = glow_geometry["baselines"][1] - glow_geometry["baselines"][0]
        self.assertAlmostEqual(glow_advance, plain_advance, places=3)
        self.assertGreater(
            glow_geometry["paint_height"], plain_geometry["paint_height"]
        )

    def test_local_stroke_width_scales_with_font_size(self):
        from manga_translator.rendering.text_render._compose import _style_stroke_ratio

        style = (
            ensure_rich_text_document(
                self._single_span_document(
                    "字", {"stroke": {"color": "#fff", "width": 0.2}}
                )
            )
            .paragraphs[0]
            .spans[0]
            .style
        )
        self.assertAlmostEqual(_style_stroke_ratio(style, 24, 0.0, None) * 24, 4.8)
        self.assertAlmostEqual(_style_stroke_ratio(style, 72, 0.0, None) * 72, 14.4)

    def test_local_line_spacing_prefers_current_lk_over_previous_nk(self):
        document = ensure_rich_text_document(
            {
                "format": RICH_TEXT_FORMAT,
                "blocks": [
                    {
                        "type": "paragraph",
                        "inlines": [
                            {
                                "type": "text",
                                "text": "上",
                                "style": {"nextKerning": 1.25},
                            },
                        ],
                    },
                    {
                        "type": "paragraph",
                        "inlines": [
                            {
                                "type": "text",
                                "text": "下",
                                "style": {"lineKerning": 0.375},
                            },
                        ],
                    },
                ],
            }
        )
        layouts = text_render._build_rich_horizontal_layout(
            document, 32, 0.0, None, False, 1.0
        )
        default_geometry = text_render._rich_horizontal_layout_geometry(
            text_render._build_rich_horizontal_layout(
                ensure_rich_text_document(
                    legacy_line_breaks_to_document("上\n下").to_dict()
                ),
                32,
                0.0,
                None,
                False,
                1.0,
            ),
            32,
            1.0,
        )
        geometry = text_render._rich_horizontal_layout_geometry(layouts, 32, 1.0)
        default_advance = (
            default_geometry["baselines"][1] - default_geometry["baselines"][0]
        )
        styled_advance = geometry["baselines"][1] - geometry["baselines"][0]
        self.assertAlmostEqual(styled_advance - default_advance, 12.0, places=3)


if __name__ == "__main__":
    unittest.main()
