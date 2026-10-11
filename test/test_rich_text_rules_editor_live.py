"""Regression of the live rich-text rules in the editor: comparison of old and new matches, skipping manual traces, the third level of the sync pipeline, IME narrowing.

Core semantics (decided by the user):
- only new hits that "did not exist before the edit" are applied (typing the first character of a two-character word and then adding the second → the whole word gets the style);
- old hits on unchanged text are never applied again (a style that was cleared is not put back);
- a matched range that carries rich text this rule cannot produce (a manual trace) → the whole range is skipped;
  when it only carries leftover styles of this rule itself → it may be completed as a whole;
- a whole-text replacement (no operation record) follows the full semantics of the render pipeline.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "desktop_qt_ui"))

import re  # noqa: E402

from manga_translator.rendering.rich_text_rules import (  # noqa: E402
    _parse_rules,
    apply_rich_text_rules,
)
from manga_translator.rendering.rich_text_sync import (  # noqa: E402
    sync_region_rich_translation,
)


def _rules(*common, horizontal=(), vertical=()):
    return _parse_rules({
        "common": list(common),
        "horizontal": list(horizontal),
        "vertical": list(vertical),
    })


RED_RULE = {"pattern": "你好", "style": {"color": "#ff0000"}}


def _doc(*inlines):
    return {
        "format": "richtext.v1",
        "blocks": [{"type": "paragraph", "inlines": list(inlines)}],
    }


def _text_run(text, style=None):
    return {"type": "text", "text": text, "style": style or {}}


def _runs_of(document):
    dumped = document.to_dict() if hasattr(document, "to_dict") else document
    return [
        (inline["text"], inline["style"])
        for inline in dumped["blocks"][0]["inlines"]
        if inline.get("type", "text") == "text"
    ]


# ─── Engine: comparing matches on previous_text and on the new text ───


def test_typed_char_completes_match_applies_whole_range():
    # Typing the first character alone gives no match; once the second is added, the whole two-character word is a new match and both characters are coloured
    rules = _rules(RED_RULE)
    document = apply_rich_text_rules(
        "你好", 0, rules, previous_text="你", styled_match_policy="skip"
    )
    assert document is not None
    assert _runs_of(document) == [("你好", {"color": "#ff0000"})]


def test_old_match_is_never_reapplied():
    # The word already matched before the edit (the user had cleared its style); typing elsewhere does not bring the style back
    rules = _rules(RED_RULE)
    plain = _doc(_text_run("你好xy", {}))
    document = apply_rich_text_rules(
        plain, 0, rules, previous_text="你好x", styled_match_policy="skip"
    )
    assert document is None  # No new match -> no change


def test_deletion_merge_creates_new_match():
    # Deleting the x in the middle joins the pieces into the pattern: the old text had no such match -> a new match
    rules = _rules({"pattern": "第1话", "style": {"bold": True}})
    document = apply_rich_text_rules(
        "第1话", 0, rules, previous_text="第1x话", styled_match_policy="skip"
    )
    assert document is not None
    assert _runs_of(document) == [("第1话", {"bold": True})]


def test_lookahead_match_outside_window_counts_as_new():
    # Lookaround: the matched range [0,1) lies entirely inside the common prefix, but the old text does not match at that position -> a new match
    rules = _rules({"pattern": "你(?=好)", "regex": True, "style": {"bold": True}})
    document = apply_rich_text_rules(
        "你好", 0, rules, previous_text="你", styled_match_policy="skip"
    )
    assert document is not None
    assert _runs_of(document) == [("你", {"bold": True}), ("好", {})]


def test_residue_style_backfills_whole_match():
    # The first character carries leftover style of this very rule (the second was deleted and typed back) -> the whole match is completed and the second is coloured too
    rules = _rules(RED_RULE)
    residue = _doc(_text_run("你", {"color": "#ff0000"}), _text_run("好", {}))
    document = apply_rich_text_rules(
        residue, 0, rules, previous_text="你", styled_match_policy="skip"
    )
    assert document is not None
    assert _runs_of(document) == [("你好", {"color": "#ff0000"})]


def test_manual_trace_skips_whole_match():
    # The first character carries a colour the rule cannot produce (a manual trace) -> the whole span is skipped and not a single field is added
    rules = _rules(RED_RULE)
    manual = _doc(_text_run("你", {"color": "#0000ff"}), _text_run("好", {}))
    document = apply_rich_text_rules(
        manual, 0, rules, previous_text="你", styled_match_policy="skip"
    )
    assert document is None


def test_manual_node_counts_as_trace():
    rules = _rules(RED_RULE)
    ruby = _doc(
        {
            "type": "ruby",
            "base": [_text_run("你", {})],
            "text": [_text_run("ニー", {})],
        },
        _text_run("好", {}),
    )
    document = apply_rich_text_rules(
        ruby, 0, rules, previous_text="你", styled_match_policy="skip"
    )
    assert document is None


def test_full_semantics_when_previous_text_is_none():
    # Whole-text replacement: previous_text=None -> every match counts as new (the same as the pipeline)
    rules = _rules(RED_RULE)
    document = apply_rich_text_rules(
        "你好在这你好", 0, rules, styled_match_policy="skip"
    )
    assert document is not None
    runs = _runs_of(document)
    assert runs[0] == ("你好", {"color": "#ff0000"})
    assert runs[-1] == ("你好", {"color": "#ff0000"})


# ─── Sync pipeline: sync_region_rich_translation followed by the rule stage ───


def _edit_info(ops, pre, post):
    return {"ops": ops, "pre_text": pre, "post_text": post}


def test_sync_plain_typing_grows_rule_style():
    # In a plain-text region (no earlier rich text), typing the one character that completes the word -> rich text appears
    result = sync_region_rich_translation(
        None,
        _edit_info([[1, 0, "好"]], "你", "你好"),
        raw_mode=False,
        new_translation="你好",
        apply_rules=True,
        old_translation="你",
        rules=_rules(RED_RULE),
    )
    assert result is not None
    assert result["blocks"][0]["inlines"][0] == {
        "type": "text",
        "text": "你好",
        "style": {"color": "#ff0000"},
    }


def test_sync_cleared_match_stays_clear_on_unrelated_edit():
    # After all styles are cleared (the rich-text field is deleted), typing elsewhere: the old match does not come back -> still plain text
    result = sync_region_rich_translation(
        None,
        _edit_info([[3, 0, "y"]], "你好x", "你好xy"),
        raw_mode=False,
        new_translation="你好xy",
        apply_rules=True,
        old_translation="你好x",
        rules=_rules(RED_RULE),
    )
    assert result is None


def test_sync_full_replacement_applies_all_matches():
    # Whole-text replacement (no record of edit operations) -> full semantics
    result = sync_region_rich_translation(
        None,
        None,
        raw_mode=False,
        new_translation="你好",
        apply_rules=True,
        rules=_rules(RED_RULE),
    )
    assert result is not None
    assert result["blocks"][0]["inlines"][0]["style"] == {"color": "#ff0000"}


def test_sync_keeps_manual_styles_and_adds_new_match():
    # With earlier rich text: replaying ops keeps the styles while the new match gets its style, and the two do not interfere
    old_rich = _doc(_text_run("蓝", {"color": "#0000ff"}), _text_run("你", {}))
    result = sync_region_rich_translation(
        old_rich,
        _edit_info([[2, 0, "好"]], "蓝你", "蓝你好"),
        raw_mode=False,
        new_translation="蓝你好",
        apply_rules=True,
        old_translation="蓝你",
        rules=_rules(RED_RULE),
    )
    assert result is not None
    runs = [
        (inline["text"], inline["style"])
        for inline in result["blocks"][0]["inlines"]
    ]
    assert runs == [("蓝", {"color": "#0000ff"}), ("你好", {"color": "#ff0000"})]


def test_sync_raw_edit_rule_hits_replacement_product():
    # Typing three periods in the "translation before replacement" box -> they are replaced by an ellipsis -> the rule matches the product of the replacement
    replacements = {
        "common": [(re.compile(re.escape("...")), "…")],
        "horizontal": [],
        "vertical": [],
    }
    old_rich = _doc(_text_run("什么", {"bold": True}))
    result = sync_region_rich_translation(
        old_rich,
        _edit_info([[2, 0, "..."]], "什么", "什么..."),
        raw_mode=True,
        new_translation="什么…",
        direction_value="h",
        replacements=replacements,
        apply_rules=True,
        old_translation="什么",
        rules=_rules({"pattern": "…", "style": {"scale": 2.0}}),
    )
    assert result is not None
    runs = [
        (inline["text"], inline["style"])
        for inline in result["blocks"][0]["inlines"]
    ]
    assert runs == [("什么", {"bold": True}), ("…", {"scale": 2.0})]


def test_sync_apply_rules_false_keeps_legacy_behavior():
    # Switch off: without earlier rich text None is always returned, exactly as in the old version
    result = sync_region_rich_translation(
        None,
        _edit_info([[1, 0, "好"]], "你", "你好"),
        raw_mode=False,
        new_translation="你好",
        old_translation="你",
        rules=_rules(RED_RULE),
    )
    assert result is None


# ─── Floating editor: narrowing an IME full replacement + the rule stage of the state machine ───


def test_ime_full_replace_report_preserves_styles():
    from editor.rich_text_editing import apply_qt_text_change

    doc = _doc(_text_run("你好", {"color": "#ff0000"}))
    # An IME commit that appends one character to a two-character word is reported as a whole-document replacement (removed=2, added=3)
    updated = apply_qt_text_change(doc, "你好", "你好呀", 0, 2, 3)
    assert _runs_of(updated) == [("你好", {"color": "#ff0000"}), ("呀", {})]


def test_state_applies_rules_on_typing():
    from editor.rich_text_editor_state import RichTextEditorState
    from manga_translator.rendering import rich_text_rules as rules_module

    state = RichTextEditorState()
    state.auto_rules_provider = lambda: True
    state.bind_region(0, {"translation": "你", "direction": "h"})

    fake_rules = _rules(RED_RULE)
    original_loader = rules_module.load_rich_text_rules
    rules_module.load_rich_text_rules = lambda file_path=None: fake_rules
    try:
        state.apply_qt_contents_change("你好", 1, 0, 1)
    finally:
        rules_module.load_rich_text_rules = original_loader

    assert _runs_of(state.document) == [("你好", {"color": "#ff0000"})]
    emitted = state.mark_document_emitted()
    assert emitted is not None
    assert emitted[2] == "你好"


def test_state_provider_off_leaves_document_plain():
    from editor.rich_text_editor_state import RichTextEditorState

    state = RichTextEditorState()
    state.bind_region(0, {"translation": "你", "direction": "h"})
    state.apply_qt_contents_change("你好", 1, 0, 1)
    assert _runs_of(state.document) == [("你好", {})]


def main():
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
