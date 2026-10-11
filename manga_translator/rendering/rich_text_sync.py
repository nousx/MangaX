"""Rich-text sync driven by edit operations.

Every edit in the plain-text translation box is reported by QTextDocument.contentsChange as an exact
``(position, removed count, inserted text)`` operation record; this module replays those operations at the same positions on the character
entries of the richtext.v1 document, so styles follow the characters that did not change, without any diff guessing.

Coordinate convention: all text is the original text of the edit box with ``\\n`` line breaks, in one-to-one correspondence with
``plain_text()`` of the document; the ``[BR]`` form of the model layer does not appear in this module.

Style policy (decided by the user):
- Characters inserted by an edit only inherit a style "in the middle" - when both neighbours exist, neither is a line break
  and both have the same style; inserted at the start or end of the text or of a style run they inherit nothing (empty style).
- A range matched by a replacement rule is replaced as a whole, and the new characters take the style of the first character of the replaced span.
- ruby/tcy nodes: when the neighbours before and after the insertion point belong to the same node, the new characters join that node
  (so the node is not cut in two); when the replaced range lies entirely inside one node, the result keeps the node.

When any validation fails (the text does not match, an operation is out of range) ``None`` is returned, and the caller falls back to
the safe path of "delete the rich text, keep only the plain text".
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, List, Optional, Sequence

from .rich_text import (
    RichTextDocument,
    ensure_rich_text_document,
    normalize_rich_linebreaks,
)
from .rich_text_rules import (
    _document_from_rule_entries,
    _rule_entries_from_document,
    _RuleEntry,
    apply_rich_text_rules,
)
from .text_replacements import load_replacements
from manga_translator.utils.swallowed import note_ignored_error

logger = logging.getLogger(__name__)


def _text_of(entries: Sequence[_RuleEntry]) -> str:
    return "".join(entry.char for entry in entries)


def _copy_entries(entries: Sequence[_RuleEntry]) -> List[_RuleEntry]:
    # node keeps the same object reference: rebuilding the document groups by identity, and a deep copy would break the nodes apart.
    return [
        _RuleEntry(entry.char, copy.deepcopy(entry.style), node=entry.node)
        for entry in entries
    ]


def _collapse_linebreak_entries(entries: Sequence[_RuleEntry]) -> List[_RuleEntry]:
    """Collapse consecutive line breaks into one, in line with the model layer's ``\\n+ -> [BR]`` convention."""
    out: List[_RuleEntry] = []
    for entry in entries:
        if entry.char == "\n" and out and out[-1].char == "\n":
            continue
        out.append(entry)
    return out


def _apply_edit_ops(entries: List[_RuleEntry], ops: Sequence[Any]) -> List[_RuleEntry]:
    """Replay the edit operations in order; a position out of range raises ValueError."""
    for op in ops:
        position, removed, inserted = int(op[0]), int(op[1]), str(op[2])
        if position < 0 or removed < 0 or position > len(entries):
            raise ValueError(
                f"edit op out of range: pos={position} removed={removed} len={len(entries)}"
            )
        # Qt quirk: when a change reaches the end of the document, contentsChange counts the final paragraph separator
        # in charsRemoved (replacing all 6 characters reports removed=7); it is clamped to the real length here,
        # and a wrong clamp is still caught by the caller's post_text check.
        removed = min(removed, len(entries) - position)
        del entries[position : position + removed]

        prev_entry = entries[position - 1] if position > 0 else None
        next_entry = entries[position] if position < len(entries) else None
        inherited_style: dict = {}
        if (
            prev_entry is not None
            and next_entry is not None
            and prev_entry.char != "\n"
            and next_entry.char != "\n"
            and prev_entry.style == next_entry.style
        ):
            inherited_style = prev_entry.style
        # Node membership is decided separately: text inside one node must be merged into it, otherwise the node is cut in two.
        inherited_node = None
        if (
            prev_entry is not None
            and next_entry is not None
            and prev_entry.node is not None
            and prev_entry.node is next_entry.node
        ):
            inherited_node = prev_entry.node

        new_entries: List[_RuleEntry] = []
        for char in inserted:
            if char == "\n":
                new_entries.append(_RuleEntry("\n", {}))
            else:
                new_entries.append(
                    _RuleEntry(
                        char, copy.deepcopy(inherited_style), node=inherited_node
                    )
                )
        entries[position:position] = new_entries
    return entries


def apply_replacements_to_entries(
    entries: List[_RuleEntry],
    direction: int,
    replacements: Optional[dict] = None,
) -> List[_RuleEntry]:
    """Run the replacement rules on a list of character entries (the entry version of text_replacements).

    Difference from the plain-text version: that one protects line-break markers such as ``[BR]`` with placeholders; here the text
    is in ``\\n`` convention, and matches that span a line break are skipped, which is equivalent. The new characters produced by a replacement
    inherit the style of the first character of the replaced span; when the range does not lie entirely inside one ruby/tcy node, the result has no node.
    """
    if replacements is None:
        replacements = load_replacements()

    group_key = "vertical" if direction == 1 else "horizontal"
    for key in ("common", group_key):
        for pattern, repl in replacements.get(key, []):
            text = _text_of(entries)
            matches = list(pattern.finditer(text))
            for match in reversed(matches):
                if match.start() == match.end():
                    continue
                if "\n" in match.group(0):
                    continue
                try:
                    replacement_text = match.expand(repl)
                except Exception as ignored_error:
                    note_ignored_error(ignored_error, "manga_translator/rendering/rich_text_sync.py:apply_replacements_to_entries")
                    replacement_text = repl
                span = entries[match.start() : match.end()]
                first = span[0]
                node = first.node if all(e.node is first.node for e in span) else None
                new_entries: List[_RuleEntry] = []
                for char in replacement_text:
                    if char == "\n":
                        new_entries.append(_RuleEntry("\n", {}))
                    else:
                        new_entries.append(
                            _RuleEntry(char, copy.deepcopy(first.style), node=node)
                        )
                entries[match.start() : match.end()] = new_entries
    return entries


def _map_styles_to_raw(
    raw_text: str,
    doc_entries: List[_RuleEntry],
    direction: int,
    replacements: Optional[dict],
) -> Optional[List[_RuleEntry]]:
    """Move the styles anchored on the text after replacement back to raw coordinates.

    The entry version of the replacement is run again with sentinel styles in which "each character carries its own index", which gives
    the correspondence translation position -> raw position: characters that were not replaced correspond one to one (exactly),
    and the output characters of a replaced range all correspond to its first character (consistent with the first-character style policy of replacement).
    None is returned when no correspondence can be made (for example when the rule version changed).
    """
    doc_text = _text_of(doc_entries)
    if raw_text == doc_text:
        return _copy_entries(doc_entries)

    corr = [_RuleEntry(char, {"__src": i}) for i, char in enumerate(raw_text)]
    corr = apply_replacements_to_entries(corr, direction, replacements)
    if _text_of(corr) != doc_text:
        return None

    style_by_src: dict = {}
    node_by_src: dict = {}
    for pos, entry in enumerate(corr):
        src = entry.style.get("__src")
        if src is None or src in style_by_src:
            continue
        style_by_src[src] = doc_entries[pos].style
        node_by_src[src] = doc_entries[pos].node

    result: List[_RuleEntry] = []
    last_style: dict = {}
    last_node = None
    for i, char in enumerate(raw_text):
        if i in style_by_src:
            last_style = style_by_src[i]
            last_node = node_by_src[i]
        # Characters without a mapping (the non-first characters of a replaced span) take the style of the span start (inherited from the left).
        result.append(_RuleEntry(char, copy.deepcopy(last_style), node=last_node))
    return result


def document_after_edit_ops(
    document_value: Any,
    ops: Sequence[Any],
    pre_text: str,
    post_text: str,
) -> Optional[RichTextDocument]:
    """Editing the "translation" directly: the operation positions are in the same coordinates as the document body and are replayed in place."""
    try:
        document = ensure_rich_text_document(document_value)
    except Exception as exc:
        logger.warning("Failed to parse rich text document: %s", exc)
        return None

    entries = _collapse_linebreak_entries(_rule_entries_from_document(document))
    if _text_of(entries) != pre_text:
        return None
    try:
        entries = _apply_edit_ops(_copy_entries(entries), ops)
    except ValueError as exc:
        logger.warning("Failed to replay edit operations: %s", exc)
        return None
    if _text_of(entries) != post_text:
        return None
    return _document_from_rule_entries(_text_of(entries), entries)


def sync_document_for_raw_edit(
    document_value: Any,
    raw_pre_text: str,
    ops: Sequence[Any],
    raw_post_text: str,
    direction: int,
    replacements: Optional[dict] = None,
) -> Optional[RichTextDocument]:
    """Editing the "translation before replacement": move the styles back to raw coordinates -> replay the operations -> run the replacement again."""
    try:
        document = ensure_rich_text_document(document_value)
    except Exception as exc:
        logger.warning("Failed to parse rich text document: %s", exc)
        return None

    doc_entries = _collapse_linebreak_entries(_rule_entries_from_document(document))
    entries = _map_styles_to_raw(raw_pre_text, doc_entries, direction, replacements)
    if entries is None:
        return None
    try:
        entries = _apply_edit_ops(entries, ops)
    except ValueError as exc:
        logger.warning("Failed to replay edit operations: %s", exc)
        return None
    if _text_of(entries) != raw_post_text:
        return None
    entries = apply_replacements_to_entries(entries, direction, replacements)
    return _document_from_rule_entries(_text_of(entries), entries)


def document_has_styling(document: RichTextDocument) -> bool:
    """Whether the document still has any style or ruby/tcy node; if not, the rich-text field is not worth keeping."""
    entries = _rule_entries_from_document(document)
    return any(entry.style or entry.node is not None for entry in entries)


def _direction_to_int(direction_value: Any) -> int:
    """Layout direction value of the region -> direction argument of the replacement rules (0 = horizontal, 1 = vertical)."""
    return 0 if direction_value in ("h", "horizontal", "hr") else 1


def _model_text_matches(document: RichTextDocument, model_text: str) -> bool:
    """Fold the document body back to the model convention (\\n+ → [BR]) and compare it with the translation field."""
    return re.sub(r"\n+", "[BR]", document.plain_text()) == model_text


def _apply_editor_rules_stage(
    document: Optional[RichTextDocument],
    incremental: bool,
    new_translation: str,
    old_translation: Optional[str],
    direction_value: Any,
    rules: Optional[dict],
) -> Optional[RichTextDocument]:
    """Third stage of the editor pipeline: apply the automatic rich-text rules to the result of the sync (or to plain text).

    With ``incremental`` (an exact operation record exists) the matches on the old and the new translation are compared and only the matches
    newly produced by the edit are applied - even when the old rich text does not exist or the sync failed, old matches that did not change
    are not styled again (a style that was cleared does not come back). A whole-text replacement has full semantics (the same as the render
    pipeline: every match counts as new). Rules only add styles and never change characters; when the body of the result does not match the translation,
    the result of the rules is dropped as a safeguard.
    """
    previous_text = None
    if incremental and old_translation is not None:
        previous_text = normalize_rich_linebreaks(str(old_translation))
    base: Any = (
        document
        if document is not None
        else normalize_rich_linebreaks(str(new_translation or ""))
    )
    try:
        ruled = apply_rich_text_rules(
            base,
            direction_value,
            rules,
            previous_text=previous_text,
            styled_match_policy="skip",
        )
    except Exception as exc:
        logger.warning("Failed to apply automatic rich text rules in the editor; preserving the synchronized result: %s", exc)
        return document
    if ruled is None:
        return document
    if not _model_text_matches(ruled, new_translation):
        logger.warning("Automatic rich text rule output does not match the translation; preserving the synchronized result")
        return document
    return ruled


def sync_region_rich_translation(
    old_rich: Any,
    edit_info: Any,
    *,
    raw_mode: bool,
    new_translation: str,
    direction_value: Any = "h",
    replacements: Optional[dict] = None,
    apply_rules: bool = False,
    old_translation: Optional[str] = None,
    rules: Optional[dict] = None,
) -> Optional[dict]:
    """Single entry point for aligning rich text with an edit of a region's translation.

    Validate the operation record -> sync the document -> fold the body back to the model convention ([BR]) and compare it with the new translation ->
    (optionally) apply the automatic rich-text rules -> degrade when there is no style. Returns a dict that can be written straight back to
    ``translation_rich``.

    With ``apply_rules=False`` the behaviour is exactly that of the old version: without old rich text, without an operation
    record or when any validation fails, ``None`` is returned (the caller deletes the rich text and goes back to plain text).
    With ``apply_rules=True`` the rules still run on the plain text after a failed sync or a whole-text replacement, so rich text may
    appear where there was none; ``old_translation`` takes the translation field from before the edit ([BR]
    convention) for comparing old and new matches, and the whole-text replacement path passes ``None``, which means full semantics.
    """
    info = edit_info if isinstance(edit_info, dict) else None
    ops = info.get("ops") if info else None

    document: Optional[RichTextDocument] = None
    if old_rich and not ops:
        logger.info("The entire translation was replaced without edit operations; reverting rich text to plain text")
    elif old_rich:
        try:
            if raw_mode:
                document = sync_document_for_raw_edit(
                    old_rich,
                    info.get("pre_text", ""),
                    ops,
                    info.get("post_text", ""),
                    _direction_to_int(direction_value),
                    replacements,
                )
            else:
                document = document_after_edit_ops(
                    old_rich,
                    ops,
                    info.get("pre_text", ""),
                    info.get("post_text", ""),
                )
        except Exception as exc:
            logger.warning("Error synchronizing rich text styles; reverting to plain text: %s", exc)
            document = None
        else:
            if document is None:
                logger.warning("Rich text style synchronization validation failed; reverting to plain text")
            elif not _model_text_matches(document, new_translation):
                logger.warning("Synchronized rich text content does not match the translation; reverting to plain text")
                document = None

    if apply_rules:
        document = _apply_editor_rules_stage(
            document,
            bool(ops),
            new_translation,
            old_translation,
            direction_value,
            rules,
        )

    if document is None or not document_has_styling(document):
        return None
    return document.to_dict()
