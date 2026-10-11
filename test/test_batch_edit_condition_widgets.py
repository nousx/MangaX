"""The condition rows of the batch edit page."""
import _bootstrap  # noqa: F401, I001

import pytest
from PyQt6.QtWidgets import QApplication

from services.batch_edit_engine import FIELDS_BY_KEY
from ui.secondary_pages.batch_edit_condition_widgets import ConditionRow, build_value_editor

TEXTS = {
    "direction_horizontal": "Horizontal",
    "direction_vertical": "Vertical",
    "direction_auto": "Automatic",
    "alignment_left": "Left",
}


def translate(key: str) -> str:
    return TEXTS.get(key, key)


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def choice_labels(editor) -> dict:
    combo = editor._combo
    return {combo.itemData(index): combo.itemText(index) for index in range(combo.count())}


def test_enum_choices_should_show_translated_names_and_keep_the_stored_codes(application):
    editor = build_value_editor(FIELDS_BY_KEY["direction"], "eq", translate)

    assert choice_labels(editor) == {"h": "Horizontal", "v": "Vertical", "auto": "Automatic"}
    editor.set_value("v")
    assert editor.value() == "v"


def test_enum_choice_without_a_translation_should_fall_back_to_its_code(application):
    editor = build_value_editor(FIELDS_BY_KEY["alignment"], "eq", translate)

    labels = choice_labels(editor)

    assert labels["left"] == "Left"
    assert labels["center"] == "center"


def test_replaced_value_editor_should_be_hidden_at_once(application):
    row = ConditionRow(translate)
    row.show()
    first = row._editor
    assert first is not None

    row._rebuild_editor()

    # deleteLater() has not run yet; the old editor must not be drawn over the new one meanwhile.
    assert first is not row._editor
    assert first.isHidden()
    row.deleteLater()
