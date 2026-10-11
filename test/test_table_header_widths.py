"""Fixed-width table columns have to grow with their translated header text."""
import _bootstrap  # noqa: F401, I001

import pytest
from PyQt6.QtWidgets import QApplication, QTableWidget

from ui.widgets.table_headers import fit_fixed_columns_to_header_text


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def table_with_headers(*labels: str) -> QTableWidget:
    table = QTableWidget(0, len(labels))
    table.setHorizontalHeaderLabels(list(labels))
    return table


def test_column_should_grow_when_its_header_text_is_wider_than_the_minimum(application):
    table = table_with_headers("A considerably longer column title", "Text")

    fit_fixed_columns_to_header_text(table, {0: 55})

    text_width = table.horizontalHeader().fontMetrics().horizontalAdvance("A considerably longer column title")
    assert table.columnWidth(0) > text_width


def test_column_should_keep_its_minimum_when_the_header_text_is_short(application):
    table = table_with_headers("On", "Text")

    fit_fixed_columns_to_header_text(table, {0: 55})

    assert table.columnWidth(0) == 55


def test_columns_that_are_not_listed_should_be_left_alone(application):
    table = table_with_headers("A considerably longer column title", "Another long column title")
    table.setColumnWidth(1, 40)

    fit_fixed_columns_to_header_text(table, {0: 55})

    assert table.columnWidth(1) == 40
