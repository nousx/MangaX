"""Column widths that follow the header text, for columns that are otherwise fixed."""

from PyQt6.QtCore import Qt

# Room for the header's own padding and the sort indicator area on both sides of the text.
_HEADER_TEXT_PADDING = 24


def fit_fixed_columns_to_header_text(table, minimum_widths: dict[int, int]) -> None:
    """Widen fixed-width columns until their header text fits.

    ``minimum_widths`` maps a column to the width it has when its header is
    short. A translated header can be much longer than the original one; a
    column that keeps a hard-coded width then shows a cut-off title.
    """
    model = table.model()
    if model is None:
        return
    metrics = table.horizontalHeader().fontMetrics()
    for column, minimum in minimum_widths.items():
        text = model.headerData(column, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        needed = metrics.horizontalAdvance(str(text or "")) + _HEADER_TEXT_PADDING
        table.setColumnWidth(column, max(minimum, needed))
