"""Family-first manga font library using the existing Fluent popup shell."""
import json
from pathlib import Path

from PyQt6.QtCore import QPoint, QSettings, Qt
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import BodyLabel, ComboBox, PushButton

from .font_list import (
    FONT_COVERAGE_FULL,
    FONT_COVERAGE_NONE,
    _FontComboBoxMenu,
    display_font_family,
    font_sample_line,
    font_text_coverage,
    fonts_directory,
    qfont_for_value,
    split_font_value,
)

ALIASES = {'itim': 'ไอติม', 'mali': 'มะลิ', 'sriracha': 'ศรีราชา', 'sarabun': 'สารบรรณ',
           'kanit': 'คณิต', 'mitr': 'มิตร', 'pridi': 'ปรีดี', 'prompt': 'พร้อมท์',
           'noto-sans-thai': 'โนโตะ ไทย', 'chonburi': 'ชลบุรี', 'pattaya': 'พัทยา',
           'purisa': 'ภูริษา', 'sawasdee': 'สวัสดี', 'garuda': 'ครุฑ', 'loma': 'โลมา',
           'umpush': 'อัมพุช', 'waree': 'วารี', 'kinnari': 'กินรี', 'norasi': 'นรสีห์',
           'laksaman': 'ลักษมัณ'}


class ThaiFontMenu(_FontComboBoxMenu):
    def __init__(self, entries, placeholder, sample, thai, parent=None, *,
                 sample_rows=False, locale_code=''):
        self.full_entries = list(entries)
        self.groups = {}
        catalog_path = Path(fonts_directory()) / 'thai-font-catalog.json'
        catalog = json.loads(catalog_path.read_text(encoding='utf-8')) if catalog_path.is_file() else {'families': []}
        metadata = {entry['family']: entry for entry in catalog['families']}
        self.faces = {}
        for original_index, entry in enumerate(entries):
            family, _ = split_font_value(entry[1])
            self.faces.setdefault(family, []).append((original_index, entry))
        families = []
        for family, faces in self.faces.items():
            default = next((entry for _, entry in faces if split_font_value(entry[1])[1] == 'Regular'), faces[0][1])
            meta = metadata.get(family, {})
            group = meta.get('group', 'other')
            self.groups[family] = group
            aliases = ALIASES.get(meta.get('source_id'), '')
            families.append((family, default[1], ' '.join((family, default[2], meta.get('note', ''), aliases))))
        self.family_entries = families
        self.thai = thai
        self.settings = QSettings('MangaX', 'FontLibrary')
        self.favorites = set(self.settings.value('favorites', [], type=list))
        sample = font_sample_line(sample)
        self.sample_text = sample or ('สวัสดี! ไปผจญภัยกันเถอะ' if thai else "I'll protect you!")
        # Rows preview the caller's text. Combos without one keep name rows unless
        # the owner asked for sample rows (then the default sentence is shown).
        # Rows show the family without the bundled-font prefix; `family_entries`
        # keeps the real family name, which is the key everywhere else.
        rows = [(display_font_family(family), value, search) for family, value, search in families]
        super().__init__(rows, placeholder, parent,
                         sample_text=self.sample_text if sample or sample_rows else '',
                         locale_code=locale_code or ('th_TH' if thai else 'en_US'))
        row = QHBoxLayout()
        self.group_combo = ComboBox(self)
        for label, value in ([('ทั้งหมด', 'all'), ('บทพูด', 'dialogue'), ('ลายมือ', 'hand'),
                               ('บรรยาย', 'story'), ('เอฟเฟกต์', 'effect'), ('รายการโปรด', 'favorites'), ('ฟอนต์อื่น', 'other')]
                              if thai else [('All', 'all'), ('Dialogue', 'dialogue'), ('Handwriting', 'hand'),
                                            ('Narration', 'story'), ('Effects', 'effect'), ('Favorites', 'favorites'), ('Other', 'other')]):
            self.group_combo.addItem(label, userData=value)
        row.addWidget(self.group_combo, 1)
        self.star_button = PushButton('☆', self)
        self.star_button.setAccessibleName('รายการโปรด' if thai else 'Favorite font')
        row.addWidget(self.star_button)
        self._content_layout.insertLayout(1, row)
        self.preview = BodyLabel(self.sample_text, self)
        self.preview.setWordWrap(True)
        self.preview.setMinimumHeight(74)
        self._content_layout.addWidget(self.preview)
        self.styles = ComboBox(self)
        self.styles.setAccessibleName('น้ำหนักและตัวเอียง' if thai else 'Weight and style')
        self._content_layout.addWidget(self.styles)
        self.apply_button = PushButton('ใช้ฟอนต์นี้' if thai else 'Use font', self)
        self._content_layout.addWidget(self.apply_button)
        self.group_combo.currentIndexChanged.connect(self._filter_family)
        self.search_edit.textChanged.disconnect(self._filter_items)
        self.search_edit.textChanged.connect(self._filter_family)
        self.styles.currentIndexChanged.connect(self._preview_style)
        self.star_button.clicked.connect(self._toggle_favorite)
        self.apply_button.clicked.connect(self._apply_style)
        self._selected_family = ''

    def _filter_family(self, *_):
        query = self.search_edit.text().strip().casefold()
        group = self.group_combo.currentData()
        # Hide source rows with the existing proxy; family metadata stays stable.
        proxy = self.view._filter_model
        proxy._family_filter = lambda row: (
            (not query or query in self.family_entries[row][2].casefold()) and
            (group == 'all' or group == 'favorites' and self.family_entries[row][0] in self.favorites
             or self.groups[self.family_entries[row][0]] == group))
        proxy.invalidateRowsFilter()
        self._layout_for_anchor()

    def adjustSize(self):
        QWidget.adjustSize(self)

    def _layout_for_anchor(self):
        if self._anchor is None:
            return
        point = self._anchor.mapToGlobal(QPoint())
        screen = QGuiApplication.screenAt(point) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        width = min(max(430, self._anchor.width()), available.width() - 16)
        # Sample rows are taller; grow the popup so about as many stay visible.
        height = min(700 if self.row_sample_text else 550, available.height() - 16)
        self.search_edit.setFixedHeight(32)
        self.preview.setFixedHeight(84)
        self.view.setFixedSize(width - 30, max(70, height - 270))
        self._opens_upward = False
        self.setFixedSize(width, height)
        self._container.setFixedSize(width - 30, height - 20)
        self._content_layout.activate()
        self.layout().activate()
        x = max(available.left()+8, min(point.x(), available.right()-width-8))
        y = max(available.top()+8, min(point.y()+self._anchor.height(), available.bottom()-height-8))
        self.move(x, y)

    def _on_index_clicked(self, index):
        row = self.view.source_row(index)
        if 0 <= row < len(self.family_entries):
            self._select_family(self.family_entries[row][0])

    def _on_index_entered(self, index):
        row = self.view.source_row(index)
        if 0 <= row < len(self.family_entries):
            self.fontHovered.emit(self.family_entries[row][1])

    def _select_family(self, family, current=None):
        self._selected_family = family
        self.styles.blockSignals(True)
        self.styles.clear()
        for original_index, entry in self.faces[family]:
            _, style = split_font_value(entry[1])
            self.styles.addItem(style or ('ปกติ' if self.thai else 'Regular'), userData=original_index)
        wanted = next((i for i, (_, entry) in enumerate(self.faces[family]) if entry[1] == current),
                      next((i for i, (_, entry) in enumerate(self.faces[family]) if split_font_value(entry[1])[1] == 'Regular'), 0))
        self.styles.setCurrentIndex(wanted)
        self.styles.blockSignals(False)
        self.star_button.setText('★' if family in self.favorites else '☆')
        self._preview_style()

    def selectValue(self, value):
        family, _ = split_font_value(value)
        if family not in self.faces:
            family = self.family_entries[0][0] if self.family_entries else ''
        if family:
            self.view.setCurrentRow(next(i for i, entry in enumerate(self.family_entries) if entry[0] == family))
            self._select_family(family, value)

    def _preview_style(self, *_):
        index = self.styles.currentData()
        if index is None:
            return
        value = self.full_entries[index][1]
        coverage = font_text_coverage(value, self.sample_text)
        if coverage == FONT_COVERAGE_NONE:
            # Same honesty rule as the rows: no substituted glyphs in the preview.
            font = QFont(self.font())
            self.preview.setText(self.no_glyph_hint)
        else:
            font = qfont_for_value(value)
            font.setPixelSize(25)
            if coverage != FONT_COVERAGE_FULL:
                font.setStyleStrategy(QFont.StyleStrategy.NoFontMerging)
            self.preview.setText(self.sample_text)
        self.preview.setFont(font)
        self.fontHovered.emit(value)

    def _toggle_favorite(self):
        family = self._selected_family
        if not family:
            return
        if family in self.favorites:
            self.favorites.remove(family)
        else:
            self.favorites.add(family)
        self.settings.setValue('favorites', sorted(self.favorites))
        self.star_button.setText('★' if family in self.favorites else '☆')
        self._filter_family()

    def _apply_style(self):
        index = self.styles.currentData()
        if index is not None:
            self.fontSelected.emit(index)
            self._hideMenu(True)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._apply_style()
        else:
            super().keyPressEvent(event)
