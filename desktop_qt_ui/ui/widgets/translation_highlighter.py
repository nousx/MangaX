"""
Translation markup highlighter.
Provides visual syntax highlighting for translation content and markup
"""

from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat


class TranslationMarkupHighlighter(QSyntaxHighlighter):
    """
    Translation markup highlighter.
    Highlights the matching text in the content box from the information in the markup box
    """
    
    def __init__(self, parent_document, markup_getter=None):
        """
        Args:
            parent_document: the document to highlight (QTextEdit.document())
            markup_getter: callback that gets the markup information; returns the markup string
        """
        super().__init__(parent_document)
        self.markup_getter = markup_getter
        self._setup_formats()
    
    def _setup_formats(self):
        """Set the formats of the different kinds of markup"""
        # Format for line break positions - shows a special symbol
        self.newline_format = QTextCharFormat()
        self.newline_format.setBackground(QColor("#FFF3E0"))  # light orange
        self.newline_format.setForeground(QColor("#F57C00"))  # orange
    
    def set_markup_getter(self, getter):
        """Set the function that gets the markup"""
        self.markup_getter = getter
    
    def highlightBlock(self, text):
        """Highlight the current text block"""
        if not self.markup_getter:
            return
        
        markup_text = self.markup_getter()
        if not markup_text:
            return
        
        # Parse the markers
        newline_positions = self._parse_markup(markup_text)
        
        # Offset of the current block in the whole document
        block_start = self.currentBlock().position()
        block_length = len(text)
        
        # Add a visual marker after the line break position
        for pos in newline_positions:
            if block_start <= pos < block_start + block_length:
                rel_pos = pos - block_start
                # Highlight one character after the line break position
                if rel_pos < block_length:
                    self.setFormat(rel_pos, 1, self.newline_format)
    
    def _parse_markup(self, markup_text):
        """
        Parse the markup text

        Returns:
            newline_positions: [pos, ...] the line break positions
        """
        newline_positions = []
        
        for mark in markup_text.split():
            if mark.startswith('↵'):
                # Line break marker
                try:
                    pos = int(mark[1:])
                    newline_positions.append(pos)
                except ValueError:
                    pass
        
        return newline_positions


class MarkupBoxHighlighter(QSyntaxHighlighter):
    """
    Syntax highlighter of the markup box.
    Adds colour to the markup symbols in the markup box
    """
    
    def __init__(self, parent_document):
        super().__init__(parent_document)
        self._setup_formats()
    
    def _setup_formats(self):
        """Set the formats"""
        # Format of the line break marker
        self.newline_mark_format = QTextCharFormat()
        self.newline_mark_format.setForeground(QColor("#F57C00"))  # orange
        self.newline_mark_format.setFontWeight(QFont.Weight.Bold)
        
        # Number format
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#00897B"))  # cyan
    
    def highlightBlock(self, text):
        """Highlight the current text block"""
        # Highlight the line break marker ↵
        newline_pattern = QRegularExpression(r'↵\d+')
        match_iterator = newline_pattern.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            self.setFormat(match.capturedStart(), 1, self.newline_mark_format)  # the ↵ symbol
            # The number part
            num_start = match.capturedStart() + 1
            num_length = match.capturedLength() - 1
            self.setFormat(num_start, num_length, self.number_format)
