"""
译文标记高亮器
为译文内容和标记提供视觉化的语法高亮
"""

from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat


class TranslationMarkupHighlighter(QSyntaxHighlighter):
    """
    译文标记高亮器
    根据标记框中的信息，在内容框中高亮显示对应的文本
    """
    
    def __init__(self, parent_document, markup_getter=None):
        """
        Args:
            parent_document: 要应用高亮的文档（QTextEdit.document()）
            markup_getter: 获取标记信息的回调函数，返回标记字符串
        """
        super().__init__(parent_document)
        self.markup_getter = markup_getter
        self._setup_formats()
    
    def _setup_formats(self):
        """设置不同类型标记的格式"""
        # Format for line break positions - shows a special symbol
        self.newline_format = QTextCharFormat()
        self.newline_format.setBackground(QColor("#FFF3E0"))  # light orange
        self.newline_format.setForeground(QColor("#F57C00"))  # orange
    
    def set_markup_getter(self, getter):
        """设置标记获取函数"""
        self.markup_getter = getter
    
    def highlightBlock(self, text):
        """高亮当前文本块"""
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
        解析标记文本
        
        Returns:
            newline_positions: [pos, ...] 换行位置
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
    标记框的语法高亮器
    为标记框中的标记符号添加颜色
    """
    
    def __init__(self, parent_document):
        super().__init__(parent_document)
        self._setup_formats()
    
    def _setup_formats(self):
        """设置格式"""
        # Format of the line break marker
        self.newline_mark_format = QTextCharFormat()
        self.newline_mark_format.setForeground(QColor("#F57C00"))  # orange
        self.newline_mark_format.setFontWeight(QFont.Weight.Bold)
        
        # Number format
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#00897B"))  # cyan
    
    def highlightBlock(self, text):
        """高亮当前文本块"""
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
