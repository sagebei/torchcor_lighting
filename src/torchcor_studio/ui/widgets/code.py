"""Python code viewer and editor (syntax highlighting, line numbers) and a live log console."""

from __future__ import annotations

import re

from PySide6.QtCore import QRect, QRegularExpression, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeyEvent,
    QPainter,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PySide6.QtWidgets import QHBoxLayout, QPlainTextEdit, QVBoxLayout, QWidget

from torchcor_studio.ui.theme import Palette, mono_font, theme
from torchcor_studio.ui.widgets.basic import IconButton, SearchField, label
from torchcor_studio.ui.widgets.controls import ToggleSwitch

_KEYWORDS = (
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif",
    "else", "except", "False", "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True", "try", "while",
    "with", "yield",
)
_BUILTINS = (
    "print", "range", "len", "float", "int", "str", "list", "dict", "tuple", "set", "enumerate",
    "zip", "open", "min", "max", "sum", "abs",
)


class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document: QTextDocument) -> None:
        super().__init__(document)
        self._rules: list[tuple[QRegularExpression, QTextCharFormat]] = []
        self._string = QTextCharFormat()
        self._tri = QRegularExpression(r'"""|\'\'\'')
        theme().changed.connect(self._restyle)
        self._restyle(theme().palette)

    def _fmt(self, color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
        f = QTextCharFormat()
        f.setForeground(QColor(color))
        if bold:
            f.setFontWeight(QFont.Weight.DemiBold)
        f.setFontItalic(italic)
        return f

    def _restyle(self, p: Palette) -> None:
        kw = self._fmt("#C792EA" if p.dark else "#7C3AED", bold=True)
        builtin = self._fmt("#82AAFF" if p.dark else "#1D4FD8")
        number = self._fmt("#F78C6C" if p.dark else "#C2410C")
        comment = self._fmt(p.text3, italic=True)
        deco = self._fmt(p.ember)
        func = self._fmt("#7FDBCA" if p.dark else "#0F766E")
        self._string = self._fmt("#C3E88D" if p.dark else "#15803D")
        self._rules = [
            (QRegularExpression(r"\b(" + "|".join(_KEYWORDS) + r")\b"), kw),
            (QRegularExpression(r"\b(" + "|".join(_BUILTINS) + r")\b(?=\()"), builtin),
            (QRegularExpression(r"\bdef\s+(\w+)|\bclass\s+(\w+)"), func),
            (QRegularExpression(r"\b[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?\b"), number),
            (QRegularExpression(r"@[\w.]+"), deco),
            (QRegularExpression(r"[rbfu]?\"[^\"\\n]*\"|[rbfu]?'[^'\\n]*'"), self._string),
            (QRegularExpression(r"#[^\n]*"), comment),
        ]
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        for pattern, fmt in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)
        # triple-quoted strings spanning blocks
        self.setCurrentBlockState(0)
        start = 0
        if self.previousBlockState() != 1:
            m = self._tri.match(text)
            start = m.capturedStart() if m.hasMatch() else -1
        while start >= 0:
            m = self._tri.match(text, start + (3 if self.previousBlockState() != 1 or start else 0))
            if m.hasMatch():
                end = m.capturedEnd()
                self.setFormat(start, end - start, self._string)
                nxt = self._tri.match(text, end)
                start = nxt.capturedStart() if nxt.hasMatch() else -1
            else:
                self.setCurrentBlockState(1)
                self.setFormat(start, len(text) - start, self._string)
                break


class CodeView(QPlainTextEdit):
    """Read-only, syntax-highlighted Python source."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("code", True)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFont(mono_font(12))
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self._hl = PythonHighlighter(self.document())
        self.setPlainText(text)


class _Gutter(QWidget):
    def __init__(self, editor: CodeEditor) -> None:
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.editor.gutter_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802
        self.editor.paint_gutter(event)


class CodeEditor(QPlainTextEdit):
    """Editable Python source: highlighting, line numbers, 4-space indents, auto-indent."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("code", True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFont(mono_font(12))
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self._hl = PythonHighlighter(self.document())
        self._gutter = _Gutter(self)
        self.blockCountChanged.connect(lambda *_: self._update_margins())
        self.updateRequest.connect(self._scroll_gutter)
        self.setPlainText(text)
        self._update_margins()

    def gutter_width(self) -> int:
        digits = max(3, len(str(self.blockCount())))
        return 14 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_margins(self) -> None:
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _scroll_gutter(self, rect: QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._gutter.setGeometry(QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def paint_gutter(self, event) -> None:
        p = theme().palette
        painter = QPainter(self._gutter)
        painter.fillRect(event.rect(), QColor(p.plot_bg))
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        current = self.textCursor().blockNumber()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor(p.text2 if number == current else p.text3))
                painter.drawText(0, top, self._gutter.width() - 8, self.fontMetrics().height(),
                                 Qt.AlignmentFlag.AlignRight, str(number + 1))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            number += 1

    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Tab and not e.modifiers():
            self.textCursor().insertText("    ")
            return
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            line = self.textCursor().block().text()
            indent = line[: len(line) - len(line.lstrip(" "))]
            if line.rstrip().endswith(":"):
                indent += "    "
            super().keyPressEvent(e)
            self.textCursor().insertText(indent)
            return
        super().keyPressEvent(e)


class LogView(QWidget):
    """Console for live run logs: batched appends, follow-tail toggle, find, copy."""

    MAX_BLOCKS = 20_000

    def __init__(self, parent: QWidget | None = None, toolbar: bool = True) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.edit = QPlainTextEdit()
        self.edit.setProperty("console", True)
        self.edit.setReadOnly(True)
        self.edit.setMaximumBlockCount(self.MAX_BLOCKS)
        self.edit.setFont(mono_font(12))
        self.edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.follow = ToggleSwitch(True)
        if toolbar:
            bar = QHBoxLayout()
            bar.setSpacing(8)
            self.search = SearchField("Find in log")
            self.search.setMaximumWidth(280)
            self.search.returnPressed.connect(self.find_next)
            bar.addWidget(self.search)
            bar.addStretch(1)
            bar.addWidget(label("Follow", "muted"))
            bar.addWidget(self.follow)
            bar.addSpacing(6)
            bar.addWidget(IconButton("copy", "Copy log", on_click=self.copy_all))
            bar.addWidget(IconButton("chevron-down", "Jump to end", on_click=self.scroll_to_end))
            lay.addLayout(bar)
        lay.addWidget(self.edit, 1)

    def clear(self) -> None:
        self.edit.clear()

    def append(self, text: str) -> None:
        if not text:
            return
        text = _strip_ansi(text).replace("\r\n", "\n")
        # carriage-return progress bars (tqdm) collapse to their last state
        text = "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))
        bar = self.edit.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 4
        cursor = QTextCursor(self.edit.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        if self.follow.isChecked() and at_end:
            self.scroll_to_end()

    def set_text(self, text: str) -> None:
        self.edit.setPlainText(_strip_ansi(text))
        self.scroll_to_end()

    def scroll_to_end(self) -> None:
        bar = self.edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def copy_all(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.edit.toPlainText())

    def find_next(self) -> None:
        needle = self.search.text()
        if not needle:
            return
        if not self.edit.find(needle):
            self.edit.moveCursor(QTextCursor.MoveOperation.Start)
            self.edit.find(needle)
        self.follow.setChecked(False)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


__all__ = ["CodeEditor", "CodeView", "LogView", "PythonHighlighter"]
