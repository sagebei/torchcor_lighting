"""The parameter grid: fields sized by meaning in equal columns, with group tiles and blocks.

:class:`FieldGrid` is a layout of 2 to 4 equal columns (by width, or a fixed count). Each
item asks for a span in columns (0: the whole row) and rows pack left to right, wrapping
when a span does not fit. A row holds fields (bottom-aligned, so their controls share a
line) or group tiles (top-aligned, so their eyebrows do), never both; a heading or a block
(a table, the input slot) takes a row of its own, except two half blocks (small tables),
which share one at half the width each. An item can start a row (``brk``) or end one
(``end``: a choice that later fields depend on), also when the items next to it are
hidden. Hidden items take no room: the layout skips them and never removes or re-adds a
widget. ``heightForWidth`` and ``setGeometry`` share one packing.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLayout,
    QLayoutItem,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import WrapLabel, label

#: ``(columns, column width) -> span``; an int span is fixed (0: the whole row).
Span = int | Callable[[int, float], int]

MIN_COLUMN = 170
COLUMN_GAP = 16
ROW_GAP = 14  # between rows of fields
BLOCK_GAP = 18  # around headings and blocks
HEADING_GAP = 8  # from a heading to what it heads


@dataclass
class _Item:
    item: QLayoutItem
    span: Span
    role: str  # "field", "tile", "heading", "block" or "half"
    brk: bool  # starts a new row
    end: bool = False  # nothing follows it on its row
    half: Callable[[], bool] | None = None  # a block that can take half the row
    natural: Callable[[], int] | None = None  # its own width: wider than half, it takes the row


def natural_span(widget: QWidget) -> Callable[[int, float], int]:
    """The columns a fixed-size control needs at its natural width (segmented controls)."""
    def span(columns: int, column: float) -> int:
        return min(columns, max(1, math.ceil((widget.sizeHint().width() + COLUMN_GAP)
                                             / (column + COLUMN_GAP))))
    return span


class FieldGrid(QLayout):
    """Equal columns, ``clamp((width + 16) // (170 + 16), 2, 4)`` of them, or ``columns``
    (a fixed count, or a function of the width: group tiles follow their span)."""

    def __init__(self, parent: QWidget | None = None, *,
                 columns: int | Callable[[], int] | None = None, max_columns: int = 4) -> None:
        self._items: list[_Item] = []
        self._columns = columns
        self._max = max_columns
        self._cache: dict[tuple[int, int], int] = {}
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

    # ------------------------------------------------------------------ building
    def add(self, widget: QWidget, *, span: Span = 1, role: str = "field", brk: bool = False,
            end: bool = False, half: Callable[[], bool] | None = None,
            natural: Callable[[], int] | None = None) -> None:
        """``role``: "field", "tile" (a group tile), "heading" or "block" (the whole row). A
        block for which ``half()`` holds takes half the row, beside another such block,
        unless ``natural()`` (its own width) is wider than half."""
        self.addChildWidget(widget)
        self._items.append(_Item(QWidgetItem(widget), span, role, brk, end, half, natural))
        self.invalidate()

    def columns(self, width: int) -> int:
        if callable(self._columns):
            return max(1, self._columns())
        if self._columns:
            return self._columns
        return max(2, min(self._max, (width + COLUMN_GAP) // (MIN_COLUMN + COLUMN_GAP)))

    def span(self, entry: _Item, columns: int, column: float) -> int:
        """The columns an item takes (whole row for 0); 0 when it is hidden."""
        widget = entry.item.widget()
        if widget is not None and widget.isHidden():
            return 0
        span = entry.span(columns, column) if callable(entry.span) else entry.span
        return columns if span <= 0 else min(span, columns)

    # ------------------------------------------------------------------ QLayout
    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802
        self._items.append(_Item(item, 1, "field", False))

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items[index].item if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items.pop(index).item if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        m = self.contentsMargins()
        inner = width - m.left() - m.right()
        key = (inner, self.columns(inner))
        if key not in self._cache:
            self._cache[key] = self._pack(QRect(0, 0, inner, 0), apply=False)
        return self._cache[key] + m.top() + m.bottom()

    def invalidate(self) -> None:
        self._cache.clear()
        super().invalidate()

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        m = self.contentsMargins()
        self._pack(rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom()), apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        m = self.contentsMargins()
        width = self._max * MIN_COLUMN + (self._max - 1) * COLUMN_GAP + m.left() + m.right()
        return QSize(width, self.heightForWidth(width))

    def minimumSize(self) -> QSize:  # noqa: N802
        return QSize(0, 0)

    # ------------------------------------------------------------------ packing
    def _rows(self, width: int) -> list[tuple[str, list[tuple[QLayoutItem, int, int, str]]]]:
        n = self.columns(width)
        column = (width - (n - 1) * COLUMN_GAP) / n
        half = (width - COLUMN_GAP) / 2
        rows: list[tuple[str, list[tuple[QLayoutItem, int, int, str]]]] = []
        row: list[tuple[QLayoutItem, int, int, str]] = []
        kind, used, pending = "", 0.0, False  # pending: the next visible item starts a row
        for entry in self._items:
            span = self.span(entry, n, column)
            if not span:
                pending = pending or entry.brk  # a hidden item's break still holds
                continue
            role = entry.role
            if entry.half is not None and entry.half() and (entry.natural is None or entry.natural() <= half):
                role = "half"
            if role == "half":  # half the row, whatever the columns
                share, x, w = 0.5, round(used * (width + COLUMN_GAP)), round(half)
            else:
                share = span / n
                x, w = round(used * n * (column + COLUMN_GAP)), round(span * column + (span - 1) * COLUMN_GAP)
            shared = role == kind and role in ("field", "tile", "half")
            if row and (pending or entry.brk or not shared or used + share > 1 + 1e-9):
                rows.append((kind, row))
                row, used = [], 0.0
                x = 0
            kind = role
            row.append((entry.item, x, w, role))
            used += share
            pending = entry.end
        if row:
            rows.append((kind, row))
        return rows

    def _pack(self, rect: QRect, apply: bool) -> int:
        y, previous = rect.y(), ""
        for kind, row in self._rows(rect.width()):
            if previous:
                y += HEADING_GAP if previous == "heading" else (
                    ROW_GAP if kind == previous and kind in ("field", "tile") else BLOCK_GAP)
            heights = [item.heightForWidth(w) if item.hasHeightForWidth() else item.sizeHint().height()
                       for item, _x, w, _role in row]
            top = max(heights)
            if apply:
                for (item, x, w, role), h in zip(row, heights, strict=True):
                    dy = top - h if role == "field" else 0  # fields share their bottom line
                    item.setGeometry(QRect(rect.x() + x, y + dy, w, h))
            y += top
            previous = kind
        return y - rect.y()


def grid_widget(**kwargs) -> tuple[QWidget, FieldGrid]:
    widget = QWidget()
    grid = FieldGrid(widget, **kwargs)
    widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    return widget, grid


# ---------------------------------------------------------------------------- painted text
_SUB = re.compile(r"(?<=\w)_(\w+)")  # "b_fs": b with fs as a subscript


def _parts(text: str) -> list[tuple[str, bool]]:
    """``text`` as (piece, subscript) pieces."""
    out, at = [], 0
    for m in _SUB.finditer(text):
        out += [(text[at:m.start()], False), (m.group(1), True)]
        at = m.end()
    return [*out, (text[at:], False)]


def _sub_font(font: QFont) -> QFont:
    small = QFont(font)
    small.setPixelSize(max(8, round(font.pixelSize() * 0.78)))
    return small


def rich_width(text: str, font: QFont) -> float:
    """The width of ``text`` drawn by :func:`draw_rich`."""
    small = _sub_font(font)
    return sum(QFontMetrics(small if sub else font).horizontalAdvance(piece) for piece, sub in _parts(text))


def draw_rich(painter: QPainter, rect: QRectF, text: str, font: QFont, color: QColor) -> float:
    """Draw ``text`` left-aligned and centred in ``rect`` with "x_y" as x and a subscript y
    (parameter names: b_f, a_fs); a text too wide for ``rect`` is elided instead (without
    subscripts). Returns the width drawn."""
    painter.setPen(color)
    fm = QFontMetrics(font)
    baseline = rect.y() + (rect.height() - fm.height()) / 2 + fm.ascent()
    if rich_width(text, font) > rect.width() + 0.5:
        plain = fm.elidedText(_SUB.sub(r"\1", text), Qt.TextElideMode.ElideRight, int(rect.width()))
        painter.setFont(font)
        painter.drawText(QPointF(rect.x(), baseline), plain)
        return fm.horizontalAdvance(plain)
    small = _sub_font(font)
    x = rect.x()
    for piece, sub in _parts(text):
        painter.setFont(small if sub else font)
        painter.drawText(QPointF(x, baseline + (fm.ascent() * 0.22 if sub else 0)), piece)
        x += QFontMetrics(small if sub else font).horizontalAdvance(piece)
    return x - rect.x()


# ---------------------------------------------------------------------------- cells
def _font(widget: QWidget, size: int, weight: QFont.Weight) -> QFont:
    font = QFont(widget.font())
    font.setPixelSize(size)
    font.setWeight(weight)
    return font


class FieldCell(QWidget):
    """A field: its title and help icon above its editor (or beside a switch). The title is
    painted (12 px, medium, text2), elided when narrow, with the full title as tooltip."""

    GAP = 6  # title to editor
    SWITCH_GAP = 8  # a switch to its title
    ICON = 12

    def __init__(self, title: str, help_text: str, editor: QWidget, *, beside: bool = False,
                 fixed: bool = False) -> None:
        """``fixed``: the editor keeps its natural width (segmented controls)."""
        super().__init__()
        self.title, self.help, self.editor = title, help_text, editor
        self._beside = beside
        self._font = _font(self, 13 if beside else 12, QFont.Weight.Normal if beside else QFont.Weight.Medium)
        self._line = QFontMetrics(self._font).height()
        self._title_w = rich_width(title, self._font)  # measured once: it is drawn often
        lay = QHBoxLayout(self) if beside else QVBoxLayout(self)
        lay.setSpacing(0)
        if beside:  # a switch: on the editors' line, its title on its right
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(editor, 0, Qt.AlignmentFlag.AlignVCenter)
            lay.addStretch(1)
            self.setFixedHeight(32)
        else:
            lay.setContentsMargins(0, self._line + self.GAP, 0, 0)
            lay.addWidget(editor, 0, Qt.AlignmentFlag.AlignLeft if fixed else Qt.AlignmentFlag(0))

    def _text_rect(self) -> QRectF:
        if self._beside:
            x = self.editor.geometry().right() + 1 + self.SWITCH_GAP
            return QRectF(x, 0, max(0, self.width() - x - self.ICON - 6), self.height())
        return QRectF(0, 0, max(0, self.width() - self.ICON - 6), self._line)

    def _shown(self) -> tuple[bool, QRectF, QRectF]:
        """Whether the whole title shows, where it is drawn and where its help icon is."""
        rect = self._text_rect()
        width = min(self._title_w, rect.width())
        whole = self._title_w <= rect.width() + 0.5
        icon = QRectF(rect.x() + width + 5, rect.center().y() - self.ICON / 2 + 0.5, self.ICON, self.ICON)
        return whole, rect, icon

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        text = math.ceil(self._title_w) + self.ICON + 6
        if self._beside:
            return QSize(self.editor.sizeHint().width() + self.SWITCH_GAP + text, 32)
        return QSize(max(hint.width(), text), hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), 48), hint.height())

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _whole, rect, icon = self._shown()
        draw_rich(painter, rect, self.title, self._font, p.qcolor("text" if self._beside else "text2"))
        if self.help:
            painter.drawPixmap(icon.topLeft().toPoint(), icons.pixmap("info", p.text3, self.ICON))

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            whole, rect, icon = self._shown()
            point = QPointF(event.pos())
            tip = (self.help if self.help and icon.adjusted(-3, -3, 3, 3).contains(point)
                   else self.title if not whole and rect.contains(point) else "")
            if tip:
                QToolTip.showText(event.globalPos(), tip, self)
            else:
                QToolTip.hideText()
                event.ignore()
            return True
        return super().event(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._beside and event.button() == Qt.MouseButton.LeftButton and \
                self._text_rect().contains(event.position()):
            self.editor.toggle()  # a click on a switch's title flips it
            return
        super().mousePressEvent(event)


class CodeCell(QWidget):
    """A code editor across the row, its help as a caption above it."""

    def __init__(self, help_text: str, editor: QWidget) -> None:
        super().__init__()
        self.editor = editor
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        if help_text:
            lay.addWidget(WrapLabel(help_text, "caption"))
        lay.addWidget(editor)


class GroupTile(QWidget):
    """Consecutive fields of one group under its eyebrow. The tile spans its visible
    members' columns, and its inner grid uses that many, so the members line up with the
    parent's columns."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self._columns = 1
        self.grid = FieldGrid(self, columns=lambda: self._columns)
        self.grid.add(label(title.upper(), "eyebrow"), span=0, role="heading")
        self.members: list[tuple[QWidget, Span]] = []

    def add(self, cell: QWidget, span: Span) -> None:
        self.members.append((cell, span))
        self.grid.add(cell, span=span, brk=len(self.members) == 1)

    def any_visible(self) -> bool:
        return any(not cell.isHidden() for cell, _span in self.members)

    def span(self, columns: int, column: float) -> int:
        """The visible members' columns, clamped to the parent's (sets the inner count)."""
        total = 0
        for cell, span in self.members:
            if not cell.isHidden():
                total += max(1, span(columns, column) if callable(span) else (span or columns))
        self._columns = max(1, min(columns, total))
        return self._columns


class Heading(QWidget):
    """A block's eyebrow and caption (the eyebrow's tooltip holds ``tip``)."""

    def __init__(self, eyebrow: str, caption: str = "", tip: str = "") -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.eyebrow = label(eyebrow.upper(), "eyebrow")
        self.eyebrow.setToolTip(tip)
        lay.addWidget(self.eyebrow)
        if caption:
            lay.addWidget(WrapLabel(caption, "caption"))
