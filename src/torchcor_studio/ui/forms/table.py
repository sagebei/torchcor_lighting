"""Lists of sections as real tables: column titles with their units, one row of cell
editors per item, and column-group headers when two or more groups show.

One grid holds everything (no widget per row): row 0 the column groups, row 1 the column
titles (a title and, beneath it, its unit: a unit is never dropped), row 2 a hairline, then
the cells, with each row's remove button in the last column. Columns take the width their
content needs, up to a cap per type, and leave the rest of the row empty, so a short number
never sits in a wide box; when the room is short every column gives some of it up, down to
a floor, and the cells end their text with "…". A ``fixed`` table (CircAdapt's tubes,
valves, ...) has no add or remove controls, and its first text column becomes a read-only
row header.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel
from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.core.models import InputInfo
from torchcor_studio.core.simulations.fields import field_meta
from torchcor_studio.core.simulations.panels import value_at
from torchcor_studio.core.simulations.types import FieldType, resolve
from torchcor_studio.ui.forms.editors import CELL_H, Editor, make_editor
from torchcor_studio.ui.forms.grid import draw_rich, rich_width
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import Button, ElidedLabel, IconButton, discard, hairline

if TYPE_CHECKING:
    from torchcor_studio.ui.forms.form import ParamForm

GROUP_ROW, HEAD_ROW, RULE_ROW, FIRST_ROW = 0, 1, 2, 3
GAP = 6  # between columns and between rows
REMOVE = 24  # the remove buttons' column


@dataclass
class _Column:
    name: str
    ftype: FieldType
    meta: dict
    index: int

    @property
    def regions(self) -> bool:
        return self.meta.get("widget") == "regions"

    @property
    def numeric(self) -> bool:
        return self.ftype.kind in ("float", "int")

    @property
    def choice(self) -> bool:
        return self.ftype.kind == "choice" or bool(self.meta.get("choices")) or self.meta.get("widget") == "vtx"

    def limits(self) -> tuple[int, int, int]:
        """(floor, cap, weight): the narrowest it gets, the widest it grows with room to
        spare (a choice: at least its longest option), and its share of that room."""
        if self.regions:
            return 110, 260, 2
        if self.choice:
            return 90, 220, 3
        if self.ftype.kind == "int":
            return 48, 80, 1
        if self.ftype.kind == "float":
            return 64, 140, 2
        return 80, 260, 3


@dataclass
class _Row:
    editors: dict[str, Editor] = field(default_factory=dict)
    cells: dict[str, QWidget] = field(default_factory=dict)  # what sits in each column
    remove: QWidget | None = None


def _font(widget: QWidget, weight: QFont.Weight) -> QFont:
    font = QFont(widget.font())
    font.setPixelSize(11)
    font.setWeight(weight)
    return font


class ColumnHead(QWidget):
    """A column title (11 px semibold, text2) with its unit (11 px, text3) on a second line
    when the table shows units; a narrow column shortens the title, never the unit."""

    def __init__(self, title: str, unit: str, *, right: bool, tip: str = "") -> None:
        super().__init__()
        self.title, self.unit, self._right = title, unit, right
        self._bold = _font(self, QFont.Weight.DemiBold)
        self._plain = _font(self, QFont.Weight.Normal)
        self._lines = 1
        self._natural = int(max(rich_width(title, self._bold),
                                QFontMetrics(self._plain).horizontalAdvance(unit))) + 10
        self.setToolTip(tip)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.set_lines(1)

    def set_lines(self, lines: int) -> None:
        """One line (title only) or two (title, then unit)."""
        self._lines = lines
        self.setFixedHeight(QFontMetrics(self._bold).height() * lines + 2)

    def lines(self) -> int:
        return self._lines

    def natural_width(self) -> int:
        """The width that shows the title and the unit whole."""
        return self._natural

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.natural_width(), self.height())

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        line = QFontMetrics(self._bold).height()
        room = self.width() - 10
        x0 = 9
        title_w = min(rich_width(self.title, self._bold), room)
        x = self.width() - 10 - title_w if self._right else x0
        draw_rich(painter, QRectF(x, 0, room, line), self.title, self._bold, p.qcolor("text2"))
        if self.unit and self._lines == 2:
            painter.setFont(self._plain)
            painter.setPen(p.qcolor("text3"))
            unit = QFontMetrics(self._plain).elidedText(self.unit, Qt.TextElideMode.ElideRight, room)
            align = Qt.AlignmentFlag.AlignRight if self._right else Qt.AlignmentFlag.AlignLeft
            painter.drawText(QRectF(x0, line, self.width() - 10 - x0 + 1, line),
                             align | Qt.AlignmentFlag.AlignVCenter, unit)


class GroupHead(QWidget):
    """A column group's name in capitals over a hairline."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title.upper()
        self._font = _font(self, QFont.Weight.DemiBold)
        self._font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(QFontMetrics(self._font).height() + 6)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setFont(self._font)
        painter.setPen(p.qcolor("text2"))
        fm = QFontMetrics(self._font)
        painter.drawText(QRectF(9, 0, self.width() - 9, fm.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         fm.elidedText(self.title, Qt.TextElideMode.ElideRight, self.width() - 9))
        painter.fillRect(QRectF(0, self.height() - 1, self.width(), 1), p.qcolor("line"))


class _Grid(QGridLayout):
    """The table's grid: before laying out, it sets each column's width for the room it
    gets (``TableEditor.widths``)."""

    def __init__(self, table: TableEditor) -> None:
        super().__init__()
        self._table = table

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        for index, width in self._table.widths(rect.width()).items():
            if self.columnMinimumWidth(index) != width:
                self.setColumnMinimumWidth(index, width)
        super().setGeometry(rect)


class TableEditor(QWidget):
    """Editable rows of a ``list[Section]`` field (stimuli, conductivities, tubes, ...)."""

    def __init__(self, model_cls: type[BaseModel], path: tuple, form: ParamForm, *, panel: str,
                 fixed: bool = False, item: str = "row") -> None:
        super().__init__()
        self.model_cls, self.path, self.form, self.panel = model_cls, path, form, panel
        self.fixed = fixed
        self._columns = [_Column(name, resolve(info.annotation), field_meta(model_cls, name), i)
                         for i, (name, info) in enumerate(model_cls.model_fields.items())]
        self._rowhead = next((c.name for c in self._columns if fixed and c.ftype.kind == "str"
                              and not c.meta.get("choices") and not c.meta.get("widget")), None)
        self._rows: list[_Row] = []
        self._shown: set[str] = set()
        self._prefs: dict[str, int] | None = None  # each column's own width, from its content
        self._longest: dict[str, int] = {}  # a choice column's width for its longest option
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._grid = _Grid(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(GAP)
        self._grid.setVerticalSpacing(GAP)
        lay.addLayout(self._grid)
        n = len(self._columns)
        self._remove_col, self._rest_col = n, n + 1
        self._grid.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum),
                           HEAD_ROW, self._rest_col)
        self._grid.setColumnStretch(self._rest_col, 1)  # the row's room to spare
        self._heads: dict[str, ColumnHead] = {}
        for col in self._columns:
            head = ColumnHead(col.meta.get("short") or col.meta["title"], col.meta.get("unit", ""),
                              right=col.numeric, tip=self._tip(col))
            self._grid.addWidget(head, HEAD_ROW, col.index)
            self._heads[col.name] = head
        self._groups: list[tuple[str, GroupHead]] = []
        start = 0
        while start < n:
            group = self._columns[start].meta.get("group", "")
            end = start
            while end + 1 < n and self._columns[end + 1].meta.get("group", "") == group:
                end += 1
            if group:
                head = GroupHead(group)
                self._grid.addWidget(head, GROUP_ROW, start, 1, end - start + 1)
                self._groups.append((group, head))
            start = end + 1
        self._grid.addWidget(hairline(), RULE_ROW, 0, 1, n + 1)
        if not fixed:
            add = Button(f"Add {item}", "plus", "ghost", size="sm", on_click=self.add_row)
            add.setProperty("inline", True)  # its plus sign lines up with the cells
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(add)
            row.addStretch(1)
            lay.addLayout(row)
        self._build_rows()

    @staticmethod
    def _tip(col: _Column) -> str:
        title = col.meta["title"] + (f" ({col.meta['unit']})" if col.meta.get("unit") else "")
        return title + (f": {col.meta['help']}" if col.meta.get("help") else "")

    # ------------------------------------------------------------------ data
    def items(self) -> list[dict]:
        return value_at(self.form.data(), self.path)

    def row_count(self) -> int:
        return len(self._rows)

    def cell(self, index: int, name: str) -> Editor | None:
        """The editor of one cell (``None`` for a row header or a row that does not exist)."""
        return self._rows[index].editors.get(name) if 0 <= index < len(self._rows) else None

    def header(self, name: str) -> ColumnHead:
        return self._heads[name]

    def editors(self) -> Iterator[tuple[tuple[int, str], Editor]]:
        for i, row in enumerate(self._rows):
            for name, editor in row.editors.items():
                yield (i, name), editor

    def first_cell(self) -> QWidget | None:
        """The first visible cell (where a list-wide issue points)."""
        for row in self._rows:
            for col in self._columns:
                widget = row.cells.get(col.name)
                if widget is not None and not widget.isHidden():
                    return widget
        return None

    # ------------------------------------------------------------------ rows
    def _build_rows(self) -> None:
        for row in self._rows:
            for widget in [*row.cells.values(), row.remove]:
                if widget is not None:
                    self._grid.removeWidget(widget)
                    discard(widget)
        self._rows = []
        for i, values in enumerate(self.items()):
            row = _Row()
            r = FIRST_ROW + i
            for col in self._columns:
                value = values.get(col.name)
                if col.name == self._rowhead:
                    head = ElidedLabel("" if value is None else str(value), "rowhead")
                    head.setFixedHeight(CELL_H)
                    head.setContentsMargins(9, 0, 0, 0)
                    row.cells[col.name] = head
                    self._grid.addWidget(head, r, col.index)
                    continue
                editor = make_editor(col.ftype, col.meta, value,
                                     lambda i=i, n=col.name: self._edited(i, n), self.form.browse,
                                     cell=True)
                widget = editor.widget
                if col.ftype.kind != "bool":  # the column decides the width
                    widget.setMinimumWidth(0)
                    widget.setSizePolicy(QSizePolicy.Policy.Ignored, widget.sizePolicy().verticalPolicy())
                row.editors[col.name] = editor
                row.cells[col.name] = widget
                align = Qt.AlignmentFlag.AlignCenter if col.ftype.kind == "bool" else Qt.AlignmentFlag(0)
                self._grid.addWidget(widget, r, col.index, align)
            if not self.fixed:
                row.remove = IconButton("x", "Remove", size=24, icon_size=12,
                                        on_click=lambda i=i: self.remove_row(i))
                self._grid.addWidget(row.remove, r, self._remove_col, Qt.AlignmentFlag.AlignCenter)
            self._rows.append(row)
        self.update_context(self.form.input_info())
        self.apply_visibility()
        self._resized()

    def sync(self) -> None:
        """Show the form's current items: in place when the row count is unchanged."""
        items = self.items()
        if len(items) != len(self._rows):
            self._build_rows()
            return
        for row, values in zip(self._rows, items, strict=True):
            for name, editor in row.editors.items():
                editor.set_quiet(values.get(name))
            if self._rowhead is not None:
                row.cells[self._rowhead].setText(str(values.get(self._rowhead, "")))
        self.apply_visibility()
        self._resized()

    def _edited(self, index: int, name: str) -> None:
        self.items()[index][name] = self._rows[index].editors[name].get()
        self.form.rows_changed()

    def add_row(self) -> None:
        """A new row, copied from the last one (or the defaults)."""
        items = self.items()
        items.append(self.model_cls.model_validate(items[-1]).model_dump(mode="python") if items
                     else self.model_cls().model_dump(mode="python"))
        self._build_rows()
        self.form.rows_changed()

    def remove_row(self, index: int) -> None:
        del self.items()[index]
        self._build_rows()
        self.form.rows_changed()

    # ------------------------------------------------------------------ state
    def update_context(self, info: InputInfo | None) -> None:
        for _key, editor in self.editors():
            editor.update_context(info)
        self._resized()

    def apply_visibility(self) -> None:
        """Cells follow ``visible_if`` and Advanced row by row; a column shown in no row goes
        with its header, and group headers show only when two or more groups do."""
        shown: set[str] = set()
        for i, row in enumerate(self._rows):
            for col in self._columns:
                visible = self.form.visible((*self.path, i), col.meta, self.panel)
                row.cells[col.name].setVisible(visible)
                if visible:
                    shown.add(col.name)
        if not self._rows:  # an empty table still shows its columns
            shown = {c.name for c in self._columns
                     if self.form.visible((*self.path, 0), c.meta, self.panel, rowless=True)}
        if shown != self._shown:
            self._shown = shown
            self._prefs = None
            lines = 2 if any(c.meta.get("unit") for c in self._columns if c.name in shown) else 1
            for col in self._columns:
                head = self._heads[col.name]
                head.setVisible(col.name in shown)
                head.set_lines(lines)
            groups = {c.meta.get("group") for c in self._columns if c.name in shown} - {"", None}
            for group, head in self._groups:
                head.setVisible(len(groups) >= 2 and group in groups)
            self._resized()

    def is_half(self) -> bool:
        """Three or fewer columns and no regions: half the width is enough."""
        return len(self._shown) <= 3 and not any(c.regions for c in self._columns if c.name in self._shown)

    # ------------------------------------------------------------------ column widths
    def _resized(self) -> None:
        """The content changed: the columns' own widths are measured again."""
        self._prefs = None
        self._grid.invalidate()

    def _visible_columns(self) -> list[_Column]:
        return [c for c in self._columns if c.name in self._shown]

    def _measure(self) -> dict[str, int]:
        """Each visible column's own width: its title and unit and its widest value, at
        least the column type's floor and, but for a choice, at most its cap. A choice
        column may grow to its longest option (``_longest``) when there is room."""
        if self._prefs is not None and self._prefs.keys() == self._shown:
            return self._prefs
        prefs: dict[str, int] = {}
        for col in self._visible_columns():
            floor, cap, _weight = col.limits()
            want = longest = self._heads[col.name].natural_width()
            for row in self._rows:
                widget = row.cells[col.name]
                fm = QFontMetrics(widget.font())
                editor = row.editors.get(col.name)
                if editor is None:  # a row header
                    want = max(want, fm.horizontalAdvance(widget.fullText()) + 18)
                elif col.ftype.kind == "bool":
                    continue
                elif col.choice:  # the option shown, and the longest one
                    want = max(want, fm.horizontalAdvance(widget.currentText()) + 52)
                    longest = max(longest, max(map(fm.horizontalAdvance, editor.labels() or [""])) + 52)
                else:
                    target = editor.focus_target()
                    text = target.text() if isinstance(target, QLineEdit | QAbstractSpinBox | QComboBox) else ""
                    if not text and isinstance(target, QLineEdit):
                        text = target.placeholderText()  # "All regions" reads as a value
                    want = max(want, fm.horizontalAdvance(text) + (64 if col.regions else 28))
            prefs[col.name] = (self._heads[col.name].natural_width() if col.ftype.kind == "bool"
                               else max(floor, want if col.choice else min(cap, want)))
            self._longest[col.name] = max(longest, prefs[col.name])
        self._prefs = prefs
        return prefs

    def natural_width(self) -> int:
        """The width at which every column shows its content whole."""
        columns = self._visible_columns()
        prefs = self._measure()
        extra = 0 if self.fixed else REMOVE + GAP
        return sum(prefs.values()) + GAP * max(0, len(columns) - 1) + extra

    def widths(self, width: int) -> dict[int, int]:
        """The width of every grid column for a table ``width`` px wide: room to spare goes
        to the columns by weight up to their caps, then stays empty at the end of the row;
        short room is taken from every column in proportion to what it has above its
        floor, and below the floors in proportion to them (the cells then elide)."""
        columns = self._visible_columns()
        prefs = self._measure()
        out = {c.index: 0 for c in self._columns}
        out[self._remove_col] = 0 if self.fixed else REMOVE
        fixed = [c for c in columns if c.ftype.kind == "bool"]
        flex = [c for c in columns if c.ftype.kind != "bool"]
        room = width - GAP * max(0, len(columns) - 1) - (0 if self.fixed else REMOVE + GAP)
        room -= sum(prefs[c.name] for c in fixed)
        for c in fixed:
            out[c.index] = prefs[c.name]
        if not flex:
            return out
        floors = {c.name: c.limits()[0] for c in flex}
        want = {c.name: prefs[c.name] for c in flex}
        low, high = sum(floors.values()), sum(want.values())
        if room <= low:
            share = {n: floors[n] * max(0, room) / low for n in floors}
        elif room <= high:
            t = (room - low) / max(1, high - low)
            share = {n: floors[n] + (want[n] - floors[n]) * t for n in floors}
        else:
            share = dict(want)
            spare = room - high
            caps = {c.name: max(c.limits()[1], self._longest.get(c.name, 0) if c.choice else 0, want[c.name])
                    for c in flex}
            open_ = [c for c in flex if share[c.name] < caps[c.name]]
            while spare > 0.5 and open_:  # by weight, up to the caps
                total = sum(c.limits()[2] for c in open_)
                step = spare
                for c in open_:
                    grow = min(step * c.limits()[2] / total, caps[c.name] - share[c.name])
                    share[c.name] += grow
                    spare -= grow
                open_ = [c for c in open_ if share[c.name] < caps[c.name] - 0.5]
        for c in flex:
            out[c.index] = int(share[c.name])
        return out
