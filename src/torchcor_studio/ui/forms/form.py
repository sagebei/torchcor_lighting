"""Schema-driven parameter form.

``ParamForm(ModelClass, panels=PANELS)`` renders any :class:`~torchcor_studio.core.simulations.Section`
model as panels (:mod:`torchcor_studio.core.simulations.panels`): each holds a grid of
fields sized by meaning, group tiles and tables, and a collapsed panel is built when it is
first opened. Every edit is validated by pydantic; the form emits :attr:`ParamForm.changed`
with a valid model or :attr:`ParamForm.errors` with ``[(path, message)]``. Values update in
place (:meth:`ParamForm.set_value`), :meth:`ParamForm.mark_issues` tints the editors that
validation issues point at, and :meth:`ParamForm.changes` lists what differs from a baseline.

``ParamSummary`` renders read-only parameters (a finished run's Setup view).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from torchcor_studio.core.models import InputInfo
from torchcor_studio.core.simulations.base import Issue, Severity
from torchcor_studio.core.simulations.fields import field_meta
from torchcor_studio.core.simulations.panels import (
    INPUT,
    SLOTS,
    Block,
    Change,
    Panel,
    ResolvedPanel,
    changes,
    leaf_meta,
    nouns,
    number,
    owner,
    resolve,
    summary,
    value_at,
)
from torchcor_studio.core.simulations.types import FieldType
from torchcor_studio.core.simulations.types import resolve as field_type
from torchcor_studio.ui.forms.editors import BrowseFn, Editor, is_segmented, make_editor
from torchcor_studio.ui.forms.grid import (
    CodeCell,
    FieldCell,
    GroupTile,
    Heading,
    Span,
    grid_widget,
    natural_span,
)
from torchcor_studio.ui.forms.panel import PanelCard, PanelGroup
from torchcor_studio.ui.forms.table import TableEditor
from torchcor_studio.ui.widgets import Card, chain_focus, label

Path_ = tuple[Any, ...]


@dataclass
class _Field:
    path: Path_
    meta: dict
    panel: str
    editor: Editor
    cell: QWidget
    span: Span


@dataclass(frozen=True)
class OutlineEntry:
    """A panel as the Setup step's outline lists it, or (``group``) the heading of a group,
    whose panels follow it at ``depth`` 1."""

    key: str
    title: str
    errors: int
    warnings: int
    edited: bool
    collapsed: bool
    summary: str
    depth: int = 0
    group: bool = False


def _set(data: Any, path: Sequence[Any], value: Any) -> None:
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value


def parse_path(dotted: str) -> Path_:
    """``"stimuli.0.site"`` -> ``("stimuli", 0, "site")``."""
    return tuple(int(p) if p.isdigit() else p for p in dotted.split(".") if p)


def field_span(ftype: FieldType, meta: dict, default: Any, cell: QWidget, editor: QWidget) -> Span:
    """How many grid columns a field takes (0: the whole row), unless ``param(span=)`` says:
    numbers one, switches and segmented controls what their text needs, combo boxes,
    text and lists two, paths, code and long or monospace text the whole row."""
    if meta.get("span") is not None:
        return meta["span"]
    hint = meta.get("widget")
    if hint in ("dir", "file", "path", "code"):
        return 0
    kind = ftype.kind
    if kind in ("float", "int"):
        return 1
    if kind == "bool":
        return natural_span(cell)
    if kind == "choice" or meta.get("choices"):
        return natural_span(editor) if is_segmented(meta, ftype) else 2
    if kind in ("int_list", "float_list", "str_list", "vec"):
        return 2
    return 0 if hint == "mono" or len(str(default or "")) > 40 else 2


class ParamForm(QWidget):
    changed = Signal(object)  # validated model
    errors = Signal(list)  # [(dotted path, message)]
    panels_changed = Signal()  # a panel opened, closed or showed its advanced parameters

    def __init__(self, model_cls: type[BaseModel], value: BaseModel | None = None, *,
                 panels: Sequence[Panel] = (), browse: BrowseFn | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model_cls = model_cls
        self.browse = browse
        self._resolved = resolve(model_cls, panels)
        self._metas = {leaf: leaf_meta(model_cls, leaf) for p in self._resolved for leaf in p.leaves}
        self._info: InputInfo | None = None
        self._model = value or model_cls()
        self._data: dict = self._model.model_dump(mode="python")
        self._fields: dict[Path_, _Field] = {}
        self._tables: dict[Path_, TableEditor] = {}
        self._table_blocks: list[tuple[Path_, list[QWidget], str]] = []
        self._headings: list[tuple[Heading, list[Path_]]] = []
        self._tiles: list[GroupTile] = []
        self._advanced = {p.key: False for p in self._resolved}
        self._auto_advanced: dict[str, bool] = {}
        self._slot_hosts: dict[str, QWidget] = {}  # slot path (INPUT, OUTPUT) → its host
        self._slots: dict[str, QWidget] = {}  # slot path → the widget it holds
        self._input_used = True
        self._baseline: dict | None = None
        self._edited_tip = ""
        self._changes: list[Change] = []
        self._pydantic: dict[Path_, str] = {}
        self._marks: dict[Path_, tuple[str, str]] = {}
        self._counts: dict[str, tuple[int, int]] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self._cards: dict[str, PanelCard] = {}
        self._groups: dict[str, PanelGroup] = {}
        for panel in self._resolved:
            card = PanelCard(panel, self._build_panel, expanded=not panel.collapsed)
            card.toggled.connect(lambda on, k=panel.key: self._toggled(k, on))
            card.advanced_toggled.connect(lambda on, k=panel.key: self._advanced_toggled(k, on))
            self._cards[panel.key] = card
            if panel.group is None:
                lay.addWidget(card)
                continue
            box = self._groups.get(panel.group.key)
            if box is None:
                box = self._groups[panel.group.key] = PanelGroup(panel.group)
                lay.addWidget(box)
            box.add(card)
        lay.addStretch(1)
        self._refresh_panels()

    # ------------------------------------------------------------------ values
    def value(self) -> BaseModel | None:
        """The last valid parameters (``None`` while a field is invalid)."""
        return self._model

    def data(self) -> dict:
        return self._data

    def set_value(self, value: BaseModel) -> None:
        """Show ``value`` in place: built editors and tables are updated, nothing is rebuilt
        unless a table's row count changes."""
        self._data = value.model_dump(mode="python")
        self._model = value
        self._pydantic = {}
        for path, f in self._fields.items():
            f.editor.set_quiet(value_at(self._data, path))
        for table in self._tables.values():
            table.sync()
        self._apply_visibility()
        self._paint_marks()
        self._refresh_panels()

    def input_info(self) -> InputInfo | None:
        return self._info

    def set_input_info(self, info: InputInfo | None) -> None:
        self._info = info
        for f in self._fields.values():
            f.editor.update_context(info)
        for table in self._tables.values():
            table.update_context(info)

    # ------------------------------------------------------------------ panels
    def panels(self) -> tuple[ResolvedPanel, ...]:
        return self._resolved

    def panel(self, key: str) -> PanelCard:
        return self._cards[key]

    def cards(self) -> list[PanelCard]:
        return list(self._cards.values())

    def anchor(self, key: str) -> QWidget:
        """What the outline's ``key`` stands for on the page: a panel, or a group's box."""
        return self._groups.get(key) or self._cards[key]

    def expand_all(self) -> None:
        for card in self._cards.values():
            card.expand()

    def set_show_advanced(self, on: bool) -> None:
        """Show (or hide) the advanced parameters of every panel."""
        for key, card in self._cards.items():
            self._advanced[key] = on
            card.set_show_advanced(on)
        self._apply_visibility()
        self._refresh_panels()
        self.panels_changed.emit()

    def advanced_shown(self, panel: str) -> bool:
        return self._advanced.get(panel, False) or self._auto_advanced.get(panel, False)

    def outline_entries(self) -> list[OutlineEntry]:
        """The panels in page order, each group's heading before its panels."""
        entries: list[OutlineEntry] = []
        for i, p in enumerate(self._resolved):
            if p.group is not None and (i == 0 or self._resolved[i - 1].group != p.group):
                entries.append(OutlineEntry(p.group.key, p.group.title, 0, 0, False, False, "", group=True))
            card = self._cards[p.key]
            entries.append(OutlineEntry(p.key, p.title, *self._counts.get(p.key, (0, 0)), card.edited,
                                        not card.is_expanded(), card.summary, depth=int(p.group is not None)))
        return entries

    def has_slot(self, name: str = INPUT) -> bool:
        """Does a panel hold the ``name`` slot (``INPUT`` or ``OUTPUT``)?"""
        return any(b.kind == SLOTS[name] for p in self._resolved for b in p.blocks)

    def set_slot(self, name: str, widget: QWidget | None) -> None:
        """Put a widget in the ``name`` slot block, ``INPUT`` or ``OUTPUT`` (``None``: take
        the current one out). Issues about that folder go to its ``set_issue(severity,
        message)``, and revealing them focuses its ``focus_target()``, when it has them."""
        old = self._slots.get(name)
        if widget is old:
            return
        host = self._slot_hosts.get(name)
        if old is not None and host is not None:
            host.layout().removeWidget(old)
        if widget is None:
            self._slots.pop(name, None)
        else:
            self._slots[name] = widget
            if host is not None:
                host.layout().addWidget(widget)
                widget.show()
                self._chain_slot(host, widget)
        self._paint_marks()

    def _chain_slot(self, host: QWidget, widget: QWidget) -> None:
        """Tab reaches a slot's widget where it sits in its panel: after the last widget
        before its host (it was made elsewhere, so it joined the focus chain there)."""
        anchor = host.previousInFocusChain()
        for _ in range(400):  # the panel's header, at the latest
            if anchor is host or (anchor.focusPolicy() & Qt.FocusPolicy.TabFocus and self.isAncestorOf(anchor)
                                  and not widget.isAncestorOf(anchor)):
                break
            anchor = anchor.previousInFocusChain()
        else:
            return
        if anchor is not host:
            chain_focus(anchor, widget)

    def set_code_height(self, height: int) -> None:
        """The minimum height of the code editors (the Setup step fits them to the page)."""
        for f in self._fields.values():
            if f.meta.get("widget") == "code":
                f.editor.widget.setMinimumHeight(height)

    def set_input_role(self, role: str, _hint: str = "") -> None:
        """An ``"unused"`` input folder hides its block (the folder itself is kept)."""
        self._input_used = role != "unused"
        if (host := self._slot_hosts.get(INPUT)) is not None:
            host.setVisible(self._input_used)

    def _toggled(self, key: str, expanded: bool) -> None:
        if not expanded:
            self._refresh_panels()  # its summary
        self.panels_changed.emit()

    def _advanced_toggled(self, key: str, on: bool) -> None:
        self._advanced[key] = on
        self._apply_visibility(key)
        self.panels_changed.emit()

    # ------------------------------------------------------------------ building
    def _build_panel(self, card: PanelCard) -> QWidget:
        body, grid = grid_widget()
        grid.setContentsMargins(20, 2, 20, 20)
        for block in card.resolved.blocks:
            if block.kind in ("input", "output"):
                name = next(path for path, kind in SLOTS.items() if kind == block.kind)
                host = self._slot_hosts[name] = QWidget()
                QVBoxLayout(host).setContentsMargins(0, 0, 0, 0)
                if (widget := self._slots.get(name)) is not None:
                    host.layout().addWidget(widget)
                grid.add(host, span=0, role="block")  # parented before it is shown (else a window)
                host.setVisible(name != INPUT or self._input_used)
            elif block.kind == "table":
                self._table_block(grid, block, card.key)
            else:
                self._fields_block(grid, block, card.key)
        self._apply_visibility(card.key)
        self._paint_marks()
        return body

    def _fields_block(self, grid, block: Block, panel: str) -> None:  # noqa: ANN001
        """A block's eyebrow, then its fields; consecutive fields of one group form a tile.
        A choice that later fields depend on ends its row (the law, then its constants), and
        so does a switch that all the later fields of its section depend on (pre-pacing on,
        then its settings below it, where they appear without moving the switch)."""
        if block.eyebrow:
            heading = Heading(block.eyebrow, block.caption, block.tip)
            grid.add(heading, span=0, role="heading")
            self._headings.append((heading, list(block.leaves)))
        tile, tile_key = None, None
        for i, leaf in enumerate(block.leaves):
            meta, ftype = self._metas[leaf]
            group = meta.get("group", "")
            f = self._make_field(leaf, panel, short=bool(group))
            if not group:
                tile = None
                later = [lf for lf in block.leaves[i + 1:] if lf[:-1] == leaf[:-1]]
                gated = [lf for lf in later if leaf[-1] in (self._metas[lf][0].get("visible_if") or {})]
                choice = ftype.kind == "choice" or bool(meta.get("choices"))
                controls = bool(gated) and (choice or (ftype.kind == "bool" and len(gated) == len(later)))
                grid.add(f.cell, span=f.span, brk=i == 0, end=controls)
                continue
            if tile is None or tile_key != (leaf[:-1], group):
                tile, tile_key = GroupTile(group), (leaf[:-1], group)
                self._tiles.append(tile)
                grid.add(tile, span=tile.span, role="tile", brk=i == 0)
            tile.add(f.cell, f.span)

    def _make_field(self, path: Path_, panel: str, *, short: bool) -> _Field:
        meta, ftype = self._metas[path]
        value = value_at(self._data, path)
        editor = make_editor(ftype, meta, value, lambda p=path: self._edited(p), self.browse)
        title = (meta.get("short") or meta["title"]) if short else meta["title"]
        if meta.get("widget") == "code":
            cell: QWidget = CodeCell(meta.get("help", ""), editor.widget)
        else:
            cell = FieldCell(title, meta.get("help", ""), editor.widget, beside=ftype.kind == "bool",
                             fixed=ftype.kind != "bool" and _fixed_width(editor.widget))
        editor.update_context(self._info)
        f = _Field(path, meta, panel, editor, cell, field_span(ftype, meta, value, cell, editor.widget))
        self._fields[path] = f
        return f

    def _table_block(self, grid, block: Block, panel: str) -> None:  # noqa: ANN001
        """A table under its eyebrow. A small table takes half the row, beside another; its
        caption then runs across the row above it, and a fixed table's eyebrow (no caption)
        stays above the table, so two of them pair up."""
        leaf = block.leaves[0]
        meta, ftype = self._metas[leaf]
        table = TableEditor(ftype.inner, leaf, self, panel=panel, fixed=bool(meta.get("fixed")),
                            item=nouns(meta)[0])
        self._tables[leaf] = table
        widget: QWidget = table
        shown = [widget]  # what shows and hides with the table
        if block.caption and block.eyebrow:
            heading = Heading(block.eyebrow, block.caption, block.tip)
            grid.add(heading, span=0, role="heading")
            shown.append(heading)
        elif block.eyebrow:
            widget = QWidget()
            lay = QVBoxLayout(widget)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(10)
            lay.addWidget(Heading(block.eyebrow, tip=block.tip))
            lay.addWidget(table)
            shown = [widget]
        self._table_blocks.append((leaf, shown, panel))
        grid.add(widget, span=0, role="block", half=table.is_half, natural=table.natural_width)

    def _ensure(self, path: Path_) -> ResolvedPanel | None:
        """Build the panel that owns ``path`` (it stays collapsed)."""
        panel = owner(self._resolved, ".".join(str(p) for p in path))
        if panel is not None:
            self._cards[panel.key].body()
        return panel

    def editor(self, path: Path_) -> Editor | None:
        """The editor of a field, or of a table cell (``("stimuli", 0, "site")``), building its
        panel if needed."""
        self._ensure(path)
        if path in self._fields:
            return self._fields[path].editor
        for tpath, table in self._tables.items():
            n = len(tpath)
            if path[:n] == tpath and len(path) == n + 2:
                return table.cell(path[n], path[n + 1])
        return None

    def table(self, path: Path_) -> TableEditor | None:
        self._ensure(path)
        return self._tables.get(tuple(path))

    def reveal(self, dotted: str) -> QWidget | None:
        """Open the panel owning a parameter (and its Advanced when needed) and return the
        widget to scroll to: the field's editor, a table cell, a list's first cell, or the
        input folder (:data:`INPUT`)."""
        panel = owner(self._resolved, dotted)
        if panel is None:
            return None
        card = self._cards[panel.key]
        card.expand()
        if dotted in SLOTS:
            slot = self._slots.get(dotted)
            return getattr(slot, "focus_target", lambda: slot)()
        path = parse_path(dotted)
        leaf = max((lf for lf in panel.leaves if path[:len(lf)] == lf), key=len, default=None)
        if leaf is None:  # a section: its first leaf in the panel
            leaf = next(lf for lf in panel.leaves if lf[:len(path)] == path)
        meta, ftype = self._metas[leaf]
        column = (leaf_meta(self.model_cls, path)[0] if ftype.kind == "rows" and len(path) == len(leaf) + 2
                  else meta)
        if (meta.get("advanced") or column.get("advanced")) and not self.advanced_shown(panel.key):
            self._advanced[panel.key] = True
            card.set_show_advanced(True)
            self._apply_visibility(panel.key)
            self.panels_changed.emit()
        if ftype.kind == "rows":
            table = self._tables[leaf]
            editor = table.cell(path[len(leaf)], path[len(leaf) + 1]) if len(path) == len(leaf) + 2 else None
            return (editor.focus_target() or editor.widget) if editor is not None else table.first_cell()
        f = self._fields[leaf]
        return f.editor.focus_target() or f.cell

    # ------------------------------------------------------------------ editing
    def _edited(self, path: Path_) -> None:
        _set(self._data, path, self._fields[path].editor.get())
        self._apply_visibility()
        self._validate()

    def rows_changed(self) -> None:
        """Called by tables after a row was added, removed or edited."""
        self._apply_visibility()
        self._validate()

    def _validate(self) -> None:
        try:
            model = self.model_cls.model_validate(self._data)
        except ValidationError as exc:
            self._pydantic = {}
            problems = []
            for err in exc.errors():
                loc = tuple(err["loc"])
                msg = err["msg"].removeprefix("Value error, ")
                self._pydantic.setdefault(loc, msg)
                problems.append((".".join(str(x) for x in loc), msg))
            self._model = None
            self._paint_marks()
            self._refresh_panels()
            self.errors.emit(problems)
            return
        self._pydantic = {}
        self._model = model
        self._paint_marks()
        self._refresh_panels()
        self.errors.emit([])
        self.changed.emit(model)

    def error_issues(self) -> list[Issue]:
        """The current pydantic errors as issues that point at their fields."""
        out = []
        for loc, msg in self._pydantic.items():
            dotted = ".".join(map(str, loc))
            try:
                title = leaf_meta(self.model_cls, loc)[0]["title"]
            except (KeyError, AttributeError):  # not a field of the model
                title = dotted
            out.append(Issue(Severity.ERROR, f"{title}: {msg}", dotted))
        return out

    # ------------------------------------------------------------------ baseline & issues
    def set_baseline(self, value: BaseModel, origin: str = "") -> None:
        """What changes are counted against; ``origin`` names it in tooltips."""
        self._baseline = value.model_dump(mode="python")
        self._edited_tip = f"Changed since {origin}" if origin else "Changed"
        self._refresh_panels()

    def changes(self) -> list[Change]:
        return self._changes

    def mark_issues(self, issues: Sequence[Issue]) -> None:
        """Tint the editors the issues point at (the input folder for :data:`INPUT`) and
        count them on their panels."""
        marks: dict[Path_, tuple[str, str]] = {}
        counts = {p.key: [0, 0] for p in self._resolved}
        for issue in sorted(issues, key=lambda i: i.severity != Severity.ERROR):
            if issue.severity == Severity.INFO or not issue.field:
                continue
            panel = owner(self._resolved, issue.field)
            if panel is not None:
                counts[panel.key][issue.severity != Severity.ERROR] += 1
            key = (INPUT,) if issue.field == INPUT else parse_path(issue.field)
            marks.setdefault(key, (str(issue.severity), issue.message))
        self._marks = marks
        self._counts = {k: (e, w) for k, (e, w) in counts.items()}
        for key, card in self._cards.items():
            card.set_issues(*self._counts[key])
        self._paint_marks()

    def issue_counts(self) -> dict[str, tuple[int, int]]:
        return dict(self._counts)

    def _paint_marks(self) -> None:
        wanted = {loc: ("error", msg) for loc, msg in self._pydantic.items()}
        for path, mark in self._marks.items():
            wanted.setdefault(path, mark)
        for path, f in self._fields.items():
            f.editor.set_issue(*wanted.get(path, (None, "")))
        for tpath, table in self._tables.items():
            for (i, name), editor in table.editors():
                editor.set_issue(*wanted.get((*tpath, i, name), (None, "")))
        for name, slot in self._slots.items():
            if hasattr(slot, "set_issue"):
                slot.set_issue(*wanted.get((name,), (None, "")))

    def _refresh_panels(self) -> None:
        """Changes and edited dots, summaries of collapsed panels and Advanced counts."""
        self._changes = (changes(self.model_cls, self._resolved, self._baseline, self._data)
                         if self._baseline is not None else [])
        edited = {c.panel for c in self._changes}
        for panel in self._resolved:
            card = self._cards[panel.key]
            card.set_edited(panel.key in edited, self._edited_tip)
            if not card.is_expanded():
                card.set_summary(summary(self.model_cls, panel, self._data))
            advanced, plain = self._advanced_census(panel)
            folders = any(b.kind in SLOTS.values() for b in panel.blocks)  # plain content too
            self._auto_advanced[panel.key] = bool(advanced) and not plain and not folders
            card.set_advanced_count(0 if self._auto_advanced[panel.key] else advanced)

    def _advanced_census(self, panel: ResolvedPanel) -> tuple[int, int]:
        """(advanced, other) parameters of a panel whose ``visible_if`` holds now; a table
        counts its columns."""
        advanced = plain = 0
        for leaf in panel.leaves:
            meta, ftype = self._metas[leaf]
            if not self.condition(leaf[:-1], meta):
                continue
            if ftype.kind != "rows":
                advanced += bool(meta.get("advanced"))
                plain += not meta.get("advanced")
                continue
            rows = value_at(self._data, leaf)
            for name in ftype.inner.model_fields:
                column = field_meta(ftype.inner, name)
                shown = (any(self.condition((*leaf, i), column) for i in range(len(rows))) if rows
                         else self.condition((*leaf, 0), column, rowless=True))
                if shown:
                    advanced += bool(column.get("advanced"))
                    plain += not column.get("advanced")
        return advanced, plain

    # ------------------------------------------------------------------ visibility
    def condition(self, scope: Path_, meta: dict, rowless: bool = False) -> bool:
        """Does the field's ``visible_if`` hold? ``"../"`` climbs one section (from a table
        row, to the section owning the table) and the rest may go down with dots.
        ``rowless``: a table with no rows, whose row-level conditions cannot be read."""
        cond = meta.get("visible_if")
        if not cond:
            return True
        for key, expected in cond.items():
            if rowless and not key.startswith("../"):
                continue
            base = list(scope)
            while key.startswith("../"):
                key = key[3:]
                base = base[:-2] if base and isinstance(base[-1], int) else base[:-1]
            try:
                actual = value_at(self._data, (*base, *key.split(".")))
            except (KeyError, IndexError, TypeError):
                return False
            actual = actual.value if hasattr(actual, "value") else actual
            allowed = expected if isinstance(expected, list | tuple | set) else [expected]
            if actual not in allowed:
                return False
        return True

    def visible(self, scope: Path_, meta: dict, panel: str, rowless: bool = False) -> bool:
        """Is a field shown: its condition holds, and it is not advanced or its panel shows
        advanced parameters."""
        if meta.get("advanced") and not self.advanced_shown(panel):
            return False
        return self.condition(scope, meta, rowless)

    def _apply_visibility(self, panel: str | None = None) -> None:
        for path, f in self._fields.items():
            if panel is None or f.panel == panel:
                f.cell.setVisible(self.visible(path[:-1], f.meta, f.panel))
        for tile in self._tiles:
            tile.setVisible(tile.any_visible())
        for heading, leaves in self._headings:
            heading.setVisible(any(not self._fields[leaf].cell.isHidden() for leaf in leaves))
        for leaf, widgets, key in self._table_blocks:
            if panel is None or key == panel:
                visible = self.visible(leaf[:-1], self._metas[leaf][0], key)
                for widget in widgets:
                    widget.setVisible(visible)
                self._tables[leaf].apply_visibility()


def _fixed_width(widget: QWidget) -> bool:
    """Controls with a natural width (segmented, switches) keep it instead of stretching."""
    return widget.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed


_LABEL_WIDTH = 196


class ParamSummary(QWidget):
    """Read-only rendering of a parameter dict (for a finished run's "Parameters" tab)."""

    def __init__(self, model_cls: type[BaseModel] | None, data: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        if model_cls is None:
            card = Card("Parameters")
            card.addWidget(_kv_grid([(k, _fmt(v, {})) for k, v in data.items()]))
            lay.addWidget(card)
        else:
            general: list[tuple[str, str]] = []
            for name, info in model_cls.model_fields.items():
                ftype = field_type(info.annotation)
                meta = field_meta(model_cls, name)
                value = data.get(name)
                if ftype.kind == "section" and isinstance(value, dict):
                    card = Card(meta["title"])
                    card.addWidget(_kv_grid(_flatten(ftype.inner, value)))
                    lay.addWidget(card)
                elif ftype.kind == "rows" and isinstance(value, list):
                    card = Card(meta["title"], f"{len(value)} entr{'y' if len(value) == 1 else 'ies'}")
                    for i, item in enumerate(value):
                        card.addWidget(label(f"#{i + 1}", "eyebrow"))
                        card.addWidget(_kv_grid(_flatten(ftype.inner, item)))
                    lay.addWidget(card)
                else:
                    general.append((meta["title"], _fmt(value, meta)))
            if general:
                card = Card("General")
                card.addWidget(_kv_grid(general))
                lay.insertWidget(0, card)
        lay.addStretch(1)


def _flatten(model_cls: type[BaseModel], data: dict, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for name, info in model_cls.model_fields.items():
        ftype = field_type(info.annotation)
        meta = field_meta(model_cls, name)
        value = data.get(name)
        title = f"{prefix}{meta['title']}"
        if ftype.kind == "section" and isinstance(value, dict):
            rows.extend(_flatten(ftype.inner, value, f"{title} › "))
        elif ftype.kind == "rows" and isinstance(value, list):
            for i, item in enumerate(value):
                rows.extend(_flatten(ftype.inner, item, f"{title} #{i + 1} › "))
        else:
            rows.append((title, _fmt(value, meta)))
    return rows


def _fmt(value: Any, meta: dict) -> str:
    unit = meta.get("unit", "")
    choices = meta.get("choices") or {}
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, list | tuple):
        return ", ".join(_fmt(v, {}) for v in value) or ("All" if meta.get("widget") == "regions" else "—")
    if isinstance(value, float):
        text = number(value)
    else:
        value = value.value if hasattr(value, "value") else value
        text = str(choices.get(value, value))
    return f"{text} {unit}".strip()


def _kv_grid(rows: list[tuple[str, str]]) -> QWidget:
    w = QWidget()
    grid = QGridLayout(w)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(8)
    grid.setColumnMinimumWidth(0, _LABEL_WIDTH)
    grid.setColumnStretch(1, 1)
    for r, (k, v) in enumerate(rows):
        key = label(k, "muted")
        key.setWordWrap(True)
        val = QLabel(v)
        val.setWordWrap(True)
        val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        grid.addWidget(key, r, 0, Qt.AlignmentFlag.AlignTop)
        grid.addWidget(val, r, 1, Qt.AlignmentFlag.AlignTop)
    return w
