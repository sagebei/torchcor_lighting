"""Left sidebar: "New" (a menu of the kinds of simulation), projects, searchable run
history, navigation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from PySide6.QtCore import (
    QAbstractListModel,
    QMimeData,
    QModelIndex,
    QPoint,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLineEdit,
    QListView,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.core.models import Project, RunRecord, RunState
from torchcor_studio.core.simulations import specs
from torchcor_studio.ui import icons, kinds
from torchcor_studio.ui.format import duration, relative_time
from torchcor_studio.ui.new_menu import NewMenu, NewMenuButton
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import Button, IconButton, SearchField, label, tone_color
from torchcor_studio.ui.widgets.titlebar import TitleBarSpace

#: Drag-and-drop payload: the keys of the dragged runs, one per line.
RUN_KEYS_MIME = "application/x-torchcor-run-keys"


@dataclass(frozen=True)
class DraftEntry:
    """A simulation being set up in a tab, listed where it will be filed."""

    id: str
    name: str  # "" until named
    kind: str = "ep"
    project_id: str | None = None

    @property
    def key(self) -> str:
        return f"draft:{self.id}"

    @property
    def title(self) -> str:
        return self.name or "New simulation"


@dataclass
class _Row:
    header: str | None = None
    record: RunRecord | None = None
    draft: DraftEntry | None = None
    project: Project | None = None
    depth: int = 0  # 1: a run shown inside its project
    count: int = 0  # project rows: runs in the project
    running: int = 0  # project rows: of which active
    expanded: bool = False


class RunListModel(QAbstractListModel):
    RecordRole = Qt.ItemDataRole.UserRole + 1
    HeaderRole = Qt.ItemDataRole.UserRole + 2
    ProjectRole = Qt.ItemDataRole.UserRole + 3  # the _Row of a project
    DepthRole = Qt.ItemDataRole.UserRole + 4
    DraftRole = Qt.ItemDataRole.UserRole + 5

    #: Runs (keys) dropped on a project (id).
    runs_dropped = Signal(list, str)
    #: A run (key) or draft (``draft:<id>``) renamed in place.
    rename_requested = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[_Row] = []
        self._machines: dict[str, str] = {}

    def set_records(self, records: list[RunRecord], query: str, machines: dict[str, str],
                    projects: list[Project] | tuple = (), expanded: set[str] | frozenset = frozenset(),
                    drafts: list[DraftEntry] | tuple = ()) -> None:
        self.beginResetModel()
        self._machines = machines
        self._rows = _rows(records, query, machines, list(projects), expanded, list(drafts))
        self.endResetModel()

    def machine_name(self, machine_id: str) -> str:
        return self._machines.get(machine_id, machine_id)

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        if role == self.RecordRole:
            return row.record
        if role == self.HeaderRole:
            return row.header
        if role == self.ProjectRole:
            return row if row.project is not None else None
        if role == self.DepthRole:
            return row.depth
        if role == self.DraftRole:
            return row.draft
        if role == Qt.ItemDataRole.EditRole:
            return row.draft.name if row.draft else row.record.name if row.record else None
        if role == Qt.ItemDataRole.DisplayRole:
            if row.draft is not None:
                return row.draft.title
            return row.header or (row.project.name if row.project else row.record.name if row.record else "")
        if role == Qt.ItemDataRole.ToolTipRole:
            if row.record:
                r = row.record
                return f"{r.name}\n{self.machine_name(r.machine_id)} · {r.run_dir}"
            if row.project:
                return f"{row.project.name}: {row.count} run{'s' if row.count != 1 else ''}. Drag runs here to file them."
            if row.draft:
                return f"{row.draft.title}: being set up in its tab (not started yet)"
        return None

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        row = self._rows[index.row()]
        if row.header is not None:
            return Qt.ItemFlag.ItemIsEnabled
        if row.project is not None:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDropEnabled
        # runs and drafts: renamed in place, dragged onto a project to file them there
        return (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
                | Qt.ItemFlag.ItemIsDragEnabled)

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802, ANN001
        """A name typed in place: the owner renames the run / draft (the list then refreshes)."""
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        row = self._rows[index.row()]
        name = str(value).strip()
        key = row.draft.key if row.draft else row.record.key if row.record else None
        if key is None or not name or name == self.data(index, Qt.ItemDataRole.EditRole):
            return False
        self.rename_requested.emit(key, name)
        return True

    # ------------------------------------------------------------------ drag and drop
    def mimeTypes(self) -> list[str]:  # noqa: N802
        return [RUN_KEYS_MIME]

    def mimeData(self, indexes) -> QMimeData:  # noqa: N802, ANN001
        """The keys of the dragged runs and drafts (``draft:<id>``)."""
        rows = [self._rows[i.row()] for i in indexes if i.isValid()]
        keys = [row.record.key if row.record else row.draft.key for row in rows if row.record or row.draft]
        data = QMimeData()
        data.setData(RUN_KEYS_MIME, "\n".join(keys).encode())
        return data

    def supportedDragActions(self) -> Qt.DropAction:  # noqa: N802
        return Qt.DropAction.MoveAction

    def supportedDropActions(self) -> Qt.DropAction:  # noqa: N802
        return Qt.DropAction.MoveAction

    def canDropMimeData(self, data, action, row, column, parent) -> bool:  # noqa: N802, ANN001
        return (data.hasFormat(RUN_KEYS_MIME) and parent.isValid()
                and self._rows[parent.row()].project is not None)

    def dropMimeData(self, data, action, row, column, parent) -> bool:  # noqa: N802, ANN001
        if not self.canDropMimeData(data, action, row, column, parent):
            return False
        keys = [k for k in bytes(data.data(RUN_KEYS_MIME)).decode().splitlines() if k]
        if keys:
            self.runs_dropped.emit(keys, self._rows[parent.row()].project.id)
        return False  # the context refiles the runs; nothing to move in this model

    # ------------------------------------------------------------------ lookups
    def index_of(self, key: str) -> QModelIndex:
        """The row of a run (its key) or of a draft (``draft:<id>``)."""
        for i, row in enumerate(self._rows):
            if (row.record is not None and row.record.key == key) or (row.draft is not None and row.draft.key == key):
                return self.index(i)
        return QModelIndex()

    def index_of_project(self, project_id: str) -> QModelIndex:
        for i, row in enumerate(self._rows):
            if row.project is not None and row.project.id == project_id:
                return self.index(i)
        return QModelIndex()

    def update_record(self, record: RunRecord) -> None:
        for i, row in enumerate(self._rows):
            if row.record is not None and row.record.key == record.key:
                row.record = record
                idx = self.index(i)
                self.dataChanged.emit(idx, idx)
                return

    def has_active(self) -> bool:
        return any((r.record is not None and r.record.status.state.is_active) or r.running for r in self._rows)


def _matches(record: RunRecord, q: str, machines: dict[str, str]) -> bool:
    return (q in record.name.lower() or q in record.kind.lower() or q in record.variant.lower()
            or q in machines.get(record.machine_id, "").lower() or q in str(record.status.state))


def _rows(records: list[RunRecord], query: str, machines: dict[str, str], projects: list[Project],
          expanded: set[str] | frozenset, drafts: list[DraftEntry] | None = None) -> list[_Row]:
    """Projects (folders with their drafts and runs when expanded), then drafts not in a
    project, then the other runs by date.

    A search shows the projects whose name or runs match, opened on the matching runs.
    """
    q = query.strip().lower()
    known = {p.id for p in projects}
    members: dict[str, list[RunRecord]] = {p.id: [] for p in projects}
    loose: list[RunRecord] = []
    for r in records:
        (members[r.project_id] if r.project_id in known else loose).append(r)
    filed: dict[str, list[DraftEntry]] = {p.id: [] for p in projects}
    loose_drafts: list[DraftEntry] = []
    for d in drafts or []:
        if q and q not in d.title.lower():
            continue
        (filed[d.project_id] if d.project_id in known else loose_drafts).append(d)
    rows: list[_Row] = []
    folders: list[_Row] = []
    for project in projects:
        runs = sorted(members[project.id], key=lambda r: r.created_at or 0.0, reverse=True)
        shown = runs if not q or q in project.name.lower() else [r for r in runs if _matches(r, q, machines)]
        in_setup = filed[project.id]
        if q and not shown and not in_setup and q not in project.name.lower():
            continue
        is_open = project.id in expanded or bool(q and (shown or in_setup))
        folders.append(_Row(project=project, count=len(runs) + len(in_setup), expanded=is_open,
                            running=sum(r.status.state.is_active for r in runs)))
        if is_open:
            folders.extend(_Row(draft=d, depth=1) for d in in_setup)
            folders.extend(_Row(record=r, depth=1) for r in shown)
    if folders:
        rows.append(_Row(header="Projects"))
        rows.extend(folders)
    if loose_drafts:
        rows.append(_Row(header="Drafts"))
        rows.extend(_Row(draft=d) for d in loose_drafts)
    if q:
        loose = [r for r in loose if _matches(r, q, machines)]
    for title, group in _group(loose):
        rows.append(_Row(header=title))
        rows.extend(_Row(record=r) for r in group)
    return rows


def _group(records: list[RunRecord]) -> list[tuple[str, list[RunRecord]]]:
    active = [r for r in records if r.status.state.is_active]
    rest = [r for r in records if not r.status.state.is_active]
    today = datetime.now().date()
    buckets: dict[str, list[RunRecord]] = {}
    order: list[str] = []
    for r in rest:
        day = datetime.fromtimestamp(r.created_at or 0).date()
        if day == today:
            name = "Today"
        elif day == today - timedelta(days=1):
            name = "Yesterday"
        elif day > today - timedelta(days=7):
            name = "Previous 7 days"
        elif day.year == today.year:
            name = day.strftime("%B")
        else:
            name = day.strftime("%B %Y")
        if name not in buckets:
            buckets[name] = []
            order.append(name)
        buckets[name].append(r)
    out = [("Running", active)] if active else []
    out.extend((name, buckets[name]) for name in order)
    return out


class RunDelegate(QStyledItemDelegate):
    """Paints history rows: status dot, name, meta line and a live progress bar. A row's
    name is edited in place (double-click or F2)."""

    def createEditor(self, parent, option, index):  # noqa: N802, ANN001, ANN201
        if index.data(RunListModel.RecordRole) is None and index.data(RunListModel.DraftRole) is None:
            return None
        editor = QLineEdit(parent)
        editor.setObjectName("RowTitleEdit")
        return editor

    def updateEditorGeometry(self, editor, option, index) -> None:  # noqa: N802, ANN001
        card = QRectF(option.rect).adjusted(6 + self.INDENT * (index.data(RunListModel.DepthRole) or 0), 1, -6, -1)
        x = int(card.left() + 24)
        editor.setGeometry(x, int(card.top() + 2), int(card.right() - 6 - x), 20)

    def setEditorData(self, editor, index) -> None:  # noqa: N802, ANN001
        editor.setText(index.data(Qt.ItemDataRole.EditRole) or "")
        editor.selectAll()

    def setModelData(self, editor, model, index) -> None:  # noqa: N802, ANN001
        model.setData(index, editor.text(), Qt.ItemDataRole.EditRole)

    ROW_H = 44
    HEADER_H = 26
    PROJECT_H = 30
    INDENT = 14  # runs inside a project

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        if index.data(RunListModel.HeaderRole) is not None:
            return QSize(option.rect.width(), self.HEADER_H)
        if index.data(RunListModel.ProjectRole) is not None:
            return QSize(option.rect.width(), self.PROJECT_H)
        return QSize(option.rect.width(), self.ROW_H)

    def _paint_draft(self, painter: QPainter, option: QStyleOptionViewItem, draft: DraftEntry, depth: int) -> None:
        """Like a run, with a hollow dot: being set up in its tab, not started yet."""
        p = theme().palette
        card = QRectF(option.rect).adjusted(6 + self.INDENT * depth, 1, -6, -1)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if selected or hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_soft(p.accent if selected else p.text, 0.16 if selected else 0.05))
            painter.drawRoundedRect(card, 8, 8)
        cx, cy = card.left() + 16, card.top() + 13
        pen = QPen(QColor(p.accent))
        pen.setWidthF(1.6)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(cx - 3.5, cy - 3.5, 7, 7))
        tag = kinds.badge(draft.kind)
        tf = QFont(option.font)
        tf.setPixelSize(10)
        tf.setWeight(QFont.Weight.Bold)
        tw = QFontMetrics(tf).horizontalAdvance(tag) + 10
        tag_rect = QRectF(card.right() - tw - 8, card.top() + 5, tw, 15)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_soft(p.text2, 0.10))
        painter.drawRoundedRect(tag_rect, 4, 4)
        painter.setFont(tf)
        painter.setPen(QColor(p.text2))
        painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, tag)
        x = card.left() + 30
        f = QFont(option.font)
        f.setPixelSize(13)
        f.setWeight(QFont.Weight.Medium)
        f.setItalic(not draft.name)
        painter.setFont(f)
        painter.setPen(QColor(p.text))
        title_w = tag_rect.left() - x - 8
        painter.drawText(QRectF(x, card.top() + 3, title_w, 20), Qt.AlignmentFlag.AlignVCenter,
                         QFontMetrics(f).elidedText(draft.title, Qt.TextElideMode.ElideRight, int(title_w)))
        mf = QFont(option.font)
        mf.setPixelSize(11)
        painter.setFont(mf)
        painter.setPen(QColor(p.accent_text))
        painter.drawText(QRectF(x, card.top() + 21, card.right() - x - 10, 15), Qt.AlignmentFlag.AlignVCenter,
                         "Draft · not started")

    def _paint_project(self, painter: QPainter, option: QStyleOptionViewItem, row: _Row) -> None:
        p = theme().palette
        card = QRectF(option.rect).adjusted(6, 1, -6, -1)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if selected or hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_soft(p.accent if selected else p.text, 0.16 if selected else 0.05))
            painter.drawRoundedRect(card, 8, 8)
        cy = card.center().y()
        chevron = icons.pixmap("chevron-down" if row.expanded else "chevron-right", p.text3, 14)
        painter.drawPixmap(int(card.left() + 6), int(cy - 7), chevron)
        folder = icons.pixmap("folder-open" if row.expanded else "folder", p.accent_text, 16)
        painter.drawPixmap(int(card.left() + 24), int(cy - 8), folder)
        # count (right) and a live dot when one of its runs is active
        count_font = QFont(option.font)
        count_font.setPixelSize(11)
        cfm = QFontMetrics(count_font)
        count = str(row.count)
        cw = cfm.horizontalAdvance(count)
        right = card.right() - 10
        painter.setFont(count_font)
        painter.setPen(QColor(p.text3))
        painter.drawText(QRectF(right - cw, card.top(), cw, card.height()), Qt.AlignmentFlag.AlignVCenter, count)
        right -= cw + 8
        if row.running:
            dot = QColor(p.state_color("running"))
            phase = (time.time() % 1.6) / 1.6
            halo = QColor(dot)
            halo.setAlphaF(0.45 * (1 - phase))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(QRectF(right - 7 - 3 * phase, cy - 3 - 3 * phase, 6 + 6 * phase, 6 + 6 * phase))
            painter.setBrush(dot)
            painter.drawEllipse(QRectF(right - 7, cy - 3, 6, 6))
            right -= 14
        name_font = QFont(option.font)
        name_font.setPixelSize(13)
        name_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(name_font)
        painter.setPen(QColor(p.text))
        x = card.left() + 48
        nfm = QFontMetrics(name_font)
        painter.drawText(QRectF(x, card.top(), right - x - 4, card.height()), Qt.AlignmentFlag.AlignVCenter,
                         nfm.elidedText(row.project.name, Qt.TextElideMode.ElideRight, int(right - x - 4)))

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        p = theme().palette
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(option.rect)
        header = index.data(RunListModel.HeaderRole)
        if header is not None:
            f = QFont(option.font)
            f.setPixelSize(11)
            f.setWeight(QFont.Weight.DemiBold)
            f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
            painter.setFont(f)
            painter.setPen(QColor(p.text3))
            painter.drawText(rect.adjusted(14, 6, -8, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             header.upper())
            painter.restore()
            return

        project_row = index.data(RunListModel.ProjectRole)
        if project_row is not None:
            self._paint_project(painter, option, project_row)
            painter.restore()
            return
        draft = index.data(RunListModel.DraftRole)
        if draft is not None:
            self._paint_draft(painter, option, draft, index.data(RunListModel.DepthRole) or 0)
            painter.restore()
            return

        record: RunRecord = index.data(RunListModel.RecordRole)
        card = rect.adjusted(6 + self.INDENT * (index.data(RunListModel.DepthRole) or 0), 1, -6, -1)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if selected or hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_soft(p.accent if selected else p.text, 0.16 if selected else 0.05))
            painter.drawRoundedRect(card, 8, 8)
        state = record.status.state
        # status dot (+ breathing halo for live runs)
        dot_c = QColor(p.state_color(str(state)))
        cx, cy = card.left() + 16, card.top() + 13
        if state.is_active:
            phase = (time.time() % 1.6) / 1.6
            halo = QColor(dot_c)
            halo.setAlphaF(0.45 * (1 - phase))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(QRectF(cx - 4 - 4 * phase, cy - 4 - 4 * phase, 8 + 8 * phase, 8 + 8 * phase))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dot_c)
        painter.drawEllipse(QRectF(cx - 4, cy - 4, 8, 8))

        # kind tag (right)
        tag = kinds.badge(record.kind)
        tf = QFont(option.font)
        tf.setPixelSize(10)
        tf.setWeight(QFont.Weight.Bold)
        tfm = QFontMetrics(tf)
        tw = tfm.horizontalAdvance(tag) + 10
        tag_rect = QRectF(card.right() - tw - 8, card.top() + 5, tw, 15)
        painter.setBrush(_soft(p.text2, 0.10))
        painter.drawRoundedRect(tag_rect, 4, 4)
        painter.setFont(tf)
        painter.setPen(QColor(p.text2))
        painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, tag)

        # title
        x = card.left() + 30
        title_w = tag_rect.left() - x - 8
        f = QFont(option.font)
        f.setPixelSize(13)
        f.setWeight(QFont.Weight.Medium)
        painter.setFont(f)
        painter.setPen(QColor(p.text))
        fm = QFontMetrics(f)
        painter.drawText(QRectF(x, card.top() + 3, title_w, 20), Qt.AlignmentFlag.AlignVCenter,
                         fm.elidedText(record.name, Qt.TextElideMode.ElideRight, int(title_w)))

        # meta line
        mf = QFont(option.font)
        mf.setPixelSize(11)
        painter.setFont(mf)
        model: RunListModel = index.model()
        machine = model.machine_name(record.machine_id)
        painter.setPen(QColor(p.danger if state == RunState.FAILED else p.text3))
        mfm = QFontMetrics(mf)
        meta_w = card.right() - x - 10
        meta = _meta(record, machine)
        if mfm.horizontalAdvance(meta) > meta_w:  # the machine goes before anything is cut
            meta = _meta(record, "")
        painter.drawText(QRectF(x, card.top() + 21, meta_w, 15), Qt.AlignmentFlag.AlignVCenter,
                         mfm.elidedText(meta, Qt.TextElideMode.ElideRight, int(meta_w)))

        # progress
        if state == RunState.RUNNING and record.status.progress is not None:
            bar = QRectF(x, card.bottom() - 4, card.right() - x - 10, 2.5)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(p.bg4))
            painter.drawRoundedRect(bar, 1.25, 1.25)
            painter.setBrush(QColor(p.accent))
            painter.drawRoundedRect(QRectF(bar.x(), bar.y(), max(2.5, bar.width() * record.status.progress),
                                           bar.height()), 1.25, 1.25)
        painter.restore()


def _soft(color: str, alpha: float) -> QColor:
    c = QColor(color)
    c.setAlphaF(alpha)
    return c


def _meta(record: RunRecord, machine: str) -> str:
    """A run's second line in the list (``machine`` "": without the machine)."""
    st = record.status
    if st.state == RunState.RUNNING:  # the stage's short name: the row is narrow
        stage = kinds.stage_label(record.kind, st.stage, st.stage_label) or "Running"
        parts = (f"{st.progress * 100:.0f}%" if st.progress is not None else "", stage, machine)
    elif st.state == RunState.QUEUED:
        parts = ("In queue" if st.stage == "waiting" else "Starting…", machine)
    elif st.state == RunState.FAILED:
        parts = ("Failed", machine)
    elif st.state == RunState.CANCELLED:
        parts = ("Stopped", relative_time(record.created_at))
    elif st.state == RunState.LOST:
        parts = ("Lost contact", machine)
    else:
        parts = (relative_time(record.created_at), duration(st.elapsed_s) if st.elapsed_s else "", machine)
    return " · ".join(part for part in parts if part)


class Sidebar(QWidget):
    new_requested = Signal(str)  # kind, chosen in the New menu
    new_in_project = Signal(str, str)  # kind, project id (the project's menu)
    run_selected = Signal(object)  # RunRecord
    run_action = Signal(str, object)  # action, RunRecord
    nav_requested = Signal(str)  # "machines" | "settings"
    new_project_requested = Signal()
    project_action = Signal(str, str)  # "rename" | "delete", project id
    #: File runs (keys) in a project (id), or take them out of their project (None).
    move_requested = Signal(list, object)
    expanded_changed = Signal(object)  # set of open project ids
    rename_requested = Signal(str, str)  # run key or "draft:<id>", new name (typed in place)
    draft_selected = Signal(str)  # draft id: show its tab
    draft_action = Signal(str, str)  # "rename" | "new_project" | "discard", draft id

    def __init__(self, parent: QWidget | None = None, *, title_bar: int = 0) -> None:
        """``title_bar``: height kept free at the top for the window buttons (0: none)."""
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 0 if title_bar else 14, 12, 12)
        lay.setSpacing(10)
        # the window's own buttons sit here when the content fills the title bar (macOS)
        self.title_space = TitleBarSpace(title_bar) if title_bar else None
        if self.title_space is not None:
            lay.addWidget(self.title_space)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        # the kinds of simulation open beside the sidebar when the pointer rests on New
        self.new_button = Button("New", "plus", "primary")
        self.new_button.setAccessibleName("New simulation")
        self.new_button.setAccessibleDescription("Choose the kind of simulation to set up (Ctrl+N)")
        self.new_menu = NewMenuButton(self.new_button, self._new_menu_place, hover=True)
        self.new_menu.chosen.connect(self.new_requested.emit)
        actions.addWidget(self.new_button, 1)
        self.project_button = IconButton("folder-plus", "New project: a folder to organise runs in",
                                         size=34, icon_size=17, on_click=self.new_project_requested.emit)
        actions.addWidget(self.project_button)
        lay.addLayout(actions)

        self.search = SearchField("Search runs")
        self.search.textChanged.connect(lambda *_: self._refilter())
        lay.addWidget(self.search)

        self.model = RunListModel()
        self.view = QListView()
        self.view.setModel(self.model)
        self.view.setItemDelegate(RunDelegate(self.view))
        self.view.setMouseTracking(True)
        self.view.setUniformItemSizes(False)
        self.view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.view.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._context_menu)
        self.view.clicked.connect(self._clicked)
        # double-click (or F2) a simulation to rename it, as on its tab
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                  | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.model.rename_requested.connect(self.rename_requested.emit)
        # drag runs onto a project to file them there
        self.view.setDragEnabled(True)
        self.view.setAcceptDrops(True)
        self.view.setDropIndicatorShown(True)
        self.view.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.view.setDragDropOverwriteMode(True)
        self.view.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.model.runs_dropped.connect(lambda keys, pid: self.move_requested.emit(keys, pid))
        self.view.setStyleSheet("QListView { background: transparent; } QListView::item { padding: 0; }")
        lay.addWidget(self.view, 1)

        self.empty = label("No runs yet. Start one with\n“New”.", "caption")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.empty)

        nav = QHBoxLayout()
        nav.setSpacing(4)
        self.machines_btn = Button("Machines", "server", "ghost", on_click=lambda: self.nav_requested.emit("machines"))
        self.settings_btn = IconButton("settings", "Settings  (Ctrl+,)",
                                       on_click=lambda: self.nav_requested.emit("settings"))
        nav.addWidget(self.machines_btn)
        nav.addStretch(1)
        nav.addWidget(self.settings_btn)
        lay.addLayout(nav)

        self._records: list[RunRecord] = []
        self._machines: dict[str, str] = {}
        self._projects: list[Project] = []
        self._drafts: list[DraftEntry] = []
        self._expanded: set[str] = set()
        self._current_project: str | None = None  # where "New" files the simulation
        self._selected_key: str | None = None
        self._pulse = QTimer(self)
        self._pulse.setInterval(60)
        self._pulse.timeout.connect(self.view.viewport().update)

    # ------------------------------------------------------------------ data
    def set_records(self, records: list[RunRecord], machines: dict[str, str],
                    projects: list[Project] | None = None) -> None:
        self._records = records
        self._machines = machines
        if projects is not None:
            self._projects = projects
            known = {p.id for p in projects}
            self._expanded &= known
            if self._current_project not in known:
                self._current_project = None
        self._refilter()

    def set_drafts(self, drafts: list[DraftEntry]) -> None:
        """The simulations being set up in tabs (listed where they will be filed)."""
        if drafts != self._drafts:
            self._drafts = list(drafts)
            self._refilter()

    def set_expanded(self, project_ids) -> None:  # noqa: ANN001 - iterable of ids
        self._expanded = set(project_ids)
        self._refilter()

    def expand(self, project_id: str) -> None:
        if project_id not in self._expanded:
            self._expanded.add(project_id)
            self.expanded_changed.emit(set(self._expanded))
            self._refilter()

    def current_project_id(self) -> str | None:
        """The project being worked in: the last one opened, or the selected run's."""
        return self._current_project

    def set_current_project(self, project_id: str | None) -> None:
        self._current_project = project_id

    def update_record(self, record: RunRecord) -> None:
        for i, r in enumerate(self._records):
            if r.key == record.key:
                self._records[i] = record
        self.model.update_record(record)
        self._sync_pulse()

    def select(self, key: str | None) -> None:
        self._selected_key = key
        idx = self.model.index_of(key) if key else QModelIndex()
        if idx.isValid():
            self.view.setCurrentIndex(idx)
            self.view.scrollTo(idx)
        else:
            self.view.clearSelection()

    def focus_search(self) -> None:
        self.search.setFocus()
        self.search.selectAll()

    def _refilter(self) -> None:
        self.model.set_records(self._records, self.search.text(), self._machines, self._projects, self._expanded,
                               self._drafts)
        self.empty.setVisible(not self._records and not self._projects and not self._drafts)
        self.select(self._selected_key)
        self._sync_pulse()

    def _sync_pulse(self) -> None:
        if self.model.has_active():
            if not self._pulse.isActive():
                self._pulse.start()
        else:
            self._pulse.stop()

    # ------------------------------------------------------------------ interaction
    def _clicked(self, index: QModelIndex) -> None:
        project_row = index.data(RunListModel.ProjectRole)
        if project_row is not None:  # open / close the folder
            pid = project_row.project.id
            self._current_project = pid
            self._expanded.symmetric_difference_update({pid})
            self.expanded_changed.emit(set(self._expanded))
            self._refilter()
            self.view.setCurrentIndex(self.model.index_of_project(pid))
            return
        draft = index.data(RunListModel.DraftRole)
        if draft is not None:
            self._selected_key = draft.key
            self._current_project = draft.project_id
            self.draft_selected.emit(draft.id)
            return
        record = index.data(RunListModel.RecordRole)
        if record is not None:
            self._selected_key = record.key
            self._current_project = record.project_id if record.project_id in {p.id for p in self._projects} else None
            self.run_selected.emit(record)


    def _context_menu(self, pos: QPoint) -> None:
        index = self.view.indexAt(pos)
        if not index.isValid():
            return
        project_row = index.data(RunListModel.ProjectRole)
        if project_row is not None:
            self._project_menu(project_row.project, pos)
            return
        draft = index.data(RunListModel.DraftRole)
        if draft is not None:
            self._draft_menu(draft, pos)
            return
        record: RunRecord | None = index.data(RunListModel.RecordRole)
        if record is None:
            return
        p = theme().palette
        menu = QMenu(self)

        def add(text: str, icon: str, action: str, enabled: bool = True) -> None:
            act = QAction(icons.icon(icon, p.text2), text, menu)
            act.setEnabled(enabled)
            act.triggered.connect(lambda: self.run_action.emit(action, record))
            menu.addAction(act)

        add("Open", "arrow-right", "open")
        add("Duplicate and edit", "duplicate", "duplicate")
        add("Rename…", "edit", "rename")
        self._add_project_menu(menu, record.key, record.project_id,
                               lambda: self.run_action.emit("new_project", record))
        add("Copy folder path", "copy", "copy_path")
        if record.machine_id == "local":
            add("Show in folder", "folder-open", "reveal")
        menu.addSeparator()
        add("Stop", "stop", "stop", record.status.state.is_active)
        add("Delete…", "trash", "delete", not record.status.state.is_active)
        menu.exec(self.view.viewport().mapToGlobal(pos))

    def _add_project_menu(self, menu: QMenu, key: str, current: str | None, new_project) -> None:  # noqa: ANN001
        """"Move to project ▸" (and "Remove from project") for a run or a draft (its key)."""
        p = theme().palette
        sub = menu.addMenu(icons.icon("folder", p.text2), "Move to project")
        for project in self._projects:
            act = sub.addAction(icons.icon("folder", p.text2), project.name)
            act.setEnabled(project.id != current)
            act.triggered.connect(lambda _=False, pid=project.id: self.move_requested.emit([key], pid))
        if self._projects:
            sub.addSeparator()
        sub.addAction(icons.icon("folder-plus", p.text2), "New project…").triggered.connect(new_project)
        if current in {pr.id for pr in self._projects}:
            out = menu.addAction(icons.icon("x", p.text2), "Remove from project")
            out.triggered.connect(lambda: self.move_requested.emit([key], None))

    def _draft_menu(self, draft: DraftEntry, pos: QPoint) -> None:
        p = theme().palette
        menu = QMenu(self)
        opened = menu.addAction(icons.icon("arrow-right", p.text2), "Open")
        opened.triggered.connect(lambda: self.draft_selected.emit(draft.id))
        rename = menu.addAction(icons.icon("edit", p.text2), "Rename…")
        rename.triggered.connect(lambda: self.draft_action.emit("rename", draft.id))
        self._add_project_menu(menu, draft.key, draft.project_id,
                               lambda: self.draft_action.emit("new_project", draft.id))
        menu.addSeparator()
        discard = menu.addAction(icons.icon("trash", p.text2), "Discard")
        discard.triggered.connect(lambda: self.draft_action.emit("discard", draft.id))
        menu.exec(self.view.viewport().mapToGlobal(pos))

    def _project_menu(self, project: Project, pos: QPoint) -> None:
        p = theme().palette
        menu = QMenu(self)
        new = menu.addMenu(icons.icon("plus", p.text2), "New simulation in this project")
        for spec in specs():
            kind = str(spec.kind)
            act = new.addAction(icons.icon(spec.icon, tone_color(p, kinds.tone(kind))), spec.title)
            act.triggered.connect(lambda _=False, k=kind: self.new_in_project.emit(k, project.id))
        menu.addSeparator()
        for text, icon, action in (("Rename project…", "edit", "rename"), ("Delete project…", "trash", "delete")):
            act = QAction(icons.icon(icon, p.text2), text, menu)
            act.triggered.connect(lambda _=False, a=action: self.project_action.emit(a, project.id))
            menu.addAction(act)
        menu.exec(self.view.viewport().mapToGlobal(pos))

    def _new_menu_place(self, _size: QSize) -> QPoint:
        """The New menu opens beside the sidebar, level with the New button."""
        x = self.mapToGlobal(QPoint(self.width(), 0)).x() + 6
        return QPoint(x, self.new_button.mapToGlobal(QPoint(0, 0)).y() - NewMenu.PAD)
