"""Browse folders and files on any :class:`Machine` (SSH hosts included).

``RemoteBrowserDialog.get_path(machine, "~/meshes", mode="dir")`` returns the chosen
path or ``None``. Listings run off the UI thread; the path bar shows clickable
breadcrumbs and turns into a text field for typing a path (click it or press Ctrl+L).
"""

from __future__ import annotations

import time
from datetime import datetime

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLineEdit,
    QMenu,
    QStackedWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.core.machines.base import Machine
from torchcor_studio.core.models import FileEntry
from torchcor_studio.ui import icons
from torchcor_studio.ui.tasks import run_async
from torchcor_studio.ui.theme import Palette, theme
from torchcor_studio.ui.widgets import (
    Badge,
    Button,
    IconButton,
    IconLabel,
    Spinner,
    ToggleSwitch,
    divider,
    label,
)
from torchcor_studio.ui.widgets.gpu import IconTile

_MESH_FILES = (".pts", ".elem", ".lon", ".vtx", ".surf")


def human_size(size: int) -> str:
    """``812 B``, ``12.4 KB``, ``3.1 MB`` …"""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def relative_time(mtime: float, now: float | None = None) -> str:
    """``Just now``, ``5 min ago``, ``3 h ago``, ``Yesterday``, ``Mar 3`` or ``Mar 3, 2023``."""
    if not mtime:
        return ""
    now = now or time.time()
    delta = now - mtime
    if delta < 60:
        return "Just now"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)} h ago"
    when = datetime.fromtimestamp(mtime)
    today = datetime.fromtimestamp(now).date()
    if (today - when.date()).days == 1:
        return "Yesterday"
    if when.year == today.year:
        return f"{when:%b} {when.day}"
    return f"{when:%b} {when.day}, {when.year}"


def mesh_summary(entries: list[FileEntry]) -> str | None:
    """Describe the openCARP mesh(es) in a listing (``.pts`` + ``.elem``), if any."""
    names = [e.name for e in entries if not e.is_dir]
    pts = {n[:-4] for n in names if n.lower().endswith(".pts")}
    elem = {n[:-5] for n in names if n.lower().endswith(".elem")}
    if not pts or not elem:
        return None
    both = sorted(pts & elem)
    if len(both) > 1:
        return f"{len(both)} openCARP meshes"
    mesh = both[0] if both else sorted(pts)[0]
    extras = []
    if any(n.lower() == f"{mesh.lower()}.lon" for n in names):
        extras.append("fibres")
    vtx = sum(1 for n in names if n.lower().endswith(".vtx"))
    if vtx:
        extras.append(f"{vtx} .vtx")
    return "openCARP mesh · " + " · ".join([mesh, *extras])


# ============================================================================ path bar
class _PathBar(QStackedWidget):
    """Breadcrumbs that become an editable path field (Ctrl+L, or click the bar)."""

    path_requested = Signal(str)

    def __init__(self, machine: Machine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.machine = machine
        self._path = ""
        self.setFixedHeight(theme().metrics.control_h)

        self._crumbs = QWidget()
        self._crumbs.setObjectName("PathCrumbs")
        self._crumbs.setCursor(Qt.CursorShape.IBeamCursor)
        self._crumb_lay = QHBoxLayout(self._crumbs)
        self._crumb_lay.setContentsMargins(6, 0, 4, 0)
        self._crumb_lay.setSpacing(0)
        self.addWidget(self._crumbs)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText("Type a path and press Enter")
        self._edit.installEventFilter(self)
        self.addWidget(self._edit)
        theme().changed.connect(self._restyle)
        self._restyle(theme().palette)

    def _restyle(self, p: Palette) -> None:
        m = theme().metrics
        self._crumbs.setStyleSheet(
            f"QWidget#PathCrumbs {{ background: {p.bg3}; border: 1px solid {p.line2}; "
            f"border-radius: {m.r_md}px; }}"
            f"QWidget#PathCrumbs:hover {{ border-color: {p.text3}; }}"
            f"QToolButton {{ border: none; background: transparent; padding: 0 6px; min-height: 24px; "
            f"color: {p.text2}; border-radius: {m.r_sm}px; }}"
            f"QToolButton:hover {{ background: {p.bg4}; color: {p.text}; }}"
            f"QToolButton[current=\"true\"] {{ color: {p.text}; font-weight: 600; }}")
        self._rebuild()

    # ------------------------------------------------------------------ api
    def set_path(self, path: str) -> None:
        self._path = path
        self._edit.setText(path)
        self._rebuild()
        self.setCurrentWidget(self._crumbs)

    def start_editing(self) -> None:
        self._edit.setText(self._path)
        self.setCurrentWidget(self._edit)
        self._edit.setFocus()
        self._edit.selectAll()

    # ------------------------------------------------------------------ internals
    def _chain(self) -> list[tuple[str, str]]:
        """``(label, path)`` from the root down to the current folder."""
        chain, path = [], self._path
        for _ in range(64):
            name = self.machine.basename(path) or path
            chain.append((name, path))
            parent = self.machine.parent(path)
            if not parent or parent == path:
                break
            path = parent
        return list(reversed(chain))

    def _rebuild(self) -> None:
        while self._crumb_lay.count():
            item = self._crumb_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not self._path:
            return
        chain = self._chain()
        p = theme().palette
        budget = max(160, self.width() - 40)
        shown: list[tuple[str, str]] = []
        used = 0
        fm = self.fontMetrics()
        for crumb in reversed(chain):
            cost = fm.horizontalAdvance(crumb[0]) + 30
            if shown and used + cost > budget:
                break
            shown.insert(0, crumb)
            used += cost
        hidden = chain[: len(chain) - len(shown)]
        if hidden:
            more = QToolButton()
            more.setText("…")
            more.setToolTip("Parent folders")
            menu = QMenu(more)
            for name, path in reversed(hidden):
                menu.addAction(name, lambda path=path: self.path_requested.emit(path))
            more.setMenu(menu)
            more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            self._crumb_lay.addWidget(more)
            self._crumb_lay.addWidget(self._chevron(p))
        for i, (name, path) in enumerate(shown):
            button = QToolButton()
            button.setText(name)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setProperty("current", i == len(shown) - 1)
            button.clicked.connect(lambda _=False, path=path: self.path_requested.emit(path))
            self._crumb_lay.addWidget(button)
            if i < len(shown) - 1:
                self._crumb_lay.addWidget(self._chevron(p))
        self._crumb_lay.addStretch(1)

    @staticmethod
    def _chevron(p: Palette) -> QWidget:
        return IconLabel("chevron-right", 12, color=p.text3)

    def _submit(self) -> None:
        text = self._edit.text().strip()
        if text:
            self.path_requested.emit(text)
        self.setCurrentWidget(self._crumbs)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self._edit and event.type() == event.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._submit()  # consumed: the dialog's default button must not fire
                return True
            if event.key() == Qt.Key.Key_Escape:
                self.setCurrentWidget(self._crumbs)
                return True
        if obj is self._edit and event.type() == event.Type.FocusOut:
            self.setCurrentWidget(self._crumbs)
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.currentWidget() is self._crumbs:  # a click beside the crumbs edits the path
            self.start_editing()
            return
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        QTimer.singleShot(0, self._rebuild)


# ============================================================================ dialog
class RemoteBrowserDialog(QDialog):
    """Pick a folder (``mode="dir"``) or a file (``mode="file"``) on a machine."""

    def __init__(self, machine: Machine, start_path: str = "", mode: str = "dir", title: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if mode not in ("dir", "file"):
            raise ValueError("mode must be 'dir' or 'file'")
        self.machine = machine
        self.mode = mode
        self._current: str | None = None
        self._entries: list[FileEntry] = []
        self._request = 0
        self._selected: str | None = None
        self.setWindowTitle(title or ("Choose a folder" if mode == "dir" else "Choose a file"))
        self.setModal(True)
        self.resize(720, 540)
        self.setMinimumSize(520, 400)
        self._build(title)
        theme().changed.connect(self._on_theme)
        QTimer.singleShot(0, lambda: self.navigate(start_path or "~"))

    @staticmethod
    def get_path(machine: Machine, start_path: str = "", mode: str = "dir", title: str = "",
                 parent: QWidget | None = None) -> str | None:
        """Run the dialog and return the chosen path, or ``None`` if cancelled."""
        dialog = RemoteBrowserDialog(machine, start_path, mode, title, parent)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        return dialog.selected_path() if accepted else None

    # ------------------------------------------------------------------ ui
    def _build(self, title: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        head = QHBoxLayout()
        head.setSpacing(14)
        head.addWidget(IconTile("folder-open" if self.mode == "dir" else "file", "accent", size=40), 0,
                       Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(3)
        col.addWidget(label(title or self.windowTitle(), "title"))
        where = self.machine.profile.name
        address = self.machine.profile.address
        col.addWidget(label(where if address == "localhost" else f"{where} · {address}", "caption"))
        head.addLayout(col, 1)
        root.addLayout(head)

        bar = QHBoxLayout()
        bar.setSpacing(4)
        self._up = IconButton("chevron-up", "Parent folder (Backspace)", on_click=self.go_up)
        self._home = IconButton("user", "Home folder", on_click=lambda: self.navigate("~"))
        self._refresh = IconButton("refresh", "Refresh", on_click=self.refresh)
        for b in (self._up, self._home, self._refresh):
            bar.addWidget(b)
        bar.addSpacing(4)
        self._path_bar = _PathBar(self.machine)
        self._path_bar.path_requested.connect(self.navigate)
        bar.addWidget(self._path_bar, 1)
        bar.addSpacing(4)
        self._new_folder = Button("New folder", icon="plus", variant="ghost", on_click=self._make_folder)
        bar.addWidget(self._new_folder)
        root.addLayout(bar)

        self._list = QTreeWidget()
        self._list.setColumnCount(3)
        self._list.setHeaderLabels(["Name", "Size", "Modified"])
        self._list.headerItem().setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._list.setRootIsDecorated(False)
        self._list.setUniformRowHeights(True)
        self._list.setIconSize(QSize(16, 16))
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self._list.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 96)
        header.resizeSection(2, 132)
        header.setSectionsClickable(False)
        self._list.itemDoubleClicked.connect(self._activate)
        self._list.itemSelectionChanged.connect(self._selection_changed)
        self._list_frame = QWidget()
        self._list_frame.setObjectName("BrowserList")
        frame_lay = QVBoxLayout(self._list_frame)
        frame_lay.setContentsMargins(6, 4, 6, 6)
        frame_lay.addWidget(self._list)
        root.addWidget(self._list_frame, 1)

        self._overlay = QWidget(self._list_frame)
        ov = QHBoxLayout(self._overlay)
        ov.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ov.setSpacing(10)
        ov.addWidget(Spinner(18))
        self._overlay_text = label("Loading…", "muted")
        ov.addWidget(self._overlay_text)
        self._overlay.hide()

        self._error_row = QWidget()
        er = QHBoxLayout(self._error_row)
        er.setContentsMargins(2, 0, 2, 0)
        er.setSpacing(8)
        er.addWidget(IconLabel("alert", 14, tone="danger"))
        self._error = label("", "error", wrap=True)
        er.addWidget(self._error, 1)
        self._error_row.hide()
        root.addWidget(self._error_row)

        root.addWidget(divider())
        foot = QHBoxLayout()
        foot.setSpacing(10)
        self._mesh = Badge("", "accent", icon="mesh")
        self._mesh.hide()
        foot.addWidget(self._mesh)
        foot.addStretch(1)
        self._hidden = ToggleSwitch(False)
        self._hidden.toggled.connect(lambda *_: self._render())
        foot.addWidget(self._hidden)
        foot.addWidget(label("Hidden files", "muted"))
        foot.addSpacing(12)
        cancel = Button("Cancel", variant="ghost", on_click=self.reject)
        foot.addWidget(cancel)
        self._choose = Button("Select", variant="primary", on_click=self._accept_current)
        self._choose.setDefault(True)
        foot.addWidget(self._choose)
        root.addLayout(foot)

        QShortcut(QKeySequence("Ctrl+L"), self, activated=self._path_bar.start_editing)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self._list, activated=self.go_up,
                  context=Qt.ShortcutContext.WidgetShortcut)
        self._restyle(theme().palette)
        theme().changed.connect(self._restyle)

    def _restyle(self, p: Palette) -> None:
        m = theme().metrics
        self._list_frame.setStyleSheet(
            f"QWidget#BrowserList {{ background: {p.bg2}; border: 1px solid {p.line}; "
            f"border-radius: {m.r_lg}px; }}")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._overlay.setGeometry(self._list_frame.rect())

    # ------------------------------------------------------------------ navigation
    @property
    def current_path(self) -> str | None:
        return self._current

    def selected_path(self) -> str | None:
        """The accepted path (after the dialog closed with *Select*)."""
        return self._selected

    def navigate(self, path: str) -> None:
        """List ``path`` (``~`` and relative paths are resolved on the machine)."""
        self._request += 1
        request = self._request
        self._set_busy(True, "Connecting…" if not self.machine.is_connected else "Loading…")
        run_async(self._list_job, self.machine, path,
                  on_done=lambda result: self._show(request, *result),
                  on_error=lambda exc: self._fail(request, path, exc), owner=self)

    @staticmethod
    def _list_job(machine: Machine, path: str) -> tuple[str, list[FileEntry]]:
        resolved = machine.expand(path)
        return resolved, machine.listdir(resolved)

    def refresh(self) -> None:
        if self._current:
            self.navigate(self._current)

    def go_up(self) -> None:
        if self._current:
            parent = self.machine.parent(self._current)
            if parent and parent != self._current:
                self.navigate(parent)

    def _show(self, request: int, path: str, entries: list[FileEntry]) -> None:
        if request != self._request:
            return
        self._current = path
        self._entries = entries
        self._error_row.hide()
        self._path_bar.set_path(path)
        self._render()
        self._set_busy(False)
        self._list.setFocus()

    def _fail(self, request: int, path: str, exc: BaseException) -> None:
        if request != self._request:
            return
        self._set_busy(False)
        reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        self._error.setText(f"Could not open {path}: {reason}")
        self._error_row.show()
        if self._current is None:
            self._path_bar.set_path(path)

    def _set_busy(self, busy: bool, text: str = "Loading…") -> None:
        self._overlay_text.setText(text)
        self._overlay.setGeometry(self._list_frame.rect())
        self._overlay.setVisible(busy)
        self._overlay.raise_()
        self._list.setEnabled(not busy)
        for widget in (self._up, self._home, self._refresh, self._new_folder):
            widget.setEnabled(not busy)
        self._update_choose()

    # ------------------------------------------------------------------ listing
    def _on_theme(self, *_args) -> None:
        self._render()

    def _render(self) -> None:
        p = theme().palette
        show_hidden = self._hidden.isChecked()
        self._list.clear()
        folder_icon = icons.icon("folder", p.text2)
        file_icon = icons.icon("file", p.text3)
        mesh_icon = icons.icon("mesh", p.accent_text)
        now = time.time()
        for entry in self._entries:
            if entry.name.startswith(".") and not show_hidden:
                continue
            item = QTreeWidgetItem([entry.name, "—" if entry.is_dir else human_size(entry.size),
                                    relative_time(entry.mtime, now)])
            if entry.is_dir:
                item.setIcon(0, folder_icon)
            elif entry.name.lower().endswith(_MESH_FILES):
                item.setIcon(0, mesh_icon)
            else:
                item.setIcon(0, file_icon)
                if self.mode == "dir":
                    item.setForeground(0, p.qcolor("text2"))
            for col, align in ((1, Qt.AlignmentFlag.AlignRight), (2, Qt.AlignmentFlag.AlignLeft)):
                item.setForeground(col, p.qcolor("text3"))
                item.setTextAlignment(col, align | Qt.AlignmentFlag.AlignVCenter)
            item.setData(0, Qt.ItemDataRole.UserRole, entry)
            if entry.mtime:
                item.setToolTip(2, datetime.fromtimestamp(entry.mtime).strftime("%Y-%m-%d %H:%M"))
            self._list.addTopLevelItem(item)
        summary = mesh_summary(self._entries)
        self._mesh.setVisible(summary is not None)
        if summary:
            self._mesh.set(summary)
        self._update_choose()

    def _entry(self, item: QTreeWidgetItem | None) -> FileEntry | None:
        return item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None

    def _selection_changed(self) -> None:
        self._update_choose()

    def _update_choose(self) -> None:
        busy = self._overlay.isVisible()
        if self.mode == "dir":
            name = self.machine.basename(self._current or "") or (self._current or "")
            self._choose.setText(f"Select “{name}”" if name else "Select folder")
            self._choose.setEnabled(bool(self._current) and not busy)
        else:
            entry = self._entry(self._list.currentItem())
            self._choose.setText("Select file")
            self._choose.setEnabled(entry is not None and not entry.is_dir and not busy)

    def _activate(self, item: QTreeWidgetItem) -> None:
        entry = self._entry(item)
        if entry is None:
            return
        if entry.is_dir:
            self.navigate(entry.path)
        elif self.mode == "file":
            self._selected = entry.path
            self.accept()

    def _accept_current(self) -> None:
        if self.mode == "dir":
            if self._current:
                self._selected = self._current
                self.accept()
            return
        entry = self._entry(self._list.currentItem())
        if entry is not None and entry.is_dir:
            self.navigate(entry.path)
        elif entry is not None:
            self._selected = entry.path
            self.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._list.hasFocus():
            item = self._list.currentItem()
            if item is not None:
                self._activate(item)
                return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ new folder
    def _make_folder(self) -> None:
        if not self._current:
            return
        name, ok = QInputDialog.getText(self, "New folder", "Folder name")
        name = name.strip()
        if not ok or not name:
            return
        if "/" in name or "\\" in name:
            self._error.setText("A folder name cannot contain slashes.")
            self._error_row.show()
            return
        target = self.machine.join(self._current, name)
        self._set_busy(True, "Creating folder…")
        run_async(self.machine.mkdir, target,
                  on_done=lambda _: self.navigate(target),
                  on_error=lambda exc: self._fail(self._request, target, exc), owner=self)

