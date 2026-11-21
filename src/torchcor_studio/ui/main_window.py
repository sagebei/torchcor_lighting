"""Main window: the sidebar on the left, a tab per simulation (and Machines, Settings) on the right.

A *draft* tab sets up a new simulation of the kind chosen in the New menu, named after it
(EP, EP_1, …); it is listed in the sidebar where it will be filed (its project, or
"Drafts") and becomes the run's tab once launched. A simulation's name is on its tab and
its sidebar entry, and either can be edited in place. Opening a run from the sidebar uses
a *preview* tab that the next run replaces, unless it is kept (the tab's "Keep open", or
"Open" in the run's menu).
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QWidget,
)

from torchcor_studio import APP_NAME
from torchcor_studio.core.models import Project, RunRecord
from torchcor_studio.ui import kinds
from torchcor_studio.ui.context import AppContext
from torchcor_studio.ui.pages.machines_page import MachinesPage
from torchcor_studio.ui.pages.new_run import NewRunPage
from torchcor_studio.ui.pages.run_page import RunPage, reveal_local
from torchcor_studio.ui.pages.settings_page import SettingsPage
from torchcor_studio.ui.sidebar import DraftEntry, Sidebar
from torchcor_studio.ui.widgets import ToastHost
from torchcor_studio.ui.widgets.layout import PanelSplitter
from torchcor_studio.ui.widgets.titlebar import UNIFIED, unify
from torchcor_studio.ui.workspace import STRIP_HEIGHT, Workspace

#: Narrowest and widest the run list can be dragged (px).
SIDEBAR_WIDTH = (220, 560)


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        # macOS: the tabs fill the title bar and the window buttons sit over the sidebar,
        # so no title text is drawn (elsewhere the native title bar shows the name)
        self.setWindowTitle("" if UNIFIED else APP_NAME)
        self.setMinimumSize(1100, 700)
        root = QWidget()
        root.setObjectName("AppRoot")
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.workspace = Workspace()
        self.sidebar = Sidebar(title_bar=STRIP_HEIGHT if UNIFIED else 0)
        # drag the sidebar's edge to see longer run names (the width is remembered)
        self.split = PanelSplitter(self.sidebar, self.workspace, width=ctx.settings.sidebar_width,
                                   minimum=SIDEBAR_WIDTH[0], maximum=SIDEBAR_WIDTH[1])
        self.split.moved.connect(lambda w: setattr(ctx.settings, "sidebar_width", w))
        lay.addWidget(self.split)
        self.setCentralWidget(root)
        unify(self, root)

        # one each, shown as tabs on demand
        self.machines = MachinesPage(ctx)
        self.settings = SettingsPage(ctx)

        # wiring -------------------------------------------------------------------
        self.sidebar.new_requested.connect(lambda kind: self.show_new_run(kind=kind))
        self.sidebar.new_in_project.connect(self._new_in_project)
        self.sidebar.run_selected.connect(self.show_run)
        self.sidebar.rename_requested.connect(self._rename_entry)
        self.sidebar.draft_selected.connect(self._focus_draft)
        self.sidebar.draft_action.connect(self._draft_action)
        self.sidebar.run_action.connect(self._run_action)
        self.sidebar.nav_requested.connect(self._nav)
        self.sidebar.new_project_requested.connect(self._new_project)
        self.sidebar.project_action.connect(self._project_action)
        self.sidebar.move_requested.connect(self._move_runs)
        self.sidebar.expanded_changed.connect(lambda ids: setattr(ctx.settings, "expanded_projects", sorted(ids)))
        self.sidebar.set_expanded(ctx.settings.expanded_projects)
        self.workspace.new_requested.connect(lambda kind: self.show_new_run(kind=kind))
        self.workspace.close_requested.connect(self._close_tab)
        self.workspace.title_edited.connect(self._tab_renamed)
        self.workspace.current_changed.connect(self._tab_changed)
        self.machines.install_requested.connect(self.install_torchcor)
        self.machines.python_requested.connect(self.pick_python)
        ctx.runs_changed.connect(self._refresh_history)
        ctx.projects_changed.connect(self._refresh_history)
        ctx.run_updated.connect(self.sidebar.update_record)
        ctx.run_updated.connect(self._run_updated)
        ctx.machines_changed.connect(self._refresh_history)
        ctx.notify.connect(self._notify)

        for keys, fn in (
            (QKeySequence.StandardKey.New, self._open_new_menu),
            (QKeySequence.StandardKey.Close, lambda: self._close_tab(self.workspace.current())),
            (QKeySequence.StandardKey.Find, self.sidebar.focus_search),
            (QKeySequence.StandardKey.Preferences, lambda: self._nav("settings")),
            (QKeySequence("Ctrl+,"), lambda: self._nav("settings")),
            (QKeySequence.StandardKey.Refresh, lambda: ctx.refresh_runs()),
        ):
            QShortcut(keys, self, activated=fn)

        self._refresh_history()
        self.show_new_run()
        self._restore_geometry()

    # ------------------------------------------------------------------ the pages shown
    @property
    def pages(self) -> QTabWidget:
        return self.workspace.tabs

    @property
    def new_run(self) -> NewRunPage | None:
        """The simulation being set up in the current tab (else the latest draft)."""
        page = self.workspace.current()
        if isinstance(page, NewRunPage):
            return page
        drafts = self.workspace.pages("draft")
        return drafts[-1] if drafts else None

    @property
    def run_page(self) -> RunPage | None:
        """The run shown in the current tab, if any."""
        page = self.workspace.current()
        return page if isinstance(page, RunPage) else None

    # ------------------------------------------------------------------ simulations (drafts)
    def show_new_run(self, project_id: str | None = None, *, record: RunRecord | None = None,
                     kind: str | None = None) -> NewRunPage:
        """A new tab to set up a simulation of ``kind`` (by default the kind last run), named
        after it (EP, EP_1, …) and filed in the project being worked in; or prefilled from
        ``record`` to duplicate it."""
        if project_id is None and record is None:
            project_id = self.sidebar.current_project_id()
        kind = record.kind if record is not None else kind or self.ctx.settings.last_kind or kinds.DEFAULT
        page = self._make_draft(kind)
        if record is not None:
            page.start_from(record)
        else:
            page.start_fresh(project_id=project_id, name=kinds.next_name(kind, self._simulation_names()))
        self.workspace.add(page, "draft", page.title(), kinds.icon(page.setup.kind))
        self._draft_moved(page, page.project_id())
        return page

    def _make_draft(self, kind: str) -> NewRunPage:
        page = NewRunPage(self.ctx, kind)
        page.run_started.connect(lambda r, pg=page: self._launched(pg, r))
        page.add_machine_requested.connect(self._add_machine)
        page.install_requested.connect(self.install_torchcor)
        page.title_changed.connect(lambda _name, pg=page: self._draft_changed(pg))
        page.project_changed.connect(lambda pid, pg=page: self._draft_moved(pg, pid))
        page.setup.kind_changed.connect(lambda _kind, pg=page: self._draft_changed(pg))
        return page

    def duplicate(self, record: RunRecord) -> None:
        self.show_new_run(record=record)

    def _edit_run(self, run_page: RunPage, record: RunRecord, step: int) -> None:
        """A stopped or failed run's tab becomes its set-up again, at ``step`` (Compute or
        Setup): the same machine, device, folders and parameters, under its name, to change
        and run again. The run stays in the sidebar."""
        draft = self._make_draft(record.kind)
        draft.start_from(record, again=True, step=step)
        self.workspace.replace(run_page, draft, "draft", draft.title(), kinds.icon(record.kind))
        run_page.deactivate()
        run_page.deleteLater()
        self.workspace.focus(draft)
        self._draft_moved(draft, draft.project_id())

    def _open_new_menu(self) -> None:
        """Ctrl+N: the New menu from the sidebar's New button (the tab strip's "+" while the
        sidebar is hidden), its first kind current."""
        opener = self.sidebar.new_menu if self.sidebar.new_button.isVisible() else self.workspace.new_menu
        opener.open(keyboard=True)

    def _new_in_project(self, kind: str, project_id: str) -> None:
        self.sidebar.set_current_project(project_id)
        self.show_new_run(project_id, kind=kind)

    def _simulation_names(self) -> list[str]:
        """The names of the simulations being set up and of the runs: a new simulation takes
        the first name of its kind that none has."""
        return [page.name() for page in self.workspace.pages("draft")] + [r.name for r in self.ctx.records()]

    def _draft_changed(self, page: NewRunPage) -> None:
        self.workspace.set_title(page, page.title())
        self._sync_drafts()

    def _draft_moved(self, page: NewRunPage, project_id: str | None) -> None:
        if project_id:
            self.sidebar.expand(project_id)  # so its entry is in view
        self._sync_drafts()
        if self.workspace.current() is page:
            self.sidebar.select(f"draft:{page.draft_id}")

    def _sync_drafts(self) -> None:
        """List every draft tab in the sidebar, where it will be filed."""
        self.sidebar.set_drafts([
            DraftEntry(id=page.draft_id, name=page.name(), kind=page.setup.kind,
                       project_id=page.project_id())
            for page in self.workspace.pages("draft")])

    def _draft(self, draft_id: str) -> NewRunPage | None:
        return next((p for p in self.workspace.pages("draft") if p.draft_id == draft_id), None)

    def _focus_draft(self, draft_id: str) -> None:
        page = self._draft(draft_id)
        if page is not None:
            self.workspace.focus(page)

    def _draft_action(self, action: str, draft_id: str) -> None:
        page = self._draft(draft_id)
        if page is None:
            return
        if action == "rename":
            name, ok = QInputDialog.getText(self, "Rename simulation", "Name", text=page.name())
            if ok and name.strip():
                page.set_name(name.strip())
        elif action == "new_project":
            if (project := self._new_project()) is not None:
                page.set_project(project.id)
        elif action == "discard":
            self._close_tab(page)

    def _launched(self, draft: NewRunPage, record: RunRecord) -> None:
        """The draft's tab becomes the run's tab."""
        run_page = self._make_run_page()
        self.workspace.replace(draft, run_page, "run", record.name, kinds.icon(record.kind),
                               key=record.key)
        draft.deleteLater()
        self._sync_drafts()
        if record.project_id:
            self.sidebar.expand(record.project_id)
        run_page.show_record(self.ctx.record(record.key) or record, tab="Overview")
        self.sidebar.select(record.key)

    # ------------------------------------------------------------------ runs
    def _make_run_page(self) -> RunPage:
        page = RunPage(self.ctx)
        page.duplicate_requested.connect(self.duplicate)
        page.edit_requested.connect(lambda record, step, pg=page: self._edit_run(pg, record, step))
        page.deleted.connect(lambda pg=page: self._close_tab(pg, ask=False))
        return page

    def show_run(self, record: RunRecord, tab: str | None = None, keep: bool = False) -> None:
        """Show a run in its tab: the preview tab, unless ``keep`` (or it has its own)."""
        latest = self.ctx.record(record.key) or record
        page = self.workspace.find("run", latest.key)
        if page is None:
            page = None if keep else self.workspace.preview_page()
            icon = kinds.icon(latest.kind)
            if page is not None:
                self.workspace.set_key(page, latest.key)
                self.workspace.set_title(page, latest.name, icon)
            else:
                page = self._make_run_page()
                self.workspace.add(page, "run", latest.name, icon, key=latest.key, preview=not keep, focus=False)
        if keep:
            self.workspace.set_preview(page, False)
        self.workspace.focus(page)
        page.show_record(latest, tab=tab)
        self.sidebar.select(latest.key)
        if self.ctx.machine(latest.machine_id) is None:
            self.ctx.connect_machine(latest.machine_id, on_ready=lambda _s: self._reshow(latest.key))

    def _reshow(self, key: str) -> None:
        """Re-render an open run once its machine is connected (results, log, script)."""
        for page in self.workspace.pages("run"):
            if self.workspace.key(page) == key and page.record is not None and page.isVisible():
                page.show_record(self.ctx.record(key) or page.record)

    def _run_updated(self, record: RunRecord) -> None:
        for page in self.workspace.pages("run"):
            if self.workspace.key(page) == record.key:
                self.workspace.set_title(page, record.name)

    # ------------------------------------------------------------------ tabs
    def _nav(self, name: str) -> None:
        page = self.machines if name == "machines" else self.settings
        if self.workspace.find(name) is None:
            self.workspace.add(page, name, "Machines" if name == "machines" else "Settings",
                               "server" if name == "machines" else "settings")
        self.workspace.focus(page)

    def _add_machine(self) -> None:
        self._nav("machines")
        self.machines.add_machine()

    def _close_tab(self, page: QWidget | None, ask: bool = True) -> None:
        if page is None:
            return
        kind = self.workspace.kind(page)
        if kind == "draft" and ask and page.is_dirty():
            answer = QMessageBox.question(
                self, "Close this simulation?",
                f"“{page.title()}” has not been started; its set-up will be lost.",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Close,
                QMessageBox.StandardButton.Close)
            if answer != QMessageBox.StandardButton.Close:
                return
        self.workspace.remove(page)
        if kind == "run":
            page.deactivate()
        if kind in ("draft", "run"):
            page.deleteLater()
        if kind == "draft":
            self._sync_drafts()
        if not self.workspace.pages():
            self.show_new_run()  # never an empty window

    def _rename_entry(self, key: str, name: str) -> None:
        """A name typed in the sidebar: the same rename as on the tab."""
        if key.startswith("draft:"):
            page = self._draft(key.removeprefix("draft:"))
            if page is not None:
                page.set_name(name)
        elif (record := self.ctx.record(key)) is not None:
            self.ctx.rename(record, name)

    def _tab_renamed(self, page: QWidget, name: str) -> None:
        kind = self.workspace.kind(page)
        if kind == "draft":
            page.set_name(name)
        elif kind == "run" and page.record is not None:
            self.ctx.rename(page.record, name)

    def _tab_changed(self, page: QWidget | None) -> None:
        """Keep the sidebar on what the current tab shows."""
        kind = self.workspace.kind(page)
        if kind == "run" and page.record is not None:
            self.sidebar.select(page.record.key)
            self.sidebar.set_current_project(page.record.project_id)
        elif kind == "draft":
            self.sidebar.select(f"draft:{page.draft_id}")
            self.sidebar.set_current_project(page.project_id())
        else:
            self.sidebar.select(None)

    def _refresh_history(self) -> None:
        names = {p.id: p.name for p in self.ctx.profiles()}
        self.sidebar.set_records(self.ctx.records(), names, self.ctx.projects())

    # ------------------------------------------------------------------ projects
    def _ask_project_name(self, title: str, text: str = "") -> str | None:
        name, ok = QInputDialog.getText(self, title, "Project name", text=text)
        return name.strip() if ok and name.strip() else None

    def _new_project(self, records: list[RunRecord] | None = None) -> Project | None:
        name = self._ask_project_name("New project")
        if name is None:
            return None
        project = self.ctx.create_project(name)
        if records:
            self.ctx.move_to_project(records, project.id)
        self.sidebar.set_current_project(project.id)
        self.sidebar.expand(project.id)
        return project

    def _project_action(self, action: str, project_id: str) -> None:
        project = self.ctx.project(project_id)
        if project is None:
            return
        if action == "rename":
            name = self._ask_project_name("Rename project", project.name)
            if name is not None:
                self.ctx.rename_project(project_id, name)
        elif action == "delete":
            runs = sum(r.project_id == project_id for r in self.ctx.records())
            answer = QMessageBox.question(
                self, "Delete this project?",
                f"The project “{project.name}” will be removed. Its {runs} run{'s' if runs != 1 else ''} "
                "are kept and listed by date again; no files are deleted.",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
                QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Yes:
                self.ctx.delete_project(project_id)

    def _move_runs(self, keys: list[str], project_id: str | None) -> None:
        """File runs and drafts (``draft:<id>``) in a project (``None``: take them out)."""
        for key in keys:
            if key.startswith("draft:") and (page := self._draft(key.removeprefix("draft:"))) is not None:
                page.set_project(project_id)  # its entry follows (project_changed)
        records = [r for r in (self.ctx.record(k) for k in keys if not k.startswith("draft:")) if r is not None]
        if records:
            self.ctx.move_to_project(records, project_id)
            if project_id is not None:
                self.sidebar.expand(project_id)

    def _notify(self, title: str, message: str, tone: str) -> None:
        ToastHost.show(self, title, message, tone)
        if tone in ("success", "danger") and not self.isActiveWindow():
            QApplication.alert(self)  # bounce the dock icon / flash the taskbar

    # ------------------------------------------------------------------ actions
    def _run_action(self, action: str, record: RunRecord) -> None:
        if action == "open":
            self.show_run(record, keep=True)
        elif action == "duplicate":
            self.duplicate(record)
        elif action == "rename":
            name, ok = QInputDialog.getText(self, "Rename run", "Name", text=record.name)
            if ok and name.strip():
                self.ctx.rename(record, name.strip())
                self._refresh_history()
        elif action == "new_project":
            self._new_project([record])
        elif action == "copy_path":
            QApplication.clipboard().setText(record.run_dir)
            self.ctx.notify.emit("Copied", record.run_dir, "success")
        elif action == "reveal":
            reveal_local(record.run_dir)
        elif action == "stop":
            self.ctx.cancel(record)
        elif action == "delete":
            answer = QMessageBox.warning(
                self, "Delete this run?",
                f"“{record.name}” and its folder will be permanently deleted:\n\n{record.run_dir}\n\n"
                "This cannot be undone.",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
                QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Yes:
                for page in self.workspace.pages("run"):
                    if self.workspace.key(page) == record.key:
                        self._close_tab(page, ask=False)
                self.ctx.delete(record)

    def install_torchcor(self, machine_id: str) -> None:
        machine = self.ctx.machine(machine_id)
        if machine is None:
            self.ctx.notify.emit("Not connected", "Connect to the machine first.", "warning")
            return
        from torchcor_studio.ui.dialogs.install_dialog import InstallDialog

        dlg = InstallDialog(machine, machine.python, self.ctx.settings.torchcor_requirement(), parent=self)
        dlg.exec()
        self.ctx.reprobe(machine_id)

    def pick_python(self, machine_id: str) -> None:
        from torchcor_studio.ui.dialogs.python_picker import pick_python

        pick_python(self, self.ctx, machine_id)

    # ------------------------------------------------------------------ window state
    def _restore_geometry(self) -> None:
        geo = self.ctx.settings.window_geometry
        if geo:
            self.restoreGeometry(QByteArray.fromBase64(geo.encode()))
        else:
            screen = QApplication.primaryScreen().availableGeometry()
            w, h = min(1480, int(screen.width() * 0.9)), min(940, int(screen.height() * 0.9))
            self.resize(w, h)
            self.move(screen.center().x() - w // 2, screen.center().y() - h // 2)

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        ToastHost.relayout(self)

    def closeEvent(self, e) -> None:  # noqa: N802
        self.ctx.settings.window_geometry = bytes(self.saveGeometry().toBase64()).decode()
        self.ctx.store.save_settings()
        self.ctx.history.save()
        for page in self.workspace.pages("run"):
            page.deactivate()
        self.ctx.shutdown()
        super().closeEvent(e)
