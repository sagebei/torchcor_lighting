"""The right panel's tabs: a tab per simulation, listed in the sidebar while being set up."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QLabel

pytestmark = pytest.mark.gui


@pytest.fixture
def window(themed, qtbot):  # noqa: ANN001, ANN201
    from torchcor_studio.core.settings import LOCAL_MACHINE_ID, Store
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.main_window import MainWindow

    store = Store()
    local = store.machine(LOCAL_MACHINE_ID)
    local.python = sys.executable
    store.upsert_machine(local)
    ctx = AppContext(store)
    win = MainWindow(ctx)
    qtbot.addWidget(win)
    win.resize(1400, 900)
    win.show()
    yield win
    ctx.shutdown()


def _sidebar_rows(win) -> list[str]:  # noqa: ANN001
    model = win.sidebar.model
    out = []
    for i in range(model.rowCount()):
        idx = model.index(i)
        header, draft = idx.data(model.HeaderRole), idx.data(model.DraftRole)
        folder, record = idx.data(model.ProjectRole), idx.data(model.RecordRole)
        indent = "  " * (idx.data(model.DepthRole) or 0)
        out.append(f"#{header}" if header else f"[{folder.project.name}]" if folder
                   else f"{indent}draft:{draft.title}" if draft else f"{indent}{record.name}")
    return out


def test_each_new_simulation_gets_a_tab_and_a_sidebar_entry(window, qtbot) -> None:
    from PySide6.QtCore import Qt

    win = window
    first = win.new_run
    assert win.workspace.kind(first) == "draft" and win.pages.tabText(0) == "EP"  # named after its kind
    assert first.setup.kind == "ep" and _sidebar_rows(win) == ["#Drafts", "draft:EP"]

    assert not first.header.findChildren(QLabel)  # the name is on the tab and in the sidebar only
    model = win.sidebar.model
    entry = model.index_of(f"draft:{first.draft_id}")
    assert model.flags(entry) & model.flags(entry).__class__.ItemIsEditable
    model.setData(entry, "Atrial pacing")  # typed in place in the sidebar
    assert win.pages.tabText(0) == "Atrial pacing" and first.name() == "Atrial pacing"
    assert _sidebar_rows(win) == ["#Drafts", "draft:Atrial pacing"]

    project = win.ctx.create_project("Benchmarks")
    win.sidebar.set_current_project(project.id)
    second = win.show_new_run()  # the kind last run, EP (the first one renamed: EP is free)
    assert win.workspace.current() is second and len(win.workspace.pages("draft")) == 2
    assert second.project_id() == project.id
    assert _sidebar_rows(win) == ["#Projects", "[Benchmarks]", "  draft:EP",
                                  "#Drafts", "draft:Atrial pacing"]

    win._tab_renamed(second, "Slab benchmark")  # double-click on the tab, then type
    assert second.name() == "Slab benchmark" and "  draft:Slab benchmark" in _sidebar_rows(win)
    win.sidebar.draft_selected.emit(first.draft_id)  # clicking an entry shows its tab
    assert win.workspace.current() is first

    win.sidebar.move_requested.emit([f"draft:{second.draft_id}"], None)  # "Remove from project"
    assert second.project_id() is None
    assert _sidebar_rows(win) == ["#Projects", "[Benchmarks]", "#Drafts", "draft:Atrial pacing",
                                  "draft:Slab benchmark"]
    entry, folder = model.index_of(f"draft:{first.draft_id}"), model.index(1)  # dragged onto the folder
    assert model.flags(entry) & Qt.ItemFlag.ItemIsDragEnabled
    assert model.dropMimeData(model.mimeData([entry]), Qt.DropAction.MoveAction, -1, -1, folder) is False
    assert first.project_id() == project.id
    assert _sidebar_rows(win) == ["#Projects", "[Benchmarks]", "  draft:Atrial pacing", "#Drafts",
                                  "draft:Slab benchmark"]


def test_new_simulations_are_named_after_their_kind() -> None:
    from torchcor_studio.ui.kinds import next_name

    assert next_name("ep", []) == "EP" and next_name("ep", ["EP", "ECG"]) == "EP_1"
    assert next_name("ep", ["ep", "EP_1", "EP_3"]) == "EP_2"  # whatever the case; the first free
    assert next_name("ep", ["EP · Duration = 200 ms"]) == "EP_1"  # a sweep's runs keep its name
    assert next_name("mechanics", []) == "MECH" and next_name("script", ["PY"]) == "PY_1"


def test_the_new_menu_starts_a_simulation_of_the_kind_chosen(window, qtbot, monkeypatch) -> None:
    """New lists the five kinds beside the sidebar (on a click, from Ctrl+N, or when the
    pointer rests on it); choosing one opens a simulation of that kind, named after it."""
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QEnterEvent, QMouseEvent
    from PySide6.QtWidgets import QApplication

    from torchcor_studio.core.simulations import get_spec

    win = window
    button, menu = win.sidebar.new_button, win.sidebar.new_menu.menu
    assert button.text() == "New" and [k for k, *_ in menu.kinds] == ["ep", "ecg", "mechanics", "em", "script"]
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(menu.isVisible, timeout=3000)
    assert menu.geometry().left() >= win.sidebar.mapToGlobal(QPoint(win.sidebar.width(), 0)).x()  # on its right
    for _ in range(2):
        qtbot.keyClick(menu, Qt.Key.Key_Down)
    assert menu.current() == "ecg"
    qtbot.keyClick(menu, Qt.Key.Key_Return)
    ecg = win.workspace.current()
    assert not menu.isVisible() and ecg.setup.kind == "ecg" and ecg.name() == "ECG"
    assert win.pages.tabText(win.pages.currentIndex()) == "ECG"
    assert ecg.setup.kind_title.text() == get_spec("ecg").title and not hasattr(ecg.setup, "kinds")

    win._open_new_menu()  # Ctrl+N: the first kind current
    qtbot.waitUntil(menu.isVisible, timeout=3000)
    assert menu.current() == "ep"
    qtbot.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu._row(1).center().toPoint())  # ECG again
    second = win.workspace.current()
    assert second is not ecg and second.setup.kind == "ecg" and second.name() == "ECG_1"

    project = win.ctx.create_project("Benchmarks")  # a project's menu: New simulation in it
    win.sidebar.new_in_project.emit("mechanics", project.id)
    mech = win.workspace.current()
    assert mech.setup.kind == "mechanics" and mech.name() == "MECH" and mech.project_id() == project.id

    from torchcor_studio.ui import new_menu

    def on_new() -> QPoint:  # New's centre now: the sidebar may still be laying out
        return button.mapToGlobal(button.rect().center())

    # Where New sees the pointer: not every CI machine lets a test move the real one.
    pointer = [on_new]
    monkeypatch.setattr(new_menu, "QCursor", type("Pointer", (), {"pos": staticmethod(lambda: pointer[0]())}))
    # The menus above closed wherever the platform left the pointer (on New, it would keep the
    # button from reopening): a pointer that comes to rest on New has left it first.
    QApplication.sendEvent(button, QEvent(QEvent.Type.Leave))
    centre = QPointF(on_new())
    QApplication.sendEvent(button, QEnterEvent(QPointF(button.rect().center()), centre, centre))
    qtbot.waitUntil(menu.isVisible, timeout=3000)  # the pointer rests on New: the menu opens...
    away = on_new() + QPoint(900, 600)
    pointer[0] = lambda: away  # ...and closes once the pointer has gone elsewhere
    QApplication.sendEvent(menu, QMouseEvent(QEvent.Type.MouseMove, QPointF(menu.mapFromGlobal(away)),
                                             QPointF(away), Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                                             Qt.KeyboardModifier.NoModifier))
    qtbot.waitUntil(lambda: not menu.isVisible(), timeout=3000)


def test_closing_tabs(window, qtbot, monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    win = window
    draft = win.new_run
    win._nav("machines")
    win._nav("machines")  # one Machines tab, focused again
    assert [win.workspace.kind(p) for p in win.workspace.pages()] == ["draft", "machines"]
    win._close_tab(win.machines)
    win._close_tab(draft)  # nothing set up: closes without asking
    qtbot.wait(10)
    assert [win.workspace.kind(p) for p in win.workspace.pages()] == ["draft"]  # never empty
    named = win.new_run
    named.set_name("Keep me")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    win._close_tab(named)  # it has a name: asks first, and Cancel keeps it
    assert named in win.workspace.pages()


def _fake_run(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    now = time.time()
    (run_dir / "manifest.json").write_text(json.dumps({"id": name, "name": name, "kind": "ep",
                                                       "created_at": now, "params": {}, "device": "cpu"}))
    (run_dir / "status.json").write_text(json.dumps({"state": "failed", "started_at": now - 5, "ended_at": now,
                                                     "updated_at": now, "error": "test"}))
    (run_dir / "run.py").write_text("print('done')\n")
    return run_dir


def test_a_stopped_run_is_set_up_again_in_its_tab(window, qtbot, tmp_path: Path) -> None:
    """After Stop (or a failure), the run's Compute and Setup steps open it to change and run
    again: its tab becomes its set-up (kind, name, parameters, folders), the run stays listed.
    A completed run's steps keep showing what it ran."""
    from torchcor_studio.core.models import RunRecord, RunState, RunStatus
    from torchcor_studio.core.simulations import ep
    from torchcor_studio.ui.pages.new_run.page import NewRunPage
    from torchcor_studio.ui.pages.run_page import RunPage

    win = window
    runs = str(tmp_path / "runs")

    def record(name: str, state: RunState) -> RunRecord:
        run = RunRecord(id=name, name=name, kind="ep", machine_id="local", run_dir=str(_fake_run(tmp_path, name)),
                        created_at=time.time(), status=RunStatus(state=state), device="cpu",
                        manifest={"params": ep.atrium().model_dump(mode="json"), "input_dir": "/data/atrium",
                                  "output_dir": runs})
        win.ctx.history.upsert(run)
        return run

    done = record("Done", RunState.COMPLETED)
    win.show_run(done, tab="Overview", keep=True)
    page = win.workspace.current()
    assert isinstance(page, RunPage) and not page.edit_btn.isVisibleTo(page)
    page.steps.step_clicked.emit(1)  # Setup of a completed run: what it ran
    assert win.workspace.current() is page and page._view == "Parameters"

    stopped = record("Atrium pacing", RunState.CANCELLED)
    win.show_run(stopped, tab="Overview", keep=True)
    page = win.workspace.current()
    assert page.edit_btn.isVisibleTo(page)  # "Edit and run again" on the stopped banner
    index = win.pages.indexOf(page)
    page.steps.step_clicked.emit(1)  # Setup
    draft = win.workspace.current()
    assert isinstance(draft, NewRunPage) and win.pages.indexOf(draft) == index  # in the same tab
    assert win.workspace.kind(draft) == "draft" and draft._step == 1
    assert draft.setup.kind == "ep" and draft.name() == "Atrium pacing" and draft.setup.demo().key == "atrium"
    assert draft.folders.input_dir() == "/data/atrium" and draft.folders.output_dir() == runs
    assert win.ctx.record(stopped.key) is not None  # the stopped run stays in the history
    assert "draft:Atrium pacing" in _sidebar_rows(win)

    failed = record("Broken", RunState.FAILED)
    win.show_run(failed, tab="Overview", keep=True)
    win.workspace.current().edit_btn.click()  # from the banner: the Setup step
    again = win.workspace.current()
    assert isinstance(again, NewRunPage) and again.name() == "Broken" and again._step == 1


def test_runs_open_in_a_preview_tab_unless_kept(window, qtbot, tmp_path: Path) -> None:
    from torchcor_studio.core.models import RunRecord, RunState, RunStatus

    win = window
    records = [RunRecord(id=n, name=n, kind="ep", machine_id="local", run_dir=str(_fake_run(tmp_path, n)),
                         created_at=time.time(), status=RunStatus(state=RunState.FAILED)) for n in ("one", "two")]
    for record in records:
        win.ctx.history.upsert(record)
    win.show_run(records[0], tab="Overview")
    preview = win.workspace.current()
    assert win.workspace.is_preview(preview) and win.pages.tabText(win.pages.indexOf(preview)) == "one"
    win.show_run(records[1], tab="Overview")  # browsing reuses the preview tab
    assert win.workspace.current() is preview and len(win.workspace.pages("run")) == 1
    assert win.pages.tabText(win.pages.indexOf(preview)) == "two" and preview.record.key == records[1].key
    win.show_run(records[0], tab="Overview", keep=True)  # double-click: its own tab
    assert len(win.workspace.pages("run")) == 2 and not win.workspace.is_preview(win.workspace.current())


def test_a_launched_draft_becomes_the_run_tab(window, qtbot, tmp_path: Path) -> None:
    from torchcor_studio.core.models import RunRecord, RunState, RunStatus

    win = window
    draft = win.new_run
    record = RunRecord(id="launched", name="Atrial pacing", kind="ep", machine_id="local",
                       run_dir=str(_fake_run(tmp_path, "launched")), created_at=time.time(),
                       status=RunStatus(state=RunState.FAILED))
    win.ctx.history.upsert(record)
    index = win.pages.indexOf(draft)
    draft.run_started.emit(record)
    page = win.pages.widget(index)
    assert win.workspace.kind(page) == "run" and page.record.key == record.key
    assert win.pages.tabText(index) == "Atrial pacing" and not win.workspace.pages("draft")
    assert not any(row.startswith("draft:") for row in _sidebar_rows(win))
    steps = page.steps  # Compute and Simulation done, the run failed, no results
    assert (steps._current, steps._reachable, steps._tone, steps._mark) == (2, 2, "danger", "x")


def test_a_runs_page_goes_through_the_steps(window, qtbot, tmp_path: Path) -> None:
    """Compute, Simulation, Run and Result are its navigation; it shows the results when done."""
    from torchcor_studio.core.models import RunRecord, RunState, RunStatus
    from torchcor_studio.ui.pages.run_page import VIEWS

    win = window
    record = RunRecord(id="r", name="Slab", kind="ep", machine_id="local", run_dir=str(_fake_run(tmp_path, "r")),
                       created_at=time.time(), status=RunStatus(state=RunState.RUNNING, started_at=time.time()),
                       manifest={"input_dir": "/data/ep/slab", "output_dir": "/runs", "python": "/py/bin/python"})
    win.ctx.history.upsert(record)
    win.show_run(record, keep=True)
    page = win.workspace.current()

    def tabs() -> list[str]:
        return [page.tabs.tabText(i) for i in range(page.tabs.count()) if page.tabs.isTabVisible(i)]

    assert tabs() == ["Overview", "Log"] and page.steps._selected == 2 and page.steps._reachable == 2
    page.steps.step_clicked.emit(0)  # Compute: where it runs
    assert tabs() == ["Machine"] and VIEWS[page.stack.currentIndex()] == "Machine"
    assert "/py/bin/python" in [w.text() for w in page._machine_host.findChildren(QLabel)]
    page.steps.step_clicked.emit(1)  # Simulation: its set-up
    assert tabs() == ["Parameters", "Script"]
    assert "/data/ep/slab" in [w.text() for w in page._params_host.findChildren(QLabel)]
    page.show_view("Log")
    page.steps.step_clicked.emit(1)
    page.steps.step_clicked.emit(2)  # back to the view last shown in the Run step
    assert VIEWS[page.stack.currentIndex()] == "Log"

    record.status = RunStatus(state=RunState.COMPLETED, started_at=time.time() - 5, ended_at=time.time())
    win.ctx.run_updated.emit(record)  # done while its run was shown: on to the results
    assert tabs() == ["Results", "Files"] and page.steps._selected == 3
    assert page.steps._current == 4 and page.steps._reachable >= 3  # every step done, all reachable
    page.steps.step_clicked.emit(0)
    assert tabs() == ["Machine"]


def test_set_up_has_two_steps_and_runs_from_the_setup_step(window, qtbot) -> None:
    from torchcor_studio.ui.pages.common import SIMULATION_STEPS

    draft = window.new_run
    assert draft.stack.count() == 2 and SIMULATION_STEPS == ("Compute", "Setup", "Run", "Result")
    assert draft.next_btn.isVisible() and not draft.run_btn.isVisible()
    draft.go(2)  # the Run step is reached with the Run button only
    assert draft._step == 1 and draft.run_btn.isVisible() and not draft.next_btn.isVisible()
    assert draft.preview_btn.isVisible() and draft.steps._reachable == 1  # Run and Result: not yet
    setup = draft.setup
    assert setup.isAncestorOf(setup.sweep) and setup.isAncestorOf(setup.demos)
    setup.sweep.toggle.setChecked(True)
    assert setup.sweep.target() is not None  # the parameters are read once the sweep is on
    setup.sweep.values_edit.setText("1, 2, 3")
    assert draft.run_btn.text() == "Run 3 simulations"
    setup.sweep.toggle.setChecked(False)
    assert draft.run_btn.text() == "Run simulation"


def test_long_names_are_shortened_on_the_tab_only(window) -> None:
    win = window
    page = win.new_run
    long = "Atrial pacing study with every pacing site of the left and right atria"
    page.set_name(long)
    shown = win.pages.tabText(win.pages.indexOf(page))
    assert shown.endswith("…") and len(shown) <= 30 and long.startswith(shown[:-1])
    assert win.workspace.title(page) == long and page.name() == long


def test_runs_are_renamed_in_place_too(window, qtbot, tmp_path: Path) -> None:
    from torchcor_studio.core.models import RunRecord, RunState, RunStatus

    win = window
    record = RunRecord(id="r", name="old name", kind="ep", machine_id="local",
                       run_dir=str(_fake_run(tmp_path, "r")), created_at=time.time(),
                       status=RunStatus(state=RunState.FAILED), manifest={"name": "old name"})
    win.ctx.history.upsert(record)
    win._refresh_history()
    win.show_run(record, tab="Overview")
    tab = win.workspace.current()
    model = win.sidebar.model
    model.setData(model.index_of(record.key), "new name")  # the sidebar entry
    assert win.ctx.record(record.key).name == "new name"
    assert win.pages.tabText(win.pages.indexOf(tab)) == "new name"  # and its tab
    win._tab_renamed(tab, "newer name")  # the tab
    assert win.ctx.record(record.key).name == "newer name"
    assert model.index_of(record.key).data() == "newer name"


def test_a_running_run_is_listed_by_its_stages_short_name() -> None:
    """The sidebar's line for a running run names its stage as the Run page's tracker does
    (the script's own label is a sentence), and a custom script's own stage by its label."""
    import time as clock

    from torchcor_studio.core.models import RunRecord, RunState, RunStatus
    from torchcor_studio.ui.sidebar import _meta

    def running(kind: str, stage: str, label: str, progress: float | None = 0.75) -> RunRecord:
        status = RunStatus(state=RunState.RUNNING, stage=stage, stage_label=label, progress=progress)
        return RunRecord(id="r", name="r", kind=kind, machine_id="local", run_dir="/runs/r",
                         created_at=clock.time(), status=status)

    em = running("em", "load", "Inflating to the start pressures")
    assert _meta(em, "GPU server") == "75% · Inflating · GPU server"
    assert _meta(em, "") == "75% · Inflating"  # without the machine, when the line is too long
    assert _meta(running("ecg", "data", "Preparing the heart-torso model", None), "GPU server") == \
        "Heart and torso · GPU server"
    assert _meta(running("script", "fit", "Fitting the model"), "") == "75% · Fitting the model"
