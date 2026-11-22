"""Projects: folders in the sidebar that runs are filed in (membership in the manifest)."""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pytest

from torchcor_studio.core.models import Project, RunRecord, RunState, RunStatus
from torchcor_studio.core.runs.history import RunHistory


def _record(name: str, project: Project | None = None, *, state: RunState = RunState.COMPLETED,
            age_h: float = 1.0, machine: str = "local", root: str = "/runs") -> RunRecord:
    manifest = {"name": name, "notes": ""}
    if project is not None:
        manifest["project"] = project.ref()
    return RunRecord(id=name, name=name, kind="ep", machine_id=machine, run_dir=f"{root}/{name}",
                     created_at=_earlier_today(age_h), status=RunStatus(state=state), manifest=manifest)


def _earlier_today(hours: float) -> float:
    """A time that many hours ago, squeezed into the part of today gone by so the sidebar
    files it under Today at any time of day (just after midnight too); ages keep their order."""
    now = time.time()
    return now - (now - datetime.combine(date.today(), datetime.min.time()).timestamp()) * hours / 24


def test_history_keeps_projects_and_adopts_them_from_manifests(tmp_path: Path) -> None:
    history = RunHistory(tmp_path / "history.json")
    atria = history.add_project("Atria")
    history.add_project("benchmarks")
    history.rename_project(atria.id, "Atrial study")
    assert [p.name for p in history.projects()] == ["Atrial study", "benchmarks"]  # by name

    # runs filed elsewhere (another computer, or before a reinstall) bring their project back
    other = Project.new("From the server")
    history.replace_machine("gpu", [_record("r1", other, machine="gpu")], ["/runs"])
    assert history.project(other.id).name == "From the server"

    history.remove_project(atria.id)
    history.upsert(_record("r2", atria))  # a stale manifest does not resurrect a deleted project
    assert history.project(atria.id) is None and history.project_of(history.get("local:/runs/r2")) is None

    reloaded = RunHistory(tmp_path / "history.json")
    assert [p.name for p in reloaded.projects()] == ["benchmarks", "From the server"]  # case-insensitive
    reloaded.upsert(_record("r3", atria))
    assert reloaded.project(atria.id) is None


def test_a_launched_run_records_its_project(tmp_path: Path) -> None:
    from torchcor_studio.core.machines.local import LocalMachine
    from torchcor_studio.core.models import MachineProfile
    from torchcor_studio.core.runs import LaunchRequest, RunService
    from torchcor_studio.core.simulations import SimulationKind, get_spec

    spec = get_spec(SimulationKind.SCRIPT)
    machine = LocalMachine(MachineProfile(name="here", python=sys.executable))
    project = Project.new("Benchmarks")
    record = RunService().create(machine, LaunchRequest(
        spec=spec, params=spec.defaults(), name="filed", input_dir="", output_dir=str(tmp_path),
        device="cpu", device_name="CPU", project=project.ref()))
    manifest = json.loads((Path(record.run_dir) / "manifest.json").read_text())
    assert manifest["project"] == {"id": project.id, "name": "Benchmarks"} and record.project_id == project.id


# ============================================================================ sidebar
pytestmark_gui = pytest.mark.gui


@pytestmark_gui
def test_sidebar_lists_projects_as_folders(themed, qtbot) -> None:
    from torchcor_studio.ui.sidebar import RunListModel

    atria, empty = Project.new("Atria"), Project.new("Empty")
    records = [_record("a1", atria, age_h=1), _record("a2", atria, state=RunState.RUNNING, age_h=0.5),
               _record("loose", age_h=2)]
    model = RunListModel()

    def rows():  # noqa: ANN202
        out = []
        for i in range(model.rowCount()):
            idx = model.index(i)
            header, folder = idx.data(model.HeaderRole), idx.data(model.ProjectRole)
            rec = idx.data(model.RecordRole)
            out.append(f"#{header}" if header else f"[{folder.project.name} {folder.count}/{folder.running}]"
                       if folder else ("  " * idx.data(model.DepthRole)) + rec.name)
        return out

    model.set_records(records, "", {}, [atria, empty], set())
    assert rows() == ["#Projects", "[Atria 2/1]", "[Empty 0/0]", "#Today", "loose"]
    assert model.has_active()  # a collapsed project with a running run keeps the pulse going
    model.set_records(records, "", {}, [atria, empty], {atria.id})
    assert rows() == ["#Projects", "[Atria 2/1]", "  a2", "  a1", "[Empty 0/0]", "#Today", "loose"]
    model.set_records(records, "a1", {}, [atria, empty], set())  # search opens matching projects
    assert rows() == ["#Projects", "[Atria 2/1]", "  a1"]

    # dropping runs on a folder asks to file them there
    model.set_records(records, "", {}, [atria, empty], set())
    loose = model.index_of("local:/runs/loose")
    folder = model.index_of_project(empty.id)
    data = model.mimeData([loose])
    assert not model.canDropMimeData(data, None, -1, 0, loose)  # only folders accept runs
    with qtbot.waitSignal(model.runs_dropped) as dropped:
        model.dropMimeData(data, None, -1, 0, folder)
    assert dropped.args == [["local:/runs/loose"], empty.id]


@pytestmark_gui
def test_runs_are_filed_in_their_folder_on_disk(themed, qtbot, tmp_path: Path) -> None:
    from torchcor_studio.core.settings import LOCAL_MACHINE_ID, Store
    from torchcor_studio.ui.context import AppContext

    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    now = time.time()
    (run_dir / "manifest.json").write_text(json.dumps({"id": "run-1", "name": "run 1", "kind": "ep",
                                                       "created_at": now, "params": {}, "device": "cpu"}))
    (run_dir / "status.json").write_text(json.dumps({"state": "completed", "started_at": now - 5,
                                                     "ended_at": now, "updated_at": now}))
    store = Store()
    local = store.machine(LOCAL_MACHINE_ID)
    local.python = sys.executable
    local.output_dir = str(tmp_path / "runs")
    store.upsert_machine(local)
    ctx = AppContext(store)
    ctx.connect_machine(LOCAL_MACHINE_ID)
    qtbot.waitUntil(lambda: len(ctx.records()) == 1, timeout=60000)
    record = ctx.records()[0]

    project = ctx.create_project("Atrial study")
    assert ctx.create_project("atrial STUDY ") is project and len(ctx.projects()) == 1  # no twins
    ctx.move_to_project([record], project.id)
    on_disk = lambda: json.loads((run_dir / "manifest.json").read_text()).get("project")  # noqa: E731
    qtbot.waitUntil(lambda: on_disk() == project.ref(), timeout=10000)
    assert ctx.records()[0].project_id == project.id

    ctx.rename_project(project.id, "AF study")
    qtbot.waitUntil(lambda: (on_disk() or {}).get("name") == "AF study", timeout=10000)
    ctx.delete_project(project.id)  # the run is kept, unfiled
    qtbot.waitUntil(lambda: on_disk() is None, timeout=10000)
    assert ctx.projects() == [] and run_dir.exists()
    ctx.shutdown()


@pytestmark_gui
def test_a_new_simulation_starts_in_the_chosen_project(themed, qtbot) -> None:
    from torchcor_studio.core.settings import Store
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.pages.new_run.page import NewRunPage

    ctx = AppContext(Store())
    page = NewRunPage(ctx)
    qtbot.addWidget(page)
    project = ctx.create_project("Benchmarks")
    page.start_fresh(project_id=project.id)
    assert page.project_id() == project.id
    page.start_fresh()
    assert page.project_id() is None
    ctx.shutdown()
