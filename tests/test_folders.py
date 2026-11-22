"""New run → the input / output folders: on the Setup step, remembered per machine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui


def test_remembered_folders_are_most_recent_first() -> None:
    from torchcor_studio.core.settings import Settings

    s = Settings()
    for path in ("/a", "/b", "/a", ""):
        s.remember_output("m", path)
    assert s.recent_outputs == {"m": ["/a", "/b"]}
    assert s.recent_inputs == {}


def _context(*profiles):  # noqa: ANN002, ANN202
    from torchcor_studio.core.settings import LOCAL_MACHINE_ID, Store
    from torchcor_studio.ui.context import AppContext

    store = Store()
    local = store.machine(LOCAL_MACHINE_ID)
    local.python = sys.executable
    store.upsert_machine(local)
    for profile in profiles:
        store.upsert_machine(profile)
    return AppContext(store)


def test_output_folder_survives_switching_machines(themed, qtbot, tmp_path: Path) -> None:
    from torchcor_studio.core.models import MachineKind, MachineProfile
    from torchcor_studio.core.settings import LOCAL_MACHINE_ID, Store
    from torchcor_studio.ui.pages.new_run.folders import Folders

    ctx = _context(MachineProfile(name="Other", id="other", kind=MachineKind.LOCAL, python=sys.executable))
    folders = Folders(ctx)
    qtbot.addWidget(folders.input_slot)
    qtbot.addWidget(folders.output)
    folders.set_machine(LOCAL_MACHINE_ID)

    typed, browsed = str(tmp_path / "typed"), str(tmp_path / "browsed")
    folders.output.edit.setText(typed)  # typed only: no editingFinished before the switch
    folders.set_machine(LOCAL_MACHINE_ID)  # the same machine again keeps what is typed
    assert folders.output_dir() == typed
    folders.set_machine("other")
    assert folders.output_dir() == ""  # nothing chosen for that machine yet
    assert folders.input_slot.machine.text() == "on Other"
    folders.output.set_text(browsed, emit=True)  # as Browse… / Recent do
    folders.set_machine(LOCAL_MACHINE_ID)
    assert folders.output_dir() == typed
    folders.set_machine("other")
    assert folders.output_dir() == browsed

    reloaded = Store()  # and after a restart
    assert reloaded.settings.recent_outputs[LOCAL_MACHINE_ID][0] == typed
    assert reloaded.settings.recent_outputs["other"][0] == browsed
    ctx.shutdown()


def test_output_falls_back_to_where_runs_were_written(themed, qtbot) -> None:
    """Folders picked before outputs were remembered are only known as history roots."""
    from torchcor_studio.core.models import MachineKind, MachineProfile
    from torchcor_studio.ui.pages.new_run.folders import Folders

    ctx = _context(MachineProfile(name="Server", id="server", kind=MachineKind.LOCAL, python=sys.executable))
    ctx.history.add_root("server", "/runs/server")
    folders = Folders(ctx)
    qtbot.addWidget(folders.input_slot)
    qtbot.addWidget(folders.output)
    folders.set_machine("server")
    assert folders.output_dir() == "/runs/server"
    ctx.shutdown()


def test_input_is_inspected_on_the_machine_it_belongs_to(themed, qtbot, tmp_path: Path) -> None:
    from torchcor_studio.core.models import MachineKind, MachineProfile
    from torchcor_studio.core.settings import LOCAL_MACHINE_ID
    from torchcor_studio.ui.pages.new_run.folders import Folders

    mesh = tmp_path / "mesh"
    mesh.mkdir()
    for ext, text in (("pts", "1\n0 0 0\n"), ("elem", "1\nTt 0 0 0 0 1\n")):
        (mesh / f"heart.{ext}").write_text(text)
    ctx = _context(MachineProfile(name="Other", id="other", kind=MachineKind.LOCAL, python=sys.executable))
    folders = Folders(ctx)
    qtbot.addWidget(folders.input_slot)
    qtbot.addWidget(folders.output)
    seen = []
    folders.input_changed.connect(seen.append)
    ctx.connect_machine(LOCAL_MACHINE_ID)
    qtbot.waitUntil(lambda: ctx.machine(LOCAL_MACHINE_ID) is not None, timeout=60000)
    folders.set_machine(LOCAL_MACHINE_ID)
    folders.input.set_text(str(mesh), emit=True)
    qtbot.waitUntil(lambda: folders.input_info is not None and folders.input_info.has_mesh, timeout=60000)
    folders.set_machine("other")  # another machine: what was found no longer applies
    assert folders.input_info is None and seen[-1] is None
    ctx.shutdown()


def test_folders_are_chosen_on_the_setup_step(themed, qtbot, tmp_path: Path) -> None:
    from torchcor_studio.ui.pages.new_run.page import NewRunPage

    ctx = _context()
    local = ctx.profile("local")
    local.output_dir = ""  # no default: the page must ask for one
    ctx.store.upsert_machine(local)
    page = NewRunPage(ctx)
    qtbot.addWidget(page)
    page.resize(1300, 900)
    page.show()
    slot = page.folders.input_slot
    assert page.setup.isAncestorOf(slot) and not page.compute.isAncestorOf(slot)
    first = page.setup.form().cards()[0]
    assert first.isAncestorOf(slot) and first.isAncestorOf(page.folders.output)  # the first panel
    page.setup.set_kind("mechanics")  # its demos need no input folder: only the output is missing
    page.go(1)
    assert page._step == 1 and page.run_btn.isVisible() and not page.next_btn.isVisible()
    qtbot.waitUntil(lambda: page.hint.text() == "Choose an output folder.", timeout=10000)
    assert not page.run_btn.isEnabled()  # can't run without an output folder
    page.folders.output.set_text(str(tmp_path / "runs"), emit=True)
    qtbot.waitUntil(lambda: page.hint.text() != "Choose an output folder.", timeout=10000)
    ctx.shutdown()


def _data_root(tmp_path: Path) -> Path:
    """A demo data folder: ``input`` with one sub-folder per kind and its case folders, and
    the matching ``output`` folders."""
    root = tmp_path / "torchcor_data"
    for sub in ("ep/atrium", "ep/ventricle", "ecg/heart", "mechanics/ellipsoid", "notes"):
        (root / "input" / sub).mkdir(parents=True)
        (root / "output" / sub).mkdir(parents=True)
    (root / ".version").write_text('{"version": "data-v1.0"}')
    return root


def test_data_folder_is_found_on_the_machine(tmp_path: Path) -> None:
    from torchcor_studio.core.data_folders import find_data_folders
    from torchcor_studio.core.machines.local import LocalMachine
    from torchcor_studio.core.models import MachineProfile

    root = _data_root(tmp_path)
    data = find_data_folders(LocalMachine(MachineProfile(name="here", data_dir=str(root))))
    assert data is not None and data.kinds == {"ep", "ecg", "mechanics"}  # "notes" is no kind
    assert data.root == str(root) and data.version == "data-v1.0"
    assert data.folder("ep") == str(root / "input" / "ep") and data.folder("em") is None
    assert data.contains(str(root / "input" / "ep"), str(root / "input" / "ep" / "atrium"))
    assert not data.contains(str(root / "input" / "ep"), str(root / "input" / "ecg" / "heart"))
    assert not data.contains(str(root / "input" / "ep"), str(root / "input" / "epx"))
    assert data.paired_output(str(root / "input" / "ep" / "atrium")) == str(root / "output" / "ep" / "atrium")
    assert data.paired_output(str(root / "input")) == str(root / "output")
    assert data.paired_output(str(root / "output" / "ep")) is None and data.paired_output("/elsewhere") is None
    assert find_data_folders(LocalMachine(MachineProfile(name="empty", data_dir=str(tmp_path / "none")))) is None
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "ep").mkdir()  # kind folders without input/: not a data folder
    assert find_data_folders(LocalMachine(MachineProfile(name="old", data_dir=str(tmp_path / "old")))) is None


def test_data_folders_list_cases(tmp_path: Path) -> None:
    from torchcor_studio.core.data_folders import find_data_folders
    from torchcor_studio.core.machines.local import LocalMachine
    from torchcor_studio.core.models import MachineProfile

    root = _data_root(tmp_path)
    data = find_data_folders(LocalMachine(MachineProfile(name="here", data_dir=str(root))))
    assert data.cases == {"ep/atrium", "ep/ventricle", "ecg/heart", "mechanics/ellipsoid"}
    assert data.has_case("ep/ventricle") and data.has_case("ecg") and not data.has_case("em")
    assert data.has_case("ecg", ("heart",)) and not data.has_case("ecg", ("KCL_torso3", "heart"))
    assert data.case_path("ep/ventricle") == str(root / "input" / "ep" / "ventricle")
    assert data.output_path("ep/ventricle") == str(root / "output" / "ep" / "ventricle")


def test_demo_data_status() -> None:
    from torchcor_studio.core.data_folders import DataFolders, DataStatus, DemoData, demo_data
    from torchcor_studio.core.simulations import get_spec

    ep, ecg = get_spec("ep"), get_spec("ecg")
    data = DataFolders("/home/u/torchcor_data", frozenset({"ep"}), frozenset({"ep/ventricle"}))

    def status(demo, folders=data, connected=True):  # noqa: ANN001, ANN202
        return demo_data(demo, folders, connected)

    assert status(get_spec("mechanics").demos[0]) == DemoData(DataStatus.BUILT_IN, "")
    assert status(ep.demo("ventricle")) == DemoData(DataStatus.FOUND, "/home/u/torchcor_data/input/ep/ventricle",
                                                    "/home/u/torchcor_data/output/ep/ventricle")
    assert status(ep.demo("atrium")) == DemoData(DataStatus.MISSING, "/home/u/torchcor_data/input/ep/atrium")
    assert status(ep.demo("atrium"), None) == DemoData(DataStatus.MISSING, "~/torchcor_data/input/ep/atrium")
    assert status(ecg.demo("healthy")) == DemoData(DataStatus.MISSING, "/home/u/torchcor_data/input/ecg")
    assert status(ecg.demo("healthy"), None) == DemoData(DataStatus.MISSING, "~/torchcor_data/input/ecg")
    assert status(ep.demo("ventricle"), connected=False) == DemoData(
        DataStatus.UNKNOWN, "~/torchcor_data/input/ep/ventricle", "~/torchcor_data/output/ep/ventricle")
    healthy = ecg.demo("healthy")
    unprepared = DataFolders("/d", frozenset({"ecg"}), frozenset({"ecg/heart"}))  # not KCL torso 3
    assert status(healthy, unprepared) == DemoData(DataStatus.MISSING, "/d/input/ecg")
    prepared = DataFolders("/d", frozenset({"ecg"}), frozenset(f"ecg/{sub}" for sub in healthy.requires))
    assert status(healthy, prepared) == DemoData(DataStatus.FOUND, "/d/input/ecg", "/d/output/ecg")


def test_input_starts_in_the_folder_of_the_simulation_kind(themed, qtbot, tmp_path: Path) -> None:
    from torchcor_studio.core.settings import LOCAL_MACHINE_ID
    from torchcor_studio.ui.pages.new_run.folders import Folders

    root = _data_root(tmp_path)
    inputs, outputs = root / "input", root / "output"
    ctx = _context()
    local = ctx.profile(LOCAL_MACHINE_ID)
    local.data_dir = str(root)  # the machine's data folder
    ctx.store.upsert_machine(local)
    ctx.settings.recent_inputs[LOCAL_MACHINE_ID] = [str(inputs / "ecg" / "heart")]  # used before
    folders = Folders(ctx)
    qtbot.addWidget(folders.input_slot)
    qtbot.addWidget(folders.output)
    folders.set_kind("ep")
    folders.set_machine(LOCAL_MACHINE_ID)
    ctx.connect_machine(LOCAL_MACHINE_ID)  # as the Compute step does
    qtbot.waitUntil(lambda: folders.input_dir() == str(inputs / "ep"), timeout=60000)  # once connected
    assert folders.output_dir() == str(outputs / "ep")  # each input folder writes to its output folder
    folders.set_kind("mechanics")
    assert (folders.input_dir(), folders.output_dir()) == (str(inputs / "mechanics"), str(outputs / "mechanics"))
    folders.set_kind("ecg")
    assert folders.input_dir() == str(inputs / "ecg" / "heart")  # the case used last in that folder
    assert folders.output_dir() == str(outputs / "ecg" / "heart")
    folders.set_kind("script")  # no folder for this kind: keep what is there
    assert folders.input_dir() == str(inputs / "ecg" / "heart")

    folders.set_kind("ep")
    folders.input.set_text(str(inputs / "ep" / "atrium"), emit=True)  # the user's choice for EP
    qtbot.waitUntil(lambda: ctx.settings.kind_inputs.get(LOCAL_MACHINE_ID, {}).get("ep") == str(inputs / "ep" / "atrium"),
                    timeout=60000)
    assert folders.output_dir() == str(outputs / "ep" / "atrium")
    folders.set_kind("mechanics")
    folders.set_kind("ep")
    assert folders.input_dir() == str(inputs / "ep" / "atrium")
    assert "mechanics" not in ctx.settings.kind_inputs[LOCAL_MACHINE_ID]  # defaults are not remembered
    folders.output.set_text(str(tmp_path / "mine"), emit=True)  # an output the user chose stays
    folders.set_kind("mechanics")
    assert folders.output_dir() == str(tmp_path / "mine")
    folders.input.set_text(str(tmp_path), emit=True)  # an input outside the data
    assert folders.output_dir() == str(tmp_path / "mine")
    ctx.shutdown()


def test_closing_the_tab_while_its_input_is_checked(themed, qtbot, tmp_path: Path, caplog, monkeypatch) -> None:
    """The input folder is checked in the background: a tab closed meanwhile is left alone."""
    import time

    from torchcor_studio.core.settings import LOCAL_MACHINE_ID
    from torchcor_studio.ui.pages.new_run.folders import Folders

    ctx = _context()
    ctx.connect_machine(LOCAL_MACHINE_ID)
    qtbot.waitUntil(lambda: ctx.machine(LOCAL_MACHINE_ID) is not None, timeout=60000)
    machine = ctx.machine(LOCAL_MACHINE_ID)
    checked = []

    def slow_inspect(*_args, **_kwargs):  # noqa: ANN202
        time.sleep(0.3)
        checked.append(True)
        return {"path": str(tmp_path), "exists": True}

    monkeypatch.setattr(machine, "runtime", slow_inspect)
    folders = Folders(ctx)
    folders.set_machine(LOCAL_MACHINE_ID)
    folders.input.set_text(str(tmp_path), emit=True)
    folders.deleteLater()  # the tab closes while the folder is being checked
    qtbot.waitUntil(lambda: bool(checked), timeout=5000)
    qtbot.wait(200)  # the result is delivered (or dropped) on the UI thread
    assert "task callback failed" not in caplog.text
    ctx.shutdown()
