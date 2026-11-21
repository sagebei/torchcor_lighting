"""Telemetry widgets and the machine, remote-browser and install dialogs (offscreen Qt).

Set ``TCS_SCREENSHOT_DIR`` to keep the rendered PNGs (they go to a temp folder otherwise).
"""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path

import keyring
import pytest
from keyring.backend import KeyringBackend
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QVBoxLayout, QWidget

from torchcor_studio.core import secrets
from torchcor_studio.core.machines.local import LocalMachine
from torchcor_studio.core.models import (
    AuthMethod,
    CommandResult,
    DeviceKind,
    EnvironmentReport,
    GPUSample,
    MachineKind,
    MachineProfile,
)
from torchcor_studio.ui.dialogs import InstallDialog, MachineDialog, RemoteBrowserDialog
from torchcor_studio.ui.dialogs.install_dialog import failure_hint
from torchcor_studio.ui.dialogs.machine_dialog import gpu_summary
from torchcor_studio.ui.dialogs.remote_browser import human_size, mesh_summary, relative_time
from torchcor_studio.ui.widgets.gpu import (
    DeviceList,
    GpuCard,
    GpuFeed,
    GpuGrid,
    device_caption,
    device_state,
    device_title,
    memory_text,
)

pytestmark = pytest.mark.gui


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self) -> None:
        super().__init__()
        self.data: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.data.get((service, username))

    def set_password(self, service, username, password):
        self.data[(service, username)] = password

    def delete_password(self, service, username):
        self.data.pop((service, username), None)


@pytest.fixture(autouse=True)
def memory_keyring():
    previous = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    yield backend
    keyring.set_keyring(previous)


@pytest.fixture
def themed(qapp):
    """The studio theme, applied once per session instead of once per test.

    Re-applying it emits ``theme().changed`` to every widget ever connected, and the
    design system's lambda connections (Badge, Sparkline, MeterBar, …) outlive the
    widgets deleted by earlier tests.
    """
    from torchcor_studio.ui.theme import theme

    if not qapp.styleSheet():
        theme().set_mode("dark")
    return qapp


@pytest.fixture
def shots(tmp_path) -> Path:
    folder = Path(os.environ.get("TCS_SCREENSHOT_DIR") or tmp_path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def devices(load: float = 87.0) -> list[GPUSample]:
    """Two A100s (one busy, one idle) and a CPU."""
    return [
        GPUSample(0, "NVIDIA A100-SXM4-80GB", DeviceKind.CUDA, "GPU-a", load, 51_200, 81_920, 67, 312, 400, 2),
        GPUSample(1, "NVIDIA A100-SXM4-80GB", DeviceKind.CUDA, "GPU-b", 1, 4, 81_920, 34, 61, 400, 0),
        GPUSample(0, "AMD EPYC 7763 · 128 cores", DeviceKind.CPU, "cpu", 23, 187_000, 515_000),
    ]


APPLE = GPUSample(0, "Apple M3 GPU · 10 cores", DeviceKind.APPLE, "apple-gpu", 12, 900, 16_384)


# ============================================================================ formatting
def test_formatting_helpers():
    busy, idle, cpu = devices()
    assert device_state(busy) == ("2 processes", "warning")
    assert device_state(idle) == ("Idle", "success")
    assert device_state(APPLE) == ("Not supported", "neutral")
    assert device_state(GPUSample(0, "x", utilization=55)) == ("Busy", "warning")
    assert device_state(GPUSample(0, "c", DeviceKind.CPU, utilization=30)) == ("Idle", "success")
    assert memory_text(busy) == "50.0 / 80.0 GB" and memory_text(GPUSample(0, "x")) == "—"
    assert device_title(busy) == "A100-SXM4-80GB" and device_caption(busy) == "cuda:0"
    assert device_title(cpu) == "AMD EPYC 7763" and device_caption(cpu) == "cpu · 128 cores"
    assert device_caption(APPLE) == "Metal · 10 cores"
    assert gpu_summary(["GPU 0: NVIDIA A100", "GPU 1: NVIDIA A100", "GPU 2: Tesla T4"]) == "2 × NVIDIA A100, Tesla T4"
    assert human_size(812) == "812 B" and human_size(12_700) == "12.4 KB" and human_size(3 * 2**20) == "3.0 MB"
    now = time.time()
    assert relative_time(now - 5, now) == "Just now" and relative_time(now - 600, now) == "10 min ago"
    assert relative_time(now - 7200, now) == "2 h ago" and relative_time(0) == ""
    assert failure_hint("error: externally-managed-environment").startswith("This Python is managed")
    assert failure_hint("all good") is None


# ============================================================================ telemetry widgets
def test_gpu_card_states(qtbot, themed):
    busy, idle, cpu = devices()
    card = GpuCard(busy, selectable=True)
    qtbot.addWidget(card)
    assert card.key == "cuda:0" and card._state.text() == "2 processes"
    assert not card._temp.isHidden() and not card._power.isHidden()
    card.set_sample(cpu)
    assert card._chips_row.isHidden() and card._index.isHidden() and card._mem_key._text == "RAM"
    card.set_sample(APPLE)
    assert card.toolTip() == "TorchCor needs an NVIDIA GPU (CUDA) or the CPU"
    assert card._content.graphicsEffect() is not None
    assert card.focusPolicy() == Qt.FocusPolicy.NoFocus
    card.set_sample(idle)
    assert card._content.graphicsEffect() is None and card.toolTip() == ""
    assert len(card.history.load) == 4
    for _ in range(100):
        card.set_sample(idle)
    assert len(card.history.load) == card.history.load.maxlen == 60


def test_gpu_grid_updates_and_selection(qtbot, themed):
    grid = GpuGrid(selectable=True)
    qtbot.addWidget(grid)
    grid.resize(1000, 600)
    assert grid._stack.currentWidget() is grid._loading
    grid.update_samples([*devices(), APPLE])
    assert [c.key for c in grid.cards()] == ["cuda:0", "cuda:1", "cpu:0", "apple:0"]
    assert grid._stack.currentWidget() is grid._page
    with qtbot.waitSignal(grid.selection_changed) as signal:
        qtbot.mouseClick(grid.cards()[1], Qt.MouseButton.LeftButton)
    assert signal.args[0].key == "cuda:1" and grid.selected().key == "cuda:1"
    assert grid.cards()[1].isSelected() and not grid.cards()[0].isSelected()
    grid.select("apple:0")  # not runnable: ignored
    assert grid.selected().key == "cuda:1"
    grid.update_samples(devices(load=10))  # Apple GPU gone, selection kept, history kept
    assert len(grid.cards()) == 3 and grid.selected().key == "cuda:1"
    assert len(grid.cards()[0].history.load) == 2 and grid.selected().utilization == 1
    with qtbot.waitSignal(grid.selection_changed) as signal:
        grid.update_samples(devices()[::2])  # the selected GPU disappears
    assert signal.args[0] is None and grid.selected() is None
    grid.update_samples([])
    assert grid._stack.currentWidget() is grid._empty


def test_gpu_grid_columns_and_messages(qtbot, themed):
    grid = GpuGrid()
    qtbot.addWidget(grid)
    widths = (200, 580, 880, 1180, 3000)
    assert [grid.columns_for(w) for w in widths] == [1, 2, 2, 2, 2]  # never one card across the row
    grid.set_error("Could not connect to gpu01")
    assert grid._stack.currentWidget() is grid._empty and grid._empty.title.text() == "Telemetry unavailable"
    grid.update_samples([*devices(), APPLE])
    assert [grid.columns_for(w) for w in widths] == [1, 2, 3, 4, 4]
    grid.update_samples(devices())  # three devices share the row, like three machine cards
    assert [grid.columns_for(w) for w in widths] == [1, 2, 3, 3, 3]
    grid.set_error("transient")  # ignored while cards are showing
    assert grid._stack.currentWidget() is grid._page
    grid.set_loading()
    assert grid._stack.currentWidget() is grid._loading


def test_device_list_selection(qtbot, themed):
    panel = DeviceList()
    qtbot.addWidget(panel)
    panel.update_samples([*devices(), APPLE])
    rows = list(panel._rows.values())
    assert panel._placeholder.isHidden() and len(rows) == 4
    with qtbot.waitSignal(panel.selection_changed) as signal:
        qtbot.mouseClick(rows[2], Qt.MouseButton.LeftButton)
    assert signal.args[0].kind == DeviceKind.CPU
    panel.select("apple:0")
    assert panel.selected().kind == DeviceKind.CPU and not rows[3].isEnabled()
    panel.update_samples([])
    assert not panel._placeholder.isHidden()


def test_gpu_feed_delivers_samples_on_the_ui_thread(qtbot, themed):
    feed = GpuFeed(LocalMachine(MachineProfile(name="local")), interval=0.2)
    grid = GpuGrid()
    qtbot.addWidget(grid)
    threads = []
    feed.samples.connect(lambda s: threads.append(threading.current_thread()))
    feed.samples.connect(grid.update_samples)
    feed.start()
    try:
        with qtbot.waitSignal(feed.samples, timeout=5000) as signal:
            pass
        assert any(s.kind == DeviceKind.CPU for s in signal.args[0])
        qtbot.waitUntil(lambda: bool(grid.cards()), timeout=5000)
        assert threads and all(t is threading.main_thread() for t in threads)
    finally:
        feed.stop()
        feed.monitor.stop()
    assert not feed.monitor.running


def test_render_gpu_grid(qtbot, themed, shots):
    root = QWidget()
    root.setObjectName("AppRoot")
    root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    lay = QVBoxLayout(root)
    lay.setContentsMargins(24, 24, 24, 24)
    grid = GpuGrid(selectable=True)
    lay.addWidget(grid)
    lay.addStretch(1)
    qtbot.addWidget(root)
    root.resize(1040, 380)
    for t in range(60):  # a minute of plausible telemetry
        busy = 74 + 12 * math.sin(t / 6) + 4 * math.sin(t * 1.7)
        grid.update_samples(devices(load=busy))
    grid.select("cuda:1")
    root.show()
    qtbot.waitExposed(root)
    qtbot.wait(50)  # let the grid settle into its final columns
    assert root.height() == 380 and {c.height() for c in grid.cards()} == {grid.cards()[0].sizeHint().height()}
    path = shots / "gpu_grid.png"
    assert root.grab().save(str(path)) and path.stat().st_size > 10_000


# ============================================================================ machine dialog
def test_machine_dialog_new_ssh_profile(qtbot, themed, memory_keyring):
    dialog = MachineDialog(None)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.host.setText("gpu01.example.org")
    dialog.port.setValue(2222)
    dialog.username.setText("alice")
    dialog.auth.set_value(AuthMethod.PASSWORD, animate=False)
    dialog._sync_auth()
    dialog.password.setText("hunter2")
    dialog.jump_host.setText("alice@bastion:22")
    dialog.input_dir.setText("~/meshes")
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    profile = dialog.result_profile()
    assert (profile.name, profile.host, profile.port, profile.username) == (
        "gpu01.example.org", "gpu01.example.org", 2222, "alice")
    assert profile.kind == MachineKind.SSH and profile.auth == AuthMethod.PASSWORD
    assert profile.jump_host == "alice@bastion:22" and profile.input_dir == "~/meshes"
    assert "hunter2" not in str(profile.to_dict())
    assert secrets.get_password(profile.id) == "hunter2"  # usable at once
    key = (secrets.SERVICE, f"{profile.id}:password")
    qtbot.waitUntil(lambda: memory_keyring.data.get(key) == "hunter2", timeout=3000)


def test_machine_dialog_validation_and_auth_rows(qtbot, themed):
    dialog = MachineDialog(None)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted and dialog.isVisible()
    assert dialog.host.property("invalid") is True and "host" in dialog._status.text()
    dialog.host.setText("h")
    dialog.auth.set_value(AuthMethod.KEY, animate=False)
    dialog._sync_auth()
    assert dialog.key_path.isVisible() and dialog.passphrase.isVisible() and not dialog.password.isVisible()
    dialog.accept()
    assert dialog.key_path.property("invalid") is True and dialog.isVisible()
    dialog.auth.set_value(AuthMethod.AGENT, animate=False)
    dialog._sync_auth()
    assert not dialog.key_path.isVisible() and not dialog.password.isVisible()
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_machine_dialog_imports_from_ssh_config(qtbot, themed):
    dialog = MachineDialog(None)
    qtbot.addWidget(dialog)
    imported = MachineProfile(name="lab", kind=MachineKind.SSH, host="lab.example.org", port=2200,
                              username="bob", auth=AuthMethod.KEY, key_path="/k/id", jump_host="j@b")
    dialog._fill_imports([imported])
    actions = [a for a in dialog._import_menu.actions() if a.isEnabled()]
    assert len(actions) == 1 and "lab.example.org" in actions[0].text()
    actions[0].trigger()
    profile = dialog.result_profile()
    assert (profile.name, profile.host, profile.port, profile.username, profile.auth, profile.key_path,
            profile.jump_host) == ("lab", "lab.example.org", 2200, "bob", AuthMethod.KEY, "/k/id", "j@b")
    assert profile.id != imported.id
    dialog._fill_imports([])
    assert not dialog._import_menu.actions()[0].isEnabled()


def test_machine_dialog_edit_keeps_id_and_cleans_old_secret(qtbot, themed, memory_keyring):
    original = MachineProfile(name="gpu", kind=MachineKind.SSH, host="h", auth=AuthMethod.PASSWORD)
    secrets.set_secret(original.id, secrets.PASSWORD, "old")
    dialog = MachineDialog(original)
    qtbot.addWidget(dialog)
    assert "keychain" in dialog.password.placeholderText()
    dialog.auth.set_value(AuthMethod.AGENT, animate=False)
    dialog._sync_auth()
    dialog.accept()
    assert dialog.result_profile().id == original.id and dialog.result_profile().auth == AuthMethod.AGENT
    qtbot.waitUntil(lambda: secrets.get_password(original.id) is None, timeout=3000)


def test_machine_dialog_for_this_computer(qtbot, themed):
    dialog = MachineDialog(MachineProfile(id="local", name="This computer", kind=MachineKind.LOCAL))
    qtbot.addWidget(dialog)
    assert not hasattr(dialog, "host") and dialog._test.text() == "Check environment"
    dialog.python.setText("/usr/bin/python3")
    dialog.accept()
    profile = dialog.result_profile()
    assert profile.id == "local" and profile.is_local and profile.python == "/usr/bin/python3"


def test_machine_dialog_checks_this_computer(qtbot, themed):
    dialog = MachineDialog(MachineProfile(id="local", name="This computer", kind=MachineKind.LOCAL))
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.mouseClick(dialog._test, Qt.MouseButton.LeftButton)
    assert not dialog._test.isEnabled()
    qtbot.waitUntil(lambda: dialog._test.isEnabled(), timeout=60_000)
    assert dialog._result_card.isVisible()
    assert "Auto-detect · found" in dialog.python.placeholderText()


def test_machine_dialog_shows_connection_failures(qtbot, themed):
    dialog = MachineDialog(None)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.host.setText("127.0.0.1")
    dialog.port.setValue(1)  # nothing listens there
    qtbot.mouseClick(dialog._test, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._test.isEnabled(), timeout=30_000)
    assert dialog._result_card.isVisible()
    texts = [w.text() for w in dialog._result_card.findChildren(type(dialog._status))]
    assert "Could not connect" in texts


def test_render_machine_dialog(qtbot, themed, shots):
    dialog = MachineDialog(None)
    qtbot.addWidget(dialog)
    dialog._apply_import(MachineProfile(name="gpu01", kind=MachineKind.SSH, host="gpu01.lab.example.org",
                                        username="alice", auth=AuthMethod.KEY, key_path="~/.ssh/id_ed25519"))
    dialog.output_dir.setText("~/torchcor-runs")
    dialog.show()
    qtbot.waitExposed(dialog)
    qtbot.wait(300)  # let the segmented control finish its animation
    path = shots / "machine_dialog.png"
    assert dialog.grab().save(str(path)) and path.stat().st_size > 10_000


# ============================================================================ remote browser
@pytest.fixture
def mesh_folder(tmp_path) -> Path:
    case = tmp_path / "cases" / "heart"
    (case / "pacing").mkdir(parents=True)
    (case / ".hidden").mkdir()
    for name in ("heart.pts", "heart.elem", "heart.lon", "apex.vtx", "notes.txt"):
        (case / name).write_text("1\n")
    return case


def test_mesh_summary(mesh_folder):
    machine = LocalMachine(MachineProfile(name="local"))
    assert mesh_summary(machine.listdir(str(mesh_folder))) == "openCARP mesh · heart · fibres · 1 .vtx"
    assert mesh_summary(machine.listdir(str(mesh_folder / "pacing"))) is None


def test_remote_browser_navigation(qtbot, themed, mesh_folder):
    machine = LocalMachine(MachineProfile(name="local"))
    dialog = RemoteBrowserDialog(machine, str(mesh_folder), mode="dir", title="Choose input")
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.current_path == str(mesh_folder), timeout=5000)
    names = [dialog._list.topLevelItem(i).text(0) for i in range(dialog._list.topLevelItemCount())]
    assert names == ["pacing", "apex.vtx", "heart.elem", "heart.lon", "heart.pts", "notes.txt"]
    assert dialog._mesh.isVisible() and "heart" in dialog._mesh.text()
    assert dialog._choose.text() == "Select “heart”"
    dialog._hidden.setChecked(True)
    assert dialog._list.topLevelItem(0).text(0) == ".hidden"
    dialog._hidden.setChecked(False)
    dialog._activate(dialog._list.topLevelItem(0))  # into "pacing"
    qtbot.waitUntil(lambda: dialog.current_path == str(mesh_folder / "pacing"), timeout=5000)
    assert not dialog._mesh.isVisible()
    dialog.go_up()
    qtbot.waitUntil(lambda: dialog.current_path == str(mesh_folder), timeout=5000)
    dialog._path_bar.path_requested.emit(str(mesh_folder.parent))
    qtbot.waitUntil(lambda: dialog.current_path == str(mesh_folder.parent), timeout=5000)
    dialog._accept_current()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.selected_path() == str(mesh_folder.parent)


def test_remote_browser_file_mode_errors_and_new_folder(qtbot, themed, mesh_folder, monkeypatch):
    machine = LocalMachine(MachineProfile(name="local"))
    dialog = RemoteBrowserDialog(machine, str(mesh_folder / "missing"), mode="file")
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog._error_row.isVisible(), timeout=5000)
    assert "Could not open" in dialog._error.text() and not dialog._choose.isEnabled()
    dialog.navigate(str(mesh_folder))
    qtbot.waitUntil(lambda: dialog.current_path == str(mesh_folder), timeout=5000)
    assert not dialog._error_row.isVisible()
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("results", True))
    dialog._make_folder()
    qtbot.waitUntil(lambda: dialog.current_path == str(mesh_folder / "results"), timeout=5000)
    assert (mesh_folder / "results").is_dir()
    dialog.navigate(str(mesh_folder))
    qtbot.waitUntil(lambda: dialog.current_path == str(mesh_folder), timeout=5000)
    item = next(dialog._list.topLevelItem(i) for i in range(dialog._list.topLevelItemCount())
                if dialog._list.topLevelItem(i).text(0) == "heart.pts")
    dialog._list.setCurrentItem(item)
    assert dialog._choose.isEnabled() and dialog._choose.text() == "Select file"
    dialog._activate(item)
    assert dialog.selected_path() == str(mesh_folder / "heart.pts")


def test_remote_browser_get_path(qtbot, themed, mesh_folder):
    machine = LocalMachine(MachineProfile(name="local"))

    def choose_when_ready() -> None:
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, RemoteBrowserDialog) and dialog.current_path:
            dialog._accept_current()
        else:
            QTimer.singleShot(50, choose_when_ready)

    QTimer.singleShot(50, choose_when_ready)
    assert RemoteBrowserDialog.get_path(machine, str(mesh_folder)) == str(mesh_folder)


# ============================================================================ install dialog
class FakeInstaller(LocalMachine):
    def __init__(self, lines: list[str], code: int = 0, block: bool = False) -> None:
        super().__init__(MachineProfile(name="gpu01"))
        self.lines, self.code, self.block = lines, code, block

    def run(self, argv, **kwargs):
        return CommandResult(0, "git version 2.43", "")

    def stream(self, argv, on_line, stop, *, env=None):
        for line in self.lines:
            on_line(line)
        if self.block:
            stop.wait(10)
            return -1 if stop.is_set() else 0
        return self.code

    def probe_environment(self, python=None):
        return EnvironmentReport(python=python or "", python_version="3.11.9", torch_version="2.5.1",
                                 cuda_available=True, cuda_version="12.4", torchcor_version="1.9.0")


def test_install_dialog_success(qtbot, themed, shots):
    machine = FakeInstaller(["Collecting torchcor", "Successfully installed torchcor-1.9.0"])
    dialog = InstallDialog(machine, "/env/bin/python", "torchcor @ git+https://github.com/sagebei/torchcor.git")
    qtbot.addWidget(dialog)
    with qtbot.waitSignal(dialog.install_finished, timeout=5000) as signal:
        dialog.show()
    assert signal.args == [0]
    qtbot.waitUntil(lambda: dialog.report is not None, timeout=5000)
    assert dialog.succeeded and dialog._status.text() == "TorchCor 1.9.0 is ready"
    assert "Successfully installed torchcor-1.9.0" in dialog._console.toPlainText()
    assert dialog._console.toPlainText().startswith("$ /env/bin/python -m pip install --upgrade")
    assert dialog.grab().save(str(shots / "install_dialog.png"))


def test_install_dialog_failure_hint(qtbot, themed):
    machine = FakeInstaller(["error: externally-managed-environment"], code=1)
    dialog = InstallDialog(machine, "/usr/bin/python3", "torchcor")
    qtbot.addWidget(dialog)
    with qtbot.waitSignal(dialog.install_finished, timeout=5000):
        dialog.show()
    assert dialog.exit_code == 1 and not dialog.succeeded
    assert "managed by the operating system" in dialog._detail.text()


def test_install_dialog_cancel(qtbot, themed):
    machine = FakeInstaller(["Collecting torch"], block=True)
    dialog = InstallDialog(machine, "/env/bin/python", "torchcor")
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: "Collecting torch" in dialog._console.toPlainText(), timeout=5000)
    with qtbot.waitSignal(dialog.install_finished, timeout=5000) as signal:
        dialog.cancel()
    assert signal.args == [-1] and dialog._status.text() == "Installation cancelled"
