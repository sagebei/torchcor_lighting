"""New run → Compute step: the environment card and the responsive layouts it sits in."""

from __future__ import annotations

import pytest
from tests.fixtures.host import offscreen_windows_text

pytestmark = pytest.mark.gui


def _badges(card) -> list[str]:  # noqa: ANN001
    items = (card.badges.itemAt(i).widget() for i in range(card.badges.count()))
    return [w._text for w in items if w is not None]


def test_environment_card_explains_gpus_this_python_cannot_use(themed, qtbot) -> None:
    from torchcor_studio.core.environment import PythonCandidate
    from torchcor_studio.core.models import EnvironmentReport
    from torchcor_studio.ui.context import MachineState
    from torchcor_studio.ui.pages.new_run.compute_step import EnvironmentCard, devices_html

    card = EnvironmentCard()
    qtbot.addWidget(card)
    card.show()
    system = EnvironmentReport(python="/usr/bin/python3", python_version="3.8.10", hostname="gpu-box",
                               nvidia_gpus=2)
    tc = PythonCandidate(path="/home/u/miniforge3/envs/tc/bin/python", version="3.11.6",
                         prefix="/home/u/miniforge3/envs/tc", has_torch=True, torch_version="2.7.1", cuda="12.8")
    card.show_state(MachineState.CONNECTED, "", system, tc)
    assert "too old for TorchCor" in card.message.text() and "2 NVIDIA GPUs" in card.message.text()
    assert "2 GPUs not usable" in _badges(card)
    assert "not usable" in devices_html(system)
    assert card.suggestion.isVisible() and "<b>tc</b>" in card.suggestion.text.text()
    with qtbot.waitSignal(card.use_python_requested) as signal:
        card.suggestion.button.click()
    assert signal.args == [tc.path]

    cuda = EnvironmentReport(python=tc.path, python_version="3.11.6", torch_version="2.7.1",
                             cuda_available=True, cuda_version="12.8", cuda_device_count=2, nvidia_gpus=2)
    card.show_state(MachineState.CONNECTED, "", cuda, None)
    assert not card.suggestion.isVisible()
    assert card.message.text() == "PyTorch can use the 2 GPUs. Install TorchCor to start simulating."
    assert "CUDA 12.8 · 2 GPUs" in _badges(card) and card.install_btn.isVisible()
    assert devices_html(cuda) == " · 2 GPUs"

    outdated = EnvironmentReport(python="/v/bin/python", python_version="3.11.10", torch_version="2.7.1",
                                 cuda_available=True, cuda_version="12.6", cuda_device_count=2, nvidia_gpus=2,
                                 torchcor_version="0.1.6")
    card.show_state(MachineState.CONNECTED, "", outdated, None)
    assert "TorchCor 0.1.6 too old" in _badges(card) and card.install_btn.text() == "Update TorchCor"
    assert card.message.text().startswith("TorchCor 0.1.6 is older than this app supports (1.9 or newer)")


@pytest.mark.parametrize("kind", ["sticky", pytest.param("grid", marks=offscreen_windows_text)])
def test_responsive_layouts_follow_children_that_grow(themed, qtbot, kind: str) -> None:
    """A card whose wrapped text grows must not stay clipped at its first height."""
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    from torchcor_studio.ui.widgets.layout import ResponsiveGrid, StickyRow

    host = QWidget()
    qtbot.addWidget(host)
    outer = QVBoxLayout(host)
    cards, texts = [], []
    for name in ("a", "b"):
        card = QWidget()
        text = QLabel(name)
        text.setWordWrap(True)
        QVBoxLayout(card).addWidget(text)
        cards.append(card)
        texts.append(text)
    if kind == "sticky":  # a main column and a side panel that both grow
        container = StickyRow(cards[0], cards[1], side_width=200, breakpoint=100)
    else:
        container = ResponsiveGrid(min_item=200, max_columns=2)
        container.set_widgets(cards)
    outer.addWidget(container)
    outer.addStretch(1)
    host.resize(300, 900)  # one grid column, then two: the layouts re-add their widgets
    host.show()
    qtbot.wait(50)
    host.resize(640, 900)
    qtbot.wait(50)
    before = cards[0].height()
    texts[0].setText(" ".join(["wrapped"] * 120))  # many lines at this width
    qtbot.waitUntil(lambda: cards[0].height() >= texts[0].heightForWidth(texts[0].width()) > before + 40,
                    timeout=3000)
    if kind == "sticky":  # the side panel too, though it is not in the row's layout
        before = cards[1].height()
        texts[1].setText(" ".join(["wrapped"] * 60))
        qtbot.waitUntil(lambda: cards[1].height() >= texts[1].heightForWidth(texts[1].width()) > before + 40,
                        timeout=3000)


def test_page_one_shows_devices_before_the_environment(themed, qtbot) -> None:
    from torchcor_studio.core.settings import Store
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.pages.new_run.compute_step import ComputeStep

    ctx = AppContext(Store())
    step = ComputeStep(ctx)
    qtbot.addWidget(step)
    layout = step.layout()
    positions = [layout.indexOf(w) for w in (step.picker, step.devices, step.env)]
    assert -1 not in positions and positions == sorted(positions)  # machines, devices, environment
    ctx.shutdown()


def test_machine_cards_and_device_tiles_share_one_grid(themed, qtbot) -> None:
    from torchcor_studio.core.settings import Store
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.pages.new_run.compute_step import ComputeStep

    ctx = AppContext(Store())
    step = ComputeStep(ctx)
    qtbot.addWidget(step)
    machines, devices = step.picker._grid, step.devices.grid
    for width in (480, 760, 1000, 1180, 1600):  # same columns whatever the number of cards
        assert machines.columns_for(width) == devices.columns_for(width)
    assert devices.columns_for(1180) == 4 and devices.compact
    ctx.shutdown()


def _gpu(index: int, load: float, processes: int = 0):  # noqa: ANN202
    from torchcor_studio.core.models import GPUSample

    return GPUSample(index=index, name="GeForce RTX 3090", utilization=load, memory_total_mb=24576,
                     processes=processes)


def test_the_device_is_picked_on_page_one(themed, qtbot) -> None:
    """The least busy device is selected until the user picks one; another machine starts over."""
    from torchcor_studio.core.models import DeviceKind, GPUSample, RunRecord, RunState, RunStatus
    from torchcor_studio.ui.pages.new_run.device_picker import DevicePicker

    picker = DevicePicker()
    qtbot.addWidget(picker)
    cpu = GPUSample(index=0, name="CPU", kind=DeviceKind.CPU, utilization=5)
    with qtbot.waitSignal(picker.device_changed):
        picker.update_samples([_gpu(0, 90, 1), _gpu(1, 5), cpu])
    assert picker.selected_device().key == "cuda:1"
    picker.update_samples([_gpu(0, 3), _gpu(1, 80, 1), cpu])  # follows the usage while automatic
    assert picker.selected_device().key == "cuda:0"
    picker.grid.select("cuda:1")  # what a click on its tile does
    assert picker.selected_device().key == "cuda:1" and not picker.auto.isChecked()
    picker.update_samples([_gpu(0, 3), _gpu(1, 95, 1), cpu])
    assert picker.selected_device().key == "cuda:1"  # the user's choice stays

    run = RunRecord(id="a", name="Atrial pacing", kind="ep", machine_id="local", run_dir="/runs/a",
                    created_at=0.0, status=RunStatus(state=RunState.RUNNING))
    picker.set_active_runs([run])
    assert picker.queue_text.text() == "Queue behind “Atrial pacing” on this device"
    picker.hide()  # asked from the Setup step, while this page is hidden
    assert picker.queue_after() == ["/runs/a"]
    picker.queue.setChecked(False)
    assert picker.queue_after() == []

    with qtbot.waitSignal(picker.device_changed):
        picker.clear()  # another machine
    assert picker.selected_device() is None and picker.queue_after() == []
    picker.update_samples([cpu])
    assert picker.selected_device() is cpu and not picker.note.isHidden()  # the CPU-only note


def test_a_run_set_up_again_keeps_its_device(themed, qtbot) -> None:
    """A stopped run opened to run again asks for the device it ran on: selected once the
    machine lists it, though another is less busy, and then kept like a choice of the user's."""
    from torchcor_studio.ui.pages.new_run.device_picker import DevicePicker

    picker = DevicePicker()
    qtbot.addWidget(picker)
    picker.prefer("cuda:1")  # before the machine's devices are known
    picker.update_samples([_gpu(0, 5), _gpu(1, 60, 1)])
    assert picker.selected_device().key == "cuda:1" and not picker.auto.isChecked()
    picker.update_samples([_gpu(0, 3), _gpu(1, 90, 1)])
    assert picker.selected_device().key == "cuda:1"


def test_machine_cards_give_their_text_the_cards_width(themed, qtbot) -> None:
    """A card without chips (machines) must not squeeze its title into a narrow column."""
    from torchcor_studio.ui.widgets.steps import SelectCard

    card = SelectCard("laptop", "This computer", "workstation", compact=True)
    qtbot.addWidget(card)
    card.resize(280, 110)
    card.show()
    qtbot.wait(10)
    assert card.title.width() > 200 and card.title.x() < 60


def test_demo_data_downloads_into_the_folder_chosen(themed, qtbot, tmp_path, monkeypatch) -> None:
    """Download runs the runtime's fetch-data on the machine (faked here) and its progress
    shows; once done, the data folder is the machine's and the card shows it. A failure says
    why, and Download tries again."""
    import json
    import sys

    from torchcor_studio.core.settings import LOCAL_MACHINE_ID, Store
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.pages.new_run.compute_step import ComputeStep

    store = Store()
    local = store.machine(LOCAL_MACHINE_ID)
    local.python = sys.executable
    store.upsert_machine(local)
    ctx = AppContext(store)
    step = ComputeStep(ctx)
    qtbot.addWidget(step)
    step.set_machine(LOCAL_MACHINE_ID)
    qtbot.waitUntil(lambda: ctx.session(LOCAL_MACHINE_ID).connected, timeout=60000)
    card, machine = step.data, ctx.machine(LOCAL_MACHINE_ID)
    data = tmp_path / "torchcor_data"
    seen = []

    def fetch(argv, on_line, stop, env=None) -> int:  # noqa: ANN001
        assert argv[-2:] == ["fetch-data", str(tmp_path)] and argv[1].endswith("tcs_runtime.py")
        on_line(json.dumps({"progress": "download", "done": 50, "total": 200, "version": "data-v1.0"}))
        seen.append((ctx.session(LOCAL_MACHINE_ID).download.stage, ctx.session(LOCAL_MACHINE_ID).download.done))
        (data / "input" / "ep" / "ventricle").mkdir(parents=True)
        (data / ".version").write_text('{"version": "data-v1.0"}')
        on_line(json.dumps({"ok": True, "version": "data-v1.0", "path": str(data)}))
        return 0

    monkeypatch.setattr(machine, "stream", fetch)
    card.set_folder(str(tmp_path))
    assert card.target.text() == f"The data goes into {tmp_path}/torchcor_data."
    card.download_btn.click()
    sess = ctx.session(LOCAL_MACHINE_ID)
    qtbot.waitUntil(lambda: sess.download.stage == "done", timeout=10000)
    assert seen == [("download", 50)]
    assert sess.data_folders.root == str(data) and sess.data_folders.version == "data-v1.0"
    assert ctx.profile(LOCAL_MACHINE_ID).data_dir == str(data) and Store().machine(LOCAL_MACHINE_ID).data_dir == str(data)
    assert card.path_label.text() == str(data) and card.download_btn.text() == "Download the latest"
    assert "v1.0" in card.message.text() and not card.folder.isVisibleTo(step)

    def private(argv, on_line, stop, env=None) -> int:  # noqa: ANN001
        on_line("could not list the releases of sagebei/torchcor_lighting (HTTP 404; the repository must be public)")
        return 1

    monkeypatch.setattr(machine, "stream", private)
    card.download_btn.click()
    qtbot.waitUntil(lambda: sess.download.stage == "failed", timeout=10000)
    assert "the repository must be public" in card.message.text() and card.download_btn.isEnabled()
    assert sess.data_folders.root == str(data)  # the data already there is still used
    ctx.shutdown()
