"""The whole product flow in a real (offscreen) window: set up → run → watch → results.

Runs a small electrophysiology simulation on this computer's CPU with TorchCor, so it is
marked slow and skipped when TorchCor isn't importable.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gui]

SHOTS = os.environ.get("TCS_SCREENSHOTS")  # folder to save screenshots of each step


def _shot(widget, name: str) -> None:
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        widget.grab().save(str(Path(SHOTS) / f"{name}.png"))


def _wait(qtbot, predicate, timeout_s: float, step_ms: int = 100) -> None:
    deadline = time.time() + timeout_s
    while not predicate():
        assert time.time() < deadline, "timed out"
        qtbot.wait(step_ms)


@pytest.mark.skipif(importlib.util.find_spec("torchcor") is None, reason="TorchCor is not installed")
def test_new_run_to_results(themed, qtbot, tmp_path: Path) -> None:
    from torchcor_studio.core.settings import LOCAL_MACHINE_ID, Store
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.main_window import MainWindow

    store = Store()
    local = store.machine(LOCAL_MACHINE_ID)
    local.python = sys.executable
    local.output_dir = str(tmp_path / "runs")
    store.upsert_machine(local)
    ctx = AppContext(store)
    window = MainWindow(ctx)
    qtbot.addWidget(window)
    window.resize(1440, 920)
    window.show()
    ctx.reattach()

    page = window.new_run
    _wait(qtbot, page.compute.is_ready, 120)
    _wait(qtbot, lambda: bool(ctx.session(LOCAL_MACHINE_ID).samples), 30)
    qtbot.wait(1500)  # a few telemetry ticks for the sparklines
    _shot(window, "1-compute")

    page.go(1)
    sim = page.setup
    sim.set_kind("ep")
    params = sim.params()
    assert params is not None, sim.issues()
    # keep the run short on a CPU: shorten the simulated time if the model exposes it
    data = params.model_dump()
    for section in data.values():
        if isinstance(section, dict) and "duration" in section:
            section["duration"] = min(section["duration"], 30.0)
    sim.load("ep", data)
    assert not sim.has_errors(), sim.issues()
    _wait(qtbot, lambda: page.compute.devices.selected_device() is not None, 30)
    _wait(qtbot, page.run_btn.isEnabled, 30)  # the Run button is on the Setup step
    _shot(window, "2-simulation")
    started = []
    page.run_started.connect(started.append)
    page._launch()
    _wait(qtbot, lambda: bool(started), 60)
    record = started[0]
    assert window.pages.currentWidget() is window.run_page

    _wait(qtbot, lambda: (ctx.record(record.key) or record).status.state == "running", 60)
    qtbot.wait(4000)
    _shot(window, "4-running")
    _wait(qtbot, lambda: (ctx.record(record.key) or record).status.state.is_terminal, 900, step_ms=500)
    final = ctx.record(record.key)
    assert final.status.state == "completed", final.status.error

    assert window.run_page._view == "Results"  # shown as the run completed
    window.run_page.show_view("Results")
    _wait(qtbot, lambda: window.run_page._results_view is not None, 120)
    qtbot.wait(1500)
    _shot(window, "5-results")
    window.run_page.show_view("Overview")
    qtbot.wait(300)
    _shot(window, "6-overview-done")
    ctx.shutdown()
