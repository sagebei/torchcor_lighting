"""Run page behaviour with real run folders on this computer."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from tests.fixtures.synthetic_bundles import write_ecg_bundle, write_mechanics_bundle

pytestmark = pytest.mark.gui


def _fake_run(root: Path, name: str, kind: str, writer) -> Path:  # noqa: ANN001
    run_dir = root / name
    (run_dir / "viz").mkdir(parents=True)
    writer(run_dir / "viz")
    now = time.time()
    (run_dir / "manifest.json").write_text(json.dumps(
        {"id": name, "name": name, "kind": kind, "created_at": now, "params": {}, "device": "cpu"}))
    (run_dir / "status.json").write_text(json.dumps(
        {"state": "completed", "started_at": now - 5, "ended_at": now, "updated_at": now, "progress": 1.0}))
    (run_dir / "run.py").write_text("print('done')\n")
    return run_dir


def _wait(qtbot, predicate, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "timed out"
        qtbot.wait(50)


def test_switching_completed_runs_reloads_results(themed, qtbot, tmp_path: Path) -> None:
    from torchcor_studio.core.settings import LOCAL_MACHINE_ID, Store
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.pages.run_page import RunPage

    runs = tmp_path / "runs"
    _fake_run(runs, "ecg-run", "ecg", write_ecg_bundle)
    _fake_run(runs, "mech-run", "mechanics", write_mechanics_bundle)
    store = Store()
    local = store.machine(LOCAL_MACHINE_ID)
    local.python = sys.executable
    local.output_dir = str(runs)
    store.upsert_machine(local)
    ctx = AppContext(store)
    ctx.connect_machine(LOCAL_MACHINE_ID)
    _wait(qtbot, lambda: len(ctx.records()) == 2)  # discovered by scanning the output folder
    records = {r.kind: r for r in ctx.records()}

    page = RunPage(ctx)
    qtbot.addWidget(page)
    page.resize(1300, 800)
    page.show()
    page.show_record(records["ecg"], tab="Results")
    _wait(qtbot, lambda: page._results_view is not None)
    first = page._results_view
    page.show_record(records["mechanics"], tab="Results")  # same tab index: must still load
    _wait(qtbot, lambda: page._results_view is not None and page._results_view is not first)
    assert page._view == "Results"
    ctx.shutdown()


def test_stage_labels_fit_the_tracker(themed, qtbot) -> None:
    """Every kind's stage labels show whole (no "…") in the Progress card of the narrowest
    window (its tracker is about 430 px wide; 400 here, for wider system fonts)."""
    from torchcor_studio.core.simulations import specs
    from torchcor_studio.ui.widgets.steps import StageTracker

    tracker = StageTracker()
    qtbot.addWidget(tracker)
    tracker.resize(400, tracker.height())
    for spec in specs():
        tracker.set_stages([(s.id, s.label) for s in spec.stages])
        shown = [text for _rect, text in tracker.labels()]
        assert shown == [s.label for s in spec.stages], spec.kind


def test_progress_says_what_the_stage_is_doing_while_it_runs(themed, qtbot) -> None:
    """The tracker names the stage, so the Progress card does not repeat it in bold; its
    message line is live detail, gone once the run has finished."""
    import time as clock

    from PySide6.QtWidgets import QLabel

    from torchcor_studio.core.models import RunRecord, RunState, RunStatus
    from torchcor_studio.core.settings import Store
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.pages.run_page import RunPage

    ctx = AppContext(Store())
    page = RunPage(ctx)
    qtbot.addWidget(page)

    def show(state: RunState, stage: str, message: str) -> list[str]:
        status = RunStatus(state=state, stage=stage, stage_label="Beating the heart", progress=0.4,
                           message=message)
        page.show_record(RunRecord(id="r", name="r", kind="em", machine_id="local", run_dir="/runs/r",
                                   created_at=clock.time(), status=status), tab="Overview")
        return [w.text() for w in page.progress.findChildren(QLabel) if w.text()]

    shown = show(RunState.RUNNING, "beats", "t = 400 / 1000 ms")
    assert "Beating the heart" not in shown and "Beating" not in shown
    assert page.progress.message.fullText() == "t = 400 / 1000 ms"
    show(RunState.COMPLETED, "done", "t = 1000 / 1000 ms")  # an older run's last message
    assert page.progress.message.fullText() == ""
    ctx.shutdown()


def test_the_overview_ends_with_the_latest_output(themed, qtbot) -> None:
    """The Run step's Overview: the outcome, the progress and device, then the latest output
    filling the rest of the page (no notes)."""
    from PySide6.QtWidgets import QLabel

    from torchcor_studio.core.settings import Store
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.pages.run_page import RunPage

    ctx = AppContext(Store())
    page = RunPage(ctx)
    qtbot.addWidget(page)
    titles = [w.text() for w in page.findChildren(QLabel) if w.property("role") == "h3"]
    assert "Notes" not in titles and "Latest output" in titles
    ctx.shutdown()
