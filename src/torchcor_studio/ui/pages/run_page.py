"""A single run: live progress, device usage and log while it runs; results afterwards."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QMenu,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QTabBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.core.models import GPUSample, RunRecord, RunState
from torchcor_studio.core.simulations import get_spec
from torchcor_studio.ui import icons
from torchcor_studio.ui.context import AppContext
from torchcor_studio.ui.format import duration, megabytes, number, size, timestamp
from torchcor_studio.ui.forms import ParamSummary
from torchcor_studio.ui.pages.common import SIMULATION_STEPS, scroll
from torchcor_studio.ui.tasks import run_async
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import (
    Badge,
    Button,
    Card,
    ElidedLabel,
    EmptyState,
    IconButton,
    KeyValueGrid,
    MeterBar,
    Sparkline,
    Spinner,
    StatTile,
    discard,
    label,
)
from torchcor_studio.ui.widgets.code import CodeView, LogView
from torchcor_studio.ui.widgets.steps import StageTracker, StepIndicator, _Disc

STATE_TEXT = {
    RunState.QUEUED: ("Starting", "neutral"),
    RunState.RUNNING: ("Running", "accent"),
    RunState.COMPLETED: ("Completed", "success"),
    RunState.FAILED: ("Failed", "danger"),
    RunState.CANCELLED: ("Stopped", "neutral"),
    RunState.LOST: ("Lost contact", "warning"),
}
#: A run's views (its stack's pages, in this order) and the step of the simulation each
#: belongs to: the steps are the page's navigation, and the current step's views its tabs.
VIEWS = ("Overview", "Results", "Log", "Parameters", "Script", "Files", "Machine")
VIEW_STEP = {"Machine": 0, "Parameters": 1, "Script": 1, "Overview": 2, "Log": 2, "Results": 3, "Files": 3}
SETUP_STEP, RUN_STEP, RESULT_STEP = 1, 2, 3
#: Runs whose Compute and Setup steps open them to change and run again.
EDITABLE = (RunState.CANCELLED, RunState.FAILED)
#: The Run step of a run that ended without completing: its tone and mark.
RUN_STEP_ENDED = {RunState.FAILED: ("danger", "x"), RunState.LOST: ("warning", "alert"),
                  RunState.CANCELLED: ("neutral", "stop")}


class DeviceLive(Card):
    """Live utilisation and memory of the device the run uses."""

    def __init__(self) -> None:
        super().__init__("Device", icon="gpu")
        self.name = label("—", "muted")
        self.addWidget(self.name)
        row = QHBoxLayout()
        self.util = label("—", "value")
        row.addWidget(self.util)
        row.addWidget(label("utilisation", "faint"), 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        self.extra = label("", "caption")
        row.addWidget(self.extra, 0, Qt.AlignmentFlag.AlignBottom)
        self.addLayout(row)
        self.spark = Sparkline(capacity=90, height=64)
        self.addWidget(self.spark)
        mem = QHBoxLayout()
        mem.addWidget(label("Memory", "faint"))
        mem.addStretch(1)
        self.mem_text = label("", "caption")
        mem.addWidget(self.mem_text)
        self.addLayout(mem)
        self.mem = MeterBar(0)
        self.addWidget(self.mem)

    def update_sample(self, s: GPUSample | None) -> None:
        if s is None:
            self.name.setText("Waiting for telemetry…")
            return
        self.name.setText(f"{s.name} · {s.device_string}")
        self.util.setText(f"{s.utilization:.0f}%")
        self.spark.push(s.utilization, s.memory_fraction * 100)
        self.mem.set_value(s.memory_fraction)
        self.mem_text.setText(f"{megabytes(s.memory_used_mb)} / {megabytes(s.memory_total_mb)}")
        bits = []
        if s.temperature_c is not None:
            bits.append(f"{s.temperature_c:.0f} °C")
        if s.power_w is not None:
            bits.append(f"{s.power_w:.0f} W")
        self.extra.setText(" · ".join(bits))

    def reset(self) -> None:
        self.spark.clear()
        self.util.setText("—")


class ProgressCard(Card):
    def __init__(self) -> None:
        super().__init__("Progress", icon="activity")
        self.tracker = StageTracker()
        self.addWidget(self.tracker)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.percent = label("—", "title")
        row.addWidget(self.percent)
        self.message = ElidedLabel("", "muted")  # what the stage is doing; the tracker names it
        row.addWidget(self.message, 1)
        self.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.addWidget(self.bar)
        tiles = QGridLayout()
        tiles.setHorizontalSpacing(12)
        self.elapsed = StatTile("Elapsed")
        self.eta = StatTile("Remaining")
        self.sim = StatTile("Simulated")
        for i, t in enumerate((self.elapsed, self.eta, self.sim)):
            tiles.addWidget(t, 0, i)
        self.addLayout(tiles)


class RunPage(QWidget):
    duplicate_requested = Signal(object)
    #: A stopped or failed run to change and run again, from its Compute or Setup step.
    edit_requested = Signal(object, int)  # RunRecord, step
    deleted = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.record: RunRecord | None = None
        #: The state last shown: polling updates the record in place, so a change of state is
        #: seen by comparing with this, not with the record.
        self._shown_state: RunState | None = None
        self._log_offset = 0
        self._loaded: set[str] = set()
        self._results_view: QWidget | None = None
        self._results_loading = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        head = QWidget()
        head.setObjectName("PageHeader")
        head.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        hl = QVBoxLayout(head)
        hl.setContentsMargins(28, 10, 24, 0)
        hl.setSpacing(4)
        # the steps where they were while the simulation was set up (its name is on its tab)
        self.steps = StepIndicator(SIMULATION_STEPS)
        self.steps.step_clicked.connect(self._step_clicked)
        hl.addWidget(self.steps, 0, Qt.AlignmentFlag.AlignHCenter)
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.tabs = QTabBar()  # the views of the step shown (see show_view)
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        self.tabs.currentChanged.connect(lambda i: i >= 0 and self._tab_changed(i))
        self._view = "Overview"  # the view shown
        bar.addWidget(self.tabs)
        bar.addStretch(1)
        self.state_badge = Badge("", "neutral", dot=True)
        bar.addWidget(self.state_badge)
        self.stop_btn = Button("Stop", "stop", "danger", size="sm", on_click=self._stop)
        self.dup_btn = Button("Duplicate", "duplicate", size="sm",
                              on_click=lambda: self.record and self.duplicate_requested.emit(self.record))
        self.folder_btn = IconButton("folder-open", "Show run folder", on_click=self._reveal)
        self.more_btn = IconButton("more", "More actions", on_click=self._more)
        for w in (self.stop_btn, self.dup_btn, self.folder_btn, self.more_btn):
            bar.addWidget(w)
        hl.addLayout(bar)
        lay.addWidget(head)
        self._step_views: dict[int, str] = {}  # the view last shown in each step

        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        self._build_overview()
        self._results_host = QWidget()
        self._results_lay = QVBoxLayout(self._results_host)
        self._results_lay.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self._results_host)
        log_page = QWidget()
        ll = QVBoxLayout(log_page)
        ll.setContentsMargins(28, 20, 28, 24)
        self.log = LogView()
        ll.addWidget(self.log)
        self.stack.addWidget(log_page)
        self._params_host = QWidget()
        self._params_lay = QVBoxLayout(self._params_host)
        self._params_lay.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self._params_host)
        self._machine_host = QWidget()
        self._machine_lay = QVBoxLayout(self._machine_host)
        self._machine_lay.setContentsMargins(0, 0, 0, 0)
        script_page = QWidget()
        sl = QVBoxLayout(script_page)
        sl.setContentsMargins(28, 20, 28, 24)
        sl.setSpacing(10)
        srow = QHBoxLayout()
        srow.addWidget(label("run.py — the exact script that ran. Re-run it with “python run.py” in the run folder.",
                             "caption"), 1)
        srow.addWidget(Button("Copy", "copy", size="sm", on_click=self._copy_script))
        srow.addWidget(Button("Save as…", "download", size="sm", on_click=self._save_script))
        sl.addLayout(srow)
        self.script = CodeView()
        sl.addWidget(self.script, 1)
        self.stack.addWidget(script_page)
        self._build_files()
        self.stack.addWidget(self._machine_host)  # after Files, as in VIEWS

        self._log_timer = QTimer(self)
        self._log_timer.setInterval(1000)
        self._log_timer.timeout.connect(self._poll_log)
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick)
        ctx.run_updated.connect(self._run_updated)
        ctx.telemetry.connect(self._telemetry)
        self.show_view("Overview")  # only the Run step's tabs until a run is shown

    # ------------------------------------------------------------------ construction
    def _build_overview(self) -> None:
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        self.outcome = Card()
        self.outcome_icon = _Disc("check-circle", "success")
        orow = QHBoxLayout()
        orow.setSpacing(14)
        orow.addWidget(self.outcome_icon, 0, Qt.AlignmentFlag.AlignTop)
        ocol = QVBoxLayout()
        ocol.setSpacing(4)
        self.outcome_title = label("", "h3")
        self.outcome_text = label("", "muted", wrap=True, selectable=True)
        ocol.addWidget(self.outcome_title)
        ocol.addWidget(self.outcome_text)
        orow.addLayout(ocol, 1)
        self.outcome_btn = Button("Open results", "arrow-right", "primary", on_click=lambda: self.show_step(RESULT_STEP))
        orow.addWidget(self.outcome_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self.edit_btn = Button("Edit and run again", "edit", "primary", on_click=lambda: self._edit(SETUP_STEP))
        self.edit_btn.setToolTip("Open its Compute and Setup steps to change them, then run it again here.")
        orow.addWidget(self.edit_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self.outcome.addLayout(orow)
        lay.addWidget(self.outcome)

        self.summary_tiles = QGridLayout()
        self.summary_tiles.setHorizontalSpacing(12)
        self.summary_tiles.setVerticalSpacing(12)
        lay.addLayout(self.summary_tiles)

        row = QHBoxLayout()
        row.setSpacing(16)
        self.progress = ProgressCard()
        row.addWidget(self.progress, 3)
        self.device = DeviceLive()
        row.addWidget(self.device, 2)
        lay.addLayout(row)
        tail = Card("Latest output", icon="terminal")  # the rest of the page's height
        self.tail = LogView(toolbar=False)
        self.tail.setMinimumHeight(220)
        tail.addWidget(self.tail)
        tail.add_action(Button("Full log", "arrow-right", "ghost", size="sm", on_click=lambda: self.show_view("Log")))
        lay.addWidget(tail, 1)
        self.stack.addWidget(scroll(content))

    def _build_files(self) -> None:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(28, 20, 28, 24)
        lay.setSpacing(10)
        bar = QHBoxLayout()
        self.files_path = ElidedLabel("", "mono")
        bar.addWidget(self.files_path, 1)
        bar.addWidget(Button("Refresh", "refresh", "ghost", size="sm", on_click=self._load_files))
        self.download_btn = Button("Download…", "download", size="sm", on_click=self._download_selected)
        bar.addWidget(self.download_btn)
        lay.addLayout(bar)
        self.files = QTreeWidget()
        self.files.setColumnCount(2)
        self.files.setHeaderLabels(["Name", "Size"])
        self.files.setRootIsDecorated(True)
        self.files.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.files.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.files.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.files.itemDoubleClicked.connect(self._open_file)
        lay.addWidget(self.files, 1)
        self.stack.addWidget(page)

    # ------------------------------------------------------------------ public
    def show_record(self, record: RunRecord, tab: str | None = None) -> None:
        """Show a run, at ``tab`` (one of ``VIEWS``): by default its results once it has
        completed, else its progress."""
        same = self.record is not None and record.key == self.record.key
        self.record = record
        if not same:
            self._reset()
        self._render_record()
        self._render_status()
        if tab is None and not same:
            tab = "Results" if record.status.state == RunState.COMPLETED else "Overview"
        self.show_view(tab or self._view)

    def _step_clicked(self, step: int) -> None:
        """A step clicked: its views, or for a run stopped or failed, its Compute or Setup
        step opened to change and run again."""
        if step <= SETUP_STEP and self.record is not None and self.record.status.state in EDITABLE:
            self._edit(step)
        else:
            self.show_step(step)

    def _edit(self, step: int) -> None:
        if self.record is not None:
            self.edit_requested.emit(self.record, step)

    def show_step(self, step: int) -> None:
        """Show a step of the simulation: the view last shown in it, else its first."""
        self.show_view(self._step_views.get(step) or next(v for v in VIEWS if VIEW_STEP[v] == step))

    def show_view(self, view: str) -> None:
        """Show one of ``VIEWS``; its step's views become the tabs."""
        views = [v for v in VIEWS if VIEW_STEP[v] == VIEW_STEP[view]]
        self.tabs.blockSignals(True)  # the view is loaded below (Qt is silent for the same index)
        if views != self._tab_views():
            while self.tabs.count():
                self.tabs.removeTab(0)
            for v in views:
                self.tabs.addTab(v)
        self.tabs.setCurrentIndex(views.index(view))
        self.tabs.blockSignals(False)
        self._tab_changed(self.tabs.currentIndex())

    def _tab_views(self) -> list[str]:
        return [self.tabs.tabText(i) for i in range(self.tabs.count())]
        self._sync_timers()

    def deactivate(self) -> None:
        self._log_timer.stop()
        self._clock.stop()
        if self._results_view is not None and hasattr(self._results_view, "set_active"):
            self._results_view.set_active(False)

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        self._sync_timers()
        if self.record is not None:
            self.ctx.watch_telemetry(self.record.machine_id)
            self._watching = self.record.machine_id
            if self._view == "Results" and self._results_view is not None and hasattr(self._results_view, "set_active"):
                self._results_view.set_active(True)

    def hideEvent(self, e) -> None:  # noqa: N802
        super().hideEvent(e)
        self.deactivate()
        watching = getattr(self, "_watching", None)
        if watching:
            self.ctx.unwatch_telemetry(watching)
            self._watching = None

    # ------------------------------------------------------------------ rendering
    def _reset(self) -> None:
        self._log_offset = 0
        self._loaded.clear()
        self.log.clear()
        self.tail.clear()
        self.device.reset()
        self.script.setPlainText("")
        self.files.clear()
        self._release_results()
        _clear(self._params_lay)
        _clear(self.summary_tiles)
        spec_stages = []
        with contextlib.suppress(KeyError, ValueError):
            spec_stages = [(s.id, s.label) for s in get_spec(self.record.kind).stages]
        self.progress.tracker.set_stages(spec_stages)
        watching = getattr(self, "_watching", None)
        if self.isVisible() and watching != self.record.machine_id:
            if watching:
                self.ctx.unwatch_telemetry(watching)
            self.ctx.watch_telemetry(self.record.machine_id)
            self._watching = self.record.machine_id

    def _render_record(self) -> None:
        r = self.record
        profile = self.ctx.profile(r.machine_id)
        self.folder_btn.setVisible(profile is not None and profile.is_local)

    def _render_status(self) -> None:
        r = self.record
        st = r.status
        text, tone = STATE_TEXT.get(st.state, ("Unknown", "neutral"))
        if st.state == RunState.QUEUED and st.stage == "waiting":
            text, tone = "In queue", "info"
        self.state_badge.set(text, tone)
        self._render_steps()
        self.stop_btn.setVisible(st.state.is_active)
        self.progress.tracker.set_state(st.stage or "", str(st.state))
        frac = st.progress if st.progress is not None else (1.0 if st.state == RunState.COMPLETED else None)
        self.progress.percent.setText(f"{frac * 100:.0f}%" if frac is not None else "—")
        self.progress.bar.setRange(0, 1000 if frac is not None or not st.state.is_active else 0)
        self.progress.bar.setValue(int((frac or 0) * 1000))
        self.progress.bar.setProperty("tone", {RunState.COMPLETED: "success", RunState.FAILED: "danger",
                                               RunState.CANCELLED: "muted", RunState.LOST: "muted"}.get(st.state, ""))
        self.progress.bar.style().polish(self.progress.bar)
        # live detail only: a finished run's last message is history (the banner says how it ended)
        self.progress.message.setText(st.message or "" if st.state.is_active else "")
        self._tick()
        sim_time = st.sim_time if st.sim_time is not None else (st.sim_total if st.state == RunState.COMPLETED else None)
        if sim_time is not None and st.sim_total:
            self.progress.sim.set(f"{sim_time:g}", f"/ {st.sim_total:g} ms")
        else:
            self.progress.sim.set("—", "")
        # outcome banner
        show = st.state.is_terminal
        self.outcome.setVisible(show)
        self.edit_btn.setVisible(st.state in EDITABLE)
        if st.state == RunState.COMPLETED:
            self.outcome_icon.set("check-circle", "success")
            self.outcome_title.setText("Simulation completed")
            took = duration(st.elapsed_s)
            self.outcome_text.setText(f"Finished in {took}. Explore the results, or duplicate this run to try other parameters.")
            self.outcome_btn.setVisible(True)
            self._load_summary_tiles()
        elif st.state == RunState.FAILED:
            self.outcome_icon.set("x-circle", "danger")
            self.outcome_title.setText("Simulation failed")
            self.outcome_text.setText(st.error or "The script exited with an error — see the log for details.")
            self.outcome_btn.setVisible(False)
        elif st.state == RunState.CANCELLED:
            self.outcome_icon.set("stop", "neutral")
            self.outcome_title.setText("Simulation stopped")
            self.outcome_text.setText("The run was stopped before it finished. Change its Compute or Setup "
                                      "step and run it again.")
            self.outcome_btn.setVisible(False)
        elif st.state == RunState.LOST:
            self.outcome_icon.set("alert", "warning")
            self.outcome_title.setText("Lost contact with the run")
            self.outcome_text.setText(st.error or "The run stopped reporting. The machine may have restarted.")
            self.outcome_btn.setVisible(False)

    def _render_steps(self) -> None:
        """Set-up done; the run under way, done or ended; the results once it completed."""
        state = self._shown_state = self.record.status.state
        shown = VIEW_STEP[self._view]
        if state == RunState.COMPLETED:
            self.steps.set_current(len(SIMULATION_STEPS), reachable=RESULT_STEP, selected=shown)
        else:
            tone, mark = RUN_STEP_ENDED.get(state, ("accent", None))
            self.steps.set_current(RUN_STEP, reachable=RUN_STEP, tone=tone, mark=mark, selected=shown)

    def _tick(self) -> None:
        if self.record is None:
            return
        st = self.record.status
        self.progress.elapsed.set(duration(st.elapsed_s) or "—")
        if st.state == RunState.RUNNING and st.remaining_s is not None:
            self.progress.eta.set(duration(st.remaining_s) or "<1s")
        elif st.state == RunState.COMPLETED:
            self.progress.eta.set("Done")
        else:
            self.progress.eta.set("—")

    def _sync_timers(self) -> None:
        active = self.record is not None and self.record.status.state.is_active and self.isVisible()
        if active:
            self._log_timer.start()
            self._clock.start()
            self._poll_log()
        else:
            self._log_timer.stop()
            self._clock.stop()

    # ------------------------------------------------------------------ live updates
    def _run_updated(self, record: RunRecord) -> None:
        if self.record is None or record.key != self.record.key:
            return
        was_active = self._shown_state is not None and self._shown_state.is_active
        self.record = record
        self._render_status()
        if was_active and record.status.state.is_terminal:
            self._poll_log()  # final lines
            self._loaded.difference_update({"Files", "Machine"})
            if record.status.state == RunState.COMPLETED and VIEW_STEP[self._view] == RUN_STEP:
                self.show_step(RESULT_STEP)  # done: on to the results
        self._sync_timers()

    def _telemetry(self, machine_id: str, samples) -> None:  # noqa: ANN001
        if self.record is None or machine_id != self.record.machine_id:
            return
        dev = self.record.device
        match = None
        for s in samples:
            if s.device_string == dev and s.runnable:
                match = s
                break
        self.device.update_sample(match)

    def _poll_log(self) -> None:
        record = self.record
        machine = self.ctx.machine(record.machine_id) if record else None
        if machine is None or getattr(self, "_log_busy", False):
            return
        self._log_busy = True

        def done(result) -> None:  # noqa: ANN001
            self._log_busy = False
            if self.record is None or self.record.key != record.key:
                return
            text, offset = result
            if text:
                self._log_offset = offset
                self.log.append(text)
                self.tail.append(text)

        def failed(_e) -> None:  # noqa: ANN001
            self._log_busy = False

        run_async(self.ctx.runs.read_log, machine, record, self._log_offset, on_done=done, on_error=failed, owner=self)

    # ------------------------------------------------------------------ tabs
    def _tab_changed(self, index: int) -> None:
        name = self._view = self.tabs.tabText(index)
        self.stack.setCurrentIndex(VIEWS.index(name))
        self._step_views[VIEW_STEP[name]] = name
        if self._results_view is not None and hasattr(self._results_view, "set_active"):
            self._results_view.set_active(name == "Results")
        if self.record is None:
            return
        self._render_steps()
        if name == "Results":
            self._load_results()
        elif name == "Log" and not self.record.status.state.is_active and "Log" not in self._loaded:
            self._loaded.add("Log")
            self._poll_log()
        elif name == "Parameters" and "Parameters" not in self._loaded:
            self._loaded.add("Parameters")
            self._load_params()
        elif name == "Script" and "Script" not in self._loaded:
            self._loaded.add("Script")
            self._load_script()
        elif name == "Files" and "Files" not in self._loaded:
            self._loaded.add("Files")
            self._load_files()
        elif name == "Machine" and "Machine" not in self._loaded:
            self._loaded.add("Machine")
            self._load_machine()
        if name == "Overview" and "Log" not in self._loaded and not self.record.status.state.is_active:
            self._loaded.add("Log")
            self._poll_log()

    def _load_params(self) -> None:
        """The Setup step: what was simulated, from which folders, with which parameters."""
        _clear(self._params_lay)
        r = self.record
        manifest = r.manifest or {}
        try:
            spec = get_spec(r.kind)
        except (KeyError, ValueError):
            spec = None
        project = self.ctx.project(r.project_id)
        sweep = manifest.get("sweep") or {}
        rows = [("Simulation", " · ".join(t for t in (spec.title if spec else r.kind, r.variant) if t)),
                ("Input", manifest.get("input_dir") or "None (built-in geometry)"),
                ("Output", manifest.get("output_dir") or "—")]
        if project is not None:
            rows.append(("Project", project.name))
        if sweep:
            rows.append(("Sweep", f"{sweep.get('parameter')} = {sweep.get('value')} {sweep.get('unit') or ''}".strip()))
        content = QWidget()
        col = QVBoxLayout(content)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(16)
        setup = Card("Set-up", icon="sliders")
        setup.addWidget(KeyValueGrid(rows))
        col.addWidget(setup)
        col.addWidget(ParamSummary(spec.params_model if spec else None, manifest.get("params") or {}))
        col.addStretch(1)
        self._params_lay.addWidget(scroll(content, max_width=980))

    def _load_machine(self) -> None:
        """The Compute step: where the run ran, with what, and when."""
        _clear(self._machine_lay)
        r, st = self.record, self.record.status
        manifest = r.manifest or {}
        profile = self.ctx.profile(r.machine_id)
        name = profile.name if profile else (manifest.get("machine") or {}).get("name") or r.machine_id
        address = "" if profile is None or profile.is_local else profile.address
        rows = [("Machine", f"{name} ({address})" if address else name),
                ("Device", f"{r.device_name} ({r.device})" if r.device_name else r.device),
                ("Python", manifest.get("python") or "—"),
                ("TorchCor", manifest.get("torchcor_version") or "—"),
                ("Run folder", r.run_dir),
                ("Created", timestamp(r.created_at)),
                ("Started", timestamp(st.started_at)),
                ("Finished", timestamp(st.ended_at) if st.ended_at else "—")]
        if after := manifest.get("after"):
            rows.append(("Queued behind", f"{len(after)} run{'s' if len(after) != 1 else ''} on the same device"))
        card = Card("Compute", icon="server")
        card.addWidget(KeyValueGrid(rows))
        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(card)
        col.addStretch(1)
        self._machine_lay.addWidget(scroll(holder, max_width=980))

    def _load_script(self) -> None:
        machine = self.ctx.machine(self.record.machine_id)
        if machine is None:
            self.script.setPlainText("# The machine is not connected.")
            return
        record = self.record
        run_async(self.ctx.runs.read_script, machine, record,
                  on_done=lambda text: self.record is record and self.script.setPlainText(text),
                  on_error=lambda e: self.script.setPlainText(f"# Could not read run.py: {e}"), owner=self)

    def _load_files(self) -> None:
        machine = self.ctx.machine(self.record.machine_id)
        self.files_path.setText(self.record.run_dir)
        self.download_btn.setVisible(machine is not None and not machine.profile.is_local)
        if machine is None:
            return
        record = self.record

        def done(entries) -> None:  # noqa: ANN001
            if self.record is not record:
                return
            self.files.clear()
            parents: dict[str, QTreeWidgetItem] = {}
            p = theme().palette
            for rel, nbytes, is_dir in entries:
                parent_key, _, name = rel.rpartition("/")
                item = QTreeWidgetItem([name, "" if is_dir else size(nbytes)])
                item.setData(0, Qt.ItemDataRole.UserRole, rel)
                item.setIcon(0, icons.icon("folder" if is_dir else "file", p.text3))
                (parents[parent_key].addChild(item) if parent_key in parents else self.files.addTopLevelItem(item))
                if is_dir:
                    parents[rel] = item

        run_async(self.ctx.runs.list_files, machine, record, on_done=done,
                  on_error=lambda e: self.ctx.notify.emit("Could not list files", str(e), "danger"), owner=self)

    def _load_summary_tiles(self) -> None:
        if self.summary_tiles.count() or self.record is None:
            return
        machine = self.ctx.machine(self.record.machine_id)
        if machine is None:
            return
        record = self.record

        def work():
            path = machine.join(record.run_dir, "viz", "index.json")
            return machine.read_json(path) if machine.exists(path) else None

        def done(index) -> None:  # noqa: ANN001
            if self.record is not record or not index or self.summary_tiles.count():
                return
            for i, item in enumerate((index.get("summary") or [])[:8]):
                value = item.get("value")
                text = number(value, 4) if isinstance(value, int | float) and not isinstance(value, bool) else str(value)
                tile = StatTile(item.get("label", ""), text, item.get("unit", ""), item.get("hint", ""))
                self.summary_tiles.addWidget(tile, i // 4, i % 4)

        run_async(work, on_done=done, owner=self)

    # ------------------------------------------------------------------ results
    def _release_results(self) -> None:
        view, self._results_view = self._results_view, None
        if view is not None:
            if hasattr(view, "release"):
                view.release()
            discard(view)
        _clear(self._results_lay)

    def _load_results(self) -> None:
        record = self.record
        if self._results_view is not None or self._results_loading:
            return
        _clear(self._results_lay)
        st = record.status.state
        if st != RunState.COMPLETED:
            msg = {
                RunState.RUNNING: ("Results will appear here", "The visualisation opens automatically when the run finishes."),
                RunState.QUEUED: ("Results will appear here", "The run is starting."),
            }.get(st, ("No results", "This run did not finish, so there is nothing to visualise."))
            self._results_lay.addWidget(EmptyState("layers", *msg))
            return
        machine = self.ctx.machine(record.machine_id)
        if machine is None:
            self._results_lay.addWidget(EmptyState("unplug", "Machine offline", "Connect to the machine to load results."))
            return
        self._results_loading = True
        loading = QWidget()
        ll = QVBoxLayout(loading)
        ll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spin = Spinner(24)
        ll.addWidget(spin, 0, Qt.AlignmentFlag.AlignHCenter)
        status = label("Loading results…", "muted")
        ll.addWidget(status, 0, Qt.AlignmentFlag.AlignHCenter)
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setFixedWidth(260)
        bar.setVisible(not machine.profile.is_local)
        ll.addWidget(bar, 0, Qt.AlignmentFlag.AlignHCenter)
        self._results_lay.addWidget(loading)

        progress_box = {"v": 0.0}
        tick = QTimer(self)
        tick.setInterval(100)
        tick.timeout.connect(lambda: bar.setValue(int(progress_box["v"] * 1000)))
        tick.start()

        def work() -> Path:
            if not self.ctx.runs.has_viz(machine, record):
                raise FileNotFoundError("This run has no visualisation bundle (viz/index.json).")
            return self.ctx.runs.fetch_viz(machine, record, progress=lambda f: progress_box.__setitem__("v", f))

        def done(path: Path) -> None:
            tick.stop()
            self._results_loading = False
            if self.record is not record:
                return
            _clear(self._results_lay)
            from torchcor_studio.ui.results import create_results_view

            view = create_results_view(path, parent=self._results_host)
            self._results_view = view
            self._results_lay.addWidget(view)
            if hasattr(view, "set_active"):
                view.set_active(self._view == "Results" and self.isVisible())

        def failed(exc) -> None:  # noqa: ANN001
            tick.stop()
            self._results_loading = False
            if self.record is not record:
                return
            _clear(self._results_lay)
            self._results_lay.addWidget(EmptyState("alert", "Couldn't load the results", str(exc)))

        run_async(work, on_done=done, on_error=failed, owner=self)

    # ------------------------------------------------------------------ actions
    def _stop(self) -> None:
        if self.record is None:
            return
        answer = QMessageBox.question(self, "Stop this run?",
                                      f"“{self.record.name}” will be terminated. Partial results stay in its folder.")
        if answer == QMessageBox.StandardButton.Yes:
            self.ctx.cancel(self.record)
            self.state_badge.set("Stopping…", "warning")

    def _reveal(self) -> None:
        if self.record is None:
            return
        reveal_local(self.record.run_dir)

    def _more(self) -> None:
        if self.record is None:
            return
        p = theme().palette
        menu = QMenu(self)

        def add(text: str, icon: str, fn, enabled: bool = True) -> None:  # noqa: ANN001
            act = QAction(icons.icon(icon, p.text2), text, menu)
            act.setEnabled(enabled)
            act.triggered.connect(fn)
            menu.addAction(act)

        add("Rename…", "edit", self._rename)
        add("Copy folder path", "copy", self._copy_path)
        add("Export script…", "file-code", self._save_script)
        menu.addSeparator()
        add("Delete run…", "trash", self._delete, not self.record.status.state.is_active)
        menu.exec(self.more_btn.mapToGlobal(self.more_btn.rect().bottomLeft()))

    def _rename(self) -> None:
        name, ok = QInputDialog.getText(self, "Rename run", "Name", text=self.record.name)
        if ok and name.strip():
            self.ctx.rename(self.record, name.strip())

    def _copy_path(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.record.run_dir)
        self.ctx.notify.emit("Copied", self.record.run_dir, "success")

    def _copy_script(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.script.toPlainText())
        self.ctx.notify.emit("Copied", "run.py is on the clipboard.", "success")

    def _save_script(self) -> None:
        if not self.script.toPlainText():
            self._load_script()
        path, _ = QFileDialog.getSaveFileName(self, "Export script", f"{self.record.id}.py", "Python (*.py)")
        if path:
            text = self.script.toPlainText()
            if not text:
                self.ctx.notify.emit("Script not loaded yet", "Open the Script tab, then export again.", "warning")
                return
            Path(path).write_text(text, encoding="utf-8")
            self.ctx.notify.emit("Script exported", path, "success")

    def _delete(self) -> None:
        record = self.record
        answer = QMessageBox.warning(
            self, "Delete this run?",
            f"“{record.name}” and its folder will be permanently deleted from {self.ctx.profile(record.machine_id).name if self.ctx.profile(record.machine_id) else 'the machine'}:\n\n{record.run_dir}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Yes:
            self._release_results()
            self.ctx.delete(record, on_done=self.deleted.emit)

    def _open_file(self, item: QTreeWidgetItem) -> None:
        machine = self.ctx.machine(self.record.machine_id)
        rel = item.data(0, Qt.ItemDataRole.UserRole)
        if machine is not None and machine.profile.is_local and rel:
            QDesktopServices.openUrl(QUrl.fromLocalFile(machine.join(self.record.run_dir, *rel.split("/"))))

    def _download_selected(self) -> None:
        machine = self.ctx.machine(self.record.machine_id)
        if machine is None:
            return
        items = self.files.selectedItems()
        rels = [i.data(0, Qt.ItemDataRole.UserRole) for i in items] or ["results", "viz"]
        target = QFileDialog.getExistingDirectory(self, "Download to…")
        if not target:
            return
        record = self.record

        def work() -> int:
            count = 0
            for rel in rels:
                remote = machine.join(record.run_dir, *rel.split("/"))
                count += _download_tree(machine, remote, Path(target) / record.id / rel)
            return count

        self.ctx.notify.emit("Downloading…", f"Saving to {target}", "info")
        run_async(work, on_done=lambda n: self.ctx.notify.emit("Download complete", f"{n} files saved.", "success"),
                  on_error=lambda e: self.ctx.notify.emit("Download failed", str(e), "danger"), owner=self)


def _download_tree(machine, remote: str, local: Path) -> int:  # noqa: ANN001
    if machine.is_dir(remote):
        local.mkdir(parents=True, exist_ok=True)
        return sum(_download_tree(machine, e.path, local / e.name) for e in machine.listdir(remote))
    local.parent.mkdir(parents=True, exist_ok=True)
    machine.download(remote, str(local))
    return 1


def reveal_local(path: str) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def _clear(layout) -> None:  # noqa: ANN001
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            discard(item.widget())
        elif item.layout():
            _clear(item.layout())
