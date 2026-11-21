"""A simulation being set up: Compute → Setup, then Run, where the tab becomes the run's
page (``RunPage``, whose header shows the same steps). Its kind is chosen in the New menu
and stays; it is named after it (EP, EP_1, …) until renamed."""

from __future__ import annotations

import uuid

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from torchcor_studio.core.models import DeviceKind, RunRecord
from torchcor_studio.core.runs import LaunchRequest
from torchcor_studio.core.runs.protocol import default_run_name
from torchcor_studio.core.simulations import Severity
from torchcor_studio.ui import kinds
from torchcor_studio.ui.browse import browse_path
from torchcor_studio.ui.context import AppContext
from torchcor_studio.ui.format import plural
from torchcor_studio.ui.pages.common import SIMULATION_STEPS, scroll
from torchcor_studio.ui.pages.new_run.compute_step import ComputeStep
from torchcor_studio.ui.pages.new_run.folders import Folders, machine_words
from torchcor_studio.ui.pages.new_run.setup_step import SetupStep
from torchcor_studio.ui.widgets import Button, label
from torchcor_studio.ui.widgets.code import CodeView
from torchcor_studio.ui.widgets.steps import StepIndicator

#: What a simulation without a name is called.
UNTITLED = "New simulation"


class NewRunPage(QWidget):
    """One simulation of ``kind`` being set up (a "draft"): where it runs, then what it
    simulates."""

    run_started = Signal(object)  # RunRecord
    add_machine_requested = Signal()
    install_requested = Signal(str)  # machine id
    title_changed = Signal(str)  # the name given to the simulation ("" until named)
    project_changed = Signal(object)  # project id it will be filed in, or None

    def __init__(self, ctx: AppContext, kind: str = kinds.DEFAULT, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.draft_id = uuid.uuid4().hex[:12]
        self.ctx = ctx
        self._step = 0
        self._launching = False
        self._name = ""  # the name given to it ("" until one is)
        self._default_name = ""  # its name until then ("EP_1")
        self._project_id: str | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        # only the steps: the simulation's name is on its tab and its sidebar entry
        self.header = QWidget()
        self.header.setObjectName("PageHeader")
        self.header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        head = QHBoxLayout(self.header)
        head.setContentsMargins(28, 10, 24, 10)  # as the run page's: the steps stay put on launch
        self.steps = StepIndicator(SIMULATION_STEPS)
        self.steps.step_clicked.connect(self.go)
        head.addWidget(self.steps, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(self.header)

        self.compute = ComputeStep(ctx)
        self.folders = Folders(ctx, self)
        self.setup = SetupStep(self._browse_for_form, self.folders, kind=kind)
        self.stack = QStackedWidget()
        for step in (self.compute, self.setup):
            self.stack.addWidget(scroll(step))
        lay.addWidget(self.stack, 1)

        footer = QWidget()
        footer.setObjectName("PageFooter")
        footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        foot = QHBoxLayout(footer)
        foot.setContentsMargins(28, 12, 28, 12)
        foot.setSpacing(10)
        self.back_btn = Button("Back", "arrow-left", "ghost", on_click=lambda: self.go(self._step - 1))
        self.hint = label("", "caption")
        self.issues_btn = Button("", "x-circle", "danger", on_click=lambda: self.setup.reveal_first_error())
        self.issues_btn.setToolTip("Show the first issue to fix")
        self.issues_btn.hide()
        self.next_btn = Button("Continue", "arrow-right", "primary", on_click=lambda: self.go(self._step + 1))
        self.preview_btn = Button("Preview script", "code", "ghost", on_click=self.preview_script)
        self.preview_btn.setToolTip("The run.py this simulation runs, as it will be saved in its run folder.")
        self.run_btn = Button("Run simulation", "play", "primary", size="lg", on_click=self._launch)
        foot.addWidget(self.back_btn)
        foot.addStretch(1)
        foot.addWidget(self.hint)
        foot.addWidget(self.issues_btn)
        foot.addWidget(self.next_btn)
        foot.addWidget(self.preview_btn)
        foot.addWidget(self.run_btn)
        lay.addWidget(footer)

        self.folders.set_kind(self.setup.kind, self.setup.demo())
        ctx.projects_changed.connect(self._projects_changed)
        ctx.machine_state_changed.connect(self._machine_state)
        self.compute.ready_changed.connect(lambda *_: self._sync())
        self.compute.machine_changed.connect(self._machine_changed)
        self.compute.devices.device_changed.connect(self._device_changed)
        self.folders.input_changed.connect(self._input_changed)
        self.folders.output.path_changed.connect(lambda *_: self._sync())
        self.compute.add_machine_requested.connect(self.add_machine_requested.emit)
        self.compute.install_requested.connect(lambda: self.install_requested.emit(self.compute.machine_id or ""))
        self.setup.params_changed.connect(self._sync)
        self.setup.kind_changed.connect(self._kind_changed)
        self.setup.demo_applied.connect(lambda _kind, demo: self.folders.use_demo(demo))
        self.setup.sweep.changed.connect(self._sweep_changed)
        self.go(0)

    # ------------------------------------------------------------------ lifecycle
    def showEvent(self, e) -> None:  # noqa: N802
        self.compute.activate()
        super().showEvent(e)

    def hideEvent(self, e) -> None:  # noqa: N802
        self.compute.deactivate()
        super().hideEvent(e)

    def start_fresh(self, project_id: str | None = None, name: str = "") -> None:
        """A new simulation called ``name`` until it is renamed ("EP_1"), filed in
        ``project_id`` (it can be moved in the sidebar)."""
        self._default_name = name
        self.set_name("")
        self.set_project(project_id)
        self.go(0)

    # ------------------------------------------------------------------ project
    def project_id(self) -> str | None:
        """The project the run will be filed in (``None``: no project)."""
        return self._project_id

    def set_project(self, project_id: str | None) -> None:
        """File the simulation in a project (its sidebar entry follows)."""
        if project_id != self._project_id:
            self._project_id = project_id
            self.project_changed.emit(project_id)

    def _projects_changed(self) -> None:
        if self._project_id and self.ctx.project(self._project_id) is None:
            self.set_project(None)  # its project was deleted

    # ------------------------------------------------------------------ name
    def name(self) -> str:
        """The simulation's name: the one given to it, else the one it started with ("EP_1";
        "" when it has neither: the run then gets a default)."""
        return self._name or self._default_name

    def title(self) -> str:
        return self.name() or UNTITLED

    def set_name(self, name: str) -> None:
        """Rename the simulation (its tab and sidebar entry follow); "" goes back to the
        name it started with."""
        self._name = name.strip()
        self.title_changed.emit(self.name())

    def is_dirty(self) -> bool:
        """Anything set up worth keeping (closing the tab would lose it)?"""
        return self._step > 0 or bool(self._name)

    def start_from(self, record: RunRecord, *, again: bool = False, step: int = 1) -> None:
        """Duplicate a run: same machine, folders, kind and parameters, shown at ``step``.
        ``again``: the run itself set up again (it stopped or failed), so its name and the
        device it ran on, not a copy's."""
        manifest = record.manifest or {}
        self.compute.set_machine(record.machine_id)
        if again:
            self.compute.devices.prefer(record.device)
        self.setup.load(record.kind, manifest.get("params") or {}, origin=record.name)  # before the folders:
        self.folders.set_machine(self.compute.machine_id, input_dir=manifest.get("input_dir", ""),
                                 output_dir=manifest.get("output_dir"))  # its own, not the kind's
        self.set_name(record.name if again else f"{record.name} (copy)")
        self.set_project(record.project_id)  # same project
        self.go(step)

    # ------------------------------------------------------------------ navigation
    def go(self, step: int) -> None:
        """Show a set-up step (the Run step is reached with the Run button)."""
        step = max(0, min(self.stack.count() - 1, step))
        if step > self._step and self.compute.machine_id is None:
            return
        self._step = step
        self.stack.setCurrentIndex(step)
        if step == 0:
            self._offer_queue()  # runs may have started or ended since
        self._sync()

    def _sync(self) -> None:
        step = self._step
        self.back_btn.setVisible(step > 0)
        self.next_btn.setVisible(step == 0)
        self.preview_btn.setVisible(step == 1)
        self.run_btn.setVisible(step == 1)
        errors = 0
        if step == 0:
            sess = self.ctx.session(self.compute.machine_id) if self.compute.machine_id else None
            if sess is None:
                self.hint.setText("Choose a machine.")
            elif not sess.connected:
                self.hint.setText("Connecting…" if sess.state == "connecting" else "")
            elif not self.compute.is_ready():
                self.hint.setText("TorchCor isn't installed in this environment yet — you can still configure the run.")
            else:
                self.hint.setText("")
            self.next_btn.setEnabled(self.compute.machine_id is not None)
        else:
            errors = self._errors()
            blocked = self._launch_blocker()
            self.issues_btn.setText(f"{plural(errors, 'issue')} to fix")
            self.hint.setText(blocked or self._destination())
            self.run_btn.setEnabled(blocked is None and not self._launching)
        self.issues_btn.setVisible(bool(errors))  # instead of the hint: a click shows the first
        self.hint.setVisible(not errors)
        self.steps.set_current(step, reachable=1 if self.compute.machine_id else 0)

    def _errors(self) -> int:
        """Errors of the set-up, as last checked (the Setup step checks on every change)."""
        return sum(i.severity == Severity.ERROR for i in self.setup.issues())

    def _launch_blocker(self) -> str | None:
        """Why the simulation can't start yet (``None`` when it can)."""
        errors = self._errors()
        if errors:
            return f"{plural(errors, 'issue')} to fix"
        if not self.folders.output_dir():
            return "Choose an output folder."
        if not self.compute.is_ready():
            return "The selected environment can't run TorchCor yet."
        device = self.compute.devices.selected_device()
        if device is None:
            return "Waiting for a device."
        if device.kind == DeviceKind.CUDA and not self.compute.cuda_usable():
            return "This Python's PyTorch can't use NVIDIA GPUs: select the CPU or another Python on the Compute step."
        return self.setup.sweep.error() or None

    def _destination(self) -> str:
        """Where the run goes, next to the Run button: the device was chosen on another page."""
        device = self.compute.devices.selected_device()
        profile = self.ctx.profile(self.compute.machine_id) if self.compute.machine_id else None
        if device is None or profile is None:
            return ""
        count = self.setup.sweep.run_count()
        if count > 1:
            devices = len({slot.device.key for slot in self._sweep_plan(count)})
            return f"On {profile.name} · {plural(devices, 'device')}"
        return f"On {profile.name} · {device.name} ({device.device_string})"

    # ------------------------------------------------------------------ helpers
    def _browse_for_form(self, mode: str, current: str) -> str | None:
        machine = self.ctx.machine(self.compute.machine_id) if self.compute.machine_id else None
        return browse_path(self, machine, mode, current)

    def _machine_changed(self, machine_id: str) -> None:
        self.folders.set_machine(machine_id)
        self._machine_state(machine_id)
        self._sync()

    def _machine_state(self, machine_id: str) -> None:
        """The machine connected (its data folder is known) or another was chosen: the
        demos' data chips follow."""
        profile = self.ctx.profile(machine_id)
        if machine_id == self.compute.machine_id and profile is not None:
            self.setup.refresh_demo_data(machine_words(profile))

    def _input_changed(self, info) -> None:  # noqa: ANN001
        self.setup.set_input(info)
        self._sync()

    def _kind_changed(self, kind: str) -> None:
        self.folders.set_kind(kind, self.setup.demo())
        self._sync()

    def _device_changed(self) -> None:
        self._offer_queue()
        self._sync()

    def _offer_queue(self) -> None:
        """Offer to queue behind this app's runs already active on the selected device."""
        device = self.compute.devices.selected_device()
        machine_id = self.compute.machine_id
        active = [r for r in self.ctx.records()
                  if r.machine_id == machine_id and r.status.state.is_active
                  and r.device == device.device_string] if device is not None and machine_id else []
        self.compute.devices.set_active_runs(active)

    def _sweep_changed(self) -> None:
        count = self.setup.sweep.run_count()
        self.run_btn.setText(f"Run {count} simulations" if count > 1 else "Run simulation")
        self._sync()

    def _sweep_plan(self, count: int):  # noqa: ANN202
        from torchcor_studio.core.sweep import plan

        samples = self.ctx.session(self.compute.machine_id).samples if self.compute.machine_id else []
        if not self.compute.cuda_usable():
            samples = [s for s in samples if s.kind != DeviceKind.CUDA]
        return plan(samples, count, spread=self.setup.sweep.spread(),
                    preferred=self.compute.devices.selected_device())

    def preview_script(self) -> None:
        machine_id = self.compute.machine_id
        machine = self.ctx.machine(machine_id) if machine_id else None
        params = self.setup.params()
        if params is None:
            self.ctx.notify.emit("Fix the parameters first", "The script can't be generated while fields are invalid.", "warning")
            return
        device = self.compute.devices.selected_device()
        from torchcor_studio.core.simulations import ScriptContext

        out = self.folders.output_dir() or "~/TorchCorLighting/runs"
        ctx = ScriptContext(device=device.device_string if device else "cuda:0",
                            input_dir=self.folders.input_dir(), run_dir=f"{out.rstrip('/')}/<run-id>",
                            run_name=self.name() or "Preview",
                            machine_name=machine.name if machine else "")
        try:
            text = self.setup.spec.build_script(params, ctx)
        except Exception as exc:  # noqa: BLE001
            text = f"# Could not generate the script:\n# {exc}"
        dlg = QDialog(self)
        dlg.setWindowTitle("Generated script")
        dlg.resize(900, 720)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)
        v.addWidget(label("run.py", "h3"))
        v.addWidget(label("This is exactly what will run. It's saved in the run folder, so you can re-run or adapt it later.",
                          "caption", wrap=True))
        v.addWidget(CodeView(text), 1)
        row = QHBoxLayout()
        row.addStretch(1)

        def copy() -> None:
            from PySide6.QtWidgets import QApplication

            QApplication.clipboard().setText(text)
            self.ctx.notify.emit("Copied", "The script is on the clipboard.", "success")

        row.addWidget(Button("Copy", "copy", on_click=copy))
        row.addWidget(Button("Close", variant="primary", on_click=dlg.accept))
        v.addLayout(row)
        dlg.exec()

    # ------------------------------------------------------------------ launch
    def _launch(self) -> None:
        blocker = self._launch_blocker()
        if blocker:
            self.ctx.notify.emit("Can't start yet", blocker, "warning")
            return
        self._offer_queue()  # the runs on the device now, not when it was selected
        machine_id = self.compute.machine_id
        sess = self.ctx.session(machine_id)
        device = self.compute.devices.selected_device()
        params = self.setup.params()
        spec = self.setup.spec
        base_name = self.name() or default_run_name(spec.title, spec.variant_of(params))
        base = LaunchRequest(
            spec=spec,
            params=params,
            name=base_name,
            input_dir=self.folders.input_dir(),
            output_dir=self.folders.output_dir(),
            device=device.device_string,
            device_name=device.name,
            torchcor_version=sess.env.torchcor_version if sess.env else None,
            after=tuple(self.compute.devices.queue_after()),
            project=project.ref() if (project := self.ctx.project(self.project_id())) else None,
        )
        items = self._sweep_items(base) if self.setup.sweep.enabled() else [(base, None)]
        self._launching = True
        self.run_btn.setText("Starting…")
        self._sync()

        def done(records: list[RunRecord]) -> None:
            self._launching = False
            self._sweep_changed()
            self.ctx.settings.last_kind = self.setup.kind
            self.ctx.store.save_settings()
            self.run_started.emit(records[0])
            self._sync()

        def failed(_exc) -> None:  # noqa: ANN001
            self._launching = False
            self._sweep_changed()
            self._sync()

        self.ctx.launch_batch(machine_id, items, on_done=done, on_error=failed)

    def _sweep_items(self, base: LaunchRequest) -> list[tuple[LaunchRequest, int | None]]:
        """One request per sweep value; runs sharing a device queue behind each other."""
        from dataclasses import replace

        from torchcor_studio.core.sweep import apply, format_value

        sweep = self.setup.sweep
        target = sweep.target()
        values = sweep.values()
        slots = self._sweep_plan(len(values))
        items: list[tuple[LaunchRequest, int | None]] = []
        for value, slot in zip(values, slots, strict=True):
            on_base_device = slot.device.device_string == base.device
            items.append((replace(
                base,
                params=apply(base.params, target, value),
                name=f"{base.name} · {target.label.split(' › ')[-1]} = {format_value(value, target.unit)}",
                device=slot.device.device_string,
                device_name=slot.device.name,
                after=base.after if (slot.after is None and on_base_device) else (),
                sweep={"parameter": target.label, "path": list(target.path), "value": value,
                       "unit": target.unit, "group": base.name},
            ), slot.after))
        return items
