"""The device a run uses, picked on the Compute step: live tiles, the least busy one picked
automatically, and an offer to queue behind this app's runs already on it."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from torchcor_studio.core.gpu import pick_best
from torchcor_studio.core.models import DeviceKind, GPUSample, RunRecord
from torchcor_studio.ui.widgets import SectionHeader, ToggleSwitch, label
from torchcor_studio.ui.widgets.gpu import GpuGrid


class DevicePicker(QWidget):
    """The "Devices" section: the least busy device is selected until the user picks one."""

    device_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(20)  # as between the other section headers and their content
        header = SectionHeader("Devices", "Select the one to run on. Usage is live.")
        header.add_action(label("Pick the least busy", "muted"))
        self.auto = ToggleSwitch(True)
        self.auto.setToolTip("Select the least busy device, following the live usage, until you pick one.")
        self.auto.toggled.connect(lambda on: on and self._auto_pick())
        header.add_action(self.auto)
        lay.addWidget(header)
        body = QVBoxLayout()
        body.setSpacing(10)
        self.grid = GpuGrid(selectable=True, compact=True)
        self.grid.selection_changed.connect(self._user_selected)
        body.addWidget(self.grid)
        self.note = label("No NVIDIA GPU on this machine: the simulation runs on the CPU, which is "
                          "fine for small meshes but much slower than a GPU.", "caption", wrap=True)
        self.note.hide()
        body.addWidget(self.note)
        self.queue_row = QWidget()
        queue = QHBoxLayout(self.queue_row)
        queue.setContentsMargins(0, 0, 0, 0)
        self.queue_text = label("", "muted", wrap=True)
        queue.addWidget(self.queue_text, 1)
        self.queue = ToggleSwitch(True)
        self.queue.setToolTip("Start only when this app's other runs on the device have finished.")
        queue.addWidget(self.queue)
        self.queue_row.hide()
        body.addWidget(self.queue_row)
        lay.addLayout(body)
        self._samples: list[GPUSample] = []
        self._selected_key: str | None = None
        self._wanted: str | None = None  # a device string to select once the machine lists it
        self._active: list[RunRecord] = []

    # ------------------------------------------------------------------ devices
    def prefer(self, device: str) -> None:
        """Select ``device`` ("cuda:1", "cpu") once the machine lists it, instead of the least
        busy one (a run set up again on the device it ran on); call after choosing the
        machine."""
        self._wanted = device or None
        self.update_samples(self._samples)

    def clear(self) -> None:
        """Another machine: drop its devices (and their history) and the choice."""
        had = self._selected_key is not None
        self._samples, self._selected_key, self._wanted = [], None, None
        self.set_active_runs([])
        self.grid.update_samples([])
        self.grid.set_loading()
        if had:
            self.device_changed.emit()

    def update_samples(self, samples: list[GPUSample]) -> None:
        self._samples = samples
        self.grid.update_samples(samples)
        wanted = next((s for s in samples if s.device_string == self._wanted), None)
        if wanted is not None:
            self._wanted = None
            self.auto.blockSignals(True)  # the device asked for, not the least busy
            self.auto.setChecked(False)
            self.auto.blockSignals(False)
            if wanted.key != self._selected_key:
                self._selected_key = wanted.key
                self.grid.select(wanted.key)
                self.device_changed.emit()
        elif self.auto.isChecked() or not any(s.key == self._selected_key for s in samples):
            self._auto_pick()
        self.note.setVisible(bool(samples) and not any(s.kind == DeviceKind.CUDA for s in samples))

    def selected_device(self) -> GPUSample | None:
        return next((s for s in self._samples if s.key == self._selected_key), None)

    def _auto_pick(self) -> None:
        best = pick_best(self._samples)
        if best is not None and best.key != self._selected_key:
            self._selected_key = best.key
            self.grid.select(best.key)
            self.device_changed.emit()

    def _user_selected(self, sample: GPUSample | None) -> None:
        if sample is None or sample.key == self._selected_key:
            return  # a card that went away, or the auto pick itself
        self._selected_key = sample.key
        best = pick_best(self._samples)
        if self.auto.isChecked() and (best is None or best.key != sample.key):
            self.auto.blockSignals(True)  # a deliberate choice: stop choosing for the user
            self.auto.setChecked(False)
            self.auto.blockSignals(False)
        self.device_changed.emit()

    # ------------------------------------------------------------------ queueing
    def set_active_runs(self, records: list[RunRecord]) -> None:
        """This app's runs already active on the selected device (offered as a queue)."""
        self._active = records
        self.queue_row.setVisible(bool(records))
        if records:
            n = len(records)
            names = ", ".join(f"“{r.name}”" for r in records[:2]) + (f" and {n - 2} more" if n > 2 else "")
            self.queue_text.setText(f"Queue behind {names} on this device")

    def queue_after(self) -> list[str]:
        """Run folders the new run waits for (when queueing is on).

        Asked from the Setup step, while this page is hidden: so it goes by the runs
        offered, not by whether the row is visible on screen.
        """
        return [r.run_dir for r in self._active] if self.queue.isChecked() else []
