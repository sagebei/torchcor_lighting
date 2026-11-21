"""The Setup rail's parameter sweep card: one set-up becomes a run per value of a parameter."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QLineEdit, QVBoxLayout, QWidget

from torchcor_studio.core.sweep import SweepTarget, format_value, numeric_parameters, parse_values
from torchcor_studio.ui.theme import repolish
from torchcor_studio.ui.widgets import Card, SegmentedControl, ToggleSwitch, label


class SweepPanel(Card):
    """The rail's PARAMETER SWEEP card. Off by default; when on, pick a numeric parameter
    and the values to run.

    ``params()`` gives the current configuration. It is only read while the sweep is on
    (call :meth:`update_parameters` when the configuration changes), so a sweep that is
    off costs nothing as the form is edited.
    """

    changed = Signal()

    def __init__(self, params: Callable[[], BaseModel | None], parent: QWidget | None = None) -> None:
        super().__init__("Parameter sweep", "One run per value of a parameter", icon="layers", padding=14,
                         parent=parent)
        self._params = params
        self.toggle = ToggleSwitch(False)
        self.toggle.setToolTip("Turn this set-up into a run per value of a parameter")
        self.toggle.toggled.connect(self._toggled)
        self.add_action(self.toggle)
        self.parameter = QComboBox()
        self.parameter.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.parameter.setMinimumContentsLength(12)
        self.parameter.view().setMinimumWidth(420)  # the popup shows whole labels
        self.parameter.currentIndexChanged.connect(self._parameter_changed)
        self.values_edit = QLineEdit()
        self.values_edit.setPlaceholderText("0.1, 0.2, 0.4  or  start:stop:count")
        self.values_edit.setToolTip("Numbers separated by commas, or a range start:stop:count "
                                    "(add :log for log spacing).")
        self.values_edit.textChanged.connect(lambda *_: self._refresh())
        self.preview = label("", "caption", wrap=True)
        self.mode = SegmentedControl([("spread", "In parallel", "grid"), ("queue", "One by one", "list")],
                                     value="spread", compact=True)
        self.mode.setToolTip("In parallel: spread the runs over the machine's GPUs (runs that share a GPU "
                             "wait for each other).\nOne by one: run them in turn on the selected device.")
        self.mode.changed.connect(lambda *_: self.changed.emit())
        self.body_widget = QWidget()  # shown while the sweep is on
        fields = QVBoxLayout(self.body_widget)
        fields.setContentsMargins(0, 2, 0, 0)
        fields.setSpacing(6)
        for title, widget in (("Parameter", self.parameter), ("Values", self.values_edit)):
            fields.addWidget(label(title, "field"))
            fields.addWidget(widget)
        fields.addWidget(self.preview)
        fields.addSpacing(4)
        fields.addWidget(label("Runs", "field"))
        fields.addWidget(self.mode, 0, Qt.AlignmentFlag.AlignLeft)
        self.body.addWidget(self.body_widget)
        self.body_widget.setVisible(False)
        self._targets: list[SweepTarget] = []
        self._error = ""

    # ------------------------------------------------------------------ public
    def enabled(self) -> bool:
        return self.toggle.isChecked()

    def spread(self) -> bool:
        return self.mode.value() == "spread"

    def target(self) -> SweepTarget | None:
        i = self.parameter.currentIndex()
        return self._targets[i] if 0 <= i < len(self._targets) else None

    def values(self) -> list[float]:
        target = self.target()
        return parse_values(self.values_edit.text(), integer=bool(target and target.integer))

    def run_count(self) -> int:
        """How many runs the Run button starts (1 unless a valid sweep is on)."""
        return max(1, len(self._safe_values())) if self.enabled() else 1

    def error(self) -> str:
        """Why the sweep can't run yet ("" when it can or when it's off)."""
        if not self.enabled():
            return ""
        if self.target() is None:
            return "This simulation has no numeric parameter to sweep."
        if self._error:
            return self._error
        if len(self._safe_values()) < 2:
            return "Enter at least two values to sweep."
        return ""

    def set_available(self, available: bool) -> None:
        """Hide the card (and turn the sweep off) when the parameters have no number to vary."""
        if not available and self.toggle.isChecked():
            self.toggle.setChecked(False)
        self.setVisible(available)

    def update_parameters(self) -> None:
        """Re-read the configuration's numeric parameters, keeping the choice (while on).

        The list is rebuilt only when the parameters themselves change (another kind, a
        stimulus added), not when a value does. An invalid form keeps the last list.
        """
        if not self.enabled():
            return
        params = self._params()
        targets = numeric_parameters(params) if params is not None else self._targets
        if [(t.key, t.label, t.unit) for t in targets] != [(t.key, t.label, t.unit) for t in self._targets]:
            current = self.target().key if self.target() else None
            self._targets = targets
            self.parameter.blockSignals(True)
            self.parameter.clear()
            for t in targets:
                text = f"{t.label} ({t.unit})" if t.unit else t.label
                self.parameter.addItem(text, t.key)
                self.parameter.setItemData(self.parameter.count() - 1, text, Qt.ItemDataRole.ToolTipRole)
            self.parameter.setCurrentIndex(max(0, self.parameter.findData(current) if current else 0))
            self.parameter.blockSignals(False)
            self._parameter_changed()  # the new target's placeholder; refreshes
        else:
            self._targets = targets  # same parameters, current values
            self._refresh()

    # ------------------------------------------------------------------ internals
    def _toggled(self, on: bool) -> None:
        self.body_widget.setVisible(on)
        if on:
            self.update_parameters()  # emits changed
        else:
            self.changed.emit()

    def _parameter_changed(self, *_args) -> None:
        if (target := self.target()) is not None:
            self.parameter.setToolTip(self.parameter.currentText())
            self.values_edit.setPlaceholderText(
                f"e.g. {format_value(target.current * 0.5)}, {format_value(target.current)}, "
                f"{format_value(target.current * 2)}  or  start:stop:count")
        self._refresh()

    def _safe_values(self) -> list[float]:
        try:
            return self.values()
        except ValueError:
            return []

    def _refresh(self) -> None:
        self._error = ""
        target = self.target()
        try:
            values = self.values()
        except ValueError as exc:
            values = []
            self._error = str(exc)
        invalid = bool(self._error)
        if self.values_edit.property("invalid") != invalid:
            self.values_edit.setProperty("invalid", invalid)
            repolish(self.values_edit)
        if invalid:
            self.preview.setText(self._error)
        elif values:
            unit = target.unit if target else ""
            shown = ", ".join(format_value(v) for v in values[:8]) + (" …" if len(values) > 8 else "")
            self.preview.setText(f"{len(values)} run{'s' if len(values) != 1 else ''}: {shown} {unit}".strip())
        else:
            self.preview.setText(f"Current value: {format_value(target.current, target.unit)}" if target else "")
        self.changed.emit()
