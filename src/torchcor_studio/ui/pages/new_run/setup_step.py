"""Step 2, Setup: what to simulate.

The simulation's kind (chosen in the New menu) as the page's title, then its parameters as
panels in physiology order (the first one holds the input and output folders) beside a
sticky rail: an outline of the panels, the checks, the kind's demos (pick one and run it)
and the parameter sweep. A kind's form is built when the kind is first shown; it keeps its
parameters, demo, baseline and panel state. Demos, Reset, Undo, fits and duplicates update
the form in place.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from shiboken6 import isValid

from torchcor_studio.core.data_folders import DemoData
from torchcor_studio.core.models import InputInfo
from torchcor_studio.core.simulations import (
    Demo,
    Issue,
    Severity,
    SimulationKind,
    SimulationSpec,
    specs,
)
from torchcor_studio.core.simulations.common import folder_name
from torchcor_studio.core.simulations.panels import INPUT, OUTPUT, owner
from torchcor_studio.core.sweep import numeric_parameters
from torchcor_studio.ui import kinds
from torchcor_studio.ui.forms import ParamForm
from torchcor_studio.ui.pages.new_run.demo_picker import DemoPicker
from torchcor_studio.ui.pages.new_run.folders import Folders
from torchcor_studio.ui.pages.new_run.rail import SetupRail
from torchcor_studio.ui.pages.new_run.sweep_panel import SweepPanel
from torchcor_studio.ui.theme import repolish
from torchcor_studio.ui.widgets import chain_focus, label
from torchcor_studio.ui.widgets.layout import StickyRow

JUMP_OFFSET = 12  # a panel jumped to sits this far below the top of the viewport
FLASH_MS = 900


@dataclass
class _Undo:
    """What a demo replaced (one level)."""

    params: BaseModel
    input: str
    source: str
    demo: str | None
    origin: str
    fitted: str
    baseline: BaseModel


@dataclass
class _KindState:
    form: ParamForm
    baseline: BaseModel  # what changes are counted against
    previous: BaseModel  # the last valid parameters (an edit is fitted against them)
    demo: str | None = None  # the demo shown selected
    origin: str = ""  # where the parameters come from: a demo's title, 'run “name”'
    fitted: str = ""  # the folder the defaults were switched to ("ventricle")
    undo: _Undo | None = None
    touched: bool = False  # edited, a demo chosen or a run loaded: no geometry switch
    fitted_path: str | None = None  # the input folder last fitted


class SetupStep(QWidget):
    kind_changed = Signal(str)
    params_changed = Signal()
    demo_applied = Signal(str, object)  # kind, Demo

    def __init__(self, browse: Callable[[str, str], str | None], folders: Folders | None = None,
                 kind: str = kinds.DEFAULT, parent: QWidget | None = None) -> None:
        """``folders``: the draft's input and output folders (none in unit tests: no folder
        slots); ``kind``: what it simulates."""
        super().__init__(parent)
        self._browse = browse
        self.folders = folders
        self._specs: dict[str, SimulationSpec] = {str(s.kind): s for s in specs()}
        self._states: dict[str, _KindState] = {}
        self._input: InputInfo | None = None
        self._issues: list[Issue] = []  # the shown kind's, as last checked
        self._demo_data: Callable[[Demo], DemoData] | None = folders.demo_data if folders else None
        self._machine = ""
        self._sweepable: dict[str, bool] = {}
        self._tops: list[tuple[int, str]] | None = None
        self._show_advanced = False
        self._expanded = False
        self._watching = False  # the page's viewport, for its height
        self._flashed: QWidget | None = None
        self._flash = QTimer(self)
        self._flash.setSingleShot(True)
        self._flash.setInterval(FLASH_MS)
        self._flash.timeout.connect(self._unflash)
        self.kind = kind if kind in self._specs else str(SimulationKind.EP)
        self._shown: str | None = None  # the kind on the page

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = QWidget()  # the kind: its tile, its name and what it simulates
        head = QHBoxLayout(self.header)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(14)
        self.kind_tile = kinds.KindTile(self.kind, 44)
        head.addWidget(self.kind_tile, 0, Qt.AlignmentFlag.AlignVCenter)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.kind_title = label("", "h2")
        self.tagline = label("", "caption")
        text.addWidget(self.kind_title)
        text.addWidget(self.tagline)
        head.addLayout(text, 1)
        lay.addWidget(self.header)
        lay.addSpacing(22)
        main = QWidget()
        column = QVBoxLayout(main)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)
        forms = QWidget()
        self._forms = QVBoxLayout(forms)
        self._forms.setContentsMargins(0, 0, 0, 0)
        column.addWidget(forms)
        self._ensure(self.kind)  # built before the rail, so Tab goes through the panels first
        tail = QWidget()  # the sweep, below the panels on a narrow page
        column.addWidget(tail)
        column.addStretch(1)
        self.demos = DemoPicker()
        self.demos.apply_requested.connect(self.apply_demo)
        self.demos.reset_requested.connect(self.reset)
        self.demos.undo_requested.connect(self.undo)
        self.sweep = SweepPanel(self.params)
        self.rail = SetupRail(self.demos, self.sweep, tail)
        self.rail.jump.connect(self.reveal_panel)
        self.rail.reveal.connect(self.reveal)
        self.body = StickyRow(main, self.rail, side_width=lambda width: 300 if width >= 1100 else 260)
        self.body.watch(self.rail.cards)
        self.body.mode_changed.connect(self._mode_changed)
        self.body.scrolled.connect(lambda *_: self._spy())
        lay.addWidget(self.body, 1)
        self._parking = QWidget(self)  # holds a folder slot while the kind has none
        self._parking.hide()
        QVBoxLayout(self._parking)
        for slot in self._folder_slots().values():
            self._parking.layout().addWidget(slot)
        self.set_kind(self.kind)

    # ------------------------------------------------------------------ state
    @property
    def spec(self) -> SimulationSpec:
        return self._specs[self.kind]

    def form(self, kind: str | None = None) -> ParamForm:
        return self._states[kind or self.kind].form

    def demo(self, kind: str | None = None) -> Demo | None:
        """The demo the kind's parameters come from (``None``: none)."""
        kind = kind or self.kind
        state = self._states.get(kind)
        return self._specs[kind].demo(state.demo) if state is not None and state.demo else None

    def params(self) -> BaseModel | None:
        state = self._states.get(self.kind)
        return state.form.value() if state else None

    def issues(self) -> list[Issue]:
        """The shown kind's issues, as last checked (after every change)."""
        return list(self._issues)

    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self._issues)

    def _check(self) -> list[Issue]:
        params = self.params()
        if params is None:
            return self.form().error_issues()
        try:
            return self.spec.validate(params, self._input)
        except Exception as exc:  # noqa: BLE001 - a buggy check must not break the page
            return [Issue(Severity.WARNING, f"Could not validate: {exc}")]

    def status_text(self) -> str:
        """Where the parameters come from: "From <demo> · 2 changes", "Applied <demo>"."""
        state = self._states[self.kind]
        if state.undo is not None:
            return f"Applied {state.origin}"
        head = (f"Defaults, fitted to {state.fitted}" if state.fitted
                else f"From {state.origin}" if state.origin else "")
        count = len(state.form.changes())
        if count:
            tail = f"{count} change{'s' * (count != 1)}"
            head = f"{head} · {tail}" if head else tail
        return head

    def _ensure(self, kind: str) -> _KindState:
        """The kind's state, its form built on first use (from its first demo)."""
        state = self._states.get(kind)
        if state is not None:
            return state
        spec = self._specs[kind]
        first = spec.demos[0] if spec.demos else None
        params = first.params() if first else spec.defaults()
        form = ParamForm(spec.params_model, params, panels=spec.panels, browse=self._browse)
        form.set_input_info(self._input)
        if self._show_advanced:
            form.set_show_advanced(True)
        if self._expanded:
            form.expand_all()
        form.changed.connect(lambda m, k=kind: self._form_changed(k, m))
        form.errors.connect(lambda e, k=kind: self._form_errors(k, e))
        form.panels_changed.connect(lambda k=kind: self._panels_changed(k))
        form.installEventFilter(self)
        state = _KindState(form, baseline=params, previous=params,
                           demo=first.key if first else None, origin=first.title if first else "")
        form.set_baseline(params, state.origin)
        self._forms.addWidget(form)
        self._states[kind] = state
        return state

    # ------------------------------------------------------------------ kind
    def set_kind(self, kind: str) -> None:
        """Show ``kind`` (the simulation's, or a duplicated run's), its form built if new."""
        if kind not in self._specs or kind == self._shown:
            return
        previous = self._states.get(self._shown) if self._shown else None
        self.kind = self._shown = kind
        built = kind not in self._states  # the simulation's own form was built before the rail
        state = self._ensure(kind)
        spec = self._specs[kind]
        self.kind_tile.set_kind(kind)
        self.kind_title.setText(spec.title)
        self.tagline.setText(f"{spec.tagline}.")
        self.header.setToolTip(spec.description)
        for k, st in self._states.items():
            st.form.setVisible(k == kind)
        self.demos.set_spec(self._specs[kind])
        self.refresh_demo_data()
        self._move_slots(previous, state)
        if built:
            self._tab_order(state)
        if kind not in self._sweepable:
            self._sweepable[kind] = bool(numeric_parameters(self._specs[kind].defaults()))
        self.sweep.set_available(self._sweepable[kind])
        self._tops = None
        self._fit_code()
        self.kind_changed.emit(kind)  # the folders follow the kind before anything is fitted
        self._fit_folder(kind)
        self._revalidate()
        self._spy()
        self.params_changed.emit()  # the parameters are another kind's

    def _tab_order(self, state: _KindState) -> None:
        """Tab goes through the panels, then the rail. The simulation's own form is built
        before the rail; a kind's form built later (just now, so its widgets are in the focus
        chain in the order they were made) has the rail move after its last widget."""
        last = next((w for w in reversed(state.form.findChildren(QWidget))
                     if w.focusPolicy() & Qt.FocusPolicy.TabFocus), None)
        if last is not None:
            chain_focus(last, self.rail)

    def _folder_slots(self) -> dict[str, QWidget]:
        """The draft's folder slots by the panel slot they go in (none without folders)."""
        if self.folders is None:
            return {}
        return {INPUT: self.folders.input_slot, OUTPUT: self.folders.output_slot}

    def _move_slots(self, old: _KindState | None, new: _KindState) -> None:
        """The draft's input and output folders move into the shown kind's first panel (a
        kind without an input block parks the input slot)."""
        for name, slot in self._folder_slots().items():
            if old is not None and old is not new:
                old.form.set_slot(name, None)
            self._parking.layout().removeWidget(slot)
            if new.form.has_slot(name):
                new.form.set_slot(name, slot)
            else:
                self._parking.layout().addWidget(slot)

    # ------------------------------------------------------------------ input folder
    def set_input(self, info: InputInfo | None) -> None:
        """What the input folder holds (``None``: not known yet)."""
        self._input = info
        for state in self._states.values():
            state.form.set_input_info(info)
        self._fit_folder(self.kind)
        self._revalidate()

    def _pristine(self, state: _KindState) -> bool:
        return not state.touched and state.form.value() == state.baseline

    def _fit_folder(self, kind: str) -> None:
        """Fit what depends on the input folder. A new folder switches an untouched set-up to
        what the folder holds (``fit_input``'s ``new_input``: EP's slab becomes the folder's
        mesh), and the status line then says so; a re-inspection never does."""
        state, spec, info = self._states[kind], self._specs[kind], self._input
        params = state.form.value()
        if info is None or spec.fit_input is None or params is None:
            return
        new_folder = info.path != state.fitted_path
        state.fitted_path = info.path
        fitted = spec.fit_input(params, info, None, new_input=False)
        if new_folder and self._pristine(state):
            switched = spec.fit_input(params, info, None, new_input=True)
            if switched is not None and switched != (fitted or params):
                fitted = switched
                state.demo, state.fitted = None, folder_name(info.path)
            elif state.fitted:  # already fitted to a folder: now to this one
                state.fitted = folder_name(info.path)
                state.form.set_baseline(state.baseline, self._origin(state))
        self._apply_fit(state, fitted)

    def _fit(self, kind: str, previous: BaseModel | None = None, edited: BaseModel | None = None) -> bool:
        """Fit after an edit (``previous``: before it) or a demo (``None``)."""
        state, spec = self._states[kind], self._specs[kind]
        params = state.form.value()
        if spec.fit_input is None or params is None or (edited is not None and params is not edited):
            return False  # edited again since; that edit is fitted on its own
        return self._apply_fit(state, spec.fit_input(params, self._input, previous, new_input=False))

    def _apply_fit(self, state: _KindState, fitted: BaseModel | None) -> bool:
        """Show fitted parameters; while untouched by the user they become the baseline, so a
        fitted demo still reads "From <demo>"."""
        if fitted is None:
            return False
        follows = state.form.value() == state.baseline
        state.form.set_value(fitted)
        state.previous = fitted
        if follows:
            state.baseline = fitted
            state.form.set_baseline(fitted, self._origin(state))
        return True

    # ------------------------------------------------------------------ demos, reset, undo
    def apply_demo(self, key: str) -> None:
        """Start from a demo: its parameters in place, then the folders follow its data
        (``demo_applied``). Edited parameters can be put back with :meth:`undo`."""
        state, spec = self._states[self.kind], self.spec
        demo = spec.demo(key)
        if state.form.changes() or state.form.value() is None:
            state.undo = _Undo(state.form.value() or state.previous,
                               self.folders.input_dir() if self.folders else "",
                               self.folders.source if self.folders else "default",
                               state.demo, state.origin, state.fitted, state.baseline)
        else:
            state.undo = None
        model = demo.params()
        state.demo, state.origin, state.fitted, state.touched = key, demo.title, "", True
        state.baseline = state.previous = model
        state.form.set_value(model)
        state.form.set_baseline(model, demo.title)
        self.demo_applied.emit(self.kind, demo)
        self._fit(self.kind)
        self._revalidate()
        self.params_changed.emit()

    def reset(self) -> None:
        """Put the baseline parameters back (the folders stay)."""
        state = self._states[self.kind]
        state.undo = None
        state.previous = state.baseline
        state.form.set_value(state.baseline)
        self._revalidate()
        self.params_changed.emit()

    def undo(self) -> None:
        """Put back what the last demo replaced: parameters, input folder, demo and baseline."""
        state = self._states[self.kind]
        record, state.undo = state.undo, None
        if record is None:
            return
        state.demo, state.origin, state.fitted = record.demo, record.origin, record.fitted
        state.baseline, state.previous = record.baseline, record.params
        state.form.set_value(record.params)
        state.form.set_baseline(record.baseline, self._origin(state))
        if self.folders is not None:
            self.folders.set_demo(self.demo())
            self.folders.set_input(record.input, record.source)
        self._revalidate()
        self.params_changed.emit()

    def load(self, kind: str, params: dict, origin: str = "") -> None:
        """Pre-fill from an earlier run (Duplicate and edit), named ``origin``. Parameters
        equal to a demo's show that demo selected."""
        spec = self._specs.get(kind)
        if spec is None:
            return
        try:
            model = spec.params_model.model_validate(params)
        except Exception:  # noqa: BLE001 - older/incompatible parameters fall back to defaults
            model = spec.defaults()
        state = self._ensure(kind)
        match = next((d for d in spec.demos if d.params() == model), None)
        state.demo = match.key if match else None
        state.origin = f"run “{origin}”" if origin else (match.title if match else "")
        state.fitted, state.undo, state.touched = "", None, True
        state.baseline = state.previous = model
        state.form.set_value(model)
        state.form.set_baseline(model, self._origin(state))
        if kind != self.kind:
            self.set_kind(kind)
            return
        if self.folders is not None:
            self.folders.set_demo(match)
        self._revalidate()

    def refresh_demo_data(self, machine: str | None = None) -> None:
        """Redraw where the demos' data stands (the machine connected, or another was chosen:
        ``machine``)."""
        if machine is not None:
            self._machine = machine
        if self._demo_data is not None:
            self.demos.set_data(self._demo_data, self._machine)

    # ------------------------------------------------------------------ panels
    def set_show_advanced(self, on: bool) -> None:
        """Show every panel's advanced parameters, in every kind (scripts and tests)."""
        self._show_advanced = on
        for state in self._states.values():
            state.form.set_show_advanced(on)

    def expand_all(self) -> None:
        """Open every panel, in every kind (scripts and tests)."""
        self._expanded = True
        for state in self._states.values():
            state.form.expand_all()

    def reveal(self, path: str) -> None:
        """Open the panel of a parameter, scroll its field into view, focus and flash it."""
        widget = self.form().reveal(path)
        if widget is not None:
            QTimer.singleShot(0, self, lambda: self._show(widget))

    def reveal_panel(self, key: str) -> None:
        """Open a panel and scroll its top just below the top of the viewport."""
        card = self.form().panel(key)
        card.expand()
        QTimer.singleShot(0, self, lambda: self._scroll_to(card, JUMP_OFFSET))

    def reveal_first_error(self) -> None:
        for issue in self._issues:
            if issue.severity == Severity.ERROR and issue.field:
                self.reveal(issue.field)
                return

    def _scroll_to(self, widget: QWidget, offset: int) -> None:
        area = self.body.scroll_area()
        if area is not None and isValid(widget):
            y = widget.mapTo(area.widget(), QPoint(0, 0)).y()
            area.verticalScrollBar().setValue(y - offset)

    def _show(self, widget: QWidget) -> None:
        if not isValid(widget):
            return
        area = self.body.scroll_area()
        self._scroll_to(widget, area.viewport().height() // 3 if area is not None else 0)
        widget.setFocus(Qt.FocusReason.OtherFocusReason)
        self._unflash()
        self._flashed = widget
        widget.setProperty("flash", True)
        repolish(widget)
        self._flash.start()

    def _unflash(self) -> None:
        widget, self._flashed = self._flashed, None
        if widget is not None and isValid(widget):
            widget.setProperty("flash", False)
            repolish(widget)

    def _panels_changed(self, kind: str) -> None:
        self._tops = None
        if kind == self.kind:
            self.rail.set_outline(self.form().outline_entries())
            self._spy()

    def _spy(self) -> None:
        """The outline's current panel: the last one whose top is near the top of the
        viewport; at the top of the page the first one, at the bottom the last one in view."""
        area = self.body.scroll_area()
        if area is None or not self.body.is_beside() or self.kind not in self._states:
            return
        if self._tops is None:
            content = area.widget()
            self._tops = [(card.mapTo(content, QPoint(0, 0)).y(), card.key)
                          for card in self.form().cards()]
        if not self._tops:
            return
        bar = area.verticalScrollBar()
        if bar.value() <= bar.minimum():  # a short first panel still leads
            self.rail.outline.set_current(self._tops[0][1])
            return
        if bar.maximum() and bar.value() >= bar.maximum():
            limit = bar.value() + area.viewport().height()
            key = [k for top, k in self._tops if top < limit][-1:] or [self._tops[0][1]]
            self.rail.outline.set_current(key[0])
            return
        # current: the last panel whose top is in the upper third of the view
        index = bisect_right([top for top, _k in self._tops], bar.value() + area.viewport().height() // 3) - 1
        self.rail.outline.set_current(self._tops[max(0, index)][1])

    def _mode_changed(self, beside: bool) -> None:
        self.rail.set_mode(beside)
        self._tops = None

    def _page_viewport(self) -> QWidget | None:
        """The viewport of the page's scroll area, asked for each time: a Python wrapper of
        it kept from earlier can be invalidated by PySide while the viewport lives on."""
        area = self.body.scroll_area()
        return area.viewport() if area is not None else None

    def _fit_code(self) -> None:
        """A code editor fills the page: at least the viewport's height less 300 px."""
        viewport = self._page_viewport()
        if viewport is not None and self.kind in self._states:
            self.form().set_code_height(max(460, viewport.height() - 300))

    # ------------------------------------------------------------------ events
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        viewport = self._page_viewport()
        if viewport is not None and not self._watching:
            self._watching = True
            viewport.installEventFilter(self)
            self._fit_code()
        self._spy()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._tops = None

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        kind = event.type()
        if kind == QEvent.Type.LayoutRequest and isinstance(obj, ParamForm):
            self._tops = None
        elif kind == QEvent.Type.Resize and obj is self._page_viewport():
            self._fit_code()
        return False

    # ------------------------------------------------------------------ edits
    def _form_changed(self, kind: str, model: BaseModel) -> None:
        state = self._states[kind]
        previous, state.previous = state.previous, model
        state.touched, state.undo = True, None
        if kind == self.kind:
            self._revalidate()
            self.params_changed.emit()
        if self._specs[kind].fit_input is not None:
            # once the edit has settled: the form must not be updated inside its own signal
            QTimer.singleShot(0, self, lambda: self._refit(kind, previous, model))

    def _refit(self, kind: str, previous: BaseModel, edited: BaseModel) -> None:
        if self._fit(kind, previous, edited) and kind == self.kind:
            self._revalidate()
            self.params_changed.emit()

    def _form_errors(self, kind: str, errors: list) -> None:
        state = self._states[kind]
        if errors:
            state.touched, state.undo = True, None
            if kind == self.kind:
                self._revalidate()
                self.params_changed.emit()

    def _origin(self, state: _KindState) -> str:
        return f"the defaults fitted to {state.fitted}" if state.fitted else state.origin

    def _revalidate(self) -> None:
        """Re-check the parameters and refresh what follows them: tints and counts, the input
        folder's role and the rail (checks, outline, the demos' status line, sweep)."""
        state, spec = self._states[self.kind], self.spec
        form = state.form
        issues = self._issues = self._check()
        form.mark_issues(issues)
        params = form.value() or state.previous
        role, hint = spec.input_role(params)
        form.set_input_role(role, hint)
        if self.folders is not None:
            self.folders.set_role(role, hint)
        panels = form.panels()
        self.rail.checks.set_issues(
            issues, lambda path: (panel.title if (panel := owner(panels, path)) else None))
        self.rail.set_outline(form.outline_entries())
        self._update_status(state)
        self.sweep.update_parameters()

    def _update_status(self, state: _KindState) -> None:
        """The demo card: the demo the parameters come from, then what the list does not say
        (the changes since, a demo applied over edits, or another origin)."""
        changes = state.form.changes()
        self.demos.set_selected(state.demo)
        tip = "\n".join(c.label for c in changes[:8])
        if len(changes) > 8:
            tip += f"\nand {len(changes) - 8} more"
        origin = self._origin(state)
        reset = f"Put back the parameters of {origin}; the folders stay." if changes and origin else (
            "Put back the parameters this set-up started from; the folders stay." if changes else "")
        demo = self.demo()
        if state.undo is not None:
            text = "Applied over your edits"
        elif demo is not None and state.origin == demo.title:  # the list names the origin
            text = f"{len(changes)} change{'s' * (len(changes) != 1)}" if changes else ""
        else:
            text = self.status_text()
        self.demos.set_status(text, tip=tip, reset=reset, undo=state.undo is not None)
