"""Input and output folders on the compute machine, remembered per machine.

:class:`Folders` owns the two folder fields of a draft. Both live in the Setup step's
first panel: the input (:class:`InputSlot`, where the mesh comes from) and the output
(:class:`FolderSlot`, where run folders are created). Paths are on the machine chosen on
the Compute step.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLineEdit, QMenu, QVBoxLayout, QWidget
from shiboken6 import isValid

from torchcor_studio.core.data_folders import DataStatus, DemoData, demo_data
from torchcor_studio.core.models import InputInfo
from torchcor_studio.core.simulations import Demo
from torchcor_studio.ui.browse import browse_path
from torchcor_studio.ui.context import AppContext
from torchcor_studio.ui.format import number
from torchcor_studio.ui.theme import repolish
from torchcor_studio.ui.widgets import (
    Button,
    IconButton,
    IconLabel,
    Spinner,
    WrapLabel,
    label,
)


def _unique(paths: list[str]) -> list[str]:
    """Non-empty paths, first occurrence kept."""
    return list(dict.fromkeys(p for p in paths if p))


def machine_words(profile) -> str:  # noqa: ANN001
    """A machine as running text names it: "this computer" for the local "This computer",
    else its name as typed."""
    name = profile.name
    return name[0].lower() + name[1:] if profile.is_local and name.startswith("This ") else name


class FolderField(QWidget):
    """Path line edit, recent-folders menu and Browse."""

    changed = Signal(str)  # chosen: typed (when done), browsed or picked from the recent ones
    path_changed = Signal(str)  # any change of the path, also while typing or from code

    def __init__(self, placeholder: str, browse) -> None:  # noqa: ANN001
        super().__init__()
        self._browse = browse
        self._where = ""
        self._path = ""
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setFixedHeight(32)
        self.edit.editingFinished.connect(self._finished)
        self.edit.textChanged.connect(self._set_path)
        lay.addWidget(self.edit, 1)
        self.recent_btn = IconButton("history", "Recent folders", size=30, icon_size=16)
        self.recent_btn.clicked.connect(self._show_recent)
        lay.addWidget(self.recent_btn)
        self.browse_btn = Button("Browse…", "folder-open", size="sm")
        self.browse_btn.clicked.connect(self._pick)
        lay.addWidget(self.browse_btn)
        self.recent: list[str] = []

    def text(self) -> str:
        return self._path.strip()

    def set_text(self, text: str, emit: bool = False) -> None:
        self._set_path(text)
        self._show()
        if emit:
            self.changed.emit(text)

    def _set_path(self, text: str) -> None:
        if text != self._path:
            self._path = text
            self.path_changed.emit(text)

    def set_machine_name(self, words: str) -> None:
        """``words``: the machine as running text names it ("this computer", "gpu-box")."""
        self._where = f"On {words}" if words else ""
        self._show()

    def _finished(self) -> None:
        self._set_path(self.edit.text())
        self.changed.emit(self._path.strip())

    def _show(self) -> None:
        if self._path != self.edit.text():
            self.edit.blockSignals(True)  # the value is already set
            self.edit.setText(self._path)
            self.edit.blockSignals(False)
            self.edit.setCursorPosition(0)  # the start of the path: where it is
        self.edit.setToolTip("\n".join(filter(None, (self._path, self._where))))

    def _pick(self) -> None:
        path = self._browse(self.text())
        if path:
            self.set_text(path, emit=True)

    def _show_recent(self) -> None:
        menu = QMenu(self)
        if not self.recent:
            act = menu.addAction("No recent folders")
            act.setEnabled(False)
        for path in self.recent:
            menu.addAction(path, lambda p=path: self.set_text(p, emit=True))
        menu.exec(self.recent_btn.mapToGlobal(self.recent_btn.rect().bottomLeft()))


class InputSummary(QFrame):
    """One line on what was found in the input folder (+ a mesh preview button), or the
    input's hint while it is empty."""

    preview_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.preview = Button("Preview", "eye", "ghost", size="sm", on_click=self.preview_requested.emit)
        self.preview.setToolTip("Look at the mesh, its regions and pacing sites in 3D")
        self.icon = IconLabel("info", 14, tone="faint")
        self.spinner = Spinner(14)
        self.text = WrapLabel("", "caption")
        self.text.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        # the icon, the first line of text and the Preview button share one centre line
        self.text.ensurePolished()
        centre = self.preview.sizeHint().height() // 2
        icon_col = QVBoxLayout()
        icon_col.setContentsMargins(0, centre - 7, 0, 0)
        icon_col.setSpacing(0)
        icon_col.addWidget(self.icon)
        icon_col.addWidget(self.spinner)
        icon_col.addStretch(1)
        lay.addLayout(icon_col)
        self.text.setContentsMargins(0, max(0, centre - self.text.fontMetrics().height() // 2), 0, 0)
        lay.addWidget(self.text, 1)
        lay.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignTop)
        self.hint = ""
        self.show_info(None)

    def show_busy(self, text: str = "Inspecting folder…") -> None:
        self.show()
        self.spinner.show()
        self.icon.hide()
        self.preview.hide()
        self.text.setText(text)

    def show_info(self, info: InputInfo | None, note: str = "") -> None:
        """What ``info`` found; ``note``: a demo's download note, shown for a folder that does
        not exist yet."""
        self.spinner.hide()
        self.icon.show()
        self.preview.setVisible(bool(info and info.has_mesh and not info.error))
        if info is None or not info.path:
            self.icon.set_icon("info", "faint")
            self.text.setText(self.hint)
            self.setVisible(bool(self.hint))
            return
        self.show()
        if info.error:
            self.icon.set_icon("alert", "warning")
            self.text.setText(info.error)
        elif not info.exists:
            self.icon.set_icon(*(("download", "info") if note else ("alert", "warning")))
            self.text.setText(note or "This folder does not exist on the machine.")
        elif info.has_mesh:
            self.icon.set_icon("mesh", "success")
            regions = f"{len(info.regions)} region{'s' if len(info.regions) != 1 else ''}"
            vtx = f"{len(info.vtx)} .vtx" if info.vtx else "no .vtx"
            types = "/".join(info.elem_types)
            self.text.setText(f"Mesh “{info.mesh_name}” · {number(info.n_nodes)} nodes · {number(info.n_elems)} "
                              f"elements ({types}) · {regions} · {vtx}"
                              + (f" · {len(info.surfaces)} .surf" if info.surfaces else ""))
        else:
            self.icon.set_icon("folder", "muted")
            parts = [f"{len(info.files)} files"]
            if info.subdirs:
                parts.append(f"{len(info.subdirs)} folders")
            self.text.setText("No openCARP mesh in this folder · " + " · ".join(parts))


class FolderSlot(QWidget):
    """A folder inside a panel: its label, with the machine it is on, and its field."""

    def __init__(self, title: str, field: FolderField) -> None:
        super().__init__()
        self.field = field
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.head = QHBoxLayout()
        self.head.setSpacing(6)
        self.head.addWidget(label(title, "field"))
        self.machine = label("", "caption")
        self.head.addWidget(self.machine)
        self.head.addStretch(1)
        lay.addLayout(self.head)
        lay.addWidget(field)

    def set_machine_name(self, words: str) -> None:
        """``words``: the machine as running text names it ("this computer", "gpu-box")."""
        self.machine.setText(f"on {words}" if words else "")

    def focus_target(self) -> QWidget:
        """Where an issue about the folder is shown and revealed."""
        return self.field.edit

    def set_issue(self, severity: str | None, message: str = "") -> None:
        """Tint the folder field for an issue about the folder (``None``: clear)."""
        edit = self.field.edit
        if (edit.property("issue") or "") != (severity or ""):
            edit.setProperty("issue", severity or "")
            repolish(edit)


class InputSlot(FolderSlot):
    """The input folder: a note when it is optional, what was found and a warning line."""

    def __init__(self, field: FolderField, summary: InputSummary) -> None:
        super().__init__("Input folder", field)
        self.optional = label("optional", "caption")
        self.optional.hide()
        self.head.addWidget(self.optional)
        lay = self.layout()
        lay.addWidget(summary)
        self.warning = QWidget()
        row = QHBoxLayout(self.warning)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(IconLabel("alert", 14, tone="warning"), 0, Qt.AlignmentFlag.AlignTop)
        self.warning_text = WrapLabel("", "caption")
        row.addWidget(self.warning_text, 1)
        self.warning.hide()
        lay.addWidget(self.warning)

    def set_optional(self, optional: bool) -> None:
        self.optional.setVisible(optional)

    def set_warning(self, text: str) -> None:
        self.warning_text.setText(text)
        self.warning.setVisible(bool(text))


class Folders(QObject):
    """The input (mesh / data) and output folders of a draft on the selected machine.

    The input starts where runs of the selected kind keep their data (see
    :meth:`_default_input`) and follows the demo applied (:meth:`use_demo`). It remembers
    where it came from: ``"default"``, ``"demo"`` or ``"user"`` (typed, browsed or picked
    from the recent folders). Only the user's choices are remembered, and the others are
    recomputed when the machine connects. Each machine also remembers the last output
    chosen (else the machine's default, else where its runs were last written).
    """

    input_changed = Signal(object)  # InputInfo | None (None at once when the folder changes)

    def __init__(self, ctx: AppContext, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.machine_id: str | None = None
        self.kind: str | None = None
        self.input_info: InputInfo | None = None
        self.source = "default"
        self._demo: Demo | None = None
        self._chosen = False  # the user chose the input since the demo was applied
        self._role: tuple[str, str] = ("optional", "")
        self._inspecting: tuple[str, str] | None = None  # (machine id, folder) in flight
        self.input = FolderField("Mesh or data folder",
                                 lambda cur: self._browse("dir", cur, "Choose the input folder"))
        self.input.changed.connect(self._input_edited)
        self.input_summary = InputSummary()
        self.input_summary.preview_requested.connect(self._preview_mesh)
        self.input_slot = InputSlot(self.input, self.input_summary)
        self.output = FolderField("Where run folders are created",
                                  lambda cur: self._browse("dir", cur, "Choose the output folder"))
        self.output.changed.connect(self._output_changed)
        self.output_slot = FolderSlot("Output folder", self.output)
        ctx.machine_state_changed.connect(self._on_state)

    # ------------------------------------------------------------------ public
    def input_dir(self) -> str:
        return self.input.text()

    def output_dir(self) -> str:
        return self.output.text()

    def set_machine(self, machine_id: str, *, input_dir: str | None = None, output_dir: str | None = None) -> None:
        """Show the folders of ``machine_id``: the given ones, else those it remembers."""
        profile = self.ctx.profile(machine_id)
        if profile is None:
            return
        switched = machine_id != self.machine_id
        if not switched and input_dir is None and output_dir is None:
            return  # the same machine again: keep what is in the fields
        if switched and self.machine_id is not None:
            # a typed path only sends editingFinished on focus loss, which a click on a
            # machine card does not always cause: keep it for when this machine comes back
            self._output_changed(self.output.text())
        self.machine_id = machine_id
        settings = self.ctx.settings
        words = machine_words(profile)
        for slot in (self.input_slot, self.output_slot):
            slot.set_machine_name(words)
            slot.field.set_machine_name(words)
        self.input.recent = list(settings.recent_inputs.get(machine_id, []))
        self.output.recent = _unique([*settings.recent_outputs.get(machine_id, []), profile.output_dir,
                                      *self.ctx.history.roots(machine_id)])
        old_input = self.input.text()
        self.source = "default" if input_dir is None else "user"
        if input_dir is None:
            default = self._default_input()
            input_dir = default if default is not None else (
                self.input.recent[0] if self.input.recent else profile.input_dir)
        self.input.set_text(input_dir)
        self.output.set_text(output_dir if output_dir is not None else
                             (self.output.recent[0] if self.output.recent else ""))
        if switched or self.input.text() != old_input:
            self.refresh_input()  # what was found belongs to the old machine / folder
        else:
            self._update_slot()

    def set_kind(self, kind: str, demo: Demo | None = None) -> None:
        """A run of another kind starts from that kind's input folder, when there is one
        (``demo``: the kind's active demo, whose data comes first)."""
        if kind == self.kind and demo is self._demo:
            return
        self.kind, self._demo = kind, demo
        if self.machine_id is None:
            return
        default = self._default_input()
        if default is not None:
            self.set_input(default, "default")
        self._update_slot()

    def set_demo(self, demo: Demo | None) -> None:
        """The kind's active demo (for its notes), without moving the input."""
        self._demo = demo
        self._chosen = self.source == "user"
        self._update_slot()

    def use_demo(self, demo: Demo) -> None:
        """A demo was applied: point the input at its data when the machine has it (or is not
        connected yet), clear it when the run downloads its data, else leave it."""
        self._demo = demo
        self._chosen = False
        data = self.demo_data(demo)
        if data.status in (DataStatus.FOUND, DataStatus.UNKNOWN):
            self.set_input(data.path, "demo")
        elif data.status == DataStatus.DOWNLOAD:
            self.set_input("", "demo")
        self._update_slot()

    def set_input(self, path: str, source: str) -> None:
        """Show ``path`` as the input (``source``: "default", "demo" or "user")."""
        self.source = source
        if path != self.input.text():
            self.input.set_text(path)
            self.refresh_input()
        else:
            self._update_slot()

    def set_role(self, role: str, hint: str) -> None:
        """What the input folder is to the current set-up: "required", "optional" or "unused"."""
        if (role, hint) != self._role:
            self._role = (role, hint)
            self.input_slot.set_optional(role == "optional")
            self.input_summary.hint = hint
            self._update_slot()

    def demo_data(self, demo: Demo) -> DemoData:
        """Where ``demo``'s data stands on the selected machine (no remote call)."""
        sess = self.ctx.session(self.machine_id) if self.machine_id else None
        return demo_data(demo, sess.data_folders if sess else None, bool(sess and sess.connected))

    def refresh_input(self) -> None:
        self._input_changed(self.input.text())

    # ------------------------------------------------------------------ internals
    def _default_input(self) -> str | None:
        """Where a run of the current kind starts on this machine (``None``: no opinion).

        The input last chosen for this kind; else the active demo's data (its folder, or
        nothing when its run downloads it); else the kind's sub-folder of the machine's data
        folder (``~/torchcor_data/ep``) or the case inside it that was used last.
        """
        if self.machine_id is None or self.kind is None:
            return None
        settings = self.ctx.settings
        remembered = settings.kind_inputs.get(self.machine_id, {}).get(self.kind)
        if remembered:
            return remembered
        if self._demo is not None and self._demo.case:
            data = self.demo_data(self._demo)
            if data.status == DataStatus.FOUND:
                return data.path
            if data.status == DataStatus.DOWNLOAD:
                return ""
        data = self.ctx.session(self.machine_id).data_folders
        folder = data.folder(self.kind) if data else None
        if folder is None:
            return None
        recent = settings.recent_inputs.get(self.machine_id, [])
        return next((p for p in recent if data.contains(folder, p)), folder)

    def _demo_target(self) -> str | None:
        """Where the active demo points the input now (``None``: it leaves it alone)."""
        if self._demo is None:
            return None
        data = self.demo_data(self._demo)
        if data.status in (DataStatus.FOUND, DataStatus.UNKNOWN):
            return data.path
        return "" if data.status == DataStatus.DOWNLOAD else None

    def _on_state(self, machine_id: str) -> None:
        if machine_id != self.machine_id:
            return
        if self.ctx.machine(machine_id) is not None:  # connected: the data folder is known now
            target = (self._default_input() if self.source == "default"
                      else self._demo_target() if self.source == "demo" else None)
            if target is not None and target != self.input.text():
                self.input.set_text(target)
                self.refresh_input()
                return
            if self.input_info is None and self.input.text() and \
                    self._inspecting != (machine_id, self.input.text()):
                self.refresh_input()
                return
        self._update_slot()

    def _input_edited(self, path: str) -> None:
        """Typed, browsed or picked from the recent folders: the user's choice."""
        self.source = "user"
        self._chosen = True
        self._input_changed(path)

    def _browse(self, mode: str, current: str, title: str) -> str | None:
        machine = self.ctx.machine(self.machine_id) if self.machine_id else None
        if machine is None:
            self.ctx.notify.emit("Not connected", "Wait for the machine to connect, then try again.", "warning")
            return None
        return browse_path(self.input_slot, machine, mode, current, title)

    def _input_changed(self, path: str) -> None:
        self.input_info = None
        self.input_changed.emit(None)  # what was found no longer applies
        if not path:
            self._update_slot()
            return
        if self.machine_id is None or self.ctx.machine(self.machine_id) is None:
            self.input_summary.show_busy("Waiting for the machine to connect…")
            return  # inspected once connected (_on_state)
        self.input_summary.show_busy()
        machine_id, requested, kind, chosen = self.machine_id, path, self.kind, self.source == "user"
        self._inspecting = (machine_id, requested)

        def done(info: InputInfo) -> None:
            if not isValid(self.input_slot):
                return  # the draft's widgets are gone (its tab closed)
            if self._inspecting == (machine_id, requested):
                self._inspecting = None
            if self.machine_id != machine_id or self.input.text() != requested:
                return  # the machine or the folder changed meanwhile
            self.input_info = info
            if chosen and info.exists and not info.error:  # defaults and demos are not remembered
                self.ctx.settings.remember_input(machine_id, requested, kind=kind)
                self.ctx.store.save_settings()
            self._update_slot()
            self.input_changed.emit(info)

        self.ctx.inspect_input(machine_id, path, done, owner=self)  # its tab may close first

    def _update_slot(self) -> None:
        """The summary line and the demo's notes under the input field: what the folder
        holds (or the role's hint while it is empty), a note when the run downloads the
        demo's data into it, and a warning when the demo's data is missing and no folder was
        chosen since (then the warning alone, not the hint as well)."""
        demo = self._demo
        data = self.demo_data(demo) if demo is not None and demo.case else None
        note = (f"{demo.data or 'Its data'} will be downloaded here on the first run, then reused."
                if demo is not None and demo.download and self._role[0] == "optional" else "")
        missing = data is not None and data.status == DataStatus.MISSING and not self._chosen
        if missing and not self.input.text():
            self.input_summary.hide()
        elif self.input_info is not None or not self.input.text():
            self.input_summary.show_info(self.input_info, note)
        profile = self.ctx.profile(self.machine_id) if missing and self.machine_id else None
        self.input_slot.set_warning(
            f"This demo reads {data.path}, which is not on {machine_words(profile) if profile else 'the machine'}: "
            "copy TorchCor's demo data there, or choose its folder." if missing else "")

    def _output_changed(self, path: str) -> None:
        if not (self.machine_id and path):
            return
        self.ctx.history.add_root(self.machine_id, path)  # runs written there show in history
        self.ctx.settings.remember_output(self.machine_id, path)
        self.ctx.store.save_settings()
        self.output.recent = _unique([path, *self.output.recent])

    def _preview_mesh(self) -> None:
        machine = self.ctx.machine(self.machine_id) if self.machine_id else None
        if machine is None or not self.input.text():
            return
        from torchcor_studio.ui.dialogs.mesh_preview import MeshPreviewDialog

        MeshPreviewDialog(machine, self.input.text(), self.input_slot).exec()
