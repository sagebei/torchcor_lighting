"""Choose the Python interpreter that runs TorchCor on a machine."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.ui import icons
from torchcor_studio.ui.browse import browse_path
from torchcor_studio.ui.tasks import run_async
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import Button, Spinner, label


class PythonPicker(QDialog):
    def __init__(self, ctx, machine_id: str, parent: QWidget | None = None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.ctx = ctx
        self.machine_id = machine_id
        self.machine = ctx.machine(machine_id)
        self.chosen: str | None = None
        self.setWindowTitle("Python environment")
        self.resize(640, 460)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(12)
        lay.addWidget(label("Python environment", "title"))
        lay.addWidget(label("Pick the interpreter that has PyTorch and TorchCor installed. Environments found on "
                            f"{self.machine.name if self.machine else 'the machine'} are listed below.",
                            "muted", wrap=True))
        busy = QHBoxLayout()
        self.spinner = Spinner(16)
        busy.addWidget(self.spinner)
        self.status = label("Searching for Python environments…", "caption")
        busy.addWidget(self.status, 1)
        lay.addLayout(busy)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda *_: self._accept())
        self.list.setStyleSheet("QListWidget::item { padding: 8px 6px; }")
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(Button("Browse…", "folder-open", "ghost", on_click=self._browse))
        row.addStretch(1)
        row.addWidget(Button("Cancel", variant="ghost", on_click=self.reject))
        self.ok = Button("Use this Python", "check", "primary", on_click=self._accept)
        row.addWidget(self.ok)
        lay.addLayout(row)
        self._search()

    def _search(self) -> None:
        if self.machine is None:
            self.status.setText("The machine is not connected.")
            self.spinner.hide()
            return
        from torchcor_studio.core.environment import find_pythons

        run_async(find_pythons, self.machine, on_done=self._show, on_error=self._failed, owner=self)

    def _show(self, candidates) -> None:  # noqa: ANN001 - ranked best first by find_pythons
        self.spinner.hide()
        p = theme().palette
        current = self.machine.python if self.machine else ""
        ranked = list(candidates)
        for cand in ranked:
            item = QListWidgetItem(icons.icon("check-circle" if cand.ready else "terminal",
                                              p.success if cand.ready else p.text3, 16),
                                   f"{cand.path}\n{cand.summary.replace(' · ', '  ·  ')}")
            item.setData(Qt.ItemDataRole.UserRole, cand.path)
            self.list.addItem(item)
            if cand.path == current:
                self.list.setCurrentItem(item)
        self.status.setText(f"Found {len(ranked)} interpreter{'s' if len(ranked) != 1 else ''}."
                            if ranked else "No Python interpreters found — browse to one.")
        if self.list.currentItem() is None and self.list.count():
            self.list.setCurrentRow(0)

    def _failed(self, exc) -> None:  # noqa: ANN001
        self.spinner.hide()
        self.status.setText(f"Search failed: {exc}")

    def _browse(self) -> None:
        path = browse_path(self, self.machine, "file", self.machine.python if self.machine else "",
                           "Choose a Python interpreter")
        if path:
            self.chosen = path
            self.accept()

    def _accept(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            self.chosen = item.data(Qt.ItemDataRole.UserRole)
            self.accept()


def pick_python(parent: QWidget, ctx, machine_id: str) -> None:  # noqa: ANN001
    """Show the picker; on a choice, persist it in the machine profile and re-probe."""
    if ctx.machine(machine_id) is None:
        ctx.notify.emit("Not connected", "Connect to the machine first.", "warning")
        return
    dlg = PythonPicker(ctx, machine_id, parent)
    if dlg.exec() and dlg.chosen:
        apply_python(ctx, machine_id, dlg.chosen)


def apply_python(ctx, machine_id: str, python: str) -> None:  # noqa: ANN001
    """Use ``python`` on this machine from now on (saved in its profile) and re-check."""
    profile = ctx.profile(machine_id)
    profile.python = python
    ctx.store.upsert_machine(profile)
    ctx.reprobe(machine_id, python=python)
    ctx.notify.emit("Python updated", python, "success")
