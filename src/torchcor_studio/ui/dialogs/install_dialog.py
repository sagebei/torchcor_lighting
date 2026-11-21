"""Install (or upgrade) TorchCor into a Python environment, with live pip output.

``InstallDialog(machine, python, settings.torchcor_requirement()).exec()`` starts at
once; :attr:`InstallDialog.succeeded` and :attr:`InstallDialog.report` tell the caller
how it went once the dialog closes.
"""

from __future__ import annotations

import threading
from collections import deque

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QTextCursor, QTextOption
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPlainTextEdit, QVBoxLayout, QWidget

from torchcor_studio.core.environment import install_torchcor
from torchcor_studio.core.machines.base import Machine
from torchcor_studio.core.models import MIN_TORCHCOR, EnvironmentReport
from torchcor_studio.ui.tasks import run_async
from torchcor_studio.ui.theme import mono_font
from torchcor_studio.ui.widgets import Button, IconLabel, Spinner, label
from torchcor_studio.ui.widgets.gpu import IconTile

#: Known pip failures and what to tell the user about them.
_HINTS = (
    ("externally-managed-environment",
     "This Python is managed by the operating system. Create a virtual environment for "
     "TorchCor and install into that instead."),
    ("No matching distribution found for torch",
     "PyTorch has no build for this Python version or platform. Try a Python between 3.10 "
     "and 3.13."),
    ("Failed to establish a new connection",
     "The machine could not reach the package index. Check its internet access or proxy."),
    ("Temporary failure in name resolution",
     "The machine could not reach the package index. Check its internet access or proxy."),
    ("Permission denied",
     "pip could not write to this environment. Use a virtual environment you own."),
    ("Cannot find command 'git'",
     "git is not installed on the machine. Install git, or switch the TorchCor source to PyPI."),
)


def failure_hint(output: str) -> str | None:
    """Advice for a failed ``pip install``, recognised from its output."""
    return next((hint for needle, hint in _HINTS if needle in output), None)


class InstallDialog(QDialog):
    """Streams ``pip install --upgrade <requirement>`` into a console view."""

    #: Emitted with the pip exit code when the installation ends (-1 if cancelled).
    install_finished = Signal(int)

    def __init__(self, machine: Machine, python: str, requirement: str, *, with_torch: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.machine = machine
        self.python = python
        self.requirement = requirement
        self.with_torch = with_torch
        self._stop = threading.Event()
        self._pending: deque[str] = deque()
        self._started = False
        self._running = False
        self._output: list[str] = []
        self.exit_code: int | None = None
        self.report: EnvironmentReport | None = None
        self.setWindowTitle("Install TorchCor")
        self.setModal(True)
        self.resize(760, 520)
        self.setMinimumSize(560, 380)
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(80)
        self._flush_timer.timeout.connect(self._flush)
        self._build()

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

    # ------------------------------------------------------------------ ui
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        head = QHBoxLayout()
        head.setSpacing(12)
        head.addWidget(IconTile("package", "accent", size=40), 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(label("Install TorchCor", "title"))
        where = self.machine.profile.name
        col.addWidget(label(f"Into {self.python} on {where}", "caption", wrap=True, selectable=True))
        head.addLayout(col, 1)
        root.addLayout(head)

        self._console = QPlainTextEdit()
        self._console.setProperty("console", True)
        self._console.setReadOnly(True)
        self._console.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self._console.setMaximumBlockCount(5000)
        self._console.setFont(mono_font())
        root.addWidget(self._console, 1)

        foot = QHBoxLayout()
        foot.setSpacing(10)
        self._spinner = Spinner(16)
        foot.addWidget(self._spinner)
        self._icon = IconLabel("check-circle", 18, tone="success")
        self._icon.hide()
        foot.addWidget(self._icon)
        status = QVBoxLayout()
        status.setSpacing(2)
        self._status = label("Preparing…")
        self._status.setStyleSheet("font-weight: 600;")
        status.addWidget(self._status)
        self._detail = label("", "muted", wrap=True)
        self._detail.hide()
        status.addWidget(self._detail)
        foot.addLayout(status, 1)
        self._cancel = Button("Cancel", variant="secondary", on_click=self.cancel)
        foot.addWidget(self._cancel, 0, Qt.AlignmentFlag.AlignTop)
        self._close = Button("Done", variant="primary", on_click=self.accept)
        self._close.hide()
        foot.addWidget(self._close, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(foot)

    # ------------------------------------------------------------------ flow
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._started:
            self.start()

    def start(self) -> None:
        """Begin the installation (called automatically when the dialog is shown)."""
        if self._started:
            return
        self._started = self._running = True
        self._status.setText("Installing TorchCor…")
        self._flush_timer.start()
        run_async(install_torchcor, self.machine, self.python, self.requirement,
                  self._pending.append, self._stop, with_torch=self.with_torch,
                  on_done=self._finished, on_error=self._crashed, owner=self)

    def cancel(self) -> None:
        if self._running:
            self._stop.set()
            self._cancel.setEnabled(False)
            self._status.setText("Cancelling…")
        else:
            self.reject()

    def reject(self) -> None:
        self._stop.set()  # closing the window stops pip too
        super().reject()

    def _flush(self) -> None:
        if not self._pending:
            return
        lines = []
        while self._pending:
            lines.append(self._pending.popleft())
        self._output.extend(lines)
        bar = self._console.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 4
        self._console.appendPlainText("\n".join(lines))
        if follow:
            self._console.moveCursor(QTextCursor.MoveOperation.End)
            bar.setValue(bar.maximum())

    def _finished(self, code: int) -> None:
        self._flush()
        self._running = False
        self._flush_timer.stop()
        self.exit_code = code
        self._cancel.hide()
        self._close.show()
        self._close.setFocus()
        if code == 0:
            self._status.setText("Checking the installation…")
            run_async(self.machine.probe_environment, self.python,
                      on_done=self._verified, on_error=lambda exc: self._verified(None), owner=self)
        elif code == -1:
            self._end("x-circle", "neutral", "Installation cancelled",
                      "Nothing was changed if pip had not started installing yet.")
            self._close.setText("Close")
        else:
            self._end("x-circle", "danger", f"Installation failed (pip exited with code {code})",
                      failure_hint("\n".join(self._output)) or "See the output above for details.")
            self._close.setText("Close")
        self.install_finished.emit(code)

    def _crashed(self, exc: BaseException) -> None:
        self._flush()
        self._running = False
        self._flush_timer.stop()
        self.exit_code = 1
        self._cancel.hide()
        self._close.setText("Close")
        self._close.show()
        self._end("x-circle", "danger", "Installation could not start", str(exc))
        self.install_finished.emit(1)

    def _verified(self, report: EnvironmentReport | None) -> None:
        self.report = report
        if report is not None and report.torchcor_outdated:
            self._end("alert", "warning", f"Installed TorchCor {report.torchcor_version}, which is too old",
                      f"This app needs TorchCor {'.'.join(map(str, MIN_TORCHCOR))} or newer. The source in "
                      "Settings → TorchCor provides an older version: point it at a newer one "
                      "(a branch, or a local folder on that machine) and install again.")
        elif report is not None and report.torchcor_version:
            torch = f"PyTorch {report.torch_version}" if report.torch_version else "PyTorch missing"
            cuda = f" · CUDA {report.cuda_version}" if report.cuda_available and report.cuda_version else ""
            self._end("check-circle", "success", f"TorchCor {report.torchcor_version} is ready",
                      f"{torch}{cuda} · Python {report.python_version}")
        else:
            detail = report.error if report is not None and report.error else \
                "pip finished, but TorchCor could not be imported afterwards."
            self._end("alert", "warning", "Installed, but not verified", detail)

    def _end(self, icon: str, tone: str, title: str, detail: str) -> None:
        self._spinner.hide()
        self._icon.set_icon(icon, tone=tone)
        self._icon.show()
        self._status.setText(title)
        self._detail.setText(detail)
        self._detail.setVisible(bool(detail))
