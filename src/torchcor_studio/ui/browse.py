"""Pick a file or folder on a machine: native dialog locally, remote browser over SSH."""

from __future__ import annotations

from PySide6.QtWidgets import QFileDialog, QWidget

from torchcor_studio.core.machines import Machine


def browse_path(parent: QWidget, machine: Machine | None, mode: str = "dir", current: str = "",
                title: str | None = None) -> str | None:
    """Return the chosen path (in the machine's convention) or ``None``."""
    if machine is None:
        return None
    title = title or ("Choose a folder" if mode == "dir" else "Choose a file")
    start = current or machine.profile.input_dir or ""
    if machine.profile.is_local:
        start = machine.expand(start) if start else machine.home()
        if mode == "dir":
            path = QFileDialog.getExistingDirectory(parent, title, start)
        else:
            path, _ = QFileDialog.getOpenFileName(parent, title, start)
        return path or None
    from torchcor_studio.ui.dialogs.remote_browser import RemoteBrowserDialog

    return RemoteBrowserDialog.get_path(machine, start or "~", mode=mode, title=title, parent=parent)
