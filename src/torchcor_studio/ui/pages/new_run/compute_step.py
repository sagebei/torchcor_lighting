"""Step 1 — where to run: the machine, the device on it, its Python environment and the
demo data on it (downloaded from GitHub by the runtime's ``fetch-data``).

The input and output folders are chosen on the Setup step (``folders.py``)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.core.models import MIN_TORCHCOR, EnvironmentReport, MachineProfile
from torchcor_studio.ui.browse import browse_path
from torchcor_studio.ui.context import AppContext, MachineSession, MachineState
from torchcor_studio.ui.format import plural, size
from torchcor_studio.ui.pages.new_run.device_picker import DevicePicker
from torchcor_studio.ui.pages.new_run.folders import FolderField, machine_words
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import (
    Badge,
    Button,
    Card,
    IconLabel,
    SectionHeader,
    Spinner,
    WrapLabel,
    discard,
    label,
)
from torchcor_studio.ui.widgets.layout import ResponsiveGrid
from torchcor_studio.ui.widgets.steps import SelectCard


def machine_caption(profile: MachineProfile) -> str:
    return "Local machine" if profile.is_local else profile.address


def state_html(state: MachineState) -> str:
    """Rich-text connection state with a coloured dot."""
    p = theme().palette
    color, text = {
        MachineState.CONNECTED: (p.success, "Connected"),
        MachineState.CONNECTING: (p.accent, "Connecting…"),
        MachineState.ERROR: (p.danger, "Connection failed"),
        MachineState.DISCONNECTED: (p.text3, "Not connected"),
    }[state]
    return f'<span style="color:{color}">●</span>&nbsp; {text}'


def devices_html(env: EnvironmentReport) -> str:
    """What the machine card says about compute devices, after the connection state."""
    if env.cuda_device_count:
        return f" · {plural(env.cuda_device_count, 'GPU')}"
    if env.nvidia_gpus:
        return f' · <span style="color:{theme().palette.warning}">{plural(env.nvidia_gpus, "GPU")} not usable</span>'
    return " · CPU only"


def environment_message(env: EnvironmentReport) -> str:
    """One sentence: can this environment run TorchCor here, and if not, what to do."""
    host = f" on {env.hostname}" if env.hostname else ""
    gpus = plural(env.nvidia_gpus, "NVIDIA GPU")
    if env.ready:
        if env.unusable_gpus:
            return (f"Ready, but on the CPU only: this PyTorch can't use the {gpus}{host}. "
                    "Choose a Python with CUDA PyTorch to use them.")
        return f"Ready to simulate{host}."
    if not env.python_supported:
        return (f"Python {env.python_version or '?'} is too old for TorchCor (it needs 3.10 or newer)"
                + (f" and can't use the {gpus}" if env.nvidia_gpus else "") + ". Choose another Python.")
    if env.torchcor_outdated:
        return (f"TorchCor {env.torchcor_version} is older than this app supports "
                f"({'.'.join(map(str, MIN_TORCHCOR))} or newer). Update TorchCor, or choose another Python.")
    if env.cuda_available:
        return f"PyTorch can use the {plural(env.cuda_device_count, 'GPU')}. Install TorchCor to start simulating."
    if env.torch_version is None:
        return "This Python has no PyTorch yet. Install TorchCor (it brings PyTorch), or choose another Python."
    if env.unusable_gpus:
        return (f"This PyTorch is a CPU-only build and can't use the {gpus}. "
                "Choose a Python with CUDA PyTorch, or install TorchCor here to run on the CPU.")
    return "TorchCor is not available in this environment yet — install it or pick another Python."


class PythonSuggestion(QFrame):
    """A better-suited Python found on the machine, with a button that switches to it."""

    use_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)
        self.button = Button("Use this Python", "check", "subtle", size="sm",
                             on_click=lambda: self.use_requested.emit(self.path))
        # the icon and the first line of text sit on the button's centre line
        centre = self.button.sizeHint().height() // 2
        icon_col = QVBoxLayout()
        icon_col.setContentsMargins(0, centre - 7, 0, 0)
        icon_col.addWidget(IconLabel("sparkles", 14, tone="accent"))
        icon_col.addStretch(1)
        lay.addLayout(icon_col)
        self.text = WrapLabel("", "caption")
        self.text.setContentsMargins(0, max(0, centre - self.text.fontMetrics().height() // 2), 0, 0)
        lay.addWidget(self.text, 1, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.button, 0, Qt.AlignmentFlag.AlignTop)
        self.path = ""
        self.hide()

    def show_candidate(self, candidate) -> None:  # noqa: ANN001 - PythonCandidate | None
        self.path = candidate.path if candidate is not None else ""
        self.setVisible(candidate is not None)
        if candidate is not None:
            self.text.setText(f"Better match here: <b>{candidate.env_name}</b> — {candidate.summary}")
            self.text.setToolTip(candidate.path)


class MachinePicker(QWidget):
    """Row of machine cards plus an "Add machine" card."""

    selected = Signal(str)
    add_requested = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._cards: dict[str, SelectCard] = {}
        self._current: str | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        # the same column grid as the device tiles below (GpuGrid, compact)
        self._grid = ResponsiveGrid(min_item=230, max_columns=4, spacing=12, fill=False)
        outer.addWidget(self._grid)
        ctx.machines_changed.connect(self.rebuild)
        ctx.machine_state_changed.connect(self._state_changed)
        ctx.environment_changed.connect(self._state_changed)  # e.g. another Python was chosen
        self.rebuild()

    def rebuild(self) -> None:
        self._cards.clear()
        cards: list[QWidget] = []
        for prof in self.ctx.profiles():
            card = SelectCard("laptop" if prof.is_local else "server", prof.name, machine_caption(prof),
                              compact=True)
            card.clicked.connect(lambda mid=prof.id: self.select(mid))
            self._cards[prof.id] = card
            cards.append(card)
        add = SelectCard("plus", "Add machine", "Connect a GPU server over SSH", tone="neutral", compact=True)
        add.clicked.connect(self.add_requested.emit)
        cards.append(add)
        self._grid.set_widgets(cards)
        for machine_id in self._cards:
            self._state_changed(machine_id)
        if self._current in self._cards:
            self._cards[self._current].set_checked(True)

    def select(self, machine_id: str, emit: bool = True) -> None:
        self._current = machine_id
        for mid, card in self._cards.items():
            card.set_checked(mid == machine_id)
        if emit:
            self.selected.emit(machine_id)

    def _state_changed(self, machine_id: str) -> None:
        card = self._cards.get(machine_id)
        if card is None:
            return
        sess = self.ctx.session(machine_id)
        extra = ""
        if sess.state == MachineState.CONNECTED and sess.env is not None:
            extra = devices_html(sess.env)
            if sess.profile.is_local and sess.env.hostname:
                card.caption.setText(sess.env.hostname)
        card.footer.setText(state_html(sess.state) + extra)
        card.footer.setVisible(True)


class EnvironmentCard(Card):
    """Python / PyTorch / CUDA / TorchCor readiness with install & re-check actions."""

    install_requested = Signal()
    python_requested = Signal()
    use_python_requested = Signal(str)  # a suggested interpreter
    recheck_requested = Signal()

    def __init__(self) -> None:
        super().__init__()  # titled by the "Environment" section header above it
        self.status_row = QHBoxLayout()
        self.status_row.setSpacing(8)
        self.spinner = Spinner(16)
        self.message = label("", "muted", wrap=True)
        self.status_row.addWidget(self.spinner)
        self.status_row.addWidget(self.message, 1)
        self.addLayout(self.status_row)
        self.badges = QHBoxLayout()
        self.badges.setSpacing(6)
        self.addLayout(self.badges)
        self.python_label = label("", "mono")
        self.python_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.addWidget(self.python_label)
        self.suggestion = PythonSuggestion()
        self.suggestion.use_requested.connect(self.use_python_requested.emit)
        self.addWidget(self.suggestion)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.install_btn = Button("Install TorchCor", "download", "subtle", size="sm",
                                  on_click=self.install_requested.emit)
        self.python_btn = Button("Change Python…", "terminal", "ghost", size="sm",
                                 on_click=self.python_requested.emit)
        self.recheck_btn = Button("Re-check", "refresh", "ghost", size="sm",
                                  on_click=self.recheck_requested.emit)
        actions.addWidget(self.install_btn)
        actions.addWidget(self.python_btn)
        actions.addWidget(self.recheck_btn)
        actions.addStretch(1)
        self.addLayout(actions)

    def _clear_badges(self) -> None:
        while self.badges.count():
            item = self.badges.takeAt(0)
            if item.widget():
                discard(item.widget())

    def show_state(self, state: MachineState, error: str, env: EnvironmentReport | None,
                   suggestion=None) -> None:  # noqa: ANN001 - PythonCandidate | None
        self._clear_badges()
        busy = state == MachineState.CONNECTING
        self.spinner.setVisible(busy)
        self.install_btn.setVisible(False)
        self.python_btn.setEnabled(state == MachineState.CONNECTED)
        self.recheck_btn.setEnabled(not busy)
        self.python_label.setVisible(False)
        self.suggestion.show_candidate(suggestion if state == MachineState.CONNECTED else None)
        if busy:
            self.message.setText("Connecting and checking the Python environment…")
            return
        if state == MachineState.ERROR:
            self.message.setText(f"Could not connect: {error}")
            self.message.setProperty("role", "error")
            self.message.style().polish(self.message)
            return
        self.message.setProperty("role", "muted")
        self.message.style().polish(self.message)
        if state != MachineState.CONNECTED or env is None:
            self.message.setText("Select a machine to check its environment.")
            return
        if env.error:
            self.message.setText(env.error)
            self.python_btn.setEnabled(True)
            return
        self.python_label.setText(env.python)
        self.python_label.setVisible(bool(env.python))
        self.badges.addWidget(Badge(f"Python {env.python_version}", "neutral" if env.python_supported else "danger"))
        if env.torch_version:
            self.badges.addWidget(Badge(f"PyTorch {env.torch_version}", "neutral"))
        else:
            self.badges.addWidget(Badge("PyTorch missing", "danger", dot=True))
        if env.cuda_available:
            self.badges.addWidget(Badge(f"CUDA {env.cuda_version or ''} · {plural(env.cuda_device_count, 'GPU')}",
                                        "success", dot=True))
        elif env.nvidia_gpus:
            self.badges.addWidget(Badge(f"{plural(env.nvidia_gpus, 'GPU')} not usable", "warning", dot=True))
        elif env.torch_version:
            self.badges.addWidget(Badge("CPU only", "neutral", dot=True))
        if env.torchcor_outdated:
            self.badges.addWidget(Badge(f"TorchCor {env.torchcor_version} too old", "warning", dot=True))
            self.install_btn.setText("Update TorchCor")
        elif env.torchcor_version:
            self.badges.addWidget(Badge(f"TorchCor {env.torchcor_version}", "accent", icon="check"))
            self.install_btn.setText("Update TorchCor")
        else:
            self.badges.addWidget(Badge("TorchCor not installed", "danger", dot=True))
            self.install_btn.setText("Install TorchCor")
        self.badges.addStretch(1)
        self.install_btn.setVisible(env.python_supported or bool(env.torchcor_version))
        self.message.setText(environment_message(env))


class DemoDataCard(Card):
    """TorchCor's demo data on the machine: its folder and version when it is there, else a
    folder to download it into; then the download's progress, with Cancel."""

    download_requested = Signal(str)  # the folder it goes into (as <folder>/torchcor_data)
    cancel_requested = Signal()

    def __init__(self, browse) -> None:  # noqa: ANN001 - (current path) -> chosen path | None
        super().__init__()  # titled by the "Demo data" section header above it
        status = QHBoxLayout()
        status.setSpacing(8)
        self.icon = IconLabel("info", 16, tone="faint")
        self.message = label("", "muted", wrap=True)
        status.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        status.addWidget(self.message, 1)
        self.addLayout(status)
        self.path_label = label("", "mono")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.addWidget(self.path_label)
        self.folder = FolderField("Folder to download into", browse)
        self.folder.recent_btn.hide()  # a download goes where it is told, not where runs went
        self.folder.path_changed.connect(self._show_target)
        self.addWidget(self.folder)
        self.target = label("", "caption")
        self.addWidget(self.target)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.addWidget(self.bar)
        self.progress = label("", "caption")
        self.addWidget(self.progress)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.download_btn = Button("Download", "download", "subtle", size="sm",
                                   on_click=lambda: self.download_requested.emit(self.folder.text().strip()))
        self.cancel_btn = Button("Cancel", "x", "ghost", size="sm", on_click=self.cancel_requested.emit)
        actions.addWidget(self.download_btn)
        actions.addWidget(self.cancel_btn)
        actions.addStretch(1)
        self.addLayout(actions)

    def set_folder(self, folder: str) -> None:
        """Where a download goes, until the user picks another folder."""
        self.folder.set_text(folder)
        self._show_target(folder)

    def _show_target(self, folder: str) -> None:
        folder = folder.strip().rstrip("/\\")
        self.target.setText(f"The data goes into {folder}/torchcor_data." if folder else "")

    def _say(self, text: str, icon: str, tone: str, role: str = "muted") -> None:
        self.icon.set_icon(icon, tone)
        self.message.setText(text)
        if self.message.property("role") != role:
            self.message.setProperty("role", role)
            self.message.style().polish(self.message)

    def show_session(self, sess: MachineSession) -> None:
        job, data = sess.download, sess.data_folders
        active = job is not None and job.active
        self.bar.setVisible(active)
        self.progress.setVisible(active)
        self.cancel_btn.setVisible(active)
        self.download_btn.setVisible(not active)
        self.download_btn.setEnabled(sess.connected)
        # with the data there, its folder is shown, and the latest goes over it
        self.folder.setVisible(data is None)
        self.target.setVisible(data is None)
        self.folder.setEnabled(sess.connected and not active)
        self.path_label.setVisible(data is not None and not active)
        self.download_btn.setText("Download the latest" if data is not None else "Download")
        if active:
            what = {"download": "Downloading", "unzip": "Unzipping"}.get(job.stage, "Finding the latest release")
            version = job.version.removeprefix("data-") if job.version else ""
            self._say(f"{what} the demo data{f' {version}' if version else ''}…", "download", "accent")
            self.bar.setValue(int(1000 * job.done / job.total) if job.total else 0)
            self.bar.setRange(0, 1000 if job.total else 0)  # busy until the size is known
            self.progress.setText(f"{size(job.done)} of {size(job.total)}" if job.total else "")
            return
        if not sess.connected:
            self._say("Connect to the machine to see its demo data.", "info", "faint")
            self.path_label.hide()
            return
        if job is not None and job.stage == "failed":
            self._say(f"Could not download the demo data: {job.error}", "alert", "danger", "error")
        elif job is not None and job.stage == "cancelled":
            self._say("The download was cancelled.", "info", "faint")
        elif data is not None:
            version = data.version.removeprefix("data-")
            self._say(f"The demos read the demo data{f' {version}' if version else ''} in this folder; "
                      "each writes its runs to the matching output folder.", "check-circle", "success")
            self.path_label.setText(data.root)
        else:
            self._say("TorchCor's example meshes and models for the demos. Download them from "
                      "GitHub onto this machine.", "info", "faint")


class ComputeStep(QWidget):
    """Machine → the device to run on → its Python environment → the demo data on it."""

    machine_changed = Signal(str)
    ready_changed = Signal(bool)
    add_machine_requested = Signal()
    install_requested = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.machine_id: str | None = None
        self._watching: str | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(20)

        lay.addWidget(SectionHeader("Compute machine", "Run on this computer or on a GPU server over SSH."))
        self.picker = MachinePicker(ctx)
        self.picker.selected.connect(self.set_machine)
        self.picker.add_requested.connect(self.add_machine_requested.emit)
        lay.addWidget(self.picker)

        self.devices = DevicePicker()  # the "Devices" section: the run's device
        lay.addWidget(self.devices)

        lay.addWidget(SectionHeader("Environment", "The Python interpreter that runs TorchCor on this machine."))
        self.env = EnvironmentCard()
        self.env.recheck_requested.connect(lambda: self.machine_id and ctx.connect_machine(self.machine_id, reprobe=True))
        self.env.install_requested.connect(self.install_requested.emit)
        self.env.python_requested.connect(self._pick_python)
        self.env.use_python_requested.connect(self._use_python)
        lay.addWidget(self.env)

        lay.addWidget(SectionHeader("Demo data", "The example meshes and models the demos read, "
                                                 "and where their runs go."))
        self.data = DemoDataCard(self._browse_data)
        self.data.download_requested.connect(self._download_data)
        self.data.cancel_requested.connect(lambda: self.machine_id and ctx.cancel_demo_data(self.machine_id))
        lay.addWidget(self.data)
        lay.addStretch(1)

        ctx.machine_state_changed.connect(self._on_state)
        ctx.environment_changed.connect(self._on_state)
        ctx.demo_data_changed.connect(self._on_data)
        ctx.telemetry.connect(self._on_telemetry)

    # ------------------------------------------------------------------ public
    def activate(self) -> None:
        if self.machine_id is None:
            self.set_machine(self.ctx.settings.last_machine_id or self.ctx.default_machine_id())
        elif self._watching is None:
            self._watch(self.machine_id)

    def deactivate(self) -> None:
        self._watch(None)

    def is_ready(self) -> bool:
        if self.machine_id is None:
            return False
        sess = self.ctx.session(self.machine_id)
        return sess.connected and sess.env is not None and sess.env.ready

    def cuda_usable(self) -> bool:
        """Can the selected environment's PyTorch run on NVIDIA GPUs?"""
        env = self.ctx.session(self.machine_id).env if self.machine_id else None
        return env is not None and env.cuda_available

    def set_machine(self, machine_id: str) -> None:
        if self.ctx.profile(machine_id) is None:
            machine_id = self.ctx.default_machine_id()
        if machine_id != self.machine_id:
            self.devices.clear()
        switched = machine_id != self.machine_id
        self.machine_id = machine_id
        self.picker.select(machine_id, emit=False)
        self.ctx.settings.last_machine_id = machine_id
        if samples := self.ctx.session(machine_id).samples:
            self.devices.update_samples(samples)
        self._watch(machine_id)
        if switched:
            self.data.set_folder(self._data_parent(machine_id))
            self.data.folder.set_machine_name(machine_words(self.ctx.profile(machine_id)))
        self.ctx.connect_machine(machine_id)
        self._on_state(machine_id)
        self.machine_changed.emit(machine_id)

    # ------------------------------------------------------------------ internals
    def _watch(self, machine_id: str | None) -> None:
        if self._watching == machine_id:
            return
        if self._watching is not None:
            self.ctx.unwatch_telemetry(self._watching)
        self._watching = machine_id
        if machine_id is not None:
            self.ctx.watch_telemetry(machine_id)

    def _on_state(self, machine_id: str) -> None:
        if machine_id != self.machine_id:
            return
        sess = self.ctx.session(machine_id)
        self.env.show_state(sess.state, sess.error, sess.env, sess.suggestion)
        self.data.show_session(sess)
        self.ready_changed.emit(self.is_ready())

    def _on_data(self, machine_id: str) -> None:
        if machine_id != self.machine_id:
            return
        sess = self.ctx.session(machine_id)
        if sess.download is not None and sess.download.stage == "done":
            self.data.set_folder(self._data_parent(machine_id))
        self.data.show_session(sess)

    def _data_parent(self, machine_id: str) -> str:
        """Where the data folder is (or would go by default): the folder that holds it."""
        sess = self.ctx.session(machine_id)
        root = sess.data_folders.root if sess.data_folders is not None else sess.profile.data_dir
        if not root:
            return "~"
        return root.rstrip("/\\").rsplit("/", 1)[0].rsplit("\\", 1)[0] or "/"

    def _browse_data(self, current: str) -> str | None:
        machine = self.ctx.machine(self.machine_id) if self.machine_id else None
        return browse_path(self, machine, "dir", current, "Choose where to put the demo data")

    def _download_data(self, folder: str) -> None:
        if self.machine_id and folder:
            self.ctx.fetch_demo_data(self.machine_id, folder)

    def _on_telemetry(self, machine_id: str, samples) -> None:  # noqa: ANN001
        if machine_id == self.machine_id:
            self.devices.update_samples(samples)

    def _pick_python(self) -> None:
        machine = self.ctx.machine(self.machine_id) if self.machine_id else None
        if machine is None:
            return
        from torchcor_studio.ui.dialogs.python_picker import pick_python

        pick_python(self, self.ctx, self.machine_id)

    def _use_python(self, python: str) -> None:
        if self.machine_id and python:
            from torchcor_studio.ui.dialogs.python_picker import apply_python

            apply_python(self.ctx, self.machine_id, python)
