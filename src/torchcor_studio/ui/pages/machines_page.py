"""Machines: manage this computer and SSH hosts, inspect environments and live GPUs."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QVBoxLayout, QWidget

from torchcor_studio.core.models import MachineProfile
from torchcor_studio.ui.context import AppContext, MachineState
from torchcor_studio.ui.format import torch_summary, torchcor_summary
from torchcor_studio.ui.pages.common import PageHeader, scroll
from torchcor_studio.ui.pages.new_run.compute_step import machine_caption, state_html
from torchcor_studio.ui.widgets import (
    Badge,
    Button,
    Card,
    KeyValueGrid,
    SectionHeader,
    discard,
    label,
)
from torchcor_studio.ui.widgets.layout import PanelSplitter
from torchcor_studio.ui.widgets.steps import SelectCard


class MachinesPage(QWidget):
    install_requested = Signal(str)
    python_requested = Signal(str)

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.selected: str | None = None
        self._watching: str | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        header = PageHeader("Machines", "Where simulations run. Connections are opened on demand.")
        header.actions.addWidget(Button("Add machine", "plus", "primary", on_click=self.add_machine))
        lay.addWidget(header)

        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.list_col = QVBoxLayout()
        self.list_col.setContentsMargins(0, 0, 15, 0)  # with the 5 px grip: 20 px to the details
        self.list_col.setSpacing(10)
        left = QWidget()
        left.setLayout(self.list_col)

        detail = QVBoxLayout()
        detail.setContentsMargins(0, 0, 0, 0)
        detail.setSpacing(16)
        self.info = Card(icon="server")
        head = QHBoxLayout()
        self.name = label("", "h2")
        head.addWidget(self.name)
        self.state = Badge("", "neutral", dot=True)
        head.addWidget(self.state)
        head.addStretch(1)
        self.connect_btn = Button("Connect", "plug", size="sm", on_click=self._toggle_connection)
        self.edit_btn = Button("Edit", "edit", "ghost", size="sm", on_click=self._edit)
        self.remove_btn = Button("Remove", "trash", "ghost", size="sm", on_click=self._remove)
        for b in (self.connect_btn, self.edit_btn, self.remove_btn):
            head.addWidget(b)
        self.info.addLayout(head)
        self.address = label("", "mono")
        self.info.addWidget(self.address)
        self.error = label("", "error", wrap=True)
        self.info.addWidget(self.error)
        self.env = KeyValueGrid()
        self.info.addWidget(self.env)
        env_actions = QHBoxLayout()
        self.install_btn = Button("Install / update TorchCor", "download", "subtle", size="sm",
                                  on_click=lambda: self.selected and self.install_requested.emit(self.selected))
        self.python_btn = Button("Change Python…", "terminal", "ghost", size="sm",
                                 on_click=lambda: self.selected and self.python_requested.emit(self.selected))
        env_actions.addWidget(self.install_btn)
        env_actions.addWidget(self.python_btn)
        env_actions.addStretch(1)
        self.info.addLayout(env_actions)
        detail.addWidget(self.info)
        detail.addWidget(SectionHeader("Devices", "Live utilisation, memory, temperature and power."))
        from torchcor_studio.ui.widgets.gpu import GpuGrid

        self.devices = GpuGrid(selectable=False)
        detail.addWidget(self.devices)
        detail.addStretch(1)
        right = QWidget()
        right.setLayout(detail)
        # drag the list's edge to see long names and addresses (the width is remembered)
        split = PanelSplitter(left, right, width=ctx.settings.machine_list_width, minimum=260, maximum=560)
        split.moved.connect(lambda w: setattr(ctx.settings, "machine_list_width", w))
        row.addWidget(split, 1)
        lay.addWidget(scroll(body, max_width=1240))

        ctx.machines_changed.connect(self._rebuild)
        ctx.machine_state_changed.connect(self._state_changed)
        ctx.environment_changed.connect(self._state_changed)
        ctx.telemetry.connect(self._telemetry)
        self._cards: dict[str, SelectCard] = {}
        self._rebuild()

    # ------------------------------------------------------------------ lifecycle
    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        if self.selected is None and self.ctx.profiles():
            self.select(self.ctx.profiles()[0].id)
        elif self.selected:
            self._watch(self.selected)

    def hideEvent(self, e) -> None:  # noqa: N802
        super().hideEvent(e)
        self._watch(None)

    def _watch(self, machine_id: str | None) -> None:
        if self._watching == machine_id:
            return
        if self._watching:
            self.ctx.unwatch_telemetry(self._watching)
        self._watching = machine_id
        if machine_id:
            self.ctx.watch_telemetry(machine_id)

    # ------------------------------------------------------------------ list
    def _rebuild(self) -> None:
        while self.list_col.count():
            item = self.list_col.takeAt(0)
            if item.widget():
                discard(item.widget())
        self._cards.clear()
        for prof in self.ctx.profiles():
            card = SelectCard("laptop" if prof.is_local else "server", prof.name, machine_caption(prof))
            card.clicked.connect(lambda mid=prof.id: self.select(mid))
            self._cards[prof.id] = card
            self.list_col.addWidget(card)
            self._state_changed(prof.id)
        self.list_col.addStretch(1)
        if self.selected not in self._cards:
            self.selected = None
            if self._cards and self.isVisible():
                self.select(next(iter(self._cards)))
        else:
            self._cards[self.selected].set_checked(True)

    def select(self, machine_id: str) -> None:
        self.selected = machine_id
        for mid, card in self._cards.items():
            card.set_checked(mid == machine_id)
        self.devices.update_samples(self.ctx.session(machine_id).samples)
        self._watch(machine_id if self.isVisible() else None)
        self._render()

    # ------------------------------------------------------------------ detail
    def _render(self) -> None:
        if self.selected is None:
            return
        sess = self.ctx.session(self.selected)
        prof: MachineProfile = sess.profile
        self.name.setText(prof.name)
        self.address.setText("Local machine" if prof.is_local else f"ssh {prof.address}"
                             + (f"  via {prof.jump_host}" if prof.jump_host else ""))
        text, tone = {
            MachineState.CONNECTED: ("Connected", "success"),
            MachineState.CONNECTING: ("Connecting…", "accent"),
            MachineState.ERROR: ("Error", "danger"),
            MachineState.DISCONNECTED: ("Disconnected", "neutral"),
        }[sess.state]
        self.state.set(text, tone)
        self.connect_btn.setText("Disconnect" if sess.state == MachineState.CONNECTED else "Connect")
        self.connect_btn.set_icon_name("unplug" if sess.state == MachineState.CONNECTED else "plug")
        self.connect_btn.setEnabled(sess.state != MachineState.CONNECTING)
        self.edit_btn.setVisible(not prof.is_local)
        self.remove_btn.setVisible(not prof.is_local)
        self.error.setText(sess.error if sess.state == MachineState.ERROR else "")
        self.error.setVisible(sess.state == MachineState.ERROR)
        env = sess.env
        connected = sess.state == MachineState.CONNECTED
        self.install_btn.setEnabled(connected)
        self.python_btn.setEnabled(connected)
        if env is None:
            self.env.set_rows([("Environment", "Connect to inspect.")])
            return
        rows = [
            ("Host", f"{env.hostname} · {env.platform}"),
            ("Hardware", f"{env.cpu_count} CPU cores · {env.memory_total_gb:.0f} GB RAM"),
            ("Python", f"{env.python_version} — {env.python}"),
            ("PyTorch", torch_summary(env)),
            ("TorchCor", torchcor_summary(env) + (f"  ({env.torchcor_path})" if env.torchcor_path else "")),
            ("Default output", prof.output_dir or "—"),
        ]
        if env.error:
            rows.insert(0, ("Problem", env.error))
        self.env.set_rows(rows)

    def _state_changed(self, machine_id: str) -> None:
        card = self._cards.get(machine_id)
        if card is not None:
            sess = self.ctx.session(machine_id)
            card.footer.setText(state_html(sess.state))
            card.footer.setVisible(True)
        if machine_id == self.selected:
            self._render()

    def _telemetry(self, machine_id: str, samples) -> None:  # noqa: ANN001
        if machine_id == self.selected:
            self.devices.update_samples(samples)

    # ------------------------------------------------------------------ actions
    def add_machine(self) -> None:
        from torchcor_studio.ui.dialogs.machine_dialog import MachineDialog

        dlg = MachineDialog(None, self)
        if dlg.exec():
            profile = dlg.result_profile()
            self.ctx.save_machine(profile)
            self.select(profile.id)
            self.ctx.connect_machine(profile.id)

    def _edit(self) -> None:
        if self.selected is None:
            return
        from torchcor_studio.ui.dialogs.machine_dialog import MachineDialog

        dlg = MachineDialog(self.ctx.profile(self.selected), self)
        if dlg.exec():
            profile = dlg.result_profile()
            self.ctx.save_machine(profile)
            self.select(profile.id)
            self.ctx.connect_machine(profile.id)

    def _remove(self) -> None:
        prof = self.ctx.profile(self.selected) if self.selected else None
        if prof is None or prof.is_local:
            return
        answer = QMessageBox.question(self, "Remove machine?",
                                      f"Remove “{prof.name}” from TorchCor Lighting? Runs on it are not deleted.")
        if answer == QMessageBox.StandardButton.Yes:
            self.ctx.remove_machine(prof.id)

    def _toggle_connection(self) -> None:
        if self.selected is None:
            return
        sess = self.ctx.session(self.selected)
        if sess.state == MachineState.CONNECTED:
            self.ctx.disconnect(self.selected)
        else:
            self.ctx.connect_machine(self.selected)
