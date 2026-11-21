"""Add or edit a machine: SSH connection, credentials, Python and default folders.

``MachineDialog(profile_or_None, parent)``; after ``exec()`` returns ``Accepted``,
:meth:`MachineDialog.result_profile` is the profile to save. Passwords and key
passphrases go to the OS keyring (never into the profile). *Test connection* connects
with the values on screen — before anything is saved — and summarises the host.
"""

from __future__ import annotations

import getpass
from collections import Counter
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.core import secrets
from torchcor_studio.core.environment import MachineReport, check_machine
from torchcor_studio.core.machines import sshconfig
from torchcor_studio.core.machines.base import Machine
from torchcor_studio.core.machines.local import LocalMachine
from torchcor_studio.core.machines.ssh import (
    HostKeyChangedError,
    SSHMachine,
    forget_host_key,
    known_hosts_path,
)
from torchcor_studio.core.models import AuthMethod, MachineKind, MachineProfile
from torchcor_studio.ui import icons
from torchcor_studio.ui.dialogs.remote_browser import RemoteBrowserDialog
from torchcor_studio.ui.format import torch_summary, torchcor_summary
from torchcor_studio.ui.tasks import run_async
from torchcor_studio.ui.theme import repolish, theme
from torchcor_studio.ui.widgets import (
    Button,
    Card,
    IconButton,
    IconLabel,
    KeyValueGrid,
    SegmentedControl,
    Spinner,
    divider,
    label,
)
from torchcor_studio.ui.widgets.gpu import IconTile

_AUTH_OPTIONS = (
    (AuthMethod.AGENT, "SSH agent", "key"),
    (AuthMethod.KEY, "Key file", "file"),
    (AuthMethod.PASSWORD, "Password", "lock"),
)
_LABEL_WIDTH = 116


def _test_connection(profile: MachineProfile, password: str | None, passphrase: str | None) -> MachineReport:
    """Worker: connect with unsaved settings, summarise the machine, disconnect."""
    machine: Machine
    if profile.is_local:
        machine = LocalMachine(profile)
    else:
        machine = SSHMachine(profile, password=password, passphrase=passphrase)
    try:
        return check_machine(machine, python=profile.python or None)
    finally:
        machine.close()


def gpu_summary(gpus: list[str]) -> str:
    """``["GPU 0: NVIDIA A100", "GPU 1: NVIDIA A100"]`` → ``"2 × NVIDIA A100"``."""
    names = Counter(g.split(":", 1)[1].strip() if ":" in g else g for g in gpus)
    return ", ".join(f"{n} × {name}" if n > 1 else name for name, n in names.items())


class _PasswordField(QLineEdit):
    """Password entry with a show / hide toggle."""

    def __init__(self, placeholder: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEchoMode(QLineEdit.EchoMode.Password)
        self.setPlaceholderText(placeholder)
        self._toggle = self.addAction(icons.icon("eye", theme().palette.text3),
                                      QLineEdit.ActionPosition.TrailingPosition)
        self._toggle.setToolTip("Show")
        self._toggle.triggered.connect(self._flip)
        theme().changed.connect(self._retint_toggle)

    def _retint_toggle(self, palette) -> None:  # noqa: ANN001
        self._toggle.setIcon(icons.icon("eye", palette.text3))

    def _flip(self) -> None:
        hidden = self.echoMode() == QLineEdit.EchoMode.Password
        self.setEchoMode(QLineEdit.EchoMode.Normal if hidden else QLineEdit.EchoMode.Password)
        self._toggle.setToolTip("Hide" if hidden else "Show")


class MachineDialog(QDialog):
    """Create or edit a :class:`MachineProfile` (an SSH host, or this computer)."""

    def __init__(self, profile: MachineProfile | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._original = profile
        self._editing = profile is not None
        self._profile = replace(profile) if profile else MachineProfile(name="", kind=MachineKind.SSH)
        self._local = self._profile.is_local
        self._result: MachineProfile | None = None
        self._browse_machine: Machine | None = None
        self._browse_key: tuple | None = None
        self._imports: list[MachineProfile] = []
        self._shown = False
        self.setWindowTitle(self._heading())
        self.setModal(True)
        self.setMinimumWidth(620)
        self._build()
        self._load(self._profile)
        if not self._local:
            run_async(sshconfig.import_profiles, on_done=self._fill_imports,
                      on_error=lambda _exc: self._fill_imports([]), owner=self)

    # ------------------------------------------------------------------ public
    def result_profile(self) -> MachineProfile:
        """The edited profile (after *Save*; before that, what the form currently says)."""
        return self._result or self._collect()

    # ------------------------------------------------------------------ layout
    def _heading(self) -> str:
        if self._local:
            return "This computer"
        return "Edit machine" if self._editing else "Add a machine"

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(18)

        head = QHBoxLayout()
        head.setSpacing(14)
        head.addWidget(IconTile("laptop" if self._local else "server", "accent", size=40), 0,
                       Qt.AlignmentFlag.AlignTop)
        titles = QVBoxLayout()
        titles.setSpacing(3)
        titles.addWidget(label(self._heading(), "title"))
        subtitle = ("The Python environment and default folders used for runs on this computer."
                    if self._local else
                    "Run simulations on a GPU server over SSH. Passwords stay in your system keychain.")
        titles.addWidget(label(subtitle, "muted", wrap=True))
        head.addLayout(titles, 1)
        if not self._local:
            self._import = Button("Import from SSH config", icon="download", variant="ghost", size="sm")
            self._import_menu = QMenu(self._import)
            self._import_menu.addAction("Reading ~/.ssh/config…").setEnabled(False)
            self._import.setMenu(self._import_menu)
            head.addWidget(self._import, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(head)

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(16)
        self._grid.setVerticalSpacing(10)
        self._grid.setColumnMinimumWidth(0, _LABEL_WIDTH)
        self._grid.setColumnStretch(1, 1)
        self._row = 0

        self.name = QLineEdit()
        self.name.setPlaceholderText("This computer" if self._local else "e.g. Lab GPU server")
        if not self._local:
            self._section("Connection")
        self._add("Name", self.name)
        if not self._local:
            self._build_connection()
        self._section("Environment")
        self._build_environment()
        root.addLayout(self._grid)

        self._result_card = Card(variant="inset", padding=14, spacing=10)
        self._result_card.hide()
        root.addWidget(self._result_card)

        root.addWidget(divider())
        foot = QHBoxLayout()
        foot.setSpacing(10)
        self._test = Button("Check environment" if self._local else "Test connection",
                            icon="activity" if self._local else "plug", on_click=self._run_test)
        foot.addWidget(self._test)
        self._spinner = Spinner(16)
        self._spinner.hide()
        foot.addWidget(self._spinner)
        self._status = label("", "muted")
        foot.addWidget(self._status, 1)
        cancel = Button("Cancel", variant="ghost", on_click=self.reject)
        foot.addWidget(cancel)
        self._save = Button("Save" if self._editing else "Add machine", variant="primary", on_click=self.accept)
        self._save.setDefault(True)
        foot.addWidget(self._save)
        root.addLayout(foot)

    def _section(self, title: str) -> None:
        if self._row:
            self._grid.addWidget(divider(), self._row, 0, 1, 2)
            self._row += 1
        eyebrow = label(title.upper(), "eyebrow")
        self._grid.addWidget(eyebrow, self._row, 0, 1, 2)
        self._row += 1

    def _add(self, text: str, field, hint: QWidget | None = None) -> QWidget:
        """Add a labelled row; returns the label (to hide rows together)."""
        lbl = label(text, "muted")
        self._grid.addWidget(lbl, self._row, 0, Qt.AlignmentFlag.AlignVCenter)
        if isinstance(field, QWidget):
            self._grid.addWidget(field, self._row, 1)
        else:
            self._grid.addLayout(field, self._row, 1)
        self._row += 1
        if hint is not None:
            self._grid.addWidget(hint, self._row, 1)
            self._row += 1
        return lbl

    def _build_connection(self) -> None:
        self.host = QLineEdit()
        self.host.setPlaceholderText("gpu01.example.org")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(22)
        self.port.setFixedWidth(84)
        self.port.setAlignment(Qt.AlignmentFlag.AlignRight)
        host_row = QHBoxLayout()
        host_row.setSpacing(10)
        host_row.addWidget(self.host, 1)
        host_row.addWidget(label("Port", "muted"))
        host_row.addWidget(self.port)
        self._add("Host", host_row)

        self.username = QLineEdit()
        self.username.setPlaceholderText(getpass.getuser())
        self._add("Username", self.username)

        self.auth = SegmentedControl(_AUTH_OPTIONS, AuthMethod.AGENT)
        self.auth.changed.connect(lambda *_: self._sync_auth())
        auth_row = QHBoxLayout()
        auth_row.addWidget(self.auth)
        auth_row.addStretch(1)
        self._add("Sign in with", auth_row)

        self.key_path = QLineEdit()
        self.key_path.setPlaceholderText("~/.ssh/id_ed25519")
        key_row = QHBoxLayout()
        key_row.setSpacing(8)
        key_row.addWidget(self.key_path, 1)
        self._key_browse = Button("Browse…", on_click=self._browse_key)
        key_row.addWidget(self._key_browse)
        self._key_label = self._add("Key file", key_row)

        self.passphrase = _PasswordField("Only if the key is encrypted")
        self._passphrase_label = self._add("Passphrase", self.passphrase)
        self.password = _PasswordField()
        self._password_label = self._add("Password", self.password)
        self._auth_hint = label("", "caption", wrap=True)
        self._grid.addWidget(self._auth_hint, self._row, 1)
        self._row += 1

        self.jump_host = QLineEdit()
        self.jump_host.setPlaceholderText("user@bastion.example.org:22 — optional")
        self.jump_host.setToolTip("Reach the server through this host first (like ssh -J).")
        self._add("Jump host", self.jump_host)

    def _build_environment(self) -> None:
        self.python = QLineEdit()
        self.python.setPlaceholderText("Auto-detect")
        self.python.setToolTip("Python interpreter with TorchCor installed. Leave empty to detect it.")
        self._add("Python", self._with_browse(self.python, "file", "Choose the Python interpreter"))
        self.input_dir = QLineEdit()
        self.input_dir.setPlaceholderText("Data folder with ep/, ecg/, … inside (default ~/torchcor_data)")
        self._add("Input folder", self._with_browse(self.input_dir, "dir", "Choose the default input folder"))
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Where runs are written, e.g. ~/torchcor-runs")
        self._add("Output folder", self._with_browse(self.output_dir, "dir", "Choose the default output folder"))

    def _with_browse(self, field: QLineEdit, mode: str, title: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(field, 1)
        button = IconButton("folder-open", "Browse…", size=theme().metrics.control_h,
                            on_click=lambda: self._browse(field, mode, title))
        row.addWidget(button)
        return row

    def _fit(self) -> None:
        """Size the window to its content (``adjustSize`` would cap it at 2/3 of the screen)."""
        layout = self.layout()
        width = max(self.width(), self.minimumWidth())
        height = layout.heightForWidth(width) if layout.hasHeightForWidth() else self.sizeHint().height()
        screen = self.screen().availableGeometry() if self.screen() is not None else None
        if screen is not None:
            height = min(height, int(screen.height() * 0.94))
        self.resize(width, max(height, layout.minimumSize().height()))

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._shown:
            self._shown = True
            self._fit()

    # ------------------------------------------------------------------ data
    def _load(self, p: MachineProfile) -> None:
        self.name.setText(p.name)
        self.python.setText(p.python)
        self.input_dir.setText(p.input_dir)
        self.output_dir.setText(p.output_dir)
        if self._local:
            return
        self.host.setText(p.host)
        self.port.setValue(p.port or 22)
        self.username.setText(p.username)
        self.auth.set_value(p.auth, animate=False)
        self.key_path.setText(p.key_path)
        self.jump_host.setText(p.jump_host)
        if self._editing and p.auth == AuthMethod.PASSWORD:
            self.password.setPlaceholderText("Saved in your keychain — type to replace")
        else:
            self.password.setPlaceholderText("")
        if self._editing and p.auth == AuthMethod.KEY:
            self.passphrase.setPlaceholderText("Unchanged unless you type a new one")
        self._sync_auth()

    def _collect(self) -> MachineProfile:
        p = self._profile
        name = self.name.text().strip()
        profile = replace(
            p,
            name=name,
            python=self.python.text().strip(),
            input_dir=self.input_dir.text().strip(),
            output_dir=self.output_dir.text().strip(),
        )
        if self._local:
            return replace(profile, name=name or "This computer")
        host = self.host.text().strip()
        return replace(
            profile,
            name=name or host,
            kind=MachineKind.SSH,
            host=host,
            port=int(self.port.value()),
            username=self.username.text().strip(),
            auth=AuthMethod(self.auth.value()),
            key_path=self.key_path.text().strip(),
            jump_host=self.jump_host.text().strip(),
        )

    def _sync_auth(self) -> None:
        method = self.auth.value()
        is_key, is_password = method == AuthMethod.KEY, method == AuthMethod.PASSWORD
        for widget in (self._key_label, self.key_path, self._key_browse, self._passphrase_label, self.passphrase):
            widget.setVisible(is_key)
        for widget in (self._password_label, self.password):
            widget.setVisible(is_password)
        self._auth_hint.setText({
            AuthMethod.AGENT: "Uses the keys loaded in ssh-agent and your ~/.ssh/id_* files.",
            AuthMethod.KEY: "The passphrase is kept in your system keychain, never on disk.",
            AuthMethod.PASSWORD: "The password is kept in your system keychain, never on disk.",
        }[method])
        self._fit()

    def _fill_imports(self, profiles: list[MachineProfile]) -> None:
        self._imports = profiles
        self._import_menu.clear()
        if not profiles:
            self._import_menu.addAction("No hosts found in ~/.ssh/config").setEnabled(False)
            return
        for profile in profiles:
            text = f"{profile.name}  —  {profile.address}"
            action = self._import_menu.addAction(icons.icon("server", theme().palette.text2), text)
            action.triggered.connect(lambda _=False, pr=profile: self._apply_import(pr))

    def _apply_import(self, imported: MachineProfile) -> None:
        self.name.setText(imported.name)
        self.host.setText(imported.host)
        self.port.setValue(imported.port or 22)
        self.username.setText(imported.username)
        self.auth.set_value(imported.auth, animate=False)
        self.key_path.setText(imported.key_path)
        self.jump_host.setText(imported.jump_host)
        self._sync_auth()
        self._status.setText(f"Imported {imported.name} from ~/.ssh/config")

    # ------------------------------------------------------------------ validation
    def _mark(self, field: QWidget, bad: bool) -> None:
        field.setProperty("invalid", bad)
        repolish(field)

    def _validate(self) -> str | None:
        """Flag invalid fields; returns the first problem, or ``None``."""
        if self._local:
            return None
        problems = []
        host_bad = not self.host.text().strip() or any(c.isspace() for c in self.host.text().strip())
        self._mark(self.host, host_bad)
        if host_bad:
            problems.append("Enter the server's host name or IP address.")
        method = self.auth.value()
        key_bad = method == AuthMethod.KEY and not self.key_path.text().strip()
        self._mark(self.key_path, key_bad)
        if key_bad:
            problems.append("Choose the private key file.")
        password_bad = (method == AuthMethod.PASSWORD and not self.password.text()
                        and not (self._editing and self._profile.auth == AuthMethod.PASSWORD))
        self._mark(self.password, password_bad)
        if password_bad:
            problems.append("Enter the password.")
        return problems[0] if problems else None

    def accept(self) -> None:
        problem = self._validate()
        if problem:
            self._show_status(problem, error=True)
            return
        profile = self._collect()
        self._store_secrets(profile)
        self._result = profile
        super().accept()

    def _store_secrets(self, profile: MachineProfile) -> None:
        """Keep typed secrets for this session at once; persist them off the UI thread."""
        if profile.is_local:
            return
        typed = {secrets.PASSWORD: self.password.text() if profile.auth == AuthMethod.PASSWORD else "",
                 secrets.PASSPHRASE: self.passphrase.text() if profile.auth == AuthMethod.KEY else ""}
        for kind, value in typed.items():
            if value:
                secrets.remember(profile.id, kind, value)
                run_async(secrets.set_secret, profile.id, kind, value)
        if self._original is not None:
            if self._original.auth == AuthMethod.PASSWORD and profile.auth != AuthMethod.PASSWORD:
                run_async(secrets.delete_secret, profile.id, secrets.PASSWORD)
            if self._original.auth == AuthMethod.KEY and profile.auth != AuthMethod.KEY:
                run_async(secrets.delete_secret, profile.id, secrets.PASSPHRASE)

    def done(self, result: int) -> None:
        machine, self._browse_machine = self._browse_machine, None
        if machine is not None:
            run_async(machine.close)
        super().done(result)

    # ------------------------------------------------------------------ browsing
    def _browse_key(self) -> None:
        start = str(Path(self.key_path.text().strip() or "~/.ssh").expanduser())
        path, _ = QFileDialog.getOpenFileName(self, "Choose a private key", start)
        if path:
            self.key_path.setText(path)

    def _machine_for_browsing(self) -> Machine:
        """A connection for the folder pickers, reused while the connection fields stay the same."""
        profile = self._collect()
        if profile.is_local:
            return LocalMachine(profile)
        key = (profile.address, profile.auth, profile.key_path, profile.jump_host,
               self.password.text(), self.passphrase.text())
        cached = self._browse_machine
        if cached is not None and self._browse_key == key:
            return cached
        if cached is not None:
            run_async(cached.close)
        self._browse_key = key
        self._browse_machine = SSHMachine(profile, password=self.password.text() or None,
                                          passphrase=self.passphrase.text() or None)
        return self._browse_machine

    def _browse(self, field: QLineEdit, mode: str, title: str) -> None:
        if not self._local and self._validate() is not None:
            self._show_status("Fill in the connection first, then browse the server.", error=True)
            return
        current = field.text().strip()
        if self._local:
            if mode == "dir":
                path = QFileDialog.getExistingDirectory(self, title, str(Path(current or "~").expanduser()))
            else:
                path, _ = QFileDialog.getOpenFileName(self, title, str(Path(current or "~").expanduser()))
        else:
            start = current if mode == "dir" else (current.rsplit("/", 1)[0] if "/" in current else "~")
            path = RemoteBrowserDialog.get_path(self._machine_for_browsing(), start or "~", mode, title, self)
        if path:
            field.setText(path)

    # ------------------------------------------------------------------ testing
    def _show_status(self, text: str, error: bool = False) -> None:
        self._status.setProperty("role", "error" if error else "muted")
        repolish(self._status)
        self._status.setText(text)

    def _run_test(self) -> None:
        problem = self._validate()
        if problem:
            self._show_status(problem, error=True)
            return
        profile = self._collect()
        self._test.setEnabled(False)
        self._spinner.show()
        self._result_card.hide()
        where = "this computer" if profile.is_local else profile.host
        self._show_status(f"Connecting to {where}…" if not profile.is_local else "Checking the Python environment…")
        password = self.password.text() if profile.auth == AuthMethod.PASSWORD and self.password.text() else None
        passphrase = self.passphrase.text() if profile.auth == AuthMethod.KEY and self.passphrase.text() else None
        if self._local:
            password = passphrase = None
        run_async(_test_connection, profile, password, passphrase,
                  on_done=self._show_report, on_error=self._show_failure, owner=self)

    def _reset_card(self) -> None:
        body = self._result_card.body
        while body.count():
            item = body.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                _clear_layout(item.layout())

    def _card_header(self, icon: str, tone: str, title: str, subtitle: str = "",
                     detail_role: str = "caption") -> None:
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(IconLabel(icon, 18, tone=tone), 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(2)
        heading = label(title)
        heading.setStyleSheet("font-weight: 600;")
        col.addWidget(heading)
        if subtitle:
            col.addWidget(label(subtitle, detail_role, wrap=True, selectable=True))
        row.addLayout(col, 1)
        self._result_card.addLayout(row)

    def _show_report(self, report: MachineReport) -> None:
        self._test.setEnabled(True)
        self._spinner.hide()
        if not report.ok:
            self._show_failure(RuntimeError(report.error or "unknown error"))
            return
        self._reset_card()
        env = report.environment
        ready = env is not None and env.ready
        latency = f" · {report.latency_ms:.0f} ms" if report.latency_ms is not None else ""
        if self._local:
            title = "TorchCor is ready on this computer" if ready else "Python environment checked"
        else:
            title = f"Connected to {report.hostname or self.host.text().strip()}"
        self._card_header("check-circle" if ready else "alert", "success" if ready else "warning",
                          title, f"{report.os}{latency}" if report.os else latency.lstrip(" ·"))
        rows: list[tuple[str, str]] = []
        rows.append(("GPUs", gpu_summary(report.gpus) if report.gpus else
                     f"None — {report.gpu_note or 'no NVIDIA GPU'} (runs use the CPU)"))
        if env is not None:
            if env.error:
                rows.append(("Python", f"{env.python or 'not found'} — {env.error.splitlines()[0]}"))
            else:
                rows.append(("Python", f"{env.python_version} · {env.python}"))
                rows.append(("PyTorch", torch_summary(env)))
                rows.append(("TorchCor", torchcor_summary(env) if env.torchcor_version
                             else "not installed — install it after saving"))
            if not self.python.text().strip() and env.python:
                self.python.setPlaceholderText(f"Auto-detect · found {env.python}")
        if report.host_key:
            rows.append(("Host key", report.host_key + (" · trusted now (first connection)"
                                                        if report.new_host_key else "")))
        self._result_card.addWidget(KeyValueGrid(rows))
        self._result_card.show()
        self._show_status("")
        self._fit()

    def _show_failure(self, exc: BaseException) -> None:
        self._test.setEnabled(True)
        self._spinner.hide()
        self._reset_card()
        message = str(exc) or type(exc).__name__
        if isinstance(exc, HostKeyChangedError):
            title, message = "The server's host key has changed", message.partition("\n")[2]
        else:
            title = "Check failed" if self._local else "Could not connect"
        self._card_header("x-circle", "danger", title, message, detail_role="muted")
        if isinstance(exc, HostKeyChangedError) and exc.known_hosts == str(known_hosts_path()):
            profile = self._collect()
            retry = Button("Forget the old key and retry", icon="refresh", variant="danger", size="sm",
                           on_click=lambda: self._forget_and_retry(profile.host, profile.port))
            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(retry)
            self._result_card.addLayout(row)
        self._result_card.show()
        self._show_status("")
        self._fit()

    def _forget_and_retry(self, host: str, port: int) -> None:
        run_async(forget_host_key, host, port, on_done=lambda _: self._run_test(), owner=self)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            _clear_layout(item.layout())
