"""SSH hosts as a :class:`Machine` (paramiko): exec channels for commands, SFTP for files.

* **Authentication** follows the profile: ssh-agent / default keys, an explicit key file
  (passphrase from the OS keyring) or a password (OS keyring). Settings missing from
  the profile are filled in from ``~/.ssh/config``, like OpenSSH does.
* **Jump hosts**: ``profile.jump_host`` ("user@host:port", or a comma-separated chain)
  is reached first and the target is tunnelled through a ``direct-tcpip`` channel.
* **Host keys** are checked against ``~/.ssh/known_hosts`` and the studio's own
  ``known_hosts``. Unknown hosts are trusted on first use and remembered in the
  studio's file; a *changed* key raises :class:`HostKeyChangedError`.
* **Threads**: paramiko transports multiplex channels safely; the shared SFTP session
  is guarded by a lock and bulk transfers get a channel of their own. A dropped
  connection is transparently re-established once.
* **Commands** run through a non-login ``sh -c`` (login profiles can print junk), so
  every host behaves the same whatever the user's login shell is.
"""

from __future__ import annotations

import base64
import contextlib
import getpass
import hashlib
import logging
import os
import posixpath
import re
import select
import shlex
import socket
import stat
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import paramiko
from paramiko.pkey import UnknownKeyType
from paramiko.ssh_exception import NoValidConnectionsError

from torchcor_studio.core import secrets
from torchcor_studio.core.machines import sshconfig
from torchcor_studio.core.machines.base import Machine, MachineError
from torchcor_studio.core.machines.deploy import REMOTE_RUNTIME_BASE, deploy_remote
from torchcor_studio.core.machines.local import machine_os_error
from torchcor_studio.core.models import AuthMethod, CommandResult, FileEntry, MachineProfile
from torchcor_studio.core.paths import config_dir

log = logging.getLogger(__name__)
T = TypeVar("T")

CONNECT_TIMEOUT = 10.0
KEEPALIVE_S = 30
#: Concurrent short commands / bulk transfers per connection (sshd's MaxSessions is 10).
EXEC_SLOTS = 4
TRANSFER_SLOTS = 2
PID_MARKER = "__TCS_PID__"
STUDIO_VENV = "~/.torchcor-studio/venv"

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_known_hosts_lock = threading.Lock()


# ============================================================================ errors
class HostKeyChangedError(MachineError):
    """The server presented a different host key than the one remembered for it."""

    def __init__(self, host: str, expected: str, received: str, known_hosts: str) -> None:
        self.host, self.expected, self.received, self.known_hosts = host, expected, received, known_hosts
        super().__init__(
            f"WARNING: the host key of {host} has changed.\n"
            f"Expected {expected}, but the server presented {received}.\n"
            "Someone could be intercepting the connection (man-in-the-middle), or the server "
            "was reinstalled. Ask its administrator; if the change is expected, remove the old "
            f"key for {host} from {known_hosts} and connect again."
        )


# ============================================================================ helpers
def known_hosts_path() -> Path:
    """The studio's own ``known_hosts`` (keys trusted on first use)."""
    return config_dir() / "known_hosts"


def fingerprint(key: paramiko.PKey) -> str:
    """OpenSSH-style ``SHA256:…`` fingerprint of a public key."""
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


def host_key_name(host: str, port: int) -> str:
    """The name a host is filed under in ``known_hosts``."""
    return host if port == 22 else f"[{host}]:{port}"


def load_private_key(path: str, passphrase: str | None = None) -> paramiko.PKey:
    """Open a private key file, turning paramiko's errors into clear messages."""
    full = os.path.expanduser(path)
    if not os.path.isfile(full):
        raise MachineError(f"The key file {path} does not exist on this computer.")
    secret = passphrase.encode() if passphrase else None
    try:
        try:
            return paramiko.PKey.from_path(full, secret)
        except TypeError:
            if secret is None:
                raise
            return paramiko.PKey.from_path(full, None)  # a passphrase for an unencrypted key
    except TypeError as exc:
        raise MachineError(
            f"The key file {path} is protected by a passphrase. Edit the machine and enter it.") from exc
    except ValueError as exc:
        raise MachineError(
            f"Could not unlock the key file {path}: the passphrase is wrong or the key format "
            "is not supported.") from exc
    except (OSError, paramiko.SSHException, UnknownKeyType) as exc:
        raise MachineError(f"Could not read the key file {path}: {exc}") from exc


def _remember_host_key(path: Path, name: str, key: paramiko.PKey) -> None:
    with _known_hosts_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{name} {key.get_name()} {key.get_base64()}\n")


def forget_host_key(host: str, port: int = 22) -> bool:
    """Remove a host from the studio's ``known_hosts`` (after a legitimate key change).

    Returns ``True`` if an entry was removed. Keys in ``~/.ssh/known_hosts`` are never
    touched; edit that file (``ssh-keygen -R host``) for those.
    """
    path = known_hosts_path()
    name = host_key_name(host, port)
    with _known_hosts_lock:
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError:
            return False
        kept = [ln for ln in lines if name not in ln.split(" ", 1)[0].split(",")]
        if len(kept) == len(lines):
            return False
        path.write_text("".join(kept), encoding="utf-8")
        return True


def _quote_arg(arg: str) -> str:
    """Shell-quote one word, keeping a leading ``~`` / ``~/`` expandable."""
    arg = str(arg)
    if arg == "~":
        return '"$HOME"'
    if arg.startswith("~/"):
        return '"$HOME"/' + shlex.quote(arg[2:]) if arg[2:] else '"$HOME"/'
    return shlex.quote(arg)


def build_command(
    argv: Sequence[str] | str,
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    echo_pid: bool = False,
) -> str:
    """The exec-request string for ``argv``: ``sh -c '<cd> && <exports> && exec <argv>'``.

    A list is quoted word by word (and ``exec``-ed, so the remote PID is the command's);
    a string is passed to ``sh`` verbatim. ``echo_pid`` prints ``__TCS_PID__<pid>`` first
    so the caller can kill the remote process later.
    """
    steps: list[str] = []
    if cwd:
        steps.append(f"cd {_quote_arg(cwd)}")
    for name, value in (env or {}).items():
        if not _ENV_NAME.match(name):
            raise ValueError(f"invalid environment variable name {name!r}")
        steps.append(f"export {name}={shlex.quote(str(value))}")
    if isinstance(argv, str):
        steps.append(argv)
    else:
        if not argv:
            raise ValueError("empty command")
        steps.append("exec " + " ".join(_quote_arg(a) for a in argv))
    script = " && ".join(steps)
    if echo_pid:
        script = f"echo {PID_MARKER}$$; {script}"
    return "sh -c " + shlex.quote(script)


def alive_script(pid: int) -> str:
    """POSIX test that succeeds while ``pid`` runs (zombies count as dead)."""
    pid = int(pid)
    return f'kill -0 {pid} 2>/dev/null && [ "$(ps -o stat= -p {pid} 2>/dev/null | cut -c1)" != Z ]'


def kill_tree_script(pid: int, grace_s: float = 5.0) -> str:
    """A one-line POSIX script that terminates ``pid`` and its descendants.

    SIGTERM goes to the whole tree first; whatever still runs after ``grace_s`` is killed.
    """
    pid = int(pid)
    ticks = max(1, int(grace_s * 10))
    return (
        't(){ echo "$1"; for c in $(pgrep -P "$1" 2>/dev/null); do t "$c"; done; }; '
        f"p=$(t {pid}); kill -TERM $p 2>/dev/null; i=0; "
        f"while [ $i -lt {ticks} ] && {alive_script(pid)}; do sleep 0.1; i=$((i+1)); done; "
        "kill -KILL $p 2>/dev/null; exit 0"
    )


#: Where :func:`~torchcor_studio.core.environment.find_pythons` looks on SSH hosts (shell
#: globs), besides conda's own list of environments and the PATH. Between equally ready
#: interpreters, the earlier pattern wins.
REMOTE_PYTHON_GLOBS = (
    f"{STUDIO_VENV}/bin/python",
    "~/miniconda3/envs/*/bin/python",
    "~/anaconda3/envs/*/bin/python",
    "~/miniforge3/envs/*/bin/python",
    "~/mambaforge/envs/*/bin/python",
    "~/micromamba/envs/*/bin/python",
    "~/.conda/envs/*/bin/python",
    "/opt/conda/envs/*/bin/python",
    "~/.venv/bin/python",
    "~/venv*/bin/python",
    "~/.virtualenvs/*/bin/python",
    "~/*/.venv/bin/python",  # project environments (uv, poetry, python -m venv)
    "~/*/venv/bin/python",
    "~/.pyenv/versions/*/bin/python",
    "~/miniconda3/bin/python",
    "~/anaconda3/bin/python",
    "~/miniforge3/bin/python",
    "~/mambaforge/bin/python",
    "/opt/conda/bin/python",
)


def remote_globs() -> str:
    """:data:`REMOTE_PYTHON_GLOBS` as unquoted shell words (``~`` → ``$HOME``)."""
    return " ".join(g.replace("~", "$HOME", 1) if g.startswith("~") else g
                    for g in REMOTE_PYTHON_GLOBS)


@dataclass
class _Endpoint:
    """One SSH hop: where to connect and how to authenticate."""

    host: str
    port: int
    user: str
    pkey: paramiko.PKey | None
    key_files: list[str]
    password: str | None
    allow_agent: bool
    look_for_keys: bool

    @property
    def label(self) -> str:
        return sshconfig.format_hop(self.user, self.host, self.port)


class _TrustOnFirstUse(paramiko.MissingHostKeyPolicy):
    """Accept an unknown host key and remember it in the studio's ``known_hosts``."""

    def __init__(self, machine: SSHMachine) -> None:
        self.machine = machine

    def missing_host_key(self, client: paramiko.SSHClient, hostname: str, key: paramiko.PKey) -> None:
        client.get_host_keys().add(hostname, key.get_name(), key)
        _remember_host_key(known_hosts_path(), hostname, key)
        self.machine.new_host_keys.append(f"{hostname} {fingerprint(key)}")
        log.info("trusted new host key for %s: %s", hostname, fingerprint(key))


# ============================================================================ machine
class SSHMachine(Machine):
    """A POSIX host reached over SSH."""

    def __init__(self, profile: MachineProfile, *, password: str | None = None,
                 passphrase: str | None = None) -> None:
        """``password`` / ``passphrase`` override the keyring (used to test unsaved settings)."""
        super().__init__(profile)
        self._password = password
        self._passphrase = passphrase
        self._client: paramiko.SSHClient | None = None
        self._hops: list[paramiko.SSHClient] = []
        self._sftp: paramiko.SFTPClient | None = None
        self._sftp_lock = threading.RLock()
        self._exec_slots = threading.BoundedSemaphore(EXEC_SLOTS)
        self._transfer_slots = threading.BoundedSemaphore(TRANSFER_SLOTS)
        self._home: str | None = None
        #: ``"host SHA256:…"`` for keys trusted on first use during this machine's life.
        self.new_host_keys: list[str] = []
        #: Fingerprint of the target's host key once connected.
        self.host_key_fingerprint: str = ""

    # ------------------------------------------------------------------ lifecycle
    @property
    def is_connected(self) -> bool:
        client = self._client
        transport = client.get_transport() if client is not None else None
        return transport is not None and transport.is_active()

    def connect(self) -> None:
        with self._lock:
            if self.is_connected:
                return
            self._drop()
            self._open()

    def close(self) -> None:
        with self._lock:
            self._drop()

    def _drop(self) -> None:
        # No _sftp_lock here: _sftp_op holds it while (re)connecting under self._lock,
        # so taking it in the other order could deadlock. Swapping the reference is atomic.
        sftp, self._sftp = self._sftp, None
        for closable in [sftp, self._client, *reversed(self._hops)]:
            if closable is not None:
                with contextlib.suppress(Exception):  # already dead
                    closable.close()
        self._client, self._hops = None, []

    # ------------------------------------------------------------------ connecting
    def endpoints(self) -> list[_Endpoint]:
        """Jump hops followed by the target, with settings completed from ``~/.ssh/config``."""
        p = self.profile
        if not p.host.strip():
            raise MachineError(f"No host name is set for {p.name}.")
        config = sshconfig.load()
        target_cfg = sshconfig.lookup(p.host, config=config)
        user = p.username or target_cfg.user or getpass.getuser()
        port = p.port if p.port and p.port != 22 else (target_cfg.port or 22)
        password: str | None = None
        pkey: paramiko.PKey | None = None
        keys: list[str] = []
        if p.auth == AuthMethod.PASSWORD:
            password = self._password if self._password is not None else secrets.get_password(p.id)
            if not password:
                raise MachineError(f"No password is saved for {p.name}. Edit the machine and enter it.")
            agent = look = False
        elif p.auth == AuthMethod.KEY:
            if not p.key_path:
                raise MachineError(f"{p.name} uses a key file, but none is set.")
            passphrase = self._passphrase if self._passphrase is not None else secrets.get_passphrase(p.id)
            pkey = load_private_key(p.key_path, passphrase)
            agent, look = True, False
        else:
            keys = [f for f in target_cfg.identity_files if os.path.isfile(f)]
            agent = look = True
        target = _Endpoint(target_cfg.hostname, port, user, pkey, keys, password, agent, look)

        jump_spec = p.jump_host or (sshconfig.resolve_jumps(target_cfg.proxy_jump, config)
                                    if target_cfg.proxy_jump else "")
        hops = []
        for spec in (s.strip() for s in jump_spec.split(",")):
            if not spec:
                continue
            j_user, j_host, j_port = sshconfig.parse_hop(spec)
            j_cfg = sshconfig.lookup(j_host, config=config)
            hops.append(_Endpoint(
                host=j_cfg.hostname,
                port=j_port if j_port != 22 else (j_cfg.port or 22),
                user=j_user or j_cfg.user or user,
                pkey=pkey,
                key_files=[*keys, *(f for f in j_cfg.identity_files if os.path.isfile(f))],
                password=password,  # bastions often share the target's credentials
                allow_agent=True, look_for_keys=True,
            ))
        return [*hops, target]

    def _new_client(self) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        try:
            client.load_system_host_keys()
        except (OSError, paramiko.SSHException) as exc:
            log.info("could not read ~/.ssh/known_hosts: %s", exc)
        ours = known_hosts_path()
        if ours.is_file():
            try:
                client.get_host_keys().load(str(ours))
            except (OSError, paramiko.SSHException) as exc:
                log.info("could not read %s: %s", ours, exc)
        client.set_missing_host_key_policy(_TrustOnFirstUse(self))
        return client

    def _open(self) -> None:
        chain = self.endpoints()
        clients: list[paramiko.SSHClient] = []
        sock = None
        try:
            for i, ep in enumerate(chain):
                client = self._new_client()
                clients.append(client)
                self._connect_one(client, ep, sock)
                if i < len(chain) - 1:
                    nxt = chain[i + 1]
                    try:
                        sock = client.get_transport().open_channel(
                            "direct-tcpip", (nxt.host, nxt.port), ("127.0.0.1", 0),
                            timeout=CONNECT_TIMEOUT)
                    except (paramiko.SSHException, OSError) as exc:
                        raise MachineError(f"{ep.label} could not reach {nxt.label}: {exc}") from exc
        except BaseException:
            for client in reversed(clients):
                client.close()
            raise
        transport = clients[-1].get_transport()
        transport.set_keepalive(KEEPALIVE_S)
        self.host_key_fingerprint = fingerprint(transport.get_remote_server_key())
        self._hops, self._client = clients[:-1], clients[-1]

    def _connect_one(self, client: paramiko.SSHClient, ep: _Endpoint, sock) -> None:
        try:
            client.connect(
                ep.host, ep.port, username=ep.user, password=ep.password, pkey=ep.pkey,
                key_filename=ep.key_files or None,
                allow_agent=ep.allow_agent, look_for_keys=ep.look_for_keys, sock=sock,
                timeout=CONNECT_TIMEOUT, banner_timeout=CONNECT_TIMEOUT,
                auth_timeout=CONNECT_TIMEOUT + 5, channel_timeout=CONNECT_TIMEOUT,
            )
        except paramiko.BadHostKeyException as exc:
            raise HostKeyChangedError(
                host_key_name(ep.host, ep.port), fingerprint(exc.expected_key),
                fingerprint(exc.key), self._known_hosts_holding(ep),
            ) from exc
        except paramiko.AuthenticationException as exc:
            how = {AuthMethod.PASSWORD: "the password", AuthMethod.KEY: "the key file"}.get(
                self.profile.auth, "ssh-agent and your default keys")
            raise MachineError(f"Authentication failed for {ep.label} using {how}: {exc}") from exc
        except TimeoutError as exc:
            raise MachineError(f"Timed out connecting to {ep.label}.") from exc
        except socket.gaierror as exc:
            raise MachineError(f"Unknown host {ep.host!r}: {exc.strerror or exc}") from exc
        except NoValidConnectionsError as exc:
            raise MachineError(
                f"Could not connect to {ep.label}: the connection was refused. "
                "Check the host name and port, and that the SSH server is running.") from exc
        except (paramiko.SSHException, OSError, EOFError) as exc:
            raise MachineError(f"Could not connect to {ep.label}: {exc or type(exc).__name__}") from exc

    @staticmethod
    def _known_hosts_holding(ep: _Endpoint) -> str:
        name = host_key_name(ep.host, ep.port)
        ours = known_hosts_path()
        try:
            if paramiko.HostKeys(str(ours)).lookup(name) is not None:
                return str(ours)
        except (OSError, paramiko.SSHException):
            pass
        return str(Path.home() / ".ssh" / "known_hosts")

    # ------------------------------------------------------------------ plumbing
    def _transport(self) -> paramiko.Transport:
        with self._lock:
            if not self.is_connected:
                self._drop()
                self._open()
            return self._client.get_transport()

    def _call(self, op: Callable[[], T]) -> T:
        """Run ``op``; if it failed because the link dropped, reconnect once and retry."""
        try:
            return op()
        except (EOFError, OSError, paramiko.SSHException):
            if self.is_connected:
                raise
            log.info("connection to %s dropped; reconnecting", self.profile.address)
            with self._lock:
                self._drop()
            return op()

    def _open_channel(self) -> paramiko.Channel:
        delay = 0.2
        for attempt in range(4):
            try:
                return self._call(lambda: self._transport().open_session(timeout=CONNECT_TIMEOUT))
            except paramiko.ChannelException as exc:  # MaxSessions reached: wait for a free slot
                if attempt == 3:
                    raise MachineError(f"{self.profile.name} refused a new session: {exc}") from exc
                time.sleep(delay)
                delay *= 2
            except MachineError:
                raise
            except (EOFError, OSError, paramiko.SSHException) as exc:
                raise MachineError(f"Lost the connection to {self.profile.name}: {exc}") from exc
        raise AssertionError("unreachable")

    def _sftp_op(self, fn: Callable[[paramiko.SFTPClient], T], path: str) -> T:
        def op() -> T:
            with self._sftp_lock:
                transport = self._transport()
                sftp = self._sftp
                if sftp is None or sftp.get_channel().get_transport() is not transport \
                        or sftp.get_channel().closed:
                    sftp = self._sftp = paramiko.SFTPClient.from_transport(transport)
                return fn(sftp)

        try:
            return self._call(op)
        except MachineError:
            raise
        except OSError as exc:
            raise machine_os_error(exc, path) from exc
        except (EOFError, paramiko.SSHException) as exc:
            raise MachineError(f"File access on {self.profile.name} failed: {exc}") from exc

    # ------------------------------------------------------------------ commands
    def run(
        self,
        argv: Sequence[str] | str,
        *,
        timeout: float | None = 60.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
    ) -> CommandResult:
        command = build_command(argv, cwd=cwd, env=env)
        with self._exec_slots:
            channel = self._open_channel()
            try:
                channel.exec_command(command)
                if stdin:
                    channel.sendall(stdin)
                channel.shutdown_write()
                out, err, code = _collect(channel, timeout)
            except TimeoutError:
                raise MachineError(f"{_program(argv)} timed out after {timeout:g} s") from None
            except (EOFError, OSError, paramiko.SSHException) as exc:
                raise MachineError(f"Lost the connection to {self.profile.name}: {exc}") from exc
            finally:
                channel.close()
        return CommandResult(code, _decode(out), _decode(err))

    def stream(
        self,
        argv: Sequence[str] | str,
        on_line: Callable[[str], None],
        stop: threading.Event,
        *,
        env: dict[str, str] | None = None,
    ) -> int:
        """Run ``argv`` remotely, calling ``on_line`` per output line (stderr merged)."""
        if stop.is_set():
            return -1
        channel = self._open_channel()
        remote_pid: int | None = None
        finished = False
        try:
            channel.set_combine_stderr(True)
            channel.exec_command(build_command(argv, env=env, echo_pid=True))
            channel.shutdown_write()
            buf = b""
            while not stop.is_set():
                readable, _, _ = select.select([channel], [], [], 0.1)
                if not readable and not channel.recv_ready():
                    if channel.closed:
                        break
                    continue
                chunk = channel.recv(65536)
                if not chunk:
                    break
                *lines, buf = (buf + chunk).split(b"\n")
                for raw in lines:
                    text = raw.decode("utf-8", errors="replace").rstrip("\r")
                    if remote_pid is None and text.startswith(PID_MARKER):
                        digits = text[len(PID_MARKER):].strip()
                        remote_pid = int(digits) if digits.isdigit() else None
                        continue
                    on_line(text)
            if stop.is_set():
                return -1
            if buf:
                on_line(buf.decode("utf-8", errors="replace").rstrip("\r"))
            finished = True
            return channel.recv_exit_status()
        except (EOFError, OSError, paramiko.SSHException) as exc:
            raise MachineError(f"Lost the connection to {self.profile.name}: {exc}") from exc
        finally:
            channel.close()
            if not finished and remote_pid and self.is_connected:
                # Stopped early: make sure the command dies (a dropped link kills it anyway).
                self._kill_quietly(remote_pid)

    def _kill_quietly(self, pid: int) -> None:
        try:
            self.run(kill_tree_script(pid, grace_s=2.0), timeout=15)
        except MachineError as exc:
            log.info("could not stop remote process %s on %s: %s", pid, self.profile.name, exc)

    # ------------------------------------------------------------------ filesystem
    def home(self) -> str:
        if self._home is None:
            self._home = self._sftp_op(lambda s: s.normalize("."), "~")
        return self._home

    def join(self, *parts: str) -> str:
        return posixpath.join(*parts)

    def parent(self, path: str) -> str:
        trimmed = path.rstrip("/") or "/"
        return posixpath.dirname(trimmed) or "/"

    def basename(self, path: str) -> str:
        return posixpath.basename(path.rstrip("/"))

    def expand(self, path: str) -> str:
        if not path or path == "~":
            return self.home()
        if path.startswith("~/"):
            return posixpath.normpath(self.home() + path[1:])
        if not path.startswith("/"):
            return posixpath.normpath(posixpath.join(self.home(), path))
        return posixpath.normpath(path)

    def _stat(self, path: str) -> paramiko.SFTPAttributes | None:
        """``stat`` that returns ``None`` for missing / inaccessible paths."""
        target = self.expand(path)
        try:
            return self._sftp_op(lambda sftp: sftp.stat(target), target)
        except MachineError as exc:
            if isinstance(exc, OSError):
                return None
            raise

    def exists(self, path: str) -> bool:
        return self._stat(path) is not None

    def is_dir(self, path: str) -> bool:
        st = self._stat(path)
        return st is not None and stat.S_ISDIR(st.st_mode or 0)

    def listdir(self, path: str) -> list[FileEntry]:
        folder = self.expand(path)

        def op(sftp: paramiko.SFTPClient) -> list[FileEntry]:
            entries = []
            for attr in sftp.listdir_attr(folder):
                full = posixpath.join(folder, attr.filename)
                mode, size, mtime = attr.st_mode or 0, attr.st_size or 0, attr.st_mtime or 0
                if stat.S_ISLNK(mode):
                    try:
                        target = sftp.stat(full)
                        mode, size, mtime = target.st_mode or 0, target.st_size or 0, target.st_mtime or 0
                    except OSError:
                        pass  # dangling link: show it as a file
                is_dir = stat.S_ISDIR(mode)
                entries.append(FileEntry(attr.filename, full, is_dir, 0 if is_dir else size,
                                         float(mtime) + self.clock_offset if mtime else 0.0))
            return entries

        entries = self._sftp_op(op, folder)
        entries.sort(key=lambda e: (not e.is_dir, e.name.lower(), e.name))
        return entries

    def mkdir(self, path: str, parents: bool = True) -> None:
        folder = self.expand(path)
        self._sftp_op(lambda s: _mkdirs(s, folder, parents), folder)

    def read_bytes(self, path: str, offset: int = 0, length: int | None = None) -> bytes:
        target = self.expand(path)

        def op(sftp: paramiko.SFTPClient) -> bytes:
            with sftp.open(target, "rb") as fh:
                if offset:
                    fh.seek(offset)
                if length is None:
                    fh.prefetch()
                    return fh.read()
                return fh.read(length)

        return self._sftp_op(op, target)

    def write_bytes(self, path: str, data: bytes) -> None:
        target = self.expand(path)
        tmp = posixpath.join(posixpath.dirname(target),
                             f".{posixpath.basename(target)}.{uuid.uuid4().hex[:8]}.tmp")

        def op(sftp: paramiko.SFTPClient) -> None:
            try:
                fh = sftp.open(tmp, "wb")
            except FileNotFoundError:
                _mkdirs(sftp, posixpath.dirname(target), True)
                fh = sftp.open(tmp, "wb")
            try:
                with fh:
                    fh.set_pipelined(True)
                    fh.write(data)
                _rename_over(sftp, tmp, target)
            except BaseException:
                _remove_quietly(sftp, tmp)
                raise

        self._sftp_op(op, target)

    def size(self, path: str) -> int:
        target = self.expand(path)
        return int(self._sftp_op(lambda s: s.stat(target).st_size or 0, target))

    def remove(self, path: str, recursive: bool = False) -> None:
        """Delete a file or folder (folders need ``recursive`` unless empty); missing paths are ignored."""
        target = self.expand(path)

        def op(sftp: paramiko.SFTPClient) -> bool:
            try:
                st = sftp.lstat(target)
            except FileNotFoundError:
                return False
            if not stat.S_ISDIR(st.st_mode or 0):
                sftp.remove(target)
                return False
            if recursive:
                return True
            sftp.rmdir(target)
            return False

        if self._sftp_op(op, target):
            self._remove_tree(target)

    def _remove_tree(self, folder: str) -> None:
        depth = len([p for p in folder.split("/") if p])
        if depth < 2 or folder in (self.home(), posixpath.dirname(self.home())):
            raise MachineError(f"Refusing to delete {folder} recursively.")
        result = self.run(["rm", "-rf", "--", folder], timeout=600)
        if not result.ok:
            raise MachineError(f"Could not delete {folder}: {result.stderr.strip() or result.returncode}")

    def _transfer(self, fn: Callable[[paramiko.SFTPClient], None], path: str) -> None:
        with self._transfer_slots:
            try:
                sftp = self._call(lambda: paramiko.SFTPClient.from_transport(self._transport()))
            except MachineError:
                raise
            except (EOFError, OSError, paramiko.SSHException) as exc:
                raise MachineError(f"File transfer with {self.profile.name} failed: {exc}") from exc
            try:
                fn(sftp)
            except MachineError:
                raise
            except OSError as exc:
                raise machine_os_error(exc, path) from exc
            except (EOFError, paramiko.SSHException) as exc:
                raise MachineError(f"File transfer with {self.profile.name} failed: {exc}") from exc
            finally:
                sftp.close()

    def download(
        self,
        remote_path: str,
        local_path: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        source = self.expand(remote_path)
        target = Path(local_path)
        part = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")

        def fn(sftp: paramiko.SFTPClient) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                sftp.get(source, str(part), callback=progress, max_concurrent_prefetch_requests=64)
                os.replace(part, target)
            except BaseException:
                part.unlink(missing_ok=True)
                raise

        self._transfer(fn, source)

    def upload(self, local_path: str, remote_path: str) -> None:
        target = self.expand(remote_path)
        tmp = posixpath.join(posixpath.dirname(target),
                             f".{posixpath.basename(target)}.{uuid.uuid4().hex[:8]}.tmp")

        def fn(sftp: paramiko.SFTPClient) -> None:
            try:
                sftp.put(local_path, tmp)
            except FileNotFoundError:
                if not os.path.isfile(local_path):
                    raise
                _mkdirs(sftp, posixpath.dirname(target), True)
                sftp.put(local_path, tmp)
            try:
                _rename_over(sftp, tmp, target)
            except BaseException:
                _remove_quietly(sftp, tmp)
                raise

        self._transfer(fn, target)

    # ------------------------------------------------------------------ python & runtime
    def runtime_dir(self) -> str:
        with self._lock:
            cached = self._runtime_dir
        if cached is None:
            cached = deploy_remote(self, REMOTE_RUNTIME_BASE)
            with self._lock:
                self._runtime_dir = cached
        return cached

    # ------------------------------------------------------------------ processes
    def is_alive(self, pid: int) -> bool:
        return self.run(alive_script(pid), timeout=30).ok

    def terminate(self, pid: int) -> None:
        self.run(kill_tree_script(pid), timeout=60)


# ============================================================================ sftp helpers
def _mkdirs(sftp: paramiko.SFTPClient, folder: str, parents: bool) -> None:
    try:
        sftp.mkdir(folder)
        return
    except FileNotFoundError:
        parent = posixpath.dirname(folder)
        if not parents or parent == folder:
            raise
        _mkdirs(sftp, parent, True)
        sftp.mkdir(folder)
    except OSError:
        # SFTP v3 has no "already exists" status: check what is there.
        try:
            st = sftp.stat(folder)
        except OSError:
            raise
        if not stat.S_ISDIR(st.st_mode or 0):
            raise FileExistsError(17, "File exists (not a folder)", folder) from None


def _rename_over(sftp: paramiko.SFTPClient, src: str, dst: str) -> None:
    """Atomic rename replacing ``dst`` (posix-rename@openssh.com, with a fallback)."""
    try:
        sftp.posix_rename(src, dst)
    except OSError:
        with contextlib.suppress(FileNotFoundError):
            sftp.remove(dst)
        sftp.rename(src, dst)


def _remove_quietly(sftp: paramiko.SFTPClient, path: str) -> None:
    with contextlib.suppress(OSError, paramiko.SSHException, EOFError):
        sftp.remove(path)


def _collect(channel: paramiko.Channel, timeout: float | None) -> tuple[bytes, bytes, int]:
    """Read stdout and stderr until the command ends; raises ``TimeoutError``.

    Both streams are drained continuously (they share the channel's flow-control
    window). The command has ended once EOF arrives, or shortly after its exit status
    if a background process keeps the output open.
    """
    out, err = bytearray(), bytearray()
    deadline = None if timeout is None else time.monotonic() + timeout
    exited_at: float | None = None
    idle = 0.002
    while True:
        got = False
        while channel.recv_ready():
            out += channel.recv(65536)
            got = True
        while channel.recv_stderr_ready():
            err += channel.recv_stderr(65536)
            got = True
        if got:
            idle = 0.002
            continue
        if channel.eof_received or channel.closed:
            break
        now = time.monotonic()
        if channel.exit_status_ready():
            exited_at = exited_at or now
            if now - exited_at > 0.5:
                break
        if deadline is not None and now > deadline:
            raise TimeoutError
        time.sleep(idle)
        idle = min(idle * 2, 0.05)
    # The exit status follows EOF by a hair; a closed channel without one reports -1.
    channel.status_event.wait(5.0)
    return bytes(out), bytes(err), channel.exit_status if channel.exit_status_ready() else -1


def _decode(data: bytes | bytearray) -> str:
    return bytes(data).decode("utf-8", errors="replace").replace("\r\n", "\n")


def _program(argv: Sequence[str] | str) -> str:
    """The program name of a command (for messages)."""
    if isinstance(argv, str):
        words = argv.split()
        return words[0] if words else argv
    return str(argv[0]) if argv else ""
