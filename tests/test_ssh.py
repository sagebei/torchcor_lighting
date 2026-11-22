"""SSH machines: command construction, ``~/.ssh/config`` import, host keys, and a real
exec + SFTP round trip against an in-process paramiko server (no sshd needed)."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import keyring
import paramiko
import pytest
from keyring.backend import KeyringBackend
from paramiko import SFTPAttributes, SFTPHandle, SFTPServer, SFTPServerInterface
from tests.fixtures.host import TORCHCOR_READY

from torchcor_studio.core import environment, secrets
from torchcor_studio.core.machines import create_machine, sshconfig
from torchcor_studio.core.machines.base import MachineError
from torchcor_studio.core.machines.deploy import runtime_digest
from torchcor_studio.core.machines.ssh import (
    PID_MARKER,
    HostKeyChangedError,
    SSHMachine,
    alive_script,
    build_command,
    fingerprint,
    forget_host_key,
    host_key_name,
    kill_tree_script,
    known_hosts_path,
    load_private_key,
)
from torchcor_studio.core.models import AuthMethod, MachineKind, MachineProfile

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the test server runs commands with /bin/sh")
PASSWORD = "s3cret-pass"


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self) -> None:
        super().__init__()
        self.data: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.data.get((service, username))

    def set_password(self, service, username, password):
        self.data[(service, username)] = password

    def delete_password(self, service, username):
        self.data.pop((service, username), None)


@pytest.fixture(autouse=True)
def isolated_ssh(tmp_path, monkeypatch):
    """No real keychain, and an empty ~/.ssh/config (the user's must not affect tests)."""
    previous = keyring.get_keyring()
    keyring.set_keyring(MemoryKeyring())
    config = tmp_path / "ssh_config"
    config.write_text("")
    monkeypatch.setattr(sshconfig, "default_path", lambda: config)
    yield
    keyring.set_keyring(previous)


def _sh(command: str, **kwargs) -> subprocess.CompletedProcess:
    """Run an exec-request string the way sshd does: through the login shell."""
    return subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True, timeout=30, **kwargs)


def _matching(pattern: str) -> list[str]:
    """PIDs of the processes whose command line holds ``pattern``. The bracket keeps the
    shell that runs pgrep, whose own command line holds the pattern, from matching (Linux's
    pgrep lists it; macOS's leaves out its ancestors)."""
    return _sh(f"pgrep -f '[{pattern[0]}]{pattern[1:]}' || true").stdout.split()


# ============================================================================ commands
def test_build_command_quotes_every_word():
    argv = [sys.executable, "-c", "import sys; print(sys.argv[1:])", "a b", "it's", "$HOME", "`x`", "*"]
    command = build_command(argv)
    assert command.startswith("sh -c ")
    assert _sh(command).stdout.strip() == "['a b', \"it's\", '$HOME', '`x`', '*']"


def test_build_command_cwd_env_and_home(tmp_path):
    (tmp_path / "sub dir").mkdir()
    command = build_command([sys.executable, "-c", "import os; print(os.getcwd(), os.environ['TCS_X'])"],
                            cwd="~/sub dir", env={"TCS_X": "va'l ue"})
    out = _sh(command, env={**os.environ, "HOME": str(tmp_path)}).stdout.strip()
    assert out.endswith("sub dir va'l ue")
    assert _sh(build_command(["true"], cwd=str(tmp_path / "missing"))).returncode != 0
    assert _sh(build_command("echo $((1 + 2))")).stdout.strip() == "3"
    with pytest.raises(ValueError):
        build_command(["true"], env={"BAD-NAME": "1"})
    with pytest.raises(ValueError):
        build_command([])


def test_build_command_echoes_the_pid_first():
    out = _sh(build_command(["sh", "-c", "echo $$"], echo_pid=True)).stdout.split()
    assert out[0].startswith(PID_MARKER) and out[0][len(PID_MARKER):] == out[1]  # exec keeps the pid


def test_kill_and_alive_scripts_handle_process_trees():
    marker = f"tcs-{uuid.uuid4().hex[:8]}"
    sleeper = f"{sys.executable} -c 'import time; time.sleep(60)' {marker}"
    proc = subprocess.Popen(["sh", "-c", f"{sleeper} & {sleeper}; true"])
    try:
        time.sleep(0.3)
        assert _matching(marker)  # the tree is running
        assert _sh(alive_script(proc.pid)).returncode == 0
        assert _sh(kill_tree_script(proc.pid, grace_s=2)).returncode == 0
        proc.wait(timeout=10)
        time.sleep(0.2)
        assert _matching(marker) == []
        assert _sh(alive_script(proc.pid)).returncode != 0
    finally:
        if proc.poll() is None:
            proc.kill()


def _fake_python(path: Path, version: str, torch: str | None = None, cuda: str | None = None,
                 torchcor: str | None = None) -> None:
    """An executable that answers the environment probe like a real interpreter would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rest = json.dumps({"version": version, "prefix": str(path.parent.parent), "torch": torch,
                       "cuda": cuda, "torchcor": torchcor})[1:]
    # the probe runs `python -c <snippet> <path>` and reports the path it was invoked as
    path.write_text("#!/bin/sh\nprintf '{\"path\": \"%s\", " + rest.replace("%", "%%") + "\\n' \"$3\"\n")
    path.chmod(0o755)


def test_scan_finds_every_kind_of_environment_and_ranks_them(tmp_path):
    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    _fake_python(bin_dir / "python3", "3.8.10")  # the system Python on PATH: too old, no torch
    _fake_python(home / "miniconda3/envs/cpu/bin/python", "3.11.6", torch="2.5.1", torchcor="1.9.0")
    custom = home / "Libraries/miniconda3/envs/tc"  # only conda's own list knows this one
    _fake_python(custom / "bin/python", "3.11.6", torch="2.7.1+cu128", cuda="12.8")
    (home / ".conda").mkdir()
    (home / ".conda/environments.txt").write_text(f"{custom}\n")
    _fake_python(home / "project/.venv/bin/python", "3.12.4", torch="2.7.1+cu128", cuda="12.8",
                 torchcor="1.9.0")
    env = {"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin"}
    scanned = environment.rank_pythons(environment.parse_scan(_sh(environment._remote_scan_script(), env=env).stdout))
    found = [c for c in scanned if c.path.startswith(str(tmp_path))]  # /usr/bin may hold a real one too
    paths = [c.path for c in found]
    assert paths == [str(home / "project/.venv/bin/python"),  # TorchCor + CUDA PyTorch
                     str(home / "miniconda3/envs/cpu/bin/python"),  # TorchCor, CPU only
                     str(custom / "bin/python"),  # CUDA PyTorch: only TorchCor is missing
                     str(bin_dir / "python3")]  # nothing usable
    assert [c.readiness for c in found] == [0, 1, 2, 5]


# ============================================================================ ssh config# ============================================================================ ssh config
SSH_CONFIG = """\
Host *
    ServerAliveInterval 30

Host bastion
    HostName bastion.example.org
    User jumper
    Port 2200

Host gpu1 gpu1-alias  # two names, same server
    HostName gpu1.example.org
    User alice
    IdentityFile ~/.ssh/id_lab
    ProxyJump bastion

Host *.lab !secret.lab
    User labuser

Host direct
    HostName 10.0.0.5
    Port 2222
"""


def test_sshconfig_import_profiles(tmp_path):
    path = tmp_path / "config"
    path.write_text(SSH_CONFIG)
    assert sshconfig.aliases(path) == ["bastion", "gpu1", "gpu1-alias", "direct"]
    profiles = {p.name: p for p in sshconfig.import_profiles(path)}
    assert set(profiles) == {"bastion", "gpu1", "direct"}  # gpu1-alias is the same server
    gpu1 = profiles["gpu1"]
    assert gpu1.kind == MachineKind.SSH and gpu1.host == "gpu1.example.org" and gpu1.username == "alice"
    assert gpu1.auth == AuthMethod.KEY and gpu1.key_path == os.path.expanduser("~/.ssh/id_lab")
    assert gpu1.jump_host == "jumper@bastion.example.org:2200"
    direct = profiles["direct"]
    assert (direct.host, direct.port, direct.auth, direct.jump_host) == ("10.0.0.5", 2222, AuthMethod.AGENT, "")
    assert sshconfig.lookup("node.lab", path).user == "labuser"
    assert sshconfig.lookup("unknown-host", path).hostname == "unknown-host"
    assert sshconfig.import_profiles(tmp_path / "missing") == []


def test_hops_parse_and_format():
    assert sshconfig.parse_hop("alice@host:2222") == ("alice", "host", 2222)
    assert sshconfig.parse_hop("host") == ("", "host", 22)
    assert sshconfig.parse_hop("ssh://bob@[fe80::1]:23") == ("bob", "fe80::1", 23)
    assert sshconfig.format_hop("bob", "fe80::1", 23) == "bob@[fe80::1]:23"
    assert sshconfig.format_hop("", "h", 22) == "h"


# ============================================================================ auth plumbing
def test_endpoints_and_credential_errors(tmp_path):
    no_password = SSHMachine(MachineProfile(name="pw", kind=MachineKind.SSH, host="h", auth=AuthMethod.PASSWORD))
    with pytest.raises(MachineError, match="No password"):
        no_password.endpoints()
    secrets.remember(no_password.id, secrets.PASSWORD, "from-keyring")
    assert no_password.endpoints()[-1].password == "from-keyring"
    with pytest.raises(MachineError, match="does not exist"):
        SSHMachine(MachineProfile(name="k", kind=MachineKind.SSH, host="h", auth=AuthMethod.KEY,
                                  key_path=str(tmp_path / "nope"))).endpoints()
    with pytest.raises(MachineError, match="No host"):
        SSHMachine(MachineProfile(name="x", kind=MachineKind.SSH)).endpoints()
    chain = SSHMachine(MachineProfile(name="j", kind=MachineKind.SSH, host="target", username="me",
                                      jump_host="a@hop1:2201, hop2")).endpoints()
    assert [(e.user, e.host, e.port) for e in chain] == [("a", "hop1", 2201), ("me", "hop2", 22), ("me", "target", 22)]


def test_private_key_messages(tmp_path):
    key = paramiko.ECDSAKey.generate()
    encrypted, plain = tmp_path / "enc", tmp_path / "plain"
    key.write_private_key_file(str(encrypted), password="pp")
    key.write_private_key_file(str(plain))
    assert load_private_key(str(encrypted), "pp").get_base64() == key.get_base64()
    assert load_private_key(str(plain), "ignored").get_base64() == key.get_base64()
    with pytest.raises(MachineError, match="protected by a passphrase"):
        load_private_key(str(encrypted))
    with pytest.raises(MachineError, match="passphrase is wrong"):
        load_private_key(str(encrypted), "bad")


def test_changed_host_key_is_refused_with_a_fake(monkeypatch):
    old, new = paramiko.ECDSAKey.generate(), paramiko.ECDSAKey.generate()

    def connect(self, hostname, *args, **kwargs):
        raise paramiko.BadHostKeyException(hostname, new, old)

    monkeypatch.setattr(paramiko.SSHClient, "connect", connect)
    machine = SSHMachine(MachineProfile(name="x", kind=MachineKind.SSH, host="example.org", port=2222,
                                        auth=AuthMethod.PASSWORD), password="x")
    with pytest.raises(HostKeyChangedError) as info:
        machine.connect()
    err = info.value
    assert err.host == "[example.org]:2222"
    assert err.expected == fingerprint(old) and err.received == fingerprint(new)
    assert "has changed" in str(err) and "man-in-the-middle" in str(err)
    assert not machine.is_connected


# ============================================================================ test server
def _status(exc: OSError) -> int:
    return SFTPServer.convert_errno(exc.errno)


class _Handle(SFTPHandle):
    def stat(self):
        try:
            return SFTPAttributes.from_stat(os.fstat(self.readfile.fileno()))
        except OSError as exc:
            return _status(exc)


class _SFTP(SFTPServerInterface):
    """Serves the real filesystem; ``.`` is the test home folder."""

    home = "/"

    def canonicalize(self, path):
        if path in (".", ""):
            return self.home
        return os.path.normpath(path if path.startswith("/") else os.path.join(self.home, path))

    def list_folder(self, path):
        try:
            out = []
            for name in os.listdir(path):
                attr = SFTPAttributes.from_stat(os.lstat(os.path.join(path, name)))
                attr.filename = name
                out.append(attr)
            return out
        except OSError as exc:
            return _status(exc)

    def stat(self, path):
        try:
            return SFTPAttributes.from_stat(os.stat(path))
        except OSError as exc:
            return _status(exc)

    def lstat(self, path):
        try:
            return SFTPAttributes.from_stat(os.lstat(path))
        except OSError as exc:
            return _status(exc)

    def open(self, path, flags, attr):
        try:
            fd = os.open(path, flags, 0o644)
        except OSError as exc:
            return _status(exc)
        writing = flags & (os.O_WRONLY | os.O_RDWR)
        mode = ("ab" if flags & os.O_APPEND else "r+b" if flags & os.O_RDWR else "wb") if writing else "rb"
        handle = _Handle(flags)
        handle.filename = path
        handle.readfile = handle.writefile = os.fdopen(fd, mode)
        return handle

    def _call(self, fn, *args):
        try:
            fn(*args)
        except OSError as exc:
            return _status(exc)
        return paramiko.SFTP_OK

    def remove(self, path):
        return self._call(os.remove, path)

    def rename(self, oldpath, newpath):
        return paramiko.SFTP_FAILURE if os.path.exists(newpath) else self._call(os.rename, oldpath, newpath)

    def posix_rename(self, oldpath, newpath):
        return self._call(os.replace, oldpath, newpath)

    def mkdir(self, path, attr):
        if os.path.exists(path):
            return paramiko.SFTP_FAILURE  # what OpenSSH answers
        return self._call(os.mkdir, path)

    def rmdir(self, path):
        return self._call(os.rmdir, path)


class _Server(paramiko.ServerInterface):
    def __init__(self, owner: SSHTestServer) -> None:
        self.owner = owner
        self.tunnels: dict[int, tuple[str, int]] = {}

    def get_allowed_auths(self, username):
        return "password,publickey"

    def check_auth_password(self, username, password):
        return paramiko.AUTH_SUCCESSFUL if password == PASSWORD else paramiko.AUTH_FAILED

    def check_auth_publickey(self, username, key):
        ok = self.owner.client_key is not None and key.get_base64() == self.owner.client_key.get_base64()
        return paramiko.AUTH_SUCCESSFUL if ok else paramiko.AUTH_FAILED

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED if kind == "session" else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_channel_exec_request(self, channel, command):
        command = command.decode() if isinstance(command, bytes) else command
        self.owner.commands.append(command)
        threading.Thread(target=_run_exec, args=(channel, command, self.owner.home), daemon=True).start()
        return True

    def check_channel_direct_tcpip_request(self, chanid, origin, destination):
        self.tunnels[chanid] = destination
        self.owner.tunnels.append(destination)
        return paramiko.OPEN_SUCCEEDED


def _run_exec(channel: paramiko.Channel, command: str, home: str) -> None:
    """Run an exec request with /bin/sh (standing in for the user's login shell)."""
    started = time.monotonic()
    proc = subprocess.Popen(command, shell=True, cwd=home, env={**os.environ, "HOME": home},
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def pump(src, send):
        for chunk in iter(lambda: src.read1(65536), b""):
            try:
                send(chunk)
            except OSError:
                break

    def feed():
        try:
            while data := channel.recv(65536):
                proc.stdin.write(data)
                proc.stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError):
                proc.stdin.close()

    pumps = [threading.Thread(target=pump, args=(proc.stdout, channel.sendall), daemon=True),
             threading.Thread(target=pump, args=(proc.stderr, channel.sendall_stderr), daemon=True)]
    for t in [*pumps, threading.Thread(target=feed, daemon=True)]:
        t.start()
    code = None
    while code is None:
        try:
            code = proc.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            if channel.closed:
                proc.kill()
    for t in pumps:
        t.join(2)
    for pipe in (proc.stdout, proc.stderr):
        pipe.close()
    time.sleep(max(0.0, 0.01 - (time.monotonic() - started)))
    try:
        channel.send_exit_status(code if code >= 0 else 128 - code)
        channel.shutdown_write()
        channel.close()
    except (OSError, EOFError):
        pass


def _bridge(chan: paramiko.Channel, host: str, port: int) -> None:
    sock = socket.create_connection((host, port))

    def upstream():
        with contextlib.suppress(OSError):
            while data := chan.recv(65536):
                sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)

    def downstream():
        with contextlib.suppress(OSError):
            while data := sock.recv(65536):
                chan.sendall(data)
        with contextlib.suppress(OSError, EOFError):  # the client may have hung up first
            chan.close()
        sock.close()

    threading.Thread(target=upstream, daemon=True).start()
    threading.Thread(target=downstream, daemon=True).start()


class SSHTestServer:
    """A tiny SSH server on 127.0.0.1: password / public-key auth, exec, SFTP, direct-tcpip."""

    def __init__(self, home: str) -> None:
        self.home = home
        self.host_key = paramiko.ECDSAKey.generate()
        self.client_key: paramiko.PKey | None = None
        self.commands: list[str] = []
        self.tunnels: list[tuple[str, int]] = []
        self.transports: list[paramiko.Transport] = []
        self._channels: list[paramiko.Channel] = []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        transport = paramiko.Transport(conn)
        transport.add_server_key(self.host_key)
        _SFTP.home = self.home
        transport.set_subsystem_handler("sftp", SFTPServer, _SFTP)
        server = _Server(self)
        self.transports.append(transport)
        try:
            transport.start_server(server=server)
        except (paramiko.SSHException, EOFError, OSError):
            return
        while transport.is_active():
            chan = transport.accept(0.5)
            if chan is None:
                continue
            dest = server.tunnels.get(chan.get_id())
            if dest is not None:
                threading.Thread(target=_bridge, args=(chan, *dest), daemon=True).start()
            else:
                self._channels.append(chan)  # paramiko only holds channels weakly

    def drop_connections(self) -> None:
        for transport in self.transports:
            transport.close()

    def close(self) -> None:
        self.sock.close()
        self.drop_connections()


@pytest.fixture
def server(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    srv = SSHTestServer(str(home))
    yield srv
    srv.close()


def _profile(server: SSHTestServer, **changes) -> MachineProfile:
    base = dict(name="test server", kind=MachineKind.SSH, host="127.0.0.1", port=server.port,
                username="tester", auth=AuthMethod.PASSWORD)
    base.update(changes)
    return MachineProfile(**base)


@pytest.fixture
def remote(server):
    machine = SSHMachine(_profile(server), password=PASSWORD)
    yield machine
    machine.close()


# ============================================================================ end to end
def test_connect_trusts_new_host_key_on_first_use(server, remote):
    assert isinstance(create_machine(remote.profile), SSHMachine)
    remote.connect()
    assert remote.is_connected
    name = host_key_name("127.0.0.1", server.port)
    assert remote.host_key_fingerprint == fingerprint(server.host_key)
    assert remote.new_host_keys == [f"{name} {fingerprint(server.host_key)}"]
    assert name in known_hosts_path().read_text()
    again = SSHMachine(remote.profile, password=PASSWORD)
    again.connect()  # now known: no new trust
    assert again.new_host_keys == []
    again.close()
    again.close()  # safe twice


def test_changed_host_key_is_refused_by_the_server(server, remote):
    remote.connect()
    remote.close()
    server.host_key = paramiko.ECDSAKey.generate()  # "reinstalled" server
    with pytest.raises(HostKeyChangedError) as info:
        SSHMachine(remote.profile, password=PASSWORD).connect()
    assert info.value.known_hosts == str(known_hosts_path())
    assert forget_host_key("127.0.0.1", server.port)
    fresh = SSHMachine(remote.profile, password=PASSWORD)
    fresh.connect()
    assert fresh.new_host_keys
    fresh.close()


def test_wrong_password_and_refused_port(server):
    with pytest.raises(MachineError, match="Authentication failed"):
        SSHMachine(_profile(server), password="wrong").connect()
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()
    with pytest.raises(MachineError, match="refused"):
        SSHMachine(_profile(server, port=closed_port), password=PASSWORD).connect()


def test_run_over_ssh(server, remote):
    py = sys.executable
    result = remote.run([py, "-c", "import sys; print(sys.argv[1:])", "a b", "it's"])
    assert result.ok and result.stdout.strip() == "['a b', \"it's\"]"
    assert server.commands[-1].startswith("sh -c ")
    assert remote.run(["cat"], stdin=b"via stdin").stdout == "via stdin"
    failed = remote.run(["sh", "-c", "echo oops >&2; exit 7"])
    assert (failed.returncode, failed.stderr) == (7, "oops\n")
    assert remote.run(["pwd"], cwd="~").stdout.strip().endswith("home")
    assert remote.run(["sh", "-c", "echo $TCS_V"], env={"TCS_V": "1 2"}).stdout.strip() == "1 2"
    assert remote.run(["definitely-not-a-command-tcs"]).returncode == 127
    with pytest.raises(MachineError, match="timed out"):
        remote.run(["sleep", "10"], timeout=0.5)


def test_stream_and_stop_over_ssh(remote):
    lines: list[str] = []
    assert remote.stream(["sh", "-c", "echo one; echo two >&2; exit 4"], lines.append, threading.Event()) == 4
    assert lines == ["one", "two"]  # stderr merged, pid marker hidden

    marker = f"tcs-{uuid.uuid4().hex[:8]}"
    stop = threading.Event()
    ticks: list[str] = []

    def on_tick(line: str) -> None:
        ticks.append(line)
        if len(ticks) == 3:
            stop.set()

    started = time.monotonic()
    code = remote.stream(["sh", "-c", f"while true; do echo {marker}; sleep 0.05; done"], on_tick, stop)
    assert code == -1 and time.monotonic() - started < 10
    time.sleep(0.5)
    assert _matching(f"echo {marker}") == []  # remote loop was killed


def test_sftp_file_operations(server, remote):
    home = server.home
    assert remote.home() == home and remote.expand("~/a/../b") == f"{home}/b"
    assert remote.expand("rel") == f"{home}/rel" and remote.expand("/abs/./x") == "/abs/x"
    remote.mkdir("~/runs/r1/viz")
    remote.mkdir("~/runs/r1/viz")  # exists: fine
    remote.write_text("~/runs/r1/status.json", '{"state": "running"}')
    remote.write_bytes("~/runs/new/deep/file.bin", b"0123456789")  # parents created
    assert remote.read_json("~/runs/r1/status.json") == {"state": "running"}
    assert remote.read_bytes("~/runs/new/deep/file.bin", offset=3, length=4) == b"3456"
    assert remote.size("~/runs/new/deep/file.bin") == 10
    assert remote.exists("~/runs") and remote.is_dir("~/runs") and not remote.is_dir("~/runs/r1/status.json")
    assert not remote.exists("~/nothing") and not remote.is_dir("~/nothing")
    os.symlink(f"{home}/runs", f"{home}/runs-link")
    names = [(e.name, e.is_dir) for e in remote.listdir("~")]
    assert names == [("runs", True), ("runs-link", True)]
    assert [e.name for e in remote.listdir("~/runs/r1")] == ["viz", "status.json"]
    assert not [n for n in os.listdir(f"{home}/runs/r1") if n.endswith(".tmp")]
    with pytest.raises(FileNotFoundError) as info:
        remote.read_bytes("~/missing.txt")
    assert isinstance(info.value, MachineError)
    remote.remove("~/runs/r1/status.json")
    remote.remove("~/runs/new", recursive=True)
    remote.remove("~/never-there")
    assert [e.name for e in remote.listdir("~/runs")] == ["r1"]
    with pytest.raises(MachineError, match="Refusing"):
        remote.remove("~", recursive=True)


def test_transfers(server, remote, tmp_path):
    payload = os.urandom(2 * 1024 * 1024 + 3)
    local = tmp_path / "local.bin"
    local.write_bytes(payload)
    remote.upload(str(local), "~/up/dir/data.bin")
    seen: list[tuple[int, int]] = []
    back = tmp_path / "back" / "data.bin"
    remote.download("~/up/dir/data.bin", str(back), progress=lambda a, b: seen.append((a, b)))
    assert back.read_bytes() == payload and seen[-1] == (len(payload), len(payload))
    assert remote.read_bytes("~/up/dir/data.bin") == payload


def test_concurrent_use_from_many_threads(remote):
    remote.write_text("~/c.txt", "x" * 1000)

    def work(i: int):
        if i % 3 == 0:
            return remote.run(["echo", str(i)]).stdout.strip()
        if i % 3 == 1:
            return len(remote.listdir("~"))
        return len(remote.read_bytes("~/c.txt", 10, 100))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(work, range(36)))
    assert results[0::3] == [str(i) for i in range(0, 36, 3)]
    assert set(results[2::3]) == {100}


def test_reconnects_once_after_the_link_drops(server, remote):
    assert remote.run(["echo", "first"]).ok
    server.drop_connections()
    time.sleep(0.3)
    assert not remote.is_connected
    assert remote.run(["echo", "second"]).stdout.strip() == "second"
    assert [e.name for e in remote.listdir("~")] == []


def test_jump_host_tunnel(server):
    machine = SSHMachine(_profile(server, jump_host=f"tester@127.0.0.1:{server.port}"), password=PASSWORD)
    try:
        assert machine.run(["echo", "through the bastion"]).stdout.strip() == "through the bastion"
        assert server.tunnels == [("127.0.0.1", server.port)]
    finally:
        machine.close()


def test_key_authentication(server, tmp_path):
    key = paramiko.ECDSAKey.generate()
    server.client_key = key
    path = tmp_path / "id_test"
    key.write_private_key_file(str(path), password="phrase")
    profile = _profile(server, auth=AuthMethod.KEY, key_path=str(path))
    secrets.remember(profile.id, secrets.PASSPHRASE, "phrase")
    machine = SSHMachine(profile)
    try:
        assert machine.run(["echo", "keyed"]).stdout.strip() == "keyed"
    finally:
        machine.close()


def test_runtime_python_and_processes_over_ssh(server, remote):
    remote.profile.python = sys.executable
    folder = remote.runtime_dir()
    assert folder == f"{server.home}/.cache/torchcor-studio/runtime/{runtime_digest()}"
    assert sorted(os.listdir(folder)) == ["tcs_runtime.py", "tcs_studio.py"]
    report = remote.probe_environment()
    assert report.error is None and report.ready == TORCHCOR_READY  # ready: torch and TorchCor
    proc = subprocess.Popen(["sleep", "60"])
    try:
        assert remote.is_alive(proc.pid)
        remote.terminate(proc.pid)
        proc.wait(timeout=10)
        assert not remote.is_alive(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_detect_and_find_pythons_over_ssh(server, remote):
    env_python = Path(server.home) / "miniconda3" / "envs" / "tc" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.symlink_to(sys.executable)  # a "conda env" whose python lacks TorchCor
    assert remote.detect_python()  # the best-ranked interpreter (here: none has TorchCor)
    found = environment.find_pythons(remote)
    paths = [c.path for c in found]
    assert str(env_python) in paths
    assert all(c.version for c in found)
