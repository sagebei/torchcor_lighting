"""This computer as a Machine, runtime deployment, credentials and Python environments."""

from __future__ import annotations

import os
import posixpath
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import keyring
import psutil
import pytest
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError
from tests.fixtures.packages import TORCHCOR_READY

from torchcor_studio.core import environment, secrets
from torchcor_studio.core.machines import create_machine
from torchcor_studio.core.machines.base import MachineError
from torchcor_studio.core.machines.deploy import (
    deploy_local,
    deploy_remote,
    runtime_bundle,
    runtime_digest,
)
from torchcor_studio.core.machines.local import LocalMachine, machine_os_error, python_candidates
from torchcor_studio.core.models import CommandResult, MachineKind, MachineProfile
from torchcor_studio.core.paths import cache_dir
from torchcor_studio.runtime import RUNTIME_FILES, runtime_path

#: These probe this Python for a ready TorchCor environment: torch and TorchCor installed.
needs_torchcor = pytest.mark.skipif(not TORCHCOR_READY, reason="needs torch and TorchCor in this Python")

POSIX = os.name != "nt"


class MemoryKeyring(KeyringBackend):
    """In-memory keyring so tests never touch the real OS keychain."""

    priority = 1

    def __init__(self) -> None:
        super().__init__()
        self.data: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.data.get((service, username))

    def set_password(self, service, username, password):
        self.data[(service, username)] = password

    def delete_password(self, service, username):
        if self.data.pop((service, username), None) is None:
            raise PasswordDeleteError(username)


@pytest.fixture(autouse=True)
def memory_keyring():
    previous = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    yield backend
    keyring.set_keyring(previous)


@pytest.fixture
def local() -> LocalMachine:
    return LocalMachine(MachineProfile(name="This computer"))


# ============================================================================ filesystem
def test_factory_and_basics(local, tmp_path):
    assert isinstance(create_machine(MachineProfile(name="x")), LocalMachine)
    assert local.is_connected
    local.connect()
    local.close()
    assert local.home() == str(Path.home())
    assert local.expand("~") == str(Path.home())
    assert local.join("a", "b") == os.path.join("a", "b")
    assert local.parent(str(tmp_path / "x")) == str(tmp_path)
    assert local.basename(str(tmp_path / "x") + os.sep) == "x"


def test_files_roundtrip_and_atomic_write(local, tmp_path):
    target = str(tmp_path / "new" / "deep" / "file.txt")
    local.write_text(target, "hello world")  # parent folders are created
    assert local.read_text(target) == "hello world"
    assert local.size(target) == 11
    assert local.read_bytes(target, offset=6) == b"world"
    assert local.read_bytes(target, offset=2, length=3) == b"llo"
    local.write_json(target, {"a": 1})
    assert local.read_json(target) == {"a": 1}
    leftovers = [p.name for p in Path(target).parent.iterdir() if p.name != "file.txt"]
    assert leftovers == []
    assert local.exists(target) and not local.is_dir(target) and local.is_dir(str(tmp_path))


def test_listdir_dirs_first_then_names(local, tmp_path):
    for name in ("beta", "Alpha"):
        (tmp_path / name).mkdir()
    for name in ("zeta.txt", "Apple.pts", "mesh.elem"):
        (tmp_path / name).write_bytes(b"12345")
    if POSIX:
        (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    entries = local.listdir(str(tmp_path))
    names = [e.name for e in entries]
    assert names[:2] == ["Alpha", "beta"]
    assert names[2:5] == ["Apple.pts", "dangling", "mesh.elem"] if POSIX else names[2:4] == ["Apple.pts", "mesh.elem"]
    pts = next(e for e in entries if e.name == "Apple.pts")
    assert not pts.is_dir and pts.size == 5 and pts.mtime > 0 and pts.path == str(tmp_path / "Apple.pts")
    assert next(e for e in entries if e.name == "beta").size == 0


def test_errors_are_machine_errors_and_os_errors(local, tmp_path):
    missing = str(tmp_path / "missing.txt")
    with pytest.raises(FileNotFoundError) as info:
        local.read_bytes(missing)
    assert isinstance(info.value, MachineError)
    assert info.value.errno is not None and missing in str(info.value)
    with pytest.raises(MachineError):
        local.listdir(missing)
    wrapped = machine_os_error(PermissionError(13, "Permission denied"), "/root")
    assert isinstance(wrapped, PermissionError) and isinstance(wrapped, MachineError)


def test_mkdir_and_remove(local, tmp_path):
    folder = tmp_path / "a" / "b"
    local.mkdir(str(folder))
    local.mkdir(str(folder))  # exists: fine
    (folder / "f.txt").write_text("x")
    with pytest.raises(MachineError):
        local.remove(str(tmp_path / "a"))  # not empty, not recursive
    local.remove(str(folder / "f.txt"))
    local.remove(str(tmp_path / "a"), recursive=True)
    assert not (tmp_path / "a").exists()
    local.remove(str(tmp_path / "never-existed"))  # ignored


def test_download_upload_with_progress(local, tmp_path):
    source = tmp_path / "big.bin"
    payload = os.urandom(3 * 1024 * 1024 + 17)
    source.write_bytes(payload)
    seen: list[tuple[int, int]] = []
    local.download(str(source), str(tmp_path / "copy" / "big.bin"), progress=lambda a, b: seen.append((a, b)))
    assert (tmp_path / "copy" / "big.bin").read_bytes() == payload
    assert seen[-1] == (len(payload), len(payload)) and len(seen) >= 3
    local.upload(str(source), str(tmp_path / "up" / "x.bin"))
    assert (tmp_path / "up" / "x.bin").read_bytes() == payload


# ============================================================================ commands
def test_run_quoting_stdin_cwd_env(local, tmp_path):
    py = sys.executable
    result = local.run([py, "-c", "import sys; print(sys.argv[1:])", "a b", "it's", "$HOME"])
    assert result.ok and result.stdout.strip() == "['a b', \"it's\", '$HOME']"
    assert local.run([py, "-c", "import sys; print(sys.stdin.read().upper())"], stdin=b"abc").stdout.strip() == "ABC"
    here = local.run([py, "-c", "import os; print(os.getcwd())"], cwd=str(tmp_path)).stdout.strip()
    assert os.path.realpath(here) == os.path.realpath(tmp_path)
    env = local.run([py, "-c", "import os; print(os.environ['TCS_TEST'])"], env={"TCS_TEST": "x y"})
    assert env.stdout.strip() == "x y"
    failing = local.run([py, "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"])
    assert failing.returncode == 3 and failing.stderr == "bad" and not failing.ok
    assert local.run(["definitely-not-a-command-tcs"]).returncode == 127
    assert local.run([py, "-V"], cwd=str(tmp_path / "nope")).returncode == 2
    if POSIX:
        assert local.run("echo $((6 * 7)) | tr -d ' '").stdout.strip() == "42"


def test_run_timeout_kills_the_command(local):
    started = time.monotonic()
    with pytest.raises(MachineError, match="timed out"):
        local.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    assert time.monotonic() - started < 5


def test_stream_lines_and_exit_code(local):
    lines: list[str] = []
    code = local.stream([sys.executable, "-u", "-c",
                         "import sys\nfor i in range(3): print(f'line {i}')\nsys.stderr.write('err\\n')\nsys.exit(5)"],
                        lines.append, threading.Event())
    assert code == 5
    assert lines[:3] == ["line 0", "line 1", "line 2"] and "err" in lines
    missing: list[str] = []
    assert local.stream(["definitely-not-a-command-tcs"], missing.append, threading.Event()) == 127
    assert missing and "not found" in missing[0]


def test_stream_stop_kills_the_process_tree(local):
    marker = f"tcs-{uuid.uuid4().hex[:8]}"
    script = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', '{marker}'])\n"
        "while True:\n    print('tick', flush=True)\n    time.sleep(0.05)\n"
    )
    stop = threading.Event()
    lines: list[str] = []

    def on_line(line: str) -> None:
        lines.append(line)
        if len(lines) == 3:
            stop.set()

    started = time.monotonic()
    assert local.stream([sys.executable, "-c", script], on_line, stop) == -1
    assert time.monotonic() - started < 10
    time.sleep(0.3)
    survivors = [p for p in psutil.process_iter(["cmdline"]) if marker in " ".join(p.info["cmdline"] or [])]
    assert survivors == []


def test_is_alive_and_terminate(local):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert local.is_alive(proc.pid)
        local.terminate(proc.pid)
        proc.wait(timeout=10)
        assert not local.is_alive(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert not local.is_alive(2**22 + 12345)


# ============================================================================ runtime
def test_runtime_is_deployed_by_content_hash(local):
    digest, files = runtime_bundle()
    assert len(digest) == 12 and set(files) == set(RUNTIME_FILES)
    folder = Path(local.runtime_dir())
    assert folder == cache_dir() / "runtime" / digest == cache_dir() / "runtime" / runtime_digest()
    for name in RUNTIME_FILES:
        assert (folder / name).read_bytes() == runtime_path(name).read_bytes()
    stamp = (folder / RUNTIME_FILES[0]).stat().st_mtime_ns
    assert LocalMachine(local.profile).runtime_dir() == str(folder)
    assert deploy_local(cache_dir() / "runtime") == folder
    assert (folder / RUNTIME_FILES[0]).stat().st_mtime_ns == stamp  # not rewritten


@needs_torchcor
def test_runtime_probe_with_the_real_python():
    machine = LocalMachine(MachineProfile(name="local", python=sys.executable))
    report = machine.probe_environment()
    assert report.error is None, report.error
    assert report.python_version == ".".join(map(str, sys.version_info[:3]))
    assert report.torch_version and report.torchcor_version and report.ready
    assert machine.runtime("alive", str(os.getpid()))["alive"] is True


def test_deploy_remote_uploads_once(local, tmp_path):
    writes: list[str] = []

    class Recording(LocalMachine):
        def write_bytes(self, path, data):
            writes.append(path)
            super().write_bytes(path, data)

    machine = Recording(local.profile)
    first = deploy_remote(machine, str(tmp_path / "rt"))
    assert first == str(tmp_path / "rt" / runtime_digest())
    assert sorted(os.path.basename(w) for w in writes) == sorted(RUNTIME_FILES)
    writes.clear()
    assert deploy_remote(machine, str(tmp_path / "rt")) == first
    assert writes == []


# ============================================================================ python detection
def test_detect_python(local):
    pinned = LocalMachine(MachineProfile(name="x", python="/opt/py/bin/python"))
    assert pinned.detect_python() == "/opt/py/bin/python" and pinned.python == "/opt/py/bin/python"
    started = time.monotonic()
    found = local.detect_python()
    assert time.monotonic() - started < 1.0
    assert found == sys.executable  # this environment has TorchCor installed
    candidates = python_candidates()
    assert candidates[0] == sys.executable
    assert len({os.path.dirname(c) for c in candidates}) == len(candidates)


@needs_torchcor
def test_find_pythons_local_is_fast(local):
    started = time.monotonic()
    found = environment.find_pythons(local)
    assert time.monotonic() - started < 15
    ours = next(c for c in found if os.path.realpath(c.path) == os.path.realpath(sys.executable))
    assert ours.ready and ours.has_torch and ours.torchcor_version and ours.version.startswith("3.")
    assert found[0].ready  # ready interpreters are listed first
    assert "TorchCor" in ours.summary


def test_probe_python_reports_failures(local):
    broken = environment.probe_python(local, "/definitely/not/python")
    assert broken.error and not broken.version
    parsed = environment._parse_probe('noise\n{"path": "/p", "version": "3.12.1", "torch": "2.5.1+cu124",'
                                      ' "cuda": "12.4", "torchcor": null, "prefix": "/e/envs/tc"}', "/p")
    assert parsed.torch_version == "2.5.1" and parsed.cuda == "12.4" and not parsed.has_torchcor
    assert parsed.env_name == "tc" and not parsed.ready


# ============================================================================ install & venv
class RecordingMachine(LocalMachine):
    """Records commands instead of running them."""

    def __init__(self, *, git: bool = True, torch: bool = True, venv_ok: bool = True) -> None:
        super().__init__(MachineProfile(name="rec", kind=MachineKind.SSH, host="h"))
        self.git, self.torch, self.venv_ok = git, torch, venv_ok
        self.streams: list[tuple[list[str], dict | None]] = []
        self.runs: list[list[str]] = []

    def run(self, argv, *, timeout=60.0, cwd=None, env=None, stdin=None):
        argv = list(argv) if not isinstance(argv, str) else [argv]
        self.runs.append(argv)
        if argv[:2] == ["git", "--version"]:
            return CommandResult(0 if self.git else 127, "git version 2.43" if self.git else "", "")
        if "-c" in argv:
            torch = '"2.5.1"' if self.torch else "null"
            return CommandResult(0, f'{{"path": "{argv[0]}", "version": "3.11.9", "torch": {torch}}}', "")
        if argv[1:3] == ["-m", "venv"]:
            return CommandResult(0 if self.venv_ok else 1, "", "" if self.venv_ok else
                                 "The virtual environment was not created successfully because ensurepip is "
                                 "not available.")
        return CommandResult(0, "", "")

    def expand(self, path):
        return path.replace("~", "/home/u", 1)

    def join(self, *parts):  # a remote (POSIX) machine, whatever this computer is
        return posixpath.join(*parts)

    def stream(self, argv, on_line, stop, *, env=None):
        self.streams.append((list(argv), env))
        on_line("Successfully installed things")
        return 0


def test_install_torchcor_streams_pip():
    machine = RecordingMachine()
    lines: list[str] = []
    requirement = "torchcor @ git+https://github.com/sagebei/torchcor.git"
    code = environment.install_torchcor(machine, "/env/bin/python", requirement, lines.append, threading.Event())
    assert code == 0
    (argv, env), = machine.streams
    assert argv == ["/env/bin/python", "-m", "pip", "install", "--upgrade", requirement]
    assert env["PIP_DISABLE_PIP_VERSION_CHECK"] == "1" and env["PYTHONUNBUFFERED"] == "1"
    assert lines[0].startswith("$ /env/bin/python -m pip install --upgrade")


def test_install_without_git_uses_the_github_archive_and_torch_first():
    machine = RecordingMachine(git=False, torch=False)
    lines: list[str] = []
    code = environment.install_torchcor(machine, "/env/bin/python",
                                        "torchcor @ git+https://github.com/sagebei/torchcor.git@v2",
                                        lines.append, threading.Event(), with_torch=True)
    assert code == 0
    assert [a[-1] for a, _ in machine.streams] == [
        "torch", "torchcor @ https://github.com/sagebei/torchcor/archive/v2.zip"]
    assert any("git is not installed" in ln for ln in lines)


def test_archive_requirement():
    assert environment.archive_requirement("torchcor @ git+https://github.com/sagebei/torchcor.git") == \
        "torchcor @ https://github.com/sagebei/torchcor/archive/HEAD.zip"
    assert environment.archive_requirement("torchcor") is None
    assert environment.archive_requirement("/home/u/torchcor") is None


def test_create_venv():
    machine = RecordingMachine()
    assert environment.create_venv(machine, "python3", "~/.torchcor-studio/venv") == \
        "/home/u/.torchcor-studio/venv/bin/python"
    assert machine.runs[0] == ["python3", "-m", "venv", "/home/u/.torchcor-studio/venv"]
    assert machine.runs[1][-3:] == ["install", "--upgrade", "pip"]
    with pytest.raises(MachineError, match="python3-venv"):
        environment.create_venv(RecordingMachine(venv_ok=False), "python3", "~/v")
    assert environment.default_venv_path(machine) == "/home/u/.torchcor-studio/venv"


@needs_torchcor
def test_check_machine_local(local):
    report = environment.check_machine(local)
    assert report.ok and report.hostname and report.os
    assert report.environment is not None and report.environment.ready
    assert report.gpus or report.gpu_note


# ============================================================================ secrets
def test_secrets_roundtrip(memory_keyring):
    machine_id = uuid.uuid4().hex[:12]
    assert secrets.keyring_available()
    assert secrets.get_password(machine_id) is None
    assert secrets.set_secret(machine_id, secrets.PASSWORD, "hunter2") is True
    assert memory_keyring.data[(secrets.SERVICE, f"{machine_id}:password")] == "hunter2"
    assert secrets.get_password(machine_id) == "hunter2"
    secrets.set_secret(machine_id, secrets.PASSPHRASE, "pp")
    secrets.delete_all(machine_id)
    assert secrets.get_password(machine_id) is None and secrets.get_passphrase(machine_id) is None
    assert memory_keyring.data == {}
    with pytest.raises(ValueError):
        secrets.get_secret(machine_id, "token")


def test_secrets_degrade_without_a_keyring():
    from keyring.backends import fail

    keyring.set_keyring(fail.Keyring())
    machine_id = uuid.uuid4().hex[:12]
    assert not secrets.keyring_available()
    assert secrets.set_secret(machine_id, secrets.PASSWORD, "s") is False
    assert secrets.get_password(machine_id) == "s"  # still usable this session
    secrets.delete_secret(machine_id, secrets.PASSWORD)
    assert secrets.get_password(machine_id) is None


# ============================================================================ choosing a Python
def _cand(name: str, version: str = "3.11.6", torch: str | None = None, cuda: str | None = None,
          torchcor: str | None = None, prefix: str | None = None) -> environment.PythonCandidate:
    return environment.PythonCandidate(path=f"/envs/{name}/bin/python", version=version,
                                       prefix=prefix or f"/envs/{name}", has_torch=torch is not None,
                                       torch_version=torch, cuda=cuda, has_torchcor=torchcor is not None,
                                       torchcor_version=torchcor)


def test_ranking_puts_what_can_run_on_a_gpu_first():
    system = _cand("system", "3.8.10")  # e.g. Ubuntu 20.04's /usr/bin/python3: too old, no torch
    bare = _cand("bare")
    cpu_torch = _cand("cpu", torch="2.5.1")
    cuda_torch = _cand("tc", torch="2.7.1", cuda="12.8")  # one install away from GPU runs
    cpu_ready = _cand("cpu-ready", torch="2.5.1", torchcor="1.9.0")
    gpu_ready = _cand("gpu-ready", torch="2.7.1", cuda="12.8", torchcor="1.9.0")
    ranked = environment.rank_pythons([system, bare, cpu_torch, cuda_torch, cpu_ready, gpu_ready])
    assert [c.env_name for c in ranked] == ["gpu-ready", "cpu-ready", "tc", "cpu", "bare", "system"]
    assert [c.readiness for c in ranked] == [0, 1, 2, 3, 4, 5]
    assert not system.supported and bare.supported
    project = environment.PythonCandidate(path="/h/proj/.venv/bin/python", prefix="/h/proj/.venv")
    conda = environment.PythonCandidate(path="/h/miniconda3/envs/venv/bin/python", prefix="/h/miniconda3/envs/venv")
    assert (project.env_name, conda.env_name) == ("proj/.venv", "venv")


def test_a_torchcor_older_than_the_scripts_need_is_not_ready():
    from torchcor_studio.core.models import EnvironmentReport, torchcor_supported

    old = _cand("venv", torch="2.7.1", cuda="12.6", torchcor="0.1.6")  # lacks torchcor.electrophysiology
    assert not old.ready and old.readiness == 2 and "TorchCor 0.1.6 (too old)" in old.summary
    assert torchcor_supported("1.9.0") and torchcor_supported("2.0") and torchcor_supported("unknown")
    assert not torchcor_supported("1.0.6") and not torchcor_supported(None)
    env = EnvironmentReport(python="/p", python_version="3.11.10", torch_version="2.7.1", torchcor_version="0.1.6")
    assert not env.ready and env.torchcor_outdated


def test_rank_pythons_keeps_one_interpreter_per_environment():
    py2 = _cand("usr", "2.7.18", prefix="/usr")
    py3 = _cand("usr3", "3.8.10", prefix="/usr")  # same prefix, different Python: both are real
    py38 = _cand("usr38", "3.8.10", prefix="/usr")  # python3 and python3.8: the same one
    broken = environment.PythonCandidate(path="/x/python", error="exec format error")
    assert [c.path for c in environment.rank_pythons([py2, py3, py38, broken])] == [py3.path]


def test_better_python_is_suggested_only_when_it_is_better(monkeypatch):
    from torchcor_studio.core.models import EnvironmentReport

    found = [_cand("tc", torch="2.7.1", cuda="12.8")]
    monkeypatch.setattr(environment, "find_pythons", lambda machine: found)
    old = EnvironmentReport(python="/usr/bin/python3", python_version="3.8.10", nvidia_gpus=2)
    assert environment.better_python(None, old) is found[0]
    same = EnvironmentReport(python="/envs/tc/bin/python", python_version="3.11.6", torch_version="2.7.1",
                             cuda_version="12.8", cuda_available=True, nvidia_gpus=2)
    assert environment.better_python(None, same) is None
    # a ready CPU environment on a machine without NVIDIA GPUs is as good as it gets
    mac = EnvironmentReport(python="/v/bin/python", python_version="3.12.4", torch_version="2.5.1",
                            torchcor_version="1.9.0")
    monkeypatch.setattr(environment, "find_pythons", lambda machine: pytest.fail("no scan needed"))
    assert environment.better_python(None, mac) is None


def test_runtime_failures_name_the_python_that_lacks_a_package():
    from torchcor_studio.core.machines.base import runtime_failure

    trace = ('Traceback (most recent call last):\n  File "tcs_runtime.py", line 775, in cmd_preview\n'
             "    import numpy as np\nModuleNotFoundError: No module named 'numpy'\n")
    message = runtime_failure("RTX 3090", "preview", "/usr/bin/python3", trace)
    assert message.startswith("NumPy is not installed in the Python used on RTX 3090 (/usr/bin/python3).")
    assert "Change Python" in message and "Traceback" not in message
    assert "'pyvista'" in runtime_failure("m", "x", "py", "ImportError: No module named pyvista.core")
    assert runtime_failure("m", "scan", "py", "boom\nValueError: bad") == "runtime 'scan' failed: boom | ValueError: bad"
