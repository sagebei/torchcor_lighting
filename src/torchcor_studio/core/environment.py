"""Python environments on a machine: discover them, install TorchCor, create a venv.

* :func:`find_pythons` lists the interpreters of a machine with their torch / CUDA /
  TorchCor versions, best first (:attr:`PythonCandidate.readiness`). Probes only *locate*
  packages (no ``import torch``), so the whole scan takes well under a second; remote
  hosts are scanned with a single command. :func:`best_python` is auto-detection.
* :func:`install_torchcor` streams ``pip install --upgrade <requirement>``.
* :func:`create_venv` makes a fresh virtual environment for the studio.
* :func:`check_machine` is the "Test connection" summary: host, OS, GPUs, Python.
"""

from __future__ import annotations

import contextlib
import json
import platform
import re
import shlex
import shutil
import socket
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from torchcor_studio.core.machines.base import Machine, MachineError
from torchcor_studio.core.machines.local import IS_WINDOWS, python_candidates, studio_venv_path
from torchcor_studio.core.machines.ssh import REMOTE_PYTHON_GLOBS, STUDIO_VENV, remote_globs
from torchcor_studio.core.models import (
    MIN_PYTHON,
    EnvironmentReport,
    torchcor_supported,
    version_tuple,
)

#: Prints one JSON line describing the interpreter running it (Python 3.6+ syntax;
#: ``sys.argv[1]`` is the path it was invoked as). Nothing heavy is imported: packages
#: are found through their metadata, and the working folder is ignored so a ``torchcor``
#: checkout in it cannot shadow the installed package.
PROBE_SNIPPET = r"""
import json, os, re, sys
sys.path = [p for p in sys.path if p not in ("", ".", os.getcwd())]
out = {"path": sys.argv[1] if len(sys.argv) > 1 else sys.executable,
       "version": "%d.%d.%d" % tuple(sys.version_info[:3]), "prefix": sys.prefix}
try:
    import importlib.util as u
    try:
        from importlib import metadata as m
    except ImportError:
        m = None
    def dist(name):
        try:
            return m.distribution(name) if m else None
        except Exception:
            return None
    def real(name):
        s = u.find_spec(name)
        return s if s is not None and s.origin not in (None, "namespace") else None
    t, td = real("torch"), dist("torch")
    out["torch"] = td.version if td else ("unknown" if t else None)
    version_py = None
    if td is not None:
        version_py = str(td.locate_file("torch/version.py"))
    elif t is not None and t.submodule_search_locations:
        version_py = os.path.join(list(t.submodule_search_locations)[0], "version.py")
    cuda = None
    if version_py:
        try:
            with open(version_py) as fh:
                mm = re.search(r"^cuda\s*(?::[^=]*)?=\s*['\"]([^'\"]+)['\"]", fh.read(), re.M)
            cuda = mm.group(1) if mm else None
        except OSError:
            pass
    out["cuda"] = cuda
    tc = dist("torchcor")
    out["torchcor"] = tc.version if tc else ("unknown" if real("torchcor") else None)
except Exception as e:
    out["error"] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(out))
""".strip()

PIP_ENV = {
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_INPUT": "1",
    "PIP_PROGRESS_BAR": "off",
    "PYTHONUNBUFFERED": "1",
}

_GITHUB_GIT = re.compile(
    r"^(?P<name>[\w.-]+)\s*@\s*git\+https://github\.com/(?P<repo>[\w.-]+/[\w.-]+?)(?:\.git)?(?:@(?P<ref>[^#\s]+))?$")


#: Virtual-environment folder names that say nothing about the project they belong to.
_GENERIC_VENV_NAMES = ("venv", ".venv", "env", ".env", "virtualenv")


@dataclass(frozen=True)
class PythonCandidate:
    """One interpreter on a machine and what it has installed."""

    path: str
    version: str = ""
    prefix: str = ""
    has_torch: bool = False
    torch_version: str | None = None
    #: CUDA version torch was built with (``None``: CPU-only build or no torch).
    cuda: str | None = None
    has_torchcor: bool = False
    torchcor_version: str | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        """Can run the studio's simulations (torch and a recent enough TorchCor)."""
        return self.has_torch and self.has_torchcor and torchcor_supported(self.torchcor_version)

    @property
    def supported(self) -> bool:
        """New enough for TorchCor."""
        return version_tuple(self.version) >= MIN_PYTHON

    @property
    def readiness(self) -> int:
        """What is missing before this Python runs TorchCor on a GPU (lower is better).

        0 TorchCor + CUDA PyTorch · 1 TorchCor + CPU-only PyTorch · 2 CUDA PyTorch, no
        TorchCor yet · 3 PyTorch, no TorchCor · 4 a Python TorchCor supports · 5 neither.
        A TorchCor older than :data:`~torchcor_studio.core.models.MIN_TORCHCOR` counts as missing.
        """
        cuda = self.has_torch and bool(self.cuda)
        if self.ready:
            return 0 if cuda else 1
        if not self.supported:
            return 5
        if cuda:
            return 2
        return 3 if self.has_torch else 4

    @property
    def env_name(self) -> str:
        """Short environment label: conda env name, venv folder (with its project when the
        folder has a generic name such as ``venv``), or ``system``."""
        prefix = self.prefix.rstrip("/\\")
        parts = re.split(r"[/\\]", prefix)
        if len(parts) >= 2 and parts[-2] == "envs":
            return parts[-1]
        name = parts[-1] if parts else ""
        if name.lower() in _GENERIC_VENV_NAMES and len(parts) >= 2 and parts[-2]:
            return f"{parts[-2]}/{name}"
        if name in ("usr", "local", "homebrew", "") or re.fullmatch(r"[\d.]+", name) \
                or ".framework" in prefix or name.lower().startswith("python"):
            return "system"
        return name

    @property
    def summary(self) -> str:
        """``"Python 3.11.9 · PyTorch 2.5.1 (CUDA 12.4) · TorchCor 1.9.0"``."""
        bits = [f"Python {self.version}" if self.version else "Python"]
        if self.has_torch:
            bits.append(f"PyTorch {self.torch_version}" + (f" (CUDA {self.cuda})" if self.cuda else " (CPU)"))
        else:
            bits.append("no PyTorch")
        if not self.has_torchcor:
            bits.append("no TorchCor")
        elif torchcor_supported(self.torchcor_version):
            bits.append(f"TorchCor {self.torchcor_version}")
        else:
            bits.append(f"TorchCor {self.torchcor_version} (too old)")
        return " · ".join(bits)

    @classmethod
    def from_report(cls, env: EnvironmentReport) -> PythonCandidate:
        """A probed environment in the same terms, to compare it with the others."""
        return cls(path=env.python, version=env.python_version, has_torch=env.torch_version is not None,
                   torch_version=env.torch_version, cuda=env.cuda_version,
                   has_torchcor=env.torchcor_version is not None, torchcor_version=env.torchcor_version,
                   error=env.error)

    @classmethod
    def from_probe(cls, data: dict) -> PythonCandidate:
        torch = data.get("torch")
        torchcor = data.get("torchcor")
        return cls(
            path=str(data.get("path", "")),
            version=str(data.get("version", "")),
            prefix=str(data.get("prefix", "")),
            has_torch=torch is not None,
            torch_version=str(torch).split("+")[0] if torch else None,
            cuda=data.get("cuda"),
            has_torchcor=torchcor is not None,
            torchcor_version=str(torchcor) if torchcor else None,
            error=data.get("error"),
        )


def _parse_probe(text: str, path: str) -> PythonCandidate:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return PythonCandidate.from_probe(json.loads(line))
            except ValueError:
                break
    return PythonCandidate(path=path, error=(text.strip().splitlines() or ["no output"])[-1][:300])


def probe_python(machine: Machine, python: str, timeout: float = 30.0) -> PythonCandidate:
    """Describe one interpreter (fast: packages are located, not imported)."""
    try:
        result = machine.run([python, "-c", PROBE_SNIPPET, python], timeout=timeout)
    except MachineError as exc:
        return PythonCandidate(path=python, error=str(exc))
    if not result.ok:
        detail = (result.stderr or result.stdout).strip().splitlines()
        return PythonCandidate(path=python, error=detail[-1][:300] if detail else f"exit {result.returncode}")
    return _parse_probe(result.stdout, python)


def _remote_scan_script() -> str:
    """One shell line probing every remote candidate in parallel (one JSON line each).

    Candidates are :data:`REMOTE_PYTHON_GLOBS`, every environment conda recorded in
    ``~/.conda/environments.txt`` (which knows envs in custom places too) and the
    ``python3`` / ``python`` on PATH; each path is probed once.
    """
    snippet = shlex.quote(PROBE_SNIPPET)
    return (
        f'{{ for p in {remote_globs()}; do echo "$p"; done; '
        "sed -e 's|/*$|/bin/python|' \"$HOME/.conda/environments.txt\"; "
        "command -v python3; command -v python; } 2>/dev/null | awk '!seen[$0]++' | "
        "{ while IFS= read -r p; do "
        f'[ -f "$p" ] && [ -x "$p" ] && {{ "$p" -c {snippet} "$p" </dev/null 2>/dev/null & }}; '
        "done; wait; }; exit 0"
    )


def parse_scan(text: str) -> list[PythonCandidate]:
    """The candidates a remote scan printed, in :data:`REMOTE_PYTHON_GLOBS` order."""
    found: list[PythonCandidate] = []
    for line in text.splitlines():
        if line.strip().startswith("{"):
            try:
                found.append(PythonCandidate.from_probe(json.loads(line)))
            except ValueError:
                continue
    # probes finish in any order: restore the preference order of the candidates
    found.sort(key=lambda c: (_remote_position(c.path), c.path))
    return found


def rank_pythons(found: Iterable[PythonCandidate]) -> list[PythonCandidate]:
    """One Python 3 per environment, best first; ties keep the discovery order.

    Interpreters that did not run are dropped, and so is Python 2. Environments are told
    apart by ``sys.prefix`` *and* version: ``/usr/bin/python3`` and ``/usr/bin/python``
    can share a prefix and still be different Pythons.
    """
    unique: dict[tuple[str, str], PythonCandidate] = {}
    for cand in found:
        if cand.version and not cand.version.startswith("2."):
            unique.setdefault((cand.prefix or cand.path, cand.version), cand)
    return sorted(unique.values(), key=lambda c: (c.readiness, c.error is not None))


def find_pythons(machine: Machine, timeout: float = 60.0) -> list[PythonCandidate]:
    """Every Python 3 on ``machine``, best first (see :func:`rank_pythons`).

    Local interpreters are probed in parallel threads; remote ones with a single shell
    command that probes them all concurrently.
    """
    if machine.profile.is_local:
        paths = python_candidates()
        if machine.profile.python and machine.profile.python not in paths:
            paths.insert(0, machine.profile.python)
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(paths)))) as pool:
            found = list(pool.map(lambda p: probe_python(machine, p, timeout), paths))
    else:
        found = parse_scan(machine.run(_remote_scan_script(), timeout=timeout).stdout)
        if machine.profile.python and not any(c.path == machine.profile.python for c in found):
            found.insert(0, probe_python(machine, machine.profile.python, timeout))
    return rank_pythons(found)


def best_python(machine: Machine) -> PythonCandidate | None:
    """Auto-detection: the interpreter closest to running TorchCor on ``machine``."""
    found = find_pythons(machine)
    return found[0] if found else None


def better_python(machine: Machine, env: EnvironmentReport) -> PythonCandidate | None:
    """An interpreter on ``machine`` closer to running TorchCor than the probed one."""
    current = PythonCandidate.from_report(env).readiness
    if current == 0 or (current == 1 and not env.nvidia_gpus):
        return None  # as good as this machine allows
    best = best_python(machine)
    return best if best is not None and best.readiness < current else None


def _glob_regex(pattern: str) -> re.Pattern[str]:
    lead = ".*" if pattern.startswith("~") else ""
    body = pattern[1:] if lead else pattern
    return re.compile(lead + "[^/]*".join(re.escape(part) for part in body.split("*")))


_REMOTE_ORDER = [_glob_regex(g) for g in REMOTE_PYTHON_GLOBS]


def _remote_position(path: str) -> int:
    """Rank of a remote interpreter in :data:`REMOTE_PYTHON_GLOBS` (PATH ones last)."""
    return next((i for i, rx in enumerate(_REMOTE_ORDER) if rx.fullmatch(path)), len(_REMOTE_ORDER))


# ============================================================================ install
def archive_requirement(requirement: str) -> str | None:
    """``name @ git+https://github.com/o/r.git@ref`` → the GitHub source-archive URL.

    Lets pip install from GitHub on machines without ``git``. ``None`` if the
    requirement is not a GitHub git URL.
    """
    m = _GITHUB_GIT.match(requirement.strip())
    if not m:
        return None
    ref = m.group("ref") or "HEAD"
    return f"{m.group('name')} @ https://github.com/{m.group('repo')}/archive/{ref}.zip"


def install_torchcor(
    machine: Machine,
    python: str,
    requirement: str,
    on_line: Callable[[str], None],
    stop: threading.Event,
    with_torch: bool = False,
) -> int:
    """Stream ``python -m pip install --upgrade <requirement>`` and return its exit code.

    ``requirement`` usually comes from :meth:`Settings.torchcor_requirement`. With
    ``with_torch``, PyTorch is installed first when the interpreter lacks it. A GitHub
    ``git+https`` requirement falls back to the source archive when ``git`` is missing.
    Returns ``-1`` if ``stop`` was set.
    """
    if archive_requirement(requirement) is not None:
        try:
            has_git = machine.run(["git", "--version"], timeout=30).ok
        except MachineError:
            has_git = False
        if not has_git:
            requirement = archive_requirement(requirement) or requirement
            on_line("git is not installed here; downloading the source archive instead.")
    if with_torch and not probe_python(machine, python).has_torch:
        code = _pip(machine, python, ["torch"], on_line, stop)
        if code != 0:
            return code
    return _pip(machine, python, [requirement], on_line, stop)


def _pip(machine: Machine, python: str, packages: list[str], on_line: Callable[[str], None],
         stop: threading.Event) -> int:
    argv = [python, "-m", "pip", "install", "--upgrade", *packages]
    on_line("$ " + " ".join(shlex.quote(a) for a in argv))
    return machine.stream(argv, on_line, stop, env=PIP_ENV)


def default_venv_path(machine: Machine) -> str:
    """Where the studio puts its own virtual environment on ``machine``."""
    return studio_venv_path() if machine.profile.is_local else machine.expand(STUDIO_VENV)


def venv_python(machine: Machine, path: str) -> str:
    """The interpreter inside a virtual environment folder."""
    if machine.profile.is_local and IS_WINDOWS:
        return machine.join(path, "Scripts", "python.exe")
    return machine.join(path, "bin", "python")


def create_venv(machine: Machine, base_python: str, path: str) -> str:
    """Create a virtual environment at ``path`` and return its interpreter.

    pip inside the new environment is upgraded (old pips cannot install current
    PyTorch wheels); that step is skipped quietly when offline.
    """
    target = machine.expand(path)
    result = machine.run([base_python, "-m", "venv", target], timeout=600)
    if not result.ok:
        detail = (result.stderr or result.stdout).strip()
        if "ensurepip" in detail:
            raise MachineError(
                f"Python's venv module is missing on {machine.name}. On Debian/Ubuntu, install "
                "it with: sudo apt install python3-venv")
        last = detail.splitlines()[-1].strip() if detail else f"exit code {result.returncode}"
        raise MachineError(f"Could not create a virtual environment: {last}")
    python = venv_python(machine, target)
    with contextlib.suppress(MachineError):  # offline: keep the bundled pip
        machine.run([python, "-m", "pip", "install", "--upgrade", "pip"], timeout=300, env=PIP_ENV)
    return python


# ============================================================================ machine check
@dataclass
class MachineReport:
    """Result of :func:`check_machine` ("Test connection")."""

    hostname: str = ""
    os: str = ""
    gpus: list[str] = field(default_factory=list)
    #: Why no GPU list is available (e.g. "nvidia-smi not found"), if so.
    gpu_note: str | None = None
    environment: EnvironmentReport | None = None
    #: SSH host key fingerprint, and whether it was trusted for the first time.
    host_key: str = ""
    new_host_key: bool = False
    latency_ms: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _system_info(machine: Machine) -> tuple[str, str]:
    if machine.profile.is_local:
        if platform.system() == "Darwin":
            release = platform.mac_ver()[0]
            os_name = f"macOS {release}" if release else "macOS"
        elif platform.system() == "Windows":
            os_name = f"Windows {platform.release()}"
        else:
            os_name = f"{platform.system()} {platform.release()}"
        return socket.gethostname(), os_name
    script = ("hostname 2>/dev/null || uname -n; "
              "( . /etc/os-release 2>/dev/null && echo \"$PRETTY_NAME\" ) || uname -sr")
    lines = [ln.strip() for ln in machine.run(script, timeout=30).stdout.splitlines() if ln.strip()]
    return (lines[0] if lines else ""), (lines[1] if len(lines) > 1 else "")


def _gpu_list(machine: Machine) -> tuple[list[str], str | None]:
    if machine.profile.is_local and not shutil.which("nvidia-smi"):
        return [], "nvidia-smi not found"
    result = machine.run(["nvidia-smi", "-L"], timeout=30)
    if result.returncode == 127:
        return [], "nvidia-smi not found"
    if not result.ok:
        detail = (result.stdout or result.stderr).strip().splitlines()
        return [], detail[0] if detail else f"nvidia-smi failed ({result.returncode})"
    gpus = [re.sub(r"\s*\(UUID:.*\)$", "", ln.strip()) for ln in result.stdout.splitlines()
            if ln.strip().startswith("GPU ")]
    return gpus, None if gpus else "no NVIDIA GPU found"


def check_machine(machine: Machine, python: str | None = None) -> MachineReport:
    """Connect and summarise a machine: host, OS, NVIDIA GPUs, Python and TorchCor."""
    report = MachineReport()
    try:
        machine.connect()
        started = time.perf_counter()
        report.hostname, report.os = _system_info(machine)
        if not machine.profile.is_local:
            report.latency_ms = (time.perf_counter() - started) * 1000.0
        report.gpus, report.gpu_note = _gpu_list(machine)
        report.environment = machine.probe_environment(python or machine.python)
    except MachineError as exc:
        report.error = str(exc)
    fingerprint = getattr(machine, "host_key_fingerprint", "")
    report.host_key = fingerprint
    report.new_host_key = bool(fingerprint) and any(
        fingerprint in k for k in getattr(machine, "new_host_keys", []))
    return report
