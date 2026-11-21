"""The :class:`Machine` abstraction: one interface for this computer and SSH hosts.

Every operation the studio performs on a compute machine goes through this interface:
running short commands, streaming long-lived ones (GPU telemetry), file access, and
launching detached simulation supervisors. Implementations must be thread-safe: the UI
calls them from a worker pool, never from the Qt main thread.

Paths are *strings in the target machine's convention* (POSIX for SSH hosts, native for
the local machine). Use :meth:`Machine.join` / :meth:`Machine.parent` rather than
``os.path`` so code works for both.
"""

from __future__ import annotations

import json
import re
import shlex
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from torchcor_studio.core.models import (
    CommandResult,
    EnvironmentReport,
    FileEntry,
    MachineProfile,
)

_MISSING_MODULE = re.compile(r"No module named '?([A-Za-z_][\w.]*)'?")
_PACKAGES = {"numpy": "NumPy", "torch": "PyTorch", "torchcor": "TorchCor", "scipy": "SciPy"}


def runtime_failure(machine: str, command: str, python: str, output: str) -> str:
    """Explain a failed runtime command; a missing package names the Python that lacks it."""
    lines = [ln.rstrip() for ln in output.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        m = _MISSING_MODULE.search(line)
        if m:
            top = m.group(1).split(".")[0]
            package = _PACKAGES.get(top, f"The Python module '{top}'")
            return (f"{package} is not installed in the Python used on {machine} ({python}). "
                    "Choose another one with Change Python… in the Environment card.")
    return f"runtime '{command}' failed: " + " | ".join(lines[-5:])


class MachineError(RuntimeError):
    """A machine operation failed (connection lost, command failed, file missing, ...)."""


class Machine(ABC):
    """A compute machine the studio can run simulations on."""

    def __init__(self, profile: MachineProfile) -> None:
        self.profile = profile
        self._lock = threading.RLock()
        self._runtime_dir: str | None = None
        self._python: str | None = profile.python or None
        #: Seconds to add to this machine's clock readings for this computer's clock. Run times
        #: (written there by the runtime) and file times are converted with it, so that times
        #: compared with this computer's (elapsed, "5 min ago") are right even when the
        #: machine's clock is off. Measured by :meth:`probe_environment`; 0 locally.
        self.clock_offset = 0.0

    # ------------------------------------------------------------------ lifecycle
    @property
    def id(self) -> str:
        return self.profile.id

    @property
    def name(self) -> str:
        return self.profile.name

    @abstractmethod
    def connect(self) -> None:
        """Open the connection (no-op locally). Raises :class:`MachineError`."""

    @abstractmethod
    def close(self) -> None:
        """Release every resource; safe to call twice."""

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    # ------------------------------------------------------------------ commands
    @abstractmethod
    def run(
        self,
        argv: Sequence[str] | str,
        *,
        timeout: float | None = 60.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
    ) -> CommandResult:
        """Run a command to completion and capture its output.

        ``argv`` is a list (preferred, quoted for you) or a shell string.
        """

    @abstractmethod
    def stream(
        self,
        argv: Sequence[str] | str,
        on_line: Callable[[str], None],
        stop: threading.Event,
        *,
        env: dict[str, str] | None = None,
    ) -> int:
        """Run a long-lived command, calling ``on_line`` for each stdout line.

        Returns the exit code once the command ends or ``stop`` is set (in which case
        the command is terminated and ``-1`` is returned).
        """

    # ------------------------------------------------------------------ filesystem
    @abstractmethod
    def home(self) -> str: ...

    @abstractmethod
    def join(self, *parts: str) -> str: ...

    @abstractmethod
    def parent(self, path: str) -> str: ...

    @abstractmethod
    def basename(self, path: str) -> str: ...

    @abstractmethod
    def expand(self, path: str) -> str:
        """Expand ``~`` and make the path absolute on the target machine."""

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def is_dir(self, path: str) -> bool: ...

    @abstractmethod
    def listdir(self, path: str) -> list[FileEntry]:
        """Directory listing, directories first then files, each sorted by name."""

    @abstractmethod
    def mkdir(self, path: str, parents: bool = True) -> None: ...

    @abstractmethod
    def read_bytes(self, path: str, offset: int = 0, length: int | None = None) -> bytes: ...

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None:
        """Write atomically where the platform allows (temp file + rename)."""

    @abstractmethod
    def size(self, path: str) -> int: ...

    @abstractmethod
    def remove(self, path: str, recursive: bool = False) -> None: ...

    @abstractmethod
    def download(
        self,
        remote_path: str,
        local_path: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Copy a file from the machine to this computer (copy locally)."""

    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> None: ...

    # ------------------------------------------------------------------ conveniences
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(path).decode(encoding, errors="replace")

    def write_text(self, path: str, text: str, encoding: str = "utf-8") -> None:
        self.write_bytes(path, text.encode(encoding))

    def read_json(self, path: str) -> Any:
        return json.loads(self.read_text(path))

    def write_json(self, path: str, data: Any) -> None:
        self.write_text(path, json.dumps(data, indent=2, sort_keys=False, default=str))

    @staticmethod
    def quote(argv: Sequence[str] | str) -> str:
        if isinstance(argv, str):
            return argv
        return " ".join(shlex.quote(a) for a in argv)

    # ------------------------------------------------------------------ python & runtime
    @property
    def python(self) -> str:
        """The Python interpreter used to run simulations on this machine."""
        if self._python is None:
            self._python = self.detect_python()
        return self._python

    @python.setter
    def python(self, value: str) -> None:
        self._python = value or None

    def detect_python(self) -> str:
        """``profile.python``; else the interpreter closest to running TorchCor here, as
        ranked by :func:`~torchcor_studio.core.environment.find_pythons`."""
        if self.profile.python:
            return self.profile.python
        from torchcor_studio.core.environment import best_python  # it builds on the machines

        best = best_python(self)
        return best.path if best is not None else self.fallback_python

    #: Interpreter to try when no Python was found at all.
    fallback_python = "python3"

    @abstractmethod
    def runtime_dir(self) -> str:
        """Folder on the machine holding the studio runtime scripts (deployed on demand)."""

    def runtime(
        self,
        command: str,
        *args: str,
        timeout: float | None = 120.0,
        python: str | None = None,
    ) -> Any:
        """Run ``tcs_runtime.py <command> <args>`` on the machine and decode its JSON."""
        script = self.join(self.runtime_dir(), "tcs_runtime.py")
        result = self.run([python or self.python, script, command, *args], timeout=timeout)
        if not result.ok:
            raise MachineError(runtime_failure(self.name, command, python or self.python,
                                               result.stderr or result.stdout))
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise MachineError(f"runtime '{command}' returned no JSON: {result.stdout[-400:]}") from exc

    def probe_environment(self, python: str | None = None) -> EnvironmentReport:
        """Interrogate the Python environment (versions, CUDA, TorchCor), and the clock."""
        try:
            report = self.runtime("probe", python=python, timeout=180)
        except MachineError as exc:
            return EnvironmentReport(python=python or self._python or "", error=str(exc))
        if report.get("clock") and not self.profile.is_local:
            # read as the reply arrives: off by the network latency only (well under a second)
            self.clock_offset = time.time() - float(report["clock"])
        return EnvironmentReport.from_dict(report)

    # ------------------------------------------------------------------ processes
    @abstractmethod
    def is_alive(self, pid: int) -> bool: ...

    @abstractmethod
    def terminate(self, pid: int) -> None:
        """Terminate a process and its children."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.profile.name!r} {self.profile.address}>"
