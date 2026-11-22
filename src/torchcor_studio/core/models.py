"""Plain data types shared by the core and the UI.

They are deliberately small, immutable where possible, and serialisable to JSON so
they can cross thread boundaries and be cached on disk.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


# --------------------------------------------------------------------------- machines
class MachineKind(StrEnum):
    LOCAL = "local"
    SSH = "ssh"


class AuthMethod(StrEnum):
    AGENT = "agent"  # ssh-agent / default keys (~/.ssh/id_*)
    KEY = "key"  # explicit private key file (passphrase in the OS keyring)
    PASSWORD = "password"  # password stored in the OS keyring, never on disk


@dataclass
class MachineProfile:
    """A place simulations can run: this computer or an SSH host."""

    name: str
    kind: MachineKind = MachineKind.LOCAL
    host: str = ""
    port: int = 22
    username: str = ""
    auth: AuthMethod = AuthMethod.AGENT
    key_path: str = ""
    #: Optional ProxyJump-style bastion "user@host:port" (one hop).
    jump_host: str = ""
    #: Python interpreter with TorchCor installed on that machine ("" = auto-detect).
    python: str = ""
    #: Default folders on that machine.
    input_dir: str = ""
    output_dir: str = ""
    #: Its demo data folder (``.../torchcor_data``, with ``input/`` and ``output/``); "" = ``~/torchcor_data``.
    data_dir: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def is_local(self) -> bool:
        return self.kind == MachineKind.LOCAL

    @property
    def address(self) -> str:
        if self.is_local:
            return "localhost"
        user = f"{self.username}@" if self.username else ""
        port = f":{self.port}" if self.port and self.port != 22 else ""
        return f"{user}{self.host}{port}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MachineProfile:
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        if "kind" in clean:
            clean["kind"] = MachineKind(clean["kind"])
        if "auth" in clean:
            clean["auth"] = AuthMethod(clean["auth"])
        return cls(**clean)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class FileEntry:
    name: str
    path: str
    is_dir: bool
    size: int = 0
    mtime: float = 0.0


# --------------------------------------------------------------------------- environment
#: TorchCor's ``requires-python``.
MIN_PYTHON = (3, 10)
#: The TorchCor API the generated scripts are written for (and were validated with).
MIN_TORCHCOR = (1, 9)


def version_tuple(version: str) -> tuple[int, ...]:
    """``"3.11.6"`` → ``(3, 11, 6)`` (``()`` if unknown)."""
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def torchcor_supported(version: str | None) -> bool:
    """Can the generated scripts use this TorchCor? (An unknown version gets the benefit
    of the doubt: it was found without package metadata, e.g. a checkout on the path.)"""
    if version is None:
        return False
    found = version_tuple(version)
    return not found or found >= MIN_TORCHCOR


@dataclass
class EnvironmentReport:
    """What a machine's Python environment can do (from the runtime ``probe`` command)."""

    python: str = ""
    python_version: str = ""
    platform: str = ""
    hostname: str = ""
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    cuda_device_count: int = 0
    #: NVIDIA GPUs the driver lists (``nvidia-smi -L``), whether or not PyTorch can use them.
    nvidia_gpus: int = 0
    mps_available: bool = False
    torchcor_version: str | None = None
    torchcor_path: str | None = None
    numpy_version: str | None = None
    cpu_count: int = 0
    memory_total_gb: float = 0.0
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.error is None and self.torch_version is not None and torchcor_supported(self.torchcor_version)

    @property
    def torchcor_outdated(self) -> bool:
        """TorchCor is installed, but older than the API the studio's scripts use."""
        return self.torchcor_version is not None and not torchcor_supported(self.torchcor_version)

    @property
    def python_supported(self) -> bool:
        """New enough for TorchCor."""
        return version_tuple(self.python_version) >= MIN_PYTHON

    @property
    def unusable_gpus(self) -> int:
        """NVIDIA GPUs on the machine that this Python's PyTorch cannot use."""
        return 0 if self.cuda_available else self.nvidia_gpus

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentReport:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------- devices
class DeviceKind(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"
    APPLE = "apple"  # Apple-silicon GPU: monitored, but TorchCor cannot run on it


@dataclass(frozen=True)
class GPUSample:
    """One telemetry reading of one compute device."""

    index: int
    name: str
    kind: DeviceKind = DeviceKind.CUDA
    uuid: str = ""
    utilization: float = 0.0  # %
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    processes: int | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def memory_fraction(self) -> float:
        return self.memory_used_mb / self.memory_total_mb if self.memory_total_mb else 0.0

    @property
    def device_string(self) -> str:
        """The torch device string a run should use for this device."""
        if self.kind == DeviceKind.CUDA:
            return f"cuda:{self.index}"
        return "cpu"

    @property
    def runnable(self) -> bool:
        return self.kind in (DeviceKind.CUDA, DeviceKind.CPU)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.index}"


# --------------------------------------------------------------------------- projects
@dataclass
class Project:
    """A folder of runs in the sidebar. Membership lives in each run's manifest
    (``"project": {"id", "name"}``), so projects come back when runs are scanned."""

    id: str
    name: str
    created_at: float = field(default_factory=time.time)

    @classmethod
    def new(cls, name: str) -> Project:
        return cls(id=uuid.uuid4().hex[:12], name=name.strip())

    def ref(self) -> dict[str, str]:
        """What a run's manifest records about its project."""
        return {"id": self.id, "name": self.name}


# --------------------------------------------------------------------------- inputs
@dataclass
class VtxInfo:
    name: str
    count: int


@dataclass
class InputInfo:
    """Summary of an input folder (from the runtime ``inspect`` command)."""

    path: str
    exists: bool = False
    files: list[str] = field(default_factory=list)
    mesh_name: str | None = None
    n_nodes: int | None = None
    n_elems: int | None = None
    elem_types: list[str] = field(default_factory=list)
    regions: list[int] = field(default_factory=list)
    fibre_columns: int | None = None
    vtx: list[VtxInfo] = field(default_factory=list)
    surfaces: list[str] = field(default_factory=list)
    subdirs: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def has_mesh(self) -> bool:
        return self.mesh_name is not None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputInfo:
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        clean["vtx"] = [VtxInfo(**v) for v in clean.get("vtx", [])]
        return cls(**clean)


# --------------------------------------------------------------------------- runs
class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"  # supervisor stopped reporting (machine rebooted, process killed, ...)

    @property
    def is_active(self) -> bool:
        return self in (RunState.QUEUED, RunState.RUNNING)

    @property
    def is_terminal(self) -> bool:
        return not self.is_active


@dataclass
class RunStatus:
    """Mirror of ``status.json`` written by the runtime supervisor."""

    state: RunState = RunState.QUEUED
    pid: int | None = None
    child_pid: int | None = None
    host: str = ""
    started_at: float | None = None
    updated_at: float | None = None
    ended_at: float | None = None
    exit_code: int | None = None
    progress: float | None = None  # 0..1 of the main solve
    stage: str = ""
    stage_label: str = ""
    message: str = ""
    sim_time: float | None = None
    sim_total: float | None = None
    eta_s: float | None = None
    error: str | None = None

    @property
    def elapsed_s(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.ended_at or (time.time() if self.state.is_active else self.updated_at)
        return max(0.0, (end or self.started_at) - self.started_at)

    @property
    def remaining_s(self) -> float | None:
        """The runtime's estimate of the time left (``eta_s``), counted down since it was
        written (``updated_at``), so that it goes down smoothly between status updates."""
        if self.eta_s is None or not self.state.is_active or not self.updated_at:
            return self.eta_s
        return max(0.0, self.eta_s - max(0.0, time.time() - self.updated_at))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunStatus:
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        if "state" in clean:
            try:
                clean["state"] = RunState(clean["state"])
            except ValueError:
                clean["state"] = RunState.LOST
        return cls(**clean)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["state"] = str(self.state)
        return out


@dataclass
class RunRecord:
    """Everything the history sidebar needs to know about one run."""

    id: str
    name: str
    kind: str  # SimulationKind value
    machine_id: str
    run_dir: str
    created_at: float
    status: RunStatus = field(default_factory=RunStatus)
    device: str = "cpu"
    device_name: str = ""
    variant: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.machine_id}:{self.run_dir}"

    @property
    def project_id(self) -> str | None:
        """The project this run belongs to (stored in its manifest), if any."""
        project = (self.manifest or {}).get("project")
        return project.get("id") if isinstance(project, dict) else None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = self.status.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        clean["status"] = RunStatus.from_dict(clean.get("status") or {})
        return cls(**clean)
