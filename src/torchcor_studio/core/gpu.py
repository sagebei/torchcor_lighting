"""Compute-device telemetry: NVIDIA GPUs, the CPU and (on this Mac) the Apple GPU.

Qt-free. :class:`GpuMonitor` samples a :class:`Machine` on a background thread and
hands every reading to a callback; :func:`sample_once` takes a single reading.

Sources, picked automatically:

* **NVIDIA** — a long-lived ``nvidia-smi --query-gpu=… -lms <interval>`` stream (works
  locally and over SSH), or NVML through ``pynvml`` on this computer when available.
  Running processes come from ``--query-compute-apps`` every few seconds.
* **CPU** — ``psutil`` locally; ``/proc/stat`` + ``/proc/meminfo`` deltas on SSH hosts
  (one small command per interval).
* **Apple GPU** — ``ioreg`` on this Mac. It is shown for context only: TorchCor cannot
  run on it (:attr:`GPUSample.runnable` is ``False``).

Samples are always ordered CUDA GPUs (by index), then the CPU, then the Apple GPU.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import psutil

from torchcor_studio.core.machines.base import Machine, MachineError
from torchcor_studio.core.models import DeviceKind, GPUSample

log = logging.getLogger(__name__)

GPU_FIELDS = ("index", "uuid", "name", "utilization.gpu", "memory.used", "memory.total",
              "temperature.gpu", "power.draw", "power.limit")
GPU_QUERY = ",".join(GPU_FIELDS)
APPS_QUERY = "gpu_uuid,pid,used_memory"
APPS_EVERY_S = 5.0
#: CUDA readings older than this are dropped (e.g. while nvidia-smi restarts).
STALE_AFTER_S = 10.0
MAX_BACKOFF_S = 60.0
NOT_SUPPORTED_HINT = "TorchCor needs an NVIDIA GPU (CUDA) or the CPU"

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def nvidia_smi_argv(loop_ms: int | None = None) -> list[str]:
    """``nvidia-smi`` arguments for the GPU query, optionally looping every ``loop_ms``."""
    argv = ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"]
    if loop_ms:
        argv += ["-lms", str(int(loop_ms))]
    return argv


def apps_argv() -> list[str]:
    return ["nvidia-smi", f"--query-compute-apps={APPS_QUERY}", "--format=csv,noheader,nounits"]


# ============================================================================ parsing
def _number(text: str) -> float | None:
    """A numeric CSV cell; ``[N/A]``, ``[Not Supported]``, ``N/A`` and blanks are ``None``."""
    text = text.strip()
    if not text or text.startswith("[") or text.upper() in ("N/A", "NA"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_nvidia_smi_line(line: str, timestamp: float | None = None) -> GPUSample | None:
    """One row of the ``--query-gpu`` CSV, or ``None`` for anything else (errors, blanks)."""
    cells = [c.strip() for c in line.split(",")]
    if len(cells) < len(GPU_FIELDS):
        return None
    extra = len(cells) - len(GPU_FIELDS)  # a GPU name containing commas
    index_text, gpu_uuid = cells[0], cells[1]
    name = ", ".join(cells[2:3 + extra])
    rest = cells[3 + extra:]
    if not index_text.isdigit():
        return None
    util, mem_used, mem_total, temp, power, limit = (_number(c) for c in rest)
    return GPUSample(
        index=int(index_text),
        name=name,
        kind=DeviceKind.CUDA,
        uuid=gpu_uuid if gpu_uuid.startswith("GPU-") else gpu_uuid.strip("[]"),
        utilization=util or 0.0,
        memory_used_mb=mem_used or 0.0,
        memory_total_mb=mem_total or 0.0,
        temperature_c=temp,
        power_w=power,
        power_limit_w=limit,
        timestamp=timestamp if timestamp is not None else time.time(),
    )


def parse_nvidia_smi(text: str, timestamp: float | None = None) -> list[GPUSample]:
    """All GPU rows of a ``--query-gpu`` CSV, sorted by index (last row per index wins)."""
    found: dict[int, GPUSample] = {}
    for line in text.splitlines():
        sample = parse_nvidia_smi_line(line, timestamp)
        if sample is not None:
            found[sample.index] = sample
    return [found[i] for i in sorted(found)]


def parse_compute_apps(text: str) -> dict[str, int]:
    """Number of compute processes per GPU UUID from ``--query-compute-apps``."""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 2 or not cells[0].startswith("GPU-") or not cells[1].isdigit():
            continue
        counts[cells[0]] = counts.get(cells[0], 0) + 1
    return counts


def with_processes(samples: list[GPUSample], apps: dict[str, int] | None) -> list[GPUSample]:
    """Fill :attr:`GPUSample.processes` from a compute-apps census (``None`` = unknown)."""
    if apps is None:
        return samples
    return [replace(s, processes=apps.get(s.uuid, 0)) for s in samples]


@dataclass(frozen=True)
class CpuTimes:
    """Aggregate jiffies from the first line of ``/proc/stat``."""

    idle: float
    total: float


def parse_proc_stat(text: str) -> CpuTimes | None:
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "cpu" and len(parts) >= 5:
            values = [float(v) for v in parts[1:9] if v.isdigit()]
            idle = values[3] + (values[4] if len(values) > 4 else 0.0)  # idle + iowait
            return CpuTimes(idle=idle, total=sum(values))
    return None


def cpu_percent(previous: CpuTimes | None, current: CpuTimes | None) -> float | None:
    """Utilisation between two ``/proc/stat`` readings, in percent."""
    if previous is None or current is None:
        return None
    total = current.total - previous.total
    if total <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - (current.idle - previous.idle) / total)))


def parse_meminfo(text: str) -> tuple[float, float] | None:
    """``(used_mb, total_mb)`` from ``/proc/meminfo`` (used = total − available)."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached"):
            try:
                values[key] = float(rest.split()[0]) / 1024.0
            except (IndexError, ValueError):
                continue
    total = values.get("MemTotal")
    if not total:
        return None
    available = values.get("MemAvailable")
    if available is None:  # kernels before 3.14
        available = values.get("MemFree", 0.0) + values.get("Buffers", 0.0) + values.get("Cached", 0.0)
    return max(0.0, total - available), total


@dataclass(frozen=True)
class AppleGpuStats:
    """What ``ioreg -c IOAccelerator`` reports about an Apple-silicon GPU."""

    model: str
    cores: int | None
    utilization: float
    memory_used_bytes: float


_IOREG_MODEL = re.compile(r'"model"\s*=\s*"([^"]+)"')
_IOREG_CORES = re.compile(r'"gpu-core-count"\s*=\s*(\d+)')
_IOREG_STATS = re.compile(r'"PerformanceStatistics"\s*=\s*\{([^}]*)\}')


def parse_ioreg(text: str) -> AppleGpuStats | None:
    """Parse ``ioreg -r -d 1 -w 0 -c IOAccelerator`` (first accelerator with statistics)."""
    stats = _IOREG_STATS.search(text)
    if stats is None:
        return None
    body = stats.group(1)

    def stat(name: str) -> float | None:
        m = re.search(r'"' + re.escape(name) + r'"\s*=\s*(\d+)', body)
        return float(m.group(1)) if m else None

    model = _IOREG_MODEL.search(text)
    cores = _IOREG_CORES.search(text)
    util = stat("Device Utilization %")
    if util is None:
        util = stat("GPU Activity(%)")
    return AppleGpuStats(
        model=model.group(1) if model else "Apple GPU",
        cores=int(cores.group(1)) if cores else None,
        utilization=util or 0.0,
        memory_used_bytes=stat("In use system memory") or 0.0,
    )


_CPU_NOISE = re.compile(r"\((?:R|TM|tm|r)\)|\bCPU\b|\b\d+-Core Processor\b|\bProcessor\b|@.*$")


def clean_cpu_model(raw: str) -> str:
    """``"Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz"`` → ``"Intel Xeon Gold 6248"``."""
    text = _CPU_NOISE.sub(" ", raw or "")
    return " ".join(text.split()) or "CPU"


def cpu_display_name(model: str, cores: int | None) -> str:
    """``"Apple M3 · 8 cores"``."""
    name = clean_cpu_model(model)
    return f"{name} · {cores} cores" if cores else name


# ============================================================================ local sources
def _run_local(argv: list[str], timeout: float = 5.0) -> str | None:
    """stdout of a quick local command, or ``None`` if it failed."""
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                             creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def local_cpu_model() -> str:
    """This computer's CPU model, best effort and without heavy dependencies."""
    if sys.platform == "darwin":
        brand = (_run_local(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip()
        if brand:
            return brand
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.lower().startswith(("model name", "hardware")):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "CPU"


class _LocalCpu:
    """CPU utilisation and RAM of this computer via psutil."""

    #: Shortest window that gives a meaningful utilisation reading.
    MIN_WINDOW_S = 0.2

    def __init__(self) -> None:
        self.name = cpu_display_name(local_cpu_model(), psutil.cpu_count(logical=True))
        self._last: float | None = None

    def sample(self) -> GPUSample:
        now = time.monotonic()
        if self._last is None or now - self._last < self.MIN_WINDOW_S:
            util = psutil.cpu_percent(interval=self.MIN_WINDOW_S)  # no usable baseline yet
        else:
            util = psutil.cpu_percent(interval=None)  # since the previous call (this thread)
        self._last = time.monotonic()
        mem = psutil.virtual_memory()
        return GPUSample(index=0, name=self.name, kind=DeviceKind.CPU, uuid="cpu",
                         utilization=float(util),
                         memory_used_mb=(mem.total - mem.available) / 2**20,
                         memory_total_mb=mem.total / 2**20)


class _AppleGpu:
    """The Apple-silicon GPU of this Mac (telemetry only)."""

    ARGV = ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"]

    def __init__(self) -> None:
        self.total_mb = psutil.virtual_memory().total / 2**20

    @staticmethod
    def available() -> bool:
        return sys.platform == "darwin" and platform.machine() == "arm64" and bool(shutil.which("ioreg"))

    def sample(self) -> GPUSample | None:
        stats = parse_ioreg(_run_local(self.ARGV) or "")
        if stats is None:
            return None
        cores = f" · {stats.cores} cores" if stats.cores else ""
        model = stats.model if stats.model.lower().endswith("gpu") else f"{stats.model} GPU"
        return GPUSample(index=0, name=f"{model}{cores}", kind=DeviceKind.APPLE, uuid="apple-gpu",
                         utilization=stats.utilization,
                         memory_used_mb=stats.memory_used_bytes / 2**20,
                         memory_total_mb=self.total_mb)


class _Nvml:
    """NVIDIA telemetry through NVML (``pynvml``): no subprocess, local only."""

    def __init__(self) -> None:
        import pynvml

        pynvml.nvmlInit()
        self.nv = pynvml
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]

    @classmethod
    def open(cls) -> _Nvml | None:
        try:
            return cls()
        except Exception as exc:  # ImportError, NVMLError (no driver / library), ...
            log.debug("NVML unavailable: %s", exc)
            return None

    def _try(self, fn, *args):
        try:
            return fn(*args)
        except self.nv.NVMLError:
            return None

    def sample(self) -> list[GPUSample]:
        nv, out, now = self.nv, [], time.time()
        for i, h in enumerate(self.handles):
            name = self._try(nv.nvmlDeviceGetName, h) or f"GPU {i}"
            gpu_uuid = self._try(nv.nvmlDeviceGetUUID, h) or ""
            util = self._try(nv.nvmlDeviceGetUtilizationRates, h)
            mem = self._try(nv.nvmlDeviceGetMemoryInfo, h)
            temp = self._try(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU)
            power = self._try(nv.nvmlDeviceGetPowerUsage, h)
            limit = self._try(nv.nvmlDeviceGetEnforcedPowerLimit, h)
            procs = self._try(nv.nvmlDeviceGetComputeRunningProcesses, h)
            out.append(GPUSample(
                index=i,
                name=name.decode() if isinstance(name, bytes) else str(name),
                kind=DeviceKind.CUDA,
                uuid=gpu_uuid.decode() if isinstance(gpu_uuid, bytes) else str(gpu_uuid),
                utilization=float(util.gpu) if util is not None else 0.0,
                memory_used_mb=mem.used / 2**20 if mem is not None else 0.0,
                memory_total_mb=mem.total / 2**20 if mem is not None else 0.0,
                temperature_c=float(temp) if temp is not None else None,
                power_w=power / 1000.0 if power is not None else None,
                power_limit_w=limit / 1000.0 if limit is not None else None,
                processes=len(procs) if procs is not None else None,
                timestamp=now,
            ))
        return out

    def close(self) -> None:
        with contextlib.suppress(Exception):  # already shut down
            self.nv.nvmlShutdown()


# ============================================================================ remote source
_SECTION = re.compile(r"^@@(\w+)$")


def _remote_stats_script(with_cpu_info: bool, with_apps: bool, double_stat: bool) -> str:
    parts = ["echo @@stat", "head -n 1 /proc/stat"]
    if double_stat:
        parts += ["sleep 0.25", "echo @@stat2", "head -n 1 /proc/stat"]
    parts += ["echo @@mem", "grep -E '^(MemTotal|MemAvailable|MemFree|Buffers|Cached):' /proc/meminfo"]
    if with_cpu_info:
        parts += ["echo @@cpu", "nproc 2>/dev/null",
                  "grep -m1 -i -E '^(model name|hardware)' /proc/cpuinfo 2>/dev/null"
                  " || lscpu 2>/dev/null | grep -m1 'Model name'"]
    if with_apps:
        parts += ["echo @@apps", "command -v nvidia-smi >/dev/null 2>&1 && " + " ".join(apps_argv())
                  + " 2>/dev/null; echo @@apps_rc=$?"]
    return "; ".join(parts) + "; exit 0"


def _sections(text: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = _SECTION.match(line.strip())
        if m:
            current = m.group(1)
            out[current] = []
        elif line.startswith("@@apps_rc="):
            out["apps_rc"] = [line.split("=", 1)[1]]
        elif current is not None:
            out[current].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


class _RemoteStats:
    """CPU (+ compute-apps census) of an SSH host in one shell command per interval."""

    def __init__(self, machine: Machine) -> None:
        self.machine = machine
        self.name: str | None = None
        self.previous: CpuTimes | None = None
        self.apps: dict[str, int] | None = None

    def sample(self, with_apps: bool, timeout: float = 30.0) -> GPUSample | None:
        first = self.previous is None
        result = self.machine.run(
            _remote_stats_script(self.name is None, with_apps, double_stat=first), timeout=timeout)
        sec = _sections(result.stdout)
        if self.name is None and "cpu" in sec:
            lines = [ln.strip() for ln in sec["cpu"].splitlines() if ln.strip()]
            cores = int(lines[0]) if lines and lines[0].isdigit() else None
            model = next((ln.split(":", 1)[1].strip() for ln in lines if ":" in ln), "CPU")
            self.name = cpu_display_name(model, cores)
        if with_apps and "apps" in sec:
            self.apps = parse_compute_apps(sec["apps"]) if sec.get("apps_rc", "1").strip() == "0" else None
        times = parse_proc_stat(sec.get("stat", ""))
        if first and "stat2" in sec:
            self.previous, times = times, parse_proc_stat(sec["stat2"])
        util = cpu_percent(self.previous, times)
        self.previous = times or self.previous
        memory = parse_meminfo(sec.get("mem", ""))
        if times is None or memory is None:
            return None  # not Linux (no /proc): no CPU telemetry
        return GPUSample(index=0, name=self.name or "CPU", kind=DeviceKind.CPU, uuid="cpu",
                         utilization=util or 0.0, memory_used_mb=memory[0], memory_total_mb=memory[1])


# ============================================================================ one-shot
def _order(samples: list[GPUSample]) -> list[GPUSample]:
    rank = {DeviceKind.CUDA: 0, DeviceKind.CPU: 1, DeviceKind.APPLE: 2}
    return sorted(samples, key=lambda s: (rank.get(s.kind, 3), s.index))


def sample_once(machine: Machine) -> list[GPUSample]:
    """A single reading of every device of ``machine`` (takes a fraction of a second)."""
    samples: list[GPUSample] = []
    if machine.profile.is_local:
        nvml = _Nvml.open()
        if nvml is not None:
            try:
                samples += nvml.sample()
            finally:
                nvml.close()
        elif shutil.which("nvidia-smi"):
            gpus = parse_nvidia_smi(_run_local(nvidia_smi_argv(), timeout=15) or "")
            apps = _run_local(apps_argv(), timeout=15)
            samples += with_processes(gpus, parse_compute_apps(apps) if apps is not None else None)
        samples.append(_LocalCpu().sample())
        if _AppleGpu.available():
            apple = _AppleGpu().sample()
            if apple is not None:
                samples.append(apple)
        return _order(samples)
    result = machine.run(" ".join(nvidia_smi_argv()) + " 2>/dev/null; exit 0", timeout=30)
    gpus = parse_nvidia_smi(result.stdout)
    stats = _RemoteStats(machine)
    cpu = stats.sample(with_apps=bool(gpus))
    samples += with_processes(gpus, stats.apps)
    if cpu is not None:
        samples.append(cpu)
    return _order(samples)


def pick_best(samples: list[GPUSample]) -> GPUSample | None:
    """The least busy CUDA GPU, else the CPU (``None`` if neither is present).

    GPUs are ranked by running processes, then by the larger of load and memory use
    (in 10 % steps, so an idle GPU whose memory is taken does not win), then by load.
    """
    cuda = [s for s in samples if s.kind == DeviceKind.CUDA]
    if cuda:
        return min(cuda, key=lambda s: (s.processes or 0,
                                        round(max(s.utilization / 100.0, s.memory_fraction) * 10),
                                        s.utilization, s.memory_fraction, s.index))
    return next((s for s in samples if s.kind == DeviceKind.CPU), None)


# ============================================================================ monitor
class GpuMonitor:
    """Background telemetry for one machine.

    ``callback(samples)`` is called from the monitor's thread about every ``interval``
    seconds with every device (CUDA GPUs first, then the CPU, then the Apple GPU). It
    must be quick and thread-safe — in Qt, emit a signal (see ``ui.widgets.gpu.GpuFeed``).
    ``on_error(message | None)`` reports telemetry trouble (``None`` once it recovers).
    """

    def __init__(
        self,
        machine: Machine,
        interval: float = 1.0,
        callback: Callable[[list[GPUSample]], None] | None = None,
        on_error: Callable[[str | None], None] | None = None,
    ) -> None:
        self.machine = machine
        self.interval = max(0.2, float(interval))
        self.callback = callback
        self.on_error = on_error
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._cuda: dict[int, GPUSample] = {}
        self._cuda_lock = threading.Lock()
        self._latest: list[GPUSample] = []
        self._error: str | None = None
        self._reported: set[str] = set()
        #: ``True`` once nvidia-smi turned out to be missing on the machine.
        self.nvidia_missing = False

    # ------------------------------------------------------------------ api
    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    @property
    def latest(self) -> list[GPUSample]:
        """The most recent reading (empty before the first one)."""
        return list(self._latest)

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> None:
        """Start sampling (a no-op while running; a stopped monitor can be restarted)."""
        if self.running and not self._stop.is_set():
            return
        # A fresh event per run: threads of a previous run may still be winding down.
        self._stop = stop = threading.Event()
        name = self.machine.profile.name
        self._threads = [threading.Thread(target=self._loop, args=(stop,), name=f"tcs-gpu-{name}",
                                          daemon=True)]
        self._threads[0].start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop sampling and join the threads (streams are terminated)."""
        self._stop.set()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))

    # ------------------------------------------------------------------ internals
    def _report(self, key: str, message: str, level: int = logging.INFO) -> None:
        """Log each distinct problem once (no log spam on every retry)."""
        if key not in self._reported:
            self._reported.add(key)
            log.log(level, "%s: %s", self.machine.profile.name, message)

    def _set_error(self, message: str | None) -> None:
        if message != self._error:
            self._error = message
            if self.on_error is not None:
                try:
                    self.on_error(message)
                except Exception:  # a UI bug must not kill telemetry
                    log.exception("GPU monitor error callback failed")

    def _start_stream(self, stop: threading.Event) -> None:
        thread = threading.Thread(target=self._stream_loop, args=(stop,), name="tcs-gpu-stream",
                                  daemon=True)
        self._threads.append(thread)
        thread.start()

    def _stream_loop(self, stop: threading.Event) -> None:
        """Keep ``nvidia-smi -lms`` running; restart with exponential backoff."""
        backoff = 2.0
        loop_ms = max(200, int(self.interval * 1000))
        while not stop.is_set():
            started = time.monotonic()
            other: list[str] = []

            def on_line(line: str, other: list[str] = other) -> None:
                sample = parse_nvidia_smi_line(line)
                if sample is None:
                    if line.strip():
                        other.append(line.strip())
                    return
                with self._cuda_lock:
                    self._cuda[sample.index] = sample

            try:
                code = self.machine.stream(nvidia_smi_argv(loop_ms), on_line, stop)
            except MachineError as exc:
                code, other = None, [str(exc)]
            if stop.is_set():
                return
            detail = other[-1] if other else f"exit code {code}"
            if code == 127 or "not found" in detail.lower():
                self.nvidia_missing = True
                self._report("nvidia-missing", "nvidia-smi not found; showing the CPU only")
                backoff = 300.0
            else:
                self._report(f"nvidia:{detail}", f"nvidia-smi stopped ({detail}); retrying")
                if time.monotonic() - started > 60:
                    backoff = 2.0
            stop.wait(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S) if not self.nvidia_missing else backoff

    def _cuda_snapshot(self) -> list[GPUSample]:
        cutoff = time.time() - max(STALE_AFTER_S, 3 * self.interval)
        with self._cuda_lock:
            for index in [i for i, s in self._cuda.items() if s.timestamp < cutoff]:
                del self._cuda[index]
            return [self._cuda[i] for i in sorted(self._cuda)]

    def _loop(self, stop: threading.Event) -> None:
        local = self.machine.profile.is_local
        nvml = _Nvml.open() if local else None
        cpu_local = _LocalCpu() if local else None
        remote = None if local else _RemoteStats(self.machine)
        apple = _AppleGpu() if local and _AppleGpu.available() else None
        if nvml is None and (not local or shutil.which("nvidia-smi")):
            self._start_stream(stop)
        apps: dict[str, int] | None = None
        last_apps = 0.0
        remote_retry_at = 0.0
        remote_backoff = 2.0
        cpu: GPUSample | None = None
        next_tick = time.monotonic()
        try:
            while not stop.is_set():
                now = time.monotonic()
                want_apps = now - last_apps >= APPS_EVERY_S
                cuda = nvml.sample() if nvml is not None else self._cuda_snapshot()
                if cpu_local is not None:
                    cpu = cpu_local.sample()
                    if want_apps and nvml is None and cuda:
                        out = _run_local(apps_argv(), timeout=10)
                        apps = parse_compute_apps(out) if out is not None else None
                        last_apps = now
                elif now >= remote_retry_at:
                    try:
                        cpu = remote.sample(with_apps=want_apps and not self.nvidia_missing,
                                            timeout=max(10.0, 5 * self.interval))
                        if want_apps:
                            apps, last_apps = remote.apps, now
                        remote_backoff = 2.0
                        self._set_error(None)
                    except MachineError as exc:
                        cpu = None
                        self._report(f"remote:{exc}", f"telemetry failed: {exc}", logging.WARNING)
                        self._set_error(str(exc))
                        remote_retry_at = time.monotonic() + remote_backoff
                        remote_backoff = min(remote_backoff * 2, MAX_BACKOFF_S)
                samples = with_processes(cuda, apps) if nvml is None else cuda
                if cpu is not None:
                    samples = [*samples, cpu]
                if apple is not None:
                    reading = apple.sample()
                    if reading is not None:
                        samples.append(reading)
                self._latest = samples
                if self.callback is not None:
                    try:
                        self.callback(list(samples))
                    except Exception:
                        log.exception("GPU monitor callback failed")
                next_tick += self.interval
                delay = next_tick - time.monotonic()
                if delay < 0:  # fell behind (slow link): don't try to catch up
                    next_tick, delay = time.monotonic(), 0.0
                stop.wait(delay)
        finally:
            if nvml is not None:
                nvml.close()
