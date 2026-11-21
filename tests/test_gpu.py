"""GPU / CPU telemetry: parsers, device picking and the background monitor."""

from __future__ import annotations

import sys
import threading
import time

import pytest

from torchcor_studio.core import gpu
from torchcor_studio.core.machines.base import MachineError
from torchcor_studio.core.machines.local import LocalMachine
from torchcor_studio.core.models import (
    CommandResult,
    DeviceKind,
    GPUSample,
    MachineKind,
    MachineProfile,
)

SMI = """\
0, GPU-11111111-aaaa-bbbb-cccc-000000000000, NVIDIA A100-SXM4-80GB, 87, 51234, 81920, 64, 312.45, 400.00
1, GPU-22222222-aaaa-bbbb-cccc-000000000000, NVIDIA A100-SXM4-80GB, 0, 4, 81920, 33, 61.20, 400.00
2, GPU-33333333-aaaa-bbbb-cccc-000000000000, NVIDIA GeForce GTX 1080, [N/A], 312, 8192, [N/A], [Not Supported], [Not Supported]
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
"""

APPS = """\
GPU-11111111-aaaa-bbbb-cccc-000000000000, 4242, 40960
GPU-11111111-aaaa-bbbb-cccc-000000000000, 4243, 1024
GPU-33333333-aaaa-bbbb-cccc-000000000000, 999, [N/A]
"""

IOREG = """\
+-o AGXAcceleratorG15G  <class AGXAcceleratorG15G, id 0x100000575, registered, matched, active, busy 0 (571 ms), retain 77>
    {
      "IOMatchedAtBoot" = Yes
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=4300275712,"Tiler Utilization %"=32,"Renderer Utilization %"=31,"Device Utilization %"=37,"In use system memory"=830521344}
      "model" = "Apple M3"
      "gpu-core-count" = 10
    }
"""

PROC_STAT_1 = "cpu  4705 356 584 3699 23 23 0 0 0 0\ncpu0 1393280 32966 572056 13343292 6130 0 17875 0 0 0\n"
PROC_STAT_2 = "cpu  5705 356 584 4699 23 23 0 0 0 0\n"
MEMINFO = "MemTotal:       16384000 kB\nMemFree:         1000000 kB\nMemAvailable:    4096000 kB\nBuffers:  1 kB\n"


# ============================================================================ parsing
def test_parse_nvidia_smi_handles_na_and_junk():
    samples = gpu.parse_nvidia_smi(SMI, timestamp=123.0)
    assert [s.index for s in samples] == [0, 1, 2]
    first, _, old = samples
    assert first.name == "NVIDIA A100-SXM4-80GB" and first.kind == DeviceKind.CUDA
    assert first.uuid.startswith("GPU-1111")
    assert first.utilization == 87 and first.memory_used_mb == 51234 and first.memory_total_mb == 81920
    assert first.temperature_c == 64 and first.power_w == pytest.approx(312.45) and first.power_limit_w == 400
    assert first.timestamp == 123.0 and first.device_string == "cuda:0"
    assert old.utilization == 0.0 and old.temperature_c is None and old.power_w is None and old.power_limit_w is None


def test_parse_nvidia_smi_line_with_commas_in_name_and_bad_rows():
    row = "3, GPU-4, NVIDIA RTX A6000, Ada, 50, 100, 200, 40, 100, 300"
    sample = gpu.parse_nvidia_smi_line(row)
    assert sample is not None and sample.name == "NVIDIA RTX A6000, Ada" and sample.power_limit_w == 300
    assert gpu.parse_nvidia_smi_line("") is None
    assert gpu.parse_nvidia_smi_line("index, uuid, name, a, b, c, d, e, f") is None
    assert gpu.parse_nvidia_smi_line("No devices were found") is None


def test_compute_apps_fill_process_counts():
    apps = gpu.parse_compute_apps(APPS + "\nNo running processes found\n")
    assert apps == {"GPU-11111111-aaaa-bbbb-cccc-000000000000": 2, "GPU-33333333-aaaa-bbbb-cccc-000000000000": 1}
    samples = gpu.with_processes(gpu.parse_nvidia_smi(SMI), apps)
    assert [s.processes for s in samples] == [2, 0, 1]
    assert gpu.with_processes(samples[:1], None)[0].processes == 2  # unknown census keeps values


def test_proc_stat_and_meminfo():
    a, b = gpu.parse_proc_stat(PROC_STAT_1), gpu.parse_proc_stat(PROC_STAT_2)
    assert a.idle == 3699 + 23 and a.total == 4705 + 356 + 584 + 3699 + 23 + 23
    assert gpu.cpu_percent(a, b) == pytest.approx(50.0)
    assert gpu.cpu_percent(None, b) is None and gpu.cpu_percent(b, b) is None
    used, total = gpu.parse_meminfo(MEMINFO)
    assert total == pytest.approx(16000.0) and used == pytest.approx(12000.0)
    legacy = "MemTotal: 2048 kB\nMemFree: 512 kB\nBuffers: 256 kB\nCached: 256 kB\n"
    assert gpu.parse_meminfo(legacy) == (pytest.approx(1.0), pytest.approx(2.0))
    assert gpu.parse_meminfo("garbage") is None
    assert gpu.parse_proc_stat("intr 1 2 3") is None


def test_parse_ioreg_apple_gpu():
    stats = gpu.parse_ioreg(IOREG)
    assert stats == gpu.AppleGpuStats(model="Apple M3", cores=10, utilization=37.0,
                                      memory_used_bytes=830521344.0)
    assert gpu.parse_ioreg("") is None


def test_cpu_names_are_cleaned():
    assert gpu.clean_cpu_model("Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz") == "Intel Xeon Gold 6248"
    assert gpu.clean_cpu_model("AMD EPYC 7763 64-Core Processor") == "AMD EPYC 7763"
    assert gpu.cpu_display_name("Apple M3", 8) == "Apple M3 · 8 cores"
    assert gpu.cpu_display_name("", None) == "CPU"


def test_samples_are_ordered_and_flags():
    apple = GPUSample(0, "Apple M3 GPU", DeviceKind.APPLE)
    cpu = GPUSample(0, "CPU", DeviceKind.CPU)
    cuda = [GPUSample(1, "B"), GPUSample(0, "A")]
    ordered = gpu._order([apple, cpu, *cuda])
    assert [s.key for s in ordered] == ["cuda:0", "cuda:1", "cpu:0", "apple:0"]
    assert not apple.runnable and cpu.runnable and cpu.device_string == "cpu"


# ============================================================================ picking
def test_pick_best_prefers_idle_cuda_then_cpu():
    busy = GPUSample(0, "A", utilization=90, memory_used_mb=70_000, memory_total_mb=80_000, processes=1)
    idle = GPUSample(1, "A", utilization=3, memory_used_mb=100, memory_total_mb=80_000, processes=0)
    loaded = GPUSample(2, "A", utilization=0, memory_used_mb=60_000, memory_total_mb=80_000, processes=0)
    cpu = GPUSample(0, "CPU", DeviceKind.CPU, utilization=10)
    apple = GPUSample(0, "Apple", DeviceKind.APPLE)
    assert gpu.pick_best([busy, idle, loaded, cpu]).index == 1
    assert gpu.pick_best([cpu, apple]) is cpu
    assert gpu.pick_best([apple]) is None
    assert gpu.pick_best([]) is None


# ============================================================================ fakes
class FakeRemote(LocalMachine):
    """Looks like an SSH host: canned nvidia-smi stream and /proc readings."""

    def __init__(self, *, nvidia: bool = True, fail_stats: bool = False) -> None:
        super().__init__(MachineProfile(name="fake", kind=MachineKind.SSH, host="fake"))
        self.nvidia = nvidia
        self.fail_stats = fail_stats
        self.commands: list[str] = []
        self.ticks = 0

    def run(self, argv, *, timeout=60.0, cwd=None, env=None, stdin=None):
        command = argv if isinstance(argv, str) else " ".join(argv)
        self.commands.append(command)
        if self.fail_stats:
            raise MachineError("Lost the connection to fake")
        self.ticks += 1
        stat = f"cpu  {1000 + 100 * self.ticks} 0 0 {1000 + 300 * self.ticks} 0 0 0 0 0 0"
        out = [f"@@stat\n{stat}"]
        if "@@stat2" in command:
            out.append(f"@@stat2\ncpu  {1100 + 100 * self.ticks} 0 0 {1300 + 300 * self.ticks} 0 0 0 0 0 0")
        out.append("@@mem\n" + MEMINFO)
        if "@@cpu" in command:
            out.append("@@cpu\n64\nmodel name\t: AMD EPYC 7763 64-Core Processor")
        if "@@apps" in command:
            out.append("@@apps\n" + (APPS if self.nvidia else "") + f"\n@@apps_rc={0 if self.nvidia else 1}")
        return CommandResult(0, "\n".join(out) + "\n", "")

    def stream(self, argv, on_line, stop, *, env=None):
        if not self.nvidia:
            on_line("sh: 1: nvidia-smi: not found")
            return 127
        while not stop.is_set():
            for line in SMI.splitlines():
                on_line(line)
            stop.wait(0.05)
        return -1


# ============================================================================ monitor
def _collect(monitor: gpu.GpuMonitor, count: int, timeout: float = 5.0) -> list[list[GPUSample]]:
    got: list[list[GPUSample]] = []
    done = threading.Event()

    def callback(samples):
        got.append(samples)
        if len(got) >= count:
            done.set()

    monitor.callback = callback
    monitor.start()
    try:
        assert done.wait(timeout), "monitor produced too few readings"
    finally:
        started = time.monotonic()
        monitor.stop()
        assert time.monotonic() - started < 3.0
    assert not monitor.running
    return got


def test_monitor_remote_cuda_then_cpu_with_processes():
    machine = FakeRemote()
    readings = _collect(gpu.GpuMonitor(machine, interval=0.2), 4)
    last = readings[-1]
    assert [s.kind for s in last] == [DeviceKind.CUDA] * 3 + [DeviceKind.CPU]
    assert [s.processes for s in last[:3]] == [2, 0, 1]
    cpu = last[-1]
    assert cpu.name == "AMD EPYC 7763 · 64 cores"
    assert cpu.utilization == pytest.approx(25.0)
    assert cpu.memory_total_mb == pytest.approx(16000.0)
    assert sum("@@cpu" in c for c in machine.commands) == 1  # CPU model fetched once


def test_monitor_survives_missing_nvidia_smi():
    machine = FakeRemote(nvidia=False)
    monitor = gpu.GpuMonitor(machine, interval=0.2)
    readings = _collect(monitor, 3)
    assert all([s.kind for s in r] == [DeviceKind.CPU] for r in readings)
    assert monitor.nvidia_missing


def test_monitor_reports_errors_once_and_backs_off():
    machine = FakeRemote(fail_stats=True, nvidia=False)
    errors: list[str | None] = []
    monitor = gpu.GpuMonitor(machine, interval=0.1, on_error=errors.append)
    readings = _collect(monitor, 5)
    assert readings[-1] == []
    assert errors == ["Lost the connection to fake"]
    assert len(machine.commands) <= 3  # 2 s backoff: not one attempt per tick


def test_monitor_callback_exceptions_do_not_stop_it():
    calls = []

    def bad(samples):
        calls.append(samples)
        raise RuntimeError("UI bug")

    monitor = gpu.GpuMonitor(FakeRemote(), interval=0.1, callback=bad)
    monitor.start()
    time.sleep(0.5)
    monitor.stop()
    assert len(calls) >= 3


def test_monitor_local_machine_has_cpu():
    machine = LocalMachine(MachineProfile(name="local"))
    readings = _collect(gpu.GpuMonitor(machine, interval=0.25), 2)
    kinds = [s.kind for s in readings[-1]]
    assert DeviceKind.CPU in kinds
    if sys.platform == "darwin" and gpu._AppleGpu.available():
        assert kinds[-1] == DeviceKind.APPLE
        assert not readings[-1][-1].runnable


def test_sample_once_local_and_remote():
    local = gpu.sample_once(LocalMachine(MachineProfile(name="local")))
    cpu = next(s for s in local if s.kind == DeviceKind.CPU)
    assert cpu.memory_total_mb > 0 and 0.0 <= cpu.utilization <= 100.0

    class OneShot(FakeRemote):
        def run(self, argv, **kwargs):
            if isinstance(argv, str) and argv.startswith("nvidia-smi"):
                return CommandResult(0, SMI, "")
            return super().run(argv, **kwargs)

    remote = gpu.sample_once(OneShot())
    assert [s.key for s in remote] == ["cuda:0", "cuda:1", "cuda:2", "cpu:0"]
    assert remote[0].processes == 2


def test_monitor_can_restart_while_the_previous_run_winds_down():
    machine = FakeRemote()
    got: list[list[GPUSample]] = []
    monitor = gpu.GpuMonitor(machine, interval=0.1, callback=got.append)
    monitor.start()
    time.sleep(0.3)
    monitor.stop(timeout=0)  # non-blocking, as the UI does
    monitor.start()
    count = len(got)
    time.sleep(0.4)
    assert monitor.running and len(got) > count
    monitor.stop()
    assert not monitor.running
