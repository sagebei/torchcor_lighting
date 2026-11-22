#!/usr/bin/env python3
"""TorchCor Lighting runtime — the small agent that runs on every compute machine.

This file is copied verbatim to the machines the studio drives (this computer or an
SSH host) and into every run folder. It depends on the Python standard library only,
so it works in any environment, before TorchCor is even installed.

Commands (each prints one JSON document on the last line of stdout)::

    probe                         describe the Python environment (torch, CUDA, TorchCor)
    inspect <dir>                 summarise an input folder (mesh, .vtx, .surf)
    scan <root> [<root> ...]      list the runs found under output folders
    launch <run_dir>              start a detached supervisor for a prepared run folder
    supervise <run_dir>           (internal) run ``run.py`` and keep ``status.json`` current
    cancel <run_dir>              ask a run to stop (terminates it if its supervisor died)
    alive <pid>                   is a process alive?
    kill <pid>                    terminate a process and its children
    gpus                          one-shot GPU telemetry (nvidia-smi)
    remove <run_dir>              delete a run folder (only folders with a manifest)
    preview <mesh_dir> <out.npz>  boundary surface + pacing sites of a mesh (needs numpy)
    fetch-data <dest> [<repo>]    download the newest demo data release into <dest>/torchcor_data
                                  (progress lines {"progress": ...} come first)

Run folder protocol (shared with the desktop app)::

    manifest.json    what was asked for — written by the app
    run.py           the generated TorchCor script
    status.json      state, progress, timing — written by the supervisor
    run.log          stdout + stderr of run.py
    events.jsonl     structured events emitted by run.py (stage, progress, metric, ...)
    cancel.request   created by the app to request cancellation
    results/         TorchCor outputs (Vm.pt, ATs.pt, displacement, ...)
    viz/             lightweight bundle the app downloads for visualisation
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

RUNTIME_VERSION = "1"

MANIFEST = "manifest.json"
STATUS = "status.json"
LOG = "run.log"
EVENTS = "events.jsonl"
SCRIPT = "run.py"
CANCEL = "cancel.request"
SUPERVISOR_LOG = "supervisor.log"
RESULTS_DIR = "results"
VIZ_DIR = "viz"

MARKER = "::tcs::"
HEARTBEAT_S = 2.0
STALE_AFTER_S = 45.0

IS_WINDOWS = os.name == "nt"


# =============================================================================== utils
def _emit(payload) -> None:
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _write_json_atomic(path, data) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    for attempt in range(5):  # Windows: the reader may briefly hold the file open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def pid_alive(pid) -> bool:
    if not pid:
        return False
    pid = int(pid)
    if IS_WINDOWS:
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return f'"{pid}"' in out
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    # A zombie child of ours still "exists"; treat it as dead.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().split()[2] != "Z"
    except OSError:
        return True


def kill_tree(pid, grace_s=8.0, alive=None) -> None:
    """Terminate ``pid`` and its process group (SIGTERM, then SIGKILL after ``grace_s``).

    ``alive`` overrides the liveness check — a parent must pass ``lambda: child.poll() is
    None`` for its own child, because an unreaped child is a zombie that still "exists".
    """
    if not pid:
        return
    pid = int(pid)
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
        return
    alive = alive or (lambda: pid_alive(pid))
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return

    def send(sig) -> None:
        try:
            if pgid == pid:
                os.killpg(pgid, sig)
            else:
                os.kill(pid, sig)
        except OSError:  # gone already (macOS reports EPERM for a zombie-only group)
            pass

    send(signal.SIGTERM)
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not alive():
            return
        time.sleep(0.2)
    send(signal.SIGKILL)


# =============================================================================== probe
def _memory_total_gb() -> float:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024**2
        elif sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            return int(out.stdout.strip()) / 1024**3
        elif IS_WINDOWS:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullTotalPhys / 1024**3
    except Exception:  # noqa: BLE001 - best effort
        pass
    return 0.0


def _nvidia_gpu_count():
    """GPUs ``nvidia-smi -L`` lists (0: no NVIDIA driver or GPU)."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def cmd_probe(_args) -> None:
    report = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count() or 0,
        "memory_total_gb": round(_memory_total_gb(), 1),
        "runtime_version": RUNTIME_VERSION,
        "nvidia_gpus": _nvidia_gpu_count(),
    }
    try:
        import numpy

        report["numpy_version"] = numpy.__version__
    except Exception:  # noqa: BLE001
        report["numpy_version"] = None
    try:
        import torch

        report["torch_version"] = torch.__version__.split("+")[0]
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = torch.version.cuda
        report["cuda_device_count"] = torch.cuda.device_count() if report["cuda_available"] else 0
        mps = getattr(torch.backends, "mps", None)
        report["mps_available"] = bool(mps and mps.is_available())
    except Exception as exc:  # noqa: BLE001
        report["torch_version"] = None
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from importlib import metadata

        report["torchcor_version"] = metadata.version("torchcor")
        import importlib.util

        spec = importlib.util.find_spec("torchcor")
        report["torchcor_path"] = os.path.dirname(spec.origin) if spec and spec.origin else None
    except Exception:  # noqa: BLE001
        report["torchcor_version"] = None
    report["clock"] = time.time()  # last: the app compares it with its own clock on arrival
    _emit(report)


# =============================================================================== inspect
_ELEM_NODES = {"Ln": 2, "Tr": 3, "Qd": 4, "Tt": 4, "Py": 5, "Pr": 6, "Hx": 8}


def _first_line_int(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return int(fh.readline().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _scan_elems(path, limit_bytes):
    """Unique element types and region tags of an ``.elem`` file."""
    types, regions = set(), set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        fh.readline()
        read = 0
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            types.add(parts[0])
            n = _ELEM_NODES.get(parts[0])
            if n is not None and len(parts) > n + 1:
                regions.add(int(parts[n + 1]))
            read += len(line)
            if limit_bytes and read > limit_bytes:
                break
    return sorted(types), sorted(regions)


def inspect_dir(path, limit_bytes=0):
    info = {"path": path, "exists": os.path.isdir(path)}
    if not info["exists"]:
        return info
    try:
        names = sorted(os.listdir(path))
    except OSError as exc:
        info["error"] = str(exc)
        return info
    files = [n for n in names if os.path.isfile(os.path.join(path, n))]
    info["files"] = files
    info["subdirs"] = [n for n in names if os.path.isdir(os.path.join(path, n)) and not n.startswith(".")]

    def first(ext):
        found = [f for f in files if f.lower().endswith(ext)]
        return found[0] if found else None

    pts, elem, lon = first(".pts"), first(".elem"), first(".lon")
    if pts and elem:
        info["mesh_name"] = os.path.splitext(pts)[0]
        info["n_nodes"] = _first_line_int(os.path.join(path, pts))
        info["n_elems"] = _first_line_int(os.path.join(path, elem))
        try:
            info["elem_types"], info["regions"] = _scan_elems(os.path.join(path, elem), limit_bytes)
        except (OSError, ValueError) as exc:
            info["error"] = f"could not read {elem}: {exc}"
        if lon:
            ncol = _first_line_int(os.path.join(path, lon))
            info["fibre_columns"] = 6 if ncol == 2 else 3
    info["vtx"] = [
        {"name": f, "count": _first_line_int(os.path.join(path, f)) or 0}
        for f in files if f.lower().endswith(".vtx")
    ]
    info["surfaces"] = [f for f in files if f.lower().endswith(".surf")]
    return info


def cmd_inspect(args) -> None:
    if not args:
        raise SystemExit("usage: inspect <dir> [--limit-mb N]")
    limit = 0
    if "--limit-mb" in args:
        limit = int(float(args[args.index("--limit-mb") + 1]) * 1024 * 1024)
    _emit(inspect_dir(os.path.abspath(os.path.expanduser(args[0])), limit))


# =============================================================================== scan
def _effective_status(run_dir, status):
    """Downgrade a 'running' status whose supervisor has silently died to 'lost'."""
    if not status:
        return {"state": "lost", "error": "status.json is missing"}
    if status.get("state") in ("queued", "running"):
        fresh = time.time() - float(status.get("updated_at") or 0) < STALE_AFTER_S
        same_host = status.get("host") in (None, "", socket.gethostname())
        alive = pid_alive(status.get("pid")) if same_host else fresh
        if status.get("state") == "queued" and not status.get("pid"):
            created = os.path.getmtime(os.path.join(run_dir, STATUS))
            alive = time.time() - created < STALE_AFTER_S
        if not alive:
            status = dict(status, state="lost", error=status.get("error") or "the run stopped reporting")
    return status


def scan_roots(roots, depth=2):
    runs = []
    seen = set()
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            continue
        frontier = [(root, 0)]
        while frontier:
            folder, level = frontier.pop()
            manifest_path = os.path.join(folder, MANIFEST)
            if os.path.isfile(manifest_path):
                real = os.path.realpath(folder)
                if real in seen:
                    continue
                seen.add(real)
                manifest = _read_json(manifest_path, {}) or {}
                status = _effective_status(folder, _read_json(os.path.join(folder, STATUS)))
                runs.append({
                    "run_dir": folder,
                    "manifest": manifest,
                    "status": status,
                    "has_viz": os.path.isfile(os.path.join(folder, VIZ_DIR, "index.json")),
                })
                continue
            if level >= depth:
                continue
            try:
                for entry in os.scandir(folder):
                    if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                        frontier.append((entry.path, level + 1))
            except OSError:
                continue
    runs.sort(key=lambda r: r["manifest"].get("created_at") or 0, reverse=True)
    return runs


def cmd_scan(args) -> None:
    _emit({"runs": scan_roots(args)})


# =============================================================================== launch
def cmd_launch(args) -> None:
    if not args:
        raise SystemExit("usage: launch <run_dir>")
    run_dir = os.path.abspath(os.path.expanduser(args[0]))
    if not os.path.isfile(os.path.join(run_dir, SCRIPT)):
        raise SystemExit(f"{run_dir} has no {SCRIPT}")
    cmd = [sys.executable, os.path.abspath(__file__), "supervise", run_dir]
    log = open(os.path.join(run_dir, SUPERVISOR_LOG), "ab")  # noqa: SIM115 - handed to the child
    kwargs = {"cwd": run_dir, "stdin": subprocess.DEVNULL, "stdout": log, "stderr": subprocess.STDOUT}
    if IS_WINDOWS:
        flags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_PROCESS_GROUP | NO_WINDOW
        try:
            proc = subprocess.Popen(cmd, creationflags=flags | 0x01000000, **kwargs)  # BREAKAWAY_FROM_JOB
        except OSError:
            proc = subprocess.Popen(cmd, creationflags=flags, **kwargs)
    else:
        proc = subprocess.Popen(cmd, start_new_session=True, close_fds=True, **kwargs)
    log.close()
    _emit({"pid": proc.pid, "run_dir": run_dir, "host": socket.gethostname()})


# =============================================================================== supervise
_RE_EP = re.compile(r"^t:\s*([-\d.eE+]+)\s*/\s*([-\d.eE+]+)\s*\|")
_RE_LOAD = re.compile(r"load step\s+(\d+):\s+lambda\s*=\s*([-\d.eE+]+)")
_RE_DYN = re.compile(r"^\s*t\s*=\s*([-\d.eE+]+)\s+newton")
_RE_EM = re.compile(r"^t\s+([-\d.]+)\s*ms\s*\|\s*p \(mmHg\)")
_RE_LEAD = re.compile(r"^\s*(V[1-6]|RA|LA|LL|RL):\s+CG iters")
_RE_UNLOAD = re.compile(r"^\s*iteration\s+(\d+):\s+largest mismatch\s+([-\d.eE+%]+)")
_RE_EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit)|Exception|KeyboardInterrupt)\b:?\s*(.*)$")


class _Progress:
    """Turns run.py's output (TorchCor's own lines + ::tcs:: markers) into status fields."""

    LEADS = ("V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "LL")

    def __init__(self, events_path):
        self.events_path = events_path
        self.stage, self.stage_label = "start", "Starting"
        self.progress = None
        self.message = ""
        self.sim_time = None
        self.sim_total = None
        self.unit = "ms"
        self.stage_started = time.time()
        self.stage_progress_started = None
        self.leads_done = set()
        self.metrics = {}
        self.auto = True  # parse TorchCor's own lines for progress in this stage

    def _event(self, payload):
        payload.setdefault("time", time.time())
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

    def _set_stage(self, sid, label):
        if sid != self.stage:
            self.stage, self.stage_label = sid, label or sid.title()
            self.progress, self.message = None, ""  # nothing of the last stage carries over
            self.stage_started = time.time()
            self.stage_progress_started = None

    def _set_progress(self, value, message=None):
        value = max(0.0, min(1.0, float(value)))
        if self.stage_progress_started is None:
            self.stage_progress_started = (time.time(), value)
        self.progress = value
        if message is not None:
            self.message = message

    def eta(self):
        if self.progress is None or self.stage_progress_started is None:
            return None
        t0, p0 = self.stage_progress_started
        done = self.progress - p0
        if done <= 1e-6:
            return None
        return (time.time() - t0) * (1.0 - self.progress) / done

    def feed(self, line):
        if line.startswith(MARKER):
            try:
                ev = json.loads(line[len(MARKER):].strip())
            except ValueError:
                return
            self._event(ev)
            kind = ev.get("type")
            if kind == "stage":
                self._set_stage(ev.get("id", "stage"), ev.get("label"))
                self.auto = bool(ev.get("auto", True))
                if ev.get("total") is not None:
                    self.sim_total = float(ev["total"])
                    self.unit = ev.get("unit", self.unit)
            elif kind == "total":
                self.sim_total = float(ev["value"])
                self.unit = ev.get("unit", self.unit)
            elif kind == "progress":  # a message describes this progress only
                self._set_progress(ev.get("value", 0.0), ev.get("message") or "")
            elif kind == "metric":
                self.metrics[ev.get("name", "metric")] = ev.get("value")
            return
        m = _RE_UNLOAD.match(line)
        if m:  # unloading converges in an unknown number of iterations: report, don't estimate
            self.message = f"iteration {m.group(1)} · largest mismatch {m.group(2)}"
            return
        if not self.auto:  # the stage reports its own progress (e.g. unloading iterations)
            return

        m = _RE_EP.match(line)
        if m:
            t, total = float(m.group(1)), float(m.group(2))
            self.sim_time, self.sim_total = t, total
            if total > 0:
                self._set_progress(t / total, f"t = {t:g} / {total:g} ms")
            return
        m = _RE_LOAD.search(line)
        if m:
            lam = float(m.group(2))
            self._set_progress(lam, f"load step {m.group(1)} · λ = {lam:.3f}")
            return
        m = _RE_EM.match(line)
        if m:
            t = float(m.group(1))
            self.sim_time = t
            if self.sim_total:
                self._set_progress(t / self.sim_total, f"t = {t:.0f} / {self.sim_total:.0f} ms")
            return
        m = _RE_DYN.match(line)
        if m:
            t = float(m.group(1))
            self.sim_time = t
            if self.sim_total:
                self._set_progress(t / self.sim_total, f"t = {t:.4g} / {self.sim_total:.4g} {self.unit}")
            return
        m = _RE_LEAD.match(line)
        if m:
            self.leads_done.add(m.group(1))
            self._set_progress(len(self.leads_done) / len(self.LEADS), f"lead field {m.group(1)}")


def _last_error(log_path):
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 64 * 1024))
            tail = fh.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        m = _RE_EXC.match(line.strip())
        if m:
            return line.strip()[:500]
    lines = [ln for ln in tail if ln.strip()]
    return lines[-1][:500] if lines else None


def _dependency_active(dep_dir) -> str | None:
    """Name of a run we must wait for, or None once it has finished (in any way)."""
    status = _read_json(os.path.join(dep_dir, STATUS))
    if not status:
        return None
    status = _effective_status(dep_dir, status)
    if status.get("state") not in ("queued", "running"):
        return None
    manifest = _read_json(os.path.join(dep_dir, MANIFEST), {}) or {}
    return manifest.get("name") or os.path.basename(dep_dir)


def _wait_for_dependencies(run_dir, status) -> bool:
    """Hold a queued run until the runs listed in ``manifest["after"]`` have finished.

    This is how the studio queues several runs on one GPU: each waits for the previous
    one, entirely on the compute machine, so the queue survives closing the app.
    Returns False if the run was cancelled while waiting.
    """
    manifest = _read_json(os.path.join(run_dir, MANIFEST), {}) or {}
    deps = [d for d in manifest.get("after") or [] if d and os.path.isdir(d)]
    status_path = os.path.join(run_dir, STATUS)
    cancel_path = os.path.join(run_dir, CANCEL)
    while deps:
        if os.path.exists(cancel_path):
            status.update(state="cancelled", ended_at=time.time(), updated_at=time.time(),
                          message="Removed from the queue", pid=os.getpid())
            _write_json_atomic(status_path, status)
            return False
        waiting = [name for name in (_dependency_active(d) for d in deps) if name]
        if not waiting:
            break
        more = f" (+{len(waiting) - 1} more)" if len(waiting) > 1 else ""
        status.update(state="queued", pid=os.getpid(), host=socket.gethostname(), updated_at=time.time(),
                      stage="waiting", stage_label="In queue", message=f"Waiting for “{waiting[0]}”{more}")
        _write_json_atomic(status_path, status)
        time.sleep(3.0)
    return True


def _activated_path(path):
    """``path`` with this Python environment's script folder first, as activating the
    environment would, and the user scheme's (``pip install --user``) last.

    The studio starts an interpreter by its path, without activating its environment, so
    the tools installed beside it would be missed: TorchCor compiles the cell models it
    pre-paces with ``ninja``, which ``torch.utils.cpp_extension`` looks for on PATH.
    """
    import sysconfig

    parts = [p for p in path.split(os.pathsep) if p]
    seen = {os.path.normcase(p) for p in parts}

    def missing(*folders):
        found = []
        for folder in folders:
            if folder and os.path.normcase(folder) not in seen and os.path.isdir(folder):
                seen.add(os.path.normcase(folder))
                found.append(folder)
        return found

    try:
        user = sysconfig.get_path("scripts", sysconfig.get_preferred_scheme("user"))
    except (AttributeError, KeyError):  # Python < 3.10, or no user scheme
        user = None
    first = missing(sysconfig.get_path("scripts"), os.path.dirname(sys.executable))
    return os.pathsep.join(first + parts + missing(user))


def cmd_supervise(args) -> None:  # noqa: C901 - one linear state machine
    run_dir = os.path.abspath(args[0])
    status_path = os.path.join(run_dir, STATUS)
    log_path = os.path.join(run_dir, LOG)
    cancel_path = os.path.join(run_dir, CANCEL)
    status = _read_json(status_path, {}) or {}
    if not _wait_for_dependencies(run_dir, status):
        return
    status.update({
        "state": "running", "pid": os.getpid(), "host": socket.gethostname(),
        "started_at": time.time(), "updated_at": time.time(), "ended_at": None,
        "exit_code": None, "error": None, "stage": "start", "stage_label": "Starting",
        "message": "",
    })
    _write_json_atomic(status_path, status)

    env = dict(os.environ)
    env.update({
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",  # cuda:N == nvidia-smi index N
        "MPLBACKEND": "Agg",
        "TCS_RUN_DIR": run_dir,
        "PATH": _activated_path(os.environ.get("PATH", os.defpath)),
    })
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    progress = _Progress(os.path.join(run_dir, EVENTS))
    log = open(log_path, "ab")  # noqa: SIM115 - lives for the whole run
    popen_kwargs = {"cwd": run_dir, "env": env, "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = 0x00000200 | 0x08000000
    else:
        popen_kwargs["start_new_session"] = True
    try:
        child = subprocess.Popen([sys.executable, "-u", SCRIPT], **popen_kwargs)
    except OSError as exc:
        status.update(state="failed", error=f"could not start {SCRIPT}: {exc}", ended_at=time.time())
        _write_json_atomic(status_path, status)
        return
    status["child_pid"] = child.pid
    _write_json_atomic(status_path, status)

    import threading

    cancelled = threading.Event()
    lock = threading.Lock()

    def pump():
        partial = b""
        for chunk in iter(lambda: child.stdout.read1(65536) if hasattr(child.stdout, "read1")
                          else child.stdout.read(4096), b""):
            log.write(chunk)
            log.flush()
            partial += chunk
            *lines, partial = re.split(rb"\r\n|\n|\r", partial)
            with lock:
                for raw in lines:
                    progress.feed(raw.decode("utf-8", errors="replace"))
        if partial:
            with lock:
                progress.feed(partial.decode("utf-8", errors="replace"))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    def write_status():
        with lock:
            status.update({
                "updated_at": time.time(),
                "stage": progress.stage,
                "stage_label": progress.stage_label,
                "progress": progress.progress,
                "message": progress.message,
                "sim_time": progress.sim_time,
                "sim_total": progress.sim_total,
                "eta_s": progress.eta(),
                "metrics": progress.metrics,
            })
        _write_json_atomic(status_path, status)

    crash = None
    try:
        last = 0.0
        while child.poll() is None:
            if not cancelled.is_set() and os.path.exists(cancel_path):
                cancelled.set()
                with lock:
                    progress.message = "Stopping…"
                write_status()
                try:
                    kill_tree(child.pid, grace_s=10.0, alive=lambda: child.poll() is None)
                except Exception:  # noqa: BLE001 - never let cancellation kill the supervisor
                    traceback.print_exc()
            if time.time() - last >= 1.0:
                write_status()
                last = time.time()
            time.sleep(0.25)
    except BaseException as exc:  # noqa: BLE001 - always leave a final status behind
        crash = f"supervisor error: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        kill_tree(child.pid, grace_s=5.0, alive=lambda: child.poll() is None)
        child.wait(timeout=30)

    reader.join(timeout=10)
    log.close()
    code = child.returncode
    try:
        write_status()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    status["exit_code"] = code
    status["ended_at"] = time.time()
    if crash:
        status["state"] = "failed"
        status["error"] = crash
    elif cancelled.is_set():
        status["state"] = "cancelled"
        status["message"] = "Stopped by user"
    elif code == 0:
        status["state"] = "completed"
        status["progress"] = 1.0
        status["stage"], status["stage_label"] = "done", "Completed"
        status["eta_s"] = 0
    else:
        status["state"] = "failed"
        status["error"] = _last_error(log_path) or f"run.py exited with code {code}"
    _write_json_atomic(status_path, status)


# =============================================================================== misc
def cmd_cancel(args) -> None:
    run_dir = os.path.abspath(os.path.expanduser(args[0]))
    with open(os.path.join(run_dir, CANCEL), "w", encoding="utf-8") as fh:
        fh.write(str(time.time()))
    status = _read_json(os.path.join(run_dir, STATUS), {}) or {}
    supervisor_alive = pid_alive(status.get("pid"))
    if not supervisor_alive:
        kill_tree(status.get("child_pid"))
        if status.get("state") in ("queued", "running"):
            status.update(state="cancelled", ended_at=time.time(), message="Stopped by user")
            _write_json_atomic(os.path.join(run_dir, STATUS), status)
    _emit({"ok": True, "supervisor_alive": supervisor_alive})


def cmd_alive(args) -> None:
    _emit({"alive": pid_alive(int(args[0]))})


def cmd_kill(args) -> None:
    kill_tree(int(args[0]))
    _emit({"ok": True})


GPU_QUERY = ("index,uuid,name,utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,power.limit")


def cmd_gpus(_args) -> None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        _emit({"gpus": [], "error": "nvidia-smi not found"})
        return
    out = subprocess.run([exe, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20)
    rows = [r.split(", ") for r in out.stdout.strip().splitlines() if r.strip()]
    _emit({"gpus": rows, "error": out.stderr.strip() or None})


def cmd_remove(args) -> None:
    run_dir = os.path.abspath(os.path.expanduser(args[0]))
    if not os.path.isfile(os.path.join(run_dir, MANIFEST)):
        raise SystemExit(f"refusing to delete {run_dir}: it is not a TorchCor Lighting run")
    status = _read_json(os.path.join(run_dir, STATUS), {}) or {}
    if status.get("state") in ("queued", "running") and pid_alive(status.get("pid")):
        raise SystemExit("the run is still active; stop it first")
    shutil.rmtree(run_dir)
    _emit({"ok": True})


# =============================================================================== preview
def _read_mesh_arrays(path):
    """Points (mm), tetrahedra, triangles and their region tags of the mesh in a folder."""
    import numpy as np

    info = inspect_dir(path, limit_bytes=1)
    if not info.get("mesh_name"):
        raise SystemExit(f"no mesh (.pts + .elem) in {path}")
    base = os.path.join(path, info["mesh_name"])
    with open(base + ".pts", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        points = np.array(fh.read().split(), dtype=np.float64).reshape(-1, 3) / 1000.0
    tets, tet_reg, tris, tri_reg = [], [], [], []
    with open(base + ".elem", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Tt" and len(parts) >= 6:
                tets.append(parts[1:5])
                tet_reg.append(parts[5])
            elif parts[0] == "Tr" and len(parts) >= 5:
                tris.append(parts[1:4])
                tri_reg.append(parts[4])
    as_int = lambda rows: np.array(rows, dtype=np.int64) if rows else None  # noqa: E731
    return info, points, as_int(tets), as_int(tet_reg), as_int(tris), as_int(tri_reg)


def cmd_preview(args) -> None:
    """``preview <mesh_dir> <out.npz>`` — boundary surface + pacing sites for a quick 3D look."""
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tcs_studio  # deployed next to this file

    mesh_dir = os.path.abspath(os.path.expanduser(args[0]))
    out = os.path.abspath(os.path.expanduser(args[1]))
    info, points, tets, tet_reg, tris, tri_reg = _read_mesh_arrays(mesh_dir)
    pts, triangles, node_ids, regions = tcs_studio.surface_of(points, tets=tets, triangles=tris,
                                                              tet_regions=tet_reg, tri_regions=tri_reg)
    names, offsets, site_points = [], [0], []
    for vtx in info.get("vtx", []):
        with open(os.path.join(mesh_dir, vtx["name"]), encoding="utf-8", errors="replace") as fh:
            lines = fh.read().split()
        ids = np.array(lines[2:], dtype=np.int64) if len(lines) > 2 else np.zeros(0, np.int64)
        ids = ids[(ids >= 0) & (ids < len(points))]
        names.append(vtx["name"])
        site_points.append(points[ids].astype(np.float32))
        offsets.append(offsets[-1] + len(ids))
    np.savez_compressed(
        out, points=pts, triangles=triangles, node_ids=node_ids,
        regions=regions if regions is not None else np.zeros(len(triangles), np.int32),
        vtx_names=np.array(names), vtx_offsets=np.array(offsets, dtype=np.int64),
        vtx_points=np.concatenate(site_points) if site_points else np.zeros((0, 3), np.float32),
    )
    _emit({"path": out, "points": len(pts), "triangles": len(triangles),
           "sites": len(names), "bytes": os.path.getsize(out)})


# =============================================================================== demo data
#: The GitHub repository whose releases tagged ``data-v<version>`` hold the demo data zip.
DATA_REPO = "sagebei/torchcor_lighting"
DATA_FOLDER = "torchcor_data"  # the zip's one top-level folder
DATA_VERSION_FILE = ".version"  # in it, once unzipped: {"version": "data-v1.0", "repo": ...}


def _http(url, accept="application/json"):
    import urllib.request

    request = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "torchcor-lighting"})
    return urllib.request.urlopen(request, timeout=60)  # noqa: S310 - https URLs from GitHub


def _data_version(tag):
    """``data-v1.10`` -> ``(1, 10)``; ``()`` for a tag that is not a data release."""
    return tuple(int(n) for n in re.findall(r"\d+", tag[len("data-v"):])) if tag.startswith("data-v") else ()


def latest_data_release(repo=DATA_REPO):
    """``(tag, url, size, sha256)`` of the zip of ``repo``'s newest ``data-v*`` release
    (``sha256`` is "" when GitHub gives no digest)."""
    import urllib.error

    try:
        with _http(f"https://api.github.com/repos/{repo}/releases?per_page=100") as resp:
            releases = json.load(resp)
    except urllib.error.HTTPError as exc:
        hint = "; the repository must be public" if exc.code == 404 else ""
        raise SystemExit(f"could not list the releases of {repo} (HTTP {exc.code}{hint})") from exc
    except OSError as exc:
        raise SystemExit(f"could not reach GitHub: {exc}") from exc
    found = [r for r in releases if not r.get("draft") and _data_version(str(r.get("tag_name", "")))]
    if not found:
        raise SystemExit(f"{repo} has no data-v* release")
    release = max(found, key=lambda r: _data_version(r["tag_name"]))
    zips = [a for a in release.get("assets", []) if str(a.get("name", "")).endswith(".zip")]
    if not zips:
        raise SystemExit(f"the release {release['tag_name']} has no .zip")
    asset = next((a for a in zips if a["name"] == f"{DATA_FOLDER}.zip"), zips[0])
    digest = str(asset.get("digest") or "")
    return (release["tag_name"], asset["browser_download_url"], int(asset.get("size") or 0),
            digest[len("sha256:"):] if digest.startswith("sha256:") else "")


def _progress(stage, done, total, version):
    print(json.dumps({"progress": stage, "done": done, "total": total, "version": version}), flush=True)


def _unzip(path, dest, version):
    """Unzip ``path`` into ``dest``: one ``torchcor_data`` folder, no path leaving it, and
    room for it; files already there are replaced, others (run outputs) kept."""
    import zipfile

    root = os.path.realpath(dest)
    folder = os.path.join(root, DATA_FOLDER)
    with zipfile.ZipFile(path) as zf:
        members = zf.infolist()
        for member in members:
            target = os.path.realpath(os.path.join(root, member.filename))
            if target != folder and not target.startswith(folder + os.sep):
                raise SystemExit(f"unexpected path in the zip: {member.filename}")
        total = sum(m.file_size for m in members)
        free = shutil.disk_usage(root).free
        if free < total:
            raise SystemExit(f"not enough free space in {dest} to unzip: {total / 1e9:.1f} GB needed, "
                             f"{free / 1e9:.1f} GB free")
        done, shown = 0, -1
        for member in members:
            zf.extract(member, root)
            done += member.file_size
            percent = int(100 * done / total) if total else 100
            if percent != shown:
                shown = percent
                _progress("unzip", done, total, version)


def cmd_fetch_data(args) -> None:
    """``fetch-data <dest> [<repo>]`` — download the newest demo data release into <dest>,
    check its SHA-256, unzip it there (``<dest>/torchcor_data``) and delete the zip."""
    import hashlib

    dest = os.path.abspath(os.path.expanduser(args[0]))
    repo = args[1] if len(args) > 1 else DATA_REPO
    os.makedirs(dest, exist_ok=True)
    version, url, size, sha256 = latest_data_release(repo)
    free = shutil.disk_usage(dest).free
    if size and free < size:
        raise SystemExit(f"not enough free space in {dest}: {size / 1e9:.1f} GB to download, "
                         f"{free / 1e9:.1f} GB free")
    part = os.path.join(dest, f"{DATA_FOLDER}.zip.part")
    digest, done = hashlib.sha256(), 0
    _progress("download", 0, size, version)
    with _http(url, accept="application/octet-stream") as resp, open(part, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or size or 0)
        shown = -1
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            percent = int(100 * done / total) if total else -1
            if percent != shown:
                shown = percent
                _progress("download", done, total, version)
    if (sha256 and digest.hexdigest() != sha256) or (size and done != size):
        os.remove(part)
        raise SystemExit("the download is damaged (its SHA-256 does not match the release's); try again")
    archive = part[: -len(".part")]
    os.replace(part, archive)
    try:
        _unzip(archive, dest, version)
    finally:
        os.remove(archive)
    folder = os.path.join(dest, DATA_FOLDER)
    with open(os.path.join(folder, DATA_VERSION_FILE), "w", encoding="utf-8") as fh:
        json.dump({"version": version, "repo": repo}, fh)
    _emit({"ok": True, "version": version, "path": folder, "bytes": done})


COMMANDS = {
    "fetch-data": cmd_fetch_data,
    "probe": cmd_probe,
    "inspect": cmd_inspect,
    "scan": cmd_scan,
    "launch": cmd_launch,
    "supervise": cmd_supervise,
    "cancel": cmd_cancel,
    "alive": cmd_alive,
    "kill": cmd_kill,
    "gpus": cmd_gpus,
    "remove": cmd_remove,
    "preview": cmd_preview,
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        sys.stderr.write(__doc__ or "")
        return 2
    try:
        COMMANDS[argv[0]](argv[1:])
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - report every failure as JSON
        traceback.print_exc()
        _emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
