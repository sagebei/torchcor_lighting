"""Create, launch, observe and clean up runs on any :class:`Machine`.

All methods are blocking and thread-safe; the UI calls them from worker threads.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from torchcor_studio.core.machines.base import Machine, MachineError
from torchcor_studio.core.models import RunRecord, RunState, RunStatus
from torchcor_studio.core.paths import run_cache_dir
from torchcor_studio.core.runs import protocol as P
from torchcor_studio.core.simulations.base import ScriptContext, SimulationSpec
from torchcor_studio.runtime import RUNTIME_FILES, runtime_path


@dataclass(frozen=True)
class LaunchRequest:
    spec: SimulationSpec
    params: BaseModel
    name: str
    input_dir: str
    output_dir: str
    device: str
    device_name: str
    torchcor_version: str | None = None
    #: Run folders (same machine) that must finish before this run starts — a GPU queue.
    after: tuple[str, ...] = ()
    #: Sweep provenance shown in the UI, e.g. {"parameter": "Timing › Duration", "value": 400}.
    sweep: dict | None = None
    #: The project the run is filed in (``Project.ref()``), if any.
    project: dict | None = None


def status_from_machine(data: dict[str, Any], clock_offset: float = 0.0) -> RunStatus:
    """A run's ``status.json``, whose times were written by its machine's clock, with the
    times on this computer's clock (``Machine.clock_offset``)."""
    status = RunStatus.from_dict(data)
    if clock_offset:
        for name in ("started_at", "updated_at", "ended_at"):
            if (value := getattr(status, name)) is not None:
                setattr(status, name, value + clock_offset)
    return status


def record_from_scan(machine_id: str, item: dict[str, Any], clock_offset: float = 0.0) -> RunRecord:
    """Build a :class:`RunRecord` from one entry of the runtime ``scan`` command."""
    manifest = item.get("manifest") or {}
    status = status_from_machine(item.get("status") or {}, clock_offset)
    run_dir = item.get("run_dir", "")
    return RunRecord(
        id=manifest.get("id") or run_dir.rstrip("/\\").split("/")[-1],
        name=manifest.get("name") or manifest.get("id") or "Untitled run",
        kind=manifest.get("kind", "ep"),
        machine_id=machine_id,
        run_dir=run_dir,
        created_at=float(manifest.get("created_at") or 0.0),
        status=status,
        device=manifest.get("device", "cpu"),
        device_name=manifest.get("device_name", ""),
        variant=manifest.get("variant", ""),
        manifest=manifest,
    )


class RunService:
    """Stateless operations on run folders."""

    # ------------------------------------------------------------------ create & launch
    def build_script(self, machine: Machine, request: LaunchRequest, run_dir: str, run_id: str) -> str:
        ctx = ScriptContext(
            device=request.device,
            input_dir=request.input_dir,
            run_dir=run_dir,
            run_id=run_id,
            run_name=request.name,
            machine_name=machine.name,
            windows=self._is_windows(machine),
        )
        return request.spec.build_script(request.params, ctx)

    def create(self, machine: Machine, request: LaunchRequest) -> RunRecord:
        """Write a complete, runnable run folder (does not start it)."""
        run_id = P.new_run_id(request.name)
        output_dir = machine.expand(request.output_dir)
        run_dir = machine.join(output_dir, run_id)
        machine.mkdir(run_dir)
        script = self.build_script(machine, request, run_dir, run_id)
        machine.write_text(machine.join(run_dir, P.SCRIPT), script)
        for name in RUNTIME_FILES:
            machine.write_bytes(machine.join(run_dir, name), runtime_path(name).read_bytes())
        params = request.params.model_dump(mode="json")
        manifest = P.build_manifest(
            run_id=run_id,
            name=request.name,
            kind=str(request.spec.kind),
            variant=request.spec.variant_of(request.params),
            params=params,
            machine={"id": machine.id, "name": machine.name, "kind": str(machine.profile.kind),
                     "address": machine.profile.address},
            device=request.device,
            device_name=request.device_name,
            python=machine.python,
            input_dir=request.input_dir,
            output_dir=output_dir,
            torchcor_version=request.torchcor_version,
        )
        if request.after:
            manifest["after"] = list(request.after)
        if request.sweep:
            manifest["sweep"] = request.sweep
        if request.project:
            manifest["project"] = dict(request.project)
        machine.write_json(machine.join(run_dir, P.MANIFEST), manifest)
        status = RunStatus(state=RunState.QUEUED, updated_at=time.time(), stage="queued",
                           stage_label="Queued")
        on_machine = {**status.to_dict(), "updated_at": status.updated_at - machine.clock_offset}
        machine.write_json(machine.join(run_dir, P.STATUS), on_machine)  # the machine's clock, as the runtime's
        return RunRecord(
            id=run_id,
            name=request.name,
            kind=str(request.spec.kind),
            machine_id=machine.id,
            run_dir=run_dir,
            created_at=manifest["created_at"],
            status=status,
            device=request.device,
            device_name=request.device_name,
            variant=manifest["variant"],
            manifest=manifest,
        )

    def launch(self, machine: Machine, record: RunRecord) -> RunRecord:
        script = machine.join(record.run_dir, "tcs_runtime.py")
        result = machine.run([machine.python, script, "launch", record.run_dir], timeout=60)
        if not result.ok:
            raise MachineError("could not start the run: " + (result.stderr or result.stdout).strip()[-600:])
        try:
            info = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise MachineError(f"unexpected launch output: {result.stdout[-300:]}") from exc
        record.status.pid = info.get("pid")
        record.status.host = info.get("host", "")
        return record

    # ------------------------------------------------------------------ observe
    def read_status(self, machine: Machine, record: RunRecord) -> RunStatus:
        try:
            data = machine.read_json(machine.join(record.run_dir, P.STATUS))
        except (MachineError, OSError, ValueError):
            return record.status
        status = status_from_machine(data, machine.clock_offset)
        if status.state.is_active and status.updated_at and time.time() - status.updated_at > P.STALE_AFTER_S:
            pid = status.pid
            alive = machine.is_alive(pid) if pid else False
            if not alive:
                status.state = RunState.LOST
                status.error = status.error or "The run stopped reporting (process ended unexpectedly)."
        return status

    def read_log(self, machine: Machine, record: RunRecord, offset: int = 0,
                 max_bytes: int = 2_000_000) -> tuple[str, int]:
        """New log text since ``offset`` and the new offset."""
        path = machine.join(record.run_dir, P.LOG)
        try:
            size = machine.size(path)
        except (MachineError, OSError):
            return "", offset
        if size < offset:  # log was truncated / replaced
            offset = 0
        if size == offset:
            return "", offset
        start = max(offset, size - max_bytes) if offset == 0 else offset
        data = machine.read_bytes(path, start, size - start)
        return data.decode("utf-8", errors="replace"), start + len(data)

    def read_script(self, machine: Machine, record: RunRecord) -> str:
        return machine.read_text(machine.join(record.run_dir, P.SCRIPT))

    def read_manifest(self, machine: Machine, record: RunRecord) -> dict[str, Any]:
        return machine.read_json(machine.join(record.run_dir, P.MANIFEST))

    def list_files(self, machine: Machine, record: RunRecord, max_depth: int = 2) -> list[tuple[str, int, bool]]:
        """(relative path, size, is_dir) of the run folder, depth-limited."""
        out: list[tuple[str, int, bool]] = []

        def walk(path: str, rel: str, depth: int) -> None:
            for entry in machine.listdir(path):
                rel_path = f"{rel}{entry.name}"
                out.append((rel_path, entry.size, entry.is_dir))
                if entry.is_dir and depth < max_depth:
                    walk(entry.path, rel_path + "/", depth + 1)

        walk(record.run_dir, "", 1)
        return out

    # ------------------------------------------------------------------ control
    def cancel(self, machine: Machine, record: RunRecord) -> None:
        machine.write_text(machine.join(record.run_dir, P.CANCEL), str(time.time()))
        script = machine.join(record.run_dir, "tcs_runtime.py")
        machine.run([machine.python, script, "cancel", record.run_dir], timeout=60)

    def delete(self, machine: Machine, record: RunRecord) -> None:
        script = machine.join(record.run_dir, "tcs_runtime.py")
        if machine.exists(script):
            result = machine.run([machine.python, script, "remove", record.run_dir], timeout=120)
            if not result.ok:
                raise MachineError((result.stderr or result.stdout).strip().splitlines()[-1])
        elif machine.exists(machine.join(record.run_dir, P.MANIFEST)):
            machine.remove(record.run_dir, recursive=True)

    # ------------------------------------------------------------------ discovery
    def scan(self, machine: Machine, roots: list[str]) -> list[RunRecord]:
        roots = [r for r in dict.fromkeys(roots) if r]
        if not roots:
            return []
        data = machine.runtime("scan", *roots, timeout=120)
        return [record_from_scan(machine.id, item, machine.clock_offset) for item in data.get("runs", [])]

    # ------------------------------------------------------------------ results
    def viz_dir(self, machine: Machine, record: RunRecord) -> str:
        return machine.join(record.run_dir, P.VIZ_DIR)

    def has_viz(self, machine: Machine, record: RunRecord) -> bool:
        return machine.exists(machine.join(self.viz_dir(machine, record), "index.json"))

    def fetch_viz(self, machine: Machine, record: RunRecord, progress=None) -> Path:
        """Local folder holding the run's visualisation bundle (downloaded if remote)."""
        remote = self.viz_dir(machine, record)
        if machine.profile.is_local:
            return Path(remote)
        local = run_cache_dir(machine.id, record.id) / P.VIZ_DIR
        local.mkdir(parents=True, exist_ok=True)
        entries = [e for e in machine.listdir(remote) if not e.is_dir]
        total = sum(e.size for e in entries) or 1
        done = 0
        for entry in sorted(entries, key=lambda e: e.name != "index.json"):
            target = local / entry.name
            if target.exists() and target.stat().st_size == entry.size and entry.name != "index.json":
                done += entry.size
                continue
            base = done

            def report(sent: int, _total: int, base: int = base) -> None:
                if progress:
                    progress((base + sent) / total)

            machine.download(entry.path, str(target), report)
            done += entry.size
        if progress:
            progress(1.0)
        return local

    @staticmethod
    def _is_windows(machine: Machine) -> bool:
        if not machine.profile.is_local:
            return False
        import os

        return os.name == "nt"
