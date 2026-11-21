"""A machine whose clock is off: its run and file times are shown on this computer's clock.

Seen on a GPU server whose clock was 3.5 min behind: a run that had just started showed
"3m 32s" elapsed, because the start time (the server's clock) was compared with now (here).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from torchcor_studio.core.models import MachineKind, MachineProfile, RunState
from torchcor_studio.core.runs import LaunchRequest, RunService, record_from_scan
from torchcor_studio.core.runs.service import status_from_machine

BEHIND = 211.8  # seconds the machine's clock is behind this computer's (its clock_offset)


def test_status_times_are_put_on_this_computers_clock() -> None:
    there = time.time() - BEHIND  # "now" on the machine
    data = {"state": "running", "started_at": there - 5, "updated_at": there, "eta_s": 60}
    status = status_from_machine(data, BEHIND)
    assert 4.5 <= status.elapsed_s <= 6  # not 3.5 min
    assert 59 <= status.remaining_s <= 60
    assert status_from_machine(data).started_at == data["started_at"]  # same clock: as written
    record = record_from_scan("server", {"run_dir": "/runs/a", "manifest": {"id": "a"}, "status": data}, BEHIND)
    assert 4.5 <= record.status.elapsed_s <= 6
    done = status_from_machine({**data, "state": "completed", "ended_at": there}, BEHIND)
    assert done.elapsed_s == pytest.approx(5)  # durations are the same on either clock


def test_the_remaining_time_counts_down_between_updates() -> None:
    status = status_from_machine({"state": "running", "started_at": time.time() - 40,
                                  "updated_at": time.time() - 30, "eta_s": 100})
    assert 69 <= status.remaining_s <= 70.5
    assert status_from_machine({"state": "completed", "eta_s": 0}).remaining_s == 0


def test_probing_measures_the_clock(monkeypatch) -> None:  # noqa: ANN001
    from torchcor_studio.core.machines.local import LocalMachine
    from torchcor_studio.core.machines.ssh import SSHMachine

    server = SSHMachine(MachineProfile(name="server", kind=MachineKind.SSH, host="gpu-server"))
    assert server.clock_offset == 0.0
    monkeypatch.setattr(server, "runtime", lambda *_a, **_k: {"python": "/usr/bin/python3",
                                                              "clock": time.time() - BEHIND})
    assert server.probe_environment().python == "/usr/bin/python3"
    assert server.clock_offset == pytest.approx(BEHIND, abs=0.5)
    local = LocalMachine(MachineProfile(name="here", python=sys.executable))
    monkeypatch.setattr(local, "runtime", lambda *_a, **_k: {"clock": time.time() - BEHIND})
    local.probe_environment()
    assert local.clock_offset == 0.0  # one clock


def test_runs_on_a_machine_whose_clock_is_behind(tmp_path: Path) -> None:
    from tests.test_run_service import SPEC, FakeParams

    from torchcor_studio.core.machines.local import LocalMachine

    machine = LocalMachine(MachineProfile(name="Local", kind=MachineKind.LOCAL, python=sys.executable))
    machine.connect()
    machine.clock_offset = BEHIND  # as if measured on a server
    service = RunService()
    record = service.create(machine, LaunchRequest(spec=SPEC, params=FakeParams(samples=80), name="Skewed",
                                                   input_dir="", output_dir=str(tmp_path / "runs"),
                                                   device="cpu", device_name="CPU"))
    status_file = Path(record.run_dir) / "status.json"
    written = json.loads(status_file.read_text())
    assert written["updated_at"] == pytest.approx(record.status.updated_at - BEHIND)  # its clock

    there = time.time() - BEHIND  # what its runtime writes while the run goes on
    status_file.write_text(json.dumps({"state": "running", "started_at": there - 5, "updated_at": there}))
    status = service.read_status(machine, record)
    assert status.state == RunState.RUNNING  # it reported just now: not lost
    assert 4.5 <= status.elapsed_s <= 6
