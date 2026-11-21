"""RunService end to end on this computer: create → launch → observe → discover → delete."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from torchcor_studio.core.models import MachineKind, MachineProfile, RunState
from torchcor_studio.core.runs import LaunchRequest, RunService
from torchcor_studio.core.simulations import (
    ScriptContext,
    Section,
    SimulationKind,
    SimulationSpec,
    Stage,
    param,
)

SCRIPT = '''
import numpy as np
import tcs_studio as studio

studio.stage("solve", "Solving")
for i in range(1, 5):
    print(f"t: {{i * 25:.1f}}/100 | Time elapsed: 0.1 | CG iter: 2", flush=True)
studio.stage("export", "Writing visualisation bundle")
bundle = studio.Bundle("ecg", "Synthetic")
t = np.arange(0, {n}, 1.0, dtype=np.float32)
bundle.add("ecg", t=t, leads=np.zeros((12, t.size), np.float32), lead_names=np.array(
    ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]))
bundle.stat("Samples", t.size)
bundle.write()
studio.stage("done", "Completed")
'''


class FakeParams(Section):
    samples: int = param(50, "Samples", min=1)

    def variant_label(self) -> str:
        return "Synthetic"


def _build(params: FakeParams, ctx: ScriptContext) -> str:
    assert ctx.device == "cpu"
    return SCRIPT.format(n=params.samples)


SPEC = SimulationSpec(
    kind=SimulationKind.ECG, title="Fake", tagline="", description="", icon="ecg",
    params_model=FakeParams, build_script=_build,
    stages=(Stage("solve", "Solve"), Stage("export", "Export")),
)


def _machine():
    from torchcor_studio.core.machines.local import LocalMachine

    machine = LocalMachine(MachineProfile(name="Local", kind=MachineKind.LOCAL, python=sys.executable))
    machine.connect()
    return machine


def test_run_lifecycle(tmp_path: Path) -> None:
    machine = _machine()
    service = RunService()
    request = LaunchRequest(spec=SPEC, params=FakeParams(samples=80), name="Synthetic ECG",
                            input_dir="", output_dir=str(tmp_path / "runs"), device="cpu",
                            device_name="CPU")
    record = service.create(machine, request)
    assert record.status.state == RunState.QUEUED
    run_dir = Path(record.run_dir)
    for name in ("run.py", "tcs_runtime.py", "tcs_studio.py", "manifest.json", "status.json"):
        assert (run_dir / name).exists(), name
    assert record.manifest["params"] == {"samples": 80}
    assert record.variant == "Synthetic"

    service.launch(machine, record)
    deadline = time.time() + 120
    while True:
        status = service.read_status(machine, record)
        if status.state.is_terminal:
            break
        assert time.time() < deadline, status
        time.sleep(0.25)
    log, offset = service.read_log(machine, record)
    assert status.state == RunState.COMPLETED, (status.error, log)
    assert "t: 100.0/100" in log and offset > 0
    more, offset2 = service.read_log(machine, record, offset)
    assert more == "" and offset2 == offset

    assert service.has_viz(machine, record)
    viz = service.fetch_viz(machine, record)
    assert (viz / "index.json").exists() and (viz / "ecg.npz").exists()

    found = service.scan(machine, [str(tmp_path / "runs")])
    assert [r.id for r in found] == [record.id]
    assert found[0].status.state == RunState.COMPLETED
    assert found[0].name == "Synthetic ECG"

    files = {rel for rel, _size, _is_dir in service.list_files(machine, record)}
    assert {"run.py", "run.log", "viz", "viz/index.json"} <= files

    service.delete(machine, record)
    assert not run_dir.exists()


@pytest.mark.slow
def test_custom_script_template_runs(tmp_path: Path) -> None:
    """The starter template of the "Custom script" kind is a real, working TorchCor run."""
    import importlib.util

    if importlib.util.find_spec("torchcor") is None:
        pytest.skip("TorchCor is not installed")
    from torchcor_studio.core.simulations import get_spec

    spec = get_spec("script")
    params = spec.params_model()
    params = params.model_copy(update={"code": params.code.replace("T=60.0", "T=12.0")})
    assert not [i for i in spec.validate(params, None) if i.severity == "error"]
    machine = _machine()
    service = RunService()
    record = service.create(machine, LaunchRequest(spec=spec, params=params, name="Template", input_dir="",
                                                   output_dir=str(tmp_path / "runs"), device="cpu",
                                                   device_name="CPU"))
    service.launch(machine, record)
    deadline = time.time() + 600
    while not (status := service.read_status(machine, record)).state.is_terminal:
        assert time.time() < deadline
        time.sleep(0.5)
    log, _ = service.read_log(machine, record)
    assert status.state == RunState.COMPLETED, (status.error, log[-2000:])
    viz = service.fetch_viz(machine, record)
    assert (viz / "ep.npz").exists() and (viz / "surface.npz").exists()
