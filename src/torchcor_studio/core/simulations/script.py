"""Custom script: run your own TorchCor code with the studio's machines, GPUs, logs and history."""

from __future__ import annotations

from torchcor_studio.core.models import InputInfo
from torchcor_studio.core.simulations.base import (
    Cost,
    Demo,
    InputUse,
    Issue,
    ScriptContext,
    Severity,
    SimulationKind,
    SimulationSpec,
    Stage,
    register,
)
from torchcor_studio.core.simulations.fields import Section, param
from torchcor_studio.core.simulations.panels import INPUT, OUTPUT, Panel

TEMPLATE = '''\
"""My TorchCor experiment.

The studio defines, above this script:
    DEVICE     the device picked at launch ("cuda:0", "cpu", ...)
    INPUT_DIR  the input folder chosen in the Setup step ("" when none)
    RESULTS    this run's results folder (created for you)
and ships `tcs_studio` (imported below as `studio`) for live progress and for the
visualisation bundle shown in the Results tab. Everything else is plain TorchCor.
"""
import torchcor as tc
from torchcor.electrophysiology import Monodomain
from torchcor.ionic import TenTusscherPanfilov

import tcs_studio as studio

studio.stage("setup", "Setting up")
tc.set_device(DEVICE)
dt = 0.01                                     # ms

# A 10 x 4 x 2 mm slab generated on the fly (swap in your own mesh folder).
mesh_dir = RESULTS / "mesh"
studio.make_slab_mesh(mesh_dir, lengths=(10.0, 4.0, 2.0), dx=0.5)

cell = TenTusscherPanfilov(dt=dt, cell_type="EPI", dtype=tc.float32)
sim = Monodomain(ionic_models=[cell], T=60.0, dt=dt, dtype=tc.float32)
sim.load_mesh(mesh_dir, unit_conversion=1000)
sim.add_conductivity(None, il=0.17, it=0.019, el=0.62, et=0.24)
sim.add_stimulus(mesh_dir / "stimulus.vtx", start=0.0, duration=2.0, intensity=50.0)

studio.stage("solve", "Solving")
Vm = sim.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100, snapshot_interval=1.0,
               verbose=True, result_path=RESULTS)
sim.save_vm(Vm)

studio.stage("post", "Activation map")
ATs = sim.compute_activation_map(Vm=Vm, snapshot_interval=1.0, threshold=0.0)

studio.stage("export", "Visualisation")
studio.export_ep(sim, Vm, snapshot_interval=1.0, ATs=ATs)
studio.stage("done", "Completed")
'''


class ScriptParams(Section):
    code: str = param(TEMPLATE, "Script", widget="code",
                      help="Python that runs on the compute machine. DEVICE, INPUT_DIR and RESULTS are "
                           "predefined; use tcs_studio for progress and visualisation.")

    def variant_label(self) -> str:
        first = next((ln.strip().strip('"').strip() for ln in self.code.splitlines() if ln.strip()), "")
        return first[:40] if first and not first.startswith(("import", "from", "#")) else "Custom script"


DEMOS = (
    Demo("starter", "Starter script",
         "A 10 × 4 × 2 mm slab in plain TorchCor, with tcs_studio progress.", ScriptParams,
         cost=Cost.SECONDS),
)

PANELS = (Panel("Script", "code", ("code", INPUT, OUTPUT)),)  # the folders after the code


def input_use(_params: ScriptParams) -> tuple[InputUse, str]:
    return "optional", "Your script reads it as INPUT_DIR (empty when none is chosen)."


def build_script(params: ScriptParams, ctx: ScriptContext) -> str:
    prelude = (
        f"# ── TorchCor Lighting · custom script · {ctx.run_name or ctx.run_id}\n"
        "# Re-run with: python run.py\n"
        "from pathlib import Path as _Path\n\n"
        f"DEVICE = {ctx.device!r}\n"
        f"INPUT_DIR = {ctx.input_dir!r}\n"
        'RESULTS = _Path(__file__).resolve().parent / "results"\n'
        "RESULTS.mkdir(parents=True, exist_ok=True)\n"
        "# ── your script ──────────────────────────────────────────────────────────────\n\n"
    )
    return prelude + params.code.rstrip() + "\n"


def check(params: ScriptParams, _info: InputInfo | None) -> list[Issue]:
    try:
        compile(params.code, "run.py", "exec")
    except SyntaxError as exc:
        return [Issue(Severity.ERROR, f"Syntax error on line {exc.lineno}: {exc.msg}", "code")]
    issues = []
    if "tcs_studio" not in params.code:
        issues.append(Issue(Severity.INFO, "The script doesn't use tcs_studio, so the Results tab will be empty "
                                           "and progress comes only from TorchCor's own output.", "code"))
    return issues


SPEC = register(SimulationSpec(
    kind=SimulationKind.SCRIPT,
    title="Custom script",
    tagline="Your own TorchCor code",
    description="Write or paste Python that uses TorchCor. It runs on the machine and GPU you choose, "
                "with live logs, history and — via tcs_studio — progress and visualisation.",
    icon="code",
    params_model=ScriptParams,
    build_script=build_script,
    stages=(Stage("setup", "Setup"), Stage("solve", "Solve"), Stage("post", "Post-process"),
            Stage("export", "Export")),
    demos=DEMOS,
    panels=PANELS,
    check=check,
    input_use=input_use,
    badge="PY",
))
