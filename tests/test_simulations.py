"""Simulation specs: parameter trees, demos, generated scripts, checks and real runs.

The slow tests run small TorchCor simulations on the CPU through the real runtime
supervisor (``tcs_runtime.py launch``), exactly as the app does on a compute machine.
"""

from __future__ import annotations

import ast
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from torchcor_studio.core.models import InputInfo, VtxInfo
from torchcor_studio.core.simulations import ecg, em, ep, mechanics
from torchcor_studio.core.simulations.base import (
    ScriptContext,
    Severity,
    SimulationKind,
    get_spec,
    specs,
)
from torchcor_studio.core.simulations.fields import Section, field_meta
from torchcor_studio.core.simulations.panels import INPUT
from torchcor_studio.runtime import runtime_path

OURS = (SimulationKind.EP, SimulationKind.ECG, SimulationKind.MECHANICS, SimulationKind.EM)
def _installed(name: str) -> bool:
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


#: The generated scripts and the slow runs execute in their own interpreter.
HAS_TORCHCOR = _installed("torchcor")
ICONS = {"zap", "ecg", "box", "heart-pulse"}


def _ctx(**kw) -> ScriptContext:
    base = dict(device="cuda:0", input_dir="/data/heart", run_dir="/runs/abc", run_id="abc",
                run_name="Test run", machine_name="gpu-box")
    base.update(kw)
    return ScriptContext(**base)


def _errors(issues):
    return [i for i in issues if i.severity == Severity.ERROR]


def _cases():
    for spec in specs():
        for demo in spec.demos:
            yield spec, demo.key, demo.params


CASES = list(_cases())


def _sections(model: BaseModel, path=""):
    """Every (dotted path, section) of a parameter tree, rows of tables included."""
    yield path, model
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, Section):
            yield from _sections(value, f"{path}{name}.")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, Section):
                    yield from _sections(item, f"{path}{name}.{i}.")


# =============================================================================== registry
def test_specs_are_registered_with_their_metadata():
    kinds = [s.kind for s in specs()]
    assert kinds[:4] == list(OURS)
    for kind in OURS:
        spec = get_spec(kind)
        assert spec.title and spec.tagline and spec.description
        assert spec.icon in ICONS
        assert spec.badge in {"EP", "ECG", "MECH", "EM"}
        assert spec.demos and spec.panels
        assert not any("RTX" in d.summary for d in spec.demos)  # no hardware-specific timings
        assert spec.stages[-1].id == "done"
        assert len({s.id for s in spec.stages}) == len(spec.stages)
    assert get_spec("ep").input_role(ep.EPParams())[0] == "required"
    assert get_spec("mechanics").input_role(mechanics.MechanicsParams())[0] == "unused"
    assert get_spec("ecg").input_role(ecg.ECGParams())[0] == "required"
    assert get_spec("em").input_role(em.EMParams())[0] == "required"


@pytest.mark.parametrize("kind", OURS)
def test_parameter_trees_have_ui_metadata(kind):
    spec = get_spec(kind)
    params = spec.defaults()
    for path, section in _sections(params):
        for name in type(section).model_fields:
            meta = field_meta(section, name)
            assert meta["title"], f"{path}{name} has no title"
            assert meta["title"][0].isupper() or not meta["title"][0].isalpha(), (
                f"{path}{name}: '{meta['title']}' is not in sentence case")
            for key in ("choices",):
                if key in meta:
                    value = getattr(section, name)
                    assert value in meta[key], f"{path}{name}: default not among its choices"


@pytest.mark.parametrize(("spec", "name", "factory"), CASES,
                         ids=[f"{s.kind}-{n}" for s, n, _ in CASES])
def test_every_preset_builds_a_script_that_compiles(spec, name, factory):
    params = factory()
    assert isinstance(params, spec.params_model)
    again = spec.params_model.model_validate(json.loads(params.model_dump_json()))
    assert again == params, "parameters must survive a JSON round trip"
    for ctx in (_ctx(), _ctx(device="cpu", input_dir="", machine_name=""),
                _ctx(input_dir=r"C:\data\heart case", windows=True)):
        source = spec.build_script(params, ctx)
        compile(source, "run.py", "exec")
    assert isinstance(spec.variant_of(params), str)
    if spec.kind in OURS:
        assert spec.variant_of(params)
        spec.validate(params, None)  # must not raise


@pytest.mark.parametrize(("spec", "name", "factory"), [c for c in CASES if c[0].kind in OURS],
                         ids=[f"{s.kind}-{n}" for s, n, _ in CASES if s.kind in OURS])
def test_scripts_follow_the_studio_conventions(spec, name, factory):
    source = spec.build_script(factory(), _ctx(device="cuda:3"))
    assert source.startswith('"""Test run — TorchCor Lighting run')
    assert "Re-run with:  python run.py" in source
    assert "import tcs_studio as studio" in source
    assert 'DEVICE = torch.device("cuda:3")' in source
    assert "cuda:0" not in source.replace('"cuda:3"', "")
    assert 'RESULTS = HERE / "results"' in source
    assert "from demo" not in source and "import demo" not in source
    assert "run_ecg" not in source and "heart_ep" not in source
    stage_ids = re.findall(r'studio\.stage\("([a-z_]+)"', source)
    assert stage_ids[-1] == "done"
    assert set(stage_ids) <= {s.id for s in spec.stages}
    assert [s for s in stage_ids if s != "done"] == [
        s.id for s in spec.stages if s.id in stage_ids and s.id != "done"], "stages out of order"


@pytest.mark.skipif(not HAS_TORCHCOR, reason="TorchCor is not installed")
def test_script_imports_resolve_against_torchcor(tmp_path):
    """Every name the generated scripts import exists in the installed TorchCor.

    Checked in a fresh interpreter outside the repository, where ``torchcor`` is the
    installed package rather than the checkout folder of the same name.
    """
    wanted: set[tuple[str, str]] = set()
    for spec, _, factory in CASES:
        if spec.kind not in OURS:
            continue
        for node in ast.walk(ast.parse(spec.build_script(factory(), _ctx(device="cpu")))):
            if isinstance(node, ast.ImportFrom):
                wanted |= {(node.module, alias.name) for alias in node.names}
            elif isinstance(node, ast.Import):
                wanted |= {(alias.name, "") for alias in node.names if alias.name != "tcs_studio"}
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import importlib, sys\n"
        "try:\n"
        "    import torchcor  # noqa: F401\n"
        "except ImportError as exc:  # installed, but its files are not there\n"
        "    print(exc)\n"
        "    sys.exit(3)\n"
        f"wanted = {sorted(wanted)!r}\n"
        "missing = [f'{m}.{n}' for m, n in wanted\n"
        "           if n and not hasattr(importlib.import_module(m), n)\n"
        "           or not n and importlib.import_module(m) is None]\n"
        "print(missing)\n"
        "sys.exit(1 if missing else 0)\n", encoding="utf-8")
    out = subprocess.run([sys.executable, str(probe)], cwd=tmp_path, capture_output=True,
                         text=True, timeout=300)
    if out.returncode == 3:
        pytest.skip(f"TorchCor is installed but cannot be imported ({out.stdout.strip()})")
    assert out.returncode == 0, out.stdout + out.stderr[-3000:]


def test_script_context_paths_reach_the_script():
    windows = ep.build_script(ep.ventricle(),
                              _ctx(input_dir=r"D:\meshes\case 1", windows=True))
    assert 'MESH_DIR = Path(r"D:\\meshes\\case 1")' in windows
    home = ep.build_script(ep.ventricle(),
                           _ctx(input_dir="~/meshes/case1"))
    assert 'Path("~/meshes/case1").expanduser()' in home


# =============================================================================== EP
def _ventricle_info(**overrides) -> InputInfo:
    info = dict(path="/data/heart", exists=True,
                files=["heart.pts", "heart.elem", "heart.lon", "LV_sf.vtx", "LV_pf.vtx",
                       "LV_af.vtx", "RV_sf.vtx", "RV_mod.vtx"],
                mesh_name="heart", n_nodes=500_000, n_elems=2_500_000, elem_types=["Tt"],
                regions=[34, 35, 44, 45, 46], fibre_columns=3,
                vtx=[VtxInfo(n, 20) for n in ("LV_sf.vtx", "LV_pf.vtx", "LV_af.vtx",
                                              "RV_sf.vtx", "RV_mod.vtx")])
    info.update(overrides)
    return InputInfo(**info)


def test_ep_defaults_are_the_ventricle_demo():
    params = ep.EPParams()
    assert params == ep.ventricle() and params.variant_label() == "Monodomain"
    assert [e.field for e in _errors(ep.check(params, None))] == [INPUT]  # the mesh: the input folder's
    assert not _errors(ep.check(params, _ventricle_info()))
    assert [d.key for d in get_spec("ep").demos] == ["ventricle", "ventricle-eikonal", "atrium"]


def test_ep_variants():
    assert ep.ventricle_eikonal().variant_label() == "Reaction–eikonal"
    params = ep.EPParams()
    params.model.solver = "re_plus"
    assert params.variant_label() == "Reaction–eikonal with diffusion"


def test_ep_input_folder_checks():
    params = ep.ventricle()
    assert not _errors(ep.check(params, _ventricle_info()))
    assert _errors(ep.check(params, None))
    no_mesh = _ventricle_info(mesh_name=None)
    assert any("no mesh" in i.message for i in _errors(ep.check(params, no_mesh)))
    no_lon = _ventricle_info(fibre_columns=None)
    assert any(".lon" in i.message for i in _errors(ep.check(params, no_lon)))
    hexes = _ventricle_info(elem_types=["Tt", "Hx"])
    assert any("Hx" in i.message for i in _errors(ep.check(params, hexes)))

    missing_site = _ventricle_info(vtx=[VtxInfo("LV_sf.vtx", 3)])
    errors = _errors(ep.check(params, missing_site))
    assert {e.field for e in errors} >= {"stimuli.1.site", "stimuli.4.site"}

    params.model.conductivities[0].regions = [34, 99]
    errors = _errors(ep.check(params, _ventricle_info()))
    assert any(e.field == "model.conductivities.0.regions" and "99" in e.message for e in errors)


def test_ep_memory_estimate():
    params = ep.ventricle()
    info = _ventricle_info(n_nodes=4_000_000)
    params.timing.duration = 1000.0
    params.timing.snapshot_interval = 0.5
    issues = ep.check(params, info)
    memory = [i for i in issues if "Vm history" in i.message]
    assert memory and memory[0].severity == Severity.WARNING
    params.timing.snapshot_interval = 5.0
    small = [i for i in ep.check(params, info) if "Vm history" in i.message]
    assert not small or small[0].severity == Severity.INFO


def test_ep_timing_and_eikonal_checks():
    params = ep.EPParams()
    params.timing.snapshot_interval = 0.015
    assert any(e.field == "timing.snapshot_interval" for e in _errors(ep.check(params, None)))

    params = ep.ventricle_eikonal()
    params.stimuli[1].period, params.stimuli[1].count = 300.0, 2
    assert any("same period" in e.message for e in _errors(ep.check(params, _ventricle_info())))

    params = ep.ventricle_eikonal()
    params.model.velocities[0].vel_n = 0.1  # the mesh has no sheet directions
    assert any(e.field == "model.velocities.0.vel_n" for e in _errors(ep.check(params, _ventricle_info())))


def test_ep_sites_and_cell_groups():
    params = ep.EPParams()
    params.stimuli[0].site = "His.vtx"  # not in the folder
    assert any(e.field == "stimuli.0.site" for e in _errors(ep.check(params, _ventricle_info())))

    params = ep.EPParams()
    params.cells = [ep.CellGroup(), ep.CellGroup(model="CourtemancheRamirezNattel")]
    assert any("Only one cell model" in e.message for e in _errors(ep.check(params, None)))
    params.cells[0].scaling = "GKs=half"
    assert any(e.field == "cells.0.scaling" for e in _errors(ep.check(params, None)))


def test_ep_atrium_demo_leaves_the_site_to_the_folder():
    params = get_spec("ep").demo("atrium").params()
    info = InputInfo(path="/data/atrium/Case_18", exists=True,
                     files=["Case_18.pts", "Case_18.elem", "Case_18.lon", "Case_18.vtx"],
                     mesh_name="Case_18", n_nodes=300_000, elem_types=["Tr"],
                     regions=[1, 2, 3, 4, 5, 6], fibre_columns=3, vtx=[VtxInfo("Case_18.vtx", 40)])
    errors = _errors(ep.check(params, info))
    assert [(e.field, e.message) for e in errors] == [
        ("stimuli.0.site", "Choose a stimulus site (it has Case_18.vtx).")]
    params.stimuli[0].site = "Case_18.vtx"
    assert not _errors(ep.check(params, info))


def test_unstable_time_steps_are_flagged():
    params = ep.EPParams()
    params.cells[0].model = "TWorld"
    params.timing.dt = 0.02
    assert any(i.field == "timing.dt" and i.severity == Severity.WARNING
               for i in ep.check(params, None))
    params.timing.dt = 0.01
    assert not any(i.field == "timing.dt" for i in ep.check(params, None))
    heart = em.EMParams()
    heart.rhythm.ep_dt = 0.05
    assert any(i.field == "rhythm.ep_dt" for i in em.check(heart, None))


def test_ep_script_for_region_specific_cells_and_prepacing():
    params = ep.EPParams()
    params.cells = [ep.CellGroup(model="AlievPanfilov"),
                    ep.CellGroup(regions=[1], model="TWorld", cell_type="MID", sex="FEMALE",
                                 scaling="GKr=0.5")]
    params.prepace.enabled = True
    params.prepace.beats = 25
    source = ep.build_script(params, _ctx(device="cpu"))
    compile(source, "run.py", "exec")
    # explicit regions first: TorchCor gives a model without region_ids the rest
    assert source.index("TWorld(DT") < source.index("AlievPanfilov(DT")
    assert 'cell_type="MID", sex="FEMALE"' in source
    assert "cell_1.GKr *= 0.5" in source
    assert "cell_2.region_ids = None" in source
    assert "beats=25" in source and "pacer(cell)" in source  # one compiled call per model
    assert "make_cells" not in source and "twin" not in source


def test_ep_script_variants():
    eikonal = ep.build_script(ep.ventricle_eikonal(), _ctx())
    assert "ReactionEikonal(CELLS" in eikonal and "diffusion=False" in eikonal
    assert eikonal.count("simulator.add_velocity(") == 2
    assert "add_conductivity" not in eikonal
    assert "max_iter" not in eikonal  # R-E has no linear solve
    atrium = ep.build_script(ep.atrium(), _ctx())
    assert "add_conductivity([1, 2, 3, 4, 5, 6], il=0.3, it=0.06)" in atrium
    assert "threshold=-10.0" in atrium
    ventricle = ep.build_script(ep.ventricle(), _ctx())
    assert "el=1.0732, et=0.4227" in ventricle
    assert 'MESH_DIR / "RV_mod.vtx", start=5.0, duration=1.0, intensity=100.0' in ventricle


# =============================================================================== ECG
def test_ecg_reads_a_prepared_folder():
    params = ecg.ECGParams()
    assert params.variant_label() == "12-lead"
    assert [e.field for e in _errors(ecg.check(params, None))] == [INPUT]  # choose the folder
    prepared = InputInfo(path="/d", exists=True, files=["fields.npz"],
                         subdirs=["KCL_torso3", "electrodes", "heart", "pacing"])
    assert not ecg.check(params, prepared)
    source = ecg.build_script(params, _ctx(input_dir="/data/ecg"))
    assert 'DATA = Path("/data/ecg")' in source and 'TORSO_DIR = DATA / "KCL_torso3"' in source
    assert "download" not in source.lower() and "prepare_" not in source
    assert "lf.add_heart_conductivity([34, 35], il=0.174, it=0.019, el=0.625, et=0.236)" in source


def test_ecg_reaction_eikonal():
    params = ecg.ECGParams()
    params.heart.solver = "eikonal"
    assert params.variant_label() == "12-lead · reaction–eikonal"
    prepared = InputInfo(path="/d", exists=True, files=["fields.npz"],
                         subdirs=["KCL_torso3", "electrodes", "heart", "pacing"])
    assert not ecg.check(params, prepared)
    source = ecg.build_script(params, _ctx(input_dir="/data/ecg"))
    assert "from torchcor.electrophysiology import ReactionEikonal" in source
    assert "ReactionEikonal([cell], T=DURATION, dt=DT, diffusion=False" in source
    assert "simulator.add_velocity([36], vel_l=1.15, vel_t=1.15)" in source
    assert "add_conductivity" not in source.split("lf = LeadField")[0]  # the heart's: velocities instead
    assert "Vm = simulator.solve(snapshot_interval=SAMPLE_INTERVAL" in source  # no CG settings
    params.heart.velocities = []
    assert any(e.field == "heart.velocities" for e in _errors(ecg.check(params, prepared)))
    params = ecg.ECGParams()
    params.heart.solver = "eikonal"
    params.heart.stimuli[0].period, params.heart.stimuli[0].count = 500.0, 2
    assert any(e.field == "heart.stimuli" for e in _errors(ecg.check(params, prepared)))

    params = ecg.ECGParams()  # with diffusion: velocities and conductivities, and the CG solve
    params.heart.solver = "re_plus"
    assert params.variant_label() == "12-lead · reaction–eikonal with diffusion"
    assert not ecg.check(params, prepared)
    source = ecg.build_script(params, _ctx(input_dir="/data/ecg"))
    assert "ReactionEikonal([cell], T=DURATION, dt=DT, diffusion=True" in source
    assert "simulator.add_velocity([36], vel_l=1.15, vel_t=1.15)" in source
    assert "simulator.add_conductivity([36], il=1.5, it=1.5)" in source
    assert "Vm = simulator.solve(a_tol=" in source
    params.heart.conductivities = []
    assert any(e.field == "heart.conductivities" for e in _errors(ecg.check(params, prepared)))


def test_ecg_checks():
    params = ecg.ECGParams()
    folder = InputInfo(path="/d", exists=True, files=["fields.npz"],
                       subdirs=["KCL_torso3", "heart", "pacing", "electrodes"])
    params.data.heart_mesh = "ventricles"
    assert any(e.field == "data.heart_mesh" for e in _errors(ecg.check(params, folder)))
    params.data.electrode_order = "V1, V2"
    assert any(e.field == "data.electrode_order" for e in _errors(ecg.check(params, folder)))


def test_ecg_without_gradients_has_no_heterogeneity():
    source = ecg.build_script(ecg.no_gradients(), _ctx())
    assert "heterogeneous_init" not in source and "default_init" not in source


# =============================================================================== mechanics
def test_mechanics_defaults_and_variants():
    assert mechanics.MechanicsParams().variant_label() == "Beam"
    labels = {d.key: d.params().variant_label() for d in mechanics.DEMOS}
    assert labels == {"beam": "Beam", "lv-inflation": "LV inflation",
                      "lv-contraction": "LV contraction", "lv-beat": "LV beat"}
    for demo in mechanics.DEMOS:
        assert not _errors(mechanics.check(demo.params(), None))


def test_mechanics_checks():
    params = mechanics.MechanicsParams()
    params.loading.pressure_surface = "endo"
    assert any(e.field == "loading.pressure_surface"
               for e in _errors(mechanics.check(params, None)))
    params = mechanics.MechanicsParams()
    params.loading.support = "springs"
    assert any(e.field == "loading.support" for e in _errors(mechanics.check(params, None)))
    params = mechanics.lv_beat()
    params.loading.density = 0.0
    assert any(e.field == "loading.density" for e in _errors(mechanics.check(params, None)))
    params = mechanics.lv_inflation()
    params.geometry.epi_radius = 5.0
    assert any(e.field == "geometry.epi_radius" for e in _errors(mechanics.check(params, None)))


def test_mechanics_scripts():
    beat = mechanics.build_script(mechanics.lv_beat(), _ctx())
    assert 'studio.total(DURATION, "ms")' in beat
    assert "observer=observer" in beat and "rho_infinity=2.0 / 3.0" in beat
    assert "RobinBC.on_surface(mesh, \"epi\"" in beat and "bulk_modulus=None" in beat
    assert "HolzapfelOgdenMaterial(" in beat and "bulk_modulus=BULK_MODULUS" in beat
    p3 = mechanics.build_script(mechanics.lv_contraction(), _ctx())
    assert "ActiveStressMaterial(passive, tension=ACTIVE_TENSION, ramp=True)" in p3
    assert "HELIX_ENDO = 90.0" in p3 and "MaterialAxes(f=helix_fibres)" in p3
    neo = mechanics.MechanicsParams()
    neo.material.law = "neo_hookean"
    neo.material.volume = "penalty"
    source = mechanics.build_script(neo, _ctx())
    compile(source, "run.py", "exec")
    assert "VolumetricPenalty(IsochoricMaterial(NeoHookeanMaterial(mu=10.0)), BULK_MODULUS)" in source
    assert "axes = None" in source


# =============================================================================== EM
def _heart_folder(**overrides) -> InputInfo:
    info = dict(
        path="/data/HCM3/sims_folder", exists=True,
        files=["myocardium_AV_FEC_BB_lvrv.pts", "myocardium_AV_FEC_BB_lvrv.elem",
               "myocardium_AV_FEC_BB_lvrv.lon", "pericardium_scale.dat", "SAN.vtx",
               "fascicles_lv.vtx", "fascicles_rv.vtx"],
        mesh_name="myocardium_AV_FEC_BB_lvrv", n_nodes=900_000, elem_types=["Tt"],
        regions=list(range(1, 30)), fibre_columns=6,
        vtx=[VtxInfo("SAN.vtx", 10), VtxInfo("fascicles_lv.vtx", 50),
             VtxInfo("fascicles_rv.vtx", 40)],
        surfaces=["LV_endo.surf", "RV_endo.surf", "LA_endo.surf", "RA_endo.surf",
                  "epicardium_for_sim.surf", "RPVs.surf", "SVC.surf"])
    info.update(overrides)
    return InputInfo(**info)


def test_em_checks():
    params = em.EMParams()
    assert params.variant_label() == "4-chamber · 5 beats"
    assert _errors(em.check(params, None))
    assert not _errors(em.check(params, _heart_folder()))
    missing = _heart_folder(surfaces=["LV_endo.surf"], vtx=[], files=[])
    fields = {e.field for e in _errors(em.check(params, missing))}
    assert {"heart.rv_endo", "support.springs.0.surface", "heart.pericardium_scale",
            "rhythm.pacing.0.site", INPUT} <= fields  # the mesh is the input folder's own
    no_mesh = [e.message for e in _errors(em.check(params, _heart_folder(mesh_name=None)))]
    assert no_mesh == ["The input folder has no mesh (.pts, .elem, .lon)."]
    two = _heart_folder(files=[*_heart_folder().files, "coarse.pts", "coarse.elem"])
    assert [(i.field, i.message) for i in em.check(params, two) if i.severity == Severity.WARNING
            and i.field == INPUT] == [(INPUT, "The input folder holds 2 meshes; the run reads "
                                              "coarse.")]
    extra_region = _heart_folder(regions=[*range(1, 30), 40])
    messages = " ".join(e.message for e in _errors(em.check(params, extra_region)))
    assert "40" in messages
    params.circulation.valves = params.circulation.valves[:5]
    assert any(e.field == "circulation.valves" for e in _errors(em.check(params, _heart_folder())))


def test_em_script_is_a_parameterised_port_of_the_demo():
    params = em.EMParams()
    params.rhythm.beats = 2
    params.support.start_lv = 10.5
    params.heart.reuse_dir = "/scratch/hcm3"
    source = em.build_script(params, _ctx(input_dir="/data/HCM3/sims_folder"))
    for snippet in ("DURATION = 1666.0", '"lv": 10.5', 'CACHE = Path("/scratch/hcm3")',
                    "ep.solve(on_step=runner, record=False, verbose=False)",
                    'studio.total(DURATION, "ms")', "Unloading(simulation, cavities=cavities",
                    "FirstOrderDynamicSolver(coupling.problem, **DYNAMICS",
                    "CoupledCavityPressures(circulation", "valve_leak_area_mm2=(0.0, 756.622",
                    '{**LAND_VENTRICLES, "T_ref": 175.668}', "recorder.write_bundle(",
                    'NAME = sorted(p.stem for p in MESH.glob("*.pts"))[0]'):  # the folder's mesh
        assert snippet in source, snippet
    assert "argparse" not in source


# =============================================================================== slow runs
def _launch(run: Path, kind: str, params, input_dir: str = "", timeout: float = 300.0) -> dict:
    """Prepare a run folder as the app does, launch it with the runtime and wait."""
    spec = get_spec(kind)
    run.mkdir(parents=True, exist_ok=True)
    ctx = ScriptContext(device="cpu", input_dir=input_dir, run_dir=str(run), run_id=run.name,
                        run_name=f"{kind} test run", machine_name="pytest")
    (run / "run.py").write_text(spec.build_script(params, ctx), encoding="utf-8")
    for name in ("tcs_runtime.py", "tcs_studio.py"):
        shutil.copy(runtime_path(name), run / name)
    (run / "manifest.json").write_text(json.dumps({"kind": kind, "created_at": time.time()}))
    (run / "status.json").write_text(json.dumps({"state": "queued"}))
    out = subprocess.run([sys.executable, str(run / "tcs_runtime.py"), "launch", str(run)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        try:
            status = json.loads((run / "status.json").read_text())
        except (OSError, ValueError):
            status = {}
        if status.get("state") not in (None, "queued", "running"):
            break
        time.sleep(1.0)
    log = (run / "run.log").read_text(errors="replace") if (run / "run.log").exists() else ""
    assert status.get("state") == "completed", f"{status}\n--- run.log ---\n{log[-4000:]}"
    return status


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _check_index(run: Path, kind: str) -> dict:
    index = json.loads((run / "viz" / "index.json").read_text())
    assert index["format"] == 1 and index["kind"] == kind and index["title"]
    assert isinstance(index["created_at"], float)
    for name, file in index["datasets"].items():
        assert (run / "viz" / file).exists(), name
    for row in index["summary"]:
        assert set(row) == {"label", "value", "unit", "hint"}
        assert isinstance(row["value"], int | float | str)
    return index


def _check_surface(run: Path) -> dict:
    surface = _load(run / "viz" / "surface.npz")
    n = len(surface["points"])
    assert surface["points"].dtype == np.float32 and surface["points"].shape == (n, 3)
    assert surface["triangles"].dtype == np.int32 and surface["triangles"].shape[1] == 3
    assert surface["triangles"].min() >= 0 and surface["triangles"].max() < n
    assert surface["node_ids"].dtype == np.int64 and surface["node_ids"].shape == (n,)
    if "regions" in surface:
        assert surface["regions"].dtype == np.int32
        assert surface["regions"].shape == (len(surface["triangles"]),)
    return surface


@pytest.mark.slow
@pytest.mark.skipif(not HAS_TORCHCOR, reason="TorchCor is not installed")
def test_ep_runs_end_to_end_on_a_mesh_folder(tmp_path):
    from torchcor_studio.runtime import tcs_runtime, tcs_studio

    mesh = tcs_studio.make_slab_mesh(tmp_path / "mesh", dx=1.0, corner=2.0)  # a small block, one region
    params = ep.EPParams(cells=[ep.CellGroup()], model=ep.EPModel(conductivities=[ep.Conductivity()]),
                         stimuli=[ep.Stimulus(site="stimulus.vtx")])
    params.timing.duration = 30.0
    params.numerics.precision = "float64"
    assert not _errors(ep.check(params, InputInfo.from_dict(tcs_runtime.inspect_dir(str(mesh)))))
    run = tmp_path / "ep-run"
    status = _launch(run, "ep", params, input_dir=str(mesh))
    assert status["stage"] == "done" and status["progress"] == 1.0
    assert status["sim_total"] == 30.0
    assert "activated" in status["metrics"]
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert [e["id"] for e in events if e["type"] == "stage"] == [
        "setup", "mesh", "solve", "maps", "export", "done"]

    index = _check_index(run, "ep")
    assert set(index["datasets"]) == {"surface", "ep"}
    labels = {row["label"] for row in index["summary"]}
    assert {"Total activation time", "Activated", "Nodes", "Elements"} <= labels
    surface = _check_surface(run)
    n = len(surface["points"])
    data = _load(run / "viz" / "ep.npz")
    frames = int(params.timing.duration / params.timing.snapshot_interval) + 1
    assert data["times"].dtype == np.float32 and data["times"].shape == (frames,)
    assert data["vm"].dtype == np.float16 and data["vm"].shape == (frames, n)
    assert data["lat"].dtype == np.float32 and data["lat"].shape == (n,)
    assert np.isfinite(data["lat"]).any() and np.isnan(data["lat"]).any()
    assert data["rt"].shape == (n,) and data["apd"].shape == (n,)
    k = len(data["trace_labels"])
    assert 3 <= k <= 6
    assert data["traces"].dtype == np.float32 and data["traces"].shape == (k, frames)
    assert data["trace_times"].shape == (frames,)
    assert data["trace_node_ids"].dtype == np.int64
    assert set(data["trace_node_ids"]) <= set(surface["node_ids"])
    assert float(data["vm"].max()) > 0.0 > float(data["vm"].min())  # an action potential
    for name in ("ATs.pt", "RTs.pt", "Vm.pt"):
        assert (run / "results" / name).exists()


@pytest.mark.slow
@pytest.mark.skipif(not HAS_TORCHCOR, reason="TorchCor is not installed")
def test_mechanics_beam_runs_end_to_end(tmp_path):
    params = mechanics.MechanicsParams()
    params.geometry.nx, params.geometry.ny, params.geometry.nz = 10, 1, 1
    run = tmp_path / "beam-run"
    status = _launch(run, "mechanics", params)
    assert status["stage"] == "done" and status["progress"] == 1.0
    index = _check_index(run, "mechanics")
    assert set(index["datasets"]) == {"surface", "mechanics"}
    summary = {row["label"]: row["value"] for row in index["summary"]}
    assert summary["Converged"] == "yes"
    assert 4.0 < summary["Tip z"] < 4.4  # Land et al. consensus 4.17 mm (coarse mesh)
    surface = _check_surface(run)
    data = _load(run / "viz" / "mechanics.npz")
    n = len(surface["points"])
    assert data["frames"].dtype == np.float32 and data["frames"].tolist() == [0.0, 1.0]
    assert str(data["frame_label"]) == "Load factor λ" and str(data["frame_unit"]) == ""
    assert data["displacement"].dtype == np.float32 and data["displacement"].shape == (2, n, 3)
    assert np.abs(data["displacement"][0]).max() == 0.0
    assert np.abs(data["displacement"][1]).max() > 1.0
    steps = len(data["conv_step"])
    assert steps >= 2 and data["conv_step"].dtype == np.int32
    assert data["conv_value"].dtype == np.float32 and data["conv_value"][-1] == 1.0
    assert data["conv_newton"].dtype == np.int32 and data["conv_newton"].shape == (steps,)
    assert (run / "results" / "displacement.vtu").exists()
