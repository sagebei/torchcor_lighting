"""Electrophysiology: monodomain and reaction–eikonal simulations of cardiac tissue.

The tissue is the mesh of the input folder (``.pts``/``.elem``/``.lon`` plus ``.vtx``
stimulus sites); the demos read the biventricle and the atrium of the demo data.
"""

from __future__ import annotations

from torchcor_studio.core.models import InputInfo
from torchcor_studio.core.simulations import codegen, common
from torchcor_studio.core.simulations.base import (
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
from torchcor_studio.core.simulations.common import (
    CellGroup,
    Conductivity,
    PrePacing,
    SolverSettings,
    Stimulus,
    Timing,
    Velocity,
)
from torchcor_studio.core.simulations.fields import Section, param
from torchcor_studio.core.simulations.panels import INPUT, OUTPUT, Panel

SOLVERS = {"monodomain": "Monodomain", "eikonal": "Reaction–eikonal",
           "re_plus": "Reaction–eikonal with diffusion"}
VARIANTS = {"monodomain": "Monodomain", "eikonal": "Reaction–eikonal",
            "re_plus": "Reaction–eikonal with diffusion"}
STAGES = (
    Stage("setup", "Setting up"),
    Stage("mesh", "Preparing the mesh"),
    Stage("solve", "Solving"),
    Stage("maps", "Activation maps"),
    Stage("export", "Saving results"),
    Stage("done", "Completed"),
)


# =============================================================================== parameters
class EPGeometry(Section):
    """How the mesh of the input folder is read."""

    unit_conversion: float = param(1000.0, "Coordinate units per mm", min=1e-6, max=1e6,
                                   advanced=True,
                                   help="1000 when .pts coordinates are in µm, 1 when they are in mm.")


class EPModel(Section):
    """The tissue model and its propagation parameters."""

    solver: str = param("monodomain", "Propagation model", choices=SOLVERS, widget="segmented",
                        summary=True, help="Monodomain solves the diffusion every step; reaction–eikonal "
                             "times each cell from an eikonal activation map (much faster).")
    conductivity_mode: str = param("bidomain", "Conductivity values",
                                   choices=common.CONDUCTIVITY_MODES,
                                   visible_if={"solver": ["monodomain", "re_plus"]},
                                   help="Give intra- and extracellular values (TorchCor takes "
                                        "their harmonic mean) or effective monodomain values.")
    conductivities: list[Conductivity] = param(
        title="Conductivities", default_factory=lambda: [Conductivity()],
        visible_if={"solver": ["monodomain", "re_plus"]}, item="region group",
        help="Tissue conductivity per group of regions.")
    velocities: list[Velocity] = param(
        title="Conduction velocities", default_factory=lambda: [Velocity(vel_l=0.6, vel_t=0.22)],
        visible_if={"solver": ["eikonal", "re_plus"]}, item="region group",
        help="Eikonal conduction velocity per group of regions.")
    mass_lumping: bool = param(False, "Lumped mass matrix", advanced=True,
                               help="Row-sum the finite-element mass matrix onto its diagonal.")


class EPOutputs(Section):
    """What is computed and saved after the solve."""

    repolarisation: bool = param(True, "Repolarisation map",
                                 help="Also compute repolarisation times and action-potential "
                                      "durations (the activation map is always computed).")
    save_vm: bool = param(True, "Vm history",
                          help="results/Vm.pt: the full (frames, nodes) history.")
    vtk: bool = param(False, "VTK frames", help="results/Vm_vtk/*.vtk (and ATs/RTs) for ParaView.")
    vtk_step: int = param(10, "VTK frame step", min=1, max=10000, visible_if={"vtk": True},
                          help="Write every n-th saved frame.")
    igb: bool = param(False, "IGB file", help="results/Vm.igb: the voltage in IGB format.")
    auto_thresholds: bool = param(True, "Thresholds from the cell model",
                                  help="Use thresholds that suit the cell model's voltage range.")
    activation_threshold: float = param(0.0, "Activation threshold", unit="mV",
                                        visible_if={"auto_thresholds": False},
                                        help="Local activation time: first upward crossing.")
    repolarisation_threshold: float = param(-70.0, "Repolarisation threshold", unit="mV",
                                            visible_if={"auto_thresholds": False},
                                            help="First downward crossing after activation.")


#: The defaults are the biventricle of the demo data (``demo/ep/monodomain_ventricle.py``):
#: walls 34/35 and the faster endocardial layer 44-46, paced at five His–Purkinje sites.
_VENTRICLE_STIMULI = [Stimulus(site=f"{site}.vtx", start=start, duration=1.0, intensity=100.0)
                      for site, start in (("LV_sf", 0.0), ("LV_pf", 0.0), ("LV_af", 0.0),
                                          ("RV_sf", 5.0), ("RV_mod", 5.0))]


def _ventricle_model() -> EPModel:
    return EPModel(solver="monodomain", conductivities=[
        Conductivity(regions=[34, 35], il=0.5272, it=0.2076, el=1.0732, et=0.4227),
        Conductivity(regions=[44, 45, 46], il=0.9074, it=0.3332, el=0.9074, et=0.3332)])


class EPParams(Section):
    """Electrophysiology parameters (the defaults: the ventricle demo)."""

    geometry: EPGeometry = param(title="Geometry", default_factory=EPGeometry)
    model: EPModel = param(title="Model", default_factory=_ventricle_model)
    cells: list[CellGroup] = param(title="Cell models",
                                   default_factory=lambda: [CellGroup(model="TenTusscherPanfilov", cell_type="ENDO")],
                                   item="cell model", help="Ionic model per group of regions.")
    prepace: PrePacing = param(title="Pre-pacing", default_factory=PrePacing)
    timing: Timing = param(title="Time", default_factory=lambda: Timing(duration=500.0, dt=0.01,
                                                                        snapshot_interval=1.0))
    stimuli: list[Stimulus] = param(title="Stimuli",
                                    default_factory=lambda: [s.model_copy() for s in _VENTRICLE_STIMULI],
                                    item="stimulus", items="stimuli",
                                    help="Where, when and how strongly the tissue is paced.")
    outputs: EPOutputs = param(title="Outputs", default_factory=EPOutputs)
    numerics: SolverSettings = param(title="Numerics", default_factory=SolverSettings)

    def variant_label(self) -> str:
        return VARIANTS[self.model.solver]

    def thresholds(self) -> tuple[float, float]:
        if not self.outputs.auto_thresholds:
            return self.outputs.activation_threshold, self.outputs.repolarisation_threshold
        info = common.ordered_groups(self.cells)[0].info
        return info.activation, info.repolarisation


# =============================================================================== demos
def ventricle() -> EPParams:
    """``demo/ep/monodomain_ventricle.py``: biventricle, five His-Purkinje sites (the defaults)."""
    return EPParams()


def ventricle_eikonal() -> EPParams:
    """``demo/ep/eikonal_ventricle.py``: reaction–eikonal with a faster endocardial layer."""
    params = ventricle()
    params.model = EPModel(solver="eikonal", velocities=[
        Velocity(regions=[34, 35], vel_l=0.6, vel_t=0.38),
        Velocity(regions=[44, 45, 46], vel_l=0.9, vel_t=0.41)])
    params.numerics = SolverSettings(precision="float64")
    return params


def atrium() -> EPParams:
    """``demo/ep/monodomain_atrium.py``: Courtemanche atrium, effective conductivities."""
    return EPParams(
        model=EPModel(solver="monodomain", conductivity_mode="monodomain", conductivities=[
            Conductivity(regions=[1, 2, 3, 4, 5, 6], il=0.3, it=0.06)]),
        cells=[CellGroup(model="CourtemancheRamirezNattel")],
        timing=Timing(duration=500.0, dt=0.01, snapshot_interval=1.0),
        # Atrial meshes name their pacing site after the case (Case_18.vtx, ...): it is
        # left for the user to pick from the folder's .vtx files.
        stimuli=[Stimulus(site="", start=0.0, duration=2.0, intensity=50.0)],
    )


DEMOS = (
    Demo("ventricle", "Biventricle, five pacing sites",
         "Ten Tusscher–Panfilov cells paced at five His–Purkinje sites.", ventricle,
         case="ep/ventricle", source="demo/ep/monodomain_ventricle.py"),
    Demo("ventricle-eikonal", "Biventricle, reaction–eikonal",
         "A faster endocardial layer; activation from an eikonal map.", ventricle_eikonal,
         case="ep/ventricle", source="demo/ep/eikonal_ventricle.py"),
    Demo("atrium", "Atrium (Courtemanche)",
         "Effective conductivities, paced at the folder's own .vtx site.", atrium,
         case="ep/atrium", source="demo/ep/monodomain_atrium.py"),
)

PANELS = (
    Panel("Geometry", "mesh", (INPUT, "geometry.unit_conversion", OUTPUT)),
    Panel("Cells", "activity", ("cells", "prepace")),
    Panel("Conduction", "wave", ("model",)),
    Panel("Pacing", "zap", ("stimuli",)),
    Panel("Time and outputs", "clock", ("timing", "outputs")),
    Panel("Solver", "cpu", ("numerics",), collapsed=True),
)


def input_use(_params: EPParams) -> tuple[InputUse, str]:
    return "required", "A mesh: .pts, .elem and .lon, with .vtx pacing sites."


# =============================================================================== input
def fit_input(params: EPParams, info: InputInfo | None, previous: EPParams | None = None, *,
              new_input: bool = False) -> EPParams | None:
    """Fill in what depends on the input folder: every stimulus points at a ``.vtx`` the
    folder has, its first one (the first that lists nodes) for a site it lacks. Only sites
    that do not exist are changed, and only after something that can make them wrong (a
    new folder, demo or stimulus row), so a site being typed or a valid choice is never
    replaced."""
    if previous is not None and len(previous.stimuli) == len(params.stimuli):
        return None
    if info is None or not info.has_mesh or not info.vtx:
        return None  # the folder's sites are not known yet
    sites = [v.name for v in info.vtx]
    first = next((v.name for v in info.vtx if v.count), sites[0])
    stimuli = [s if s.site in sites else s.model_copy(update={"site": first}) for s in params.stimuli]
    if all(new is old for new, old in zip(stimuli, params.stimuli, strict=True)):
        return None
    return params.model_copy(update={"stimuli": stimuli})


# =============================================================================== checks
def check(params: EPParams, info: InputInfo | None) -> list[Issue]:
    issues: list[Issue] = common.mesh_issues(info, INPUT)
    model = params.model
    ok = info is not None and info.has_mesh
    regions = info.regions if ok and info.regions else None
    nodes = info.n_nodes if ok else None
    sites = [v.name for v in info.vtx] if ok else None
    if ok:
        empty = [v.name for v in info.vtx if v.count == 0]
        for i, stim in enumerate(params.stimuli):
            if stim.site in empty:
                issues.append(Issue(Severity.ERROR, f"{stim.site} lists no nodes.",
                                    f"stimuli.{i}.site"))
    issues += common.stimulus_issues(params.stimuli, sites, "stimuli", params.timing.duration,
                                     "the input folder")
    issues += common.timing_issues(params.timing)
    issues += common.cell_group_issues(params.cells, regions)
    issues += common.cell_dt_issues((g.model for g in params.cells), params.timing.dt, "timing.dt")
    issues += common.prepace_issues(params.prepace, params.cells)
    fibres = info.fibre_columns if info is not None else None
    if model.solver in ("monodomain", "re_plus"):
        issues += common.conductivity_issues(model.conductivities, model.conductivity_mode,
                                             regions, "model.conductivities")
    if model.solver in ("eikonal", "re_plus"):
        issues += common.velocity_issues(model.velocities, regions, fibres, "model.velocities")
        issues += common.eikonal_schedule_issues(params.stimuli, "stimuli")
    history = common.vm_history_issue(params.timing.frames, nodes, params.numerics.precision)
    if history:
        issues.append(history)
    if params.outputs.vtk and nodes and params.timing.frames / params.outputs.vtk_step > 500:
        issues.append(Issue(Severity.WARNING,
                            f"{params.timing.frames // params.outputs.vtk_step} VTK frames will "
                            "be written; raise the VTK frame step.", "outputs.vtk_step"))
    return issues


# =============================================================================== script
def build_script(params: EPParams, ctx: ScriptContext) -> str:
    geo, model, timing, out, num = (params.geometry, params.model, params.timing,
                                    params.outputs, params.numerics)
    eikonal = model.solver in ("eikonal", "re_plus")
    groups = common.ordered_groups(params.cells)
    title = ctx.run_name or "Electrophysiology run"
    cells_label = "; ".join(
        g.info.label + (f", {common.CELL_TYPES[g.cell_type].lower()}" if g.info.cell_types else "")
        + (f" (regions {', '.join(map(str, g.regions))})" if g.regions else "")
        for g in groups)
    script = codegen.Script(codegen.docstring(
        title, f"Electrophysiology · {VARIANTS[model.solver]} on the mesh of the input folder.", ctx,
        [f"Cell model: {cells_label}.",
         f"{timing.duration:g} ms simulated at dt = {timing.dt:g} ms, Vm saved every "
         f"{timing.snapshot_interval:g} ms."]))

    solver_class = "ReactionEikonal" if eikonal else "Monodomain"
    script.imports.add("import torchcor as tc")
    script.imports.add(f"from torchcor.electrophysiology import {solver_class}"
                       + (", PrePace" if params.prepace.enabled else ""))
    script.imports.add(common.cell_imports(groups))

    s = script.settings
    codegen.device_setting(script, ctx)
    s.add("DTYPE", common.dtype_expr(num.precision),
          "precision of Vm and the cell models (CG runs in float64)")
    s.add("DURATION", codegen.lit(float(timing.duration)), "ms of simulated time")
    s.add("DT", codegen.lit(float(timing.dt)), "ms per time step")
    s.add("SNAPSHOT_INTERVAL", codegen.lit(float(timing.snapshot_interval)),
          "ms between saved Vm frames")
    codegen.output_settings(script)
    s.add("MESH_DIR", codegen.path(ctx.input_dir or ".", ctx.windows),
          "the mesh (.pts, .elem, .lon) and .vtx sites")

    # 1. set up ---------------------------------------------------------------------------
    script.step("Set up", "setup", "Setting up")
    script.code("tc.set_device(DEVICE)")
    script.blank()
    script.comment("Cell models (ionic currents), one per group of regions; a model given "
                   "no regions takes every region the models before it leave.")
    body, names = common.cell_models_code(groups)
    script.code(body)
    script.code(f"CELLS = [{', '.join(names)}]")
    if params.prepace.enabled:
        script.blank()
        script.comment(f"Pre-pace one cell of each model to steady state at "
                       f"{params.prepace.cycle_length:g} ms cycle length (torchcor PrePace): its "
                       "step is compiled once and paced on the CPU; the state then seeds every node.")
        script.code(common.prepace_code(params.prepace))

    # 2. mesh -----------------------------------------------------------------------------
    script.step("Mesh and tissue", "mesh", "Preparing the mesh")
    kwargs = {"T": "DURATION", "dt": "DT"}
    if eikonal:
        kwargs["diffusion"] = codegen.lit(model.solver == "re_plus")
    if model.mass_lumping:
        kwargs["mass_lumping"] = "True"
    kwargs.update(device="DEVICE", dtype="DTYPE")
    script.code(codegen.call(solver_class, "CELLS", prefix="simulator = ", **kwargs))
    unit = geo.unit_conversion
    unit_text = codegen.lit(int(unit) if float(unit).is_integer() else float(unit))
    script.code(f"simulator.load_mesh(MESH_DIR, unit_conversion={unit_text})"
                + ("  # .pts in µm -> mm" if unit == 1000 else ""))
    if model.solver in ("monodomain", "re_plus"):
        script.blank()
        script.comment("Tissue conductivities (S/m) per group of regions (None = all regions)"
                       + ("; TorchCor combines intra- and extracellular values by their "
                          "harmonic mean." if model.conductivity_mode == "bidomain" else "."))
        script.code(common.conductivity_code(model.conductivities, model.conductivity_mode))
    if eikonal:
        script.blank()
        script.comment("Conduction velocities (mm/ms) per group of regions, for the eikonal "
                       "activation times.")
        script.code(common.velocity_code(model.velocities))
    script.blank()
    script.comment("Stimuli: strength in mV/ms (µA/µF), times in ms.")
    for stim in params.stimuli:
        script.code(codegen.call("simulator.add_stimulus", f"MESH_DIR / {codegen.lit(stim.site)}",
                                 **common.stimulus_kwargs(stim)))

    # 3. solve ----------------------------------------------------------------------------
    note = {"monodomain": "Vm: (frames, nodes) on DEVICE, frame 0 at t = 0.",
            "eikonal": "The eikonal activation times come first, then each cell fires when its "
                       "front arrives; there is no linear solve.",
            "re_plus": "The eikonal activation times come first, then the reaction with the "
                       "monodomain diffusion term (a CG solve per step)."}[model.solver]
    script.step("Solve", "solve", "Solving", note=note)
    solve_kwargs = {}
    if model.solver != "eikonal":
        solve_kwargs = {"a_tol": codegen.lit(num.a_tol), "r_tol": codegen.lit(num.r_tol),
                        "max_iter": codegen.lit(int(num.max_iter)),
                        "linear_guess": codegen.lit(bool(num.linear_guess))}
    solve_kwargs.update(snapshot_interval="SNAPSHOT_INTERVAL", verbose="True",
                        result_path="RESULTS")
    script.code(codegen.call("simulator.solve", prefix="Vm = ", **solve_kwargs))
    if eikonal:
        script.code("RESULTS.mkdir(parents=True, exist_ok=True)")
        script.code('torch.save(simulator.eikonal_AT.cpu(), RESULTS / "eikonal_AT.pt")'
                    "  # prescribed arrival times")

    # 4. maps -----------------------------------------------------------------------------
    activation, repolarisation = params.thresholds()
    unit_name = groups[0].info.unit or "(normalised)"
    script.step("Activation and repolarisation", "maps", "Activation maps",
                note=f"Local activation time: first upward crossing of {activation:g} {unit_name}"
                     + (f"; repolarisation: first downward crossing of {repolarisation:g} "
                        f"{unit_name} after it." if out.repolarisation else "."))
    script.code(codegen.call("simulator.compute_activation_map", "Vm", prefix="ATs = ",
                             snapshot_interval="SNAPSHOT_INTERVAL",
                             threshold=codegen.lit(float(activation)), save="True"))
    if out.repolarisation:
        script.code(codegen.call("simulator.compute_repolarization_map", "Vm", prefix="RTs = ",
                                 snapshot_interval="SNAPSHOT_INTERVAL",
                                 threshold=codegen.lit(float(repolarisation)),
                                 search_after="ATs", save="True"))
    else:
        script.code("RTs = None")
    script.code('''
        activated = float(torch.isfinite(ATs).float().mean()) * 100
        studio.metric("activated", round(activated, 1), "%")
        print(f"activated {activated:.1f}% of the nodes", flush=True)
    ''')

    # 5. save -----------------------------------------------------------------------------
    script.step("Save", "export", "Saving results")
    if out.save_vm:
        script.code("simulator.save_vm(Vm)  # results/Vm.pt")
    if out.vtk:
        script.code(f"simulator.vm_to_vtk(Vm, step={int(out.vtk_step)})"
                    "  # results/Vm_vtk/*.vtk, ATs.vtk, RTs.vtk")
    if out.igb:
        script.code("simulator.vm_to_igb(Vm)  # results/Vm.igb")
    script.code(f'''
        bundle = studio.Bundle("ep", title={codegen.lit(title)})
        studio.export_ep(simulator, Vm, SNAPSHOT_INTERVAL, ATs=ATs, RTs=RTs, bundle=bundle)
        bundle.write()

        studio.stage("done", "Completed")
    ''')
    return script.render()


# =============================================================================== spec
SPEC = register(SimulationSpec(
    kind=SimulationKind.EP,
    title="Electrophysiology",
    tagline="Monodomain or reaction–eikonal activation of tissue",
    description="Simulate the spread of electrical activation through cardiac tissue with a "
                "cell model of your choice, on atria or whole ventricles. Produces Vm movies, activation and repolarisation maps.",
    icon="zap",
    params_model=EPParams,
    build_script=build_script,
    stages=STAGES,
    demos=DEMOS,
    panels=PANELS,
    check=check,
    fit_input=fit_input,
    input_use=input_use,
    badge="EP",
))
