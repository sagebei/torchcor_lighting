"""12-lead ECG: ventricular electrophysiology on a heart-torso model, then lead fields.

The model is a prepared heart-torso folder: the torso mesh, the ventricles cut out of it,
pacing sites, electrodes and nodal heart coordinates. The defaults follow TorchCor's
``demo/ecg`` recipe and the folder its ``demo_download.py`` prepares.
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
    CELL_MODELS,
    CELL_TYPES,
    Conductivity,
    SolverSettings,
    Stimulus,
    Velocity,
)
from torchcor_studio.core.simulations.fields import Section, param
from torchcor_studio.core.simulations.panels import INPUT, OUTPUT, Group, Panel

#: How the activation spreads through the heart: TorchCor's Monodomain or ReactionEikonal.
PROPAGATION = {"monodomain": "Monodomain", "eikonal": "Reaction–eikonal",
               "re_plus": "Reaction–eikonal with diffusion"}
#: The models that solve the diffusion (with conductivities), and the eikonal ones (with velocities).
DIFFUSION, EIKONAL = ["monodomain", "re_plus"], ["eikonal", "re_plus"]
KCL_FOLDER = "KCL_torso3"
ELECTRODE_ORDER = "V1, V2, V3, V4, V5, V6, RA, LA, RL, LL"
_VENTRICULAR = {name: m.label for name, m in CELL_MODELS.items() if m.cell_types}

STAGES = (
    Stage("setup", "Setting up"),
    Stage("data", "Heart and torso"),
    Stage("ep", "Heart activity"),
    Stage("leadfield", "Lead fields"),
    Stage("ecg", "12-lead ECG"),
    Stage("export", "Saving results"),
    Stage("done", "Completed"),
)


# =============================================================================== parameters
class ECGData(Section):
    """Where the parts of the prepared heart-torso model are in the input folder."""

    torso_mesh: str = param(KCL_FOLDER, "Torso mesh folder", span=1,
                            help="Sub-folder of the input folder with the full heart-torso mesh.")
    heart_mesh: str = param("heart", "Heart mesh folder", span=1,
                            help="Sub-folder with the ventricles cut out of the torso mesh, in "
                                 "torso node order (as LeadField requires).")
    pacing: str = param("pacing", "Pacing sites folder", span=1,
                        help="Sub-folder holding the stimulus .vtx files.")
    electrodes: str = param("electrodes/lf_src.vtx", "Electrodes", span=1,
                            help="The 10 electrode nodes of the torso mesh, relative to the "
                                 "input folder.")
    electrode_order: str = param(ELECTRODE_ORDER, "Electrode order", advanced=True, widget="mono",
                                 help="Names of the electrodes in file order; RL is the ground.")
    fields: str = param("fields.npz", "Heart coordinates", advanced=True,
                        help="npz with nodal apico-basal 'ab' and transmural 'tm' coordinates "
                             "for the repolarisation gradients; empty = none.")


class ECGHeart(Section):
    """Ventricular electrophysiology, whose Vm drives the ECG."""

    model: str = param("TenTusscherPanfilov", "Cell model", choices=_VENTRICULAR, widget="combo",
                       help="Ventricular ionic model.")
    cell_type: str = param("EPI", "Cell type", choices=CELL_TYPES,
                           help="Base cell type; the gradients below modulate it.")
    heterogeneity: bool = param(True, "Repolarisation gradients",
                                visible_if={"model": "TenTusscherPanfilov"},
                                help="Larger IKs towards epicardium and apex and an endo-to-epi "
                                     "Ito gradient, so the T wave is concordant with the QRS.")
    duration: float = param(600.0, "Duration", unit="ms", min=10.0, max=5000.0, step=50.0,
                            help="One beat including the T wave.")
    dt: float = param(0.01, "Time step", unit="ms", min=0.001, max=0.1, step=0.005, decimals=3,
                      help="Integration step of the EP solve.")
    sample_interval: float = param(1.0, "ECG sample interval", unit="ms", min=0.1, max=10.0,
                                   step=0.5, help="Time between Vm frames, i.e. ECG samples.")
    solver: str = param("monodomain", "Propagation model", choices=PROPAGATION, widget="segmented",
                        summary=True,
                        help="Monodomain solves the diffusion every step. Reaction–eikonal times "
                             "each cell from an eikonal activation map: much faster, but its cells "
                             "are not coupled, so the ECG is less faithful. With diffusion, the "
                             "cells fire at their eikonal times and stay coupled (a linear solve "
                             "per step); give conductivities and velocities of the same speed.")
    conductivity_mode: str = param("monodomain", "Conductivity values",
                                   choices=common.CONDUCTIVITY_MODES,
                                   visible_if={"solver": DIFFUSION},
                                   help="The KCL values are effective monodomain values tuned "
                                        "for its ~0.9 mm mesh.")
    conductivities: list[Conductivity] = param(
        title="Conductivities",
        default_factory=lambda: [Conductivity(regions=[34, 35], il=0.408, it=0.141),
                                 Conductivity(regions=[36], il=1.5, it=1.5)],
        visible_if={"solver": DIFFUSION}, item="region group",
        help="Heart conductivity per region: walls 34/35 and the fast endocardial layer 36.")
    velocities: list[Velocity] = param(
        title="Conduction velocities",
        # the walls as TorchCor's eikonal ventricle demo; the layer scaled from the walls by
        # the square root of the conductivity ratio, sqrt(1.5 / 0.408)
        default_factory=lambda: [Velocity(regions=[34, 35], vel_l=0.6, vel_t=0.38),
                                 Velocity(regions=[36], vel_l=1.15, vel_t=1.15)],
        visible_if={"solver": EIKONAL}, item="region group",
        help="Eikonal conduction velocity per region: walls 34/35 and the fast endocardial "
             "layer 36.")
    stimuli: list[Stimulus] = param(
        title="Pacing sites",
        default_factory=lambda: [Stimulus(site=f"{site}.vtx", start=start, duration=1.0,
                                          intensity=100.0)
                                 for site, start in (("LV_septal", 0.0), ("LV_anterior", 0.0),
                                                     ("RV_septal", 10.0),
                                                     ("RV_moderator", 10.0))],
        item="site", help="Durrer-type early activation sites in the pacing folder.")
    mass_lumping: bool = param(True, "Lumped mass matrix", advanced=True,
                               help="Row-sum the finite-element mass matrix (as the demo).")


class TorsoConductivity(Section):
    """Isotropic conductivity of some torso tissues (a row of the torso table)."""

    tissue: str = param("", "Tissue", help="Name for your reference.")
    tags: list[int] = param(title="Region tags", widget="regions", default_factory=list,
                            help="Torso mesh tags with this conductivity.")
    g: float = param(0.2, "Conductivity", unit="S/m", min=0.0, step=0.01, decimals=4,
                     help="Isotropic conductivity.")


class HeartConductivity(Section):
    """Bidomain conductivity of heart regions in the volume conductor."""

    regions: list[int] = param(title="Regions", widget="regions", default_factory=list,
                               help="Heart tags of the torso mesh.")
    il: float = param(0.174, "Intracellular, along fibres", unit="S/m", min=0.0, decimals=4,
                      group="Intracellular", short="Along fibres")
    it: float = param(0.019, "Intracellular, across fibres", unit="S/m", min=0.0, decimals=4,
                      group="Intracellular", short="Across")
    el: float = param(0.625, "Extracellular, along fibres", unit="S/m", min=0.0, decimals=4,
                      group="Extracellular", short="Along fibres")
    et: float = param(0.236, "Extracellular, across fibres", unit="S/m", min=0.0, decimals=4,
                      group="Extracellular", short="Across")


#: The KCL table of the TorchCor ECG demo: (tissue, tags, S/m).
KCL_TORSO = (
    ("Bath and ICD hardware", [1, 26, 30, 31, 32, 33], 0.24725),
    ("Skin", [2], 0.117),
    ("Bone", [3], 0.05),
    ("Kidneys and liver", [4, 5], 0.1667),
    ("Stomach and spleen", [6, 7], 0.1),
    ("Lungs", [8], 0.0714),
    ("Atrial walls", [20, 23], 0.25),
    ("Blood and vessels", [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 25, 27, 28, 29],
     0.6667),
)


class VolumeConductor(Section):
    """The torso as a passive volume conductor and the lead-field solves."""

    torso: list[TorsoConductivity] = param(
        title="Torso tissues",
        default_factory=lambda: [TorsoConductivity(tissue=n, tags=list(t), g=g)
                                 for n, t, g in KCL_TORSO],
        item="tissue", summary=True,
        help="Every non-heart tag of the torso mesh needs a conductivity.")
    heart: list[HeartConductivity] = param(
        title="Heart in the torso", default_factory=lambda: [HeartConductivity(regions=[34, 35])],
        item="heart region", summary=True, help="Anisotropic bidomain conductivity of the ventricles (intra and extra).")
    a_tol: float = param(1e-10, "Absolute tolerance", min=1e-16, max=1e-2, log=True,
                         advanced=True, help="CG tolerance of each reciprocal lead-field solve.")
    r_tol: float = param(1e-10, "Relative tolerance", min=1e-16, max=1e-2, log=True,
                         advanced=True, help="CG relative tolerance of each lead-field solve.")
    max_iter: int = param(20000, "Iteration limit", min=10, max=10_000_000, advanced=True,
                          help="CG iterations per electrode.")


class ECGOutputs(Section):
    """What is saved besides the ECG."""

    plot: bool = param(True, "12-lead plot", help="results/ecg_12lead.png (TorchCor's own plot).")
    save_vm: bool = param(False, "Heart Vm history", help="results/Vm.pt, the full heart history.")


class ECGParams(Section):
    """ECG parameters."""

    data: ECGData = param(title="In the input folder", default_factory=ECGData)
    heart: ECGHeart = param(title="Heart electrophysiology", default_factory=ECGHeart)
    conductor: VolumeConductor = param(title="Volume conductor", default_factory=VolumeConductor)
    numerics: SolverSettings = param(title="Numerics",
                                     default_factory=lambda: SolverSettings(precision="float64"))
    outputs: ECGOutputs = param(title="Outputs", default_factory=ECGOutputs)

    def variant_label(self) -> str:
        solver = self.heart.solver
        return "12-lead" if solver == "monodomain" else f"12-lead · {PROPAGATION[solver].lower()}"


# =============================================================================== demos
def no_gradients() -> ECGParams:
    params = ECGParams()
    params.heart = ECGHeart(heterogeneity=False)
    return params


#: What TorchCor's demo_download.py leaves in the ecg folder once KCL torso 3 is prepared.
KCL_PREPARED = (KCL_FOLDER, "heart", "pacing", "electrodes")

DEMOS = (
    Demo("healthy", "KCL torso 3, healthy beat",
         "A 12-lead ECG of one beat on a CT-based heart–torso model.", ECGParams, case="ecg",
         requires=KCL_PREPARED, source="demo/ecg/demo_ecg.py"),
    Demo("uniform", "KCL torso 3, no repolarisation gradients",
         "The same beat without the IKs and Ito gradients.", no_gradients, case="ecg",
         requires=KCL_PREPARED, source="demo/ecg/demo_ecg.py"),
)

HEART_EP = Group("Heart EP", "heart")

PANELS = (
    Panel("Heart and torso", "user", (INPUT, "data", OUTPUT)),
    Panel("Cells", "activity", ("heart.model", "heart.cell_type", "heart.heterogeneity"),
          group=HEART_EP),
    Panel("Models", "wave", ("heart.solver", "heart.conductivity_mode", "heart.conductivities",
                             "heart.velocities", "heart.mass_lumping"), group=HEART_EP),
    Panel("Pacing sites", "zap", ("heart.stimuli",), group=HEART_EP),
    Panel("Torso conductivities", "layers", (("conductor.torso", "Torso tissues"), "conductor.heart"),
          collapsed=True),
    Panel("Time and outputs", "clock", ("heart.duration", "heart.dt", "heart.sample_interval",
                                        "outputs")),
    Panel("Solver", "cpu", (("numerics", "Heart solve"), ("conductor", "Lead fields")),
          collapsed=True),
)


def input_use(_params: ECGParams) -> tuple[InputUse, str]:
    return "required", ("A prepared heart–torso folder: torso and heart meshes, pacing sites "
                        "and electrodes.")


# =============================================================================== checks
def check(params: ECGParams, info: InputInfo | None) -> list[Issue]:
    issues: list[Issue] = []
    data, heart = params.data, params.heart
    if info is None or not info.exists:
        issues.append(Issue(Severity.ERROR, "Choose the input folder with the prepared "
                                            "heart–torso model.", INPUT))
    else:
        for field in ("torso_mesh", "heart_mesh", "pacing"):
            folder = getattr(data, field).strip("/\\")
            if folder.split("/")[0] not in info.subdirs:
                issues.append(Issue(Severity.ERROR, f"The input folder has no '{folder}' "
                                                    "sub-folder.", f"data.{field}"))
        top = data.electrodes.replace("\\", "/").split("/")[0]
        if top not in info.subdirs and top not in info.files:
            issues.append(Issue(Severity.ERROR, f"{data.electrodes} is not in the input "
                                                "folder.", "data.electrodes"))
        if heart.heterogeneity and data.fields and data.fields not in info.files \
                and heart.model == "TenTusscherPanfilov":
            issues.append(Issue(Severity.ERROR, f"{data.fields} is not in the input folder; "
                                                "clear it to run without gradients.",
                                "data.fields"))
    names = [n.strip() for n in data.electrode_order.split(",") if n.strip()]
    required = {"V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"}
    if set(names) != required or len(names) != 10:
        issues.append(Issue(Severity.ERROR, "The electrode order must name V1-V6, RA, LA, RL "
                                            "and LL once each.", "data.electrode_order"))
    issues += common.stimulus_issues(heart.stimuli, None, "heart.stimuli", heart.duration)
    issues += common.cell_dt_issues([heart.model], heart.dt, "heart.dt")
    if heart.solver in DIFFUSION:
        issues += common.conductivity_issues(heart.conductivities, heart.conductivity_mode, None,
                                             "heart.conductivities")
    if heart.solver in EIKONAL:
        issues += common.velocity_issues(heart.velocities, None, None, "heart.velocities")
        issues += common.eikonal_schedule_issues(heart.stimuli, "heart.stimuli")
    for i, row in enumerate(params.conductor.heart):
        if not row.regions:
            issues.append(Issue(Severity.ERROR, "Name the heart tags of the torso mesh.",
                                f"conductor.heart.{i}.regions"))
        if min(row.il, row.it, row.el, row.et) <= 0:
            issues.append(Issue(Severity.ERROR, "Conductivities must be positive.",
                                f"conductor.heart.{i}.il"))
    if not params.conductor.heart:
        issues.append(Issue(Severity.ERROR, "Add the heart conductivity.", "conductor.heart"))
    for i, row in enumerate(params.conductor.torso):
        if not row.tags:
            issues.append(Issue(Severity.ERROR, "Give the torso tags of this tissue.",
                                f"conductor.torso.{i}.tags"))
    ratio = heart.sample_interval / heart.dt
    if abs(ratio - round(ratio)) > 1e-6 * ratio:
        issues.append(Issue(Severity.ERROR, "The sample interval must be a whole number of time "
                                            "steps.", "heart.sample_interval"))
    return issues


# =============================================================================== script
def build_script(params: ECGParams, ctx: ScriptContext) -> str:
    data, heart, conductor, num, out = (params.data, params.heart, params.conductor,
                                        params.numerics, params.outputs)
    lit = codegen.lit
    model = CELL_MODELS[heart.model]
    eikonal, diffusion = heart.solver in EIKONAL, heart.solver in DIFFUSION
    solver_class = "ReactionEikonal" if eikonal else "Monodomain"
    gradients = heart.heterogeneity and heart.model == "TenTusscherPanfilov" and bool(data.fields)
    title = ctx.run_name or "ECG run"
    script = codegen.Script(codegen.docstring(
        title, "ECG · 12-lead ECG from ventricular electrophysiology by the lead-field "
               "(reciprocity) method.", ctx,
        ["Heart-torso model: a prepared folder.",
         f"Heart activation: {PROPAGATION[heart.solver].lower()}.",
         f"Cell model: {model.label}, {CELL_TYPES[heart.cell_type].lower()}"
         + (", with transmural and apico-basal repolarisation gradients." if gradients else "."),
         f"{heart.duration:g} ms simulated at dt = {heart.dt:g} ms; ECG sampled every "
         f"{heart.sample_interval:g} ms."]))
    script.imports.add("import numpy as np")
    script.imports.add("import torchcor as tc")
    script.imports.add("from torchcor.electrocardiogram import LeadField")
    script.imports.add(f"from torchcor.electrophysiology import {solver_class}")
    script.imports.add(f"from torchcor.ionic import {heart.model}")

    s = script.settings
    codegen.device_setting(script, ctx)
    s.add("DTYPE", common.dtype_expr(num.precision), "Vm precision (lead fields run in float64)")
    s.add("DURATION", lit(float(heart.duration)), "ms: one beat with its T wave")
    s.add("DT", lit(float(heart.dt)), "ms per EP time step")
    s.add("SAMPLE_INTERVAL", lit(float(heart.sample_interval)), "ms between Vm frames = ECG samples")
    codegen.output_settings(script)
    s.add("DATA", codegen.path(ctx.input_dir or ".", ctx.windows), "the prepared heart-torso model")
    s.add("TORSO_DIR", f"DATA / {lit(data.torso_mesh)}", "full heart-torso mesh")
    s.add("HEART_DIR", f"DATA / {lit(data.heart_mesh)}", "ventricles, in torso node order")
    s.add("PACING_DIR", f"DATA / {lit(data.pacing)}", "early-activation sites")
    s.add("ELECTRODES", f"DATA / {lit(data.electrodes)}", "the 10 electrode nodes")
    if gradients:
        s.add("FIELDS", f"DATA / {lit(data.fields)}", "apico-basal and transmural coordinates")

    # 1. set up ---------------------------------------------------------------------------
    script.step("Set up", "setup", "Setting up")
    script.code("""
        tc.set_device(DEVICE)
        RESULTS.mkdir(parents=True, exist_ok=True)
    """)
    script.code(f'bundle = studio.Bundle("ecg", title={lit(title)})')

    # 2. data -----------------------------------------------------------------------------
    script.step("Heart-torso model", "data", "Checking the heart-torso model")
    script.code("""
        for path in (TORSO_DIR, HEART_DIR, PACING_DIR, ELECTRODES):
            if not path.exists():
                raise SystemExit(f"missing {path}")
    """)

    # 3. heart EP -------------------------------------------------------------------------
    script.step("Heart electrophysiology", "ep", "Heart electrophysiology",
                note={"monodomain": "Monodomain EP on the ventricles.",
                      "eikonal": "Reaction–eikonal EP on the ventricles: the eikonal activation "
                                 "times come first, then each cell fires when its front arrives "
                                 "(no linear solve).",
                      "re_plus": "Reaction–eikonal EP with diffusion on the ventricles: the "
                                 "eikonal activation times come first, then the reaction with the "
                                 "monodomain diffusion term (a linear solve per step)."}[heart.solver]
                     + " Its transmembrane potential Vm is what the lead fields turn into "
                       "body-surface potentials.")
    kwargs = {"cell_type": lit(model.cell_types[heart.cell_type]), "device": "DEVICE",
              "dtype": "DTYPE"}
    script.code(codegen.call(heart.model, "DT", prefix="cell = ", **kwargs))
    if gradients:
        script.code('''
            fields = np.load(FIELDS)
            ab = torch.from_numpy(fields["ab"])  # apico-basal, 0 apex -> 1 base
            tm = torch.from_numpy(fields["tm"])  # transmural,  0 endo -> 1 epi
            # IKs larger towards the epicardium and apex -> shorter APD there. These
            # repolarisation gradients make the T wave concordant with the QRS.
            gks = 0.392 * (0.8 * (1.5 / 0.8) ** tm) * (1.25 * (0.8 / 1.25) ** ab)
            ramp = torch.clamp((tm - 0.15) / 0.30, 0, 1)
            gto = 0.073 + (0.294 - 0.073) * ramp * ramp * (3 - 2 * ramp)  # ENDO -> EPI Ito
            default_init = cell.initialize
        ''')
        script.definitions('''
            def heterogeneous_init(n_nodes):
                """The model's own initial state, then per-node GKs and Gto."""
                vm = default_init(n_nodes)
                idx = cell.node_indices.cpu()
                cell.GKs = gks[idx].to(DEVICE, DTYPE)
                cell.Gto = gto[idx].to(DEVICE, DTYPE)
                return vm
        ''')
        script.code("cell.initialize = heterogeneous_init")
        script.blank()
    sim_kwargs = {"T": "DURATION", "dt": "DT"}
    if eikonal:
        sim_kwargs["diffusion"] = lit(diffusion)  # without it, each cell fires on its own
    sim_kwargs.update(device="DEVICE", dtype="DTYPE")
    if heart.mass_lumping:
        sim_kwargs["mass_lumping"] = "True"
    script.code(codegen.call(solver_class, "[cell]", prefix="simulator = ", **sim_kwargs))
    script.code("simulator.load_mesh(HEART_DIR, unit_conversion=1000)  # µm -> mm")
    if diffusion:
        script.comment("Heart conductivities (S/m); the KCL values are tuned for its ~0.9 mm "
                       "mesh (about 80-100 ms total activation), region 36 is the fast "
                       "endocardial layer.")
        script.code(common.conductivity_code(heart.conductivities, heart.conductivity_mode))
    if eikonal:
        script.comment("Conduction velocities (mm/ms) per region, for the eikonal activation "
                       "times; region 36 is the fast endocardial layer.")
        script.code(common.velocity_code(heart.velocities))
    script.comment("Pacing sites: strength in mV/ms, times in ms.")
    for stim in heart.stimuli:
        script.code(codegen.call("simulator.add_stimulus", f"PACING_DIR / {lit(stim.site)}",
                                 **common.stimulus_kwargs(stim)))
    solve_kwargs = {"a_tol": lit(num.a_tol), "r_tol": lit(num.r_tol), "max_iter": lit(int(num.max_iter)),
                    "linear_guess": lit(bool(num.linear_guess))} if diffusion else {}
    script.code(codegen.call("simulator.solve", prefix="Vm = ", **solve_kwargs,
                             snapshot_interval="SAMPLE_INTERVAL", verbose="True",
                             result_path="RESULTS") + ".cpu()")
    act = model.activation
    script.code(codegen.call("simulator.compute_activation_map", "Vm", prefix="ATs = ",
                             snapshot_interval="SAMPLE_INTERVAL", threshold=lit(float(act)),
                             save="True"))
    if out.save_vm:
        script.code("simulator.save_vm(Vm)  # results/Vm.pt")
    script.code("studio.export_ep(simulator, Vm, SAMPLE_INTERVAL, ATs=ATs, bundle=bundle)"
                "  # heart activation")
    script.blank()
    script.comment("Free the EP solver's matrices before assembling the lead-field ones.")
    script.code("del simulator")
    script.code("""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    """)

    # 4. lead fields ----------------------------------------------------------------------
    script.step("Lead fields", "leadfield", "Lead fields",
                note="One reciprocal solve per electrode on the whole heart-torso mesh; "
                     "TorchCor prints a line per electrode.")
    script.code("lf = LeadField(TORSO_DIR, HEART_DIR, device=DEVICE, dtype=DTYPE)")
    script.comment("Passive torso and organ conductivities (S/m, isotropic).")
    for row in conductor.torso:
        line = codegen.call("lf.add_torso_conductivity", lit([int(t) for t in row.tags]),
                            g=lit(float(row.g)))
        if row.tissue and "\n" in line:
            script.comment(row.tissue)
        script.code(line + (f"  # {row.tissue}" if row.tissue and "\n" not in line else ""))
    script.comment("Anisotropic bidomain heart conductivities (intra/extra, S/m).")
    for row in conductor.heart:
        script.code(codegen.call("lf.add_heart_conductivity",
                                 lit([int(r) for r in row.regions]), il=lit(float(row.il)),
                                 it=lit(float(row.it)), el=lit(float(row.el)),
                                 et=lit(float(row.et))))
    script.code("lf.build()")
    if data.electrode_order.replace(" ", "") == ELECTRODE_ORDER.replace(" ", ""):
        script.code("lf.load_electrodes(ELECTRODES)  # V1-V6, RA, LA, RL, LL (RL = ground)")
    else:
        names = [n.strip() for n in data.electrode_order.split(",") if n.strip()]
        script.code(f"lf.load_electrodes(ELECTRODES, names={lit(names)})  # RL = ground")
    script.code(codegen.call("lf.precompute_all", a_tol=lit(conductor.a_tol),
                             r_tol=lit(conductor.r_tol), max_iter=lit(int(conductor.max_iter))))

    # 5. ecg ------------------------------------------------------------------------------
    script.step("12-lead ECG", "ecg", "12-lead ECG")
    script.code("""
        ecg = lf.compute_12lead(Vm)  # I, II, III, aVR, aVL, aVF, V1..V6 in mV
        np.savez(RESULTS / "ecg.npz", **{lead: v.cpu().numpy() for lead, v in ecg.items()})
    """)
    if out.plot:
        script.code('lf.plot_ecg(ecg, filename=RESULTS / "ecg_12lead.png", dt=SAMPLE_INTERVAL)')

    # 6. save -----------------------------------------------------------------------------
    script.step("Save", "export", "Saving results")
    script.code("""
        studio.export_ecg(ecg, SAMPLE_INTERVAL, bundle)
        bundle.write()

        studio.stage("done", "Completed")
    """)
    return script.render()


# =============================================================================== spec
SPEC = register(SimulationSpec(
    kind=SimulationKind.ECG,
    title="ECG",
    tagline="12-lead ECG from a heart in a torso",
    description="Simulate a ventricular beat and compute the 12-lead ECG on a prepared "
                "heart-torso model with the lead-field method.",
    icon="ecg",
    params_model=ECGParams,
    build_script=build_script,
    stages=STAGES,
    demos=DEMOS,
    panels=PANELS,
    check=check,
    input_use=input_use,
    badge="ECG",
))
