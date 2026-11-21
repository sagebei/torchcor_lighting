"""12-lead ECG: ventricular electrophysiology on a heart-torso model, then lead fields.

The default reproduces TorchCor's ``demo/ecg`` recipe on the KCL whole-torso model 3,
which the script downloads from Zenodo and prepares on first use (into the input folder,
or a per-user data folder on the compute machine when none is chosen).
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
)
from torchcor_studio.core.simulations.fields import Section, param
from torchcor_studio.core.simulations.panels import INPUT, OUTPUT, Panel

SOURCES = {"kcl": "KCL torso 3", "custom": "Prepared folder"}
KCL_FOLDER = "KCL_torso3"
#: Where a run with no input folder downloads KCL torso 3 (on the compute machine).
KCL_SHARED = "~/TorchCorStudio/data/KCL_torso3"
KCL_SITES = ("LV_septal", "LV_anterior", "RV_septal", "RV_moderator")
#: Region tags of the prepared KCL heart mesh: RV wall, LV wall, fast endocardial layer.
KCL_HEART_REGIONS = (34, 35, 36)
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
    """The heart-torso model and where its parts are."""

    source: str = param("kcl", "Model", choices=SOURCES, widget="segmented",
                        help="KCL torso 3 (Zenodo 10.5281/zenodo.7253863, CC-BY-4.0) is "
                             "downloaded and prepared on first use; or use your own prepared "
                             "heart-torso folder.")
    torso_mesh: str = param(KCL_FOLDER, "Torso mesh folder", visible_if={"source": "custom"},
                            span=1, help="Sub-folder of the input folder with the full heart-torso mesh.")
    heart_mesh: str = param("heart", "Heart mesh folder", visible_if={"source": "custom"},
                            span=1, help="Sub-folder with the ventricles cut out of the torso mesh, in "
                                 "torso node order (as LeadField requires).")
    pacing: str = param("pacing", "Pacing sites folder", visible_if={"source": "custom"},
                        span=1, help="Sub-folder holding the stimulus .vtx files.")
    electrodes: str = param("electrodes/lf_src.vtx", "Electrodes", visible_if={"source": "custom"},
                            span=1, help="The 10 electrode nodes of the torso mesh, relative to the "
                                 "input folder.")
    electrode_order: str = param(ELECTRODE_ORDER, "Electrode order", advanced=True,
                                 visible_if={"source": "custom"}, widget="mono",
                                 help="Names of the electrodes in file order; RL is the ground.")
    fields: str = param("fields.npz", "Heart coordinates", visible_if={"source": "custom"},
                        advanced=True,
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
    conductivity_mode: str = param("monodomain", "Conductivity values",
                                   choices=common.CONDUCTIVITY_MODES,
                                   help="The KCL values are effective monodomain values tuned "
                                        "for its ~0.9 mm mesh.")
    conductivities: list[Conductivity] = param(
        title="Conductivities",
        default_factory=lambda: [Conductivity(regions=[34, 35], il=0.408, it=0.141),
                                 Conductivity(regions=[36], il=1.5, it=1.5)],
        item="region group",
        help="Heart conductivity per region: walls 34/35 and the fast endocardial layer 36.")
    stimuli: list[Stimulus] = param(
        title="Early activation sites",
        default_factory=lambda: [Stimulus(site=f"{site}.vtx", start=start, duration=1.0,
                                          intensity=100.0)
                                 for site, start in (("LV_septal", 0.0), ("LV_anterior", 0.0),
                                                     ("RV_septal", 10.0),
                                                     ("RV_moderator", 10.0))],
        item="site", help="Durrer-type sites in the pacing folder.")
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

    data: ECGData = param(title="Heart and torso", default_factory=ECGData)
    heart: ECGHeart = param(title="Heart electrophysiology", default_factory=ECGHeart)
    conductor: VolumeConductor = param(title="Volume conductor", default_factory=VolumeConductor)
    numerics: SolverSettings = param(title="Numerics",
                                     default_factory=lambda: SolverSettings(precision="float64"))
    outputs: ECGOutputs = param(title="Outputs", default_factory=ECGOutputs)

    def variant_label(self) -> str:
        return "12-lead · KCL torso 3" if self.data.source == "kcl" else "12-lead"


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
         requires=KCL_PREPARED, download=True, source="demo/ecg/demo_ecg.py", data="KCL torso 3"),
    Demo("uniform", "KCL torso 3, no repolarisation gradients",
         "The same beat without the IKs and Ito gradients.", no_gradients, case="ecg",
         requires=KCL_PREPARED, download=True, source="demo/ecg/demo_ecg.py", data="KCL torso 3"),
)

PANELS = (
    Panel("Heart and torso", "user", ("data.source", INPUT, "data", OUTPUT)),
    Panel("Cells", "activity", ("heart.model", "heart.cell_type", "heart.heterogeneity")),
    Panel("Conduction", "wave", ("heart.conductivity_mode", "heart.conductivities",
                                 "heart.mass_lumping")),
    Panel("Early activation", "zap", ("heart.stimuli",)),
    Panel("Torso conductivities", "layers", (("conductor.torso", "Torso tissues"), "conductor.heart"),
          collapsed=True),
    Panel("Time and outputs", "clock", ("heart.duration", "heart.dt", "heart.sample_interval",
                                        "outputs")),
    Panel("Solver", "cpu", (("numerics", "Heart solve"), ("conductor", "Lead fields")),
          collapsed=True),
)


def input_use(params: ECGParams) -> tuple[InputUse, str]:
    if params.data.source == "kcl":  # the hint shows while the folder is empty
        return "optional", (f"Empty: KCL torso 3 (122 MB) is downloaded into {KCL_SHARED} on the "
                            "first run, then reused. Choose a folder to keep it elsewhere.")
    return "required", ("A prepared heart–torso folder: torso and heart meshes, pacing sites "
                        "and electrodes.")


# =============================================================================== checks
def _kcl_prepared(info: InputInfo) -> bool:
    return set(KCL_PREPARED) <= set(info.subdirs) and "fields.npz" in info.files


def check(params: ECGParams, info: InputInfo | None) -> list[Issue]:
    issues: list[Issue] = []
    data, heart = params.data, params.heart
    known: list[int] | None = None
    if data.source == "kcl":
        known = list(KCL_HEART_REGIONS)
        # an empty or new input folder says where the model goes; a folder with other
        # files is not one the user would expect it in
        if info is not None and info.exists and not _kcl_prepared(info):
            issues.append(Issue(Severity.INFO, "KCL torso 3 (122 MB) will be downloaded into the "
                                               "input folder and prepared there on the first run.",
                                INPUT))
        sites = {f"{s}.vtx" for s in KCL_SITES}
        for i, stim in enumerate(heart.stimuli):
            if stim.site not in sites:
                issues.append(Issue(Severity.ERROR,
                                    f"The KCL pacing sites are {', '.join(sorted(sites))}.",
                                    f"heart.stimuli.{i}.site"))
    else:
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
    issues += common.conductivity_issues(heart.conductivities, heart.conductivity_mode, known,
                                         "heart.conductivities")
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
def _data_dir(params: ECGParams, ctx: ScriptContext) -> tuple[str, str]:
    if ctx.input_dir:
        return codegen.path(ctx.input_dir, ctx.windows), "the input folder"
    return ('Path.home() / "TorchCorStudio" / "data" / "KCL_torso3"',
            "shared by every run on this machine")


def build_script(params: ECGParams, ctx: ScriptContext) -> str:
    data, heart, conductor, num, out = (params.data, params.heart, params.conductor,
                                        params.numerics, params.outputs)
    lit = codegen.lit
    kcl = data.source == "kcl"
    model = CELL_MODELS[heart.model]
    gradients = heart.heterogeneity and heart.model == "TenTusscherPanfilov" and bool(
        kcl or data.fields)
    title = ctx.run_name or "ECG run"
    script = codegen.Script(codegen.docstring(
        title, "ECG · 12-lead ECG from ventricular electrophysiology by the lead-field "
               "(reciprocity) method.", ctx,
        [f"Heart-torso model: {SOURCES[data.source] if kcl else 'a prepared folder'}.",
         f"Cell model: {model.label}, {CELL_TYPES[heart.cell_type].lower()}"
         + (", with transmural and apico-basal repolarisation gradients." if gradients else "."),
         f"{heart.duration:g} ms simulated at dt = {heart.dt:g} ms; ECG sampled every "
         f"{heart.sample_interval:g} ms."]))
    script.imports.add("import numpy as np")
    script.imports.add("import torchcor as tc")
    script.imports.add("from torchcor.electrocardiogram import LeadField")
    script.imports.add("from torchcor.electrophysiology import Monodomain")
    script.imports.add(f"from torchcor.ionic import {heart.model}")

    s = script.settings
    codegen.device_setting(script, ctx)
    s.add("DTYPE", common.dtype_expr(num.precision), "Vm precision (lead fields run in float64)")
    s.add("DURATION", lit(float(heart.duration)), "ms: one beat with its T wave")
    s.add("DT", lit(float(heart.dt)), "ms per EP time step")
    s.add("SAMPLE_INTERVAL", lit(float(heart.sample_interval)), "ms between Vm frames = ECG samples")
    codegen.output_settings(script)
    data_expr, data_note = _data_dir(params, ctx)
    s.add("DATA", data_expr, f"heart-torso model ({data_note})")
    if kcl:
        s.add("TORSO_DIR", f'DATA / "{KCL_FOLDER}"', "full heart-torso mesh (µm)")
        s.add("HEART_DIR", 'DATA / "heart"', "ventricles, in torso node order")
        s.add("PACING_DIR", 'DATA / "pacing"', "early-activation sites")
        s.add("ELECTRODES", 'DATA / "electrodes" / "lf_src.vtx"', "V1-V6, RA, LA, RL, LL")
        s.add("FIELDS", 'DATA / "fields.npz"', "apico-basal and transmural coordinates")
    else:
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
    script.step("Heart-torso model", "data", "Preparing the heart-torso model")
    if kcl:
        script.comment("KCL whole-torso model 3 (Qian, Bishop et al., King's College London; "
                       "Zenodo 10.5281/zenodo.7253863, CC-BY-4.0): downloaded and prepared "
                       "(heart, pacing sites, electrodes, coordinates) on first use, then reused.")
        script.code("studio.prepare_kcl_torso(DATA)")
    else:
        script.code("""
            for path in (TORSO_DIR, HEART_DIR, PACING_DIR, ELECTRODES):
                if not path.exists():
                    raise SystemExit(f"missing {path}")
        """)

    # 3. heart EP -------------------------------------------------------------------------
    script.step("Heart electrophysiology", "ep", "Heart electrophysiology",
                note="Monodomain EP on the ventricles gives the transmembrane potential Vm "
                     "that the lead fields turn into body-surface potentials.")
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
    sim_kwargs = {"T": "DURATION", "dt": "DT", "device": "DEVICE", "dtype": "DTYPE"}
    if heart.mass_lumping:
        sim_kwargs["mass_lumping"] = "True"
    script.code(codegen.call("Monodomain", "[cell]", prefix="simulator = ", **sim_kwargs))
    script.code("simulator.load_mesh(HEART_DIR, unit_conversion=1000)  # µm -> mm")
    script.comment("Heart conductivities (S/m); the KCL values are tuned for its ~0.9 mm mesh "
                   "(about 80-100 ms total activation), region 36 is the fast endocardial layer.")
    script.code(common.conductivity_code(heart.conductivities, heart.conductivity_mode))
    script.comment("Early-activation sites: strength in mV/ms, times in ms.")
    for stim in heart.stimuli:
        script.code(codegen.call("simulator.add_stimulus", f"PACING_DIR / {lit(stim.site)}",
                                 **common.stimulus_kwargs(stim)))
    script.code(codegen.call(
        "simulator.solve", prefix="Vm = ", a_tol=lit(num.a_tol), r_tol=lit(num.r_tol),
        max_iter=lit(int(num.max_iter)), linear_guess=lit(bool(num.linear_guess)),
        snapshot_interval="SAMPLE_INTERVAL", verbose="True", result_path="RESULTS") + ".cpu()")
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
    if kcl or data.electrode_order.replace(" ", "") == ELECTRODE_ORDER.replace(" ", ""):
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
    description="Simulate a ventricular beat and compute the 12-lead ECG on a CT-based "
                "heart-torso model with the lead-field method. The KCL torso model is "
                "downloaded and prepared automatically on the first run.",
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
