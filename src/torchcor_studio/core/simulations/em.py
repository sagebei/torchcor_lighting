"""Electromechanics: beats of a four-chamber heart coupled to a closed-loop circulation.

The generated script is a faithful port of TorchCor's ``demo/em/main.py`` (one virtual
patient of the HCM3 cohort): every module constant of the demo is a parameter here with
the demo's value as default, and the script adds the studio's stages, progress and
visualisation bundle. Only the heart's mesh folder is read from disk.
"""

from __future__ import annotations

import math
from string import Template

from torchcor_studio.core.models import InputInfo
from torchcor_studio.core.simulations import codegen, common
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

CHAMBERS = ("lv", "rv", "la", "ra")
_VENTRICLE_TYPES = {"ENDO": "ENDO", "MID": "M", "EPI": "EPI"}  # ToRORdDynCl spelling

STAGES = (
    Stage("setup", "Setting up"),
    Stage("unload", "Unloading"),
    Stage("load", "Inflating"),
    Stage("prepace", "Pre-pacing"),
    Stage("beats", "Beating"),
    Stage("export", "Saving results"),
    Stage("done", "Completed"),
)


# =============================================================================== parameters
class EMHeart(Section):
    """The heart's mesh folder (the input folder): its mesh, and the names of its other files."""

    lv_endo: str = param("LV_endo", "LV cavity surface", advanced=True,
                         group="Cavity surfaces", short="LV", span=1,
                         help=".surf lining the left-ventricular cavity.")
    rv_endo: str = param("RV_endo", "RV cavity surface", advanced=True,
                         group="Cavity surfaces", short="RV", span=1,
                         help=".surf lining the right-ventricular cavity.")
    la_endo: str = param("LA_endo", "LA cavity surface", advanced=True,
                         group="Cavity surfaces", short="LA", span=1,
                         help=".surf lining the left-atrial cavity.")
    ra_endo: str = param("RA_endo", "RA cavity surface", advanced=True,
                         group="Cavity surfaces", short="RA", span=1,
                         help=".surf lining the right-atrial cavity.")
    pericardium_scale: str = param("pericardium_scale.dat", "Pericardium scaling", advanced=True,
                                   help="Per-element scale of the pericardial spring stiffness.")
    reuse_dir: str = param("", "Reuse paced states from", widget="dir", span=2,
                           placeholder="compute them in this run",
                           help="Folder with unloaded.pt and paced cell states from an earlier "
                                "run of the same heart and parameters; empty = this run's "
                                "results folder.")

    def cavities(self) -> dict[str, str]:
        return {"lv": self.lv_endo, "rv": self.rv_endo, "la": self.la_endo, "ra": self.ra_endo}


class PacingSite(Section):
    """A pacing site (a row of the "Pacing" table)."""

    site: str = param("SAN", "Site", help="A .vtx file of the mesh folder (without extension).")
    first: float = param(0.0, "First firing", unit="ms", min=0.0, step=10.0,
                         help="Time of the first pulse; it repeats every cycle.")


class EMRhythm(Section):
    """Heart rate, pacing and time steps. Time 0 is the start of ventricular activation."""

    cycle_length: float = param(833.0, "Cycle length", unit="ms", min=200.0, max=5000.0,
                                step=10.0, help="833 ms = 72 beats per minute.")
    beats: int = param(5, "Beats", min=1, max=100, help="Number of heartbeats simulated.")
    stimulus_intensity: float = param(60.0, "Stimulus strength", unit="mV/ms", min=0.0,
                                      help="Strength of each pacing pulse.")
    stimulus_duration: float = param(2.0, "Stimulus duration", unit="ms", min=0.01,
                                     help="Length of each pacing pulse.")
    pacing: list[PacingSite] = param(
        title="Pacing",
        default_factory=lambda: [PacingSite(site="SAN", first=833.0 - 120.0),
                                 PacingSite(site="fascicles_lv", first=0.0),
                                 PacingSite(site="fascicles_rv", first=10.0)],
        item="site", help="Sinus node 120 ms before the ventricles, then the LV and RV fascicles.")
    ep_dt: float = param(0.02, "EP time step", unit="ms", min=0.001, max=0.1, decimals=3,
                         help="Step of activation, cells and contraction.")
    mechanics_dt: float = param(1.0, "Mechanics interval", unit="ms", min=0.1, max=20.0,
                                help="Time between mechanics solves.")
    duration: float = param(0.0, "Duration override", unit="ms", min=0.0, step=100.0,
                            advanced=True, placeholder="Whole beats",
                            help="Simulate this long instead of whole beats (0).")

    def total(self) -> float:
        return self.duration if self.duration > 0 else self.beats * self.cycle_length


class ConductionRegion(Section):
    """Conduction velocities of one tissue (a row of the conduction table)."""

    name: str = param("", "Tissue", help="Name for your reference.")
    tags: list[int] = param(title="Region tags", widget="regions", default_factory=list)
    vel_l: float = param(0.6, "Along fibres", unit="mm/ms", min=1e-6, decimals=6,
                         short="Along fibres")
    vel_t: float = param(0.25, "Across, in sheet", unit="mm/ms", min=1e-6, decimals=6,
                         short="Across, in sheet")
    vel_n: float = param(0.25, "Sheet normal", unit="mm/ms", min=1e-6, decimals=6,
                         short="Sheet normal")


def _conduction() -> list[ConductionRegion]:
    return [ConductionRegion(name=n, tags=t, vel_l=along, vel_t=across, vel_n=normal)
            for n, t, along, across, normal in (
        ("ventricles", [1, 2], 0.646649, 0.258660, 0.258660),
        ("fast endocardium", [25, 28, 29], 3.233245, 1.293298, 1.293298),
        ("atria", [3, 4], 0.711895, 0.284758, 0.284758),
        ("Bachmann's bundle", [26], 2.491633, 0.996653, 0.996653))]


class EMConduction(Section):
    """Reaction–eikonal activation."""

    regions: list[ConductionRegion] = param(title="Conduction velocities",
                                            default_factory=_conduction, item="tissue")
    passive_tags: list[int] = param(
        title="Non-conducting tags", widget="regions",
        default_factory=lambda: [5, 6, *range(7, 25), 27],
        help="Valves, vessels and rings: no cells, no conduction.")


class EMCells(Section):
    """Cell models: ToR-ORd-dynCl in the ventricles, Courtemanche in the atria."""

    ventricular_cell_type: str = param("ENDO", "Ventricular cell type", choices=common.CELL_TYPES)
    prepace_beats: int = param(200, "Pre-pacing beats", min=1, max=5000,
                               help="Each cell model alone, before the heart beats.")
    land_beats: int = param(3, "Beats with Land", min=0, max=100,
                            help="The last pre-pacing beats with the contraction model beside it.")
    ventricular: str = param("PCa_b=1.21492e-4, Gncx_b=1.50438e-3, trpnmax=0.0996883, "
                             "Nae=140.0, clo=150.0", "Ventricular parameters (ToR-ORd-dynCl)",
                             widget="mono", help="ToR-ORd-dynCl parameters set to these values.")
    atrial: str = param("GCaL=0.160342, maxIup=0.006237, maxTrpn=0.0639378",
                        "Atrial parameters (Courtemanche)", widget="mono",
                        help="Courtemanche parameters set to these values.")
    pacing_stimulus: float = param(60.0, "Stimulus strength", unit="mV/ms", min=0.0,
                                   advanced=True, help="Pulse strength while pre-pacing.")
    pacing_duration: float = param(1.0, "Stimulus duration", unit="ms", min=0.01, advanced=True,
                                   help="Pulse length while pre-pacing.")


_LAND_VENTRICLES = ("TRPN_50=0.343453, n_Tm=2.32157, n_TRPN=2.83059, r_s=0.183866, r_w=0.639168, "
                    "A_eff=27.4896, k_u=0.0176373, beta_1=-1.537282, ca50=0.51563, k_ws=0.0119524")
_LAND_ATRIA = ("T_ref=66.5134, TRPN_50=0.391704, n_Tm=6.6936, n_TRPN=1.54176, r_s=0.202019, "
               "r_w=0.654954, A_eff=13.5921, beta_1=-2.216336, phi=1.59582, ca50=0.794187, "
               "k_ws=0.0273741")


class EMContraction(Section):
    """Land contraction model per region (LandParameters defaults otherwise)."""

    lv_tags: list[int] = param(title="LV tags", widget="regions", span=1,
                               default_factory=lambda: [1, 25, 29])
    rv_tags: list[int] = param(title="RV tags", widget="regions", span=1,
                               default_factory=lambda: [2, 28])
    atria_tags: list[int] = param(title="Atrial tags", widget="regions", span=1,
                                  default_factory=lambda: [3, 4, 26])
    lv_tref: float = param(175.668, "LV reference tension", unit="kPa", min=0.0,
                           group="Reference tension", short="LV",
                           help="Land's T_ref in the left ventricle.")
    rv_tref: float = param(118.028, "RV reference tension", unit="kPa", min=0.0,
                           group="Reference tension", short="RV",
                           help="Land's T_ref in the right ventricle.")
    ventricles: str = param(_LAND_VENTRICLES, "Ventricular Land parameters", advanced=True,
                            widget="mono")
    atria: str = param(_LAND_ATRIA, "Atrial Land parameters", advanced=True, widget="mono")


class GuccioneRegion(Section):
    tissue: str = param("", "Tissue")
    tags: list[int] = param(title="Region tags", widget="regions", default_factory=list)
    a: float = param(2.0, "Stiffness a", unit="kPa", min=0.0, decimals=6, short="a")
    b_f: float = param(8.0, "Fibre exponent", min=0.0, short="b_f")
    b_t: float = param(3.0, "Transverse exponent", min=0.0, short="b_t")
    b_fs: float = param(4.0, "Shear exponent", min=0.0, short="b_fs")


class NeoHookeanRegion(Section):
    tissue: str = param("", "Tissue")
    tags: list[int] = param(title="Region tags", widget="regions", default_factory=list)
    mu: float = param(10.0, "Shear modulus", unit="kPa", min=0.0, decimals=4, short="μ")


class EMPassive(Section):
    """Passive tissue (kPa): Guccione myocardium, neo-Hookean valves and vessels."""

    guccione: list[GuccioneRegion] = param(title="Myocardium (Guccione)", item="Guccione region",
                                           summary=True, default_factory=lambda: [
        GuccioneRegion(tissue="LV", tags=[1, 25, 29, 27], a=2.46687, b_f=8.0, b_t=3.0, b_fs=4.0),
        GuccioneRegion(tissue="RV", tags=[2, 28], a=3.700305, b_f=8.0, b_t=3.0, b_fs=4.0),
        GuccioneRegion(tissue="atria", tags=[3, 4, 26], a=1.80867, b_f=16.0, b_t=6.0, b_fs=8.0)])
    neo_hookean: list[NeoHookeanRegion] = param(title="Valves and vessels (neo-Hookean)",
                                                item="neo-Hookean region", summary=True,
                                                default_factory=lambda: [
        NeoHookeanRegion(tissue="valve planes", tags=list(range(7, 18)), mu=1000.0),
        NeoHookeanRegion(tissue="aorta", tags=[5], mu=26.66),
        NeoHookeanRegion(tissue="pulmonary artery", tags=[6], mu=3.7),
        NeoHookeanRegion(tissue="vein rings", tags=list(range(18, 25)), mu=7.45)])
    kappa: float = param(1000.0, "Volumetric penalty κ", unit="kPa", min=0.0, short="κ",
                         summary=True, help="Penalty κ/2 ln(J)² on volume change.")


class Spring(Section):
    surface: str = param("epicardium_for_sim", "Surface", help=".surf file (without extension).")
    stiffness: float = param(1.0, "Stiffness", unit="kPa/mm", min=0.0, decimals=4)
    normal_only: bool = param(False, "Normal only", help="Restrain only the normal direction.")
    scaled: bool = param(False, "Scaled", help="Scale by the pericardium map, element by element.")


class EMSupport(Section):
    """Springs, start pressures and first-order dynamics."""

    springs: list[Spring] = param(title="Springs", item="spring", summary=True,
                                  default_factory=lambda: [
        Spring(surface="epicardium_for_sim", stiffness=12.667, normal_only=True, scaled=True),
        Spring(surface="RPVs", stiffness=1.0), Spring(surface="SVC", stiffness=1.0)])
    start_lv: float = param(9.53762, "LV start pressure", unit="mmHg", min=0.0,
                            group="Start pressures", short="LV", summary=True)
    start_rv: float = param(5.93714, "RV start pressure", unit="mmHg", min=0.0,
                            group="Start pressures", short="RV", summary=True)
    start_la: float = param(9.53762, "LA start pressure", unit="mmHg", min=0.0,
                            group="Start pressures", short="LA", summary=True)
    start_ra: float = param(5.93714, "RA start pressure", unit="mmHg", min=0.0,
                            group="Start pressures", short="RA", summary=True)
    density: float = param(1.06, "Density", unit="kPa·ms²/mm²", min=0.0, group="Dynamics")
    rho_infinity: float = param(0.0, "High-frequency damping ρ∞", min=0.0, max=1.0,
                                group="Dynamics", short="Damping ρ∞")
    mass_damping: float = param(0.1, "Mass damping", min=0.0, group="Dynamics",
                                help="Rayleigh damping, mass part.")
    stiffness_damping: float = param(0.1, "Stiffness damping", min=0.0, group="Dynamics",
                                     help="Rayleigh damping, stiffness part.")

    def start(self) -> dict[str, float]:
        return {"lv": self.start_lv, "rv": self.start_rv, "la": self.start_la, "ra": self.start_ra}


class Tube(Section):
    name: str = param("AO", "Vessel")
    length: float = param(400.0, "Length", unit="mm", min=0.0, decimals=5)
    wall_area: float = param(150.0, "Wall area", unit="mm²", min=0.0, decimals=6)
    reference_area: float = param(400.0, "Reference area", unit="mm²", min=0.0, decimals=4,
                                  short="Ref. area")
    reference_pressure: float = param(100.0, "Reference pressure", unit="mmHg", min=0.0,
                                      decimals=5, short="Ref. pressure")
    stiffness_exponent: float = param(8.0, "Stiffness exponent", min=0.0, decimals=5,
                                      short="Stiffness exp.")


class Valve(Section):
    name: str = param("aortic", "Valve")
    open_area: float = param(400.0, "Open area", unit="mm²", min=0.0, decimals=4, short="Open")
    leak_area: float = param(0.0, "Leak area", unit="mm²", min=0.0, decimals=4, short="Leak")


class Peripheral(Section):
    name: str = param("systemic", "Bed")
    reference_drop: float = param(90.0, "Reference pressure drop", unit="mmHg", min=0.0,
                                  decimals=6, short="Pressure drop")
    resistance_factor: float = param(1.0, "Resistance factor", min=0.0, decimals=6,
                                     short="Resistance ×")


class Chamber(Section):
    name: str = param("LV", "Chamber")
    wall_volume: float = param(100.0, "Wall volume", unit="mL", min=0.0, decimals=4,
                               short="Wall")
    initial_volume: float = param(100.0, "Initial volume", unit="mL", min=0.0, decimals=5,
                                  short="Initial")


class EMCirculation(Section):
    """CircAdapt 2015 closed loop, coupled through the four cavity volumes."""

    blood_density: float = param(1050.0, "Blood density", unit="kg/m³", min=0.0)
    reference_flow: float = param(85.0, "Reference flow", unit="mL/s", min=0.0, summary=True)
    tubes: list[Tube] = param(title="Tubes", fixed=True, item="tube", summary=True,
                              default_factory=lambda: [
        Tube(name=n, length=a, wall_area=b, reference_area=c, reference_pressure=d,
             stiffness_exponent=e) for n, a, b, c, d, e in (
            ("aorta", 400.1115, 157.376262, 417.183, 131.136, 7.88972),
            ("pulmonary artery", 200.05575, 171.099463, 500.0, 9.37164, 8.0),
            ("venae cavae", 400.1115, 100.650528, 756.622, 9.40684, 10.0),
            ("pulmonary veins", 200.05575, 76.285933, 425.747, 3.75, 10.0))],
        help="In this order: aorta, pulmonary artery, venae cavae, pulmonary veins.")
    valves: list[Valve] = param(title="Valves", fixed=True, item="valve", summary=True,
                                default_factory=lambda: [
        Valve(name=n, open_area=a, leak_area=b) for n, a, b in (
            ("aortic", 417.183, 0.0), ("systemic venous orifice", 756.622, 756.622),
            ("tricuspid", 1380.1, 0.0), ("pulmonary", 500.0, 0.0),
            ("pulmonary venous orifice", 425.747, 425.747), ("mitral", 1329.83, 0.0))],
        help="In this order: aortic, systemic venous orifice, tricuspid, pulmonary, pulmonary "
             "venous orifice, mitral (the orifices have no valve).")
    chambers: list[Chamber] = param(title="Chambers", fixed=True, item="chamber", summary=True,
                                    default_factory=lambda: [
        Chamber(name=n, wall_volume=a, initial_volume=b) for n, a, b in (
            ("LV", 100.0, 108.375), ("RV", 50.0, 78.03), ("LA", 12.0, 54.1875),
            ("RA", 7.0, 44.43375))], help="LV, RV, LA, RA.")
    peripheral: list[Peripheral] = param(title="Peripheral beds", fixed=True, item="bed",
                                          summary=True, default_factory=lambda: [
        Peripheral(name="systemic", reference_drop=90.007389, resistance_factor=1.28866),
        Peripheral(name="pulmonary", reference_drop=11.250924, resistance_factor=0.846025)],
        help="Systemic, then pulmonary.")


class EMSolver(Section):
    """Tolerances and output."""

    volume_tolerance: float = param(0.001, "Volume tolerance", unit="mL", min=1e-8, log=True,
                                    summary=True, help="Heart and circulation volumes agree to this.")
    unloading_tolerance: float = param(0.01, "Unloading tolerance", min=1e-6, max=0.5,
                                       summary=True, help="Re-inflated cavity volumes within this fraction.")
    frame_interval: float = param(10.0, "Frame interval", unit="ms", min=1.0,
                                  help="Time between saved surface frames (VTK and the app).")
    chunk: int = param(32768, "Assembly chunk", min=256, max=10_000_000, advanced=True,
                       help="Tetrahedra assembled together; bounds peak memory.")


class EMParams(Section):
    """Electromechanics parameters."""

    heart: EMHeart = param(title="Heart mesh", default_factory=EMHeart)
    rhythm: EMRhythm = param(title="Rhythm", default_factory=EMRhythm)
    conduction: EMConduction = param(title="Conduction", default_factory=EMConduction)
    cells: EMCells = param(title="Cell models", default_factory=EMCells)
    contraction: EMContraction = param(title="Land contraction", default_factory=EMContraction)
    passive: EMPassive = param(title="Passive tissue", default_factory=EMPassive)
    support: EMSupport = param(title="Supports and start pressures", default_factory=EMSupport)
    circulation: EMCirculation = param(title="Circulation", default_factory=EMCirculation)
    solver: EMSolver = param(title="Solver", default_factory=EMSolver)

    def variant_label(self) -> str:
        if self.rhythm.duration > 0:
            return f"4-chamber · {self.rhythm.duration:g} ms"
        return f"4-chamber · {self.rhythm.beats} beat{'s' if self.rhythm.beats != 1 else ''}"


# =============================================================================== demos
def one_beat() -> EMParams:
    params = EMParams()
    params.rhythm = EMRhythm(beats=1)
    return params


DEMOS = (
    Demo("five-beats", "Four-chamber heart, five beats",
         "HCM3 virtual patient #1267 on the CircAdapt closed loop.", EMParams, case="em/heart",
         cost=Cost.HOURS, source="demo/em/main.py"),
    Demo("one-beat", "Four-chamber heart, one beat",
         "The same heart for one beat; reuse its paced states to iterate.", one_beat,
         case="em/heart", cost=Cost.HOURS, source="demo/em/main.py"),
)

PANELS = (
    Panel("Heart", "mesh", (INPUT, "heart", OUTPUT)),
    Panel("Cells and contraction", "activity", (
        "cells.ventricular_cell_type", "cells.ventricular", "cells.atrial",
        ("cells.prepace_beats", "Pre-pacing"), "cells.land_beats", "cells.pacing_stimulus",
        "cells.pacing_duration", "heart.reuse_dir",
        ("contraction", "Land contraction"))),
    Panel("Conduction", "wave", ("conduction",)),
    Panel("Passive tissue", "layers", (("passive.guccione", "Myocardium (Guccione)"), "passive"),
          collapsed=True),
    Panel("Rhythm and pacing", "heart-pulse", ("rhythm",)),
    Panel("Supports and start pressures", "target", ("support",), collapsed=True),
    Panel("Circulation", "repeat", ("circulation",), collapsed=True),
    Panel("Time and outputs", "clock", ("rhythm.ep_dt", "rhythm.mechanics_dt",
                                        "solver.frame_interval", "rhythm.duration")),
    Panel("Solver", "cpu", ("solver",), collapsed=True),
)


def input_use(_params: EMParams) -> tuple[InputUse, str]:
    return "required", ("The heart's mesh folder: .pts, .elem and .lon, cavity .surf and pacing "
                        ".vtx files.")


# =============================================================================== checks
def _assignments(text: str, field: str, issues: list[Issue]) -> list[tuple[str, float]]:
    try:
        return common.parse_scaling(text)
    except ValueError as exc:
        issues.append(Issue(Severity.ERROR, f"{exc}.", field))
        return []


def check(params: EMParams, info: InputInfo | None) -> list[Issue]:
    issues: list[Issue] = []
    heart, rhythm = params.heart, params.rhythm
    for field, text in (("cells.ventricular", params.cells.ventricular),
                        ("cells.atrial", params.cells.atrial),
                        ("contraction.ventricles", params.contraction.ventricles),
                        ("contraction.atria", params.contraction.atria)):
        _assignments(text, field, issues)
    counts = {"tubes": 4, "valves": 6, "peripheral": 2, "chambers": 4}
    for name, count in counts.items():
        if len(getattr(params.circulation, name)) != count:
            issues.append(Issue(Severity.ERROR, f"CircAdapt needs exactly {count} {name}.",
                                f"circulation.{name}"))
    for i, site in enumerate(rhythm.pacing):
        if site.first >= rhythm.cycle_length:
            issues.append(Issue(Severity.WARNING, "This site first fires after the first cycle.",
                                f"rhythm.pacing.{i}.first"))
    issues += common.cell_dt_issues(["ToRORdDynCl", "CourtemancheRamirezNattel"], rhythm.ep_dt,
                                    "rhythm.ep_dt")
    if rhythm.mechanics_dt < rhythm.ep_dt:
        issues.append(Issue(Severity.ERROR, "The mechanics interval must be at least one EP "
                                            "step.", "rhythm.mechanics_dt"))
    if info is None or not info.exists:
        issues.append(Issue(Severity.ERROR, "Choose the heart's mesh folder as the input folder.",
                            INPUT))
        return issues
    files = set(info.files)
    meshes = sorted(f[:-4] for f in files if f.endswith(".pts"))
    if not info.mesh_name:
        issues.append(Issue(Severity.ERROR, "The input folder has no openCARP mesh (.pts, .elem, "
                                            ".lon).", INPUT))
    elif f"{info.mesh_name}.lon" not in files:
        issues.append(Issue(Severity.ERROR, f"{info.mesh_name}.lon (fibres and sheets) is not in "
                                            "the input folder.", INPUT))
    elif len(meshes) > 1:
        issues.append(Issue(Severity.WARNING, f"The input folder holds {len(meshes)} meshes; the "
                                              f"run reads {meshes[0]}.", INPUT))
    surfaces = set(info.surfaces)
    for field, name in (("lv_endo", heart.lv_endo), ("rv_endo", heart.rv_endo),
                        ("la_endo", heart.la_endo), ("ra_endo", heart.ra_endo)):
        if f"{name}.surf" not in surfaces:
            issues.append(Issue(Severity.ERROR, f"{name}.surf is not in the input folder.",
                                f"heart.{field}"))
    for i, spring in enumerate(params.support.springs):
        if f"{spring.surface}.surf" not in surfaces:
            issues.append(Issue(Severity.ERROR, f"{spring.surface}.surf is not in the input "
                                                "folder.", f"support.springs.{i}.surface"))
    if heart.pericardium_scale not in files:
        issues.append(Issue(Severity.ERROR, f"{heart.pericardium_scale} is not in the input "
                                            "folder.", "heart.pericardium_scale"))
    vtx = {v.name for v in info.vtx}
    for i, site in enumerate(rhythm.pacing):
        if f"{site.site}.vtx" not in vtx:
            issues.append(Issue(Severity.ERROR, f"{site.site}.vtx is not in the input folder.",
                                f"rhythm.pacing.{i}.site"))
    if info.regions:
        known = set(info.regions)
        conducting = {t for r in params.conduction.regions for t in r.tags}
        missing = sorted(known - conducting - set(params.conduction.passive_tags))
        if missing:
            issues.append(Issue(Severity.ERROR, f"Regions {', '.join(map(str, missing))} have no "
                                                "conduction velocity and are not listed as "
                                                "non-conducting.", "conduction.regions"))
        materials = ({t for r in params.passive.guccione for t in r.tags}
                     | {t for r in params.passive.neo_hookean for t in r.tags})
        bare = sorted(known - materials)
        if bare:
            issues.append(Issue(Severity.ERROR, f"Regions {', '.join(map(str, bare))} have no "
                                                "passive material.", "passive.guccione"))
    if info.fibre_columns == 3 and any(not math.isclose(r.vel_n, r.vel_t)
                                       for r in params.conduction.regions):
        issues.append(Issue(Severity.ERROR, "The .lon file has no sheet directions, so the "
                                            "sheet-normal velocities must equal the in-sheet "
                                            "ones.", "conduction.regions"))
    issues.append(Issue(Severity.INFO, "Needs a CUDA GPU with plenty of memory; several beats "
                                       "take hours.", "rhythm.beats"))
    return issues


# =============================================================================== script
_SCRIPT = Template('''
# ===================================================================== the heart
MESH = $MESH  # the input folder
NAME = sorted(p.stem for p in MESH.glob("*.pts"))[0]  # its mesh: NAME.pts/.elem/.lon, in um
CHAMBERS = ("lv", "rv", "la", "ra")
CAVITY = $CAVITY  # .surf files
PERICARDIUM_SCALE = $PERICARDIUM_SCALE  # per-element spring scale
MMHG = 0.133322387415  # kPa per mmHg

# ===================================================================== parameters
# Units: ms, mm, kPa (mmHg where named), mm/ms for velocities.

# --- Heart rhythm.  Time 0 is the start of ventricular activation.
CYCLE = $CYCLE  # ms
DURATION = $DURATION  # ms simulated
# .vtx files; first firing time (ms)
STIMULI = $STIMULI
STIMULUS = dict(intensity=$STIMULUS_INTENSITY, duration=$STIMULUS_DURATION)
EP_DT = $EP_DT  # ms, activation, cells and contraction
MECHANICS_DT = $MECHANICS_DT  # ms between mechanics solves

# --- Conduction velocity: tags, then along fibre, across in-sheet, sheet normal.
CONDUCTION = $CONDUCTION
# Tags that do not conduct (valves, vessels, rings).
PASSIVE = $PASSIVE

# --- Cells.  The published models, with these values changed.
VENTRICULAR_CELL = $VENTRICULAR_CELL
ATRIAL_CELL = $ATRIAL_CELL
PREPACE_BEATS = $PREPACE_BEATS  # each cell alone, before the heart beats
LAND_BEATS = $LAND_BEATS  # the last of them with Land beside it
PACING = dict(stimulus=$PACING_STIMULUS, duration=$PACING_DURATION)  # single-cell stimulus

# --- Land contraction.  LandParameters defaults, with these values changed.
LAND_VENTRICLES = $LAND_VENTRICLES
LAND_ATRIA = $LAND_ATRIA

# --- Contracting regions: tags, cell model, cell type, Land parameters (T_ref in kPa).
REGIONS = [($LV_TAGS, ToRORdDynCl, $CELL_TYPE, {**LAND_VENTRICLES, "T_ref": $LV_TREF}),  # LV
           ($RV_TAGS, ToRORdDynCl, $CELL_TYPE, {**LAND_VENTRICLES, "T_ref": $RV_TREF}),  # RV
           ($ATRIA_TAGS, CourtemancheRamirezNattel, None, LAND_ATRIA)]  # atria

# --- Passive tissue, kPa.  Guccione myocardium and neo-Hookean valves and
#     vessels, all with the volumetric penalty KAPPA/2 ln(J)^2.
GUCCIONE = $GUCCIONE
NEO_HOOKEAN = $NEO_HOOKEAN
KAPPA = $KAPPA

# --- Supports: springs on surfaces, kPa/mm.  A scaled spring is scaled element by
#     element by PERICARDIUM_SCALE.
SPRINGS = $SPRINGS

# --- Mechanics in time: first-order dynamics with Rayleigh damping.
DYNAMICS = dict(density=$DENSITY, rho_infinity=$RHO_INFINITY, mass_damping=$MASS_DAMPING,
                stiffness_damping=$STIFFNESS_DAMPING)

# --- Start pressures (mmHg): the heart is imaged at these, and the beats start from them.
START = $START

# --- Circulation (CircAdapt 2015).  Tubes: aorta, pulmonary artery, venae cavae,
#     pulmonary veins.  Valves: aortic, systemic venous orifice, tricuspid,
#     pulmonary, pulmonary venous orifice, mitral.  Chambers: LV, RV, LA, RA.
CIRCULATION = CircAdaptParameters(
    blood_density_kg_m3=$BLOOD_DENSITY,
    tube_length_mm=$TUBE_LENGTH,
    tube_wall_area_mm2=$TUBE_WALL_AREA,
    tube_reference_area_mm2=$TUBE_REFERENCE_AREA,
    tube_reference_pressure_mmhg=$TUBE_REFERENCE_PRESSURE,
    tube_stiffness_exponent=$TUBE_STIFFNESS_EXPONENT,
    valve_open_area_mm2=$VALVE_OPEN_AREA,
    valve_leak_area_mm2=$VALVE_LEAK_AREA,
    reference_flow_ml_s=$REFERENCE_FLOW,
    peripheral_reference_drop_mmhg=$PERIPHERAL_DROP,  # systemic, pulmonary
    peripheral_resistance_factor=$PERIPHERAL_RESISTANCE,
    chamber_wall_volume_ml=$CHAMBER_WALL_VOLUME,
    initial_chamber_volume_ml=$CHAMBER_INITIAL_VOLUME)

# --- Solver tolerances.
VOLUME_TOLERANCE = $VOLUME_TOLERANCE  # mL, heart and circulation volumes agree
UNLOADING_TOLERANCE = $UNLOADING_TOLERANCE  # re-inflated cavity volumes within this
FRAME = $FRAME  # ms between surface frames
CHUNK = $CHUNK  # tetrahedra assembled together


# ===================================================================== the model
def read_heart(device, points=None):
    """``(mesh, tags, fibres, sheets, owners)``, in mm, optionally on other ``points``.

    Surfaces are registered after ``points`` is set, since they are oriented on it.
    ``owners`` maps each surface to the tetrahedron of each of its triangles.
    """
    mesh, tags, fibres, sheets = read_carp(MESH / NAME, device=device, dtype=torch.float64)
    if points is not None:
        mesh.points = points
    owners = {s: add_surface(mesh, s, read_surface(MESH / f"{s}.surf"))
              for s in (*CAVITY.values(), *SPRINGS)}
    return mesh, tags, fibres / fibres.norm(dim=-1, keepdim=True), sheets, owners


def per_element(tags, values):
    """One value per tetrahedron from {tags: value}, zero elsewhere."""
    field = torch.zeros(len(tags), dtype=torch.float64, device=tags.device)
    for ids, value in values.items():
        field[torch.isin(tags, torch.tensor(ids, device=tags.device))] = value
    return field


def passive_tissue(tags, unloading):
    """Guccione and neo-Hookean tissue with one volumetric penalty.

    Unloading splits both laws into isochoric and volumetric parts; in the
    beats Guccione sees the full Green strain and only the neo-Hookean part is
    split.
    """
    guccione = GuccioneMaterial(*(per_element(tags, {t: p[k] for t, p in GUCCIONE.items()})
                                  for k in ("a", "b_f", "b_t", "b_fs")))
    neo_hookean = NeoHookeanMaterial(per_element(tags, NEO_HOOKEAN))
    if unloading:
        return VolumetricPenalty(IsochoricMaterial(CompositeMaterial(guccione, neo_hookean)),
                                 KAPPA)
    return VolumetricPenalty(CompositeMaterial(guccione, IsochoricMaterial(neo_hookean)), KAPPA)


def loads_and_supports(mesh, owners, pressure):
    """Pressure ``pressure[c]`` (kPa, a number or a function of time) on each cavity
    lining, and the springs."""
    scale = read_element_data(MESH / PERICARDIUM_SCALE, device=mesh.device)
    bcs = [FollowerPressure.on_surface(mesh, CAVITY[c], pressure[c]) for c in CHAMBERS]
    for surface, s in SPRINGS.items():
        k = s["stiffness"] * scale[owners[surface]][:, None, None] if s["scaled"] else s["stiffness"]
        bcs.append(RobinBC(mesh.face_set(surface), stiffness=k, normal_only=s["normal_only"],
                           name=surface))
    return bcs


def centroid(mesh):
    """One quadrature point per tetrahedron, at its centre."""
    return (torch.tensor([[0.25, 0.25, 0.25]], device=mesh.device, dtype=mesh.dtype),
            torch.tensor([1.0 / 6.0], device=mesh.device, dtype=mesh.dtype))


def unload(device, out):
    """The stress-free nodes that the start pressures inflate onto the imaged heart."""
    path = out / "unloaded.pt"
    if path.exists():
        return torch.load(path, map_location=device)
    mesh, tags, fibres, sheets, owners = read_heart(device)
    pressure = {c: START[c] * MMHG for c in CHAMBERS}
    simulation = Mechanics(mesh, passive_tissue(tags, unloading=True),
                           axes=MaterialAxes(f=fibres, s=sheets),
                           boundary=loads_and_supports(mesh, owners, pressure),
                           bulk_modulus=None, quadrature_order=2, quadrature_rule=centroid(mesh),
                           chunk_size=CHUNK)
    cavities = {c: mesh.face_set(CAVITY[c]).connectivity for c in CHAMBERS}
    points = Unloading(simulation, cavities=cavities, volume_tol=UNLOADING_TOLERANCE,
                       newton_rtol=1e-6, max_newton=15, max_load_step=0.1, max_krylov=5000,
                       reference_continuation=False).run()
    torch.save(points, path)
    return points


def build_heart(device, points):
    """The unloaded heart with its tissue, active stress, loads and supports."""
    mesh, tags, fibres, sheets, owners = read_heart(device, points)
    load = {c: START[c] * MMHG for c in CHAMBERS}  # kPa; the circulation updates it
    boundary = loads_and_supports(mesh, owners, {c: (lambda t, c=c: load[c]) for c in CHAMBERS})
    # After loading, the pericardial springs come to rest at the loaded shape
    # and this pressure takes over the force they carried.
    pericardium = boundary[len(CHAMBERS)]
    prestress = torch.zeros(len(pericardium.face_set.connectivity), device=device,
                            dtype=mesh.dtype)
    boundary.append(FollowerPressure(pericardium.face_set, prestress, name="prestress"))
    tension, stiffness, reference = (torch.zeros(mesh.n_cells, 1, device=device, dtype=mesh.dtype),
                                     torch.zeros(mesh.n_cells, 1, device=device, dtype=mesh.dtype),
                                     torch.ones(mesh.n_cells, 1, device=device, dtype=mesh.dtype))
    material = StabilizedActiveStressMaterial(passive_tissue(tags, unloading=False), tension,
                                              stiffness, reference, stress_measure="pk1")
    simulation = Mechanics(mesh, material, axes=MaterialAxes(f=fibres, s=sheets),
                           boundary=boundary, bulk_modulus=None, quadrature_order=2,
                           quadrature_rule=centroid(mesh), chunk_size=CHUNK,
                           tangent_dtype=torch.float32, compile=True)
    contracts = torch.isin(tags, torch.tensor([t for r in REGIONS for t in r[0]], device=device))
    amg = SmoothedAggregationAMG(mesh.points, refresh_interval=3)
    coupling = ActiveStressCoupling(simulation._build(), mesh, fibres, tension,
                                    stiffness=stiffness, reference_stretch=reference,
                                    active=contracts, load_steps=(1,), newton_atol=1e-5,
                                    newton_rtol=1e-4, max_newton=15, max_krylov=1000,
                                    preconditioner=amg, max_attempts=1)

    # Loading, then the pericardium at rest at the loaded shape.
    def load_at(fraction):
        for c in CHAMBERS:
            load[c] = fraction * START[c] * MMHG
    coupling.inflate(load_at, steps=10, log=print)
    prestress.copy_(normal_spring_pressure(pericardium, mesh.points, coupling.displacement))
    coupling.problem.anchor_supports(coupling.displacement, names=[pericardium.name])

    dynamics = FirstOrderDynamicSolver(coupling.problem, **DYNAMICS,
                                       displacement=coupling.displacement, preconditioner=amg,
                                       newton_atol=1e-6, newton_rtol=1e-5, max_newton=15,
                                       forcing_initial=1e-3, max_krylov=1000)
    amg.force_refresh()
    surfaces = {c: mesh.face_set(CAVITY[c]).connectivity for c in CHAMBERS}
    cavity = CavityMechanics(coupling, load, surfaces, dynamic=dynamics, pressure_scale=MMHG,
                             volume_unit=1e3)
    return mesh, tags, fibres, sheets, coupling, cavity, amg


def cell_model(model, cell_type, device, ids=None):
    """A cell model with this heart's parameters."""
    cell = (model(EP_DT, region_ids=ids, cell_type=cell_type, device=device) if cell_type
            else model(EP_DT, region_ids=ids, device=device))
    for name, value in (VENTRICULAR_CELL if model is ToRORdDynCl else ATRIAL_CELL).items():
        setattr(cell, name, value)
    return cell


def prepace(out):
    """Pace each cell model alone to steady state, with Land beside it at fixed length
    over the last beats, and save it as <model>.pt in ``out``; a saved one is kept.

    Regions with the same cell model share its paced state: here LV and RV,
    whose Land parameters differ only in T_ref, which scales tension but no state.
    """
    pacer = PrePace(CYCLE, beats=PREPACE_BEATS, contraction_beats=LAND_BEATS, **PACING)
    cpu = torch.device("cpu")  # one cell paces fastest on the CPU
    for _, model, cell_type, land_parameters in REGIONS:
        path = out / f"{model.__name__}.pt"
        if path.exists():
            continue
        cell = cell_model(model, cell_type, cpu)
        land = LandModel(1, device=cpu, parameters=LandParameters(**land_parameters),
                         integrator="rush_larsen")
        pacer(cell, contraction=land)
        pacer.save_state(path, cell, contraction=land)
        beat, peak, diastolic = pacer.history[-1]
        print(f"paced {model.__name__} for {beat} beats: calcium {diastolic:.3f}-{peak:.3f} uM",
              flush=True)


def electrophysiology(mesh, tags, fibres, sheets, duration, out, device):
    """Reaction-eikonal activation with each region's cell model, starting from its
    paced state; also each region's paced Land state and the nodes with cells."""
    (out / "mesh").mkdir(exist_ok=True)  # the eikonal solver reads its mesh from disk
    write_carp(out / "mesh" / "heart", mesh.points, mesh.cells, tags, fibres, sheets)
    cells, land_states = [], []
    for ids, model, cell_type, _ in REGIONS:
        cell = cell_model(model, cell_type, device, ids)
        land_states.append(PrePace.load_state(CACHE / f"{model.__name__}.pt", cell))
        cells.append(cell)
    ep = ReactionEikonal(cells, T=duration, dt=EP_DT, device=device, dtype=mesh.dtype)
    ep.load_mesh(out / "mesh")
    owner = nodal_regions(mesh.cells, tags, [r[0] for r in REGIONS] + [PASSIVE], mesh.n_points)
    for i, cell in enumerate(cells):
        cell.node_indices = torch.nonzero(owner == i).flatten()
    for ids, along, across, normal in CONDUCTION.values():
        ep.add_velocity(ids, vel_l=along, vel_t=across, vel_n=normal)
    ep.add_velocity(PASSIVE, vel_l=1e-5, vel_t=1e-5, vel_n=1e-5)  # does not conduct
    for site, first in STIMULI.items():
        ep.add_stimulus(MESH / f"{site}.vtx", start=first, period=CYCLE,
                        count=int(np.ceil(duration / CYCLE)), **STIMULUS)
    has_cells = torch.zeros(ep.n_nodes, dtype=torch.bool, device=device)
    for cell in cells:
        has_cells[cell.node_indices] = True
    return ep, cells, land_states, has_cells


def contraction(cells, land_states, n_nodes, device, dtype):
    """One Land model over every node, each region with its own parameters and its
    paced state; nodes without cells rest fully blocked, at zero calcium."""
    defaults = LandParameters()
    names = {name for *_, land in REGIONS for name in land}
    values = {name: torch.full((n_nodes,), float(getattr(defaults, name)), device=device,
                               dtype=dtype) for name in names}
    for (*_, land), cell in zip(REGIONS, cells):
        for name, value in land.items():
            values[name][cell.node_indices] = value
    model = LandModel(n_nodes, device=device, dtype=dtype, parameters=LandParameters(**values),
                      integrator="rush_larsen", velocity=True)
    model.steady_state(torch.zeros(n_nodes, device=device, dtype=dtype))
    for cell, land_state in zip(cells, land_states):
        model.state[:, cell.node_indices] = land_state.to(device, dtype)[:, None]
    return model


# ===================================================================== output
class Recorder:
    """A pv.csv row per mechanics step, a printed line every 10 ms, PV loops per beat,
    and the surface every FRAME ms (VTK frames and the studio's bundle)."""

    def __init__(self, out, mesh, coupling, pressures, circulation, driver, t0):
        self.out, self.mesh, self.coupling, self.t0 = out, mesh, coupling, t0
        self.pressures, self.circulation, self.driver = pressures, circulation, driver
        self.rows, self.next = [], {"print": 0.0, "beat": CYCLE, "frame": 0.0}
        faces = boundary_faces(mesh)
        self.nodes, self.triangles = torch.unique(faces, return_inverse=True)
        self.frames = VtkSeries(out / "vtk", "heart")
        self.surface = studio.FrameRecorder(studio.surface_of(mesh.points, triangles=faces),
                                            interval=FRAME)

    def after_mechanics(self, t, iterations, converged):
        p, v = self.pressures.p, self.pressures.volume
        last = self.pressures.history[-1] if self.pressures.history else {}
        vessels = self.circulation.pressures(p)
        flows = last.get("flows", {})
        self.rows.append({"t": t, **{f"p_{c}": x for c, x in zip(CHAMBERS, p)},
                          **{f"V_{c}": x for c, x in zip(CHAMBERS, v)},
                          **{f"p_{k}": vessels[k] for k in ("sat", "pat", "svn", "pvn")},
                          **{f"q_{k}": flows.get(k, 0.0) for k in ("ao", "mi", "po", "ti")},
                          "iterations": iterations, "converged": converged,
                          "blood_ml": self.circulation.blood_volume(),
                          "wall_s": time.time() - self.t0})
        if t >= self.next["print"] - 1e-9:
            print(f"t {t:7.1f} ms | p (mmHg) " + " ".join(f"{c.upper()} {x:5.1f}"
                                                         for c, x in zip(CHAMBERS, p))
                  + " | V (mL) " + " ".join(f"{c.upper()} {x:5.1f}" for c, x in zip(CHAMBERS, v))
                  + f" | aorta {vessels['sat']:5.1f} | {iterations} it | "
                  f"{time.time() - self.t0:6.0f} s", flush=True)
            self.save()
            self.next["print"] += 10.0
        if t >= self.next["beat"] - 1e-9:
            self.plot_beat(int(round(self.next["beat"] / CYCLE)))
            self.next["beat"] += CYCLE

    def after_step(self, t, vm, calcium):
        if FRAME and t >= self.next["frame"] - 1e-9:
            u = self.coupling.displacement.reshape(-1, 3)
            self.frames.write(round(t, 6), 1000.0 * (self.mesh.points + u)[self.nodes],
                              self.triangles,
                              point_data=dict(vm=vm[self.nodes],
                                              tension=self.driver.tension[self.nodes],
                                              displacement=1000.0 * u[self.nodes]))
            self.next["frame"] += FRAME
        if self.surface.due(t):
            self.surface.add(t, displacement=self.coupling.displacement.reshape(-1, 3), vm=vm,
                             tension=self.driver.tension)

    def save(self):
        pd.DataFrame(self.rows).to_csv(self.out / "pv.csv", index=False)

    def plot_beat(self, beat):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = pd.DataFrame(self.rows)
        rows = rows[(rows.t >= (beat - 1) * CYCLE) & (rows.t <= beat * CYCLE)]
        print(f"\\nbeat {beat}: EDV / ESV (mL), peak pressure (mmHg)")
        fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
        for ax, c in zip(axes.flat, CHAMBERS):
            v, p = rows[f"V_{c}"], rows[f"p_{c}"]
            print(f"  {c.upper()}  {v.max():6.1f} / {v.min():6.1f}   {p.max():6.1f}")
            ax.plot(v, p, color="#1F5FA8", lw=1.8)
            ax.set(title=c.upper(), xlabel="volume (mL)", ylabel="pressure (mmHg)")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Pressure-volume loops, beat {beat}")
        fig.savefig(self.out / f"pv_loops_beat{beat}.png", dpi=150)
        plt.close(fig)

    def write_bundle(self, title):
        """The studio's visualisation bundle: surface frames and the PV traces."""
        if self.surface.times:
            studio.export_em(self.surface, self.rows, title=title, cycle_length=CYCLE)


# ===================================================================== the run
studio.stage("setup", "Setting up")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA starts
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
torch.set_float32_matmul_precision("highest")
t0 = time.time()

# 1-2. The unloaded heart, inflated to the start pressures.
studio.stage("unload", "Unloading the geometry", auto=False)
points = unload(DEVICE, CACHE)
studio.stage("load", "Inflating to the start pressures")
mesh, tags, fibres, sheets, coupling, cavity, amg = build_heart(DEVICE, points)
start = cavity.volumes()
print(f"{mesh.n_points} nodes, {mesh.n_cells} tetrahedra; loaded in {time.time() - t0:.0f} s "
      "to " + ", ".join(f"{c.upper()} {v:.1f}" for c, v in zip(CHAMBERS, start)) + " mL",
      flush=True)

# 3. The circulation, coupled through the four cavity volumes.
circulation = CircAdapt(CIRCULATION, volumes=start)
pressures = CoupledCavityPressures(circulation, np.array([START[c] for c in CHAMBERS]),
                                   start, np.eye(4), tol=VOLUME_TOLERANCE)

# 4. Activation, cells, contraction and the beats.
studio.stage("prepace", "Pre-pacing the cell models")
prepace(CACHE)
studio.stage("beats", "Beating")
ep, cells, land_states, has_cells = electrophysiology(mesh, tags, fibres, sheets, DURATION,
                                                      RESULTS, DEVICE)
land = contraction(cells, land_states, ep.n_nodes, DEVICE, mesh.dtype)
driver = ActiveTensionDriver(land, stretch=HeldStretch(ep.n_nodes, device=DEVICE,
                                                       dtype=mesh.dtype))
recorder = Recorder(RESULTS, mesh, coupling, pressures, circulation, driver, t0)


def next_solve(t):  # every MECHANICS_DT, never across a beat boundary or the end
    next_beat = (np.floor((t + 1e-9) / CYCLE) + 1) * CYCLE
    return min(t + MECHANICS_DT, next_beat, DURATION)


runner = ElectromechanicsRunner(
    driver, cavity, pressures, next_solve, circulation_time=1e-3,  # ms -> s, same clock
    active_nodes=has_cells, preconditioner=amg, after_mechanics=recorder.after_mechanics,
    after_step=recorder.after_step, log=lambda message: print(message, flush=True))
runner.start(torch.ones(ep.n_nodes, device=DEVICE, dtype=mesh.dtype), MECHANICS_DT)
recorder.after_mechanics(0.0, 0, True)
studio.total(DURATION, "ms")  # the "t ... ms | p (mmHg)" lines become progress
try:
    ep.solve(on_step=runner, record=False, verbose=False)
finally:
    recorder.save()  # pv.csv, and the bundle, also when a run stops early
    recorder.write_bundle($TITLE)

studio.stage("export", "Saving results")
print(f"done in {(time.time() - t0) / 3600:.1f} h; results in {RESULTS}", flush=True)
studio.stage("done", "Completed")
''')

_IMPORTS = (
    "import time",
    "import numpy as np",
    "import pandas as pd",
    "from torchcor.circulation import CircAdapt, CircAdaptParameters",
    "from torchcor.electromechanics.coupling import ActiveStressCoupling, ActiveTensionDriver, "
    "HeldStretch",
    "from torchcor.electromechanics.heart import CavityMechanics, CoupledCavityPressures, "
    "ElectromechanicsRunner",
    "from torchcor.electromechanics.land import LandModel, LandParameters",
    "from torchcor.electromechanics.material import StabilizedActiveStressMaterial",
    "from torchcor.electromechanics.regions import nodal_regions",
    "from torchcor.electrophysiology import PrePace, ReactionEikonal",
    "from torchcor.ionic import CourtemancheRamirezNattel, ToRORdDynCl",
    "from torchcor.mechanics import Mechanics",
    "from torchcor.mechanics.boundary import FollowerPressure, RobinBC, normal_spring_pressure",
    "from torchcor.mechanics.dynamics import FirstOrderDynamicSolver",
    "from torchcor.mechanics.material import CompositeMaterial, GuccioneMaterial, "
    "IsochoricMaterial, MaterialAxes, NeoHookeanMaterial, VolumetricPenalty",
    "from torchcor.mechanics.mesh import add_surface, boundary_faces, read_carp, "
    "read_element_data, read_surface, write_carp",
    "from torchcor.mechanics.preconditioner import SmoothedAggregationAMG",
    "from torchcor.mechanics.unloading import Unloading",
    "from torchcor.mechanics.visualisation import VtkSeries",
)


def _float(value: float) -> str:
    return codegen.lit(float(value))


def _tuple(values) -> str:
    return codegen.lit(tuple(float(v) for v in values))


def _table(rows: list[tuple[str, str, str]], indent: int) -> str:
    """``{key: value,  # comment`` rows of a dict literal, aligned under its brace."""
    pad = " " * (indent + 1)
    lines = []
    for i, (key, value, comment) in enumerate(rows):
        end = "}" if i == len(rows) - 1 else ","
        head = ("{" if i == 0 else pad) + f"{key}: {value}{end}"
        lines.append(head + (f"  # {comment}" if comment else ""))
    return "\n".join(lines) if rows else "{}"


def _settings(params: EMParams) -> dict[str, str]:
    lit = codegen.lit
    rhythm, cells, land, passive, support, circ, solver = (
        params.rhythm, params.cells, params.contraction, params.passive, params.support,
        params.circulation, params.solver)
    indent_of = len  # dict literal rows align under the opening brace
    values = {
        "CAVITY": lit(params.heart.cavities()),
        "PERICARDIUM_SCALE": lit(params.heart.pericardium_scale),
        "CYCLE": _float(rhythm.cycle_length),
        "DURATION": _float(rhythm.total()),
        "STIMULI": lit({p.site: float(p.first) for p in rhythm.pacing}),
        "STIMULUS_INTENSITY": _float(rhythm.stimulus_intensity),
        "STIMULUS_DURATION": _float(rhythm.stimulus_duration),
        "EP_DT": _float(rhythm.ep_dt),
        "MECHANICS_DT": _float(rhythm.mechanics_dt),
        "CONDUCTION": _table([(lit(r.name or f"tags {r.tags}"),
                               f"({lit([int(t) for t in r.tags])}, {_float(r.vel_l)}, "
                               f"{_float(r.vel_t)}, {_float(r.vel_n)})", "")
                              for r in params.conduction.regions], indent_of("CONDUCTION = ")),
        "PASSIVE": lit([int(t) for t in params.conduction.passive_tags]),
        "VENTRICULAR_CELL": _assignment_dict(cells.ventricular, "VENTRICULAR_CELL = "),
        "ATRIAL_CELL": _assignment_dict(cells.atrial, "ATRIAL_CELL = "),
        "PREPACE_BEATS": lit(int(cells.prepace_beats)),
        "LAND_BEATS": lit(int(cells.land_beats)),
        "PACING_STIMULUS": _float(cells.pacing_stimulus),
        "PACING_DURATION": _float(cells.pacing_duration),
        "LAND_VENTRICLES": _assignment_dict(land.ventricles, "LAND_VENTRICLES = "),
        "LAND_ATRIA": _assignment_dict(land.atria, "LAND_ATRIA = "),
        "LV_TAGS": lit([int(t) for t in land.lv_tags]),
        "RV_TAGS": lit([int(t) for t in land.rv_tags]),
        "ATRIA_TAGS": lit([int(t) for t in land.atria_tags]),
        "CELL_TYPE": lit(_VENTRICLE_TYPES[cells.ventricular_cell_type]),
        "LV_TREF": _float(land.lv_tref),
        "RV_TREF": _float(land.rv_tref),
        "GUCCIONE": _table([(lit(tuple(int(t) for t in r.tags)),
                             f"dict(a={_float(r.a)}, b_f={_float(r.b_f)}, b_t={_float(r.b_t)}, "
                             f"b_fs={_float(r.b_fs)})", r.tissue)
                            for r in passive.guccione], indent_of("GUCCIONE = ")),
        "NEO_HOOKEAN": _table([(lit(tuple(int(t) for t in r.tags)), _float(r.mu), r.tissue)
                               for r in passive.neo_hookean], indent_of("NEO_HOOKEAN = ")),
        "KAPPA": _float(passive.kappa),
        "SPRINGS": _table([(lit(s.surface),
                            f"dict(stiffness={_float(s.stiffness)}, "
                            f"normal_only={lit(bool(s.normal_only))}, scaled={lit(bool(s.scaled))})",
                            "") for s in support.springs], indent_of("SPRINGS = ")),
        "DENSITY": _float(support.density),
        "RHO_INFINITY": _float(support.rho_infinity),
        "MASS_DAMPING": _float(support.mass_damping),
        "STIFFNESS_DAMPING": _float(support.stiffness_damping),
        "START": lit({c: float(v) for c, v in support.start().items()}),
        "BLOOD_DENSITY": _float(circ.blood_density),
        "TUBE_LENGTH": _tuple(t.length for t in circ.tubes),
        "TUBE_WALL_AREA": _tuple(t.wall_area for t in circ.tubes),
        "TUBE_REFERENCE_AREA": _tuple(t.reference_area for t in circ.tubes),
        "TUBE_REFERENCE_PRESSURE": _tuple(t.reference_pressure for t in circ.tubes),
        "TUBE_STIFFNESS_EXPONENT": _tuple(t.stiffness_exponent for t in circ.tubes),
        "VALVE_OPEN_AREA": _tuple(v.open_area for v in circ.valves),
        "VALVE_LEAK_AREA": _tuple(v.leak_area for v in circ.valves),
        "REFERENCE_FLOW": _float(circ.reference_flow),
        "PERIPHERAL_DROP": _tuple(p.reference_drop for p in circ.peripheral),
        "PERIPHERAL_RESISTANCE": _tuple(p.resistance_factor for p in circ.peripheral),
        "CHAMBER_WALL_VOLUME": _tuple(c.wall_volume for c in circ.chambers),
        "CHAMBER_INITIAL_VOLUME": _tuple(c.initial_volume for c in circ.chambers),
        "VOLUME_TOLERANCE": _float(solver.volume_tolerance),
        "UNLOADING_TOLERANCE": _float(solver.unloading_tolerance),
        "FRAME": _float(solver.frame_interval),
        "CHUNK": lit(int(solver.chunk)),
    }
    return values


def _assignment_dict(text: str, prefix: str) -> str:
    """``"a=1, b=2"`` -> ``dict(a=1.0, b=2.0)``, wrapped as if it followed ``prefix``."""
    pairs = {name: codegen.lit(value) for name, value in common.parse_scaling(text)}
    return codegen.call("dict", prefix=prefix, **pairs)[len(prefix):]


def build_script(params: EMParams, ctx: ScriptContext) -> str:
    rhythm = params.rhythm
    title = ctx.run_name or "Electromechanics run"
    script = codegen.Script(codegen.docstring(
        title, "Electromechanics · four-chamber heart coupled to the CircAdapt circulation "
               f"({params.variant_label()}).", ctx,
        ["A port of TorchCor's demo/em/main.py (HCM3 cohort, combination 1267) with this "
         "run's parameters. Stages: unloading to the stress-free geometry, inflation to the "
         "start pressures, pre-pacing each cell model alone, then the beats: reaction-"
         "eikonal activation, ToR-ORd-dynCl ventricles and Courtemanche atria, Land "
         f"contraction and mechanics every {rhythm.mechanics_dt:g} ms."],
        outputs="results/: pv.csv (pressures, volumes and flows at every mechanics step), "
                "pv_loops_beat<N>.png, vtk/heart_*.vtk (the surface every frame), "
                "unloaded.pt and <cell model>.pt (reused when they exist); viz/: the studio's "
                "visualisation bundle."))
    for line in _IMPORTS:
        script.imports.add(line, "stdlib" if line == "import time" else "third")
    script.imports.add("import os", "stdlib")
    s = script.settings
    codegen.device_setting(script, ctx)
    codegen.output_settings(script)
    reuse = params.heart.reuse_dir.strip()
    s.add("CACHE", codegen.path(reuse, ctx.windows) if reuse else "RESULTS",
          "unloaded.pt and the paced cell states")
    s.add("MESH_DIR", codegen.path(ctx.input_dir or ".", ctx.windows), "the heart's mesh folder")
    values = _settings(params)
    values["MESH"] = "MESH_DIR"
    values["TITLE"] = codegen.lit(title)
    script.code(_SCRIPT.substitute(values))
    return script.render()


# =============================================================================== spec
SPEC = register(SimulationSpec(
    kind=SimulationKind.EM,
    title="Electromechanics",
    tagline="Beating four-chamber heart with a circulation",
    description="Couple activation, cell models, Land contraction and finite-strain mechanics "
                "of a four-chamber heart to the CircAdapt circulation, and watch it beat with "
                "its pressure-volume loops.",
    icon="heart-pulse",
    params_model=EMParams,
    build_script=build_script,
    stages=STAGES,
    demos=DEMOS,
    panels=PANELS,
    check=check,
    input_use=input_use,
    badge="EM",
))
