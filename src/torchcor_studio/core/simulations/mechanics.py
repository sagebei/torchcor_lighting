"""Cardiac mechanics: finite-strain hyperelasticity on generated benchmark geometries.

The geometries are built by ``torchcor.mechanics`` itself (a box or a truncated-ellipsoid
ventricle), so nothing needs to be uploaded. Quasi-static runs ramp the loads from zero
to full (the Land et al. 2015 benchmarks); the dynamic run drives one heartbeat with
time-varying cavity pressure and active stress (after Aróstica et al. 2025, benchmark 1).
"""

from __future__ import annotations

import math

from torchcor_studio.core.models import InputInfo
from torchcor_studio.core.simulations import codegen
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
from torchcor_studio.core.simulations.panels import OUTPUT, Panel

SHAPES = {"box": "Beam (box)", "ellipsoid": "Idealised left ventricle"}
LAWS = {"guccione": "Guccione (transversely isotropic)",
        "holzapfel_ogden": "Holzapfel–Ogden (orthotropic)",
        "neo_hookean": "Neo-Hookean (isotropic)"}
FIBRES = {"x": "Along the x axis", "helix": "Helical, rule-based"}
VOLUME = {"constraint": "Incompressible (augmented Lagrangian)",
          "penalty": "Nearly incompressible (volumetric penalty)"}
MODES = {"static": "Quasi-static load ramp", "dynamic": "Dynamic heartbeat"}
SUPPORTS = {"clamp": "Clamp a surface", "springs": "Elastic springs (epicardium and base)"}
SURFACES = {"x-": "Face x = 0", "x+": "Face x = length", "y-": "Face y = 0",
            "y+": "Face y = width", "z-": "Face z = 0", "z+": "Face z = height",
            "base": "Base", "endo": "Endocardium", "epi": "Epicardium"}
SHAPE_SURFACES = {"box": ("x-", "x+", "y-", "y+", "z-", "z+"), "ellipsoid": ("base", "endo", "epi")}

STAGES = (
    Stage("setup", "Setting up"),
    Stage("mesh", "Building the model"),
    Stage("solve", "Solving"),
    Stage("export", "Saving results"),
    Stage("done", "Completed"),
)


# =============================================================================== parameters
class MechGeometry(Section):
    """The generated mesh (quadratic hexahedra by default)."""

    shape: str = param("box", "Geometry", choices=SHAPES, widget="segmented",
                       help="A box (Land problem 1) or a truncated ellipsoid (problems 2 and 3).")
    length: float = param(10.0, "Length (x)", unit="mm", min=0.1, max=1000.0, step=1.0,
                          visible_if={"shape": "box"}, group="Size", help="Box size along x.")
    width: float = param(1.0, "Width (y)", unit="mm", min=0.1, max=1000.0, step=0.5,
                         visible_if={"shape": "box"}, group="Size", help="Box size along y.")
    height: float = param(1.0, "Height (z)", unit="mm", min=0.1, max=1000.0, step=0.5,
                          visible_if={"shape": "box"}, group="Size", help="Box size along z.")
    nx: int = param(20, "Elements along x", min=1, max=500, visible_if={"shape": "box"},
                    group="Elements", short="Along x", help="Number of hexahedra along the length.")
    ny: int = param(2, "Elements along y", min=1, max=500, visible_if={"shape": "box"},
                    group="Elements", short="Along y", help="Number of hexahedra across the width.")
    nz: int = param(2, "Elements along z", min=1, max=500, visible_if={"shape": "box"},
                    group="Elements", short="Along z",
                    help="Number of hexahedra through the height.")
    endo_radius: float = param(7.0, "Endocardial short radius", unit="mm", min=0.5, max=500.0,
                               visible_if={"shape": "ellipsoid"}, group="Endocardium",
                               short="Short radius",
                               help="Radius of the inner surface around the long axis.")
    endo_length: float = param(17.0, "Endocardial long radius", unit="mm", min=0.5, max=500.0,
                               visible_if={"shape": "ellipsoid"}, group="Endocardium",
                               short="Long radius",
                               help="Half-length of the inner ellipsoid along the long axis.")
    epi_radius: float = param(10.0, "Epicardial short radius", unit="mm", min=0.5, max=500.0,
                              visible_if={"shape": "ellipsoid"}, group="Epicardium",
                              short="Short radius",
                              help="Radius of the outer surface around the long axis.")
    epi_length: float = param(20.0, "Epicardial long radius", unit="mm", min=0.5, max=500.0,
                              visible_if={"shape": "ellipsoid"}, group="Epicardium",
                              short="Long radius",
                              help="Half-length of the outer ellipsoid along the long axis.")
    base_z: float = param(5.0, "Base plane", unit="mm", min=-500.0, max=500.0,
                          visible_if={"shape": "ellipsoid"}, group="Base", short="Base plane",
                          help="Height z of the plane the ellipsoids are cut at.")
    transmural: int = param(1, "Elements through the wall", min=1, max=50,
                            visible_if={"shape": "ellipsoid"}, group="Elements",
                            short="Through the wall", help="Hexahedra from endo to epi.")
    longitudinal: int = param(8, "Elements apex to base", min=1, max=200,
                              visible_if={"shape": "ellipsoid"}, group="Elements",
                              short="Apex to base", help="Hexahedra along the wall.")
    circumferential: int = param(16, "Elements around", min=3, max=400,
                                 visible_if={"shape": "ellipsoid"}, group="Elements",
                                 short="Around", help="Hexahedra around the long axis.")
    order: int = param(2, "Element order", min=1, max=3, advanced=True,
                       help="Polynomial degree of the hexahedra; 2 (quadratic) as the benchmarks.")

    def n_points(self) -> int:
        p = self.order
        if self.shape == "box":
            return (p * self.nx + 1) * (p * self.ny + 1) * (p * self.nz + 1)
        n1, n2, n3 = p * self.transmural + 1, p * self.longitudinal + 1, p * self.circumferential
        return n1 * (n2 - 1) * n3 + n1


class MechMaterial(Section):
    """Constitutive law of the passive tissue and the fibre field (stresses in kPa)."""

    law: str = param("guccione", "Constitutive law", choices=LAWS, widget="combo",
                     help="Strain-energy function of the passive myocardium.")
    C: float = param(2.0, "Stiffness C", unit="kPa", min=0.0, step=0.5,
                     visible_if={"law": "guccione"}, help="Guccione scaling constant.")
    bf: float = param(8.0, "Fibre exponent b_f", min=0.0, step=0.5,
                      visible_if={"law": "guccione"}, help="Guccione fibre stiffening.")
    bt: float = param(2.0, "Transverse exponent b_t", min=0.0, step=0.5,
                      visible_if={"law": "guccione"}, help="Guccione cross-fibre stiffening.")
    bfs: float = param(4.0, "Shear exponent b_fs", min=0.0, step=0.5,
                       visible_if={"law": "guccione"}, help="Guccione fibre-sheet shear stiffening.")
    mu: float = param(10.0, "Shear modulus μ", unit="kPa", min=0.0, step=1.0,
                      visible_if={"law": "neo_hookean"}, help="Neo-Hookean shear modulus.")
    a: float = param(0.059, "Matrix modulus a", unit="kPa", min=0.0, step=0.01, decimals=4,
                     visible_if={"law": "holzapfel_ogden"}, group="Matrix", short="Modulus a",
                     help="Isotropic matrix stiffness.")
    b: float = param(8.023, "Matrix exponent b", min=0.0, step=0.5,
                     visible_if={"law": "holzapfel_ogden"}, group="Matrix", short="Exponent b",
                     help="Isotropic matrix stiffening.")
    a_f: float = param(18.472, "Fibre modulus a_f", unit="kPa", min=0.0, step=0.5,
                       visible_if={"law": "holzapfel_ogden"}, group="Fibre", short="Modulus a_f",
                       help="Fibre stiffness.")
    b_f: float = param(16.026, "Fibre exponent b_f", min=0.0, step=0.5,
                       visible_if={"law": "holzapfel_ogden"}, group="Fibre", short="Exponent b_f",
                       help="Fibre stiffening.")
    a_s: float = param(2.481, "Sheet modulus a_s", unit="kPa", min=0.0, step=0.5,
                       visible_if={"law": "holzapfel_ogden"}, group="Sheet", short="Modulus a_s",
                       help="Sheet stiffness.")
    b_s: float = param(11.12, "Sheet exponent b_s", min=0.0, step=0.5,
                       visible_if={"law": "holzapfel_ogden"}, group="Sheet", short="Exponent b_s",
                       help="Sheet stiffening.")
    a_fs: float = param(0.216, "Shear modulus a_fs", unit="kPa", min=0.0, step=0.05,
                        visible_if={"law": "holzapfel_ogden"}, group="Shear", short="Modulus a_fs",
                        help="Fibre-sheet shear stiffness.")
    b_fs: float = param(11.436, "Shear exponent b_fs", min=0.0, step=0.5,
                        visible_if={"law": "holzapfel_ogden"}, group="Shear",
                        short="Exponent b_fs", help="Fibre-sheet shear stiffening.")
    fibres: str = param("x", "Fibres", choices=FIBRES, widget="segmented", group="Fibres",
                        short="Direction",
                        help="Fibre direction: along x (beam) or a helix whose angle varies "
                             "linearly through the ventricular wall.")
    helix_endo: float = param(90.0, "Helix angle at the endocardium", unit="°", min=-90.0,
                              max=90.0, step=5.0, visible_if={"fibres": "helix"}, group="Fibres",
                              short="Helix, endo",
                              help="Angle from the circumferential direction on the inner wall.")
    helix_epi: float = param(-90.0, "Helix angle at the epicardium", unit="°", min=-90.0,
                             max=90.0, step=5.0, visible_if={"fibres": "helix"}, group="Fibres",
                             short="Helix, epi",
                             help="Angle from the circumferential direction on the outer wall.")
    volume: str = param("constraint", "Volume change", choices=VOLUME, group="Volume",
                        help="Enforce J = 1 by an augmented Lagrangian, or penalise volume "
                             "change in the strain energy.")
    bulk_modulus: float = param(100.0, "Bulk modulus", unit="kPa", min=0.1, max=1e7, log=True,
                                group="Volume", help="Incompressibility penalty; larger is stiffer in volume but "
                                     "harder for the iterative solver.")


class MechLoading(Section):
    """Supports, loads and how they evolve."""

    mode: str = param("static", "Time course", choices=MODES, widget="segmented",
                      help="Ramp the loads to full (quasi-static) or march one heartbeat in "
                           "time with inertia (dynamic).")
    support: str = param("clamp", "Support", choices=SUPPORTS, group="Support",
                         help="Hold one surface fixed, or rest the ventricle on springs.")
    fixed: str = param("x-", "Clamped surface", choices=SURFACES, visible_if={"support": "clamp"},
                       group="Support", help="Surface held fixed in all directions.")
    epi_stiffness: float = param(100.0, "Epicardial spring stiffness", unit="kPa/mm", min=0.0,
                                 visible_if={"support": "springs"}, group="Support",
                                 short="Epicardial spring",
                                 help="Normal support of the epicardium (pericardium).")
    base_stiffness: float = param(0.1, "Base spring stiffness", unit="kPa/mm", min=0.0,
                                  visible_if={"support": "springs"}, group="Support",
                                  short="Base spring", help="Support of the base in all directions.")
    spring_damping: float = param(5.0, "Spring damping", unit="kPa·ms/mm", min=0.0,
                                  visible_if={"support": "springs"}, group="Support",
                                  help="Viscous damping of both springs.")
    pressure_surface: str = param("z-", "Pressure surface", choices=SURFACES, group="Load",
                                  help="Surface the follower pressure acts on.")
    pressure: float = param(0.004, "Pressure", unit="kPa", min=0.0, step=0.5, decimals=4,
                            visible_if={"mode": "static"}, group="Load",
                            help="Follower pressure at full load (1 kPa ≈ 7.5 mmHg).")
    active_tension: float = param(0.0, "Active tension", unit="kPa", min=0.0, step=5.0,
                                  visible_if={"mode": "static"}, group="Load",
                                  help="Fibre stress of contraction at full load; 0 = passive.")
    peak_pressure: float = param(16.07, "Peak pressure", unit="kPa", min=0.0, step=1.0,
                                 visible_if={"mode": "dynamic"}, group="Load",
                                 help="Highest cavity pressure of the beat (16 kPa ≈ 120 mmHg).")
    peak_tension: float = param(118.8, "Peak active stress", unit="kPa", min=0.0, step=5.0,
                                visible_if={"mode": "dynamic"}, group="Load",
                                help="Highest fibre stress of the beat; 0 = passive filling only.")
    contraction_onset: float = param(160.0, "Contraction onset", unit="ms", min=0.0, step=10.0,
                                     visible_if={"mode": "dynamic"}, group="Beat",
                                     help="When the active stress starts to rise.")
    pressure_onset: float = param(170.0, "Pressure rise onset", unit="ms", min=0.0, step=10.0,
                                  visible_if={"mode": "dynamic"}, group="Beat",
                                  help="When the cavity pressure starts to rise.")
    relaxation_onset: float = param(484.0, "Relaxation onset", unit="ms", min=0.0, step=10.0,
                                    visible_if={"mode": "dynamic"}, group="Beat",
                                    help="When active stress and pressure start to fall.")
    transition: float = param(5.0, "Transition width", unit="ms", min=0.1, max=100.0,
                              visible_if={"mode": "dynamic"}, advanced=True, group="Beat",
                              help="Width of the smoothed switches between phases.")
    density: float = param(1.0, "Density", unit="kPa·ms²/mm²", min=0.0, step=0.01,
                           visible_if={"mode": "dynamic"}, group="Tissue dynamics",
                           help="Tissue density; 1.0 = 1000 kg/m³ in mm, kPa and ms.")
    viscosity: float = param(100.0, "Viscosity", unit="kPa·ms", min=0.0, step=10.0,
                             visible_if={"mode": "dynamic"}, group="Tissue dynamics",
                             help="Kelvin–Voigt viscosity of the tissue.")
    rho_infinity: float = param(2.0 / 3.0, "High-frequency damping ρ∞", min=0.0, max=1.0,
                                step=0.05, visible_if={"mode": "dynamic"}, advanced=True,
                                group="Tissue dynamics", short="Damping ρ∞",
                                help="Generalized-α spectral radius: 1 undamped, 0 maximal.")
    load_steps: int = param(10, "Load steps", min=1, max=1000, visible_if={"mode": "static"},
                            help="The first increment is 1/n of the load; the solver adapts it.")
    duration: float = param(1000.0, "Duration", unit="ms", min=1.0, max=1e5, step=50.0,
                            visible_if={"mode": "dynamic"}, help="Simulated time (one beat).")
    dt: float = param(2.5, "Time step", unit="ms", min=0.01, max=100.0, step=0.5,
                      visible_if={"mode": "dynamic"}, help="Mechanics time step.")


class MechSolver(Section):
    """Newton–Krylov settings."""

    newton_rtol: float = param(1e-9, "Relative tolerance", min=1e-14, max=1e-2, log=True,
                               advanced=True, short="Newton rtol", summary=True, help="Residual relative to the force scale.")
    newton_atol: float = param(1e-10, "Absolute tolerance", unit="mN", min=1e-16, max=1.0,
                               log=True, advanced=True, short="atol", summary=True,
                               help="Residual floor in force units (kPa·mm² = mN).")
    max_newton: int = param(50, "Newton iteration limit", min=1, max=1000, advanced=True,
                            short="Newton iterations", summary=True, help="Corrections per step before the step is cut back.")


class MechOutputs(Section):
    """What is saved."""

    frame_interval: float = param(10.0, "Frame interval", unit="ms", min=0.01, step=5.0,
                                  visible_if={"../loading.mode": "dynamic"}, help="Time between saved surface frames of a dynamic run.")
    vtk: bool = param(True, "VTK file", help="results/displacement.vtu (warp by vector in "
                                             "ParaView) with the final displacement.")


class MechanicsParams(Section):
    """Mechanics parameters."""

    geometry: MechGeometry = param(title="Geometry", default_factory=MechGeometry)
    material: MechMaterial = param(title="Material", default_factory=MechMaterial)
    loading: MechLoading = param(title="Loading", default_factory=MechLoading)
    solver: MechSolver = param(title="Solver", default_factory=MechSolver)
    outputs: MechOutputs = param(title="Outputs", default_factory=MechOutputs)

    def variant_label(self) -> str:
        if self.geometry.shape == "box":
            return "Beam"
        if self.loading.mode == "dynamic":
            return "LV beat"
        return "LV contraction" if self.loading.active_tension > 0 else "LV inflation"


# =============================================================================== demos
def beam() -> MechanicsParams:
    """Land et al. (2015) problem 1: beam under 0.004 kPa follower pressure."""
    return MechanicsParams()


def _lv(**loading) -> MechanicsParams:
    return MechanicsParams(
        geometry=MechGeometry(shape="ellipsoid"),
        material=MechMaterial(law="guccione", fibres="helix", bulk_modulus=1000.0),
        loading=MechLoading(fixed="base", pressure_surface="endo", **loading))


def lv_inflation() -> MechanicsParams:
    """Land problem 2: isotropic ventricle inflated to 10 kPa."""
    params = _lv(pressure=10.0)
    params.material = MechMaterial(law="guccione", C=10.0, bf=1.0, bt=1.0, bfs=1.0,
                                   fibres="helix", bulk_modulus=1000.0)
    return params


def lv_contraction() -> MechanicsParams:
    """Land problem 3: 15 kPa and 60 kPa active stress along helical fibres."""
    return _lv(pressure=15.0, active_tension=60.0)


def lv_beat() -> MechanicsParams:
    """One heartbeat of a human-sized ventricle, after Aróstica et al. (2025), benchmark 1.

    Their geometry (long axis along z here, base cut by a plane at z = 25 mm), material,
    supports and load schedules in mm, kPa and ms. The soft base springs let the base descend
    towards the apex in systole: on this mesh the benchmark point p0 moves 34 mm along the
    axis at peak systole, against 36 mm in TorchCor's run of the benchmark's own mesh.
    """
    return MechanicsParams(
        geometry=MechGeometry(shape="ellipsoid", endo_radius=25.0, endo_length=90.0,
                              epi_radius=35.0, epi_length=97.0, base_z=25.0),
        material=MechMaterial(law="holzapfel_ogden", fibres="helix", helix_endo=60.0,
                              helix_epi=-60.0, volume="penalty", bulk_modulus=1000.0),
        loading=MechLoading(mode="dynamic", support="springs", fixed="base",
                            pressure_surface="endo"),
        solver=MechSolver(newton_atol=1e-6),
    )


DEMOS = (
    Demo("beam", "Beam (Land problem 1)",
         "A 10 × 1 × 1 mm beam bent by a 0.004 kPa follower pressure.", beam,
         cost=Cost.SECONDS, source="Land et al. (2015)"),
    Demo("lv-inflation", "Idealised LV, inflation (Land problem 2)",
         "An isotropic ventricle inflated to 10 kPa.", lv_inflation,
         cost=Cost.SECONDS, source="Land et al. (2015)"),
    Demo("lv-contraction", "Idealised LV, contraction (Land problem 3)",
         "60 kPa of active stress on helical fibres, 15 kPa of pressure.",
         lv_contraction, cost=Cost.SECONDS, source="Land et al. (2015)"),
    Demo("lv-beat", "Idealised LV, one beat (Aróstica benchmark 1)",
         "A dynamic beat on springs, with pressure and active stress.", lv_beat,
         source="Aróstica et al. (2025)"),
)

PANELS = (
    Panel("Geometry", "mesh", ("geometry", OUTPUT)),
    Panel("Material", "layers", ("material",)),
    Panel("Loading", "gauge", ("loading",)),
    Panel("Time and outputs", "clock", ("loading.load_steps", "loading.duration", "loading.dt",
                                        "outputs")),
    Panel("Solver", "cpu", ("solver",), collapsed=True),
)


def input_use(_params: MechanicsParams) -> tuple[InputUse, str]:
    return "unused", ""  # torchcor.mechanics builds its geometry


# =============================================================================== checks
def check(params: MechanicsParams, _info: InputInfo | None = None) -> list[Issue]:
    geo, mat, load = params.geometry, params.material, params.loading
    issues: list[Issue] = []
    surfaces = SHAPE_SURFACES[geo.shape]
    for field in ("fixed", "pressure_surface"):
        if field == "fixed" and load.support != "clamp":
            continue
        value = getattr(load, field)
        if value not in surfaces:
            issues.append(Issue(Severity.ERROR,
                                f"The {SHAPES[geo.shape].lower()} has no surface "
                                f"'{SURFACES.get(value, value)}'; choose one of "
                                f"{', '.join(SURFACES[s] for s in surfaces)}.", f"loading.{field}"))
    if load.support == "springs" and geo.shape != "ellipsoid":
        issues.append(Issue(Severity.ERROR, "Spring supports need the ventricle geometry "
                                            "(its epicardium and base).", "loading.support"))
    if load.support == "clamp" and load.fixed == load.pressure_surface:
        issues.append(Issue(Severity.ERROR, "The clamped surface cannot also carry the pressure.",
                            "loading.pressure_surface"))
    if geo.shape == "ellipsoid":
        if not (geo.epi_radius > geo.endo_radius and geo.epi_length > geo.endo_length):
            issues.append(Issue(Severity.ERROR, "The epicardium must enclose the endocardium.",
                                "geometry.epi_radius"))
        if not abs(geo.base_z) < min(geo.endo_length, geo.epi_length):
            issues.append(Issue(Severity.ERROR, "The base plane must cut both ellipsoids "
                                                "(|z| below both long radii).", "geometry.base_z"))
    if mat.fibres == "helix" and geo.shape != "ellipsoid":
        issues.append(Issue(Severity.ERROR, "Helical fibres are defined on the ventricle only.",
                            "material.fibres"))
    if load.mode == "static":
        if load.pressure <= 0 and load.active_tension <= 0:
            issues.append(Issue(Severity.WARNING, "Neither pressure nor active tension loads "
                                                  "the body.", "loading.pressure"))
    else:
        if load.density <= 0:
            issues.append(Issue(Severity.ERROR, "A dynamic run needs a positive density.",
                                "loading.density"))
        if load.dt >= load.duration:
            issues.append(Issue(Severity.ERROR, "The time step is longer than the run.",
                                "loading.dt"))
        if params.outputs.frame_interval < load.dt:
            issues.append(Issue(Severity.WARNING, "Frames are saved at most once per time step.",
                                "outputs.frame_interval"))
        if not (load.contraction_onset < load.relaxation_onset
                and load.pressure_onset < load.relaxation_onset):
            issues.append(Issue(Severity.ERROR, "Relaxation must start after contraction and the "
                                                "pressure rise.", "loading.relaxation_onset"))
        if load.relaxation_onset >= load.duration:
            issues.append(Issue(Severity.WARNING, "The beat ends before relaxation starts.",
                                "loading.duration"))
        if load.support == "clamp" and load.mode == "dynamic" and geo.shape == "ellipsoid":
            issues.append(Issue(Severity.INFO, "A clamped base makes the beat stiffer than the "
                                               "spring-supported benchmark.", "loading.support"))
    dofs = 3 * geo.n_points()
    if dofs > 3_000_000:
        issues.append(Issue(Severity.WARNING, f"{dofs:,} degrees of freedom: expect a long solve "
                                              "and high GPU memory use.", "geometry.shape"))
    return issues


# =============================================================================== script
_SCHEDULES = '''
def smooth_step(t, onset, rising):
    """Smoothed Heaviside switch of width TRANSITION (Aróstica et al. 2025, eq. 6)."""
    half = 0.5 * (1.0 + math.tanh((t - onset) / TRANSITION))
    return half if rising else 1.0 - half


def tabulate(rate, peak, steps=20_000):
    """Integrate dy/dt = -decay(t) y + source(t) once (implicit Euler) and scale its peak."""
    t = np.linspace(0.0, DURATION, steps + 1)
    y = np.zeros_like(t)
    h = t[1] - t[0]
    for i in range(1, len(t)):
        decay, source = rate(t[i])
        y[i] = (y[i - 1] + h * source) / (1.0 + h * decay)
    return t, y * (peak / y.max() if y.max() > 0 else 0.0)


def active_rate(t):
    """Active stress, eqs. (5)-(6): driven while the activation a(t) is positive."""
    f = smooth_step(t, CONTRACTION_ONSET, True) * smooth_step(t, RELAXATION_ONSET, False)
    a = 0.005 * f - 0.030 * (1.0 - f)          # 1/ms: +5 /s contracting, -30 /s relaxing
    return abs(a), max(a, 0.0)


def pressure_rate(t):
    """Cavity pressure, eqs. (7)-(8): a filling (pre-load) term plus the systolic rise."""
    f = smooth_step(t, PRESSURE_ONSET, True) * smooth_step(t, RELAXATION_ONSET, False)
    g = smooth_step(t, RELAXATION_ONSET, False)
    b = 0.005 * f - 0.030 * (1.0 - f) + 0.005 * g + 0.001
    return abs(b), 16.0 * max(b, 0.0) + 0.007 * max(g, 0.0)   # table 4, kPa and kPa/ms


TENSION_TABLE = tabulate(active_rate, PEAK_TENSION)
PRESSURE_TABLE = tabulate(pressure_rate, PEAK_PRESSURE)


def tension(t):
    """Active fibre stress (kPa) at time t (ms)."""
    return float(np.interp(t, *TENSION_TABLE))


def pressure(t):
    """Endocardial pressure (kPa) at time t (ms)."""
    return float(np.interp(t, *PRESSURE_TABLE))
'''

_HELIX = '''
def helix_fibres(points):
    """Fibres at reference points: helix angle linear through the wall, from HELIX_ENDO at
    the endocardium to HELIX_EPI at the epicardium (Land et al. 2015, eqs. 2.7-2.12)."""
    shape = points.shape
    t, s, v = mesh.to_parametric(points.reshape(-1, 3))
    rs, rl = mesh.radii(t)
    u = -torch.pi + s * (torch.pi - torch.arccos(mesh.base_z / rl))
    along = torch.stack((rs * torch.cos(u) * torch.cos(v), rs * torch.cos(u) * torch.sin(v),
                         -rl * torch.sin(u)), dim=-1)                     # apex -> base
    around = torch.stack((-rs * torch.sin(u) * torch.sin(v), rs * torch.sin(u) * torch.cos(v),
                          torch.zeros_like(v)), dim=-1)                  # circumferential
    alpha = torch.deg2rad(HELIX_ENDO + t * (HELIX_EPI - HELIX_ENDO))
    f = (along / along.norm(dim=-1, keepdim=True) * torch.sin(alpha)[:, None]
         + around / around.norm(dim=-1, keepdim=True) * torch.cos(alpha)[:, None])
    return f.reshape(shape)
'''


def _material_code(mat: MechMaterial, active: str | None) -> tuple[str, list[str], bool]:
    """Source of ``material = ...``, the material classes used, and the bulk-modulus mode."""
    penalty = mat.volume == "penalty"
    lit = codegen.lit
    if mat.law == "guccione":
        base = codegen.call("GuccioneMaterial", C=lit(mat.C), bf=lit(mat.bf), bt=lit(mat.bt),
                            bfs=lit(mat.bfs))
        classes = ["GuccioneMaterial", "IsochoricMaterial"]
    elif mat.law == "neo_hookean":
        base = f"NeoHookeanMaterial(mu={lit(mat.mu)})"
        classes = ["NeoHookeanMaterial", "IsochoricMaterial"]
    else:
        base = None
        classes = ["HolzapfelOgdenMaterial"]
    name = "passive" if active else "material"
    lines = []
    if mat.law == "holzapfel_ogden":
        kwargs = {k: lit(float(getattr(mat, k)))
                  for k in ("a", "b", "a_f", "b_f", "a_s", "b_s", "a_fs", "b_fs")}
        kwargs["bulk_modulus"] = "BULK_MODULUS" if penalty else "0.0"
        lines.append(codegen.call("HolzapfelOgdenMaterial", prefix=f"{name} = ", **kwargs))
    elif penalty:
        lines.append(f"{name} = VolumetricPenalty(IsochoricMaterial({base}), BULK_MODULUS)")
        classes.append("VolumetricPenalty")
    else:
        lines.append(f"{name} = IsochoricMaterial({base})")
    if active:
        classes.append("ActiveStressMaterial")
        lines.append(f"material = ActiveStressMaterial(passive, {active})")
    return "\n".join(lines), classes, penalty


def build_script(params: MechanicsParams, ctx: ScriptContext) -> str:
    geo, mat, load, solver, out = (params.geometry, params.material, params.loading,
                                   params.solver, params.outputs)
    lit = codegen.lit
    dynamic = load.mode == "dynamic"
    box = geo.shape == "box"
    title = ctx.run_name or "Mechanics run"
    variant = params.variant_label()
    if box:
        where = (f"a {geo.length:g} × {geo.width:g} × {geo.height:g} mm beam "
                 f"({geo.nx} × {geo.ny} × {geo.nz} order-{geo.order} hexahedra)")
    else:
        where = (f"a truncated-ellipsoid ventricle (endo {geo.endo_radius:g}/{geo.endo_length:g} "
                 f"mm, epi {geo.epi_radius:g}/{geo.epi_length:g} mm, base at z = {geo.base_z:g} "
                 f"mm; {geo.transmural} × {geo.longitudinal} × {geo.circumferential} "
                 f"order-{geo.order} hexahedra)")
    how = ("one heartbeat in time (generalized-α), with cavity pressure and active stress "
           "after Aróstica et al. (2025), benchmark 1" if dynamic else
           "loads ramped from zero to full (quasi-static), as in Land et al. (2015)")
    script = codegen.Script(codegen.docstring(
        title, f"Mechanics · {variant}: {LAWS[mat.law]} tissue on {where}.", ctx,
        [f"Solved with {how}.", "Units: mm, kPa, ms."]))

    script.imports.add("from torchcor.mechanics import Mechanics")
    mesh_class = "StructuredBoxMesh" if box else "TruncatedEllipsoidMesh"
    script.imports.add(f"from torchcor.mechanics.mesh import {mesh_class}")
    if dynamic:
        script.imports.add("import math", "stdlib")
        script.imports.add("import numpy as np")

    s = script.settings
    codegen.device_setting(script, ctx)
    s.add("DTYPE", "torch.float64", "mechanics runs in double precision")
    codegen.output_settings(script)
    if mat.fibres == "helix" and mat.law != "neo_hookean":
        s.add("HELIX_ENDO", lit(float(mat.helix_endo)), "° fibre angle, endocardium")
        s.add("HELIX_EPI", lit(float(mat.helix_epi)), "° fibre angle, epicardium")
    s.add("BULK_MODULUS", lit(float(mat.bulk_modulus)), "kPa, incompressibility")
    if dynamic:
        s.add("DURATION", lit(float(load.duration)), "ms, one heartbeat")
        s.add("DT", lit(float(load.dt)), "ms per mechanics step")
        s.add("FRAME_INTERVAL", lit(float(out.frame_interval)), "ms between saved frames")
        s.add("PEAK_PRESSURE", lit(float(load.peak_pressure)), "kPa, endocardial pressure")
        s.add("PEAK_TENSION", lit(float(load.peak_tension)), "kPa, active fibre stress")
        s.add("CONTRACTION_ONSET", lit(float(load.contraction_onset)), "ms")
        s.add("PRESSURE_ONSET", lit(float(load.pressure_onset)), "ms")
        s.add("RELAXATION_ONSET", lit(float(load.relaxation_onset)), "ms")
        s.add("TRANSITION", lit(float(load.transition)), "ms, width of the phase switches")
        s.add("DENSITY", lit(float(load.density)), "kPa·ms²/mm² (1.0 = 1000 kg/m³)")
        s.add("VISCOSITY", lit(float(load.viscosity)), "kPa·ms, Kelvin-Voigt")
    else:
        s.add("PRESSURE", lit(float(load.pressure)), "kPa at full load")
        if load.active_tension > 0:
            s.add("ACTIVE_TENSION", lit(float(load.active_tension)), "kPa at full load")
        s.add("LOAD_STEPS", lit(int(load.load_steps)), "first increment = 1/LOAD_STEPS of the load")

    # 1. set up ---------------------------------------------------------------------------
    script.step("Set up", "setup", "Setting up")
    script.code("RESULTS.mkdir(parents=True, exist_ok=True)")
    if dynamic:
        script.definitions(_SCHEDULES, comment=(
            "Time courses of the cavity pressure and the active stress, integrated once from "
            "their relaxation equations and scaled to the requested peaks."))

    # 2. model ----------------------------------------------------------------------------
    script.step("Mesh, material and loads", "mesh", "Building the model")
    if box:
        script.code(codegen.call(
            "StructuredBoxMesh", prefix="mesh = ", origin="(0.0, 0.0, 0.0)",
            lengths=lit((float(geo.length), float(geo.width), float(geo.height))),
            divisions=lit((int(geo.nx), int(geo.ny), int(geo.nz))), order=lit(int(geo.order)),
            device="DEVICE", dtype="DTYPE"))
        script.comment("Named surfaces: x-, x+, y-, y+, z-, z+.")
    else:
        script.code(codegen.call(
            "TruncatedEllipsoidMesh", prefix="mesh = ",
            divisions=lit((int(geo.transmural), int(geo.longitudinal), int(geo.circumferential))),
            order=lit(int(geo.order)), endo=lit((float(geo.endo_radius), float(geo.endo_length))),
            epi=lit((float(geo.epi_radius), float(geo.epi_length))),
            base_z=lit(float(geo.base_z)), device="DEVICE", dtype="DTYPE"))
        script.comment("Named surfaces: endo, epi, base. Radii are (short, long) in mm.")
    script.code('print(f"{mesh.n_cells} cells, {mesh.n_points} nodes, {mesh.n_dofs} DOF",'
                ' flush=True)')

    script.blank()
    active = None
    if dynamic and load.peak_tension > 0:
        active = "tension"
    elif not dynamic and load.active_tension > 0:
        active = "tension=ACTIVE_TENSION, ramp=True"
    material, classes, penalty = _material_code(mat, active)
    script.imports.add("from torchcor.mechanics.material import "
                       + ", ".join(sorted({*classes, "MaterialAxes"}
                                          if mat.law != "neo_hookean" else set(classes))))
    script.comment({"guccione": "Guccione law (kPa) on the volume-preserving part of the "
                                "deformation",
                    "neo_hookean": "Neo-Hookean law (kPa) on the volume-preserving part",
                    "holzapfel_ogden": "Holzapfel–Ogden law (kPa)"}[mat.law]
                   + ("; volume change is penalised with BULK_MODULUS" if penalty else
                      "; the solver enforces J = 1 with BULK_MODULUS as the augmented "
                      "Lagrangian penalty")
                   + ("; active stress acts along the fibres." if active else "."))
    script.code(material)
    if mat.law == "neo_hookean":
        axes = "None"
    elif mat.fibres == "helix":
        script.definitions(_HELIX)
        axes = "MaterialAxes(f=helix_fibres)"
    else:
        axes = "MaterialAxes(f=(1.0, 0.0, 0.0))"
    script.code(f"axes = {axes}" + ("  # fibres along x" if mat.fibres == "x" and axes != "None"
                                     else ""))

    script.blank()
    boundary_classes = {"FollowerPressure"}
    lines = ["boundary = ["]
    if load.support == "clamp":
        boundary_classes.add("DirichletBC")
        lines.append(f"    DirichletBC.on_surface(mesh, {lit(load.fixed)}),  # held fixed")
    else:
        boundary_classes.add("RobinBC")
        lines.append(codegen.call("RobinBC.on_surface", "mesh", '"epi"', prefix="    ",
                                  stiffness=lit(float(load.epi_stiffness)),
                                  damping=lit(float(load.spring_damping)),
                                  normal_only="True") + ",  # pericardium")
        lines.append(codegen.call("RobinBC.on_surface", "mesh", '"base"', prefix="    ",
                                  stiffness=lit(float(load.base_stiffness)),
                                  damping=lit(float(load.spring_damping))) + ",")
    lines.append(f"    FollowerPressure.on_surface(mesh, {lit(load.pressure_surface)}, "
                 f"{'pressure' if dynamic else 'PRESSURE'}),")
    lines.append("]")
    script.imports.add("from torchcor.mechanics.boundary import "
                       + ", ".join(sorted(boundary_classes)))
    script.comment("Supports and the follower pressure (it turns with the deforming surface).")
    script.code("\n".join(lines))
    sim_kwargs = {"axes": "axes", "boundary": "boundary",
                  "bulk_modulus": "None" if penalty else "BULK_MODULUS"}
    if dynamic:
        sim_kwargs.update(density="DENSITY", viscosity="VISCOSITY")
    script.code(codegen.call("Mechanics", "mesh", "material", prefix="sim = ", **sim_kwargs))

    # 3. solve ----------------------------------------------------------------------------
    options = {"newton_rtol": lit(solver.newton_rtol), "newton_atol": lit(solver.newton_atol),
               "max_newton": lit(int(solver.max_newton))}
    if dynamic:
        script.step("Solve one heartbeat", "solve", "Solving",
                    note="TorchCor prints a line per time step; studio.total turns them into "
                         "progress. The observer keeps a surface frame every FRAME_INTERVAL.")
        script.code('''
            frames = studio.FrameRecorder(studio.mechanics_surface(mesh), interval=FRAME_INTERVAL)


            def observer(t, u, v, a):
                """Called after every accepted time step with displacement, velocity, acceleration."""
                if frames.due(t):
                    frames.add(t, displacement=u.reshape(-1, 3))


            studio.total(DURATION, "ms")
            with studio.SolverLog() as log:
        ''')
        rho = ("2.0 / 3.0" if math.isclose(load.rho_infinity, 2.0 / 3.0)
               else lit(float(load.rho_infinity)))
        script.code(codegen.call("sim.solve_dynamic", "DURATION", dt="DT", verbose="True",
                                 observer="observer", raise_on_failure="False",
                                 rho_infinity=rho, **options), indent="    ")
        script.code("""
            if frames.times[-1] < sim.report.load_factor:  # always keep the final state
                frames.add(sim.report.load_factor, displacement=sim.displacement)
        """)
        label, unit = "Time", "ms"
    else:
        script.step("Solve", "solve", "Solving",
                    note="TorchCor prints a line per accepted load step (the app shows them as "
                         "progress); SolverLog keeps them for the convergence plot.")
        script.code('''
            frames = studio.FrameRecorder(studio.mechanics_surface(mesh))
            frames.add(0.0, displacement=torch.zeros_like(mesh.points))  # reference state
            with studio.SolverLog() as log:
        ''')
        script.code(codegen.call("sim.solve", load_steps="LOAD_STEPS", verbose="True",
                                 raise_on_failure="False", **options), indent="    ")
        script.code("frames.add(sim.report.load_factor, displacement=sim.displacement)")
        label, unit = "Load factor λ", ""
    script.code('print(f"solve: {sim.report}", flush=True)')

    # 4. save -----------------------------------------------------------------------------
    script.step("Save", "export", "Saving results")
    script.code('torch.save(sim.displacement.cpu(), RESULTS / "displacement.pt")'
                "  # (n_points, 3) mm")
    if out.vtk:
        script.code('sim.to_vtk(RESULTS / "displacement.vtu")')
    script.code(f'bundle = studio.Bundle("mechanics", title={lit(title)})')
    if box:
        point = f"({geo.length:g}, {geo.width / 2:g}, {geo.height:g})"
        script.code(f'''
            # Benchmark quantity of Land et al. problem 1: the deformed far top edge.
            tip = sim.probe([[{geo.length!r}, {geo.width / 2!r}, {geo.height!r}]])[0].tolist()
            print(f"deformed z of {point}: {{tip[2]:.4f}} mm", flush=True)
            bundle.stat("Tip z", tip[2], "mm", "Deformed z of the point {point}")
        ''')
    else:
        script.code('''
            # Benchmark quantities of Land et al. problems 2 and 3: deformed apex heights.
            apex = sim.probe(mesh.points_at([0.0, 1.0], [0.0, 0.0], [0.0, 0.0]))[:, 2].tolist()
            print(f"apex z: endocardium {apex[0]:.3f} mm, epicardium {apex[1]:.3f} mm", flush=True)
            bundle.stat("Endocardial apex z", apex[0], "mm", "Deformed height of the inner apex")
            bundle.stat("Epicardial apex z", apex[1], "mm", "Deformed height of the outer apex")
        ''')
    script.code(f'''
        studio.export_mechanics(frames, bundle, frame_label={lit(label)}, frame_unit={lit(unit)},
                                log=log, report=sim.report, n_dofs=mesh.n_dofs)
        bundle.write()
        if not sim.report.converged:
            raise SystemExit(f"the mechanics solve did not converge: {{sim.report}}")

        studio.stage("done", "Completed")
    ''')
    return script.render()


# =============================================================================== spec
SPEC = register(SimulationSpec(
    kind=SimulationKind.MECHANICS,
    title="Mechanics",
    tagline="Finite-strain deformation of a beam or a ventricle",
    description="Inflate, contract or beat an idealised ventricle, or bend the benchmark beam, "
                "with hyperelastic myocardium laws and follower pressures. Shows the "
                "deforming geometry and the solver's convergence.",
    icon="box",
    params_model=MechanicsParams,
    build_script=build_script,
    stages=STAGES,
    demos=DEMOS,
    panels=PANELS,
    check=check,
    input_use=input_use,
    badge="MECH",
))
