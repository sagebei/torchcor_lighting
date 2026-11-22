"""Parameter sections shared by several simulation kinds, and what they mean for TorchCor.

Each section is a :class:`~.fields.Section` whose defaults are sensible on their own; the
simulation modules combine them into their parameter trees, use the ``*_issues`` helpers
in their ``check`` functions and the ``*_code`` helpers when writing scripts.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from torchcor_studio.core.models import InputInfo
from torchcor_studio.core.simulations import codegen
from torchcor_studio.core.simulations.base import Issue, Severity
from torchcor_studio.core.simulations.fields import Section, param

PRECISIONS = {"float32": "Single (float32)", "float64": "Double (float64)"}
CELL_TYPES = {"ENDO": "Endocardial", "MID": "Mid-myocardial", "EPI": "Epicardial"}
SEXES = {"MIX": "Mixed", "MALE": "Male", "FEMALE": "Female"}


# =============================================================================== cell models
@dataclass(frozen=True)
class CellModel:
    """What the studio needs to know about one ``torchcor.ionic`` model."""

    label: str
    #: Studio cell type -> the model's own spelling (None: the model has no cell types).
    cell_types: dict[str, str] | None = None
    sex: bool = False
    #: Does the constructor accept ``region_ids``?
    region_ids: bool = True
    #: Does the model report cytosolic calcium (``calcium()``)?
    calcium: bool = True
    #: Activation / repolarisation thresholds, in the model's voltage unit.
    activation: float = 0.0
    repolarisation: float = -70.0
    #: A stimulus that reliably excites tissue: (strength, duration ms).
    stimulus: tuple[float, float] = (50.0, 2.0)
    #: Voltage unit ("mV", or "" for a normalised 0..1 potential).
    unit: str = "mV"
    #: Largest time step (ms) at which a paced cell stays finite (None: no limit seen).
    max_dt: float | None = None

    @property
    def ventricular(self) -> bool:
        return self.cell_types is not None


_VENTRICLE = {"ENDO": "ENDO", "MID": "MCELL", "EPI": "EPI"}

CELL_MODELS: dict[str, CellModel] = {
    "TenTusscherPanfilov": CellModel("ten Tusscher–Panfilov 2006 (ventricle)", _VENTRICLE),
    "OHaraRudy": CellModel("O'Hara–Rudy 2011 (ventricle)", _VENTRICLE),
    "ToRORd": CellModel("ToR-ORd 2019 (ventricle)", _VENTRICLE),
    "ToRORdDynCl": CellModel("ToR-ORd-dynCl 2020 (ventricle)",
                             {"ENDO": "ENDO", "MID": "M", "EPI": "EPI"}, max_dt=0.02),
    "TWorld": CellModel("T-World (human ventricle)", {"ENDO": "ENDO", "MID": "MID", "EPI": "EPI"},
                        sex=True, max_dt=0.01),
    "CourtemancheRamirezNattel": CellModel("Courtemanche–Ramirez–Nattel 1998 (atrium)",
                                           activation=-10.0),
    "BuenoOrovio": CellModel("Bueno-Orovio–Cherry–Fenton 2008 (minimal ventricle)",
                             calcium=False, activation=-40.0),
    "ModifiedMS2v": CellModel("Modified Mitchell–Schaeffer (two-variable, mV)", calcium=False,
                              activation=-40.0),
    "MitchellSchaeffer": CellModel("Mitchell–Schaeffer 2003 (two-variable, normalised)",
                                   calcium=False, activation=0.5, repolarisation=0.1,
                                   stimulus=(0.5, 2.0), unit=""),
    "AlievPanfilov": CellModel("Aliev–Panfilov 1996 (two-variable, mV)", region_ids=False,
                               calcium=False, activation=-40.0, stimulus=(40.0, 2.0)),
}
_TYPED = [name for name, model in CELL_MODELS.items() if model.cell_types]
_SEXED = [name for name, model in CELL_MODELS.items() if model.sex]


class CellGroup(Section):
    """One cell model and the regions it covers (a row of the "Cell models" table)."""

    regions: list[int] = param(
        title="Regions", widget="regions", default_factory=list,
        help="Element tags this model applies to; empty = every region no other row lists.")
    model: str = param("TenTusscherPanfilov", "Cell model", widget="combo",
                       choices={name: model.label for name, model in CELL_MODELS.items()},
                       help="Ionic model of the membrane currents (torchcor.ionic).")
    cell_type: str = param("EPI", "Cell type", choices=CELL_TYPES, visible_if={"model": _TYPED},
                           help="Transmural variant of the ventricular model.")
    sex: str = param("MIX", "Sex", choices=SEXES, visible_if={"model": _SEXED}, advanced=True,
                     help="Sex-specific parameter set of the T-World model.")
    scaling: str = param("", "Parameter scaling", advanced=True, placeholder="GKs=0.5, Gto=2",
                         widget="mono",
                         help="Multiply model parameters, e.g. 'GNa=1.2, GCaL=0.5' "
                              "(names as in the TorchCor model).")

    @property
    def info(self) -> CellModel:
        return CELL_MODELS[self.model]


class PrePacing(Section):
    """Pace each cell model alone to its steady state before the simulation."""

    enabled: bool = param(False, "Pre-pace each cell model to steady state",
                          help="Pace one cell of each model to steady state first "
                               "(torchcor PrePace); every node then starts from that state.")
    cycle_length: float = param(800.0, "Cycle length", unit="ms", min=50.0, max=10000.0,
                                step=50.0, visible_if={"enabled": True},
                                help="Time between pre-pacing stimuli: the heart rate to settle at.")
    beats: int = param(50, "Beats", min=1, max=5000, step=10, visible_if={"enabled": True},
                       help="Number of beats paced; ion concentrations take tens to hundreds.")
    stimulus: float = param(60.0, "Stimulus strength", unit="mV/ms", min=0.0, step=5.0,
                            visible_if={"enabled": True}, advanced=True,
                            help="Single-cell stimulus; about 0.5 for the normalised "
                                 "Mitchell–Schaeffer model.")
    duration: float = param(1.0, "Stimulus duration", unit="ms", min=0.01, step=0.5,
                            visible_if={"enabled": True}, advanced=True,
                            help="Length of each pre-pacing stimulus.")


# =============================================================================== time
class Timing(Section):
    """Simulated time, time step and output interval of an EP solve."""

    duration: float = param(100.0, "Duration", unit="ms", min=1.0, max=1e6, step=50.0,
                            help="Simulated time.")
    dt: float = param(0.01, "Time step", unit="ms", min=0.001, max=1.0, step=0.005, decimals=3,
                      help="Integration step of the cell models and the diffusion solve.")
    snapshot_interval: float = param(1.0, "Output interval", unit="ms", min=0.01, max=1000.0,
                                     step=0.5, help="Time between saved Vm frames; a whole "
                                                    "number of time steps.")

    @property
    def frames(self) -> int:
        return int(self.duration // self.snapshot_interval) + 1


def _whole_multiple(value: float, step: float) -> bool:
    ratio = value / step
    return step > 0 and abs(ratio - round(ratio)) < 1e-6 * max(1.0, ratio)


def timing_issues(timing: Timing, prefix: str = "timing") -> list[Issue]:
    issues = []
    if not _whole_multiple(timing.snapshot_interval, timing.dt):
        issues.append(Issue(Severity.ERROR,
                            f"The output interval ({timing.snapshot_interval:g} ms) must be a "
                            f"whole number of time steps ({timing.dt:g} ms).",
                            f"{prefix}.snapshot_interval"))
    if timing.snapshot_interval > timing.duration:
        issues.append(Issue(Severity.ERROR, "The output interval is longer than the simulation.",
                            f"{prefix}.snapshot_interval"))
    elif timing.snapshot_interval == timing.duration:
        issues.append(Issue(Severity.WARNING,
                            "With one output frame TorchCor prints no progress and the run "
                            "keeps only the start and end states.",
                            f"{prefix}.snapshot_interval"))
    if not _whole_multiple(timing.duration, timing.dt):
        issues.append(Issue(Severity.WARNING,
                            f"The duration is not a whole number of time steps; TorchCor runs "
                            f"{int(timing.duration / timing.dt)} steps.", f"{prefix}.duration"))
    return issues


def vm_history_issue(frames: int, nodes: int | None, precision: str,
                     field: str = "timing.snapshot_interval") -> Issue | None:
    """The Vm history (frames x nodes) stays on the device for the whole solve."""
    if not nodes:
        return None
    size = frames * nodes * (4 if precision == "float32" else 8)
    gib = size / 2**30
    text = (f"The Vm history ({frames:,} frames × {nodes:,} nodes) needs about "
            f"{_bytes(size)} on the device during the solve")
    if gib > 8:
        return Issue(Severity.WARNING, text + "; increase the output interval or use float32.",
                     field)
    if gib > 0.25:
        return Issue(Severity.INFO, text + ".", field)
    return None


def _bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# =============================================================================== stimuli
class Stimulus(Section):
    """One stimulus (a row of the "Stimuli" table)."""

    site: str = param("stimulus.vtx", "Site", widget="vtx", placeholder="Choose a .vtx site",
                      help="Nodes to stimulate: a .vtx file of the mesh folder.")
    start: float = param(0.0, "Start", unit="ms", min=0.0, step=1.0, help="First pulse.")
    duration: float = param(2.0, "Duration", unit="ms", min=0.01, step=0.5,
                            help="Length of each pulse.")
    intensity: float = param(50.0, "Strength", unit="mV/ms", min=0.0, step=10.0,
                             help="Stimulus current per membrane capacitance (µA/µF ≡ mV/ms).")
    period: float = param(0.0, "Period", unit="ms", min=0.0, step=50.0,
                          help="Time between pulses; 0 fires once.")
    count: int = param(1, "Beats", min=1, max=1000, help="Number of pulses when a period is set.")


def stimulus_kwargs(stim: Stimulus) -> dict[str, str]:
    """``add_stimulus`` keyword arguments (source text) of one stimulus row."""
    kwargs = {"start": codegen.lit(float(stim.start)),
              "duration": codegen.lit(float(stim.duration)),
              "intensity": codegen.lit(float(stim.intensity))}
    if stim.period > 0:
        kwargs["period"] = codegen.lit(float(stim.period))
        kwargs["count"] = codegen.lit(int(stim.count))
    return kwargs


def stimulus_issues(stimuli: Sequence[Stimulus], sites: Iterable[str] | None,
                    prefix: str = "stimuli", duration: float | None = None,
                    where: str = "the input folder") -> list[Issue]:
    """Missing sites (when ``sites`` is known), empty tables, pulses after the end."""
    issues = []
    if not stimuli:
        issues.append(Issue(Severity.ERROR, "Add at least one stimulus.", prefix))
    known = None if sites is None else set(sites)
    for i, stim in enumerate(stimuli):
        field = f"{prefix}.{i}"
        name = stim.site.strip()
        if not name:
            issues.append(Issue(Severity.ERROR,
                                "Choose a stimulus site" + ("." if known is None
                                                            else f"{_choices(known)}."),
                                f"{field}.site"))
        elif known is not None and name not in known:
            issues.append(Issue(Severity.ERROR, f"{name} is not in {where}{_choices(known)}.",
                                f"{field}.site"))
        if duration is not None and stim.start >= duration:
            issues.append(Issue(Severity.WARNING,
                                f"The stimulus at {stim.site} starts after the simulation ends.",
                                f"{field}.start"))
        if stim.period > 0 and stim.duration >= stim.period:
            issues.append(Issue(Severity.ERROR, "A pulse must be shorter than its period.",
                                f"{field}.duration"))
    return issues


def _choices(names: Iterable[str]) -> str:
    names = sorted(names)
    if not names:
        return " (it has no .vtx files)"
    shown = ", ".join(names[:6]) + (", …" if len(names) > 6 else "")
    return f" (it has {shown})"


# =============================================================================== tissue
class Conductivity(Section):
    """Tissue conductivity of some regions (a row of the "Conductivities" table)."""

    regions: list[int] = param(title="Regions", widget="regions", default_factory=list,
                               help="Element tags these values apply to; empty = all regions.")
    il: float = param(0.17, "Intracellular, along fibres", unit="S/m", min=0.0, step=0.01,
                      decimals=4, group="Intracellular", short="Along fibres",
                      help="Intracellular conductivity along the fibre direction.")
    it: float = param(0.019, "Intracellular, across fibres", unit="S/m", min=0.0, step=0.01,
                      decimals=4, group="Intracellular", short="Across",
                      help="Intracellular conductivity across the fibres.")
    el: float = param(0.62, "Extracellular, along fibres", unit="S/m", min=0.0, step=0.01,
                      decimals=4, visible_if={"../conductivity_mode": "bidomain"},
                      group="Extracellular", short="Along fibres",
                      help="Extracellular conductivity along the fibres.")
    et: float = param(0.24, "Extracellular, across fibres", unit="S/m", min=0.0, step=0.01,
                      decimals=4, visible_if={"../conductivity_mode": "bidomain"},
                      group="Extracellular", short="Across",
                      help="Extracellular conductivity across the fibres.")


CONDUCTIVITY_MODES = {
    "bidomain": "Intra- and extracellular (harmonic mean)",
    "monodomain": "Effective monodomain values",
}


def conductivity_code(rows: Sequence[Conductivity], mode: str, target: str = "simulator") -> str:
    lines = []
    for row in rows:
        kwargs = {"il": codegen.lit(float(row.il)), "it": codegen.lit(float(row.it))}
        if mode == "bidomain":
            kwargs.update(el=codegen.lit(float(row.el)), et=codegen.lit(float(row.et)))
        lines.append(codegen.call(f"{target}.add_conductivity", codegen.regions(row.regions),
                                  **kwargs))
    return "\n".join(lines)


def conductivity_issues(rows: Sequence[Conductivity], mode: str, known: Sequence[int] | None,
                        prefix: str) -> list[Issue]:
    issues = []
    if not rows:
        issues.append(Issue(Severity.ERROR, "Add the tissue conductivity.", prefix))
        return issues
    for i, row in enumerate(rows):
        if row.il <= 0 or row.it <= 0:
            issues.append(Issue(Severity.ERROR, "Conductivities must be positive.",
                                f"{prefix}.{i}.il"))
        if mode == "bidomain" and (row.el <= 0 or row.et <= 0):
            issues.append(Issue(Severity.ERROR, "Extracellular conductivities must be positive.",
                                f"{prefix}.{i}.el"))
        issues += region_issues(row.regions, known, f"{prefix}.{i}.regions")
    if known:
        covered = set(known) if any(not r.regions for r in rows) else {
            rid for r in rows for rid in r.regions}
        missing = sorted(set(known) - covered)
        if missing:
            issues.append(Issue(Severity.WARNING,
                                f"Regions {_ids(missing)} have no conductivity and will not "
                                "conduct.", prefix))
    return issues


class Velocity(Section):
    """Conduction velocities of some regions (a row of the "Conduction velocities" table)."""

    regions: list[int] = param(title="Regions", widget="regions", default_factory=list,
                               help="Element tags these values apply to; empty = all regions.")
    vel_l: float = param(0.6, "Along fibres", unit="mm/ms", min=0.001, max=10.0, step=0.05,
                         decimals=3, short="Along fibres",
                         help="Conduction velocity along the fibre direction (m/s).")
    vel_t: float = param(0.3, "Across fibres", unit="mm/ms", min=0.001, max=10.0, step=0.05,
                         decimals=3, short="Across", help="Conduction velocity across the fibres.")
    vel_n: float = param(0.0, "Sheet normal", unit="mm/ms", min=0.0, max=10.0, step=0.05,
                         decimals=3, advanced=True, short="Sheet normal",
                         help="Velocity normal to the sheet; 0 = same as across fibres "
                              "(meshes without sheet directions need this).")


def velocity_code(rows: Sequence[Velocity], target: str = "simulator") -> str:
    lines = []
    for row in rows:
        kwargs = {"vel_l": codegen.lit(float(row.vel_l)), "vel_t": codegen.lit(float(row.vel_t))}
        if row.vel_n > 0 and not math.isclose(row.vel_n, row.vel_t):
            kwargs["vel_n"] = codegen.lit(float(row.vel_n))
        lines.append(codegen.call(f"{target}.add_velocity", codegen.regions(row.regions),
                                  **kwargs))
    return "\n".join(lines)


def velocity_issues(rows: Sequence[Velocity], known: Sequence[int] | None,
                    fibre_columns: int | None, prefix: str) -> list[Issue]:
    issues = []
    if not rows:
        issues.append(Issue(Severity.ERROR, "Add the conduction velocities.", prefix))
        return issues
    for i, row in enumerate(rows):
        issues += region_issues(row.regions, known, f"{prefix}.{i}.regions")
        orthotropic = row.vel_n > 0 and not math.isclose(row.vel_n, row.vel_t)
        if orthotropic and fibre_columns == 3:
            issues.append(Issue(Severity.ERROR,
                                "The mesh has no sheet directions, so the sheet-normal velocity "
                                "must equal the across-fibre one (set it to 0).",
                                f"{prefix}.{i}.vel_n"))
    if known and not any(not r.regions for r in rows):
        missing = sorted(set(known) - {rid for r in rows for rid in r.regions})
        if missing:
            issues.append(Issue(Severity.ERROR,
                                f"Regions {_ids(missing)} have no conduction velocity; the "
                                "eikonal solve needs one for every region.", prefix))
    return issues


def eikonal_schedule_issues(stimuli: Sequence[Stimulus], field: str) -> list[Issue]:
    """Reaction–eikonal repeats one activation sequence: every stimulus needs the same
    period and number of beats."""
    if len({(s.period, s.count if s.period > 0 else 1) for s in stimuli}) > 1:
        return [Issue(Severity.ERROR, "Reaction–eikonal repeats one activation sequence, so every "
                                      "stimulus needs the same period and beats.", field)]
    return []


# =============================================================================== regions
def _ids(ids: Iterable[int]) -> str:
    return ", ".join(str(i) for i in ids)


def region_issues(ids: Sequence[int], known: Sequence[int] | None, field: str) -> list[Issue]:
    if not known:
        return []
    unknown = sorted(set(ids) - set(known))
    if unknown:
        return [Issue(Severity.ERROR,
                      f"Region {_ids(unknown)} is not in the mesh (it has {_ids(known)}).", field)]
    return []


# =============================================================================== numerics
class SolverSettings(Section):
    """Precision and the per-step conjugate-gradient solve."""

    precision: str = param("float32", "Precision", choices=PRECISIONS, widget="segmented",
                           summary=True, help="Storage precision of Vm and the cell models; the linear solve "
                                "always runs in float64.")
    a_tol: float = param(1e-5, "Absolute tolerance", min=1e-14, max=1e-1, log=True,
                         advanced=True, short="CG tolerance", summary=True,
                         help="Conjugate-gradient absolute tolerance per step.")
    r_tol: float = param(1e-5, "Relative tolerance", min=1e-14, max=1e-1, log=True,
                         advanced=True, help="Conjugate-gradient relative tolerance per step.")
    max_iter: int = param(100, "Iteration limit", min=1, max=100000, advanced=True,
                          short="Iteration limit", summary=True,
                          help="The run stops if one time step needs more CG iterations.")
    linear_guess: bool = param(True, "Extrapolated initial guess", advanced=True,
                               help="Start each solve from a linear extrapolation of the last "
                                    "two steps.")


def dtype_expr(precision: str) -> str:
    return "tc.float32" if precision == "float32" else "tc.float64"


# =============================================================================== cell code
_SCALING = re.compile(r"\s*([A-Za-z_]\w*)\s*[=*:]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*")


def parse_scaling(text: str) -> list[tuple[str, float]]:
    """``"GNa=1.2, Gto*0.5"`` -> ``[("GNa", 1.2), ("Gto", 0.5)]``; raises ValueError."""
    pairs = []
    for item in filter(None, (part.strip() for part in re.split(r"[,;\n]", text))):
        match = _SCALING.fullmatch(item)
        if not match:
            raise ValueError(f"cannot read {item!r}: write name=factor")
        pairs.append((match.group(1), float(match.group(2))))
    return pairs


def cell_group_issues(groups: Sequence[CellGroup], known: Sequence[int] | None,
                      prefix: str = "cells") -> list[Issue]:
    issues = []
    if not groups:
        return [Issue(Severity.ERROR, "Add a cell model.", prefix)]
    catch_all = [i for i, g in enumerate(groups) if not g.regions]
    if len(catch_all) > 1:
        issues.append(Issue(Severity.ERROR,
                            "Only one cell model may leave its regions empty (= all others).",
                            f"{prefix}.{catch_all[1]}.regions"))
    seen: dict[int, int] = {}
    for i, group in enumerate(groups):
        for rid in group.regions:
            if rid in seen:
                issues.append(Issue(Severity.ERROR,
                                    f"Region {rid} is given two cell models (rows {seen[rid] + 1} "
                                    f"and {i + 1}).", f"{prefix}.{i}.regions"))
            seen.setdefault(rid, i)
        issues += region_issues(group.regions, known, f"{prefix}.{i}.regions")
        try:
            parse_scaling(group.scaling)
        except ValueError as exc:
            issues.append(Issue(Severity.ERROR, f"Parameter scaling: {exc}.",
                                f"{prefix}.{i}.scaling"))
    if known and not catch_all:
        missing = sorted(set(known) - set(seen))
        if missing:
            issues.append(Issue(Severity.WARNING,
                                f"Regions {_ids(missing)} have no cell model; their nodes keep "
                                "a constant potential.", prefix))
    units = {g.info.unit for g in groups}
    if len(units) > 1:
        issues.append(Issue(Severity.ERROR,
                            "The normalised Mitchell–Schaeffer model cannot be mixed with "
                            "models in mV.", prefix))
    return issues


def ordered_groups(groups: Sequence[CellGroup]) -> list[CellGroup]:
    """Groups with explicit regions first: TorchCor gives a model without ``region_ids``
    every region that the models before it have not claimed."""
    return sorted(groups, key=lambda g: not g.regions)


def cell_constructor(group: CellGroup, prefix: str = "", dt: str = "DT",
                     device: str = "DEVICE", dtype: str = "DTYPE") -> str:
    """``prefix + Model(dt, region_ids=..., cell_type=..., device=..., dtype=...)``."""
    info = group.info
    kwargs: dict[str, str] = {}
    if group.regions and info.region_ids:
        kwargs["region_ids"] = codegen.regions(group.regions)
    if info.cell_types:
        kwargs["cell_type"] = codegen.lit(info.cell_types[group.cell_type])
    if info.sex:
        kwargs["sex"] = codegen.lit(group.sex)
    kwargs.update(device=device, dtype=dtype)
    return codegen.call(group.model, dt, prefix=prefix, **kwargs)


def cell_names(groups: Sequence[CellGroup]) -> list[str]:
    return ["cell"] if len(groups) == 1 else [f"cell_{i + 1}" for i in range(len(groups))]


def cell_models_code(groups: Sequence[CellGroup], indent: str = "", dt: str = "DT",
                     device: str = "DEVICE", dtype: str = "DTYPE") -> tuple[str, list[str]]:
    """Source building one cell model per group, and the variable names it defines."""
    ordered = ordered_groups(groups)
    names = cell_names(ordered)
    lines = []
    for name, group in zip(names, ordered, strict=True):
        built = cell_constructor(group, prefix=f"{name} = ", dt=dt, device=device, dtype=dtype)
        lines += [indent + line for line in built.split("\n")]
        if not group.info.region_ids:
            lines.append(f"{indent}{name}.region_ids = {codegen.regions(group.regions)}"
                         f"  # {group.model} takes no region_ids argument")
        lines += [f"{indent}{name}.{attr} *= {codegen.lit(factor)}"
                  for attr, factor in parse_scaling(group.scaling)]
    return "\n".join(lines), names


def cell_imports(groups: Iterable[CellGroup]) -> str:
    names = sorted({g.model for g in groups})
    return f"from torchcor.ionic import {', '.join(names)}"


def prepace_code(prepace: PrePacing, cells: str = "CELLS") -> str:
    """Source of the loop that pre-paces each of ``cells``: TorchCor's PrePace compiles the
    model's step once and paces one cell on the CPU, whatever device the model is on."""
    pacer = codegen.call("PrePace", prefix="pacer = ",
                         cycle_length=codegen.lit(float(prepace.cycle_length)),
                         beats=codegen.lit(int(prepace.beats)),
                         stimulus=codegen.lit(float(prepace.stimulus)),
                         duration=codegen.lit(float(prepace.duration)))
    return f'''{pacer}
for i, cell in enumerate({cells}):
    studio.progress(i / len({cells}), f"pre-pacing {{type(cell).__name__}}")
    pacer(cell)  # every node starts from the paced state
    if pacer.history:  # models with calcium
        beat, peak, diastolic = pacer.history[-1]
        print(f"paced {{type(cell).__name__}} for {{beat}} beats: calcium "
              f"{{diastolic:.3f}}-{{peak:.3f}} uM", flush=True)'''


def prepace_issues(prepace: PrePacing, groups: Sequence[CellGroup],
                   prefix: str = "prepace") -> list[Issue]:
    if not prepace.enabled:
        return []
    issues = []
    if any(g.info.unit == "" for g in groups) and prepace.stimulus > 2.0:
        issues.append(Issue(Severity.WARNING,
                            "The Mitchell–Schaeffer model has a normalised voltage: use a "
                            "pre-pacing stimulus of about 0.5.", f"{prefix}.stimulus"))
    if prepace.duration >= prepace.cycle_length:
        issues.append(Issue(Severity.ERROR, "The stimulus must be shorter than the cycle.",
                            f"{prefix}.duration"))
    return issues


def cell_dt_issues(models: Iterable[str], dt: float, field: str) -> list[Issue]:
    """Warn when a cell model is known to blow up (NaN) at this time step."""
    issues = []
    for name in dict.fromkeys(models):
        limit = CELL_MODELS[name].max_dt
        if limit is not None and dt > limit + 1e-12:
            issues.append(Issue(Severity.WARNING,
                                f"{CELL_MODELS[name].label} becomes unstable (NaN) above "
                                f"dt = {limit:g} ms; use a smaller time step.", field))
    return issues


# =============================================================================== input folder
def folder_name(path: str) -> str:
    """The last part of a folder path in either convention ("~/runs/abc/" -> "abc")."""
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def mesh_issues(info: InputInfo | None, field: str) -> list[Issue]:
    """A mesh TorchCor can read: .pts, .elem with Tt/Tr/Ln elements, and .lon
    (``field``: where the issues point, the input folder)."""
    if info is None or not info.exists:
        return [Issue(Severity.ERROR, "Choose an input folder with a mesh (.pts, .elem, .lon).",
                      field)]
    if info.error:
        return [Issue(Severity.ERROR, f"Could not read the input folder: {info.error}", field)]
    if not info.has_mesh:
        return [Issue(Severity.ERROR, "The input folder has no mesh (.pts and .elem).",
                      field)]
    issues = []
    if info.fibre_columns is None:
        issues.append(Issue(Severity.ERROR, "The mesh has no .lon fibre file; TorchCor needs one.",
                            field))
    unsupported = sorted(set(info.elem_types) - {"Tt", "Tr", "Ln"})
    if unsupported:
        issues.append(Issue(Severity.ERROR,
                            f"TorchCor reads tetrahedra, triangles and lines only; the mesh "
                            f"also has {', '.join(unsupported)}.", field))
    for ext in (".pts", ".elem", ".lon"):
        found = [f for f in info.files if f.lower().endswith(ext)]
        if len(found) > 1:
            issues.append(Issue(Severity.WARNING,
                                f"The folder holds several {ext} files ({', '.join(found)}); "
                                "TorchCor reads one of them, so keep one mesh per folder.", field))
    return issues
