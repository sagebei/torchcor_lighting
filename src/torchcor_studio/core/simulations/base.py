"""Simulation specifications: the plug-in point for each kind of TorchCor study.

A :class:`SimulationSpec` bundles everything the studio needs for one simulation kind:

* ``params_model`` — a pydantic model (built from :func:`~.fields.param` fields) whose
  defaults are a complete, runnable configuration;
* ``build_script`` — renders a readable, self-contained ``run.py`` for TorchCor;
* ``demos`` — ready-to-run set-ups, most of them ports of TorchCor's own demos, with the
  data case each one reads ("Biventricle, five pacing sites" reads ``ep/ventricle``);
* ``panels`` — how the Setup step groups the parameters (:mod:`.panels`), with
  ``input_use`` (does the input folder matter?);
* ``check`` — validation against the chosen input folder (missing mesh, unknown region);
* ``stages`` — the pipeline shown as a stage tracker while the run is live.

Register a spec with :func:`register`; the UI discovers specs through :func:`specs`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from torchcor_studio.core.models import InputInfo

if TYPE_CHECKING:
    from torchcor_studio.core.simulations.panels import Panel


class SimulationKind(StrEnum):
    EP = "ep"
    ECG = "ecg"
    MECHANICS = "mechanics"
    EM = "em"
    SCRIPT = "script"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    """A validation finding; errors block the Run button, warnings do not."""

    severity: Severity
    message: str
    field: str = ""  # dotted path of the offending parameter ("stimuli.0.site")


@dataclass(frozen=True)
class Stage:
    id: str
    label: str  # a word or two: the Run page's tracker gives it two stages' width at most


@dataclass(frozen=True)
class ScriptContext:
    """Where and how a generated script will run."""

    device: str  # "cuda:0", "cpu"
    input_dir: str  # folder on the target machine ("" when the geometry is generated)
    run_dir: str  # absolute run folder on the target machine
    run_id: str = ""
    run_name: str = ""
    machine_name: str = ""
    windows: bool = False  # target machine path convention

    @property
    def results_dir(self) -> str:
        sep = "\\" if self.windows else "/"
        return f"{self.run_dir.rstrip(sep)}{sep}results"


class Cost(StrEnum):
    """Rough wall time on one current GPU; never a hardware-specific figure."""

    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"


#: What the input folder is to a set-up: needed, used when given, or ignored.
InputUse = Literal["required", "optional", "unused"]


@dataclass(frozen=True)
class Demo:
    """A ready-to-run set-up, most often a port of one of TorchCor's own demos."""

    key: str  # stable id, unique in its kind: "ventricle"
    title: str  # the card title: "Biventricle, five pacing sites"
    summary: str  # one sentence for the card
    params: Callable[[], BaseModel]  # a fresh parameter model on every call
    case: str | None = None  # its data under the data folder: "ep/ventricle"; None: built in
    #: Sub-folders of the case that show its data is there, when the case is a kind folder
    #: (the prepared ``ecg`` folder holds ``KCL_torso3``, ``heart``, ...).
    requires: tuple[str, ...] = ()
    download: bool = False  # the run downloads the case itself when it is missing
    cost: Cost = Cost.MINUTES
    source: str = ""  # the TorchCor demo it ports (tooltip)
    data: str = ""  # its data set in messages: "KCL torso 3 will be downloaded here..."


ScriptBuilder = Callable[[Any, ScriptContext], str]
Checker = Callable[[Any, InputInfo | None], list[Issue]]
#: ``(params, input folder, params before the edit or None, *, new_input) -> adjusted | None``
#: (``new_input``: the folder was just chosen, or the form just created).
InputFitter = Callable[..., Any]
#: ``(params) -> (use, hint)``: the input folder's role and a one-line hint about it.
InputRole = Callable[[Any], tuple[InputUse, str]]
#: ``(params, input folder) -> [(key, value)]``: two to four facts of the run ("Simulated", ...).


@dataclass(frozen=True)
class SimulationSpec:
    kind: SimulationKind
    title: str  # "Electrophysiology"
    tagline: str  # one line for the type picker
    description: str  # two-three sentences for the picker card
    icon: str  # icon name from ui.icons
    params_model: type[BaseModel]
    build_script: ScriptBuilder
    stages: tuple[Stage, ...]
    #: Ready-to-run set-ups; the first one is the defaults.
    demos: tuple[Demo, ...] = ()
    #: How the Setup step groups the parameters (empty: one panel per section).
    panels: tuple[Panel, ...] = ()
    check: Checker | None = None
    #: Short label for badges ("EP", "ECG", "MECH", "EM").
    badge: str = ""
    #: Fills in parameters that depend on the input folder (e.g. stimulus files). It is
    #: given the previous parameters after an edit, or None when the folder, demo or
    #: kind changed, and returns new parameters, or None to keep them.
    fit_input: InputFitter | None = None
    #: The input folder's role for given parameters (None: optional, no hint).
    input_use: InputRole | None = None

    def defaults(self) -> BaseModel:
        return self.params_model()

    def demo(self, key: str) -> Demo:
        """The demo with this key (``KeyError`` if there is none)."""
        for demo in self.demos:
            if demo.key == key:
                return demo
        raise KeyError(key)

    def input_role(self, params: BaseModel) -> tuple[InputUse, str]:
        """Is the input folder required, optional or unused for ``params``, and a hint."""
        return self.input_use(params) if self.input_use else ("optional", "")

    def validate(self, params: BaseModel, info: InputInfo | None) -> list[Issue]:
        return list(self.check(params, info)) if self.check else []

    def variant_of(self, params: BaseModel) -> str:
        """Short sub-type label for the history list (e.g. "Monodomain")."""
        fn = getattr(params, "variant_label", None)
        return fn() if callable(fn) else ""


_REGISTRY: dict[SimulationKind, SimulationSpec] = {}


def register(spec: SimulationSpec) -> SimulationSpec:
    _REGISTRY[spec.kind] = spec
    return spec


def specs() -> list[SimulationSpec]:
    _ensure_loaded()
    order = [SimulationKind.EP, SimulationKind.ECG, SimulationKind.MECHANICS, SimulationKind.EM,
             SimulationKind.SCRIPT]
    return [_REGISTRY[k] for k in order if k in _REGISTRY]


def get_spec(kind: str | SimulationKind) -> SimulationSpec:
    _ensure_loaded()
    return _REGISTRY[SimulationKind(kind)]


_MODULES = ("ep", "ecg", "mechanics", "em", "script")
_loaded = False
#: Simulation modules that failed to import, with the reason (shown in the UI, never fatal).
LOAD_ERRORS: dict[str, str] = {}


def _ensure_loaded() -> None:
    """Import every simulation module once; a broken one is reported, not fatal."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    import importlib
    import logging

    for name in _MODULES:
        try:
            importlib.import_module(f"torchcor_studio.core.simulations.{name}")
        except Exception as exc:  # noqa: BLE001 - one bad kind must not take down the app
            LOAD_ERRORS[name] = f"{type(exc).__name__}: {exc}"
            logging.getLogger(__name__).exception("could not load simulation kind %r", name)
