"""Simulation kinds the studio can configure: EP, ECG, mechanics and electromechanics."""

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
    get_spec,
    register,
    specs,
)
from torchcor_studio.core.simulations.fields import Section, field_meta, param

__all__ = [
    "Cost",
    "Demo",
    "InputUse",
    "Issue",
    "ScriptContext",
    "Section",
    "Severity",
    "SimulationKind",
    "SimulationSpec",
    "Stage",
    "field_meta",
    "get_spec",
    "param",
    "register",
    "specs",
]
