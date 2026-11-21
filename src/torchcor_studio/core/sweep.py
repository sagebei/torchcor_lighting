"""Parameter sweeps: vary one numeric parameter over several values, spread over devices.

A sweep is a set of ordinary runs. Runs that share a device are chained with the
runtime's queue (``manifest["after"]``) so a GPU never runs two of them at once.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from torchcor_studio.core.models import DeviceKind, GPUSample
from torchcor_studio.core.simulations.fields import field_meta

Path_ = tuple[Any, ...]


@dataclass(frozen=True)
class SweepTarget:
    path: Path_
    label: str  # "Timing › Duration", "Stimuli #2 › Start"
    unit: str
    current: float
    integer: bool = False

    @property
    def key(self) -> str:
        return ".".join(str(p) for p in self.path)


def numeric_parameters(model: BaseModel) -> list[SweepTarget]:
    """Every int/float parameter of a filled-in model, with a readable label."""
    out: list[SweepTarget] = []

    def walk(obj: BaseModel, path: Path_, prefix: str) -> None:
        for name, info in type(obj).model_fields.items():
            value = getattr(obj, name)
            meta = field_meta(type(obj), name)
            title = meta.get("title") or name
            label = f"{prefix}{title}"
            if isinstance(value, BaseModel):
                walk(value, (*path, name), f"{label} › ")
            elif isinstance(value, list) and value and all(isinstance(v, BaseModel) for v in value):
                for i, item in enumerate(value):
                    walk(item, (*path, name, i), f"{label} #{i + 1} › ")
            elif isinstance(value, bool) or meta.get("widget") in ("code",):
                continue
            elif isinstance(value, int | float) and info.annotation is not None:
                out.append(SweepTarget((*path, name), label, meta.get("unit", ""), float(value),
                                       integer=isinstance(value, int)))

    walk(model, (), "")
    return out


_RANGE = re.compile(r"^\s*([-+\d.eE]+)\s*:\s*([-+\d.eE]+)\s*:\s*(\d+)\s*(?::\s*(log))?\s*$")


def parse_values(text: str, integer: bool = False) -> list[float]:
    """``"0.1, 0.2, 0.4"`` or ``"0.1:0.5:5"`` (linear) or ``"1e-5:1e-3:3:log"`` (geometric)."""
    text = text.strip()
    if not text:
        return []
    m = _RANGE.match(text)
    if m:
        start, stop, count = float(m.group(1)), float(m.group(2)), int(m.group(3))
        if count < 1:
            raise ValueError("a range needs at least one value")
        if count == 1:
            values = [start]
        elif m.group(4):
            if start <= 0 or stop <= 0:
                raise ValueError("a log range needs positive bounds")
            ratio = (stop / start) ** (1 / (count - 1))
            values = [start * ratio**i for i in range(count)]
        else:
            step = (stop - start) / (count - 1)
            values = [start + step * i for i in range(count)]
    else:
        try:
            values = [float(v) for v in re.split(r"[,\s;]+", text) if v]
        except ValueError as exc:
            raise ValueError("use numbers separated by commas, or start:stop:count") from exc
    if integer:
        values = [float(round(v)) for v in values]
    out: list[float] = []
    for v in values:  # de-duplicate, keep order
        if not any(math.isclose(v, w, rel_tol=1e-12, abs_tol=1e-15) for w in out):
            out.append(v)
    return out


def apply(model: BaseModel, target: SweepTarget, value: float) -> BaseModel:
    """A validated copy of ``model`` with the target parameter set to ``value``."""
    data = model.model_dump(mode="python")
    node = data
    for key in target.path[:-1]:
        node = node[key]
    node[target.path[-1]] = round(value) if target.integer else value
    return type(model).model_validate(data)


def format_value(value: float, unit: str = "") -> str:
    text = f"{value:.4g}"
    return f"{text} {unit}".strip()


@dataclass(frozen=True)
class Slot:
    """Where one sweep run goes: the device, and which earlier run (index) it queues behind."""

    device: GPUSample
    after: int | None


def plan(samples: list[GPUSample], count: int, *, spread: bool, preferred: GPUSample | None) -> list[Slot]:
    """Assign ``count`` runs to devices.

    ``spread`` round-robins over every runnable CUDA GPU (least busy first, preferred
    first); otherwise all runs go to ``preferred``. Runs sharing a device are chained.
    """
    runnable = [s for s in samples if s.runnable]
    if not runnable and preferred is None:
        raise ValueError("no device available")
    if spread:
        gpus = sorted((s for s in runnable if s.kind == DeviceKind.CUDA), key=lambda s: (s.utilization, s.index))
        if preferred is not None and preferred.kind == DeviceKind.CUDA:
            gpus = [preferred, *[g for g in gpus if g.key != preferred.key]]
        devices = gpus or [preferred or runnable[0]]
    else:
        devices = [preferred or runnable[0]]
    last_on: dict[str, int] = {}
    slots: list[Slot] = []
    for i in range(count):
        device = devices[i % len(devices)]
        slots.append(Slot(device, last_on.get(device.key)))
        last_on[device.key] = i
    return slots
