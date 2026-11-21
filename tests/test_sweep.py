"""Parameter sweeps: targets, value parsing, application and device planning."""

from __future__ import annotations

import pytest

from torchcor_studio.core.models import DeviceKind, GPUSample
from torchcor_studio.core.simulations.fields import Section, param
from torchcor_studio.core.sweep import apply, numeric_parameters, parse_values, plan


class Stim(Section):
    start: float = param(0.0, "Start", unit="ms")
    count: int = param(1, "Count")


class Timing(Section):
    duration: float = param(500.0, "Duration", unit="ms", min=1)
    adaptive: bool = param(False, "Adaptive")


class Model(Section):
    timing: Timing = param(default_factory=Timing, title="Timing")
    stimuli: list[Stim] = param(default_factory=lambda: [Stim(), Stim(start=5.0)], title="Stimuli")
    name: str = param("x", "Name")


def test_targets_cover_nested_and_rows_but_skip_bools_and_text() -> None:
    labels = {t.label: t for t in numeric_parameters(Model())}
    assert set(labels) == {"Timing › Duration", "Stimuli #1 › Start", "Stimuli #1 › Count",
                           "Stimuli #2 › Start", "Stimuli #2 › Count"}
    assert labels["Stimuli #2 › Start"].path == ("stimuli", 1, "start")
    assert labels["Stimuli #1 › Count"].integer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.1, 0.2 0.4", [0.1, 0.2, 0.4]),
        ("1:3:3", [1.0, 2.0, 3.0]),
        ("1e-5:1e-3:3:log", [1e-5, 1e-4, 1e-3]),
        ("2, 2, 3", [2.0, 3.0]),
        ("", []),
    ],
)
def test_parse_values(text: str, expected: list[float]) -> None:
    assert parse_values(text) == pytest.approx(expected)


def test_parse_values_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_values("fast, slow")
    with pytest.raises(ValueError):
        parse_values("0:1:3:log")


def test_apply_validates_and_rounds_integers() -> None:
    targets = {t.label: t for t in numeric_parameters(Model())}
    out = apply(Model(), targets["Stimuli #2 › Start"], 12.5)
    assert out.stimuli[1].start == 12.5 and out.stimuli[0].start == 0.0
    out = apply(Model(), targets["Stimuli #1 › Count"], 2.6)
    assert out.stimuli[0].count == 3
    with pytest.raises(ValueError):
        apply(Model(), targets["Timing › Duration"], 0.0)  # below min=1


def _gpu(i: int, util: float) -> GPUSample:
    return GPUSample(index=i, name=f"GPU {i}", kind=DeviceKind.CUDA, utilization=util,
                     memory_total_mb=1000)


def test_plan_spreads_and_chains_per_device() -> None:
    gpus = [_gpu(0, 90), _gpu(1, 5), GPUSample(0, "CPU", DeviceKind.CPU)]
    slots = plan(gpus, 5, spread=True, preferred=gpus[1])
    assert [s.device.index for s in slots] == [1, 0, 1, 0, 1]
    assert [s.after for s in slots] == [None, None, 0, 1, 2]


def test_plan_queue_mode_stays_on_one_device() -> None:
    gpus = [_gpu(0, 0), _gpu(1, 0)]
    slots = plan(gpus, 3, spread=False, preferred=gpus[0])
    assert {s.device.key for s in slots} == {"cuda:0"}
    assert [s.after for s in slots] == [None, 0, 1]


def test_plan_falls_back_to_cpu_without_gpus() -> None:
    cpu = GPUSample(0, "CPU", DeviceKind.CPU)
    slots = plan([cpu], 2, spread=True, preferred=cpu)
    assert [s.device.kind for s in slots] == [DeviceKind.CPU, DeviceKind.CPU]
    assert [s.after for s in slots] == [None, 0]
