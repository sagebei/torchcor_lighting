"""Qt-free loader for TorchCor Lighting visualisation bundles (``viz/``).

The bundle format is specified in :mod:`torchcor_studio.runtime.tcs_studio`.
:func:`load_bundle` reads and validates ``index.json`` only; every dataset is loaded
the first time it is accessed (``bundle.surface``, ``bundle.ep`` …), validated against
the surface it belongs to, and cached. Loading is thread-safe, so views call
:meth:`ResultBundle.preload` on a worker thread and then read the datasets on the UI
thread for free.

Per-frame fields stay in their compact on-disk dtype (float16) inside a
:class:`FrameStack` and are converted to float32 one frame at a time.

    bundle = load_bundle(run_dir / "viz")
    bundle.kind, bundle.title, bundle.summary
    vm = bundle.ep.vm.frame(42)            # (N,) float32
"""

from __future__ import annotations

import json
import math
import threading
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from torchcor_studio.runtime.tcs_studio import CHAMBERS, LEADS, VIZ_FORMAT

__all__ = [
    "CHAMBERS",
    "KINDS",
    "LEADS",
    "BundleError",
    "BundleNotFound",
    "ChamberMetrics",
    "Circulation",
    "Convergence",
    "ECGData",
    "EMData",
    "EPData",
    "FrameStack",
    "MechanicsData",
    "ResultBundle",
    "SummaryItem",
    "Surface",
    "Traces",
    "cardiac_summary",
    "format_value",
    "load_bundle",
    "write_ecg_csv",
]

#: Result kinds a bundle can declare, in the order the app presents them. The 12-lead
#: order (:data:`LEADS`) and the chambers (:data:`CHAMBERS`) come from the bundle writer.
KINDS = ("ep", "ecg", "mechanics", "em")

#: Datasets each kind cannot be shown without.
_REQUIRED = {
    "ep": ("surface", "ep"),
    "ecg": ("ecg",),
    "mechanics": ("surface", "mechanics"),
    "em": ("surface", "em"),
}
_KIND_NAMES = {"ep": "electrophysiology", "ecg": "ECG", "mechanics": "mechanics", "em": "electromechanics"}

T = TypeVar("T")


class BundleError(Exception):
    """A visualisation bundle is incomplete or corrupt. The message is user-facing."""


class BundleNotFound(BundleError):
    """There is no visualisation bundle in the folder (yet)."""


# =============================================================================== values
MINUS = "\u2212"


def format_value(value: Any) -> str:
    """Format a summary value compactly: ``1,234`` · ``98.6`` · ``0.999`` · ``−90`` · ``—``."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        text = f"{value:,}"
    elif isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        magnitude = abs(value)
        if magnitude >= 1000:
            text = f"{value:,.0f}"
        elif magnitude >= 100:
            text = f"{value:.0f}"
        elif magnitude >= 10:
            text = f"{value:.1f}".rstrip("0").rstrip(".")
        else:
            text = f"{value:.3g}"
    elif value is None:
        return "—"
    else:
        text = str(value)
        return {"yes": "Yes", "no": "No", "true": "Yes", "false": "No"}.get(text.strip().lower(), text)
    return "0" if text in ("-0", "0.0") else MINUS + text[1:] if text.startswith("-") else text


@dataclass(frozen=True)
class SummaryItem:
    """One headline number from ``index.json`` (shown as a stat tile)."""

    label: str
    value: float | int | str
    unit: str = ""
    hint: str = ""

    @property
    def text(self) -> str:
        return format_value(self.value)

    @classmethod
    def parse(cls, raw: Any) -> SummaryItem | None:
        if not isinstance(raw, dict) or not str(raw.get("label", "")).strip():
            return None
        value = raw.get("value")
        if value is None:
            value = "—"
        elif not isinstance(value, int | float | str):
            value = str(value)
        return cls(str(raw["label"]), value, str(raw.get("unit") or ""), str(raw.get("hint") or ""))


# =============================================================================== datasets
class FrameStack:
    """A ``(frames, points)`` scalar field kept in its compact on-disk dtype.

    Frames are converted to float32 on demand, so a 40M-value float16 field costs
    80 MB of memory instead of 160 MB.
    """

    def __init__(self, data: np.ndarray, name: str = "") -> None:
        if data.ndim != 2:
            raise BundleError(f"{name or 'field'} must be a (frames, points) array, got shape {data.shape}")
        if data.dtype.kind != "f" or data.dtype.itemsize > 4:
            data = data.astype(np.float32)
        self._data = data
        self.name = name
        self._range: tuple[float, float] | None = None
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._data.shape[0]

    @property
    def n_frames(self) -> int:
        return self._data.shape[0]

    @property
    def n_points(self) -> int:
        return self._data.shape[1]

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def frame(self, index: int, out: np.ndarray | None = None) -> np.ndarray:
        """Frame ``index`` as float32 (written into ``out`` when given)."""
        row = self._data[int(index)]
        if out is None:
            return row.astype(np.float32)
        np.copyto(out, row)
        return out

    def blend(self, position: float, out: np.ndarray | None = None) -> np.ndarray:
        """Linear interpolation between the frames around a fractional ``position``."""
        return _blend_frames(self._data, position, out)

    def column(self, point: int) -> np.ndarray:
        """The time course of one point, as float32 ``(frames,)``."""
        return self._data[:, int(point)].astype(np.float32)

    @property
    def range(self) -> tuple[float, float]:
        """Finite ``(min, max)`` over all frames (computed once, chunk by chunk)."""
        with self._lock:
            if self._range is None:
                lo, hi = math.inf, -math.inf
                step = max(1, 4_000_000 // max(1, self.n_points))
                for start in range(0, self.n_frames, step):
                    chunk = self._data[start:start + step]
                    finite = chunk[np.isfinite(chunk)]
                    if finite.size:
                        lo = min(lo, float(finite.min()))
                        hi = max(hi, float(finite.max()))
                self._range = (lo, hi) if lo <= hi else (0.0, 1.0)
            return self._range


@dataclass(frozen=True)
class Surface:
    """Renderable surface: points in mm, outward triangles, full-mesh node ids."""

    points: np.ndarray  # (N, 3) float32
    triangles: np.ndarray  # (M, 3) int32
    node_ids: np.ndarray  # (N,) int64
    regions: np.ndarray | None = None  # (M,) int32

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    def local_index(self, node_ids: Iterable[int]) -> np.ndarray:
        """Surface index of each full-mesh node id (``-1`` when it is not on the surface)."""
        ids = np.asarray(list(node_ids) if not isinstance(node_ids, np.ndarray) else node_ids,
                         dtype=np.int64).reshape(-1)
        order = np.argsort(self.node_ids, kind="stable")
        sorted_ids = self.node_ids[order]
        pos = np.clip(np.searchsorted(sorted_ids, ids), 0, max(0, len(sorted_ids) - 1))
        found = sorted_ids[pos] == ids if len(sorted_ids) else np.zeros(len(ids), bool)
        return np.where(found, order[pos], -1).astype(np.int64)


@dataclass(frozen=True)
class Traces:
    """Probe traces at full time resolution."""

    times: np.ndarray  # (T,) ms
    values: np.ndarray  # (K, T) mV
    labels: tuple[str, ...]
    node_ids: np.ndarray  # (K,) full-mesh ids

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class EPData:
    times: np.ndarray  # (F,) ms
    vm: FrameStack  # (F, N) mV
    lat: np.ndarray | None = None  # (N,) ms, NaN = never activated
    rt: np.ndarray | None = None
    apd: np.ndarray | None = None
    traces: Traces | None = None

    @property
    def n_frames(self) -> int:
        return len(self.times)


@dataclass(frozen=True)
class ECGData:
    t: np.ndarray  # (S,) ms
    leads: np.ndarray  # (L, S) mV
    lead_names: tuple[str, ...]

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def lead(self, name: str) -> np.ndarray | None:
        """A lead by name (case-insensitive), or ``None`` when the bundle lacks it."""
        key = name.lower()
        for i, lead in enumerate(self.lead_names):
            if lead.lower() == key:
                return self.leads[i]
        return None


@dataclass(frozen=True)
class Convergence:
    step: np.ndarray  # (S,) int32
    value: np.ndarray  # (S,) float32 — load factor or time
    newton: np.ndarray  # (S,) int32 Newton iterations


@dataclass(frozen=True)
class MechanicsData:
    frames: np.ndarray  # (F,)
    frame_label: str
    frame_unit: str
    displacement: np.ndarray  # (F, N, 3) mm
    fields: dict[str, FrameStack]
    max_displacement: np.ndarray  # (F,) peak |u| of every frame
    convergence: Convergence | None = None

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def magnitude(self, index: int) -> np.ndarray:
        """``|u|`` of frame ``index`` as float32 ``(N,)``."""
        return np.linalg.norm(self.displacement[int(index)], axis=1).astype(np.float32)

    def blend_displacement(self, position: float, out: np.ndarray | None = None) -> np.ndarray:
        """Displacement interpolated at a fractional frame ``position``."""
        return _blend_frames(self.displacement, position, out)


@dataclass(frozen=True)
class ChamberMetrics:
    """Volumes and pressure of one chamber over the recorded beat(s)."""

    edv: float  # mL
    esv: float  # mL
    peak_pressure: float  # mmHg

    @property
    def stroke_volume(self) -> float:
        return self.edv - self.esv

    @property
    def ejection_fraction(self) -> float:
        return 100.0 * self.stroke_volume / self.edv if self.edv > 0 else math.nan


@dataclass(frozen=True)
class Circulation:
    """Lumped circulation traces (pressure in mmHg, volume in mL) per chamber."""

    t: np.ndarray  # (S,) ms
    pressure: dict[str, np.ndarray]
    volume: dict[str, np.ndarray]

    @property
    def chambers(self) -> tuple[str, ...]:
        """Chambers with both a pressure and a volume trace, in :data:`CHAMBERS` order."""
        return tuple(c for c in CHAMBERS if c in self.pressure and c in self.volume)

    def metrics(self, chamber: str = "lv") -> ChamberMetrics | None:
        if chamber not in self.volume:
            return None
        v = self.volume[chamber]
        p = self.pressure.get(chamber)
        return ChamberMetrics(
            edv=float(np.nanmax(v)),
            esv=float(np.nanmin(v)),
            peak_pressure=float(np.nanmax(p)) if p is not None else math.nan,
        )


@dataclass(frozen=True)
class EMData:
    times: np.ndarray  # (F,) ms
    displacement: np.ndarray  # (F, N, 3) mm
    vm: FrameStack | None = None
    tension: FrameStack | None = None
    circulation: Circulation | None = None

    @property
    def n_frames(self) -> int:
        return len(self.times)

    def magnitude(self, index: int) -> np.ndarray:
        return np.linalg.norm(self.displacement[int(index)], axis=1).astype(np.float32)

    def blend_displacement(self, position: float, out: np.ndarray | None = None) -> np.ndarray:
        """Displacement interpolated at a fractional frame ``position``."""
        return _blend_frames(self.displacement, position, out)


def _blend_frames(stack: np.ndarray, position: float, out: np.ndarray | None) -> np.ndarray:
    """``stack[i] * (1 - w) + stack[i + 1] * w`` as float32 for a fractional frame position."""
    last = len(stack) - 1
    position = min(max(float(position), 0.0), float(last))
    i0 = int(position)
    w = np.float32(position - i0)
    if out is None:
        out = np.empty(stack.shape[1:], dtype=np.float32)
    if w < 1e-6 or i0 >= last:
        np.copyto(out, stack[i0])
        return out
    np.multiply(stack[i0], np.float32(1.0) - w, out=out)
    out += stack[i0 + 1] * w
    return out


# =============================================================================== bundle
@dataclass
class ResultBundle:
    """A validated ``viz/`` folder. Datasets load lazily (thread-safe) and are cached."""

    root: Path
    kind: str
    title: str
    created_at: float | None
    datasets: dict[str, str]
    summary: list[SummaryItem]
    notes: list[str]
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def has(self, name: str) -> bool:
        return name in self.datasets

    def loaded(self, name: str) -> bool:
        """Whether dataset ``name`` has been read already."""
        return name in self._cache

    # ------------------------------------------------------------------ datasets
    @property
    def surface(self) -> Surface:
        return self._get("surface", self._parse_surface)

    @property
    def ep(self) -> EPData:
        return self._get("ep", self._parse_ep)

    @property
    def ecg(self) -> ECGData:
        return self._get("ecg", self._parse_ecg)

    @property
    def mechanics(self) -> MechanicsData:
        return self._get("mechanics", self._parse_mechanics)

    @property
    def em(self) -> EMData:
        return self._get("em", self._parse_em)

    def preload(self, *names: str) -> None:
        """Load (and validate) datasets now — meant for worker threads.

        Without ``names``, loads what a view of this bundle's kind needs, and computes the
        value ranges it will ask for.
        """
        for name in names or _REQUIRED[self.kind]:
            data = getattr(self, name)
            for stack in _frame_stacks(data):
                stack.range  # noqa: B018 - warm the cached range

    def _get(self, name: str, parse: Callable[[dict[str, np.ndarray]], T]) -> T:
        with self._lock:
            if name not in self._cache:
                self._cache[name] = parse(self._read(name))
            return self._cache[name]

    def _read(self, name: str) -> dict[str, np.ndarray]:
        file = self.datasets.get(name)
        if file is None:
            raise BundleError(f"The bundle has no “{name}” dataset.")
        path = self.root / file
        if not path.is_file():
            raise BundleError(f"{file} is missing from the results folder.")
        try:
            with np.load(path, allow_pickle=False) as npz:
                return {key: npz[key] for key in npz.files}
        except (zipfile.BadZipFile, ValueError, OSError, EOFError, KeyError) as exc:
            raise BundleError(f"{file} is damaged or incomplete ({exc}).") from exc

    # ------------------------------------------------------------------ parsers
    def _parse_surface(self, a: dict[str, np.ndarray]) -> Surface:
        f = self.datasets["surface"]
        points = _array(a, "points", f, np.float32)
        triangles = _array(a, "triangles", f, np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise BundleError(f"{f}: points must be a non-empty (N, 3) array, got {points.shape}.")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
            raise BundleError(f"{f}: triangles must be a non-empty (M, 3) array, got {triangles.shape}.")
        if triangles.min() < 0 or triangles.max() >= len(points):
            raise BundleError(f"{f}: triangles reference points outside 0…{len(points) - 1}.")
        if not np.isfinite(points).all():
            raise BundleError(f"{f}: points contain NaN or infinite coordinates.")
        node_ids = a.get("node_ids")
        node_ids = np.arange(len(points), dtype=np.int64) if node_ids is None else \
            np.ascontiguousarray(node_ids, dtype=np.int64).reshape(-1)
        _expect(len(node_ids) == len(points), f"{f}: node_ids has {len(node_ids)} entries for {len(points)} points.")
        regions = a.get("regions")
        if regions is not None:
            regions = np.ascontiguousarray(regions, dtype=np.int32).reshape(-1)
            if len(regions) != len(triangles):
                regions = None
        return Surface(np.ascontiguousarray(points), np.ascontiguousarray(triangles, dtype=np.int32),
                       node_ids, regions)

    def _parse_ep(self, a: dict[str, np.ndarray]) -> EPData:
        f = self.datasets["ep"]
        n = self.surface.n_points
        times = _vector(a, "times", f)
        vm = FrameStack(_array(a, "vm", f), "vm")
        _expect(vm.n_frames == len(times), f"{f}: vm has {vm.n_frames} frames but times has {len(times)}.")
        _expect(vm.n_points == n, f"{f}: vm has {vm.n_points} values per frame but the surface has {n} points.")
        point_fields = {}
        for key in ("lat", "rt", "apd"):
            if key in a:
                values = np.ascontiguousarray(a[key], dtype=np.float32).reshape(-1)
                _expect(len(values) == n, f"{f}: {key} has {len(values)} values but the surface has {n} points.")
                point_fields[key] = values
        return EPData(times=times, vm=vm, traces=_parse_traces(a, f), **point_fields)

    def _parse_ecg(self, a: dict[str, np.ndarray]) -> ECGData:
        f = self.datasets["ecg"]
        t = _vector(a, "t", f)
        leads = _array(a, "leads", f, np.float32)
        if leads.ndim == 1:
            leads = leads[None, :]
        _expect(leads.ndim == 2 and leads.shape[1] == len(t),
                f"{f}: leads must be (leads, {len(t)}) to match t, got {leads.shape}.")
        names = a.get("lead_names")
        if names is None:
            names = LEADS[: len(leads)] if len(leads) <= len(LEADS) else \
                tuple(f"Lead {i + 1}" for i in range(len(leads)))
        names = tuple(str(x) for x in np.asarray(names).reshape(-1))
        _expect(len(names) == len(leads), f"{f}: {len(names)} lead names for {len(leads)} leads.")
        _expect(len(t) >= 2, f"{f}: the ECG needs at least two samples.")
        return ECGData(t=t, leads=np.ascontiguousarray(leads), lead_names=names)

    def _parse_mechanics(self, a: dict[str, np.ndarray]) -> MechanicsData:
        f = self.datasets["mechanics"]
        n = self.surface.n_points
        frames = _vector(a, "frames", f)
        disp = _displacement(a, f, len(frames), n)
        fields = {}
        for key in sorted(a):
            if key.startswith("field_"):
                stack = FrameStack(a[key], key)
                _expect(stack.n_frames == len(frames) and stack.n_points == n,
                        f"{f}: {key} must be ({len(frames)}, {n}), got ({stack.n_frames}, {stack.n_points}).")
                fields[key.removeprefix("field_")] = stack
        convergence = None
        if all(k in a for k in ("conv_step", "conv_value", "conv_newton")):
            step, value, newton = (np.asarray(a[k]).reshape(-1) for k in ("conv_step", "conv_value", "conv_newton"))
            _expect(len(step) == len(value) == len(newton), f"{f}: convergence arrays differ in length.")
            convergence = Convergence(step.astype(np.int32), value.astype(np.float32), newton.astype(np.int32))
        return MechanicsData(
            frames=frames,
            frame_label=_text(a, "frame_label", "Step"),
            frame_unit=_text(a, "frame_unit", ""),
            displacement=disp,
            fields=fields,
            max_displacement=np.array([np.linalg.norm(u, axis=1).max(initial=0.0) for u in disp],
                                      dtype=np.float32),
            convergence=convergence,
        )

    def _parse_em(self, a: dict[str, np.ndarray]) -> EMData:
        f = self.datasets["em"]
        n = self.surface.n_points
        times = _vector(a, "times", f)
        disp = _displacement(a, f, len(times), n)
        stacks = {}
        for key in ("vm", "tension"):
            if key in a:
                stack = FrameStack(a[key], key)
                _expect(stack.n_frames == len(times) and stack.n_points == n,
                        f"{f}: {key} must be ({len(times)}, {n}), got ({stack.n_frames}, {stack.n_points}).")
                stacks[key] = stack
        return EMData(times=times, displacement=disp, circulation=_parse_circulation(a, f), **stacks)


def _frame_stacks(data: Any) -> list[FrameStack]:
    if isinstance(data, EPData):
        return [data.vm]
    if isinstance(data, MechanicsData):
        return list(data.fields.values())
    if isinstance(data, EMData):
        return [s for s in (data.vm, data.tension) if s is not None]
    return []


# =============================================================================== validation
def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def _array(a: dict[str, np.ndarray], key: str, file: str, dtype=None) -> np.ndarray:
    if key not in a:
        raise BundleError(f"{file} is missing the “{key}” array.")
    value = a[key]
    if value.dtype.kind not in "biuf":
        raise BundleError(f"{file}: “{key}” is not numeric (dtype {value.dtype}).")
    return np.ascontiguousarray(value, dtype=dtype) if dtype is not None else value


def _vector(a: dict[str, np.ndarray], key: str, file: str) -> np.ndarray:
    value = _array(a, key, file, np.float32).reshape(-1)
    _expect(len(value) > 0, f"{file}: “{key}” is empty.")
    return value


def _text(a: dict[str, np.ndarray], key: str, default: str) -> str:
    if key not in a:
        return default
    value = np.asarray(a[key])
    return str(value.reshape(-1)[0]) if value.size else default


def _displacement(a: dict[str, np.ndarray], file: str, frames: int, points: int) -> np.ndarray:
    disp = _array(a, "displacement", file, np.float32)
    _expect(disp.shape == (frames, points, 3),
            f"{file}: displacement must be ({frames}, {points}, 3), got {disp.shape}.")
    return disp


def _parse_traces(a: dict[str, np.ndarray], file: str) -> Traces | None:
    if "traces" not in a or "trace_times" not in a:
        return None
    times = _vector(a, "trace_times", file)
    values = _array(a, "traces", file, np.float32)
    if values.ndim == 1:
        values = values[None, :]
    _expect(values.ndim == 2 and values.shape[1] == len(times),
            f"{file}: traces must be (probes, {len(times)}), got {values.shape}.")
    k = len(values)
    labels = a.get("trace_labels")
    labels = tuple(str(x) for x in np.asarray(labels).reshape(-1)) if labels is not None else \
        tuple(f"Probe {i + 1}" for i in range(k))
    node_ids = a.get("trace_node_ids")
    node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1) if node_ids is not None else \
        np.full(k, -1, dtype=np.int64)
    _expect(len(labels) == k and len(node_ids) == k,
            f"{file}: {k} traces but {len(labels)} labels and {len(node_ids)} node ids.")
    return Traces(times, values, labels, node_ids)


def _parse_circulation(a: dict[str, np.ndarray], file: str) -> Circulation | None:
    if "pv_t" not in a:
        return None
    t = _vector(a, "pv_t", file)
    pressure, volume = {}, {}
    for chamber in CHAMBERS:
        for key, target in ((f"p_{chamber}", pressure), (f"V_{chamber}", volume)):
            if key in a:
                values = np.asarray(a[key], dtype=np.float32).reshape(-1)
                _expect(len(values) == len(t), f"{file}: {key} has {len(values)} samples but pv_t has {len(t)}.")
                target[chamber] = values
    if not pressure and not volume:
        return None
    return Circulation(t, pressure, volume)


# =============================================================================== loading
def load_bundle(viz_dir: Path | str) -> ResultBundle:
    """Read and validate ``index.json`` of a bundle (datasets load on first access).

    ``viz_dir`` is the ``viz/`` folder itself or the run folder that contains it.
    Raises :class:`BundleNotFound` when there is no bundle and :class:`BundleError` when
    it cannot be used; both carry a message suitable for the UI.
    """
    root = Path(viz_dir)
    if not (root / "index.json").is_file() and (root / "viz" / "index.json").is_file():
        root = root / "viz"
    index_path = root / "index.json"
    if not index_path.is_file():
        raise BundleNotFound(f"No visualisation bundle was found in {root}.")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"index.json is not valid JSON (line {exc.lineno}: {exc.msg}).") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise BundleError(f"index.json could not be read ({exc}).") from exc
    if not isinstance(index, dict):
        raise BundleError("index.json must contain a JSON object.")

    fmt = index.get("format", VIZ_FORMAT)
    if not isinstance(fmt, int) or fmt < 1:
        raise BundleError(f"index.json declares an invalid format ({fmt!r}).")
    if fmt > VIZ_FORMAT:
        raise BundleError(f"These results use bundle format {fmt}; this version of TorchCor Lighting "
                          f"reads format {VIZ_FORMAT}. Please update the app.")
    kind = index.get("kind")
    if kind not in KINDS:
        raise BundleError(f"Unknown result kind {kind!r} (expected one of {', '.join(KINDS)}).")

    datasets = index.get("datasets") or {}
    if not isinstance(datasets, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                 for k, v in datasets.items()):
        raise BundleError("index.json: “datasets” must map dataset names to file names.")
    for name, file in datasets.items():
        if Path(file).name != file or file in ("", ".", ".."):
            raise BundleError(f"index.json: dataset “{name}” points outside the bundle ({file!r}).")
    for name in _REQUIRED[kind]:
        if name not in datasets:
            raise BundleError(f"These {_KIND_NAMES[kind]} results have no “{name}” dataset.")
        if not (root / datasets[name]).is_file():
            raise BundleError(f"{datasets[name]} is missing from the results folder.")

    summary = [item for item in (SummaryItem.parse(raw) for raw in _list(index.get("summary"))) if item]
    notes = [str(note) for note in _list(index.get("notes")) if str(note).strip()]
    created = index.get("created_at")
    return ResultBundle(
        root=root,
        kind=kind,
        title=str(index.get("title") or "").strip(),
        created_at=float(created) if isinstance(created, int | float) else None,
        datasets=dict(datasets),
        summary=summary,
        notes=notes,
    )


def _list(value: Any) -> list:
    return list(value) if isinstance(value, list | tuple) else []


# =============================================================================== metrics
#: Headline haemodynamics: (label, unit, labels a bundle may use for it, value from LV metrics).
_CARDIAC_METRICS: tuple[tuple[str, str, tuple[str, ...], Callable[[ChamberMetrics], float]], ...] = (
    ("EDV", "mL", ("edv", "lv edv", "end-diastolic volume", "lv end-diastolic volume"), lambda m: m.edv),
    ("ESV", "mL", ("esv", "lv esv", "end-systolic volume", "lv end-systolic volume"), lambda m: m.esv),
    ("Stroke volume", "mL", ("sv", "lv sv", "stroke volume", "lv stroke volume"), lambda m: m.stroke_volume),
    ("Ejection fraction", "%", ("ef", "lv ef", "ejection fraction", "lv ejection fraction"),
     lambda m: m.ejection_fraction),
    ("Peak LV pressure", "mmHg", ("peak lv pressure", "peak pressure", "max lv pressure"),
     lambda m: m.peak_pressure),
)


def cardiac_summary(summary: Iterable[SummaryItem], circulation: Circulation | None) -> list[SummaryItem]:
    """EDV, ESV, SV, EF and peak LV pressure first, then the rest of ``summary``.

    Values the bundle reports are used as they are; missing ones are computed from the
    LV pressure–volume traces (when there are any).
    """
    items = list(summary)
    lv = circulation.metrics("lv") if circulation is not None else None
    # A run shorter than one cycle reports an "LV volume" range instead of EDV/ESV/EF;
    # recomputing those from a partial trace would be misleading.
    partial = any(item.label.strip().lower() == "lv volume" for item in items)
    used: set[int] = set()
    head: list[SummaryItem] = []
    for label, unit, aliases, compute in _CARDIAC_METRICS:
        match = next((item for item in items if item.label.strip().lower() in aliases), None)
        if match is not None:
            used.add(id(match))
            head.append(match)
        elif partial and label != "Peak LV pressure":
            continue
        elif lv is not None and math.isfinite(value := compute(lv)):
            head.append(SummaryItem(label, round(value, 1), unit, "Computed from the LV pressure–volume trace"))
    return head + [item for item in items if id(item) not in used]


# =============================================================================== export
def write_ecg_csv(path: Path | str, ecg: ECGData) -> Path:
    """Write the ECG as CSV: ``t_ms`` followed by one column per lead (mV)."""
    path = Path(path)
    table = np.column_stack([ecg.t, ecg.leads.T])
    header = ",".join(["t_ms", *(f"{name}_mV" for name in ecg.lead_names)])
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.6g")
    return path
