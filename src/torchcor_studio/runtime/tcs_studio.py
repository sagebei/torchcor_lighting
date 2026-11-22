"""TorchCor Lighting helpers for generated ``run.py`` scripts.

Copied next to every generated script, so a run folder is self-contained: ``python run.py``
reproduces the run anywhere TorchCor is installed. Importing it needs the standard library
only; numpy is imported on first use and torch only by the functions that receive tensors.
Four jobs:

1. **Progress markers** — :func:`stage`, :func:`progress`, :func:`total`, :func:`metric`
   print ``::tcs:: {json}`` lines that the supervisor (``tcs_runtime.py``) turns into the
   live status shown in the app. They are harmless when the script runs on its own.
2. **Inputs** — :func:`make_slab_mesh` writes the built-in benchmark slab and
   :func:`prepare_kcl_torso` downloads and prepares the KCL heart-torso model.
3. **Recording** — :class:`FrameRecorder` keeps evenly spaced surface frames during a
   solve and :class:`SolverLog` keeps TorchCor's mechanics step lines.
4. **Visualisation bundle** — small, float16-packed arrays in ``viz/`` that the app
   downloads instead of the full results (:func:`export_ep`, :func:`export_ecg`,
   :func:`export_mechanics`, :func:`export_em`).

Visualisation bundle format (``viz/``, format 1)
-----------------------------------------------
``index.json``::

    {"format": 1, "kind": "ep|ecg|mechanics|em", "title": str, "created_at": float,
     "datasets": {"surface": "surface.npz", "<name>": "<file>.npz", ...},
     "summary": [{"label": str, "value": float|str, "unit": str, "hint": str}, ...],
     "notes": [str, ...]}

``surface.npz`` (every kind that has a geometry)
    ``points`` (N, 3) float32 mm · ``triangles`` (M, 3) int32 · ``node_ids`` (N,) int64 —
    index of each surface point in the full mesh · optional ``regions`` (M,) int32.

``ep.npz``
    ``times`` (F,) float32 ms · ``vm`` (F, N) float16 mV on surface points ·
    ``lat`` (N,) float32 ms (NaN = never activated) · optional ``rt`` (N,), ``apd`` (N,) ·
    ``trace_times`` (Ft,) float32 · ``traces`` (K, Ft) float32 mV full time resolution ·
    ``trace_labels`` (K,) str · ``trace_node_ids`` (K,) int64 (full-mesh ids).

``ecg.npz``
    ``t`` (S,) float32 ms · ``leads`` (12, S) float32 mV · ``lead_names`` (12,) str in the
    order I, II, III, aVR, aVL, aVF, V1 … V6. An ECG bundle may also carry the heart's
    ``surface`` and ``ep`` datasets (as above), so the activation can be shown beside it.

``mechanics.npz``
    ``frames`` (F,) float32 · ``frame_label`` str ("Load factor λ" or "Time") ·
    ``frame_unit`` str ("", "ms", "s") · ``displacement`` (F, N, 3) float32 mm on surface
    points · optional ``field_<name>`` (F, N) float16 point scalars ·
    convergence: ``conv_step`` (S,) int32, ``conv_value`` (S,) float32 (λ or t),
    ``conv_newton`` (S,) int32.

``em.npz``
    ``times`` (F,) float32 ms · ``displacement`` (F, N, 3) float32 mm · optional
    ``vm`` (F, N) float16 mV · ``tension`` (F, N) float16 kPa ·
    circulation: ``pv_t`` (S,) float32 ms, ``p_lv p_rv p_la p_ra`` (S,) mmHg,
    ``V_lv V_rv V_la V_ra`` (S,) mL (absent chambers are omitted), optional ``pv_cycle``
    () float32 ms: the cycle length; beat k spans ``k * pv_cycle`` to ``(k + 1) * pv_cycle``.

Strings are stored as numpy unicode arrays, so every dataset loads with
``numpy.load(path, allow_pickle=False)``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

MARKER = "::tcs::"
VIZ_FORMAT = 1

#: Upper bound on the size of a per-frame field in the bundle (float16 values).
MAX_FIELD_VALUES = 40_000_000

#: Standard 12-lead order of ``ecg.npz``.
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

#: Cavities of the four-chamber electromechanics bundle, in circulation order.
CHAMBERS = ("lv", "rv", "la", "ra")


# =============================================================================== markers
def _mark(payload: dict) -> None:
    payload.setdefault("time", time.time())
    sys.stdout.write(f"{MARKER} {json.dumps(payload, default=_jsonable)}\n")
    sys.stdout.flush()


def _jsonable(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def stage(stage_id: str, label: str | None = None, *, total: float | None = None,
          unit: str = "ms", auto: bool = True) -> None:
    """Announce the start of a pipeline stage (``setup``, ``mesh``, ``solve``, ...).

    ``auto=False`` tells the supervisor not to derive progress from TorchCor's own output
    in this stage (for stages that repeat inner solves, such as unloading iterations); the
    stage then reports only what :func:`progress` sends.
    """
    payload = {"type": "stage", "id": stage_id, "label": label or stage_id.title()}
    if total is not None:
        payload.update(total=float(total), unit=unit)
    if not auto:
        payload["auto"] = False
    _mark(payload)
    print(f"── {label or stage_id}", flush=True)


def progress(value: float, message: str | None = None) -> None:
    """Report the fraction (0..1) of the current stage that is done."""
    _mark({"type": "progress", "value": max(0.0, min(1.0, float(value))), "message": message})


def total(value: float, unit: str = "ms") -> None:
    """Declare the simulated end time, so TorchCor's own ``t = ...`` lines become progress."""
    _mark({"type": "total", "value": float(value), "unit": unit})


def metric(name: str, value, unit: str = "") -> None:
    _mark({"type": "metric", "name": name, "value": value, "unit": unit})


def artifact(path, kind: str = "file", label: str | None = None) -> None:
    _mark({"type": "artifact", "path": str(path), "kind": kind, "label": label})


def run_dir() -> Path:
    return Path(os.environ.get("TCS_RUN_DIR") or Path(sys.argv[0]).resolve().parent)


# =============================================================================== numpy/torch
def _np():
    import numpy as np

    return np


def to_numpy(x):
    """torch tensor / array-like -> numpy array on the host."""
    np = _np()
    if hasattr(x, "detach"):
        x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        return x.numpy()
    return np.asarray(x)


def _rows(x, rows):
    """``x[rows]`` (rows along the first axis) as numpy, indexing on x's own device."""
    if hasattr(x, "detach"):
        import torch

        x = x.detach()
        index = torch.as_tensor(_np().asarray(rows), dtype=torch.long, device=x.device)
        return to_numpy(x.index_select(0, index))
    return _np().asarray(x)[rows]


def _columns(x, columns, stride: int = 1):
    """``x[::stride, columns]`` as numpy, indexing on x's own device first."""
    if hasattr(x, "detach"):
        import torch

        x = x.detach()[::stride]
        index = torch.as_tensor(_np().asarray(columns), dtype=torch.long, device=x.device)
        return to_numpy(x.index_select(1, index))
    return _np().asarray(x)[::stride][:, columns]


# =============================================================================== geometry
class Surface(NamedTuple):
    """A renderable triangle surface of a mesh (see ``surface.npz``)."""

    points: object  # (N, 3) float32 mm
    triangles: object  # (M, 3) int32, local numbering
    node_ids: object  # (N,) int64, full-mesh index of each point
    regions: object = None  # (M,) int32 or None


def _face_keys(faces, n_nodes):
    """Unique int64 key per (sorted) triangle; falls back to row hashing for huge meshes."""
    np = _np()
    if n_nodes >= (1 << 21):  # 3 x 21 bits no longer fit in an int64
        return None
    s = np.sort(faces, axis=1).astype(np.int64)
    return (s[:, 0] << 42) | (s[:, 1] << 21) | s[:, 2]


def boundary_of_tets(points, tets, regions=None):
    """Outward-oriented boundary triangles of a tetrahedral mesh.

    Returns ``(triangles, triangle_regions)`` in full-mesh node numbering.
    """
    np = _np()
    tets = np.asarray(tets, dtype=np.int64)
    local = np.array([[1, 2, 3, 0], [0, 3, 2, 1], [0, 1, 3, 2], [0, 2, 1, 3]])
    faces = tets[:, local[:, :3]].reshape(-1, 3)
    opposite = tets[:, local[:, 3]].reshape(-1)
    owner = np.repeat(np.arange(len(tets)), 4)
    keys = _face_keys(faces, len(points))
    if keys is None:
        _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True,
                                       return_counts=True)
        once = counts[inverse.reshape(-1)] == 1
    else:
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        once = counts[inverse] == 1
    tri, opp, own = faces[once], opposite[once], owner[once]
    p = np.asarray(points, dtype=np.float64)
    normal = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    inward = np.einsum("ij,ij->i", normal, p[opp] - p[tri[:, 0]]) > 0
    tri[inward] = tri[inward][:, [0, 2, 1]]
    reg = None if regions is None else np.asarray(regions)[own]
    return tri, reg


#: The six faces of a VTK hexahedron (corners 0-3 bottom, 4-7 top).
_HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def boundary_of_hexes(points, hexes):
    """Outward-oriented boundary triangles of an 8-node hexahedral mesh (VTK node order).

    Hexahedra may have collapsed edges (e.g. at the apex of an ellipsoidal ventricle):
    faces are matched by their sorted node sets and degenerate triangles are dropped.
    """
    np = _np()
    hexes = np.asarray(hexes, dtype=np.int64)
    quads = hexes[:, np.array(_HEX_FACES)].reshape(-1, 4)
    owner = np.repeat(np.arange(len(hexes)), len(_HEX_FACES))
    _, inverse, counts = np.unique(np.sort(quads, axis=1), axis=0, return_inverse=True,
                                   return_counts=True)
    once = counts[inverse.reshape(-1)] == 1
    quads, owner = quads[once], owner[once]
    tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    own = np.concatenate([owner, owner])
    proper = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri, own = tri[proper], own[proper]
    p = np.asarray(points, dtype=np.float64)
    centre = p[hexes].mean(axis=1)[own]
    normal = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    inward = np.einsum("ij,ij->i", normal, centre - p[tri[:, 0]]) > 0
    tri[inward] = tri[inward][:, [0, 2, 1]]
    return tri


def compact_surface(points, triangles, triangle_regions=None):
    """Keep only the nodes the triangles use. Returns (points, triangles, node_ids, regions)."""
    np = _np()
    triangles = np.asarray(triangles, dtype=np.int64)
    node_ids = np.unique(triangles)
    local = np.searchsorted(node_ids, triangles).astype(np.int32)
    pts = np.asarray(points, dtype=np.float32)[node_ids]
    reg = None if triangle_regions is None else np.asarray(triangle_regions, dtype=np.int32)
    return Surface(pts, local, node_ids.astype(np.int64), reg)


def surface_of(points, tets=None, triangles=None, tet_regions=None, tri_regions=None):
    """Renderable surface of a mesh with tetrahedra and/or surface triangles."""
    np = _np()
    points = to_numpy(points)
    if tets is not None and len(tets):
        tri, reg = boundary_of_tets(points, to_numpy(tets),
                                    None if tet_regions is None else to_numpy(tet_regions))
    elif triangles is not None and len(triangles):
        tri = to_numpy(triangles).astype(np.int64)
        reg = None if tri_regions is None else to_numpy(tri_regions)
    else:
        raise ValueError("the mesh has neither tetrahedra nor triangles to draw")
    return compact_surface(points, tri, reg)


def simulator_surface(simulator):
    """Surface of a TorchCor EP simulator's mesh (``Monodomain``, ``ReactionEikonal``)."""
    elems = simulator.elems
    tets, tris = elems.Tt, elems.Tr
    return surface_of(simulator.nodes,
                      tets=tets.data if tets.data is not None else None,
                      triangles=tris.data if tris.data is not None else None,
                      tet_regions=tets.region if tets.data is not None else None,
                      tri_regions=tris.region if tris.data is not None else None)


def mechanics_surface(mesh):
    """Surface of a ``torchcor.mechanics`` mesh (hexahedral or tetrahedral, any order).

    Higher-order cells are split into linear ones first, so every surface node of a
    quadratic mesh is a vertex of the drawn surface.
    """
    points = to_numpy(mesh.points)
    cells = mesh.linear_cells()
    if cells.shape[1] == 8:
        return compact_surface(points, boundary_of_hexes(points, cells))
    return surface_of(points, tets=cells)


def frame_stride(n_frames: int, n_points: int, components: int = 1,
                 budget: int = MAX_FIELD_VALUES) -> int:
    """Stride that keeps ``frames x points x components`` under the bundle budget."""
    per_frame = max(1, n_points * components)
    allowed = max(2, budget // per_frame)
    return max(1, -(-n_frames // allowed))


def pick_trace_nodes(points, lat=None, count: int = 6):
    """Indices into ``points`` of ~``count`` probe nodes, and a label for each.

    The earliest, median and latest activated nodes first (when activation times are
    given), then nodes spread over the surface by farthest-point sampling.
    """
    np = _np()
    points = np.asarray(points, dtype=np.float64)
    chosen: list[int] = []
    labels: list[str] = []
    if lat is not None:
        lat = np.asarray(lat, dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(lat))
        if finite.size:
            order = finite[np.argsort(lat[finite], kind="stable")]
            for label, index in (("Earliest activation", order[0]),
                                 ("Median activation", order[len(order) // 2]),
                                 ("Latest activation", order[-1])):
                if int(index) not in chosen:
                    chosen.append(int(index))
                    labels.append(label)
    if not chosen and len(points):
        centre = points.mean(axis=0)
        chosen.append(int(np.argmin(np.linalg.norm(points - centre, axis=1))))
        labels.append("Probe 1")
    if not chosen:
        return np.zeros(0, dtype=np.int64), labels
    distance = np.min([np.linalg.norm(points - points[i], axis=1) for i in chosen], axis=0)
    while len(chosen) < min(count, len(points)):
        index = int(np.argmax(distance))
        if distance[index] <= 0.0:
            break
        chosen.append(index)
        labels.append(f"Probe {len(chosen)}")
        distance = np.minimum(distance, np.linalg.norm(points - points[index], axis=1))
    return np.asarray(chosen, dtype=np.int64), labels


# =============================================================================== inputs
def make_slab_mesh(out_dir, lengths=(20.0, 7.0, 3.0), dx: float = 0.5, corner: float = 1.5,
                   name: str = "slab") -> Path:
    """Write a structured tetrahedral slab as a .pts/.elem/.lon mesh and return its folder.

    The defaults are the N-version benchmark of Niederer et al. (2011): a 20 x 7 x 3 mm
    block with fibres along x. Writes ``<name>.pts`` (µm), ``<name>.elem`` (``Tt a b c d 1``,
    six tetrahedra per cube), ``<name>.lon`` (one fibre direction per element) and two
    stimulus sites: ``stimulus.vtx``, the nodes in the cube of side ``corner`` at the
    origin, and ``left_face.vtx``, every node on x = 0 (a planar wave). TorchCor reads
    whichever mesh it finds in a folder, so keep the folder for the slab alone.
    """
    np = _np()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lengths = tuple(float(v) for v in lengths)
    nx, ny, nz = (max(2, round(length / dx) + 1) for length in lengths)
    axes = [np.linspace(0.0, length, n) for length, n in zip(lengths, (nx, ny, nz), strict=True)]
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.stack([g.ravel() for g in grid], axis=1)  # mm, node (i, j, k) at (i*ny + j)*nz + k

    ids = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    cube = np.stack([ids[:-1, :-1, :-1], ids[1:, :-1, :-1], ids[1:, 1:, :-1], ids[:-1, 1:, :-1],
                     ids[:-1, :-1, 1:], ids[1:, :-1, 1:], ids[1:, 1:, 1:], ids[:-1, 1:, 1:]],
                    axis=-1).reshape(-1, 8)
    # Kuhn split of each cube along its main diagonal (0-6): six positively oriented tets.
    tets = np.concatenate([cube[:, [0, 1, 2, 6]], cube[:, [0, 2, 3, 6]], cube[:, [0, 3, 7, 6]],
                           cube[:, [0, 7, 4, 6]], cube[:, [0, 4, 5, 6]], cube[:, [0, 5, 1, 6]]])

    with open(out / f"{name}.pts", "w", encoding="utf-8") as fh:
        fh.write(f"{len(points)}\n")
        np.savetxt(fh, points * 1000.0, fmt="%.3f")
    with open(out / f"{name}.elem", "w", encoding="utf-8") as fh:
        fh.write(f"{len(tets)}\n")
        np.savetxt(fh, tets, fmt="Tt %d %d %d %d 1")
    with open(out / f"{name}.lon", "w", encoding="utf-8") as fh:
        fh.write("1\n")
        fh.write("1 0 0\n" * len(tets))
    write_vtx(out / "stimulus.vtx", np.flatnonzero(np.all(points <= corner + 1e-9, axis=1)))
    write_vtx(out / "left_face.vtx", np.flatnonzero(points[:, 0] <= 1e-9))
    print(f"slab mesh: {len(points)} nodes, {len(tets)} tetrahedra in {out}", flush=True)
    return out


def write_vtx(path, node_ids, domain: str = "extra") -> Path:
    """Write a ``.vtx`` node list (count, domain line, one id per line)."""
    np = _np()
    ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(ids)}\n{domain}\n")
        np.savetxt(fh, ids, fmt="%d")
    return path


# ------------------------------------------------------------------------ KCL torso
# Ported from TorchCor's demo/ecg/demo_download.py (MIT licence, same author).
# Data: KCL whole-torso computational models (Qian, Bishop et al., King's College
# London; Zenodo 10.5281/zenodo.7253863, CC-BY-4.0). Torso 3 is used: of the five it
# gives the most typical healthy 12-lead ECG with the default settings.
KCL_TORSO = "KCL_torso3"
KCL_URL = f"https://zenodo.org/records/7253863/files/{KCL_TORSO}.zip?download=1"
KCL_PACING_SITES = ("LV_septal", "LV_anterior", "RV_septal", "RV_moderator")
_RV_WALL, _LV_WALL, _FAST_ENDO = 34, 35, 36  # 36 is created here (heart mesh only)
_RV_BLOOD, _LV_BLOOD, _BONE = 24, 25, 3
_BASE_TAGS = list(range(9, 24))  # great vessels, valves, atria


def kcl_torso_ready(data_dir) -> bool:
    """Is ``data_dir`` a prepared KCL torso folder (see :func:`prepare_kcl_torso`)?"""
    data = Path(data_dir)
    needed = [data / KCL_TORSO / f"{KCL_TORSO}.elem", data / "heart" / "heart.elem",
              data / "electrodes" / "lf_src.vtx", data / "fields.npz",
              *(data / "pacing" / f"{site}.vtx" for site in KCL_PACING_SITES)]
    return all(path.exists() for path in needed)


def prepare_kcl_torso(data_dir, url: str = KCL_URL) -> Path:
    """Download and prepare the KCL heart-torso model 3 in ``data_dir``; returns it.

    Nothing is done when the folder is already prepared. Otherwise:

    1. download and unzip ``KCL_torso3`` (a .pts/.elem/.lon mesh in µm, fibres in the ventricles);
    2. cut the ventricles out of the torso mesh, keeping the torso node ordering that
       ``LeadField`` requires                                  -> ``heart/``
    3. classify the heart surfaces (LV/RV endocardium, epicardium, base) and compute
       apico-basal and transmural coordinates                  -> ``fields.npz``
    4. tag a fast-conducting sub-endocardial layer (tag 36, a stand-in for the Purkinje
       network) in the heart mesh;
    5. four Durrer-type early-activation sites                 -> ``pacing/*.vtx``
    6. place the 10 ECG electrodes (V1-V6 + Mason-Likar limb electrodes) on the skin
       from anatomical landmarks                               -> ``electrodes/lf_src.vtx``
    """
    data = Path(data_dir)
    if kcl_torso_ready(data):
        print(f"KCL torso model found in {data}", flush=True)
        return data
    data.mkdir(parents=True, exist_ok=True)
    if not (data / KCL_TORSO / f"{KCL_TORSO}.elem").exists():
        _download_kcl_torso(data, url)
    _preprocess_kcl_torso(data)
    return data


def _download_kcl_torso(data: Path, url: str, attempts: int = 5) -> None:
    import urllib.request
    import zipfile

    archive = data / f"{KCL_TORSO}.zip"
    part = archive.with_suffix(".part")
    for attempt in range(1, attempts + 1):
        if archive.exists() and zipfile.is_zipfile(archive):
            break
        print(f"Downloading {KCL_TORSO} from Zenodo (attempt {attempt}) ...", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as fh:
                size = int(response.headers.get("Content-Length", 0))
                done = 0
                while chunk := response.read(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if size:
                        progress(done / size, f"downloading {done / 2**20:.0f} / {size / 2**20:.0f} MB")
            if size and part.stat().st_size != size:  # the connection dropped
                raise OSError(f"got {part.stat().st_size} of {size} bytes")
            part.replace(archive)
        except Exception as err:  # noqa: BLE001 - retried, then reported below
            print(f"  download failed: {err}", flush=True)
            time.sleep(10)
    if not (archive.exists() and zipfile.is_zipfile(archive)):
        raise RuntimeError(f"Could not download {url}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data)
    archive.unlink()


def _read_mesh(base):
    import numpy as np
    import pandas as pd

    pts = pd.read_csv(f"{base}.pts", sep=r"\s+", skiprows=1, header=None,
                      dtype=np.float64).to_numpy()
    el = pd.read_csv(f"{base}.elem", sep=r"\s+", skiprows=1, header=None,
                     usecols=[1, 2, 3, 4, 5]).to_numpy()
    lon = pd.read_csv(f"{base}.lon", sep=r"\s+", skiprows=1, header=None,
                      dtype=np.float64).to_numpy()[:, :3]
    return pts, el[:, :4].astype(np.int64), el[:, 4].astype(np.int64), lon


def _face_adjacency(tets):
    """All tet faces with owner element and the pairs of coincident faces."""
    np = _np()
    faces = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]],
                            tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]])
    opposite = tets.T.reshape(-1)  # node opposite each face
    owner = np.tile(np.arange(len(tets)), 4)
    s = np.sort(faces, axis=1)
    key = (s[:, 0] << 42) | (s[:, 1] << 21) | s[:, 2]
    order = np.argsort(key, kind="stable")
    same = key[order[1:]] == key[order[:-1]]
    first, second = order[:-1][same], order[1:][same]
    paired = np.zeros(len(faces), bool)
    paired[first] = paired[second] = True
    return faces, opposite, owner, first, second, paired


def _outward_normals(faces, opposite, pts):
    np = _np()
    p0, p1, p2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    n[np.einsum("ij,ij->i", n, pts[opposite] - p0) > 0] *= -1
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _preprocess_kcl_torso(data: Path) -> None:  # noqa: C901 - a straight port of the demo
    import numpy as np
    from scipy.spatial import cKDTree

    t0 = time.time()
    print("Pre-processing the KCL heart-torso model ...", flush=True)
    pts_um, tets, tags, lon = _read_mesh(data / KCL_TORSO / KCL_TORSO)
    pts = pts_um / 1000.0  # mm
    faces, opp, owner, first, second, paired = _face_adjacency(tets)

    # ---------------- heart sub-mesh (torso node ordering) ----------------
    heart_el = np.isin(tags, [_RV_WALL, _LV_WALL])
    h2t = np.unique(tets[heart_el])  # sorted, as LeadField
    t2h = np.full(len(pts), -1, np.int64)
    t2h[h2t] = np.arange(len(h2t))
    hx = pts[h2t]

    # ---------------- heart surfaces from neighbour tags ------------------
    is_heart = heart_el[owner]
    a, b = first, second
    mask = is_heart[a] ^ is_heart[b]
    hf = np.where(is_heart[a[mask]], a[mask], b[mask])  # heart-side face
    nb_tag = tags[owner[np.where(is_heart[a[mask]], b[mask], a[mask])]]
    surf = {
        "lv_endo": np.unique(faces[hf[nb_tag == _LV_BLOOD]]),
        "rv_endo": np.unique(faces[hf[nb_tag == _RV_BLOOD]]),
        "base": np.unique(faces[hf[np.isin(nb_tag, _BASE_TAGS)]]),
        "epi": np.unique(faces[hf[~np.isin(nb_tag, [*_BASE_TAGS, _LV_BLOOD, _RV_BLOOD])]]),
    }
    surf = {k: t2h[v] for k, v in surf.items()}

    # ---------------- apico-basal coordinate ------------------------------
    base_c = hx[surf["base"]].mean(0)
    lv_nodes = np.unique(t2h[tets[tags == _LV_WALL]])
    lv_epi = np.intersect1d(surf["epi"], lv_nodes)
    apex = hx[lv_epi[np.argmax(np.linalg.norm(hx[lv_epi] - base_c, axis=1))]]
    axis = (base_c - apex) / np.linalg.norm(base_c - apex)
    # distance ratio rather than a projection on the LV long axis, so that the RV free
    # wall (which bulges far "above" the LV base along that axis) is graded by its
    # distance to the valve rings
    d_apex = np.linalg.norm(hx - apex, axis=1)
    d_base = cKDTree(hx[surf["base"]]).query(hx)[0]
    ab = d_apex / (d_apex + d_base + 1e-9)  # 0 apex, 1 base

    # ---------------- transmural coordinate -------------------------------
    endo = np.concatenate([surf["lv_endo"], surf["rv_endo"]])
    d_endo = cKDTree(hx[endo]).query(hx)[0]
    d_epi = cKDTree(hx[surf["epi"]]).query(hx)[0]
    tm = d_endo / (d_endo + d_epi + 1e-9)  # 0 endo, 1 epi

    # ---------------- circumferential angle around the LV -----------------
    def perp(v):
        return v - np.outer(v @ axis, axis)

    lv_c, rv_c = hx[surf["lv_endo"]].mean(0), hx[surf["rv_endo"]].mean(0)
    e_sep = perp((rv_c - lv_c)[None])[0]
    e_sep /= np.linalg.norm(e_sep)
    e_ant = np.cross(axis, e_sep)
    if e_ant @ np.array([0, -1, 0]) < 0:  # LPS: anterior = -y
        e_ant = -e_ant
    v = perp(hx - lv_c)
    theta = np.degrees(np.arctan2(v @ e_ant, v @ e_sep))  # 0 septum, +90 anterior

    # ---------------- heart mesh with fast endocardial layer --------------
    htets = t2h[tets[heart_el]]
    htags = tags[heart_el].copy()
    fast = (d_endo[htets].mean(1) < 1.0) & (ab[htets].mean(1) < 0.95)
    htags[fast] = _FAST_ENDO

    heart_dir = data / "heart"
    heart_dir.mkdir(parents=True, exist_ok=True)
    with open(heart_dir / "heart.pts", "w", encoding="utf-8") as fh:
        fh.write(f"{len(h2t)}\n")
        np.savetxt(fh, pts_um[h2t], fmt="%.6f")
    with open(heart_dir / "heart.elem", "w", encoding="utf-8") as fh:
        fh.write(f"{len(htets)}\n")
        np.savetxt(fh, np.column_stack([htets, htags]), fmt="Tt %d %d %d %d %d")
    with open(heart_dir / "heart.lon", "w", encoding="utf-8") as fh:
        fh.write("1\n")
        np.savetxt(fh, lon[heart_el], fmt="%.6f")

    # ---------------- Durrer early-activation sites -----------------------
    lv_endo, rv_endo = surf["lv_endo"], surf["rv_endo"]
    d_lv = cKDTree(hx[lv_endo]).query(hx)[0]
    rv_sept = rv_endo[d_lv[rv_endo] < 15]
    rv_free = rv_endo[d_lv[rv_endo] > 25]

    def site(cands, ab_t, th_t=None):
        cost = np.abs(ab[cands] - ab_t) / 0.05
        if th_t is not None:
            dth = (theta[cands] - th_t + 180) % 360 - 180
            cost = cost + np.abs(dth) / 15
        c = hx[cands[np.argmin(cost)]]
        return np.where((np.linalg.norm(hx - c, axis=1) < 3.0) & (d_endo < 1.5))[0]

    # LV mid-septal + anterior paraseptal (high), RV septal + moderator band.
    # (Durrer's posterior paraseptal site is not used.)
    sites = {
        "LV_septal": site(lv_endo, 0.55, 0),
        "LV_anterior": site(lv_endo, 0.70, 60),
        "RV_septal": site(rv_sept, 0.50),
        "RV_moderator": site(rv_free, 0.35),
    }
    for name, ids in sites.items():
        write_vtx(data / "pacing" / f"{name}.vtx", ids)

    # ---------------- electrodes on the skin ------------------------------
    bfaces = faces[~paired]
    bnorm = _outward_normals(bfaces, opp[~paired], pts)
    side = np.abs(bnorm[:, 2]) < 0.7  # drop the cut planes
    skin_n = np.zeros_like(pts)
    np.add.at(skin_n, bfaces[side].reshape(-1), np.repeat(bnorm[side], 3, 0))
    skin = np.unique(bfaces[side])
    sx = pts[skin]
    sn = skin_n[skin] / np.linalg.norm(skin_n[skin], axis=1, keepdims=True)

    z_v4 = apex[2] + 5.0  # 5th ICS ~ apex beat
    z_v12 = z_v4 + 20.0  # 4th ICS
    sl = np.abs(sx[:, 2] - z_v4) < 4
    x_mid = 0.5 * (sx[sl, 0].min() + sx[sl, 0].max())
    bone = pts[np.unique(tets[tags == _BONE])]
    st = bone[(np.abs(bone[:, 0] - x_mid) < 40) & (np.abs(bone[:, 2] - z_v12) < 30)]
    x_mid = st[np.argsort(st[:, 1])[:200], 0].mean()  # sternum (most anterior bone)
    half_w = sx[sl, 0].max() - x_mid

    def front(x_t, z_t):  # anterior skin node
        c = np.where((np.abs(sx[:, 2] - z_t) < 4) & (sn[:, 1] < -0.3))[0]
        return skin[c[np.argmin(np.abs(sx[c, 0] - x_t))]]

    def lateral(z_t, phi_t):  # by angle in a slice
        c = np.where(np.abs(sx[:, 2] - z_t) < 4)[0]
        yc = 0.5 * (sx[c, 1].min() + sx[c, 1].max())
        phi = np.degrees(np.arctan2(sx[c, 0] - x_mid, -(sx[c, 1] - yc)))
        return skin[c[np.argmin(np.abs(phi - phi_t))]], yc

    v1, v2 = front(x_mid - 25, z_v12), front(x_mid + 25, z_v12)
    v4 = front(x_mid + 0.55 * half_w, z_v4)
    v3 = front(0.5 * (pts[v2, 0] + pts[v4, 0]), 0.5 * (z_v12 + z_v4))
    v6, yc = lateral(z_v4, 90)
    phi4 = np.degrees(np.arctan2(pts[v4, 0] - x_mid, -(pts[v4, 1] - yc)))
    v5, _ = lateral(z_v4, 0.5 * (phi4 + 90))
    z_top, z_bot = sx[:, 2].max(), sx[:, 2].min()
    ra = front(x_mid - 0.70 * half_w, z_top - 35)
    la = front(x_mid + 0.70 * half_w, z_top - 35)
    rl = front(x_mid - 0.60 * half_w, z_bot + 35)
    ll = front(x_mid + 0.60 * half_w, z_bot + 35)
    write_vtx(data / "electrodes" / "lf_src.vtx", [v1, v2, v3, v4, v5, v6, ra, la, rl, ll])

    # nodal coordinates used for the ionic heterogeneity
    np.savez_compressed(data / "fields.npz", ab=ab, tm=tm)
    print(f"  heart: {len(h2t):,} nodes, {len(htets):,} tets "
          f"({fast.sum():,} fast-endocardial) | done in {time.time() - t0:.0f} s", flush=True)


# =============================================================================== recording
class FrameRecorder:
    """Evenly spaced frames of nodal fields on a surface, kept within the bundle budget.

    ``add(t, displacement=u, vm=v)`` takes full-mesh nodal arrays (torch or numpy, shape
    ``(n_nodes,)`` or ``(n_nodes, k)``) and keeps their surface values on the host. Call
    it whenever :meth:`due` says so; ``interval`` is the spacing in the caller's unit.
    When a field would exceed ``budget`` values, every other frame is dropped and the
    interval doubles, so long runs keep evenly spaced frames at bounded memory.
    """

    def __init__(self, surface: Surface, interval: float = 0.0,
                 budget: int = MAX_FIELD_VALUES) -> None:
        self.surface = surface
        self.interval = float(interval)
        self.budget = int(budget)
        self.times: list[float] = []
        self.fields: dict[str, list] = {}
        self._next: float | None = None

    def due(self, t: float) -> bool:
        return self._next is None or t >= self._next - 1e-9

    def add(self, t: float, **fields) -> None:
        np = _np()
        for name, value in fields.items():
            values = _rows(value, self.surface.node_ids).astype(np.float32)
            self.fields.setdefault(name, []).append(values)
        self.times.append(float(t))
        self._next = float(t) + self.interval
        while len(self.times) > 2 and self._largest() > self.budget:
            self.times = self.times[::2]
            self.fields = {name: frames[::2] for name, frames in self.fields.items()}
            self.interval *= 2.0
            self._next = self.times[-1] + self.interval

    def _largest(self) -> int:
        return max((len(frames) * frames[0].size for frames in self.fields.values()), default=0)

    def array(self, name: str, dtype="float32"):
        """``(frames, points, ...)`` array of one field, or None when it was not recorded."""
        frames = self.fields.get(name)
        return None if not frames else _np().stack(frames).astype(dtype)


class SolverLog:
    """Tee standard output and keep TorchCor's mechanics step lines::

        with studio.SolverLog() as log:
            sim.solve(verbose=True)

    ``steps``, ``values`` (load factor λ or time) and ``newton`` (Newton iterations) get
    one entry per accepted load step (``load step N: lambda = ...``) or time step
    (``t = ...  newton = ...``); the output itself still reaches the console unchanged.
    """

    _LOAD = re.compile(r"load step\s+(\d+):\s+lambda\s*=\s*([-\d.eE+]+)\s+newton\s*=\s*(\d+)")
    _TIME = re.compile(r"^\s*t\s*=\s*([-\d.eE+]+)\s+newton\s*=\s*(\d+)")

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.values: list[float] = []
        self.newton: list[int] = []
        self._stream = None
        self._partial = ""

    def __enter__(self) -> SolverLog:
        self._stream, sys.stdout = sys.stdout, self
        return self

    def __exit__(self, *exc) -> None:
        if self._partial:
            self._parse(self._partial)
            self._partial = ""
        sys.stdout = self._stream

    def write(self, text: str) -> int:
        self._stream.write(text)
        *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            self._parse(line)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):  # encoding, fileno, isatty, ... of the real stream
        return getattr(self._stream, name)

    def _parse(self, line: str) -> None:
        load = self._LOAD.search(line)
        if load:
            self._record(int(load.group(1)), float(load.group(2)), int(load.group(3)))
            return
        step = self._TIME.match(line)
        if step:
            self._record(len(self.steps) + 1, float(step.group(1)), int(step.group(2)))

    def _record(self, step: int, value: float, newton: int) -> None:
        self.steps.append(step)
        self.values.append(value)
        self.newton.append(newton)


# =============================================================================== bundle
class Bundle:
    """Writes ``viz/index.json`` and its ``.npz`` datasets."""

    def __init__(self, kind: str, title: str = "", directory=None) -> None:
        self.kind = kind
        self.title = title
        self.dir = Path(directory) if directory is not None else run_dir() / "viz"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.datasets: dict[str, str] = {}
        self.summary: list[dict] = []
        self.notes: list[str] = []

    def add(self, name: str, **arrays) -> Path:
        np = _np()
        clean = {k: np.asarray(v) for k, v in arrays.items() if v is not None}
        path = self.dir / f"{name}.npz"
        np.savez_compressed(path, **clean)
        self.datasets[name] = path.name
        return path

    def add_surface(self, points, triangles, node_ids, regions=None) -> None:
        np = _np()
        self.add("surface", points=np.asarray(points, dtype=np.float32),
                 triangles=np.asarray(triangles, dtype=np.int32),
                 node_ids=np.asarray(node_ids, dtype=np.int64),
                 regions=None if regions is None else np.asarray(regions, dtype=np.int32))

    def stat(self, label: str, value, unit: str = "", hint: str = "") -> None:
        np = _np()
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = "—"
        self.summary.append({"label": label, "value": value, "unit": unit, "hint": hint})

    def note(self, text: str) -> None:
        self.notes.append(text)

    def write(self) -> Path:
        index = {
            "format": VIZ_FORMAT,
            "kind": self.kind,
            "title": self.title,
            "created_at": time.time(),
            "datasets": self.datasets,
            "summary": self.summary,
            "notes": self.notes,
        }
        path = self.dir / "index.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index, indent=2, default=_jsonable), encoding="utf-8")
        os.replace(tmp, path)
        artifact(path, kind="viz", label="Visualisation bundle")
        return path


def _finite_stat(values, fn, default=float("nan")) -> float:
    np = _np()
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if finite.size else default


# =============================================================================== exports
def export_ep(simulator, Vm, snapshot_interval: float, ATs=None, RTs=None, bundle=None, *,
              title: str = "", traces: int = 6, start_time: float = 0.0) -> Bundle:
    """Add an electrophysiology result to ``bundle``; returns the bundle.

    ``simulator`` is a solved ``Monodomain``/``ReactionEikonal``; ``Vm`` its
    ``(frames, nodes)`` history (any device) with frame 0 at ``start_time``; ``ATs`` and
    ``RTs`` full-mesh activation and repolarisation times (ms, NaN where none). Adds
    ``surface.npz``, ``ep.npz`` and the summary statistics. Without a ``bundle`` a new
    ``"ep"`` bundle is made and written; a given one is left for the caller to write.
    """
    np = _np()
    owned = bundle is None
    bundle = Bundle("ep", title) if owned else bundle
    surface = simulator_surface(simulator)
    bundle.add_surface(*surface)
    n_frames, n_nodes = int(Vm.shape[0]), int(Vm.shape[1])

    stride = frame_stride(n_frames, len(surface.node_ids))
    vm = _columns(Vm, surface.node_ids, stride).astype(np.float16)
    times = (start_time + np.arange(0, n_frames, stride) * snapshot_interval).astype(np.float32)

    at_full = None if ATs is None else to_numpy(ATs).astype(np.float32).reshape(-1)
    rt_full = None if RTs is None else to_numpy(RTs).astype(np.float32).reshape(-1)
    lat = (np.full(len(surface.node_ids), np.nan, np.float32) if at_full is None
           else at_full[surface.node_ids])
    rt = None if rt_full is None else rt_full[surface.node_ids]
    apd = None if rt is None else (rt - lat).astype(np.float32)

    probe, labels = pick_trace_nodes(surface.points, None if at_full is None else lat, traces)
    trace_ids = surface.node_ids[probe]
    trace_values = _columns(Vm, trace_ids).T.astype(np.float32)
    trace_times = (start_time + np.arange(n_frames) * snapshot_interval).astype(np.float32)

    bundle.add("ep", times=times, vm=vm, lat=lat, rt=rt, apd=apd,
               trace_times=trace_times, traces=trace_values,
               trace_labels=np.asarray(labels, dtype=str), trace_node_ids=trace_ids)

    if at_full is not None:
        finite = np.isfinite(at_full)
        bundle.stat("Total activation time",
                    _finite_stat(at_full, lambda v: v.max() - v.min()), "ms",
                    "Latest minus earliest activation over the whole mesh")
        bundle.stat("Activated", 100.0 * finite.mean(), "%",
                    "Share of nodes whose Vm crossed the activation threshold")
    if rt_full is not None and at_full is not None:
        bundle.stat("Mean APD", _finite_stat(rt_full - at_full, np.mean), "ms",
                    "Repolarisation minus activation time, averaged over the mesh")
    bundle.stat("Peak Vm", _finite_stat(vm.astype(np.float32), np.max), "mV",
                "Highest membrane potential on the surface")
    bundle.stat("Nodes", n_nodes, "", "Mesh nodes")
    bundle.stat("Elements", int(simulator.elems.elems_num()), "", "Mesh elements")
    bundle.stat("Simulated", float((n_frames - 1) * snapshot_interval), "ms",
                f"{n_frames} frames every {snapshot_interval:g} ms")
    if stride > 1:
        bundle.note(f"Vm is shown every {stride} frames ({stride * snapshot_interval:g} ms) to "
                    "keep the bundle small; traces keep every frame.")
    if owned:
        bundle.write()
    return bundle


def export_ecg(ecg, dt_ms: float, bundle=None, *, title: str = "", simulator=None, Vm=None,
               ATs=None, RTs=None, snapshot_interval: float | None = None) -> Bundle:
    """Add a 12-lead ECG (``LeadField.compute_12lead`` output) to ``bundle``.

    ``ecg`` maps lead names to ``(samples,)`` signals in mV, ``dt_ms`` apart. With the
    heart's ``simulator`` and ``Vm`` the bundle also carries the heart surface and its
    activation (:func:`export_ep`). Without a ``bundle`` a new ``"ecg"`` bundle is made
    and written; a given one is left for the caller to write.
    """
    np = _np()
    owned = bundle is None
    bundle = Bundle("ecg", title) if owned else bundle
    missing = [name for name in LEADS if name not in ecg]
    if missing:
        raise KeyError(f"the ECG has no lead(s) {missing}")
    leads = np.stack([to_numpy(ecg[name]).astype(np.float32).reshape(-1) for name in LEADS])
    t = (np.arange(leads.shape[1]) * dt_ms).astype(np.float32)
    bundle.add("ecg", t=t, leads=leads, lead_names=np.asarray(LEADS, dtype=str))

    lead_ii = leads[LEADS.index("II")]
    bundle.stat("Duration", float(t[-1]) if len(t) else 0.0, "ms",
                f"{leads.shape[1]} samples every {dt_ms:g} ms")
    bundle.stat("Lead II peak-to-peak", float(np.ptp(lead_ii)) if len(lead_ii) else 0.0, "mV",
                "Largest minus smallest value of lead II")
    if len(lead_ii):
        bundle.stat("Lead II peak at", float(t[int(np.argmax(np.abs(lead_ii)))]), "ms",
                    "Time of the largest |lead II| (usually the R wave)")
    if simulator is not None and Vm is not None:
        export_ep(simulator, Vm, snapshot_interval or dt_ms, ATs=ATs, RTs=RTs, bundle=bundle)
    if owned:
        bundle.write()
    return bundle


def export_mechanics(frames: FrameRecorder, bundle=None, *, title: str = "",
                     frame_label: str = "Load factor λ", frame_unit: str = "",
                     log: SolverLog | None = None, report=None, n_dofs: int | None = None
                     ) -> Bundle:
    """Add a mechanics result to ``bundle`` (a new ``"mechanics"`` bundle when None).

    ``frames`` holds the surface displacement frames (field ``displacement``, mm) and any
    point scalars (each written as ``field_<name>``). ``log`` supplies the convergence
    history and ``report`` (``Mechanics.report``) the summary. Without a ``bundle`` a new
    one is made and written; a given one is left for the caller to write.
    """
    np = _np()
    owned = bundle is None
    bundle = Bundle("mechanics", title) if owned else bundle
    surface = frames.surface
    bundle.add_surface(*surface)
    displacement = frames.array("displacement")
    if displacement is None:
        raise ValueError("no displacement frames were recorded")
    values = np.asarray(frames.times, dtype=np.float32)
    stride = frame_stride(len(values), len(surface.node_ids), 3)
    keep = np.unique(np.append(np.arange(0, len(values), stride), len(values) - 1))
    scalars = {f"field_{name}": frames.array(name, "float16")[keep]
               for name in frames.fields if name != "displacement"}
    conv = {} if log is None else {
        "conv_step": np.asarray(log.steps, dtype=np.int32),
        "conv_value": np.asarray(log.values, dtype=np.float32),
        "conv_newton": np.asarray(log.newton, dtype=np.int32)}
    bundle.add("mechanics", frames=values[keep], frame_label=np.asarray(frame_label),
               frame_unit=np.asarray(frame_unit), displacement=displacement[keep],
               **scalars, **conv)

    magnitude = np.linalg.norm(displacement, axis=-1)
    bundle.stat("Max displacement", float(magnitude.max()) if magnitude.size else 0.0, "mm",
                "Largest displacement of a surface point over the run")
    if report is not None:
        bundle.stat("Converged", "yes" if report.converged else "no", "",
                    "Whether the solver reached the full load / end time")
        bundle.stat("Steps", int(report.load_steps), "",
                    "Accepted load or time steps")
        bundle.stat("Newton iterations", int(report.newton_iterations), "",
                    "Nonlinear iterations over all steps")
        bundle.stat("Min J", float(report.min_jacobian), "",
                    "Smallest volume ratio det(F) at the end; must stay positive")
        bundle.stat("Wall time", float(report.wall_time), "s", "Time spent in the solver")
    if n_dofs is not None:
        bundle.stat("Degrees of freedom", int(n_dofs), "", "3 per mesh node")
    if len(keep) < len(values):
        bundle.note(f"Every {stride}th of {len(values)} frames is kept to keep the bundle small.")
    if owned:
        bundle.write()
    return bundle


def export_em(frames: FrameRecorder, pv=None, bundle=None, *, title: str = "",
              cycle_length: float | None = None) -> Bundle:
    """Add a four-chamber electromechanics result to ``bundle``.

    ``frames`` holds surface frames (fields ``displacement`` mm, optional ``vm`` mV and
    ``tension`` kPa) recorded at times in ms; ``pv`` is a mapping or a list of row
    mappings with ``t`` (ms), ``p_<chamber>`` (mmHg) and ``V_<chamber>`` (mL). Without a
    ``bundle`` a new one is made and written; a given one is left for the caller to write.
    """
    np = _np()
    owned = bundle is None
    bundle = Bundle("em", title) if owned else bundle
    surface = frames.surface
    bundle.add_surface(*surface)
    displacement = frames.array("displacement")
    if displacement is None:
        raise ValueError("no displacement frames were recorded")
    times = np.asarray(frames.times, dtype=np.float32)
    stride = frame_stride(len(times), len(surface.node_ids), 3)
    keep = np.arange(0, len(times), stride)
    vm, tension = frames.array("vm", "float16"), frames.array("tension", "float16")

    columns = _pv_columns(pv)
    traces = {}
    if columns:
        traces["pv_t"] = np.asarray(columns["t"], dtype=np.float32)
        for chamber in CHAMBERS:
            for prefix in ("p", "V"):
                key = f"{prefix}_{chamber}"
                if key in columns:
                    traces[key] = np.asarray(columns[key], dtype=np.float32)
        if cycle_length:
            traces["pv_cycle"] = np.float32(cycle_length)
    bundle.add("em", times=times[keep], displacement=displacement[keep],
               vm=None if vm is None else vm[keep],
               tension=None if tension is None else tension[keep], **traces)

    magnitude = np.linalg.norm(displacement, axis=-1)
    bundle.stat("Max displacement", float(magnitude.max()) if magnitude.size else 0.0, "mm",
                "Largest displacement of a surface point")
    if "V_lv" in traces and "p_lv" in traces:
        t, volume, pressure = traces["pv_t"], traces["V_lv"], traces["p_lv"]
        whole_beat = bool(cycle_length) and len(t) > 1 and (t[-1] - t[0]) >= 0.99 * cycle_length
        last = t >= (t[-1] - cycle_length) if cycle_length else np.ones_like(t, bool)
        if whole_beat:
            edv, esv = float(volume[last].max()), float(volume[last].min())
            bundle.stat("LV end-diastolic volume", edv, "mL", "Largest LV volume of the last beat")
            bundle.stat("LV end-systolic volume", esv, "mL", "Smallest LV volume of the last beat")
            bundle.stat("LV ejection fraction", 100.0 * (edv - esv) / edv if edv > 0 else 0.0, "%",
                        "(EDV - ESV) / EDV over the last beat")
        else:  # EDV/ESV/EF need a whole cycle; report the volume range instead
            bundle.stat("LV volume", f"{float(volume.min()):.1f}–{float(volume.max()):.1f}", "mL",
                        "Range over the simulated time")
            bundle.note("Less than one cardiac cycle was simulated, so EDV, ESV and ejection "
                        "fraction are not reported.")
        bundle.stat("Peak LV pressure", float(pressure[last].max()), "mmHg",
                    "Highest LV pressure of the last beat" if whole_beat else "Highest LV pressure so far")
    bundle.stat("Simulated", float(times[-1]) if len(times) else 0.0, "ms",
                f"{len(times)} surface frames")
    if stride > 1:
        bundle.note(f"Every {stride}th surface frame is kept to keep the bundle small.")
    if owned:
        bundle.write()
    return bundle


def _pv_columns(pv) -> dict:
    """Column dictionary from a mapping of sequences or a list of row mappings."""
    if not pv:
        return {}
    if isinstance(pv, dict):
        return dict(pv)
    rows = list(pv)
    return {key: [row[key] for row in rows] for key in rows[0]}
