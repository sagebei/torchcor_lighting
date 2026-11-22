"""``tcs_studio``: inputs, recorders and the visualisation bundle, on synthetic data.

The bundle format is the contract with the results viewers (see the module docstring of
``tcs_studio``); these tests check every dataset against it. The KCL torso preparation is
exercised on a small synthetic heart-torso model, never by downloading the real one.
"""

from __future__ import annotations

import importlib.metadata
import io
import json
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from torchcor_studio.runtime import runtime_path

sys.path.insert(0, str(runtime_path("tcs_studio.py").parent))
import tcs_studio as studio  # noqa: E402 - the runtime file, imported as run.py does


def _installed(name: str) -> bool:
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


HAS_TORCHCOR = _installed("torchcor")


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _index(directory: Path) -> dict:
    return json.loads((directory / "index.json").read_text(encoding="utf-8"))


# =============================================================================== fakes
def _read_slab(folder: Path):
    points = np.loadtxt(folder / "slab.pts", skiprows=1) / 1000.0
    elems = np.loadtxt(folder / "slab.elem", skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=np.int64)
    return points, elems[:, :4], elems[:, 4]


class _Element:
    def __init__(self, data=None, region=None):
        self.data, self.region = data, region


class _Elems:
    def __init__(self, tets, regions):
        self.Tt, self.Tr, self.Ln = _Element(tets, regions), _Element(), _Element()

    def elems_num(self) -> int:
        return len(self.Tt.data)


def _fake_simulator(tmp_path: Path):
    folder = studio.make_slab_mesh(tmp_path / "slab", lengths=(6.0, 3.0, 2.0), dx=1.0, corner=1.0)
    points, tets, regions = _read_slab(folder)
    return SimpleNamespace(nodes=points, elems=_Elems(tets, regions)), points


def _wave(points, frames=40, dt=1.0, speed=0.25):
    """Vm of a planar wave along x: (frames, nodes) mV, and its activation times."""
    at = points[:, 0] / speed
    t = np.arange(frames)[:, None] * dt
    vm = -85.0 + 125.0 / (1.0 + np.exp(-(t - at[None, :]) / 0.5))
    lat = np.where(at <= (frames - 1) * dt, at, np.nan)
    return vm.astype(np.float32), lat


class _HexMesh:
    """The parts of a torchcor.mechanics mesh that the exports use."""

    def __init__(self, n=(3, 2, 2), size=(3.0, 2.0, 2.0)):
        axes = [np.linspace(0.0, s, k + 1) for s, k in zip(size, n, strict=True)]
        grid = np.meshgrid(*axes, indexing="ij")
        self.points = np.stack([g.ravel() for g in grid], axis=1)
        ids = np.arange(len(self.points)).reshape(*(k + 1 for k in n))
        self._cells = np.stack([ids[:-1, :-1, :-1], ids[1:, :-1, :-1], ids[1:, 1:, :-1],
                                ids[:-1, 1:, :-1], ids[:-1, :-1, 1:], ids[1:, :-1, 1:],
                                ids[1:, 1:, 1:], ids[:-1, 1:, 1:]], axis=-1).reshape(-1, 8)
        self.n_dofs = 3 * len(self.points)

    def linear_cells(self):
        return self._cells


def _signed_volume(points, triangles) -> float:
    p = np.asarray(points, dtype=np.float64)
    a, b, c = (p[triangles[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# =============================================================================== import
def test_tcs_studio_imports_with_the_standard_library_only(tmp_path):
    shutil.copy(runtime_path("tcs_studio.py"), tmp_path / "tcs_studio.py")
    code = ("import sys; sys.modules['numpy'] = None; sys.modules['torch'] = None\n"
            "import tcs_studio as studio\n"
            "studio.stage('setup', 'Setting up'); studio.total(10); studio.progress(0.5)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    markers = [json.loads(line.split(" ", 1)[1]) for line in out.stdout.splitlines()
               if line.startswith(studio.MARKER)]
    assert [m["type"] for m in markers] == ["stage", "total", "progress"]


# =============================================================================== inputs
def test_make_slab_mesh_writes_an_opencarp_mesh(tmp_path):
    folder = studio.make_slab_mesh(tmp_path / "mesh", lengths=(4.0, 2.0, 1.0), dx=0.5, corner=1.0)
    nx, ny, nz = 9, 5, 3
    lines = (folder / "slab.pts").read_text().splitlines()
    assert int(lines[0]) == nx * ny * nz == len(lines) - 1
    assert lines[-1].split() == ["4000.000", "2000.000", "1000.000"]  # µm
    elem = (folder / "slab.elem").read_text().splitlines()
    assert int(elem[0]) == 6 * (nx - 1) * (ny - 1) * (nz - 1) == len(elem) - 1
    assert all(row.startswith("Tt ") and row.endswith(" 1") for row in elem[1:])
    lon = (folder / "slab.lon").read_text().splitlines()
    assert lon[0] == "1" and len(lon) == len(elem) and lon[1] == "1 0 0"

    points, tets, _ = _read_slab(folder)
    vol = np.linalg.det(np.stack([points[tets[:, i]] - points[tets[:, 0]] for i in (1, 2, 3)],
                                 axis=1))
    assert (vol > 0).all(), "every tetrahedron is positively oriented"
    assert np.isclose(vol.sum() / 6.0, 4.0 * 2.0 * 1.0)

    for name, expected in (("stimulus.vtx", 3 * 3 * 3), ("left_face.vtx", ny * nz)):
        vtx = (folder / name).read_text().splitlines()
        assert int(vtx[0]) == expected == len(vtx) - 2 and vtx[1] == "extra"
        ids = np.array(vtx[2:], dtype=np.int64)
        assert ids.min() >= 0 and ids.max() < len(points)
    corner = np.array((folder / "stimulus.vtx").read_text().split()[2:], dtype=np.int64)
    assert (points[corner] <= 1.0 + 1e-9).all()


def test_prepared_kcl_folder_is_not_downloaded_again(tmp_path, monkeypatch):
    for path in ("KCL_torso3/KCL_torso3.elem", "heart/heart.elem", "electrodes/lf_src.vtx",
                 "fields.npz", *(f"pacing/{s}.vtx" for s in studio.KCL_PACING_SITES)):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("")
    import urllib.request

    def no_network(*_args, **_kwargs):
        raise AssertionError("must not download")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert studio.kcl_torso_ready(tmp_path)
    assert studio.prepare_kcl_torso(tmp_path) == tmp_path
    (tmp_path / "fields.npz").unlink()
    assert not studio.kcl_torso_ready(tmp_path)


# ------------------------------------------------------------------ synthetic heart-torso
_LV, _RV = np.array([88.0, 60.0, 100.0]), np.array([52.0, 60.0, 100.0])


def _synthetic_torso(data: Path, h: float = 4.0) -> None:
    """A box torso (160 x 120 x 200 mm) with two ventricles, a base and a sternum, tagged
    like the KCL model, written as ``KCL_torso3/KCL_torso3.{pts,elem,lon}`` (µm)."""
    size = (160.0, 120.0, 200.0)
    n = [round(s / h) + 1 for s in size]
    grid = np.meshgrid(*[np.linspace(0.0, s, k) for s, k in zip(size, n, strict=True)],
                       indexing="ij")
    points = np.stack([g.ravel() for g in grid], axis=1)
    ids = np.arange(len(points)).reshape(n)
    cube = np.stack([ids[:-1, :-1, :-1], ids[1:, :-1, :-1], ids[1:, 1:, :-1], ids[:-1, 1:, :-1],
                     ids[:-1, :-1, 1:], ids[1:, :-1, 1:], ids[1:, 1:, 1:], ids[:-1, 1:, 1:]],
                    axis=-1).reshape(-1, 8)
    tets = np.concatenate([cube[:, [0, 1, 2, 6]], cube[:, [0, 2, 3, 6]], cube[:, [0, 3, 7, 6]],
                           cube[:, [0, 7, 4, 6]], cube[:, [0, 4, 5, 6]], cube[:, [0, 5, 1, 6]]])
    c = points[tets].mean(axis=1)

    def inside(centre, radii):
        return np.linalg.norm((c - centre) / np.asarray(radii), axis=1) < 1.0

    lv_in, lv_out = inside(_LV, (14, 14, 22)), inside(_LV, (24, 24, 32))
    rv_in, rv_out = inside(_RV, (12, 18, 22)), inside(_RV, (20, 26, 30))
    heart = lv_out | rv_out
    tags = np.ones(len(tets), dtype=np.int64)  # bath
    sternum = ((c[:, 0] > 72) & (c[:, 0] < 104) & (c[:, 1] > 4) & (c[:, 1] < 16)
               & (c[:, 2] > 60) & (c[:, 2] < 140))
    tags[sternum] = 3
    tags[rv_out] = 34
    tags[lv_out] = 35
    tags[rv_in & ~lv_out] = 24
    tags[lv_in] = 25
    tags[heart & (c[:, 2] > 116)] = 9  # valve planes and atria above the base
    folder = data / studio.KCL_TORSO
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{studio.KCL_TORSO}.pts", "w") as fh:
        fh.write(f"{len(points)}\n")
        np.savetxt(fh, points * 1000.0, fmt="%.3f")
    with open(folder / f"{studio.KCL_TORSO}.elem", "w") as fh:
        fh.write(f"{len(tets)}\n")
        np.savetxt(fh, np.column_stack([tets, tags]), fmt="Tt %d %d %d %d %d")
    with open(folder / f"{studio.KCL_TORSO}.lon", "w") as fh:
        fh.write("1\n" + "1 0 0\n" * len(tets))


@pytest.fixture(scope="module")
def prepared_torso(tmp_path_factory) -> Path:
    pytest.importorskip("scipy", reason="preparing the torso needs scipy, as on a compute machine")
    data = tmp_path_factory.mktemp("kcl")
    _synthetic_torso(data)
    with redirect_stdout(io.StringIO()):
        studio.prepare_kcl_torso(data)  # the torso is there: preprocess only
    return data


def test_kcl_preprocessing_on_a_synthetic_torso(prepared_torso):
    data = prepared_torso
    assert studio.kcl_torso_ready(data)
    torso = np.loadtxt(data / studio.KCL_TORSO / f"{studio.KCL_TORSO}.elem", skiprows=1,
                       usecols=(1, 2, 3, 4, 5), dtype=np.int64)
    heart_nodes = np.unique(torso[np.isin(torso[:, 4], [34, 35]), :4])
    pts = np.loadtxt(data / "heart" / "heart.pts", skiprows=1)
    torso_pts = np.loadtxt(data / studio.KCL_TORSO / f"{studio.KCL_TORSO}.pts", skiprows=1)
    assert len(pts) == len(heart_nodes) and np.allclose(pts, torso_pts[heart_nodes])
    heart = np.loadtxt(data / "heart" / "heart.elem", skiprows=1, usecols=(1, 2, 3, 4, 5),
                       dtype=np.int64)
    assert set(np.unique(heart[:, 4])) <= {34, 35, 36}
    assert heart[:, :4].max() == len(pts) - 1
    fields = np.load(data / "fields.npz")
    for key in ("ab", "tm"):
        assert fields[key].shape == (len(pts),)
        assert fields[key].min() >= 0.0 and fields[key].max() <= 1.0
    apex = np.argmin(fields["ab"])
    assert pts[apex][2] / 1000.0 < 80.0  # the apex is at the bottom of the LV
    for site in studio.KCL_PACING_SITES:
        vtx = (data / "pacing" / f"{site}.vtx").read_text().split()
        assert int(vtx[0]) >= 1 and int(vtx[0]) == len(vtx) - 2
    electrodes = np.array((data / "electrodes" / "lf_src.vtx").read_text().split()[2:], dtype=int)
    assert len(electrodes) == 10
    skin = torso_pts[electrodes] / 1000.0
    assert (skin[:4, 1] < 1e-6).all(), "V1-V4 sit on the anterior skin (y = 0)"
    assert skin[5, 0] > 150.0, "V6 is lateral"
    assert skin[6, 0] < skin[7, 0]  # RA on the right (smaller x), LA on the left


# =============================================================================== geometry
def test_boundary_of_hexes_is_closed_and_outward():
    mesh = _HexMesh()
    surface = studio.mechanics_surface(mesh)
    assert len(surface.triangles) == 2 * 2 * (3 * 2 + 3 * 2 + 2 * 2)
    assert np.isclose(_signed_volume(surface.points, surface.triangles), 12.0)
    assert set(surface.node_ids) == set(range(len(mesh.points))) - {
        int(i) for i in np.flatnonzero((mesh.points > 0).all(1) & (mesh.points[:, 0] < 3)
                                       & (mesh.points[:, 1] < 2) & (mesh.points[:, 2] < 2))}


def test_boundary_of_hexes_drops_collapsed_faces():
    # one hexahedron whose top face collapses to an edge (a wedge), as at an apex
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0.5, 1], [1, 0.5, 1]],
                      dtype=float)
    hexes = np.array([[0, 1, 2, 3, 4, 5, 5, 4]])
    tri = studio.boundary_of_hexes(points, hexes)
    assert ((tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])).all()
    assert np.isclose(_signed_volume(points, tri), 0.5)


def test_pick_trace_nodes():
    points = np.random.default_rng(0).random((200, 3))
    lat = points[:, 0] * 10.0
    lat[:20] = np.nan
    ids, labels = studio.pick_trace_nodes(points, lat, count=6)
    assert len(ids) == 6 == len(set(ids.tolist()))
    assert labels[:3] == ["Earliest activation", "Median activation", "Latest activation"]
    assert ids[0] == np.nanargmin(lat) and ids[2] == np.nanargmax(lat)
    ids, labels = studio.pick_trace_nodes(points, None, count=4)
    assert len(ids) == 4 and labels[0] == "Probe 1"


def test_frame_stride_respects_the_budget():
    assert studio.frame_stride(100, 1000) == 1
    assert studio.frame_stride(1000, 100_000, budget=10_000_000) == 10
    stride = studio.frame_stride(1001, 500_000, components=3)
    assert -(-1001 // stride) * 500_000 * 3 <= studio.MAX_FIELD_VALUES


# =============================================================================== recording
def test_frame_recorder_keeps_evenly_spaced_frames_within_budget():
    surface = studio.compact_surface(np.random.default_rng(1).random((10, 3)),
                                     np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 9]]))
    recorder = studio.FrameRecorder(surface, interval=1.0, budget=10 * 3 * 8)
    t = 0.0
    while t <= 40.0:
        if recorder.due(t):
            recorder.add(t, displacement=np.full((12, 3), t), vm=np.arange(12.0))
        t += 0.25
    assert len(recorder.times) <= 8
    gaps = np.diff(recorder.times)
    assert np.allclose(gaps, gaps[0]) and gaps[0] >= 4.0
    assert recorder.array("displacement").shape == (len(recorder.times), 10, 3)
    assert recorder.array("vm", "float16").dtype == np.float16
    assert recorder.array("tension") is None


def test_frame_recorder_takes_torch_tensors():
    torch = pytest.importorskip("torch")
    surface = studio.compact_surface(np.zeros((4, 3)), np.array([[3, 1, 2]]))
    recorder = studio.FrameRecorder(surface)
    recorder.add(0.0, displacement=torch.arange(12.0).reshape(4, 3))
    np.testing.assert_array_equal(recorder.array("displacement")[0],
                                  np.arange(12.0).reshape(4, 3)[[1, 2, 3]])


def test_solver_log_keeps_step_lines_and_passes_output_through():
    out = io.StringIO()
    with redirect_stdout(out), studio.SolverLog() as log:
        print("    tangent is symmetric; using ConjugateGradient")
        print("  load step  1: lambda = 0.100000  newton =  13  mixed constraint = 9e-07")
        print("  load step  2: lambda = 0.178468  newton =   6  mixed constraint = 9e-07")
        sys.stdout.write("  t = 2.500000  newton =   4  min J = 0.99\n  t = 5.0")
        sys.stdout.write("00000  newton =   5  min J = 0.98")
    assert sys.stdout is not log
    assert log.steps == [1, 2, 3, 4]
    assert log.values == [0.1, 0.178468, 2.5, 5.0]
    assert log.newton == [13, 6, 4, 5]
    assert "load step  2" in out.getvalue() and out.getvalue().endswith("min J = 0.98")


# =============================================================================== bundles
def test_bundle_index_and_statistics(tmp_path):
    bundle = studio.Bundle("ep", title="Test", directory=tmp_path)
    bundle.add("extra", values=np.arange(3), label="x")
    bundle.stat("Plain", 1.5, "ms", "hint")
    bundle.stat("Numpy", np.float32(2.0))
    bundle.stat("Missing", float("nan"))
    bundle.note("a note")
    with redirect_stdout(io.StringIO()):
        bundle.write()
    index = _index(tmp_path)
    assert index["format"] == 1 and index["kind"] == "ep" and index["title"] == "Test"
    assert index["datasets"] == {"extra": "extra.npz"}
    assert [row["value"] for row in index["summary"]] == [1.5, 2.0, "—"]
    assert index["notes"] == ["a note"]
    assert str(_load(tmp_path / "extra.npz")["label"]) == "x"


def _check_surface(directory: Path) -> dict:
    surface = _load(directory / "surface.npz")
    n = len(surface["points"])
    assert surface["points"].dtype == np.float32 and surface["points"].shape == (n, 3)
    assert surface["triangles"].dtype == np.int32 and surface["triangles"].max() < n
    assert surface["node_ids"].dtype == np.int64 and surface["node_ids"].shape == (n,)
    return surface


def test_export_ep_follows_the_bundle_spec(tmp_path):
    simulator, points = _fake_simulator(tmp_path)
    vm, lat = _wave(points)
    rt = lat + 200.0
    bundle = studio.Bundle("ep", title="EP", directory=tmp_path / "viz")
    with redirect_stdout(io.StringIO()):
        studio.export_ep(simulator, vm, 1.0, ATs=lat, RTs=rt, bundle=bundle)
        bundle.write()
    index = _index(tmp_path / "viz")
    assert set(index["datasets"]) == {"surface", "ep"}
    summary = {row["label"]: row for row in index["summary"]}
    assert summary["Total activation time"]["value"] == pytest.approx(np.nanmax(lat), abs=1e-3)
    assert summary["Mean APD"]["value"] == pytest.approx(200.0, abs=1e-3)
    assert summary["Nodes"]["value"] == len(points)
    assert summary["Elements"]["value"] == len(simulator.elems.Tt.data)
    surface = _check_surface(tmp_path / "viz")
    assert set(np.unique(_load(tmp_path / "viz" / "surface.npz")["regions"])) == {1}
    n = len(surface["points"])
    data = _load(tmp_path / "viz" / "ep.npz")
    assert data["times"].dtype == np.float32 and data["times"].tolist() == list(range(40))
    assert data["vm"].dtype == np.float16 and data["vm"].shape == (40, n)
    np.testing.assert_allclose(data["vm"], vm[:, surface["node_ids"]], atol=0.1)
    assert data["lat"].dtype == np.float32
    np.testing.assert_array_equal(np.isnan(data["lat"]), np.isnan(lat[surface["node_ids"]]))
    np.testing.assert_allclose(data["apd"][np.isfinite(data["apd"])], 200.0)
    k = len(data["trace_labels"])
    assert k == 6 and data["traces"].shape == (k, 40) and data["traces"].dtype == np.float32
    assert data["trace_times"].shape == (40,)
    assert data["trace_node_ids"].dtype == np.int64
    np.testing.assert_allclose(data["traces"], vm[:, data["trace_node_ids"]].T)
    assert set(data["trace_node_ids"]) <= set(surface["node_ids"])


def test_export_ep_writes_its_own_bundle_and_takes_tensors(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    simulator, points = _fake_simulator(tmp_path)
    simulator.nodes = torch.as_tensor(points)
    simulator.elems.Tt.data = torch.as_tensor(simulator.elems.Tt.data)
    simulator.elems.Tt.region = torch.as_tensor(simulator.elems.Tt.region)
    vm, lat = _wave(points)
    monkeypatch.setenv("TCS_RUN_DIR", str(tmp_path / "run"))
    with redirect_stdout(io.StringIO()):
        bundle = studio.export_ep(simulator, torch.as_tensor(vm), snapshot_interval=1.0,
                                  ATs=torch.as_tensor(lat))
    index = _index(tmp_path / "run" / "viz")  # written without an explicit bundle
    assert index["kind"] == "ep" and bundle.dir == tmp_path / "run" / "viz"
    data = _load(tmp_path / "run" / "viz" / "ep.npz")
    assert "rt" not in data and "apd" not in data and data["vm"].dtype == np.float16


def test_export_ecg_follows_the_bundle_spec(tmp_path):
    t = np.arange(500)
    ecg = {name: np.sin(t / (20.0 + i)) * (i + 1) for i, name in enumerate(studio.LEADS)}
    simulator, points = _fake_simulator(tmp_path)
    vm, lat = _wave(points)
    bundle = studio.Bundle("ecg", title="ECG", directory=tmp_path / "viz")
    with redirect_stdout(io.StringIO()):
        studio.export_ecg(ecg, 2.0, bundle, simulator=simulator, Vm=vm, ATs=lat,
                          snapshot_interval=1.0)
        bundle.write()
    index = _index(tmp_path / "viz")
    assert index["kind"] == "ecg" and set(index["datasets"]) == {"ecg", "surface", "ep"}
    data = _load(tmp_path / "viz" / "ecg.npz")
    assert data["t"].dtype == np.float32 and data["t"][-1] == 998.0
    assert data["leads"].dtype == np.float32 and data["leads"].shape == (12, 500)
    assert data["lead_names"].tolist() == list(studio.LEADS)
    np.testing.assert_allclose(data["leads"][1], ecg["II"], rtol=1e-6)
    labels = {row["label"] for row in index["summary"]}
    assert {"Duration", "Lead II peak-to-peak", "Total activation time"} <= labels
    with pytest.raises(KeyError):
        studio.export_ecg({"I": t}, 1.0, studio.Bundle("ecg", directory=tmp_path / "other"))


def test_export_mechanics_follows_the_bundle_spec(tmp_path):
    mesh = _HexMesh()
    frames = studio.FrameRecorder(studio.mechanics_surface(mesh), interval=5.0)
    for t in np.arange(0.0, 101.0, 2.5):
        if frames.due(t):
            u = np.zeros_like(mesh.points)
            u[:, 2] = 0.01 * t * mesh.points[:, 0]
            frames.add(t, displacement=u, stretch=1.0 + 0.001 * t * mesh.points[:, 0])
    log = studio.SolverLog()
    log.steps, log.values, log.newton = [1, 2, 3], [2.5, 5.0, 7.5], [4, 5, 4]
    report = SimpleNamespace(converged=True, load_steps=40, newton_iterations=180,
                             min_jacobian=0.97, wall_time=12.5)
    bundle = studio.Bundle("mechanics", title="Mech", directory=tmp_path / "viz")
    studio.export_mechanics(frames, bundle, frame_label="Time", frame_unit="ms", log=log,
                            report=report, n_dofs=mesh.n_dofs)
    with redirect_stdout(io.StringIO()):
        bundle.write()
    surface = _check_surface(tmp_path / "viz")
    n = len(surface["points"])
    data = _load(tmp_path / "viz" / "mechanics.npz")
    f = len(data["frames"])
    assert f == 21 and data["frames"].dtype == np.float32 and data["frames"][-1] == 100.0
    assert str(data["frame_label"]) == "Time" and str(data["frame_unit"]) == "ms"
    assert data["displacement"].dtype == np.float32 and data["displacement"].shape == (f, n, 3)
    assert data["field_stretch"].dtype == np.float16 and data["field_stretch"].shape == (f, n)
    assert data["conv_step"].dtype == np.int32 and data["conv_step"].tolist() == [1, 2, 3]
    assert data["conv_value"].dtype == np.float32
    assert data["conv_newton"].dtype == np.int32 and data["conv_newton"].tolist() == [4, 5, 4]
    summary = {row["label"]: row["value"] for row in _index(tmp_path / "viz")["summary"]}
    assert summary["Max displacement"] == pytest.approx(3.0)
    assert summary["Converged"] == "yes" and summary["Newton iterations"] == 180
    assert summary["Degrees of freedom"] == mesh.n_dofs


def test_export_em_follows_the_bundle_spec(tmp_path):
    mesh = _HexMesh()
    surface = studio.mechanics_surface(mesh)
    frames = studio.FrameRecorder(surface, interval=10.0)
    n_nodes = len(mesh.points)
    for t in np.arange(0.0, 1666.0, 0.5):
        if frames.due(t):
            phase = np.sin(2 * np.pi * t / 833.0)
            frames.add(t, displacement=np.full((n_nodes, 3), phase),
                       vm=np.full(n_nodes, -85.0 + 100.0 * max(phase, 0.0)),
                       tension=np.full(n_nodes, 40.0 * max(phase, 0.0)))
    t = np.arange(0.0, 1666.0, 1.0)
    volume = 100.0 + 40.0 * np.cos(2 * np.pi * t / 833.0)  # 60..140 mL
    rows = [{"t": ti, "p_lv": 8.0 + 112.0 * max(np.sin(2 * np.pi * ti / 833.0), 0.0),
             "V_lv": vi, "p_rv": 5.0, "V_rv": 90.0, "q_ao": 0.0} for ti, vi in zip(t, volume, strict=True)]
    bundle = studio.Bundle("em", title="EM", directory=tmp_path / "viz")
    studio.export_em(frames, rows, bundle, cycle_length=833.0)
    with redirect_stdout(io.StringIO()):
        bundle.write()
    index = _index(tmp_path / "viz")
    assert set(index["datasets"]) == {"surface", "em"}
    surface_data = _check_surface(tmp_path / "viz")
    n = len(surface_data["points"])
    data = _load(tmp_path / "viz" / "em.npz")
    f = len(data["times"])
    assert f == 167 and data["times"].dtype == np.float32
    assert data["displacement"].dtype == np.float32 and data["displacement"].shape == (f, n, 3)
    assert data["vm"].dtype == np.float16 and data["vm"].shape == (f, n)
    assert data["tension"].dtype == np.float16 and data["tension"].shape == (f, n)
    assert data["pv_t"].dtype == np.float32 and len(data["pv_t"]) == len(t)
    assert {"p_lv", "V_lv", "p_rv", "V_rv"} <= set(data) and "p_la" not in data
    assert "q_ao" not in data
    summary = {row["label"]: row["value"] for row in index["summary"]}
    assert summary["LV end-diastolic volume"] == pytest.approx(140.0, abs=0.1)
    assert summary["LV end-systolic volume"] == pytest.approx(60.0, abs=0.1)
    assert summary["LV ejection fraction"] == pytest.approx(100.0 * 80.0 / 140.0, abs=0.1)
    assert summary["Peak LV pressure"] == pytest.approx(120.0, abs=0.1)


# =============================================================================== ECG run
@pytest.mark.slow
@pytest.mark.skipif(not HAS_TORCHCOR, reason="TorchCor is not installed")
def test_ecg_runs_end_to_end_on_a_synthetic_torso(tmp_path, prepared_torso):
    """The whole ECG pipeline (EP, lead fields, 12 leads, bundle) on the synthetic torso."""
    from torchcor_studio.core.simulations import ecg
    from torchcor_studio.core.simulations.base import ScriptContext

    params = ecg.ECGParams()
    params.heart.duration = 30.0
    params.heart.dt = 0.02
    params.heart.heterogeneity = True
    params.conductor.r_tol = 1e-8
    params.outputs.plot = False
    run = tmp_path / "ecg-run"
    run.mkdir()
    ctx = ScriptContext(device="cpu", input_dir=str(prepared_torso), run_dir=str(run),
                        run_name="ECG on a synthetic torso")
    (run / "run.py").write_text(ecg.build_script(params, ctx), encoding="utf-8")
    for name in ("tcs_runtime.py", "tcs_studio.py"):
        shutil.copy(runtime_path(name), run / name)
    (run / "manifest.json").write_text(json.dumps({"kind": "ecg", "created_at": time.time()}))
    (run / "status.json").write_text(json.dumps({"state": "queued"}))
    subprocess.run([sys.executable, str(run / "tcs_runtime.py"), "launch", str(run)],
                   check=True, capture_output=True, timeout=60)
    deadline, status = time.time() + 600, {}
    while time.time() < deadline:
        status = json.loads((run / "status.json").read_text() or "{}")
        if status.get("state") not in ("queued", "running"):
            break
        time.sleep(1.0)
    log = (run / "run.log").read_text(errors="replace")
    assert status.get("state") == "completed", f"{status}\n{log[-4000:]}"
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert [e["id"] for e in events if e["type"] == "stage"] == [
        "setup", "data", "ep", "leadfield", "ecg", "export", "done"]
    assert log.count("CG iters =") == 9  # one reciprocal solve per electrode but the ground

    index = _index(run / "viz")
    assert index["kind"] == "ecg" and set(index["datasets"]) == {"ecg", "surface", "ep"}
    data = _load(run / "viz" / "ecg.npz")
    samples = int(params.heart.duration / params.heart.sample_interval) + 1
    assert data["leads"].shape == (12, samples) and np.isfinite(data["leads"]).all()
    assert data["lead_names"].tolist() == list(studio.LEADS)
    leads = dict(zip(studio.LEADS, data["leads"], strict=True))
    np.testing.assert_allclose(leads["II"], leads["I"] + leads["III"], atol=1e-5)  # Einthoven
    assert (run / "results" / "ecg.npz").exists()
