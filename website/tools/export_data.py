"""Export compact surface data for the website videos. Runs ON THE COMPUTE MACHINE.

It reads simulation results with that machine's TorchCor Python environment and writes one
compressed .npz to stdout (notes go to stderr), so nothing is written on the server:

    ssh HOST 'PYTHON - ep-run RUN_DIR' < export_data.py > data/ep_biventricle.npz
    ssh HOST 'PYTHON - ep-surface RESULT_DIR' < export_data.py > data/ep_heart.npz
    ssh HOST 'PYTHON - em PV_CSV VTK_DIR [VTK_DIR ...]' < export_data.py > data/em_keyframes.npz

ep-run      a TorchCor Lighting run folder: results/Vm.pt sliced to the surface points of
            viz/surface.npz; every frame to 100 ms (the activation), every 2nd to 440 ms.
ep-surface  a folder whose surface.npz holds points, triangles, vm and snapshot_ms
            (the manuscript's whole-heart sinus beat); every 2nd frame.
em          heart_NNNN.vtk surface keyframes (points, vm, tension, displacement), NNNN in
            units of 10 ms, plus the run's pv.csv; keyframes found in several folders are
            merged by index.
"""
import csv
import glob
import io
import json
import os
import sys

import numpy as np


def note(*args):
    print(*args, file=sys.stderr)


def ep_run(run_dir):
    import torch

    surface = np.load(os.path.join(run_dir, "viz", "surface.npz"), allow_pickle=False)
    node_ids = surface["node_ids"]
    snapshot = 1.0
    try:
        with open(os.path.join(run_dir, "manifest.json")) as f:
            snapshot = float(json.load(f)["params"]["timing"]["snapshot_interval"])
    except (OSError, KeyError, ValueError):
        pass
    vm = torch.load(os.path.join(run_dir, "results", "Vm.pt"), map_location="cpu")
    if vm.shape[-1] != len(node_ids) and vm.shape[0] > vm.shape[-1]:
        vm = vm.T
    last = vm.shape[0] - 1
    frames = [k for k in list(range(0, 101)) + list(range(102, 441, 2)) if k <= last]
    sub = vm[frames][:, torch.as_tensor(node_ids, dtype=torch.long)].numpy().astype(np.float16)
    note("ep-run:", tuple(vm.shape), "->", sub.shape)
    return dict(points=surface["points"], triangles=surface["triangles"],
                times=(np.asarray(frames) * snapshot).astype(np.float32), vm=sub)


def ep_surface(result_dir):
    s = np.load(os.path.join(result_dir, "surface.npz"), allow_pickle=False)
    keep = np.arange(0, s["vm"].shape[0], 2)
    note("ep-surface:", s["vm"].shape, "->", len(keep), "frames")
    return dict(points=s["points"], triangles=s["triangles"].astype(np.int32),
                times=(keep * float(s["snapshot_ms"])).astype(np.float32),
                vm=s["vm"][keep].astype(np.float16))


def em(pv_csv, *vtk_dirs):
    import pyvista as pv

    files = {}
    for d in vtk_dirs:
        for f in sorted(glob.glob(os.path.join(d, "heart_*.vtk"))):
            files.setdefault(int(os.path.basename(f)[6:-4]), f)
    idx = sorted(files)
    frames = [pv.read(files[i]) for i in idx]
    note("em keyframes:", idx)
    tri = frames[0].cells_dict.get(5)                  # VTK_TRIANGLE
    if tri is None:
        tri = frames[0].extract_surface().faces.reshape(-1, 4)[:, 1:]
    out = dict(points=np.asarray(frames[0].points, np.float32), triangles=np.asarray(tri, np.int32),
               frame_index=np.asarray(idx, np.int32),
               deformed_points=np.stack([np.asarray(g.points, np.float32) for g in frames]),
               displacement=np.stack([np.asarray(g["displacement"], np.float32) for g in frames]),
               tension=np.stack([np.asarray(g["tension"], np.float16) for g in frames]),
               vm=np.stack([np.asarray(g["vm"], np.float16) for g in frames]))
    with open(pv_csv) as f:
        rows = list(csv.DictReader(f))
    for col in ("t", "p_lv", "p_rv", "p_la", "p_ra", "V_lv", "V_rv", "V_la", "V_ra"):
        out[f"pv_{col}"] = np.asarray([float(r[col]) for r in rows], np.float32)
    note("pv rows:", len(rows))
    return out


if __name__ == "__main__":
    command, args = sys.argv[1], [os.path.expanduser(a) for a in sys.argv[2:]]
    data = {"ep-run": ep_run, "ep-surface": ep_surface, "em": em}[command](*args)
    buf = io.BytesIO()
    np.savez_compressed(buf, **data)
    sys.stdout.buffer.write(buf.getvalue())
    note("bytes:", buf.tell())
