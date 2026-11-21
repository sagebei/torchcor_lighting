"""Mechanics card: the idealised left ventricle beating through one cycle.

    python render_lv.py        # data/lv_b1.npz -> mech-lv.mp4 (+ poster)

The input is the TorchCor benchmark result (points in m, 10-node tetrahedra, displacement u
for 101 frames over 1 s). Quadratic cells are rendered through VTK's curved-surface
subdivision; colours show the displacement magnitude on the brand scale (render.BRAND).
"""
import numpy as np
import pyvista as pv

import render as r

CLIM = (0.0, 40.0)            # displacement, mm
UP = (1.0, 0.0, 0.0)          # the benchmark's long axis is x, with the open base at the top
VTK_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]   # mid-edge node order in VTK


def vtk_order(points, cells):
    """Permutation of a 10-node tet's columns into VTK order, found from the geometry."""
    order = [0, 1, 2, 3]
    for a, b in VTK_EDGES:
        mid = 0.5 * (points[cells[:, a]] + points[cells[:, b]])
        err = [np.abs(points[cells[:, k]] - mid).max() for k in range(4, 10)]
        order.append(4 + int(np.argmin(err)))
    return np.array(order)


d = np.load(r.DATA / "lv_b1.npz")
P = d["points"].astype(np.float64)
U = d["u"].astype(np.float64)
CELLS = d["cells"][:, vtk_order(P, d["cells"])]
GRID = pv.UnstructuredGrid({pv.CellType.QUADRATIC_TETRA: CELLS}, P)


def surface(f):
    """Smooth surface at (fractional) frame f, coloured by |u| in mm."""
    i0 = int(np.floor(f))
    i1, w = min(i0 + 1, len(U) - 1), f - i0
    disp = (1 - w) * U[i0] + w * U[i1]
    g = GRID.copy()
    g.points = P + disp
    g["disp"] = np.linalg.norm(disp, axis=1) * 1000.0
    s = g.extract_surface(nonlinear_subdivision=2, algorithm="dataset_surface")
    return s.compute_normals(split_vertices=False, auto_orient_normals=True)


def video(az=30.0, el=24.0, size=(520, 600), swing=6.0, fade=14, dist=3.75):
    schedule = list(np.arange(6.0, 84.0, 0.5))    # the beat, at half-frame steps
    ref = surface(0)
    center, radius = np.asarray(ref.center), ref.length / 2
    mesh = surface(schedule[0])
    p = r.studio(size)
    r.add_surface(p, mesh, "disp", r.lut(r.BRAND, CLIM), CLIM, specular=0.4)
    enc = r.Encoder(r.OUT / "mech-lv.mp4", size)
    total = len(schedule) + fade
    angle = lambda i: az + swing * np.sin(2 * np.pi * i / total)
    first, poster = surface(schedule[0]), None
    for i, f in enumerate(schedule):
        mesh.copy_from(surface(f))
        r.camera(p, center, radius, UP, angle(i), el, dist)
        img = r.grab(p)
        enc.write(img)
        if poster is None and f >= 40:
            poster = img
    last = surface(schedule[-1])
    for j in range(fade):
        r.camera(p, center, radius, UP, angle(len(schedule) + j), el, dist)
        mesh.copy_from(last)
        a = r.grab(p).astype(np.float32)
        mesh.copy_from(first)
        b = r.grab(p).astype(np.float32)
        w = r.smoothstep((j + 1) / (fade + 1))
        enc.write(np.ascontiguousarray((a * (1 - w) + b * w).round().astype(np.uint8)))
    enc.close()
    p.close()
    r.save_poster(poster, "mech-lv-poster.webp")
    print("mech-lv frames", total)


if __name__ == "__main__":
    video()
