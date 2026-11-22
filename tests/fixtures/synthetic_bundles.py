"""Synthetic but physiologically plausible visualisation bundles.

Each ``write_*_bundle(folder, ...)`` writes a complete bundle (``index.json`` plus its
``.npz`` datasets, exactly as :class:`torchcor_studio.runtime.tcs_studio.Bundle` writes
them for real runs) into ``folder`` and returns that folder. Defaults are small so tests
stay fast; pass a higher ``level`` / finer ``dt`` for showcase-quality data.

* EP — a biventricular-looking closed surface, an activation wave from an apical septal
  site, action potentials (−85 … +30 mV), LAT, RT, APD and four probe traces.
* ECG — 12 leads from a rotating cardiac dipole (QRS + T), 600 ms at 1 ms.
* Mechanics — a cantilever beam bending under 10 load steps, with von Mises stress.
* EM — an ellipsoidal ventricle contracting and twisting over an 800 ms beat, with
  voltage, active tension and four-chamber pressure–volume traces.
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path

import numpy as np

from torchcor_studio.runtime.tcs_studio import Bundle

V_REST = -85.0


# =============================================================================== geometry
def icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere: ``10·4^level + 2`` points, outward triangles."""
    t = (1 + 5 ** 0.5) / 2
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
                  (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
                  (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
                  (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)], np.int64)
    for _ in range(level):
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        mids = v[unique[:, 0]] + v[unique[:, 1]]
        mids /= np.linalg.norm(mids, axis=1, keepdims=True)
        mid = len(v) + inverse.reshape(3, -1).T
        v = np.vstack([v, mids])
        a, b, c = f.T
        ab, bc, ca = mid.T
        f = np.vstack([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                       np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
    return v, f


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    s = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return s * s * (3 - 2 * s)


def _ellipsoid_field(p: np.ndarray, center, radii, angle: float = 0.0) -> np.ndarray:
    """Approximate signed distance (mm) to an ellipsoid rotated by ``angle`` about z."""
    c, s = math.cos(angle), math.sin(angle)
    q = p - np.asarray(center, float)
    local = np.column_stack([c * q[:, 0] + s * q[:, 1], -s * q[:, 0] + c * q[:, 1], q[:, 2]])
    radii = np.asarray(radii, float)
    return (np.linalg.norm(local / radii, axis=1) - 1.0) * radii.min()


def _smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    h = np.clip(k - np.abs(a - b), 0.0, None) / k
    return np.minimum(a, b) - h * h * k / 4.0


def heart_surface(level: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """A closed biventricular surface in mm: LV and RV lobes, a groove, a flat valve plane.

    The shape is the smooth union of two ellipsoids cut by the base plane; icosphere
    directions are ray-cast onto it, so the triangulation stays clean.
    """
    unit, tri = icosphere(level)
    rv_angle = math.radians(40.0)
    rv_center = (19.0 * math.cos(rv_angle), 19.0 * math.sin(rv_angle), 5.0)

    def field(p: np.ndarray) -> np.ndarray:
        taper = 0.58 + 0.42 * _smoothstep(-52.0, 20.0, p[:, 2])  # conical toward the apex
        q = np.column_stack([p[:, 0] / taper, p[:, 1] / taper, p[:, 2]])
        lv = _ellipsoid_field(q, (0.0, 0.0, -4.0), (29.0, 28.0, 52.0))
        rv = _ellipsoid_field(q, rv_center, (17.0, 33.0, 40.0), rv_angle)
        heart = _smooth_min(lv, rv, 9.0)
        return -_smooth_min(-heart, -(p[:, 2] - 24.0), 8.0)  # smooth cut at the valve plane

    lo, hi = np.zeros(len(unit)), np.full(len(unit), 90.0)
    for _ in range(36):  # bisection along each direction from the LV centre
        mid = 0.5 * (lo + hi)
        inside = field(unit * mid[:, None]) < 0
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    pts = unit * (0.5 * (lo + hi))[:, None]
    pts *= 1.0 + 0.006 * np.sin(3 * np.arctan2(pts[:, 1], pts[:, 0]) + pts[:, 2] / 12.0)[:, None]
    # tilt as the heart sits in the chest: apex forward, down and to the left
    tilt, yaw = math.radians(24.0), math.radians(-20.0)
    rx = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return (pts @ (rz @ rx).T).astype(np.float32), tri.astype(np.int32)


def ventricle_surface(level: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A truncated prolate ellipsoid (LV-like) in mm, apex at ``z = -48``."""
    unit, tri = icosphere(level)
    x, y, z = unit.T
    zz = np.where(z > 0.55, 0.55 + (z - 0.55) * 0.22, z)
    taper = 0.78 + 0.22 * _smoothstep(-1.0, 0.5, z)
    pts = np.column_stack([25.0 * taper * x, 25.0 * taper * y, 48.0 * zz])
    return pts.astype(np.float32), tri.astype(np.int32)


def beam_surface(nx: int = 40, ny: int = 6, nz: int = 6, length: float = 100.0,
                 width: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Closed surface of a ``length × width × width`` box along x (mm), outward triangles."""
    shape = (nx + 1, ny + 1, nz + 1)
    ids = np.full(shape, -1, np.int64)
    gx, gy, gz = np.meshgrid(np.linspace(0, length, nx + 1), np.linspace(-width / 2, width / 2, ny + 1),
                             np.linspace(-width / 2, width / 2, nz + 1), indexing="ij")
    on_surface = np.zeros(shape, bool)
    on_surface[[0, -1], :, :] = on_surface[:, [0, -1], :] = on_surface[:, :, [0, -1]] = True
    ids[on_surface] = np.arange(on_surface.sum())
    points = np.column_stack([gx[on_surface], gy[on_surface], gz[on_surface]])

    faces = []
    for axis, n in ((0, nx), (1, ny), (2, nz)):
        for end in (0, n):
            face = np.take(ids, end, axis=axis)
            a, b, c, d = face[:-1, :-1], face[1:, :-1], face[1:, 1:], face[:-1, 1:]
            faces += [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
    tri = np.vstack(faces)
    # the box is convex: orient every triangle away from its centre
    p = points[tri]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = np.einsum("ij,ij->i", normal, p.mean(axis=1) - points.mean(axis=0)) < 0
    tri[inward] = tri[inward][:, ::-1]
    return points.astype(np.float32), tri.astype(np.int32)


def _orient_outward(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Flip every triangle if the surface is wound inward (keeps fixtures honest)."""
    p = points.astype(np.float64)
    n = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    outward = np.einsum("ij,ij->i", n, p[tri].mean(axis=1) - p.mean(axis=0)).sum() > 0
    return tri if outward else tri[:, ::-1].copy()


# =============================================================================== physiology
def action_potential(tau: np.ndarray, apd90: np.ndarray | float) -> np.ndarray:
    """Ventricular AP (mV) at time ``tau`` after activation: spike, dome, repolarisation."""
    tau = np.asarray(tau, dtype=np.float32)
    up = 1.0 / (1.0 + np.exp(np.clip(-tau / 0.35, -60.0, 60.0)))
    after = np.maximum(tau, 0.0)
    plateau = 10.0 * np.exp(-after / 110.0) + 35.0 * np.exp(-after / 3.0)
    repol = 1.0 / (1.0 + np.exp(np.clip((tau - (apd90 - 24.0)) / 11.0, -60.0, 60.0)))
    return (V_REST + (plateau - V_REST) * up * repol).astype(np.float32)


def active_tension(tau: np.ndarray, peak: float = 60.0, time_to_peak: float = 140.0) -> np.ndarray:
    """Active tension (kPa) after electromechanical delay; zero before activation."""
    s = np.maximum(np.asarray(tau, dtype=np.float32) - 15.0, 0.0) / time_to_peak
    return (peak * s ** 2 * np.exp(2.0 * (1.0 - s))).astype(np.float32)


def _activation(points: np.ndarray, site: np.ndarray, velocity: float, rng: np.random.Generator) -> np.ndarray:
    d = points - site
    d[:, 2] *= 0.85  # faster apex-to-base conduction
    lat = np.linalg.norm(d, axis=1) / velocity
    phase = rng.uniform(0, 2 * math.pi, 2)
    lat += 2.0 * np.sin(points[:, 0] / 14.0 + phase[0]) * np.cos(points[:, 1] / 17.0 + phase[1])
    return np.maximum(lat - lat.min(), 0.0).astype(np.float32)


def _node_ids(n: int, rng: np.random.Generator) -> np.ndarray:
    """Surface points are a subset of a (pretend) volume mesh: sorted, sparse ids."""
    return np.sort(rng.choice(6 * n, size=n, replace=False)).astype(np.int64)


def _write(bundle: Bundle) -> Path:
    with contextlib.redirect_stdout(io.StringIO()):  # silence the ::tcs:: artifact marker
        bundle.write()
    return bundle.dir


# =============================================================================== bundles
def _ep_arrays(level: int, dt: float, duration: float, seed: int) -> tuple[dict, dict, dict]:
    """Surface arrays, ``ep.npz`` arrays and headline numbers of a synthetic EP run."""
    rng = np.random.default_rng(seed)
    points, tri = heart_surface(level)
    tri = _orient_outward(points, tri)
    n = len(points)
    node_ids = _node_ids(n, rng)

    apex = points[np.argmin(points[:, 2])]
    septal_dir = np.array([math.cos(math.radians(30)), math.sin(math.radians(30)), 0.9])
    site = points[np.argmin(np.linalg.norm(points - (apex + 14 * septal_dir), axis=1))]
    lat = _activation(points, site, velocity=0.75, rng=rng)
    apd = (285.0 - 0.32 * lat + 6.0 * np.sin(points[:, 2] / 14.0)).astype(np.float32)

    times = np.arange(0.0, duration + 1e-6, dt, dtype=np.float32)
    vm = action_potential(times[:, None] - lat[None, :], apd[None, :]).astype(np.float16)

    probes = {
        "Apex": int(np.argmin(points[:, 2])),
        "Septum": int(np.argmin(np.linalg.norm(points - site, axis=1))),
        "LV free wall": int(np.argmax(points @ np.array([-0.9, -0.4, 0.15]))),
        "RV free wall": int(np.argmax(points @ np.array([0.25, 0.95, 0.1]))),
    }
    idx = np.array(list(probes.values()))
    trace_times = np.arange(0.0, duration + 1e-6, 0.5, dtype=np.float32)
    traces = action_potential(trace_times[None, :] - lat[idx, None], apd[idx, None])

    surface = dict(points=points, triangles=tri, node_ids=node_ids)
    ep = dict(times=times, vm=vm, lat=lat, rt=(lat + apd).astype(np.float32), apd=apd,
              trace_times=trace_times, traces=traces, trace_labels=np.array(list(probes)),
              trace_node_ids=node_ids[idx])
    stats = dict(total=float(lat.max() - lat.min()), apd_mean=float(apd.mean()),
                 apd_spread=float(apd.max() - apd.min()), points=n)
    return surface, ep, stats


def write_ep_bundle(folder: Path, *, level: int = 3, dt: float = 10.0, duration: float = 400.0,
                    title: str = "Biventricular activation", seed: int = 7) -> Path:
    """EP bundle: activation from an apical septal site, APs, LAT/RT/APD, probe traces."""
    surface, ep, stats = _ep_arrays(level, dt, duration, seed)
    bundle = Bundle("ep", title, directory=folder)
    bundle.add_surface(**surface)
    bundle.add("ep", **ep)
    bundle.stat("Total activation", stats["total"], "ms", "First to last activated point")
    bundle.stat("Mean APD₉₀", stats["apd_mean"], "ms")
    bundle.stat("APD dispersion", stats["apd_spread"], "ms", "Max − min APD₉₀")
    bundle.stat("Activated", 100.0, "%")
    bundle.stat("Surface points", stats["points"])
    bundle.stat("Solve time", 41.7, "s", "cuda:0 · NVIDIA A100")
    bundle.note("Synthetic data generated for TorchCor Lighting tests.")
    return _write(bundle)


def ecg_leads(duration: float = 600.0, dt: float = 1.0, qrs_onset: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """12 leads (mV) projected from a rotating cardiac dipole: septal, free wall, basal, T."""
    t = np.arange(0.0, duration + 1e-6, dt, dtype=np.float64)
    tau = t - qrs_onset

    def unit(v):
        v = np.asarray(v, float)
        return v / np.linalg.norm(v)

    def pulse(center, width_up, width_down=None):
        width = np.where(tau < center, width_up, width_down or width_up)
        return np.exp(-0.5 * ((tau - center) / width) ** 2)

    # dipole in (left, inferior, anterior) coordinates
    dipole = (0.28 * pulse(12, 6)[:, None] * unit([-0.6, 0.2, 0.75])
              + 1.65 * pulse(44, 11)[:, None] * unit([0.75, 0.6, -0.35])
              + 0.45 * pulse(72, 9)[:, None] * unit([-0.3, -0.55, -0.75])
              + 0.42 * pulse(300, 52, 32)[:, None] * unit([0.62, 0.55, 0.32]))
    lead_i = dipole @ np.array([1.0, 0.0, 0.0])
    lead_ii = dipole @ np.array([0.5, math.sqrt(3) / 2, 0.0])
    limb = [lead_i, lead_ii, lead_ii - lead_i, -(lead_i + lead_ii) / 2, lead_i - lead_ii / 2, lead_ii - lead_i / 2]
    precordial = []
    for angle, inferior in ((115, 0.0), (95, 0.0), (75, 0.1), (55, 0.2), (30, 0.2), (5, 0.2)):
        a = math.radians(angle)
        precordial.append(1.4 * dipole @ unit([math.cos(a), inferior, math.sin(a)]))
    return t.astype(np.float32), np.vstack(limb + precordial).astype(np.float32)


def write_ecg_bundle(folder: Path, *, with_heart: bool = True, level: int = 3,
                     title: str = "12-lead ECG") -> Path:
    """ECG bundle; with ``with_heart`` it also carries the EP surface + fields (Heart tab)."""
    t, leads = ecg_leads()
    bundle = Bundle("ecg", title, directory=folder)
    if with_heart:
        surface, ep, _stats = _ep_arrays(level, dt=10.0, duration=400.0, seed=7)
        bundle.add_surface(**surface)
        bundle.add("ep", **ep)
    bundle.add("ecg", t=t, leads=leads, lead_names=np.array(
        ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]))
    bundle.stat("QRS duration", 94.0, "ms")
    bundle.stat("QT interval", 388.0, "ms")
    bundle.stat("Electrical axis", 41.0, "°", "Frontal plane")
    bundle.stat("Peak R", float(leads.max()), "mV", "Lead V5")
    bundle.note("Leads computed with the lead-field method from the simulated heart.")
    return _write(bundle)


def write_mechanics_bundle(folder: Path, *, nx: int = 40, ny: int = 6, steps: int = 10,
                           title: str = "Cantilever beam") -> Path:
    """Mechanics bundle: a 100 mm cantilever bending down under a growing tip load."""
    length, width, tip = 100.0, 10.0, 28.0
    points, tri = beam_surface(nx, ny, ny, length, width)
    tri = _orient_outward(points, tri)
    rng = np.random.default_rng(3)
    node_ids = _node_ids(len(points), rng)
    x, z = points[:, 0].astype(np.float64), points[:, 2].astype(np.float64)

    lam = np.linspace(1.0 / steps, 1.0, steps).astype(np.float32)
    shape = x ** 2 * (3 * length - x) / (2 * length ** 3)  # tip-normalised deflection
    slope = (6 * length * x - 3 * x ** 2) / (2 * length ** 3)
    grid = np.linspace(0, length, 512)
    slope_grid = (6 * length * grid - 3 * grid ** 2) / (2 * length ** 3)
    shortening = np.interp(x, grid, np.concatenate([[0], np.cumsum(0.5 * (slope_grid[1:] ** 2 + slope_grid[:-1] ** 2)
                                                                     * np.diff(grid))]))
    disp = np.zeros((steps, len(points), 3), np.float32)
    von_mises = np.zeros((steps, len(points)), np.float32)
    strain = np.zeros((steps, len(points)), np.float32)
    inertia = width ** 4 / 12
    for i, s in enumerate(lam):
        delta = tip * float(s) ** 1.05
        disp[i, :, 2] = -delta * shape
        disp[i, :, 0] = z * delta * slope - 0.5 * delta ** 2 * shortening
        moment = float(s) * 1.0 * (length - x)
        bending = np.abs(moment * z / inertia) * 60.0
        shear = float(s) * 18.0 * (1 - (2 * z / width) ** 2)
        von_mises[i] = np.sqrt(bending ** 2 + 3 * shear ** 2)
        strain[i] = 100 * np.abs(z) * float(s) * (length - x) / length * 0.012
    newton = np.array([3, 3, 4, 4, 4, 5, 5, 5, 6, 6] + [6] * max(0, steps - 10), np.int32)[:steps]

    bundle = Bundle("mechanics", title, directory=folder)
    bundle.add_surface(points, tri, node_ids)
    bundle.add("mechanics", frames=lam, frame_label=np.array("Load factor λ"), frame_unit=np.array(""),
               displacement=disp, field_von_mises=von_mises.astype(np.float16),
               field_strain=strain.astype(np.float16), conv_step=np.arange(1, steps + 1, dtype=np.int32),
               conv_value=lam, conv_newton=newton)
    bundle.stat("Max displacement", float(np.linalg.norm(disp[-1], axis=1).max()), "mm", "At full load")
    bundle.stat("Load steps", steps)
    bundle.stat("Newton iterations", int(newton.sum()), "", "Across all load steps")
    bundle.stat("Peak von Mises", float(von_mises[-1].max()), "kPa")
    return _write(bundle)


def circulation(pv_t: np.ndarray) -> dict[str, np.ndarray]:
    """Four-chamber pressures (mmHg) and volumes (mL) over one 800 ms beat."""
    t = np.mod(pv_t.astype(np.float64), 800.0)

    def ventricle(edv, esv, p_ed, p_iso, p_peak, p_es, p_min):
        s_ej = np.clip((t - 40) / 260, 0, 1)
        ejected = 1 - (1 - s_ej) ** 2.1
        fill_e = _smoothstep(380, 540, t) * 0.72
        fill_slow = _smoothstep(540, 690, t) * 0.08
        fill_a = _smoothstep(690, 790, t) * 0.20
        volume = np.where(t < 300, edv - (edv - esv) * ejected, esv + (edv - esv) * (fill_e + fill_slow + fill_a))
        ivc = p_ed + (p_iso - p_ed) * _smoothstep(0, 40, t)
        ejection = p_iso + (p_peak - p_iso) * np.sin(np.pi * s_ej ** 0.8) ** 0.75 + (p_es - p_iso) * s_ej ** 3
        ivr = p_min + (p_es - p_min) * np.exp(-np.clip(t - 300, 0, None) / 14.0)
        filling = p_min + (p_ed - p_min) * ((volume - esv) / (edv - esv)) ** 2 + 1.2 * np.exp(-((t - 745) / 22) ** 2)
        pressure = np.select([t < 40, t < 300, t < 380], [ivc, ejection, ivr], filling)
        return pressure, volume

    def atrium(v_min, v_max, p_base, v_wave, a_wave):
        fill = _smoothstep(40, 380, t)
        empty_e = _smoothstep(380, 540, t) * 0.55
        empty_a = _smoothstep(690, 790, t) * 0.45
        volume = v_min + (v_max - v_min) * np.where(t < 380, fill, 1 - empty_e - empty_a)
        pressure = (p_base + v_wave * np.exp(-((t - 360) / 45) ** 2) + a_wave * np.exp(-((t - 730) / 30) ** 2)
                    + 0.8 * _smoothstep(0, 40, t) * (1 - _smoothstep(40, 120, t)))
        return pressure, volume

    p_lv, v_lv = ventricle(120.0, 50.0, 9.0, 80.0, 118.0, 98.0, 4.0)
    p_rv, v_rv = ventricle(132.0, 62.0, 4.0, 11.0, 25.0, 19.0, 2.0)
    p_la, v_la = atrium(38.0, 74.0, 6.0, 6.0, 4.0)
    p_ra, v_ra = atrium(42.0, 80.0, 2.5, 3.5, 3.0)
    return {k: v.astype(np.float32) for k, v in dict(
        p_lv=p_lv, V_lv=v_lv, p_rv=p_rv, V_rv=v_rv, p_la=p_la, V_la=v_la, p_ra=p_ra, V_ra=v_ra).items()}


def write_em_bundle(folder: Path, *, level: int = 3, dt: float = 10.0, duration: float = 800.0,
                    title: str = "Beating left ventricle", summary_metrics: bool = False,
                    cycle_length: float | None = None) -> Path:
    """EM bundle: a contracting, twisting ventricle with voltage, tension and PV loops.

    The circulation repeats every 800 ms: a ``duration`` of 2400 ms holds three beats.
    ``cycle_length`` is stored as ``pv_cycle`` (bundles written before it have none).
    The summary deliberately omits EDV/ESV/EF unless ``summary_metrics`` is set, so the
    view has to compute them from the circulation traces.
    """
    rng = np.random.default_rng(11)
    points, tri = ventricle_surface(level)
    tri = _orient_outward(points, tri)
    n = len(points)
    node_ids = _node_ids(n, rng)

    times = np.arange(0.0, duration + 1e-6, dt, dtype=np.float32)
    pv_t = np.arange(0.0, duration + 1e-6, 2.0, dtype=np.float32)
    circ = circulation(pv_t)
    v_lv = np.interp(times, pv_t, circ["V_lv"])
    contraction = (v_lv.max() - v_lv) / (v_lv.max() - v_lv.min())

    apex_z, top_z = float(points[:, 2].min()), float(points[:, 2].max())
    height = (points[:, 2] - apex_z) / (top_z - apex_z)
    disp = np.zeros((len(times), n, 3), np.float32)
    for i, c in enumerate(contraction):
        radial = 1.0 - 0.13 * c * (0.6 + 0.4 * height)
        longitudinal = 1.0 - 0.11 * c
        angle = np.radians(9.0) * c * (2 * height - 1.0) * -1.0
        x, y = points[:, 0] * radial, points[:, 1] * radial
        cos, sin = np.cos(angle), np.sin(angle)
        moved = np.column_stack([cos * x - sin * y, sin * x + cos * y,
                                 apex_z + (points[:, 2] - apex_z) * longitudinal])
        disp[i] = moved - points

    site = points[np.argmin(points[:, 2])]
    lat = _activation(points, site, velocity=0.7, rng=rng)
    apd = (270.0 - 0.3 * lat).astype(np.float32)
    tau = times[:, None] - lat[None, :]
    vm = action_potential(tau, apd[None, :]).astype(np.float16)
    relax = np.exp(np.minimum((tau - apd[None, :] - 40.0) / 25.0, 50.0))  # capped: no overflow on long runs
    tension = (active_tension(tau) / (1.0 + relax)).astype(np.float16)

    bundle = Bundle("em", title, directory=folder)
    bundle.add_surface(points, tri, node_ids)
    cycle = {} if cycle_length is None else {"pv_cycle": np.float32(cycle_length)}
    bundle.add("em", times=times, displacement=disp, vm=vm, tension=tension, pv_t=pv_t, **circ, **cycle)
    if summary_metrics:
        edv, esv = float(circ["V_lv"].max()), float(circ["V_lv"].min())
        bundle.stat("EDV", edv, "mL")
        bundle.stat("ESV", esv, "mL")
        bundle.stat("Ejection fraction", 100 * (edv - esv) / edv, "%")
    bundle.stat("Peak tension", float(tension.astype(np.float32).max()), "kPa")
    bundle.stat("Beat duration", duration, "ms")
    bundle.note("Synthetic electromechanics: displacement is prescribed, not solved.")
    return _write(bundle)


def write_all(folder: Path) -> dict[str, Path]:
    """One (small) bundle of every kind under ``folder/<kind>``."""
    folder = Path(folder)
    return {
        "ep": write_ep_bundle(folder / "ep"),
        "ecg": write_ecg_bundle(folder / "ecg"),
        "mechanics": write_mechanics_bundle(folder / "mechanics"),
        "em": write_em_bundle(folder / "em"),
    }
