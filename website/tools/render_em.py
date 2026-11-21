"""Electromechanics card: a four-chamber heart beating, coloured by active tension (brand scale).

    python render_em.py        # data/em_keyframes.npz -> em-heart.mp4 (+ poster), data/em_pv.json

The run kept surface keyframes over its last beat only (3390-4130 ms) plus the full pv.csv;
export keyframes and pv.csv from the same run. Motion between keyframes is a smooth, periodic
monotone-cubic (PCHIP) interpolation of those simulated states; the pressure-volume loops
come straight from pv.csv. Run update_page.py afterwards to put em_pv.json into index.html.
"""
import json

import numpy as np
from scipy.interpolate import PchipInterpolator

import render as r

CYCLE = 833.0                                            # ms, the run's heart period
START = 4 * CYCLE                                        # 3332 ms: ventricles fire, the last beat
CLIM = (0.0, 90.0)                                       # active tension, kPa

d = np.load(r.DATA / "em_keyframes.npz")
TRI = d["triangles"]
KT = d["frame_index"].astype(float) * 10.0               # keyframe times, ms
POS = d["deformed_points"].astype(np.float64)
TEN = d["tension"].astype(np.float32)
T0 = START

# periodic padding: the last keyframe one cycle earlier, the first two one cycle later
kt = np.concatenate([[KT[-1] - CYCLE], KT, KT[:2] + CYCLE])
POS_AT = PchipInterpolator(kt, np.concatenate([POS[-1:], POS, POS[:2]]), axis=0)
TEN_AT = PchipInterpolator(kt, np.concatenate([TEN[-1:], TEN, TEN[:2]]), axis=0)


def mesh_at(t):
    m = r.polydata(POS_AT(t), TRI)
    m["tension"] = np.clip(TEN_AT(t), 0, None).astype(np.float32)
    return m.compute_normals(split_vertices=False, auto_orient_normals=True)


def up_vector():
    """Long axis pointing at the atria: they carry little tension while the ventricles contract."""
    pts = POS[0]
    systole = TEN[int(np.argmax(np.median(TEN, axis=1)))]
    return r.orient(r.principal_axis(pts), pts, pts[systole < np.percentile(systole, 25)].mean(0))


def video(az=75.0, el=12.0, frames=136, size=(520, 600), swing=8.0, dist=3.3):
    up = up_vector()
    ref = mesh_at(T0)
    center, radius = np.asarray(ref.center), ref.length / 2
    times = T0 + CYCLE * np.arange(frames) / frames
    mesh = mesh_at(times[0])
    p = r.studio(size)
    r.add_surface(p, mesh, "tension", r.lut(r.BRAND, CLIM), CLIM, specular=0.4)
    shift, dist = r.autoframe(p, center, radius, up, az, el, swing, dist)
    enc = r.Encoder(r.OUT / "em-heart.mp4", size)
    poster = None
    for i, t in enumerate(times):
        mesh.copy_from(mesh_at(t))
        r.camera(p, center, radius, up, az + swing * np.sin(2 * np.pi * i / frames), el, dist, shift=shift)
        img = r.grab(p)
        enc.write(img)
        if poster is None and t >= 3540:
            poster = img
    enc.close()
    p.close()
    r.save_poster(poster, "em-heart-poster.webp")

    # Pressure-volume data for the page: each loop (every 4 ms) and the values at every video
    # frame, so main.js can trace the loops in step with the video.
    pv_t = d["pv_t"].astype(np.float64)
    col = lambda name: d[f"pv_{name}"].astype(np.float64)
    beat = (pv_t >= T0) & (pv_t < T0 + CYCLE)
    rnd = lambda a: [round(float(x), 1) for x in a]
    at_frames = lambda name: rnd(np.interp(times, pv_t, col(name)))
    pv = {
        "lv": {"V": rnd(col("V_lv")[beat][::4]), "p": rnd(col("p_lv")[beat][::4])},
        "rv": {"V": rnd(col("V_rv")[beat][::4]), "p": rnd(col("p_rv")[beat][::4])},
        "frames": {"lvV": at_frames("V_lv"), "lvp": at_frames("p_lv"),
                   "rvV": at_frames("V_rv"), "rvp": at_frames("p_rv")},
    }
    (r.DATA / "em_pv.json").write_text(json.dumps(pv, separators=(",", ":")))
    print("em-heart frames", frames)


if __name__ == "__main__":
    video()
