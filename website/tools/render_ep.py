"""Electrophysiology card: a sinus beat on a four-chamber heart and a paced biventricle.

    python render_ep.py            # both
    python render_ep.py heart      # data/ep_heart.npz       -> ep-heart.mp4 (+ poster)
    python render_ep.py biv        # data/ep_biventricle.npz -> ep-biventricle.mp4 (+ poster)

Colours: the website's brand scale over -100..50 mV (render.BRAND).
"""
import sys

import numpy as np

import render as r

CLIM = (-100.0, 50.0)


def ep_video(data, up, az, el, schedule, out, poster_t, size=(520, 600), swing=9.0,
             hold=8, fade=16, dist=3.25):
    """Loop: play `schedule` (ms), hold, then cross-fade back to the first frame."""
    times, vm = data["times"], data["vm"]
    mesh = r.polydata(data["points"], data["triangles"]).compute_normals(
        split_vertices=False, auto_orient_normals=True)
    center, radius = np.asarray(mesh.center), mesh.length / 2
    frame = lambda t: vm[int(np.argmin(np.abs(times - t)))].astype(np.float32)
    mesh["vm"] = frame(schedule[0])
    p = r.studio(size)
    r.add_surface(p, mesh, "vm", r.lut(r.BRAND, CLIM), CLIM)
    shift, dist = r.autoframe(p, center, radius, up, az, el, swing, dist)
    enc = r.Encoder(r.OUT / f"{out}.mp4", size)
    total = len(schedule) + hold + fade
    angle = lambda i: az + swing * np.sin(2 * np.pi * i / total)   # gentle sway, seamless
    poster = None
    for i, t in enumerate(schedule):
        mesh["vm"] = frame(t)
        r.camera(p, center, radius, up, angle(i), el, dist, shift=shift)
        img = r.grab(p)
        enc.write(img)
        if poster is None and t >= poster_t:
            poster = img
    for j in range(hold):
        r.camera(p, center, radius, up, angle(len(schedule) + j), el, dist, shift=shift)
        enc.write(r.grab(p))
    for j in range(fade):
        r.camera(p, center, radius, up, angle(len(schedule) + hold + j), el, dist, shift=shift)
        mesh["vm"] = frame(schedule[-1])
        a = r.grab(p).astype(np.float32)
        mesh["vm"] = frame(schedule[0])
        b = r.grab(p).astype(np.float32)
        w = r.smoothstep((j + 1) / (fade + 1))
        enc.write(np.ascontiguousarray((a * (1 - w) + b * w).round().astype(np.uint8)))
    enc.close()
    p.close()
    r.save_poster(poster, f"{out}-poster.webp")
    print(out, "frames", total)


def heart():
    d = np.load(r.DATA / "ep_heart.npz")
    pts = d["points"]
    # Atria on top: point the long axis at the tissue that depolarises first (the sinus node).
    vm = d["vm"]
    first = next(k for k in range(len(vm)) if (vm[k] > -20).mean() > 0.005)
    up = r.orient(r.principal_axis(pts), pts, pts[vm[first] > -20].mean(0))
    schedule = list(np.arange(0, 351, 2.0))       # 2 ms per frame: atria, AV delay, ventricles
    ep_video(d, up, 330, 12, schedule, "ep-heart", poster_t=176)


def biventricle():
    d = np.load(r.DATA / "ep_biventricle.npz")
    up = r.flat_base_normal(d["points"])          # valve plane: the openings point up
    schedule = (list(np.arange(0, 81, 1.0))       # activation in slow motion
                + list(np.arange(84, 240, 6.0))   # plateau, fast-forwarded
                + list(np.arange(240, 405, 4.0)))  # repolarisation
    ep_video(d, up, 60, 18, schedule, "ep-biventricle", poster_t=34)


if __name__ == "__main__":
    which = sys.argv[1:] or ["heart", "biv"]
    if "heart" in which:
        heart()
    if "biv" in which:
        biventricle()
