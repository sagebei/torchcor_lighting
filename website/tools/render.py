"""Shared offscreen renderer for the website's simulation videos (PyVista/VTK + ffmpeg).

Inputs are read from ``tools/data`` and videos are written to ``assets/media``; set the
``TCL_DATA`` / ``TCL_OUT`` environment variables to use other folders.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw

TOOLS = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TCL_DATA", TOOLS / "data"))
OUT = Path(os.environ.get("TCL_OUT", TOOLS.parent / "assets" / "media"))
BG = "#0d1016"  # close to the card background; the page blends the rest away

# The website's brand scale: the stops of the "Lighting" text gradient (--grad in style.css),
# from deep indigo (low) through blue, violet, rose and orange to amber (high). The page's
# legends use the same stops (--brand-map), so videos, ECG, loops and text match.
BRAND = [(0.00, "#18226A"), (0.18, "#2C4FD8"), (0.36, "#4C8DFF"), (0.54, "#8A63FF"),
         (0.70, "#FF3F7A"), (0.85, "#FF6A2C"), (1.00, "#FFB04A")]

# Colormaps as in TorchCor Lighting (src/torchcor_studio/ui/results/colormaps.py)
TORCHCOR = [(0.00, "#0E1B4F"), (0.16, "#17318A"), (0.31, "#2559DB"), (0.43, "#5B93FF"),
            (0.53, "#EAF0FB"), (0.63, "#FFC08E"), (0.76, "#FF7A2F"), (0.89, "#FFA62E"),
            (1.00, "#FFE36A")]


def _even(*colors):
    last = len(colors) - 1
    return [(i / last, c) for i, c in enumerate(colors)]


VIRIDIS = _even("#440154", "#48186A", "#472D7B", "#424086", "#3B528B", "#33638D", "#2C728E",
                "#26828E", "#21918C", "#1FA088", "#28AE80", "#3FBC73", "#5EC962", "#84D44B",
                "#ADDC30", "#D8E219", "#FDE725")
INFERNO = _even("#000004", "#0B0724", "#210C4A", "#3D0965", "#57106E", "#71196E", "#8A226A",
                "#A32C61", "#BC3754", "#D24644", "#E45A31", "#F1731D", "#F98E09", "#FCAC11",
                "#F9CB35", "#F2EA69", "#FCFFA4")


def lut(stops, clim, n=256):
    xs = np.array([s for s, _ in stops])
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for _, c in stops], float)
    t = np.linspace(0, 1, n)
    table = np.stack([np.interp(t, xs, rgb[:, k]) for k in range(3)], axis=1)
    table = pv.LookupTable(values=np.column_stack([table, np.full(n, 255)]).astype(np.uint8))
    table.scalar_range = clim
    return table


def polydata(points, triangles):
    faces = np.column_stack([np.full(len(triangles), 3), triangles]).ravel()
    return pv.PolyData(np.asarray(points, np.float32), faces)


def studio(size):
    """Offscreen plotter with studio lighting: key (upper left), fill, rim (behind), bounce."""
    p = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    p.set_background(BG)
    p.enable_anti_aliasing("ssaa")
    for pos, intensity in (((-1.0, 1.1, 1.4), 0.95), ((1.3, -0.2, 1.0), 0.45),
                           ((0.4, 1.4, -1.6), 0.75), ((0.0, -1.2, 0.6), 0.2)):
        p.add_light(pv.Light(position=pos, focal_point=(0, 0, 0), light_type="camera light",
                             intensity=intensity))
    return p


def add_surface(p, mesh, scalars, table, clim, **kw):
    style = dict(smooth_shading=True, ambient=0.22, diffuse=0.9, specular=0.35,
                 specular_power=22, show_scalar_bar=False)
    style.update(kw)
    return p.add_mesh(mesh, scalars=scalars, cmap=table, clim=clim, **style)


# ------------------------------------------------------------------ orientation helpers
def principal_axis(points, step=5):
    """Unit vector of the first principal axis (sign arbitrary; see ``orient``)."""
    c = np.asarray(points, np.float64)
    c = c - c.mean(0)
    _, _, vt = np.linalg.svd(c[::step], full_matrices=False)
    return vt[0]


def orient(axis, points, toward):
    """Flip ``axis`` so that it points from the centre of ``points`` towards ``toward``."""
    return axis if np.dot(np.asarray(toward) - np.asarray(points).mean(0), axis) > 0 else -axis


def flat_base_normal(points, step=5, tol_deg=8.0):
    """Outward normal of the largest flat patch of the convex hull.

    For a biventricle cut at the valve plane this is the base: the openings point along it.
    """
    from scipy.spatial import ConvexHull
    hull = ConvexHull(np.asarray(points, np.float64)[::step])
    n = hull.equations[:, :3]                      # Qhull normals point outwards
    q = hull.points[hull.simplices]
    area = 0.5 * np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
    near = (n @ n.T) > np.cos(np.radians(tol_deg))
    i = int(np.argmax(near @ area))
    normal = (n[near[i]] * area[near[i], None]).sum(0)
    return normal / np.linalg.norm(normal)


# ------------------------------------------------------------------ camera and frames
def basis(up):
    up = np.asarray(up, float) / np.linalg.norm(up)
    ref = np.array([0, 0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0, 0])
    side = np.cross(up, ref)
    side /= np.linalg.norm(side)
    return up, side, np.cross(up, side)


def camera(p, center, radius, up, az, el, dist=3.3, view_angle=28.0, shift=(0, 0)):
    """Orbit camera: azimuth around ``up``, elevation above the equator, ``up`` kept vertical."""
    u, s, t = basis(up)
    a, e = np.radians(az), np.radians(el)
    view = np.cos(e) * (np.cos(a) * s + np.sin(a) * t) + np.sin(e) * u
    right = np.cross(view, u)
    right /= np.linalg.norm(right)
    focal = np.asarray(center) + shift[0] * radius * right + shift[1] * radius * u
    p.camera.focal_point = focal
    p.camera.position = focal + view * radius * dist
    p.camera.up = u
    p.camera.view_angle = view_angle
    p.reset_camera_clipping_range()


def grab(p):
    p.render()  # screenshot() alone would return the first cached frame
    return np.ascontiguousarray(p.screenshot(return_img=True)[:, :, :3])


def autoframe(p, center, radius, up, az, el, swing, dist=3.3, fill=0.86, view_angle=28.0, iters=3):
    """Shift and distance that centre the object and fit it to ``fill`` of the frame over the sway."""
    bg = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)])
    shift = np.zeros(2)
    for _ in range(iters):
        boxes = []
        for a in (az - swing, az - swing / 2, az, az + swing / 2, az + swing):
            camera(p, center, radius, up, a, el, dist, view_angle, shift)
            img = grab(p)
            ys, xs = np.nonzero(np.abs(img.astype(int) - bg).sum(2) > 24)
            boxes.append((xs.min(), xs.max(), ys.min(), ys.max()))
        h, w = img.shape[:2]
        x0, x1 = min(b[0] for b in boxes), max(b[1] for b in boxes)
        y0, y1 = min(b[2] for b in boxes), max(b[3] for b in boxes)
        px = 2 * dist * radius * np.tan(np.radians(view_angle / 2)) / h   # world units per pixel
        shift[0] -= ((x0 + x1) / 2 - w / 2) * px / radius
        shift[1] -= ((y0 + y1) / 2 - h / 2) * px / radius
        dist *= max((x1 - x0) / w, (y1 - y0) / h) / fill
    return tuple(shift), dist


def smoothstep(x):
    return x * x * (3 - 2 * x)


class Encoder:
    """H.264 MP4 (yuv420p, faststart) fed with raw RGB frames through ffmpeg."""

    def __init__(self, path, size, fps=30, crf=22):
        w, h = size
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, img):
        self.ff.stdin.write(img.tobytes())

    def close(self):
        self.ff.stdin.close()
        self.ff.wait()


def save_poster(img, name):
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(OUT / name, quality=88, method=6)


def contact_sheet(images, labels, path, cols=4):
    """Tile labelled renders: handy for choosing a camera angle."""
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows))
    for i, (im, lab) in enumerate(zip(images, labels)):
        tile = Image.fromarray(im)
        ImageDraw.Draw(tile).text((8, 8), lab, fill=(255, 255, 255))
        sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
    sheet.save(path)
