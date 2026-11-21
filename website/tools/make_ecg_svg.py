"""ECG card: the animated 12-lead ECG as inline SVG, from a TorchCor Lighting ECG run.

    python make_ecg_svg.py PATH/TO/viz/ecg.npz      # -> data/ecg.svg

ecg.npz is the run's visualisation bundle (t in ms, 12 leads in mV, lead names). The leads
are drawn on navy ECG paper in the standard 3x4 layout plus a lead II rhythm strip, inked
with the site's brand gradient across the whole chart (amber on the left to blue on the
right, like the "Lighting" text). SMIL clip animations sweep the traces in, hold, fade and
repeat. Simplified with Ramer-Douglas-Peucker, so the SVG stays around 10 kB. Run
update_page.py afterwards.
"""
import sys
from pathlib import Path

import numpy as np

import render as r

W, H = 160, 112                     # cell size
PAD_X, BASE, MV = 7, 66, 47         # trace inset, 0 mV line within a cell, pixels per mV
LAYOUT = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
DUR, SWEEP, HOLD = 5.2, 0.55, 0.9   # loop seconds; sweep ends / fade starts, as fractions
INK = [(0, "#FFB04A"), (0.30, "#FF6A2C"), (0.55, "#FF3F7A"), (0.78, "#8A63FF"), (1, "#4C8DFF")]


def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.hypot(*ab) or 1.0
    dist = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
    i = int(np.argmax(dist))
    if dist[i] > eps:
        return np.vstack([rdp(pts[: i + 1], eps)[:-1], rdp(pts[i:], eps)])
    return np.vstack([a, b])


def build(src):
    d = np.load(src, allow_pickle=False)
    t, leads, names = d["t"], d["leads"], [str(n) for n in d["lead_names"]]
    lead = dict(zip(names, leads))
    anim = f'dur="{DUR}s" repeatCount="indefinite"'

    def path(values, x0, y0, width):
        """Trace in chart coordinates, so one gradient spans the whole chart."""
        x = x0 + PAD_X + t / t[-1] * (width - 2 * PAD_X)
        pts = rdp(np.column_stack([x, y0 + BASE - values * MV]), 0.22)
        return "M" + " L".join(f"{px:.1f} {py:.1f}" for px, py in pts)

    def sweep(cid, x0, y0, width, height):
        return (f'<clipPath id="{cid}"><rect x="{x0}" y="{y0}" width="0" height="{height}">'
                f'<animate attributeName="width" values="0;{width};{width}" keyTimes="0;{SWEEP};1" {anim}/>'
                f'</rect></clipPath>')

    def pen(x0, y0, width, height):
        start, end = x0 + PAD_X - 14, x0 + width - PAD_X - 14
        return (f'<rect class="ecg__pen" y="{y0 + 6}" width="14" height="{height - 12}" fill="url(#ecg-pen-grad)" opacity="0">'
                f'<animate attributeName="x" values="{start};{end};{end}" keyTimes="0;{SWEEP};1" {anim}/>'
                f'<animate attributeName="opacity" values="0.75;0.75;0;0" keyTimes="0;{SWEEP};{SWEEP + 0.04:.2f};1" {anim}/>'
                f'</rect>')

    clips = [sweep(f"ecg-col-{c}", c * W, 0, W, 3 * H) for c in range(4)]
    clips.append(sweep("ecg-rhythm", 0, 3 * H, 4 * W, H))
    labels, traces = [], []
    for row_i, row in enumerate(LAYOUT):
        for col_i, name in enumerate(row):
            x0, y0 = col_i * W, row_i * H
            labels.append(f'<text class="ecg__label" x="{x0 + 8}" y="{y0 + 15}">{name}</text>')
            traces.append(f'<path clip-path="url(#ecg-col-{col_i})" d="{path(lead[name], x0, y0, W)}"/>')
    labels.append(f'<text class="ecg__label" x="8" y="{3 * H + 15}">II rhythm</text>')
    traces.append(f'<path clip-path="url(#ecg-rhythm)" d="{path(lead["II"], 0, 3 * H, 4 * W)}"/>')
    pens = [pen(c * W, 0, W, 3 * H) for c in range(4)] + [pen(0, 3 * H, 4 * W, H)]
    stops = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in INK)

    width, height = 4 * W, 4 * H
    nl = "\n    "
    return f'''<svg class="ecg" viewBox="0 0 {width} {height}" role="img" data-still="{DUR * SWEEP + 0.2:.2f}"
     aria-labelledby="ecg-title">
  <title id="ecg-title">A simulated 12-lead ECG from TorchCor, drawn lead by lead on ECG paper</title>
  <defs>
    <pattern id="ecg-minor" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M8 0H0V8" class="ecg__minor"/></pattern>
    <pattern id="ecg-major" width="40" height="40" patternUnits="userSpaceOnUse"><rect width="40" height="40" fill="url(#ecg-minor)"/><path d="M40 0H0V40" class="ecg__major"/></pattern>
    <linearGradient id="ecg-ink" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="{width}" y2="0">{stops}</linearGradient>
    <linearGradient id="ecg-pen-grad" x1="0" x2="1"><stop offset="0" stop-color="#ffffff" stop-opacity="0"/><stop offset="1" stop-color="#ffffff" stop-opacity=".85"/></linearGradient>
    <filter id="ecg-glow" x="-5%" y="-10%" width="110%" height="120%"><feGaussianBlur stdDeviation="1.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    {nl.join(clips)}
  </defs>
  <rect class="ecg__paper" width="{width}" height="{height}" rx="10"/>
  <rect width="{width}" height="{height}" fill="url(#ecg-major)" rx="10"/>
  {"".join(labels)}
  <g class="ecg__traces" filter="url(#ecg-glow)" stroke="url(#ecg-ink)">
    <animate attributeName="opacity" values="1;1;0" keyTimes="0;{HOLD};1" {anim}/>
    {nl.join(traces)}
  </g>
  {"".join(pens)}
</svg>'''


if __name__ == "__main__":
    svg = build(Path(sys.argv[1]).expanduser())
    r.DATA.mkdir(parents=True, exist_ok=True)
    (r.DATA / "ecg.svg").write_text(svg)
    print("ecg.svg bytes", len(svg))
