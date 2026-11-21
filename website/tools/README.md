# Media tools

Scripts that make the simulation media on the overview page (`index.html`, "Four kinds of
simulation") from real TorchCor results. The website itself does not need them: they are
here so the media can be regenerated, for example from newer runs.

`serve.py` is separate: it serves the website to your local network
(`python3 website/tools/serve.py`), with the byte-range support Safari needs for video.

| Card | Output in `assets/media/` | Made by | From |
| :-- | :-- | :-- | :-- |
| Electrophysiology | `ep-heart.mp4`, `ep-biventricle.mp4` (+ `-poster.webp`) | `render_ep.py` | whole-heart sinus beat; app EP run on the 637k-node biventricle |
| 12-lead ECG | inline SVG in `index.html` | `make_ecg_svg.py`, `update_page.py` | `viz/ecg.npz` of an app ECG run |
| Mechanics | `mech-lv.mp4` (+ poster) | `render_lv.py` | idealised LV benchmark (`b1.npz`) |
| Electromechanics | `em-heart.mp4` (+ poster), pressure–volume data in `index.html` | `render_em.py`, `update_page.py` | four-chamber EM keyframes and `pv.csv` |

## Requirements

On this computer: Python 3.11+ with `numpy`, `scipy`, `pyvista` and `pillow` (the app's
development environment has them), and `ffmpeg` on the PATH. Rendering is offscreen.

On the compute machine: its TorchCor Python environment (`torch`, `numpy`, `pyvista`).

## 1. Export the data

`export_data.py` runs on the compute machine and streams a compact `.npz` back over SSH;
nothing is written there. Exports go to `tools/data/` (git-ignored, ~180 MB in all).
Replace `HOST` and `PYTHON` with your SSH host and that machine's Python.

```bash
mkdir -p data
ssh HOST 'PYTHON - ep-run ~/torchcor_output/ep/20261121-104718-monodomain-nov-21-10-40-copy-98c2' < export_data.py > data/ep_biventricle.npz
ssh HOST 'PYTHON - ep-surface ~/torchcor/manuscript/ep_plot/results/heart' < export_data.py > data/ep_heart.npz
ssh HOST 'PYTHON - em ~/torchcor/manuscript/em_plot/data/torchcor/pv.csv ~/torchcor/manuscript/em_plot/data/torchcor/vtk' < export_data.py > data/em_keyframes.npz
scp HOST:torchcor/manuscript/mechanics_plot/results/b1.npz data/lv_b1.npz
```

The ECG needs no export: `make_ecg_svg.py` reads the run's `viz/ecg.npz` directly, either
from the run folder or from the app's local cache (on macOS,
`~/Library/Caches/TorchCorStudio/runs/<machine>/<run>/viz/ecg.npz`).

## 2. Render

```bash
python render_ep.py          # heart and biventricle; or: python render_ep.py heart | biv
python render_lv.py
python render_em.py          # also writes data/em_pv.json
python make_ecg_svg.py ~/Library/Caches/TorchCorStudio/runs/<machine>/<run>/viz/ecg.npz
python update_page.py        # puts data/ecg.svg and data/em_pv.json into ../index.html
```

Videos are 520×600 H.264 MP4s (30 fps, about 0.1–0.35 MB each) that loop seamlessly; each
also gets a poster frame, which the page shows while loading and to visitors who prefer
reduced motion. Set `TCL_OUT` to write the media elsewhere, `TCL_DATA` to read the data
from another folder, and `TCL_PAGE` to update a different HTML file.

## How the renders are set up

- **Look.** Studio lighting (key, fill, rim), supersampled anti-aliasing, and a background
  close to the card colour; the page blends the rest away (`mix-blend-mode: lighten`).
  Colours use the site's brand scale (`render.BRAND`): the stops of the "Lighting" text
  gradient, from deep indigo (low) through blue, violet, rose and orange to amber (high),
  the same as the page's legends (`--brand-map` in `style.css`). Voltage spans −100…50 mV,
  displacement 0–40 mm and active tension 0–90 kPa. The ECG traces are inked with the
  gradient itself, amber on the left to blue on the right.
- **Orientation.** Computed, not hand-tuned: the biventricle's valve plane is the largest
  flat patch of its convex hull (so the openings point up); the whole heart's long axis
  points at the tissue that activates first (the sinus node, so the atria are on top);
  the EM heart's long axis points at the low-tension atria; the benchmark LV's axis is x.
  Each heart is then centred and sized to fit the frame over its whole camera sway.
- **Timing.** EP: activation in slow motion (1 ms per frame for the biventricle, 2 ms for
  the whole heart), plateau fast-forwarded, then a cross-fade back to rest. Mechanics: the
  beat at half-frame steps. EM: one 833 ms beat over 136 frames.
- **Electromechanics keyframes.** The EM run kept five surface states over its last beat
  (3390–4130 ms), so the motion between them is a smooth periodic interpolation of those
  states; the pressure–volume loops and their dots come from the same run's full `pv.csv`.
  Always export keyframes and `pv.csv` from one run (`overview_plot/data/heart` is a
  different run). With a complete frame series (`vtk/heart_*.vtk` every 10 ms),
  `render_em.py` renders it unchanged.
- **Choosing other views.** `render.contact_sheet` tiles labelled renders, which is the
  quickest way to compare camera angles (`az`, `el` in each script).
