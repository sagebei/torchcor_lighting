# TorchCor Lighting website

A static two-page site: `index.html` (overview) and `download.html` (downloads).
No build step and no external dependencies.

## Preview

Open `index.html` in a browser, or serve the folder to your local network (from the repo root):

```bash
python3 website/tools/serve.py
```

It prints the address to share (`http://<this computer's IP>:8000/`). Unlike
`python3 -m http.server`, it answers byte-range requests, which Safari needs to play the
videos.

## Deploy

Any static host works (GitHub Pages, Netlify, Cloudflare Pages): publish this folder as is.
The `tools/` folder is not needed by the site and can be left out of the upload.

## Simulation media

The videos in `assets/media/`, the animated ECG and the pressure–volume data in
`index.html` are rendered from real TorchCor results. `tools/README.md` explains how to
export the data and regenerate them.

## Download links

The download buttons point to the GitHub Release assets of `sagebei/torchcor_lighting`
(`TorchCor-Lighting-<version>-<platform>`), as built by `.github/workflows/build.yml`.
On page load, `assets/js/main.js` asks the GitHub API for the latest release and updates
the version, file sizes and links automatically, so a new release needs no change here.
The repository must be public for this, and for the download links to work.

If the API cannot be reached, the links fall back to the version in `FALLBACK_VERSION`
(`assets/js/main.js`) and the hard-coded links in `download.html`. Update both when you
release, to keep that fallback current.
