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

The site always offers the newest app release on GitHub, with nothing to update here when
you release. The buttons are the release assets of `sagebei/torchcor_lighting`
(`TorchCor-Lighting-<version>-<platform>`), as built by `.github/workflows/build.yml`: the
macOS dmgs, the Windows installer (`-windows-x64-setup.exe`) and portable zip, and the
Ubuntu/Debian `.deb` and `.tar.gz` for other Linux distributions.

On page load, `assets/js/main.js` asks the GitHub API for the newest app release (tagged
`v1.2.3`; pre-releases and the demo data releases, `data-v1.0`, are skipped) and fills in
its version, date, file links and sizes (cached for 10 minutes). Until it answers, or if it
cannot (offline, or GitHub's limit of 60 requests an hour), the buttons open the release
page at `/releases/latest`, from which every file can be downloaded. No version is written
into the pages.

That page must be the app's: the build workflow marks each app release as the latest, so
publish demo data releases as pre-releases (or untick "Set as the latest release"). The
repository must be public for the API and the downloads to work.
