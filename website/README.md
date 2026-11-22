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
you release. The release files have the same names in every version
(`TorchCor-Lighting-<platform>`, as built by `.github/workflows/build.yml`): the macOS dmgs,
the Windows installer (`-windows-x64-setup.exe`) and portable zip, and the Ubuntu/Debian
`.deb` and `.tar.gz` for other Linux distributions. So every button is a direct link,
`/releases/latest/download/TorchCor-Lighting-<platform>`, which downloads the newest file
without any lookup; the build workflow keeps "latest" on the newest app release, so
publish demo data releases as pre-releases (or untick "Set as the latest release").

The overview's main Download button downloads the visitor's own file straight away: the
Apple silicon dmg on a Mac (the Intel one when a Chromium browser reports an Intel Mac), the
Windows installer, or the `.deb` on Linux (the `.tar.gz` when Firefox names a distribution
outside Debian's family). The line under it names the file and links to the download page,
which lists every file. Phones, ChromeOS and ARM Linux go to the download page instead.

On page load, `assets/js/main.js` also asks the GitHub API for that release (tagged
`v1.2.3`; pre-releases and the demo data releases, `data-v1.0`, are skipped) to show its
version, date and file sizes (cached for 10 minutes). Without an answer (offline, or
GitHub's limit of 60 requests an hour) the page says "Latest version" and the links work
all the same. The repository must be public for visitors to download.
