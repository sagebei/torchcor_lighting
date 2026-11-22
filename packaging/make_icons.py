#!/usr/bin/env python3
"""Regenerate the platform icon files from ``resources/app_icon.png``.

    python packaging/make_icons.py            # rebuild both icons
    python packaging/make_icons.py --check    # exit 1 if an icon is missing or stale

Outputs (next to the source PNG, in ``src/torchcor_studio/resources/``):

* ``app_icon.icns`` (macOS) -- a full ``.iconset`` (16 to 512 pt at 1x and 2x, so
  16 to 1024 px) packed with Apple's ``iconutil``. Where ``iconutil`` is unavailable
  (Windows, Linux) Pillow's ICNS writer is used instead.
* ``app_icon.ico`` (Windows) -- 16, 20, 24, 32, 40, 48, 64, 96, 128 and 256 px.

The Linux .deb's PNG icons (16 to 512 px) are resized from the same PNG when
``build.py`` packages it (``make_hicolor``); they are not kept in the repository.

The source is a 1024 x 1024 rounded tile drawn on Apple's icon grid (transparent
margin around the tile). macOS expects that margin, so the ``.icns`` uses the image
as is. Windows icons conventionally fill their square, so the ``.ico`` is cropped to
the tile first, which keeps the 16 and 24 px taskbar sizes legible.

Needs Pillow (already present wherever the app's dependencies are installed).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "src" / "torchcor_studio" / "resources"
SOURCE = RESOURCES / "app_icon.png"
ICNS = RESOURCES / "app_icon.icns"
ICO = RESOURCES / "app_icon.ico"

#: (point size, scale) pairs of a complete macOS .iconset.
ICONSET = [(pt, scale) for pt in (16, 32, 128, 256, 512) for scale in (1, 2)]
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
HICOLOR_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)


def _load(path: Path) -> Image.Image:
    image = Image.open(path)
    image.load()
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if image.width != image.height:
        raise SystemExit(f"{path} must be square (got {image.width}x{image.height})")
    if image.width < 1024:
        print(f"warning: {path.name} is {image.width}px; 1024px gives the sharpest Retina icons",
              file=sys.stderr)
    return image


def _resized(image: Image.Image, px: int) -> Image.Image:
    return image if image.width == px else image.resize((px, px), Image.Resampling.LANCZOS)


def _tile(image: Image.Image) -> Image.Image:
    """The image cropped to its visible tile, as a centred square."""
    alpha = image.getchannel("A").point(lambda a: 255 if a > 8 else 0)
    box = alpha.getbbox()
    if box is None:
        return image
    left, top, right, bottom = box
    side = max(right - left, bottom - top)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half = side / 2
    square = (round(cx - half), round(cy - half), round(cx + half), round(cy + half))
    return image.crop(square)


def make_icns(image: Image.Image, out: Path = ICNS) -> str:
    """Write ``out``; returns the tool that was used."""
    iconutil = shutil.which("iconutil") if sys.platform == "darwin" else None
    if iconutil:
        with tempfile.TemporaryDirectory() as tmp:
            iconset = Path(tmp) / "app_icon.iconset"
            iconset.mkdir()
            for pt, scale in ICONSET:
                suffix = "" if scale == 1 else f"@{scale}x"
                _resized(image, pt * scale).save(iconset / f"icon_{pt}x{pt}{suffix}.png")
            subprocess.run([iconutil, "--convert", "icns", "--output", str(out), str(iconset)], check=True)
        return "iconutil"
    # Pillow writes every ICNS size it supports (16 to 1024 px) from the closest image.
    sizes = sorted({pt * scale for pt, scale in ICONSET})
    _resized(image, 1024).save(out, format="ICNS",
                               append_images=[_resized(image, px) for px in sizes if px != 1024])
    return "Pillow"


def make_ico(image: Image.Image, out: Path = ICO) -> None:
    tile = _tile(image)
    frames = {px: _resized(tile, px) for px in ICO_SIZES}
    frames[256].save(out, format="ICO", sizes=[(px, px) for px in ICO_SIZES],
                     append_images=[frames[px] for px in ICO_SIZES if px != 256])


def make_hicolor(theme: Path, name: str) -> None:
    """``<theme>/<px>x<px>/apps/<name>.png`` for each of HICOLOR_SIZES (the Linux .deb's
    icons). The image as is: freedesktop icons keep a margin around the tile too."""
    image = _load(SOURCE)
    for px in HICOLOR_SIZES:
        folder = theme / f"{px}x{px}" / "apps"
        folder.mkdir(parents=True, exist_ok=True)
        _resized(image, px).save(folder / f"{name}.png", optimize=True)


def stale(target: Path, source: Path = SOURCE) -> bool:
    return not target.is_file() or target.stat().st_mtime < source.stat().st_mtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="only report whether the icons are missing or older than the PNG")
    parser.add_argument("--source", type=Path, default=SOURCE, help="square PNG to build from")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"error: {args.source} not found", file=sys.stderr)
        return 2
    if args.check:
        outdated = [p.name for p in (ICNS, ICO) if stale(p, args.source)]
        if outdated:
            print("missing or stale: " + ", ".join(outdated))
            return 1
        print("icons are up to date")
        return 0

    image = _load(args.source)
    tool = make_icns(image)
    make_ico(image)
    for path in (ICNS, ICO):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")
    print(f"(.icns packed with {tool})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
