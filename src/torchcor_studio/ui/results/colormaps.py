"""Curated colour maps shared by the 3D viewport (VTK) and the plots (pyqtgraph).

The TorchCor map follows the brand — deep navy at rest, royal blue, a near-white
wavefront, ember orange and a warm yellow peak — and is the default for voltage. The
others are perceptual standards. Maps are stored as compact colour stops and expanded
to lookup tables on demand, so nothing here imports matplotlib.

    lut = vtk_lookup_table("torchcor", clim=(-90, 40))
    cmap = pg_colormap("turbo")
    rgb = get("viridis").table(256)        # (256, 3) floats in 0..1
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PySide6.QtGui import QColor, QLinearGradient

__all__ = ["COLORMAPS", "DEFAULT", "Colormap", "get", "gradient", "names", "pg_colormap", "vtk_lookup_table"]


def _even(*colors: str) -> tuple[tuple[float, str], ...]:
    last = len(colors) - 1
    return tuple((i / last, c) for i, c in enumerate(colors))


@dataclass(frozen=True)
class Colormap:
    """A named colour map defined by ``(position, "#RRGGBB")`` stops on ``[0, 1]``."""

    name: str
    label: str
    stops: tuple[tuple[float, str], ...]
    diverging: bool = False

    def table(self, n: int = 256, reverse: bool = False) -> np.ndarray:
        """``(n, 3)`` float RGB in ``[0, 1]``, interpolated between the stops."""
        return _table(self.name, n, reverse).copy()

    def table_rgba8(self, n: int = 256, reverse: bool = False) -> np.ndarray:
        """``(n, 4)`` uint8 RGBA (opaque)."""
        rgb = np.round(_table(self.name, n, reverse) * 255).astype(np.uint8)
        return np.column_stack([rgb, np.full(n, 255, np.uint8)])

    def color(self, x: float) -> QColor:
        """The colour at position ``x`` in ``[0, 1]``."""
        rgb = _table(self.name, 256, False)[round(min(max(x, 0.0), 1.0) * 255)]
        return QColor.fromRgbF(*rgb)


COLORMAPS: dict[str, Colormap] = {
    cm.name: cm for cm in (
        Colormap("torchcor", "TorchCor", (
            (0.00, "#0E1B4F"), (0.16, "#17318A"), (0.31, "#2559DB"), (0.43, "#5B93FF"),
            (0.53, "#EAF0FB"), (0.63, "#FFC08E"), (0.76, "#FF7A2F"), (0.89, "#FFA62E"),
            (1.00, "#FFE36A"),
        )),
        Colormap("turbo", "Turbo", _even(
            "#30123B", "#3B2F80", "#434EBA", "#466BE3", "#4685FA", "#3BA0FD", "#28BCEB", "#1AD2D2",
            "#1AE4B6", "#32F298", "#55FA76", "#80FF53", "#A4FC3C", "#BEF434", "#D9E436", "#EECF3A",
            "#FABA39", "#FE9E2F", "#FB7E21", "#F26014", "#E4450A", "#D02F05", "#B91E02", "#9B0F01",
            "#7A0403",
        )),
        Colormap("viridis", "Viridis", _even(
            "#440154", "#48186A", "#472D7B", "#424086", "#3B528B", "#33638D", "#2C728E", "#26828E",
            "#21918C", "#1FA088", "#28AE80", "#3FBC73", "#5EC962", "#84D44B", "#ADDC30", "#D8E219",
            "#FDE725",
        )),
        Colormap("inferno", "Inferno", _even(
            "#000004", "#0B0724", "#210C4A", "#3D0965", "#57106E", "#71196E", "#8A226A", "#A32C61",
            "#BC3754", "#D24644", "#E45A31", "#F1731D", "#F98E09", "#FCAC11", "#F9CB35", "#F2EA69",
            "#FCFFA4",
        )),
        Colormap("coolwarm", "Cool–warm", _even(
            "#3B4CC0", "#4E68D8", "#6282EA", "#779AF7", "#8DB0FE", "#A3C2FE", "#B9D0F9", "#CCD9ED",
            "#DDDCDC", "#ECD3C5", "#F5C4AC", "#F7B093", "#F4987A", "#EB7D62", "#DD5F4B", "#CA3B37",
            "#B40426",
        ), diverging=True),
    )
}

DEFAULT = "torchcor"


def names() -> list[str]:
    return list(COLORMAPS)


def get(name: str | None) -> Colormap:
    """The map called ``name`` (falls back to the TorchCor map)."""
    return COLORMAPS.get(name or DEFAULT, COLORMAPS[DEFAULT])


@lru_cache(maxsize=64)
def _table(name: str, n: int, reverse: bool) -> np.ndarray:
    cm = get(name)
    pos = np.array([p for p, _ in cm.stops])
    rgb = np.array([QColor(c).getRgbF()[:3] for _, c in cm.stops])
    x = np.linspace(0.0, 1.0, n)
    if reverse:
        x = x[::-1]
    table = np.column_stack([np.interp(x, pos, rgb[:, k]) for k in range(3)])
    table.setflags(write=False)
    return table


def gradient(name: str, start, end, reverse: bool = False, stops: int = 24) -> QLinearGradient:
    """A ``QLinearGradient`` from ``start`` to ``end`` (QPointF) for painted colour bars."""
    grad = QLinearGradient(start, end)
    table = _table(name, stops, reverse)
    for i, rgb in enumerate(table):
        grad.setColorAt(i / (stops - 1), QColor.fromRgbF(*rgb))
    return grad


def vtk_lookup_table(name: str, clim: tuple[float, float] = (0.0, 1.0), reverse: bool = False,
                     nan_color: str = "#7F8899", n: int = 256):
    """A ``pyvista.LookupTable`` for the map (imports pyvista lazily)."""
    import pyvista as pv

    lut = pv.LookupTable()
    lut.values = get(name).table_rgba8(n, reverse)
    lut.scalar_range = (float(clim[0]), float(clim[1]))
    lut.nan_color = nan_color
    return lut


def pg_colormap(name: str, reverse: bool = False):
    """A ``pyqtgraph.ColorMap`` for the map (imports pyqtgraph lazily)."""
    import pyqtgraph as pg

    table = get(name).table(64, reverse)
    return pg.ColorMap(np.linspace(0.0, 1.0, len(table)), np.round(table * 255).astype(np.uint8), name=name)
