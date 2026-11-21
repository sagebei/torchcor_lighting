"""Electrophysiology results: animated voltage and activation maps, APs, LAT histogram."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from torchcor_studio.core.results import ResultBundle
from torchcor_studio.ui.results import colormaps
from torchcor_studio.ui.results.common import (
    VIEW_MARGINS,
    ColormapCombo,
    SidePanel,
    Stage,
    StatGrid,
    card_title,
    finite_range,
    nice_bounds,
    nice_step,
    notes_card,
    panel_header,
    split,
    switch,
    toolbar,
)
from torchcor_studio.ui.results.playback import PlaybackController, TimelineBar
from torchcor_studio.ui.results.plots import (
    HoverReadout,
    LegendChips,
    LegendItem,
    ThemedPlot,
    TimeCursor,
    series_pen,
)
from torchcor_studio.ui.results.viewer3d import Viewer3D
from torchcor_studio.ui.theme import Palette, theme
from torchcor_studio.ui.widgets import Button, Card, SegmentedControl

MAX_PICKS = 6


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    title: str
    unit: str
    cmap: str


FIELDS = (
    FieldSpec("voltage", "Voltage", "Transmembrane voltage", "mV", "torchcor"),
    FieldSpec("activation", "Activation", "Activation time", "ms", "turbo"),
    FieldSpec("repolarisation", "Repolarisation", "Repolarisation time", "ms", "turbo"),
    FieldSpec("apd", "APD", "Action potential duration", "ms", "viridis"),
)


def _has_values(values: np.ndarray | None) -> bool:
    return values is not None and bool(np.isfinite(values).any())


class EPView(QWidget):
    """EP bundle view. ``set_active`` pauses playback while hidden; ``release`` frees VTK.

    Embedded in another view (the ECG view's Heart page), pass ``show_stats=False``, a
    ``title`` and ``margins=(0, 0, 0, 0)``.
    """

    def __init__(self, bundle: ResultBundle, parent: QWidget | None = None, *, show_stats: bool = True,
                 title: str = "", margins: tuple[int, int, int, int] = VIEW_MARGINS) -> None:
        super().__init__(parent)
        self.bundle = bundle
        self.surface = bundle.surface
        self.ep = bundle.ep
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._static = {"activation": self.ep.lat, "repolarisation": self.ep.rt, "apd": self.ep.apd}
        self.fields = [f for f in FIELDS if f.key == "voltage" or _has_values(self._static[f.key])]
        self._field = self.fields[0].key
        self._cmaps = {f.key: f.cmap for f in self.fields}
        self._clims = {f.key: self._clim(f.key) for f in self.fields}
        self._picks: list[int] = []
        self._hidden: set[str] = set()
        self._was_playing = False
        self._released = False
        times = self.ep.times
        self._dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0

        # ------------------------------------------------------------------ viewport
        span = float(times[-1] - times[0]) if len(times) > 1 else 1.0
        self.controller = PlaybackController(len(times), fps=min(30.0, max(6.0, len(times) / 8.0)), parent=self)
        self.viewer = Viewer3D(export_name="torchcor-ep")
        self.viewer.set_mesh(self.surface.points, self.surface.triangles)
        self.timeline = TimelineBar(self.controller, self._time_text, shortcut_scope=self)
        self.stage = Stage(self.viewer, self.timeline)

        # ------------------------------------------------------------------ toolbar
        self.field_control = SegmentedControl([(f.key, f.label) for f in self.fields], value=self._field)
        if len(self.fields) < 2:
            self.field_control.hide()
        self.field_control.changed.connect(self.set_field)
        self.cmap_combo = ColormapCombo(self._cmaps[self._field])
        self.cmap_combo.colormap_changed.connect(self._set_cmap)
        lat_span = float(np.subtract(*finite_range(self.ep.lat)[::-1])) if _has_values(self.ep.lat) else span
        self._iso_auto = max(1.0, nice_step(lat_span, 10))
        steps = sorted({5.0, 10.0, 20.0, 50.0, self._iso_auto})
        iso_holder, self.iso_switch = switch("Isochrones", False, self._iso_toggled,
                                             "Lines of equal activation time")
        self.iso_step = QComboBox()
        for step in steps:
            self.iso_step.addItem(f"every {step:g} ms", step)
        self.iso_step.setCurrentIndex(steps.index(self._iso_auto))
        self.iso_step.setEnabled(False)
        self.iso_step.setToolTip("Isochrone spacing")
        self.iso_step.currentIndexChanged.connect(lambda *_: self._update_isolines())
        if not _has_values(self.ep.lat):
            iso_holder.hide()
            self.iso_step.hide()
        edges_holder, self.edges_switch = switch("Edges", False, self.viewer.set_edges, "Show the mesh triangles")
        self.viewer.edges_changed.connect(self._edges_changed)

        # ------------------------------------------------------------------ side panel
        self.side = SidePanel()
        self.side.add(panel_header("Electrophysiology", title or bundle.title or "Activation and repolarisation"))
        self.stats = self.side.add(StatGrid(bundle.summary if show_stats else ()))
        self._build_ap_card()
        self._build_histogram()
        if bundle.notes and show_stats:
            self.side.add(notes_card(bundle.notes))

        root = QVBoxLayout(self)
        root.setContentsMargins(*margins)
        root.setSpacing(14)
        root.addLayout(toolbar(self.field_control, self.cmap_combo, iso_holder, self.iso_step, edges_holder, None))
        root.addWidget(split(self.stage, self.side), 1)

        self.controller.frame_changed.connect(self._show_frame)
        self.viewer.point_picked.connect(self.add_pick)
        theme().changed.connect(self._on_theme)
        self.set_field(self._field)
        self.controller.seek(self._initial_frame())
        self._show_frame(self.controller.frame)

    # ------------------------------------------------------------------ side cards
    def _build_ap_card(self) -> None:
        card = Card(padding=16, spacing=10)
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(card_title("Action potentials", "Click the surface to probe a point"), 1)
        self.clear_button = Button("Clear", variant="ghost", size="sm", on_click=self.clear_picks)
        self.clear_button.setVisible(False)
        top.addWidget(self.clear_button, 0, Qt.AlignmentFlag.AlignTop)
        card.addLayout(top)
        self.ap_plot = ThemedPlot(x_label="Time", x_unit="ms", y_label="Vm", y_unit="mV", interactive=True,
                                  min_height=190)
        lo, hi = self._clims["voltage"]
        self.ap_plot.getPlotItem().setYRange(lo, hi, padding=0.02)
        self.ap_plot.getPlotItem().getViewBox().enableAutoRange(y=False)
        self.ap_cursor = TimeCursor(self.ap_plot, on_drag=self._seek_time)
        self.ap_hover = HoverReadout(self.ap_plot, lambda name, x, y: f"{name}  ·  {x:.0f} ms  ·  {y:.1f} mV")
        self.legend = LegendChips()
        self.legend.toggled.connect(self._legend_toggled)
        self.legend.removed.connect(self._legend_removed)
        card.addWidget(self.ap_plot)
        card.addWidget(self.legend)
        self.side.add(card)
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._probe_points = self._probe_indices()
        self._rebuild_traces()

    def _build_histogram(self) -> None:
        self.hist_card = Card(padding=16, spacing=10)
        self.hist_card.addWidget(card_title("Activation times", "Share of the surface (%) activated in each bin"))
        self.hist_plot = ThemedPlot(x_label="Activation time", x_unit="ms", min_height=120)
        self.hist_bars: pg.BarGraphItem | None = None
        self.hist_cursor = TimeCursor(self.hist_plot)
        self.hist_card.addWidget(self.hist_plot)
        self.side.add(self.hist_card)
        self.hist_card.setVisible(_has_values(self.ep.lat))
        self._rebuild_histogram()

    def _rebuild_histogram(self) -> None:
        if not _has_values(self.ep.lat):
            return
        lat = self.ep.lat[np.isfinite(self.ep.lat)]
        lo, hi = float(lat.min()), float(lat.max())
        width = max(1e-3, nice_step(hi - lo, 22))
        edges = np.arange(np.floor(lo / width) * width, hi + width, width)
        if len(edges) < 2:
            edges = np.array([lo, lo + width])
        counts, edges = np.histogram(lat, bins=edges)
        share = 100.0 * counts / max(1, len(self.ep.lat))
        centers = 0.5 * (edges[:-1] + edges[1:])
        cmap = colormaps.get(self._cmaps.get("activation", "turbo"))
        clo, chi = self._clims.get("activation", (lo, hi))
        brushes = [pg.mkBrush(cmap.color((c - clo) / max(1e-9, chi - clo))) for c in centers]
        if self.hist_bars is not None:
            self.hist_plot.removeItem(self.hist_bars)
        self.hist_bars = pg.BarGraphItem(x=centers, height=share, width=width * 0.84, brushes=brushes,
                                         pen=pg.mkPen(None))
        self.hist_plot.addItem(self.hist_bars)
        self.hist_plot.getPlotItem().setXRange(edges[0], edges[-1], padding=0.02)

    # ------------------------------------------------------------------ fields
    def set_field(self, key: str) -> None:
        """Show ``voltage`` (animated), ``activation``, ``repolarisation`` or ``apd``."""
        spec = next((f for f in self.fields if f.key == key), None)
        if spec is None:
            return
        self._field = key
        self.field_control.set_value(key, animate=False)
        self.cmap_combo.set_value(self._cmaps[key])
        animated = key == "voltage"
        if not animated:
            self.controller.pause()
        self.stage.set_footer_visible(animated)
        values = self.ep.vm.frame(self.controller.frame) if animated else self._static[key]
        self.viewer.set_scalars(values, self._clims[key], self._cmaps[key], spec.title, unit=spec.unit)
        self._update_isolines()

    @property
    def field(self) -> str:
        return self._field

    def _clim(self, key: str) -> tuple[float, float]:
        if key == "voltage":
            return nice_bounds(*self.ep.vm.range, target=16)
        return nice_bounds(*finite_range(self._static[key]), target=12)

    def _set_cmap(self, name: str) -> None:
        self._cmaps[self._field] = name
        spec = next(f for f in self.fields if f.key == self._field)
        self.viewer.set_scalars(self._current_values(), cmap=name, title=spec.title, unit=spec.unit)
        if self._field == "activation":
            self._rebuild_histogram()

    def _current_values(self) -> np.ndarray:
        if self._field == "voltage":
            return self.ep.vm.frame(self.controller.frame)
        return self._static[self._field]

    def _iso_toggled(self, on: bool) -> None:
        self.iso_step.setEnabled(on)
        self._update_isolines()

    def _update_isolines(self) -> None:
        source = self.ep.lat if self._field == "voltage" else self._static.get(self._field)
        if not self.iso_switch.isChecked() or not _has_values(source):
            self.viewer.set_isolines(None)
            return
        step = float(self.iso_step.currentData() or self._iso_auto)
        lo, hi = finite_range(source)
        levels = np.arange(np.ceil(lo / step) * step, hi, step)
        levels = levels[levels > lo]
        self.viewer.set_isolines(source, levels)

    def _edges_changed(self, on: bool) -> None:
        if self.edges_switch.isChecked() != on:
            self.edges_switch.setChecked(on)

    # ------------------------------------------------------------------ playback
    def _time_text(self, frame: int) -> str:
        t = float(self.ep.times[min(frame, len(self.ep.times) - 1)])
        return f"t = {t:.0f} ms" if self._dt >= 1 else f"t = {t:.1f} ms"

    def _initial_frame(self) -> int:
        """Open on the frame where half of the surface is activated: the wavefront is visible."""
        if not _has_values(self.ep.lat):
            return 0
        lat = self.ep.lat[np.isfinite(self.ep.lat)]
        return int(np.argmin(np.abs(self.ep.times - np.median(lat))))

    def _show_frame(self, frame: int) -> None:
        if self._released:
            return
        if self._field == "voltage":
            self.viewer.set_scalars(self.ep.vm.frame(frame))
        t = float(self.ep.times[frame])
        self.ap_cursor.set_value(t)
        self.hist_cursor.set_value(t)

    def _seek_time(self, t: float) -> None:
        self.controller.seek(int(np.argmin(np.abs(self.ep.times - t))))

    # ------------------------------------------------------------------ probes & picks
    def _probe_indices(self) -> list[int]:
        traces = self.ep.traces
        if traces is None:
            return []
        return [int(i) for i in self.surface.local_index(traces.node_ids)]

    def add_pick(self, point: int) -> None:
        """Plot the action potential of surface point ``point`` (keeps the last six)."""
        if point in self._picks or not 0 <= point < self.surface.n_points:
            return
        self._picks.append(int(point))
        del self._picks[:-MAX_PICKS]
        self._rebuild_traces()

    def clear_picks(self) -> None:
        self._picks.clear()
        self._rebuild_traces()

    @property
    def picks(self) -> list[int]:
        return list(self._picks)

    def _legend_toggled(self, key: str, visible: bool) -> None:
        (self._hidden.discard if visible else self._hidden.add)(key)
        curve = self._curves.get(key)
        if curve is not None:
            curve.setVisible(visible)
        self._update_markers()

    def _legend_removed(self, key: str) -> None:
        if key.startswith("pick:"):
            point = int(key.split(":", 1)[1])
            if point in self._picks:
                self._picks.remove(point)
                self._rebuild_traces()

    def _series(self) -> list[tuple[str, str, int, np.ndarray, np.ndarray, int, bool]]:
        """``(key, label, colour index, x, y, surface point, removable)`` for every trace."""
        out = []
        traces = self.ep.traces
        if traces is not None:
            for k in range(len(traces)):
                out.append((f"probe:{k}", traces.labels[k], k, traces.times, traces.values[k],
                            self._probe_points[k], False))
        offset = len(out)
        for j, point in enumerate(self._picks):
            node = int(self.surface.node_ids[point])
            out.append((f"pick:{point}", f"Node {node:,}", offset + j, self.ep.times, self.ep.vm.column(point),
                        point, True))
        return out

    def _rebuild_traces(self) -> None:
        palette = theme().palette
        for curve in self._curves.values():
            self.ap_plot.removeItem(curve)
        self._curves.clear()
        items = []
        for key, name, color_index, x, y, _point, removable in self._series():
            color = palette.series[color_index % len(palette.series)]
            curve = pg.PlotDataItem(x, y, pen=series_pen(color, 1.7 if removable else 1.5), name=name,
                                    antialias=True)
            curve.setVisible(key not in self._hidden)
            self.ap_plot.addItem(curve)
            self._curves[key] = curve
            items.append(LegendItem(key, name, color, removable, key not in self._hidden))
        self.legend.set_items(items)
        self.clear_button.setVisible(bool(self._picks))
        self._update_markers()

    def _update_markers(self) -> None:
        palette = theme().palette
        indices, colors = [], []
        for key, _name, color_index, _x, _y, point, _removable in self._series():
            if point >= 0 and key not in self._hidden:
                indices.append(point)
                colors.append(palette.series[color_index % len(palette.series)])
        self.viewer.set_markers(indices, colors)

    # ------------------------------------------------------------------ lifecycle
    def _on_theme(self, _palette: Palette) -> None:
        if not self._released:
            self._rebuild_traces()

    def set_active(self, active: bool) -> None:
        """Pause playback and rendering while hidden; resume when shown again."""
        if not active:
            self._was_playing = self.controller.playing
            self.controller.pause()
        elif self._was_playing:
            self._was_playing = False
            self.controller.play()
        self.viewer.set_active(active)

    def release(self) -> None:
        """Stop playback and free the 3D view's VTK resources."""
        self._released = True
        self.controller.pause()
        self.viewer.release()
