"""Electromechanics results: the beating heart, pressure–volume loops and haemodynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from torchcor_studio.core.results import ResultBundle, cardiac_summary
from torchcor_studio.ui.results.common import (
    VIEW_MARGINS,
    ColormapCombo,
    SidePanel,
    Stage,
    StatGrid,
    card_title,
    nice_bounds,
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
from torchcor_studio.ui.widgets import Card, SegmentedControl

CHAMBER_NAMES = {"lv": "LV", "rv": "RV", "la": "LA", "ra": "RA"}


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    title: str
    unit: str
    cmap: str


class EMView(QWidget):
    """EM bundle view. ``set_active`` pauses playback while hidden; ``release`` frees VTK."""

    def __init__(self, bundle: ResultBundle, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bundle = bundle
        self.surface = bundle.surface
        self.em = bundle.em
        self.circulation = self.em.circulation
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.fields = [f for f, present in (
            (FieldSpec("voltage", "Voltage", "Transmembrane voltage", "mV", "torchcor"), self.em.vm is not None),
            (FieldSpec("tension", "Tension", "Active tension", "kPa", "inferno"), self.em.tension is not None),
            (FieldSpec("displacement", "Displacement", "Displacement", "mm", "viridis"), True),
        ) if present]
        self._field = self.fields[0].key
        self._cmaps = {f.key: f.cmap for f in self.fields}
        peak = max(float(np.linalg.norm(u, axis=1).max(initial=0.0)) for u in self.em.displacement)
        self._clims = {"displacement": nice_bounds(0.0, max(peak, 1e-6), target=8)}
        if self.em.vm is not None:
            self._clims["voltage"] = nice_bounds(*self.em.vm.range, target=16)
        if self.em.tension is not None:
            self._clims["tension"] = nice_bounds(0.0, max(self.em.tension.range[1], 1e-6), target=8)
        self._was_playing = False
        self._released = False
        n_points = self.surface.n_points
        self._u = np.empty((n_points, 3), np.float32)
        self._points = np.empty((n_points, 3), np.float32)
        self._values = np.empty(n_points, np.float32)
        self._frame_index = np.arange(self.em.n_frames, dtype=np.float64)
        self._hidden: set[str] = set()

        # ------------------------------------------------------------------ viewport
        n = self.em.n_frames
        self.controller = PlaybackController(n, fps=min(30.0, max(2.0, n / 4.0)), parent=self)
        self.viewer = Viewer3D(export_name="torchcor-em")
        self.viewer.set_mesh(self.surface.points, self.surface.triangles)
        self.timeline = TimelineBar(self.controller, self._time_text, shortcut_scope=self)
        self.stage = Stage(self.viewer, self.timeline)
        self.field_control = SegmentedControl([(f.key, f.label) for f in self.fields], value=self._field)
        self.field_control.changed.connect(self.set_field)
        self.cmap_combo = ColormapCombo(self._cmaps[self._field])
        self.cmap_combo.colormap_changed.connect(self._set_cmap)
        edges_holder, self.edges_switch = switch("Edges", False, self.viewer.set_edges, "Show the mesh triangles")
        self.viewer.edges_changed.connect(lambda on: self.edges_switch.isChecked() != on and
                                          self.edges_switch.setChecked(on))

        # ------------------------------------------------------------------ side panel
        self.side = SidePanel()
        self.side.add(panel_header("Electromechanics", bundle.title or "Beating heart"))
        self.stats = self.side.add(StatGrid(cardiac_summary(bundle.summary, self.circulation)))
        self._curves: dict[str, list[pg.PlotDataItem]] = {}
        self.pv_markers = pg.ScatterPlotItem(size=10)
        if self.circulation is not None and self.circulation.chambers:
            self._build_circulation()
        if bundle.notes:
            self.side.add(notes_card(bundle.notes))

        root = QVBoxLayout(self)
        root.setContentsMargins(*VIEW_MARGINS)
        root.setSpacing(14)
        root.addLayout(toolbar(self.field_control, self.cmap_combo, edges_holder, None))
        root.addWidget(split(self.stage, self.side), 1)

        self.controller.position_changed.connect(self._show)
        theme().changed.connect(self._on_theme)
        self._on_theme(theme().palette)
        self.set_field(self._field)
        self.controller.seek(self._initial_frame())
        self._show(self.controller.position)

    # ------------------------------------------------------------------ circulation cards
    def _build_circulation(self) -> None:
        circ = self.circulation
        pv_card = Card(padding=16, spacing=10)
        pv_card.addWidget(card_title("Pressure–volume loops", "One loop per beat; the dots follow playback"))
        self.pv_plot = ThemedPlot(x_label="Volume", x_unit="mL", y_label="Pressure", y_unit="mmHg",
                                  grid_x=True, min_height=230)
        self.pv_markers.setZValue(30)
        self.pv_plot.addItem(self.pv_markers)
        self.legend = LegendChips()
        self.legend.toggled.connect(self._toggle_chamber)
        pv_card.addWidget(self.pv_plot)
        pv_card.addWidget(self.legend)
        self.side.add(pv_card)

        trace_card = Card(padding=16, spacing=10)
        trace_card.addWidget(card_title("Pressure and volume", "Drag the cursor to scrub"))
        self.p_plot = ThemedPlot(y_label="Pressure", y_unit="mmHg", interactive=True, min_height=120)
        self.v_plot = ThemedPlot(x_label="Time", x_unit="ms", y_label="Volume", y_unit="mL", interactive=True,
                                 min_height=130)
        self.v_plot.setXLink(self.p_plot)
        self.p_plot.getPlotItem().getAxis("bottom").setStyle(showValues=False)
        self.p_cursor = TimeCursor(self.p_plot, on_drag=self._seek_time)
        self.v_cursor = TimeCursor(self.v_plot, on_drag=self._seek_time)
        HoverReadout(self.p_plot, lambda name, x, y: f"{name}  ·  {x:.0f} ms  ·  {y:.0f} mmHg")
        HoverReadout(self.v_plot, lambda name, x, y: f"{name}  ·  {x:.0f} ms  ·  {y:.0f} mL")
        trace_card.addWidget(self.p_plot)
        trace_card.addWidget(self.v_plot)
        self.side.add(trace_card)

        for chamber in circ.chambers:
            name = CHAMBER_NAMES[chamber]
            loop = pg.PlotDataItem(circ.volume[chamber], circ.pressure[chamber], name=name, antialias=True)
            pressure = pg.PlotDataItem(circ.t, circ.pressure[chamber], name=name, antialias=True)
            volume = pg.PlotDataItem(circ.t, circ.volume[chamber], name=name, antialias=True)
            self.pv_plot.addItem(loop)
            self.p_plot.addItem(pressure)
            self.v_plot.addItem(volume)
            self._curves[chamber] = [loop, pressure, volume]

    def _toggle_chamber(self, key: str, visible: bool) -> None:
        (self._hidden.discard if visible else self._hidden.add)(key)
        for curve in self._curves.get(key, []):
            curve.setVisible(visible)
        self._show(self.controller.position)

    # ------------------------------------------------------------------ fields
    def set_field(self, key: str) -> None:
        spec = next((f for f in self.fields if f.key == key), None)
        if spec is None:
            return
        self._field = key
        self.field_control.set_value(key, animate=False)
        self.cmap_combo.set_value(self._cmaps[key])
        self.viewer.set_scalars(self._field_values(self.controller.position), self._clims[key], self._cmaps[key],
                                spec.title, unit=spec.unit)

    @property
    def field(self) -> str:
        return self._field

    def _set_cmap(self, name: str) -> None:
        self._cmaps[self._field] = name
        self.viewer.set_scalars(self._values, cmap=name)

    def _field_values(self, position: float) -> np.ndarray:
        if self._field == "voltage":
            return self.em.vm.blend(position, out=self._values)
        if self._field == "tension":
            return self.em.tension.blend(position, out=self._values)
        u = self.em.blend_displacement(position, out=self._u)
        np.sqrt(np.einsum("ij,ij->i", u, u), out=self._values)
        return self._values

    # ------------------------------------------------------------------ playback
    def _time_text(self, frame: int) -> str:
        return f"t = {float(self.em.times[frame]):.0f} ms"

    def _initial_frame(self) -> int:
        """Open at peak contraction (largest mean displacement)."""
        mean = np.linalg.norm(self.em.displacement, axis=2).mean(axis=1)
        return int(np.argmax(mean)) if len(mean) else 0

    def _seek_time(self, t: float) -> None:
        self.controller.seek(int(np.argmin(np.abs(self.em.times - t))))

    def _show(self, position: float) -> None:
        if self._released:
            return
        u = self.em.blend_displacement(position, out=self._u)
        np.add(self.surface.points, u, out=self._points)
        self.viewer.set_points(self._points)
        self.viewer.set_scalars(self._field_values(position))
        if self.circulation is None or not self._curves:
            return
        t = float(np.interp(position, self._frame_index, self.em.times))
        self.p_cursor.set_value(t)
        self.v_cursor.set_value(t)
        p = theme().palette
        xs, ys, brushes = [], [], []
        for i, chamber in enumerate(self.circulation.chambers):
            if chamber in self._hidden:
                continue
            xs.append(float(np.interp(t, self.circulation.t, self.circulation.volume[chamber])))
            ys.append(float(np.interp(t, self.circulation.t, self.circulation.pressure[chamber])))
            brushes.append(pg.mkBrush(QColor(p.series[i % len(p.series)])))
        self.pv_markers.setData(xs, ys, brush=brushes, pen=pg.mkPen(QColor(p.bg2), width=2))

    # ------------------------------------------------------------------ lifecycle
    def _on_theme(self, p: Palette) -> None:
        if self.circulation is None:
            return
        items = []
        for i, chamber in enumerate(self.circulation.chambers):
            color = p.series[i % len(p.series)]
            for curve, width in zip(self._curves.get(chamber, []), (1.8, 1.5, 1.5), strict=False):
                curve.setPen(series_pen(color, width))
            items.append(LegendItem(chamber, CHAMBER_NAMES[chamber], color, visible=chamber not in self._hidden))
        if self._curves:
            self.legend.set_items(items)
            if not self._released:
                self._show(self.controller.position)

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
