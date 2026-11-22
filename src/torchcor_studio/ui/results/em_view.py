"""Electromechanics results: the beating heart, pressure–volume loops and haemodynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

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
from torchcor_studio.ui.widgets import Card, SegmentedControl, label

CHAMBER_NAMES = {"lv": "LV", "rv": "RV", "la": "LA", "ra": "RA"}
CHAMBER_TITLES = {"lv": "Left ventricle", "rv": "Right ventricle", "la": "Left atrium", "ra": "Right atrium"}
#: The beat picker's value for every beat at once.
ALL_BEATS = "all"
#: Up to this many beats the picker is a row of segments; beyond it, a drop-down list.
BEAT_SEGMENTS = 7
#: Width of each loop's pressure axis, the same for all so that the four plots line up.
PV_AXIS_WIDTH = 46


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
        self._hidden: set[str] = set()  # chambers switched off in the traces' legend
        self._window = (-np.inf, np.inf)  # times of the beat(s) in the loops

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
        self._curves: dict[str, list[pg.PlotDataItem]] = {}  # chamber: [pressure, volume] over time
        self.loops: dict[str, pg.PlotDataItem] = {}
        self.pv_markers: dict[str, pg.ScatterPlotItem] = {}
        self.beat_picker: SegmentedControl | QComboBox | None = None
        self._beats = self.circulation.beats() if self.circulation is not None else []
        self._beat: int | str = len(self._beats) - 1  # the last beat, closest to steady state
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
        several = len(self._beats) > 1
        pv_card = Card(padding=16, spacing=10)
        pv_card.addWidget(card_title("Pressure–volume loops",
                                     "Pick a beat; the dots follow playback" if several else "The dots follow playback"))
        if several:
            pv_card.addLayout(self._beat_row())
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        columns = 2 if len(circ.chambers) > 1 else 1
        for i, chamber in enumerate(circ.chambers):
            plot = ThemedPlot(x_label="Volume", x_unit="mL", y_label="Pressure", y_unit="mmHg",
                              grid_x=True, min_height=150)
            plot.getPlotItem().getAxis("left").setWidth(PV_AXIS_WIDTH)
            loop = pg.PlotDataItem(name=CHAMBER_NAMES[chamber], antialias=True)
            marker = pg.ScatterPlotItem(size=9)
            marker.setZValue(30)
            plot.addItem(loop)
            plot.addItem(marker)
            cell = QVBoxLayout()
            cell.setSpacing(2)
            cell.addWidget(label(f"{CHAMBER_TITLES[chamber]} ({CHAMBER_NAMES[chamber]})", "field"))
            cell.addWidget(plot)
            grid.addLayout(cell, *divmod(i, columns))
            self.loops[chamber], self.pv_markers[chamber] = loop, marker
        pv_card.addLayout(grid)
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
        # the beats shown in the loops, shaded behind the traces
        self.beat_spans = [pg.LinearRegionItem(movable=False, pen=pg.mkPen(None)) for _ in range(2)]
        for plot, span in zip((self.p_plot, self.v_plot), self.beat_spans, strict=True):
            span.setZValue(-10)
            plot.addItem(span, ignoreBounds=True)
        self.legend = LegendChips()
        self.legend.toggled.connect(self._toggle_chamber)
        trace_card.addWidget(self.p_plot)
        trace_card.addWidget(self.v_plot)
        trace_card.addWidget(self.legend)
        self.side.add(trace_card)

        for chamber in circ.chambers:
            name = CHAMBER_NAMES[chamber]
            pressure = pg.PlotDataItem(circ.t, circ.pressure[chamber], name=name, antialias=True)
            volume = pg.PlotDataItem(circ.t, circ.volume[chamber], name=name, antialias=True)
            self.p_plot.addItem(pressure)
            self.v_plot.addItem(volume)
            self._curves[chamber] = [pressure, volume]
        self.set_beat(self._beat)

    def _beat_row(self) -> QHBoxLayout:
        """``Beat  1 2 3 4 5 All``: segments for a few beats, a drop-down list for many."""
        options = [(i, str(i + 1)) for i in range(len(self._beats))] + [(ALL_BEATS, "All")]
        if len(self._beats) <= BEAT_SEGMENTS:
            self.beat_picker = SegmentedControl(options, value=self._beat, compact=True)
            self.beat_picker.changed.connect(self.set_beat)
        else:
            self.beat_picker = picker = QComboBox()
            for value, text in options:
                picker.addItem("All beats" if value == ALL_BEATS else f"Beat {text}", value)
            picker.currentIndexChanged.connect(lambda index: self.set_beat(picker.itemData(index)))
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(label("Beat", "field"))
        row.addWidget(self.beat_picker)
        row.addStretch(1)
        return row

    def set_beat(self, beat: int | str) -> None:
        """Show beat ``beat`` (0-based) in the loops, or all of them (:data:`ALL_BEATS`)."""
        circ = self.circulation
        if circ is None or not self.loops:
            return
        self._beat = beat
        window = slice(None) if beat == ALL_BEATS else self._beats[beat]
        for chamber, loop in self.loops.items():
            loop.setData(circ.volume[chamber][window], circ.pressure[chamber][window])
        times = circ.t[window]
        self._window = (float(times[0]), float(times[-1]))
        for span in self.beat_spans:
            span.setRegion(self._window)
            span.setVisible(beat != ALL_BEATS and len(self._beats) > 1)
        picker = self.beat_picker
        if isinstance(picker, SegmentedControl) and picker.value() != beat:
            picker.set_value(beat, animate=False)
        elif isinstance(picker, QComboBox) and picker.currentData() != beat:
            picker.blockSignals(True)
            picker.setCurrentIndex(picker.findData(beat))
            picker.blockSignals(False)
        self._show_markers(self.controller.position)

    @property
    def beat(self) -> int | str:
        return self._beat

    def _toggle_chamber(self, key: str, visible: bool) -> None:
        (self._hidden.discard if visible else self._hidden.add)(key)
        for curve in self._curves.get(key, []):
            curve.setVisible(visible)

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
        """Open at peak contraction (largest mean displacement) of the beat in the loops."""
        mean = np.linalg.norm(self.em.displacement, axis=2).mean(axis=1)
        shown = (self.em.times >= self._window[0]) & (self.em.times <= self._window[1])
        if shown.any():
            mean = np.where(shown, mean, -np.inf)
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
        self._show_markers(position)

    def _show_markers(self, position: float) -> None:
        """The playback time on the traces, and as a dot on each loop while it is shown."""
        if self.circulation is None or not self.loops:
            return
        circ = self.circulation
        t = float(np.interp(position, self._frame_index, self.em.times))
        self.p_cursor.set_value(t)
        self.v_cursor.set_value(t)
        shown = self._window[0] <= t <= self._window[1]
        p = theme().palette
        for i, chamber in enumerate(circ.chambers):
            marker = self.pv_markers[chamber]
            if not shown:
                marker.setData([], [])
                continue
            marker.setData([float(np.interp(t, circ.t, circ.volume[chamber]))],
                           [float(np.interp(t, circ.t, circ.pressure[chamber]))],
                           brush=pg.mkBrush(QColor(p.series[i % len(p.series)])),
                           pen=pg.mkPen(QColor(p.bg2), width=2))

    # ------------------------------------------------------------------ lifecycle
    def _on_theme(self, p: Palette) -> None:
        if self.circulation is None or not self.loops:
            return
        items = []
        for i, chamber in enumerate(self.circulation.chambers):
            color = p.series[i % len(p.series)]
            self.loops[chamber].setPen(series_pen(color, 1.8))
            for curve in self._curves[chamber]:
                curve.setPen(series_pen(color, 1.5))
            items.append(LegendItem(chamber, CHAMBER_NAMES[chamber], color, visible=chamber not in self._hidden))
        self.legend.set_items(items)
        shade = QColor(p.accent)
        shade.setAlphaF(0.10 if p.dark else 0.07)
        for span in self.beat_spans:
            span.setBrush(pg.mkBrush(shade))
        if not self._released:
            self._show_markers(self.controller.position)

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
