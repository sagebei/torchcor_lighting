"""Mechanics results: the deformed surface over load (or time) steps, convergence and peak |u|."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QSlider, QVBoxLayout, QWidget

from torchcor_studio.core.results import ResultBundle
from torchcor_studio.ui.results.common import (
    VIEW_MARGINS,
    ColormapCombo,
    SidePanel,
    Stage,
    StatGrid,
    card_title,
    frame_text,
    nice_bounds,
    notes_card,
    panel_header,
    prettify,
    split,
    switch,
    toolbar,
)
from torchcor_studio.ui.results.playback import PlaybackController, TimelineBar
from torchcor_studio.ui.results.plots import HoverReadout, ThemedPlot, TimeCursor, series_pen
from torchcor_studio.ui.results.viewer3d import Viewer3D
from torchcor_studio.ui.theme import Palette, mono_font, theme
from torchcor_studio.ui.widgets import Card, SegmentedControl, label

#: Front three-quarter view, z up: shows bending in the vertical plane well.
FRONT_VIEW = ((0.42, -1.0, 0.38), (0.0, 0.0, 1.0))
MAX_WARP = 10.0


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    title: str
    unit: str
    cmap: str


class MechanicsView(QWidget):
    """Mechanics bundle view. ``set_active`` pauses playback while hidden; ``release`` frees VTK."""

    def __init__(self, bundle: ResultBundle, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bundle = bundle
        self.surface = bundle.surface
        self.mech = bundle.mechanics
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.fields = [FieldSpec("displacement", "|u|", "Displacement", "mm", "viridis")] + [
            FieldSpec(name, prettify(name), prettify(name), "", "inferno") for name in self.mech.fields]
        self._field = "displacement"
        self._cmaps = {f.key: f.cmap for f in self.fields}
        peak = float(self.mech.max_displacement.max(initial=0.0))
        self._clims = {"displacement": nice_bounds(0.0, max(peak, 1e-6), target=8)}
        for name, stack in self.mech.fields.items():
            self._clims[name] = nice_bounds(*stack.range, target=8)
        self._warp = 1.0
        self._was_playing = False
        self._released = False
        n_points = self.surface.n_points
        self._u = np.empty((n_points, 3), np.float32)
        self._points = np.empty((n_points, 3), np.float32)
        self._values = np.empty(n_points, np.float32)
        self._frame_index = np.arange(self.mech.n_frames, dtype=np.float64)

        # ------------------------------------------------------------------ viewport
        n = self.mech.n_frames
        self.controller = PlaybackController(n, fps=min(24.0, max(1.0, (n - 1) / 4.0)), parent=self)
        self.viewer = Viewer3D(export_name="torchcor-mechanics")
        self.viewer.set_mesh(self.surface.points, self.surface.triangles, view=FRONT_VIEW)
        self.timeline = TimelineBar(self.controller, self._frame_text, shortcut_scope=self)
        self.stage = Stage(self.viewer, self.timeline)
        if n < 2:
            self.stage.set_footer_visible(False)

        # ------------------------------------------------------------------ toolbar
        if len(self.fields) <= 4:
            self.field_control = SegmentedControl([(f.key, f.label) for f in self.fields], value=self._field)
            self.field_control.changed.connect(self.set_field)
        else:
            self.field_control = QComboBox()
            for f in self.fields:
                self.field_control.addItem(f.label, f.key)
            self.field_control.currentIndexChanged.connect(
                lambda *_: self.set_field(self.field_control.currentData()))
        if len(self.fields) < 2:
            self.field_control.hide()
        self.cmap_combo = ColormapCombo(self._cmaps[self._field])
        self.cmap_combo.colormap_changed.connect(self._set_cmap)
        self.warp_slider = QSlider(Qt.Orientation.Horizontal)
        self.warp_slider.setRange(0, int(MAX_WARP * 10))
        self.warp_slider.setValue(10)
        self.warp_slider.setFixedWidth(132)
        self.warp_slider.setToolTip("Scale the displacement for display (0–10×)")
        self.warp_slider.valueChanged.connect(lambda v: self.set_warp(v / 10.0))
        self.warp_value = label("1.0×", "mono")
        self.warp_value.setFont(mono_font(12))
        self.warp_value.setFixedWidth(40)
        warp = QWidget()
        wl = QHBoxLayout(warp)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(8)
        wl.addWidget(label("Warp", "muted"))
        wl.addWidget(self.warp_slider)
        wl.addWidget(self.warp_value)
        ref_holder, self.reference_switch = switch("Reference", False, self.set_reference_visible,
                                                   "Show the undeformed geometry")
        edges_holder, self.edges_switch = switch("Edges", False, self.viewer.set_edges, "Show the mesh triangles")
        self.viewer.edges_changed.connect(lambda on: self.edges_switch.isChecked() != on and
                                          self.edges_switch.setChecked(on))

        # ------------------------------------------------------------------ side panel
        self.side = SidePanel()
        self.side.add(panel_header("Mechanics", bundle.title or "Deformation"))
        self.stats = self.side.add(StatGrid(bundle.summary))
        unit = f" ({self.mech.frame_unit})" if self.mech.frame_unit else ""
        x_label = self.mech.frame_label
        self._build_convergence(x_label, unit)
        disp_card = Card(padding=16, spacing=10)
        disp_card.addWidget(card_title("Peak displacement", f"Largest |u| on the surface per {x_label.lower()}{unit}"))
        self.disp_plot = ThemedPlot(x_label=x_label, x_unit=self.mech.frame_unit, y_label="|u|", y_unit="mm",
                                    min_height=150)
        self.disp_curve = pg.PlotDataItem(self.mech.frames, self.mech.max_displacement, name="Peak |u|",
                                          symbol="o" if self.mech.n_frames <= 40 else None, symbolSize=5)
        self.disp_plot.addItem(self.disp_curve)
        self.disp_dot = pg.ScatterPlotItem(size=10)
        self.disp_dot.setZValue(25)
        self.disp_plot.addItem(self.disp_dot)
        self.disp_cursor = TimeCursor(self.disp_plot, on_drag=self._seek_value)
        HoverReadout(self.disp_plot, lambda _n, x, y: f"{frame_text(x_label, self.mech.frame_unit, x)}  ·  {y:.2f} mm")
        disp_card.addWidget(self.disp_plot)
        self.side.add(disp_card)
        if bundle.notes:
            self.side.add(notes_card(bundle.notes))

        root = QVBoxLayout(self)
        root.setContentsMargins(*VIEW_MARGINS)
        root.setSpacing(14)
        root.addLayout(toolbar(self.field_control, self.cmap_combo, warp, ref_holder, edges_holder, None))
        root.addWidget(split(self.stage, self.side), 1)

        self.controller.position_changed.connect(self._show)
        theme().changed.connect(self._on_theme)
        self._on_theme(theme().palette)
        self.set_field(self._field)
        self.controller.seek(n - 1)
        self._show(self.controller.position)

    # ------------------------------------------------------------------ side cards
    def _build_convergence(self, x_label: str, unit: str) -> None:
        conv = self.mech.convergence
        self.conv_card = Card(padding=16, spacing=10)
        self.conv_card.addWidget(card_title("Newton convergence", f"Iterations per step over {x_label.lower()}{unit}"))
        self.conv_plot = ThemedPlot(x_label=x_label, x_unit=self.mech.frame_unit, y_label="Iterations",
                                    min_height=140)
        self.conv_done = pg.BarGraphItem(x=[], height=[], width=1.0, pen=pg.mkPen(None))
        self.conv_todo = pg.BarGraphItem(x=[], height=[], width=1.0, pen=pg.mkPen(None))
        self.conv_plot.addItem(self.conv_todo)
        self.conv_plot.addItem(self.conv_done)
        self._conv_done = -1
        self.conv_cursor = TimeCursor(self.conv_plot)
        self.conv_card.addWidget(self.conv_plot)
        self.side.add(self.conv_card)
        if conv is None or not len(conv.step):
            self.conv_card.hide()
            return
        x = conv.value.astype(np.float64)
        if len(x) > 1 and not np.all(np.diff(x) > 0):
            x = conv.step.astype(np.float64)
        self._conv_x = x
        spacing = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
        self._conv_width = spacing * 0.62
        self.conv_plot.getPlotItem().setYRange(0, max(1, int(conv.newton.max())) + 1, padding=0)
        HoverReadout(self.conv_plot, lambda _n, xv, y: f"step at {xv:g}  ·  {y:.0f} iterations")

    def _paint_convergence(self, value: float, force: bool = False) -> None:
        """Steps up to the current frame in the accent colour, later ones muted."""
        conv = self.mech.convergence
        if conv is None or not len(conv.step):
            return
        done = int(np.searchsorted(self._conv_x, value + 1e-9))
        if done == self._conv_done and not force:
            return
        self._conv_done = done
        p = theme().palette
        future = QColor(p.text3)
        future.setAlphaF(0.35)
        height = conv.newton.astype(np.float64)
        self.conv_done.setOpts(x=self._conv_x[:done], height=height[:done], width=self._conv_width,
                               brush=pg.mkBrush(QColor(p.accent)))
        self.conv_todo.setOpts(x=self._conv_x[done:], height=height[done:], width=self._conv_width,
                               brush=pg.mkBrush(future))

    # ------------------------------------------------------------------ fields
    def set_field(self, key: str) -> None:
        spec = next((f for f in self.fields if f.key == key), None)
        if spec is None:
            return
        self._field = key
        if isinstance(self.field_control, SegmentedControl):
            self.field_control.set_value(key, animate=False)
        else:
            self.field_control.setCurrentIndex(self.field_control.findData(key))
        self.cmap_combo.set_value(self._cmaps[key])
        self.viewer.set_scalars(self._field_values(self.controller.position), self._clims[key], self._cmaps[key],
                                spec.title, unit=spec.unit)

    @property
    def field(self) -> str:
        return self._field

    def _set_cmap(self, name: str) -> None:
        self._cmaps[self._field] = name
        self.viewer.set_scalars(self._values, cmap=name)

    def set_warp(self, scale: float) -> None:
        """Display the displacement ``scale`` times (0–10×)."""
        self._warp = min(max(float(scale), 0.0), MAX_WARP)
        self.warp_value.setText(f"{self._warp:.1f}×")
        if self.warp_slider.value() != round(self._warp * 10):
            self.warp_slider.blockSignals(True)
            self.warp_slider.setValue(round(self._warp * 10))
            self.warp_slider.blockSignals(False)
        self._show(self.controller.position)

    def set_reference_visible(self, visible: bool) -> None:
        self.viewer.set_reference(self.surface.points if visible else None)
        if self.reference_switch.isChecked() != visible:
            self.reference_switch.setChecked(visible)

    def _field_values(self, position: float) -> np.ndarray:
        if self._field == "displacement":
            u = self.mech.blend_displacement(position, out=self._u)
            np.sqrt(np.einsum("ij,ij->i", u, u), out=self._values)
            return self._values
        return self.mech.fields[self._field].blend(position, out=self._values)

    # ------------------------------------------------------------------ playback
    def _frame_text(self, frame: int) -> str:
        return frame_text(self.mech.frame_label, self.mech.frame_unit, float(self.mech.frames[frame]))

    def _value_at(self, position: float) -> float:
        return float(np.interp(position, self._frame_index, self.mech.frames))

    def _seek_value(self, value: float) -> None:
        self.controller.seek(int(np.argmin(np.abs(self.mech.frames - value))))

    def _show(self, position: float) -> None:
        if self._released:
            return
        u = self.mech.blend_displacement(position, out=self._u)
        np.multiply(u, np.float32(self._warp), out=self._points)
        self._points += self.surface.points
        self.viewer.set_points(self._points)
        self.viewer.set_scalars(self._field_values(position))
        value = self._value_at(position)
        peak = float(np.interp(position, self._frame_index, self.mech.max_displacement))
        self.disp_cursor.set_value(value)
        self.disp_dot.setData([value], [peak])
        self.conv_cursor.set_value(value)
        self._paint_convergence(value)

    # ------------------------------------------------------------------ lifecycle
    def _on_theme(self, p: Palette) -> None:
        accent = p.series[0]
        self.disp_curve.setPen(series_pen(accent, 1.8))
        self.disp_curve.setSymbolBrush(pg.mkBrush(QColor(accent)))
        self.disp_curve.setSymbolPen(pg.mkPen(QColor(p.bg2), width=1))
        self.disp_dot.setBrush(pg.mkBrush(QColor(p.ember)))
        self.disp_dot.setPen(pg.mkPen(QColor(p.bg2), width=2))
        self._paint_convergence(self._value_at(self.controller.position), force=True)

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
