"""The 3D surface viewport, styled from the design tokens.

:class:`Viewer3D` draws one triangulated surface coloured by a point field. It prefers
pyvistaqt's interactive ``QtInteractor``; when Qt cannot create an OpenGL context
(headless CI, some VMs and remote desktops) it renders VTK offscreen into an image with
the same orbit/pan/zoom/pick behaviour, and when neither works it shows an
:class:`EmptyState`. Set ``TCS_DISABLE_3D=1`` to skip 3D entirely.

Per-frame updates (:meth:`Viewer3D.set_scalars`, :meth:`Viewer3D.set_points`) write
into the VTK arrays in place and coalesce renders to one per event-loop pass; no actor
is ever rebuilt per frame. The colour legend and the tool buttons are Qt overlays, so
they use the app's typography and follow the theme.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QStandardPaths, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QImage,
    QOpenGLContext,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QFileDialog, QFrame, QStackedLayout, QVBoxLayout, QWidget
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkRenderingCore import vtkCellPicker

from torchcor_studio.core.results import MINUS
from torchcor_studio.ui.results import colormaps
from torchcor_studio.ui.results.common import nice_step
from torchcor_studio.ui.theme import Palette, theme
from torchcor_studio.ui.widgets import EmptyState, IconButton, ToastHost

__all__ = ["ColorBar", "Viewer3D", "opengl_available", "vertex_normals"]

log = logging.getLogger(__name__)

CLICK_SLOP = 4  # px a press may travel and still count as a click
ZOOM_STEP = 1.12

#: Camera presets: "auto" frames a heart by its principal axes; "iso" is the classic view.
ISO_VIEW = ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0))


@lru_cache(maxsize=1)
def opengl_available() -> bool:
    """Whether Qt can create an OpenGL context here (the interactive viewport needs one)."""
    if QGuiApplication.platformName() in ("offscreen", "minimal", "vnc"):
        return False
    return bool(QOpenGLContext().create())


def vertex_normals(points: np.ndarray, triangles: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Area-weighted unit vertex normals ``(N, 3)`` float32 (written into ``out`` if given)."""
    p = np.asarray(points, dtype=np.float32)
    a, b, c = p[triangles[:, 0]], p[triangles[:, 1]], p[triangles[:, 2]]
    face = np.cross(b - a, c - a)
    n = len(p)
    acc = np.zeros((n, 3), np.float64)
    for k in range(3):
        idx = triangles[:, k]
        for axis in range(3):
            acc[:, axis] += np.bincount(idx, weights=face[:, axis], minlength=n)
    length = np.linalg.norm(acc, axis=1, keepdims=True)
    length[length == 0] = 1.0
    acc /= length
    if out is None:
        return acc.astype(np.float32)
    np.copyto(out, acc)
    return out


def _format_tick(value: float, step: float) -> str:
    decimals = max(0, -math.floor(math.log10(step))) if step > 0 and step < 1 else 0
    if step >= 1 and abs(value - round(value)) > 1e-6 and step < 10:
        decimals = 1
    text = f"{value:.{decimals}f}"
    if text.startswith("-"):
        text = MINUS + text[1:]
    return "0" if text in ("−0", "-0") else text


# =============================================================================== overlays
class ColorBar(QWidget):
    """Compact horizontal colour legend: title, unit, gradient and a few ticks."""

    W, H = 236, 62

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFixedSize(self.W, self.H)
        self.title, self.unit = "", ""
        self.clim = (0.0, 1.0)
        self.cmap, self.reverse = colormaps.DEFAULT, False
        theme().changed.connect(self._repaint)

    def set(self, title: str, unit: str, clim: tuple[float, float], cmap: str, reverse: bool = False) -> None:
        state = (title, unit, (float(clim[0]), float(clim[1])), cmap, bool(reverse))
        if state != (self.title, self.unit, self.clim, self.cmap, self.reverse):
            self.title, self.unit, self.clim, self.cmap, self.reverse = state
            self.update()

    def _repaint(self, _palette=None) -> None:
        self.update()

    def ticks(self, width: float) -> list[tuple[float, str]]:
        """``(fraction, text)`` pairs: both ends plus nice interior values that fit."""
        lo, hi = self.clim
        span = hi - lo
        if span <= 0 or not math.isfinite(span):
            return [(0.0, _format_tick(lo, 1.0))]
        step = nice_step(span, 4)
        out = [(0.0, _format_tick(lo, step)), (1.0, _format_tick(hi, step))]
        value = math.ceil(lo / step + 1e-9) * step
        while value < hi - 1e-9:
            x = (value - lo) / span
            if x * width > 34 and (1 - x) * width > 34:
                out.append((x, _format_tick(value, step)))
            value += step
        return sorted(out)

    def paint_into(self, painter: QPainter, rect: QRectF) -> None:
        p = theme().palette
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        glass = QColor(p.bg2)
        glass.setAlphaF(0.84 if p.dark else 0.9)
        border = QColor(p.line2 if p.dark else p.line)
        painter.setPen(QPen(border, 1))
        painter.setBrush(glass)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)

        pad = 12.0
        title_font = QFont()
        title_font.setPixelSize(12)
        title_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QColor(p.text))
        fm = QFontMetrics(title_font)
        title = fm.elidedText(self.title, Qt.TextElideMode.ElideRight, int(rect.width() - 2 * pad - 40))
        baseline = rect.top() + pad + fm.ascent() - 2
        painter.drawText(QPointF(rect.left() + pad, baseline), title)
        if self.unit:
            unit_font = QFont()
            unit_font.setPixelSize(11)
            painter.setFont(unit_font)
            painter.setPen(QColor(p.text3))
            painter.drawText(QPointF(rect.left() + pad + fm.horizontalAdvance(title) + 6, baseline), self.unit)

        bar = QRectF(rect.left() + pad, rect.top() + 28, rect.width() - 2 * pad, 8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colormaps.gradient(self.cmap, bar.topLeft(), bar.topRight(), self.reverse))
        painter.drawRoundedRect(bar, 4, 4)

        tick_font = QFont()
        tick_font.setPixelSize(11)
        painter.setFont(tick_font)
        tfm = QFontMetrics(tick_font)
        y = bar.bottom() + 4 + tfm.ascent()
        for x, text in self.ticks(bar.width()):
            cx = bar.left() + x * bar.width()
            w = tfm.horizontalAdvance(text)
            left = cx if x == 0.0 else cx - w if x == 1.0 else cx - w / 2
            if 0.0 < x < 1.0:
                mark = QColor(p.text3)
                mark.setAlphaF(0.7)
                painter.setPen(QPen(mark, 1))
                painter.drawLine(QPointF(cx, bar.bottom() + 1), QPointF(cx, bar.bottom() + 3))
            painter.setPen(QColor(p.text2))
            painter.drawText(QPointF(left, y), text)
        painter.restore()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        self.paint_into(painter, QRectF(self.rect()))


class _ToolPill(QFrame):
    """Floating glass column of viewport tools."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(4, 4, 4, 4)
        self._lay.setSpacing(2)
        theme().changed.connect(self._repaint)

    def add(self, button: QWidget) -> None:
        self._lay.addWidget(button)
        self.adjustSize()

    def _repaint(self, _palette=None) -> None:
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        glass = QColor(p.bg2)
        glass.setAlphaF(0.84 if p.dark else 0.9)
        painter.setPen(QPen(QColor(p.line2 if p.dark else p.line), 1))
        painter.setBrush(glass)
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)


# =============================================================================== canvases
class _InteractiveCanvas(QObject):
    """pyvistaqt's OpenGL interactor; reports clicks and keeps VTK's key bindings out."""

    clicked = Signal(QPointF)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        os.environ.setdefault("QT_API", "pyside6")  # pyvistaqt reaches Qt through qtpy
        from pyvistaqt import QtInteractor

        self.plotter = QtInteractor(parent, auto_update=False, multi_samples=8)
        self.widget: QWidget = self.plotter.interactor
        self.widget.installEventFilter(self)
        self._press: QPointF | None = None

    def render(self) -> None:
        self.plotter.render()

    def snapshot(self) -> np.ndarray:
        """The current scene as ``(h, w, 3)`` uint8, rendered now (not the last frame shown)."""
        self.plotter.render()  # marks a render pending; screenshot() then runs it synchronously
        return np.ascontiguousarray(self.plotter.screenshot(return_img=True)[:, :, :3])

    def close(self) -> None:
        self.widget.removeEventFilter(self)
        self.plotter.close()

    def eventFilter(self, _obj, event) -> bool:  # noqa: N802
        kind = event.type()
        if kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return True  # VTK's single-letter bindings (w, s, q, …) would restyle or close the scene
        if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._press = event.position()
        elif kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if self._press is not None and (event.position() - self._press).manhattanLength() <= CLICK_SLOP:
                self.clicked.emit(event.position())
            self._press = None
        return False


class _ImageCanvas(QWidget):
    """VTK rendered offscreen into an image, for displays without Qt OpenGL."""

    clicked = Signal(QPointF)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
        self.plotter.enable_anti_aliasing("msaa", multi_samples=8)
        self.widget: QWidget = self
        self._image: QImage | None = None
        self._dirty = True
        self._press: QPointF | None = None
        self._last: QPointF | None = None
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def render(self) -> None:
        self._dirty = True
        self.update()

    def close(self) -> None:
        self.plotter.close()

    def sync_size(self) -> None:
        """Match the offscreen render window to the widget (in device pixels)."""
        dpr = self.devicePixelRatioF()
        size = [max(16, round(self.width() * dpr)), max(16, round(self.height() * dpr))]
        if list(self.plotter.window_size) != size:
            self.plotter.window_size = size

    def snapshot(self) -> np.ndarray:
        """The current scene as ``(h, w, 3)`` uint8, rendered now.

        pyvista's ``screenshot()`` renders only on its very first call and afterwards reads
        back the last frame, so render explicitly first (a no-op before that first call).
        """
        self.sync_size()
        self.plotter.render()
        return np.ascontiguousarray(self.plotter.screenshot(return_img=True)[:, :, :3])

    def _refresh(self) -> None:
        dpr = self.devicePixelRatioF()
        rgb = self.snapshot()
        h, w = rgb.shape[:2]
        self._image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._image.setDevicePixelRatio(dpr)
        self._dirty = False

    def paintEvent(self, _event) -> None:  # noqa: N802
        if self._dirty or self._image is None:
            self._refresh()
        painter = QPainter(self)
        painter.drawImage(QPointF(0, 0), self._image)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.sync_size()
        self._dirty = True

    # ------------------------------------------------------------------ camera
    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._press = self._last = event.position()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._last is None:
            return
        delta = event.position() - self._last
        self._last = event.position()
        camera = self.plotter.camera
        orbit = event.buttons() & Qt.MouseButton.LeftButton and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        if orbit:
            camera.Azimuth(-delta.x() * 0.45)
            camera.Elevation(delta.y() * 0.45)
            camera.OrthogonalizeViewUp()
        else:
            self._pan(delta)
        self.plotter.renderer.ResetCameraClippingRange()
        self.render()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._press is not None and event.button() == Qt.MouseButton.LeftButton and \
                (event.position() - self._press).manhattanLength() <= CLICK_SLOP:
            self.clicked.emit(event.position())
        self._press = self._last = None

    def wheelEvent(self, event) -> None:  # noqa: N802
        self.plotter.camera.Zoom(ZOOM_STEP ** (event.angleDelta().y() / 120.0))
        self.plotter.renderer.ResetCameraClippingRange()
        self.render()

    def _pan(self, delta: QPointF) -> None:
        camera = self.plotter.camera
        focal, position = np.array(camera.GetFocalPoint()), np.array(camera.GetPosition())
        up = np.array(camera.GetViewUp())
        direction = focal - position
        distance = float(np.linalg.norm(direction)) or 1.0
        right = np.cross(direction / distance, up)
        scale = 2 * distance * math.tan(math.radians(camera.GetViewAngle()) / 2) / max(1, self.height())
        shift = (-delta.x() * right + delta.y() * up) * scale
        camera.SetFocalPoint(*(focal + shift))
        camera.SetPosition(*(position + shift))


def _create_canvas(parent: QWidget) -> _InteractiveCanvas | _ImageCanvas | None:
    if os.environ.get("TCS_DISABLE_3D"):
        return None
    if opengl_available():
        return _InteractiveCanvas(parent)
    return _ImageCanvas(parent)


# =============================================================================== viewer
class Viewer3D(QWidget):
    """A themed VTK viewport for one surface. See the module docstring."""

    point_picked = Signal(int)
    edges_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None, *, export_name: str = "torchcor") -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 200)
        self.export_name = export_name
        self._canvas: _InteractiveCanvas | _ImageCanvas | None = None
        self._mesh: pv.PolyData | None = None
        self._actor = None
        self._triangles: np.ndarray | None = None
        self._reference: np.ndarray | None = None
        self._lut = None
        self._clim = (0.0, 1.0)
        self._cmap, self._reverse = colormaps.DEFAULT, False
        self._title, self._unit = "", ""
        self._edges = False
        self._view: str | tuple = "auto"
        self._isolines: tuple[np.ndarray, np.ndarray] | None = None
        self._markers: tuple[list[int], list[str]] | None = None
        self._ghost: np.ndarray | None = None
        self._extra: dict[str, object] = {}
        self._active = True
        self._released = False
        self._render_pending = False
        self._needs_render = False
        self.colorbar: ColorBar | None = None
        self.tools: _ToolPill | None = None

        self._stack = QStackedLayout(self)
        try:
            self._canvas = _create_canvas(self)
        except Exception:  # noqa: BLE001 - any VTK/GL failure falls back to the empty state
            log.exception("could not create a 3D viewport")
            self._canvas = None
        if self._canvas is None:
            self._stack.addWidget(EmptyState(
                "box", "3D view unavailable",
                "OpenGL could not be initialised on this display, so the surface can't be drawn. "
                "Plots and statistics are still available."))
            return
        self._stack.addWidget(self._canvas.widget)
        self._canvas.clicked.connect(self._pick)
        self.colorbar = ColorBar(self)
        self.colorbar.hide()
        self.tools = _ToolPill(self)
        self.reset_button = IconButton("focus", "Reset view", on_click=self.reset_camera)
        self.edges_button = IconButton("mesh", "Show mesh edges", checkable=True)
        self.edges_button.toggled.connect(self.set_edges)
        self.screenshot_button = IconButton("camera", "Save screenshot…", on_click=self._ask_screenshot)
        for button in (self.reset_button, self.edges_button, self.screenshot_button):
            self.tools.add(button)
        self._setup_scene(theme().palette)
        theme().changed.connect(self._on_theme)

    # ------------------------------------------------------------------ state
    @property
    def available(self) -> bool:
        """False when no 3D backend could be created (the viewer shows an empty state)."""
        return self._canvas is not None and not self._released

    @property
    def interactive(self) -> bool:
        return isinstance(self._canvas, _InteractiveCanvas)

    @property
    def plotter(self):
        """The underlying pyvista plotter (``None`` without a 3D backend)."""
        return self._canvas.plotter if self._canvas is not None else None

    @property
    def mesh(self) -> pv.PolyData | None:
        return self._mesh

    @property
    def edges(self) -> bool:
        return self._edges

    def scalars(self) -> np.ndarray | None:
        """The values currently drawn (a view into the VTK array)."""
        return None if self._mesh is None else vtk_to_numpy(self._mesh.GetPointData().GetArray("values"))

    # ------------------------------------------------------------------ scene
    def _setup_scene(self, p: Palette) -> None:
        plotter = self._canvas.plotter
        plotter.set_background(p.viewport_bottom, top=p.viewport_top)
        plotter.remove_all_lights()
        for position, intensity in (((1.0, 1.3, 1.5), 0.70), ((-1.6, -0.3, 0.9), 0.34),
                                    ((0.2, 1.4, -1.4), 0.42), ((0.0, 0.0, 1.0), 0.14)):
            plotter.add_light(pv.Light(position=position, focal_point=(0, 0, 0),
                                       light_type="camera light", intensity=intensity))

    def set_mesh(self, points: np.ndarray, triangles: np.ndarray, *, view: str | tuple = "auto") -> None:
        """Replace the surface. ``view``: ``"auto"`` (principal axes), ``"iso"`` or ``(direction, up)``."""
        if not self.available:
            return
        plotter = self._canvas.plotter
        plotter.clear_actors()
        self._extra.clear()
        pts = np.array(points, dtype=np.float32, copy=True)
        tri = np.ascontiguousarray(triangles, dtype=np.int64)
        mesh = pv.PolyData.from_regular_faces(pts, tri)
        mesh.point_data["values"] = np.zeros(mesh.n_points, np.float32)
        mesh.point_data["Normals"] = vertex_normals(pts, tri)
        mesh.point_data.active_normals_name = "Normals"
        p = theme().palette
        self._lut = colormaps.vtk_lookup_table(self._cmap, self._clim, self._reverse, nan_color=p.text3)
        actor = plotter.add_mesh(mesh, scalars="values", cmap=self._lut, clim=self._clim,
                                 show_scalar_bar=False, reset_camera=False, name="surface")
        prop = actor.prop
        prop.interpolation = "phong"
        prop.ambient, prop.diffuse = 0.24, 0.80
        prop.specular, prop.specular_power = 0.28, 32.0
        actor.mapper.interpolate_before_map = True
        actor.mapper.scalar_visibility = False
        prop.color = self._neutral(p)
        self._mesh, self._actor, self._triangles = mesh, actor, tri
        self._reference = pts.copy()
        self._isolines = self._markers = self._ghost = None
        self._apply_edges(p)
        self._view = view
        self.reset_camera()

    def set_scalars(self, values: np.ndarray | None, clim: tuple[float, float] | None = None,
                    cmap: str | None = None, title: str | None = None, *, unit: str | None = None,
                    reverse: bool | None = None) -> None:
        """Colour the surface by ``values`` (N,) in place; ``None`` shows the bare surface."""
        if self._actor is None:
            return
        mapper = self._actor.mapper
        if values is None:
            mapper.scalar_visibility = False
            self._actor.prop.color = self._neutral(theme().palette)
            self.colorbar.hide()
            self.request_render()
            return
        array = self._mesh.GetPointData().GetArray("values")
        np.copyto(vtk_to_numpy(array), values, casting="unsafe")
        array.Modified()
        if not mapper.scalar_visibility:
            mapper.scalar_visibility = True
        if cmap is not None or reverse is not None:
            name, rev = cmap or self._cmap, self._reverse if reverse is None else bool(reverse)
            if (name, rev) != (self._cmap, self._reverse):
                self._cmap, self._reverse = name, rev
                self._lut.values = colormaps.get(name).table_rgba8(256, rev)
        if clim is not None:
            clim = (float(clim[0]), float(clim[1]))
            if clim[1] <= clim[0]:
                clim = (clim[0], clim[0] + 1e-6)
            if clim != self._clim:
                self._clim = clim
                mapper.scalar_range = clim
        if title is not None:
            self._title = title
        if unit is not None:
            self._unit = unit
        self.colorbar.set(self._title, self._unit, self._clim, self._cmap, self._reverse)
        self.colorbar.show()
        self.request_render()

    def set_points(self, points: np.ndarray) -> None:
        """Move the surface points in place (deformation); normals follow."""
        if self._mesh is None:
            return
        data = self._mesh.GetPoints().GetData()
        view = vtk_to_numpy(data)
        np.copyto(view, points, casting="unsafe")
        data.Modified()
        normals = self._mesh.GetPointData().GetArray("Normals")
        vertex_normals(view, self._triangles, out=vtk_to_numpy(normals))
        normals.Modified()
        self._mesh.GetPoints().Modified()
        self._canvas.plotter.renderer.ResetCameraClippingRange()
        self.request_render()

    # ------------------------------------------------------------------ overlays
    def set_reference(self, points: np.ndarray | None) -> None:
        """Show a translucent ghost of the undeformed geometry (``None`` hides it)."""
        if self._mesh is None:
            return
        self._ghost = None if points is None else np.asarray(points, dtype=np.float32)
        self._rebuild_reference(theme().palette)
        self.request_render()

    def set_isolines(self, values: np.ndarray | None, levels: Sequence[float] = ()) -> None:
        """Contour lines of ``values`` at ``levels`` drawn on the surface (``None`` hides them)."""
        if self._mesh is None:
            return
        self._isolines = None if values is None or not len(levels) else \
            (np.asarray(values, dtype=np.float32), np.asarray(levels, dtype=np.float64))
        self._rebuild_isolines(theme().palette)
        self.request_render()

    def set_markers(self, indices: Sequence[int], colors: Sequence[str]) -> None:
        """Small spheres on surface points (picked probes), one colour each."""
        if self._mesh is None:
            return
        self._markers = (list(indices), list(colors)) if len(indices) else None
        self._rebuild_markers()
        self.request_render()

    def set_edges(self, visible: bool) -> None:
        visible = bool(visible)
        if self.edges_button is not None and self.edges_button.isChecked() != visible:
            self.edges_button.setChecked(visible)
            return  # the button's toggled signal comes back here
        if visible == self._edges:
            return
        self._edges = visible
        if self._actor is not None:
            self._apply_edges(theme().palette)
            self.request_render()
        self.edges_changed.emit(visible)

    def _apply_edges(self, p: Palette) -> None:
        prop = self._actor.prop
        prop.show_edges = self._edges
        prop.edge_color = "#0A0E16" if p.dark else "#3B4456"
        prop.line_width = 0.6

    def _remove(self, name: str) -> None:
        actor = self._extra.pop(name, None)
        if actor is not None:
            self._canvas.plotter.remove_actor(actor, render=False)

    def _rebuild_reference(self, p: Palette) -> None:
        self._remove("reference")
        self._remove("reference-edges")
        if self._ghost is None:
            return
        plotter = self._canvas.plotter
        ghost = pv.PolyData.from_regular_faces(self._ghost, self._triangles)
        actor = plotter.add_mesh(ghost, color=p.text3, opacity=0.16, ambient=0.35, diffuse=0.6, specular=0.0,
                                 show_scalar_bar=False, reset_camera=False, name="reference")
        self._extra["reference"] = actor
        edges = ghost.extract_feature_edges(feature_angle=35.0, boundary_edges=True, non_manifold_edges=False,
                                            manifold_edges=False)
        if edges.n_cells:
            self._extra["reference-edges"] = plotter.add_mesh(
                edges, color=p.text2, opacity=0.55, line_width=1.2, lighting=False,
                show_scalar_bar=False, reset_camera=False, name="reference-edges")

    def _rebuild_isolines(self, p: Palette) -> None:
        self._remove("isolines")
        if self._isolines is None:
            return
        values, levels = self._isolines
        points = vtk_to_numpy(self._mesh.GetPoints().GetData())
        normals = vtk_to_numpy(self._mesh.GetPointData().GetArray("Normals"))
        lifted = points + normals * np.float32(self._mesh.length * 0.003)
        source = pv.PolyData.from_regular_faces(lifted.astype(np.float32), self._triangles)
        finite = np.isfinite(values)
        fill = float(values[finite].max()) + 1e6 if finite.any() else 0.0
        source.point_data["iso"] = np.where(finite, values, fill).astype(np.float32)
        lines = source.contour(isosurfaces=levels, scalars="iso")
        if lines.n_points == 0:
            return
        color, opacity = ("#FFFFFF", 0.62) if p.dark else ("#0F1728", 0.55)
        self._extra["isolines"] = self._canvas.plotter.add_mesh(
            lines, color=color, opacity=opacity, line_width=1.8 * self.devicePixelRatioF(), lighting=False,
            show_scalar_bar=False, reset_camera=False, name="isolines")

    def _rebuild_markers(self) -> None:
        self._remove("markers")
        if self._markers is None:
            return
        points = vtk_to_numpy(self._mesh.GetPoints().GetData())
        radius = self._mesh.length * 0.011
        spheres = []
        for index, color in zip(*self._markers, strict=False):
            sphere = pv.Sphere(radius=radius, center=points[int(index)], theta_resolution=24, phi_resolution=16)
            rgb = np.array(QColor(color).getRgb()[:3], np.uint8)
            sphere.point_data["rgb"] = np.tile(rgb, (sphere.n_points, 1))
            spheres.append(sphere)
        merged = pv.merge(spheres) if len(spheres) > 1 else spheres[0]
        self._extra["markers"] = self._canvas.plotter.add_mesh(
            merged, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.28, specular=0.45,
            specular_power=40, show_scalar_bar=False, reset_camera=False, name="markers")

    # ------------------------------------------------------------------ camera
    def reset_camera(self) -> None:
        """Back to the default framing of the surface."""
        if self._mesh is None:
            return
        direction, up = self._preset()
        plotter = self._canvas.plotter
        center = np.array(self._mesh.center)
        plotter.camera_position = [tuple(center + np.asarray(direction) * self._mesh.length), tuple(center), up]
        plotter.reset_camera()
        plotter.camera.zoom(1.12)
        self.request_render()

    def view_isometric(self) -> None:
        if self._mesh is None:
            return
        self._view = "iso"
        self.reset_camera()

    def _preset(self) -> tuple[tuple, tuple]:
        if isinstance(self._view, tuple):
            return self._view
        if self._view == "iso":
            return ISO_VIEW
        return self._principal_view()

    def _principal_view(self) -> tuple[tuple, tuple]:
        """A three-quarter view with the long axis up and the widest side facing the camera."""
        pts = self._reference.astype(np.float64)
        centered = pts - pts.mean(axis=0)
        _, vectors = np.linalg.eigh(centered.T @ centered)
        thin, mid, long = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        s = centered @ long
        radial = np.linalg.norm(centered - np.outer(s, long), axis=1)
        low, high = np.quantile(s, [0.12, 0.88])
        if radial[s <= low].mean() > radial[s >= high].mean():  # the apex is the narrow end: put the base up
            long = -long
        if thin @ np.array([0.2, -1.0, 0.3]) < 0:
            thin = -thin
        direction = math.cos(math.radians(28)) * thin + math.sin(math.radians(28)) * mid + 0.22 * long
        direction /= np.linalg.norm(direction)
        return tuple(direction), tuple(long)

    # ------------------------------------------------------------------ picking
    def _pick(self, pos: QPointF) -> None:
        if self._actor is None:
            return
        plotter = self._canvas.plotter
        widget = self._canvas.widget
        width, height = plotter.window_size
        sx, sy = width / max(1, widget.width()), height / max(1, widget.height())
        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.PickFromListOn()
        picker.AddPickList(self._actor)
        if not picker.Pick(pos.x() * sx, (widget.height() - pos.y()) * sy, 0, plotter.renderer):
            return
        cell = picker.GetCellId()
        if cell < 0:
            return
        corners = self._triangles[cell]
        points = vtk_to_numpy(self._mesh.GetPoints().GetData())
        hit = np.array(picker.GetPickPosition())
        nearest = corners[int(np.argmin(np.linalg.norm(points[corners] - hit, axis=1)))]
        self.point_picked.emit(int(nearest))

    def pick_at(self, x: float, y: float) -> None:
        """Pick the surface point under widget coordinates ``(x, y)`` (as a click would)."""
        if self.available:
            self._pick(QPointF(x, y))

    # ------------------------------------------------------------------ rendering
    def request_render(self) -> None:
        """Schedule one render for this event-loop pass (skipped while inactive)."""
        if not self.available:
            return
        if not self._active:
            self._needs_render = True
            return
        if not self._render_pending:
            self._render_pending = True
            QTimer.singleShot(0, self._render_now)

    def _render_now(self) -> None:
        self._render_pending = False
        if self.available and self._active:
            self._canvas.render()

    def set_active(self, active: bool) -> None:
        """Pause rendering while hidden; catch up when shown again."""
        self._active = bool(active)
        if self._active and self._needs_render:
            self._needs_render = False
            self.request_render()

    def release(self) -> None:
        """Free the VTK render window and GL resources (the widget stays, empty)."""
        if self._released or self._canvas is None:
            self._released = True
            return
        self._released = True
        self._render_pending = False
        canvas, self._canvas = self._canvas, None
        self._mesh = self._actor = self._lut = None
        try:
            canvas.close()
        except Exception:  # noqa: BLE001 - teardown must not raise into the caller
            log.exception("closing the 3D viewport failed")
        canvas.widget.hide()
        for overlay in (self.colorbar, self.tools):
            overlay.hide()

    # ------------------------------------------------------------------ export
    def save_screenshot(self, path: Path | str) -> Path:
        """Write the viewport (with its colour legend) to a PNG at device resolution."""
        path = Path(path)
        if not self.available:
            raise RuntimeError("the 3D view is not available")
        rgb = self._canvas.snapshot()
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).convertToFormat(
            QImage.Format.Format_ARGB32_Premultiplied)
        if self.colorbar.isVisible():
            scale = w / max(1, self._canvas.widget.width())
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            painter.scale(scale, scale)
            self.colorbar.paint_into(painter, QRectF(self.colorbar.geometry()))
            painter.end()
        if not image.save(str(path), "PNG"):
            raise OSError(f"could not write {path}")
        return path

    def _ask_screenshot(self) -> None:
        folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation) or str(Path.home())
        default = Path(folder) / f"{self.export_name}-{datetime.now():%Y%m%d-%H%M%S}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", str(default), "PNG image (*.png)")
        if not path:
            return
        try:
            saved = self.save_screenshot(path)
        except (OSError, RuntimeError) as exc:
            ToastHost.show(self.window(), "Couldn't save the screenshot", str(exc), "danger")
            return
        ToastHost.show(self.window(), "Screenshot saved", saved.name, "success")

    # ------------------------------------------------------------------ layout & theme
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.tools is not None:
            self.tools.adjustSize()
            self.tools.move(self.width() - self.tools.width() - 12, 12)
            self.tools.raise_()
        if self.colorbar is not None:
            self.colorbar.move(14, self.height() - self.colorbar.height() - 14)
            self.colorbar.raise_()

    def _on_theme(self, p: Palette) -> None:
        if not self.available:
            return
        self._canvas.plotter.set_background(p.viewport_bottom, top=p.viewport_top)
        if self._actor is not None:
            self._lut.nan_color = p.text3
            if not self._actor.mapper.scalar_visibility:
                self._actor.prop.color = self._neutral(p)
            self._apply_edges(p)
            self._rebuild_reference(p)
            self._rebuild_isolines(p)
            self._rebuild_markers()
        self.request_render()

    @staticmethod
    def _neutral(p: Palette) -> str:
        return "#AEB8CC" if p.dark else "#C4CCDA"
