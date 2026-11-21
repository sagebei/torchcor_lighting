"""12-lead ECG on ECG paper — plus the simulated heart when the bundle carries it.

:class:`ECGCanvas` paints each lead on a panel of ECG paper of its own (QPainter, cached),
so the leads never run into each other, the grid is crisp at any size and the page exports
at print resolution. All leads share one time window and one gain: scroll to zoom, drag to
pan, double-click to reset; the hover cursor runs through every lead at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, QSize, QStandardPaths, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from torchcor_studio.core.results import LEADS, MINUS, ECGData, ResultBundle, write_ecg_csv
from torchcor_studio.ui.results.common import VIEW_MARGINS, StatGrid, switch, toolbar
from torchcor_studio.ui.results.container import AsyncPage
from torchcor_studio.ui.theme import Palette, repolish, theme
from torchcor_studio.ui.widgets import Button, SegmentedControl, ToastHost, label

__all__ = ["GRID_LAYOUT", "ECGCanvas", "ECGView"]

#: Standard 3 × 4 arrangement; lead II also runs as the full-width rhythm strip.
GRID_LAYOUT = (("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6"))
RHYTHM_LEAD = "II"
MS_PER_MM = 40.0  # 25 mm/s: one small square is 40 ms
CLINICAL_GAIN = 10.0  # mm/mV
#: Gains (mm/mV) tried, largest first, when the ECG is fitted to the window.
GAINS = (40.0, 30.0, 20.0, 15.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25,
         0.2, 0.15, 0.1)  # the smallest: twelve stacked rows of a beat spread over the width
GUTTER = 34.0  # grid: room for the calibration pulse left of each row
NAME_GUTTER = 46.0  # stacked: the lead names left of the rows
PAD = 1.0  # around the panels (room for their borders)
GAP = 12.0  # between the panels of the grid
ROW_GAP = 6.0  # between the stacked rows
INSET = 10.0  # the trace's margin inside its panel, left and right
RADIUS = 8.0
#: A recording this long gets lead II as a rhythm strip under the grid (one simulated beat
#: would only be stretched across the page); the clinical page always has one.
RHYTHM_MIN_MS = 2000.0
MIN_WINDOW_MS = 40.0
#: Clinical layout: 2.5 s per column (10 s rhythm strip), 10 s per row when stacked.
CLINICAL_SPAN = {"grid": 2500.0, "stacked": 10000.0, "overlay": 10000.0}


@dataclass(frozen=True)
class Cell:
    """Where one lead is drawn and how time/voltage map to pixels there: ``rect`` is the
    trace's area, inside its panel (``frame``)."""

    lead: str
    rect: QRectF
    baseline: float  # y of 0 mV
    t0: float
    t1: float
    px_per_ms: float
    px_per_mv: float
    rhythm: bool = False
    panel: QRectF | None = None

    @property
    def frame(self) -> QRectF:
        """The lead's panel of paper."""
        return self.panel if self.panel is not None else self.rect

    @property
    def px_per_mm(self) -> float:
        """Paper millimetre in pixels (the grid is square: 1 mm = 40 ms)."""
        return self.px_per_ms * MS_PER_MM

    def x(self, t: float | np.ndarray):
        return self.rect.left() + (t - self.t0) * self.px_per_ms

    def y(self, v: float | np.ndarray):
        return self.baseline - v * self.px_per_mv

    def t_at(self, x: float) -> float:
        return self.t0 + (x - self.rect.left()) / self.px_per_ms


def _signed(value: float, digits: int) -> str:
    text = f"{value:.{digits}f}"
    return MINUS + text[1:] if text.startswith("-") else text


class ECGCanvas(QWidget):
    """The ECG paper. ``mode``: ``grid`` · ``stacked`` · ``overlay``; ``clinical`` locks 25 mm/s : 10 mm/mV."""

    hovered = Signal(str)
    window_changed = Signal(float, float)
    gain_changed = Signal(float)

    MODES = ("grid", "stacked", "overlay")

    def __init__(self, ecg: ECGData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ecg = ecg
        self.mode = "grid"
        self.clinical = False
        self._index = {name.lower(): i for i, name in enumerate(ecg.lead_names)}
        self._order = [name for name in LEADS if name.lower() in self._index] + \
            [name for name in ecg.lead_names if name not in LEADS]
        self.full_window = (float(ecg.t[0]), float(ecg.t[-1]))
        self._window = self.full_window
        self._zoomed = False
        lo, hi = float(np.nanmin(ecg.leads)), float(np.nanmax(ecg.leads))
        self._amp = (min(lo, -0.25), max(hi, 0.25))
        self.gain = CLINICAL_GAIN
        self._hover: QPointF | None = None
        self._press: tuple[QPointF, tuple[float, float], float] | None = None
        self._cache: QImage | None = None
        self._cache_key: tuple | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(520, 340)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        theme().changed.connect(self._invalidate)

    # ------------------------------------------------------------------ api
    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(960, 560)

    def set_mode(self, mode: str) -> None:
        if mode in self.MODES and mode != self.mode:
            self.mode = mode
            self.reset_window()

    def set_clinical(self, on: bool) -> None:
        if bool(on) != self.clinical:
            self.clinical = bool(on)
            self.reset_window()

    @property
    def time_window(self) -> tuple[float, float]:
        """The time span every lead shows (all leads share it)."""
        return self._window if self._zoomed else self.default_window()

    @property
    def zoomed(self) -> bool:
        return self._zoomed

    def default_window(self) -> tuple[float, float]:
        """The whole recording, or the clinical page (2.5 s per column) in clinical mode."""
        start, end = self.full_window
        return (start, start + CLINICAL_SPAN[self.mode]) if self.clinical else (start, end)

    def set_window(self, t0: float, t1: float) -> None:
        """Show ``[t0, t1]`` (clamped to the recording; at least 40 ms)."""
        start, end = self.full_window
        longest = max(end - start, self.default_window()[1] - start)
        span = min(max(t1 - t0, MIN_WINDOW_MS), longest)
        t0 = min(max(t0, start), max(start, end - span))
        window = (t0, t0 + span)
        if window != self.time_window:
            self._window, self._zoomed = window, True
            self._invalidate()
            self.window_changed.emit(*window)

    def reset_window(self) -> None:
        self._zoomed = False
        self._invalidate()
        self.window_changed.emit(*self.time_window)

    def lead(self, name: str) -> np.ndarray | None:
        i = self._index.get(name.lower())
        return None if i is None else self.ecg.leads[i]

    def _invalidate(self, *_args) -> None:
        self._cache_key = None
        self.update()

    # ------------------------------------------------------------------ geometry
    def rhythm_strip(self) -> bool:
        """Lead II along the bottom of the grid: on the clinical page, or when the recording
        is long enough to hold beats (:data:`RHYTHM_MIN_MS`)."""
        start, end = self.full_window
        return self.lead(RHYTHM_LEAD) is not None and (self.clinical or end - start >= RHYTHM_MIN_MS)

    def cells(self, rect: QRectF | None = None) -> list[Cell]:
        """Layout of every drawn lead for the current mode, scale and window: a panel each,
        apart from the others."""
        inner = (rect or QRectF(self.rect())).adjusted(PAD, PAD, -PAD, -PAD)
        t0, t1 = self.time_window
        span = max(t1 - t0, 1e-6)
        lo, hi = self._amp
        amp = hi - lo

        def scale(px_per_ms: float, height: float, fill: float) -> float:
            """Pixels per mV: the clinical gain, or the largest standard gain that fits."""
            px_per_mm = px_per_ms * MS_PER_MM
            gain = CLINICAL_GAIN if self.clinical else \
                next((g for g in GAINS if g * px_per_mm * amp <= height * fill), GAINS[-1])
            if gain != self.gain:
                self.gain = gain
                self.gain_changed.emit(gain)
            return gain * px_per_mm

        dpr = self.devicePixelRatioF() or 1.0

        def baseline(top: float, height: float, px_per_mv: float) -> float:
            """0 mV centred on the amplitude range, snapped to a device-pixel centre (crisp flats)."""
            y = top + height / 2 + (hi + lo) / 2 * px_per_mv
            return (math.floor(y * dpr) + 0.5) / dpr

        def cell(name: str, panel: QRectF, px_ms: float, px_mv: float, end: float = t1,
                 rhythm: bool = False) -> Cell:
            return Cell(name, panel.adjusted(INSET, 0, -INSET, 0), baseline(panel.top(), panel.height(), px_mv),
                        t0, end, px_ms, px_mv, rhythm, panel)

        cells: list[Cell] = []
        if self.mode == "grid":
            rhythm = self.rhythm_strip()
            rows = len(GRID_LAYOUT) + (1 if rhythm else 0)
            row_h = (inner.height() - (rows - 1) * GAP) / rows
            col_w = (inner.width() - GUTTER - 3 * GAP) / 4
            px_ms = (col_w - 2 * INSET) / span
            px_mv = scale(px_ms, row_h, 0.86)
            for r, names in enumerate(GRID_LAYOUT):
                top = inner.top() + r * (row_h + GAP)
                for c, name in enumerate(names):
                    if self.lead(name) is not None:
                        panel = QRectF(inner.left() + GUTTER + c * (col_w + GAP), top, col_w, row_h)
                        cells.append(cell(name, panel, px_ms, px_mv))
            if rhythm:
                strip = QRectF(inner.left() + GUTTER, inner.top() + len(GRID_LAYOUT) * (row_h + GAP),
                               inner.width() - GUTTER, row_h)
                width = strip.width() - 2 * INSET
                rhythm_ms = px_ms if self.clinical else width / span
                cells.append(cell(RHYTHM_LEAD, strip, rhythm_ms, px_mv, end=t0 + width / rhythm_ms, rhythm=True))
        elif self.mode == "stacked":
            n = max(1, len(self._order))
            row_h = (inner.height() - (n - 1) * ROW_GAP) / n
            width = inner.width() - NAME_GUTTER
            px_ms = (width - 2 * INSET) / span
            px_mv = scale(px_ms, row_h, 0.88)
            for i, name in enumerate(self._order):
                panel = QRectF(inner.left() + NAME_GUTTER, inner.top() + i * (row_h + ROW_GAP), width, row_h)
                cells.append(cell(name, panel, px_ms, px_mv))
        else:
            px_ms = (inner.width() - 2 * INSET) / span
            px_mv = scale(px_ms, inner.height(), 0.9)
            cells.append(cell("*", inner, px_ms, px_mv))
        return cells

    # ------------------------------------------------------------------ painting
    def render_image(self, scale: float = 2.0) -> QImage:
        """The page (no hover) as an image at ``scale`` × the widget size."""
        size = self.size() if self.width() > 0 else self.sizeHint()
        image = QImage(round(size.width() * scale), round(size.height() * scale),
                       QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(theme().palette.bg0))  # the page around the panels
        painter = QPainter(image)
        painter.scale(scale, scale)
        self._paint_paper(painter, QRectF(0, 0, size.width(), size.height()))
        painter.end()
        return image

    def paintEvent(self, _event) -> None:  # noqa: N802
        dpr = self.devicePixelRatioF()
        key = (self.width(), self.height(), dpr, self.mode, self.clinical, self.time_window, theme().palette.name)
        if key != self._cache_key or self._cache is None:
            self._cache = QImage(round(self.width() * dpr), round(self.height() * dpr),
                                 QImage.Format.Format_ARGB32_Premultiplied)
            self._cache.setDevicePixelRatio(dpr)
            self._cache.fill(Qt.GlobalColor.transparent)
            cache_painter = QPainter(self._cache)
            self._paint_paper(cache_painter, QRectF(self.rect()))
            cache_painter.end()
            self._cache_key = key
        painter = QPainter(self)
        painter.drawImage(QPointF(0, 0), self._cache)
        if self._hover is not None:
            self._paint_hover(painter, self._hover)

    def _paint_paper(self, painter: QPainter, rect: QRectF) -> None:
        """Every lead on a rounded panel of paper of its own: the grid, its trace, a border."""
        p = theme().palette
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cells = self.cells(rect)
        border = QColor(p.ecg_major)
        border.setAlphaF(0.9 if p.dark else 0.6)
        for cell in cells:
            panel = QPainterPath()
            panel.addRoundedRect(cell.frame.adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS)
            painter.fillPath(panel, QColor(p.ecg_paper))
            painter.save()
            painter.setClipPath(panel)
            self._paint_grid(painter, cell, p)
            self._paint_traces(painter, cell, p)
            painter.restore()
            painter.setPen(QPen(border, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(panel)
        if self.mode == "grid":
            self._paint_calibration(painter, cells, p, rect)
        self._paint_labels(painter, cells, p)

    @staticmethod
    def _paint_grid(painter: QPainter, cell: Cell, p: Palette) -> None:
        """The panel's paper: 1 mm and 5 mm squares from the start of its trace and from its
        0 mV line (a 5 mm line), so every panel reads the same."""
        frame, step_mm = cell.frame, cell.px_per_mm
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for minimum, color, step in ((4.0, p.ecg_minor, step_mm), (2.0, p.ecg_major, 5 * step_mm)):
            if step < minimum:
                continue
            pen = QPen(QColor(color), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            x0, y0 = cell.rect.left(), cell.baseline
            for k in range(math.ceil((frame.left() - x0) / step), math.floor((frame.right() - x0) / step) + 1):
                x = math.floor(x0 + k * step) + 0.5
                painter.drawLine(QPointF(x, frame.top()), QPointF(x, frame.bottom()))
            for k in range(math.ceil((frame.top() - y0) / step), math.floor((frame.bottom() - y0) / step) + 1):
                y = math.floor(y0 + k * step) + 0.5
                painter.drawLine(QPointF(frame.left(), y), QPointF(frame.right(), y))
        painter.restore()

    def _paint_calibration(self, painter: QPainter, cells: list[Cell], p: Palette, rect: QRectF) -> None:
        """A 1 mV pulse left of each row of the grid."""
        pen = QPen(QColor(p.text2), 1.4)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        left = min(c.frame.left() for c in cells)
        for cell in cells:
            if cell.frame.left() > left + 1:
                continue
            base, top = cell.baseline, cell.y(1.0)
            width = min(GUTTER - 16, max(6.0, 200.0 * cell.px_per_ms)) if self.clinical else 10.0
            x0 = rect.left() + PAD + 6
            path = QPainterPath(QPointF(x0, base))
            for point in ((x0 + 3, base), (x0 + 3, top), (x0 + 3 + width, top), (x0 + 3 + width, base),
                          (x0 + 6 + width, base)):
                path.lineTo(QPointF(*point))
            painter.drawPath(path)

    def _paint_traces(self, painter: QPainter, cell: Cell, p: Palette) -> None:
        t = self.ecg.t
        i0 = max(0, int(np.searchsorted(t, cell.t0)) - 1)
        i1 = min(len(t), int(np.searchsorted(t, cell.t1)) + 1)
        if i1 - i0 < 2:
            return
        names = self._order if cell.lead == "*" else [cell.lead]
        colors = _overlay_colors(p, len(names)) if cell.lead == "*" else [p.ecg_trace]
        x = np.asarray(cell.x(t[i0:i1]), dtype=np.float64)
        for name, color in zip(names, colors, strict=False):
            values = self.lead(name)
            path = pg.arrayToQPath(x, np.asarray(cell.y(values[i0:i1]), dtype=np.float64))
            pen = QPen(QColor(color), 1.5 if cell.lead != "*" else 1.3)
            # samples are a fraction of a pixel apart: round joins leave dark specks on flat
            # stretches, bevel joins look the same without them
            pen.setJoinStyle(Qt.PenJoinStyle.BevelJoin)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPath(path)

    def _paint_labels(self, painter: QPainter, cells: list[Cell], p: Palette) -> None:
        """Each lead's name: at the top left of its panel, or left of its row when stacked."""
        font = QFont()
        font.setPixelSize(12)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        fm = QFontMetrics(font)
        if self.mode == "overlay":
            self._paint_overlay_legend(painter, cells[0], p, fm)
            return
        if self.mode == "stacked":
            painter.setPen(QColor(p.text2))
            for cell in cells:
                painter.drawText(QRectF(cell.frame.left() - NAME_GUTTER, cell.baseline - 10, NAME_GUTTER - 12, 20),
                                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, cell.lead)
            return
        ink = QColor(p.ecg_trace)
        ink.setAlphaF(0.85)
        halo = QPen(QColor(p.ecg_paper), 4)
        halo.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        for cell in cells:
            origin = QPointF(cell.frame.left() + 9, cell.frame.top() + 7 + fm.ascent())
            outline = QPainterPath()
            outline.addText(origin, font, cell.lead)
            painter.strokePath(outline, halo)
            painter.fillPath(outline, ink)
            if cell.rhythm:
                small = QFont(font)
                small.setWeight(QFont.Weight.Normal)
                small.setPixelSize(11)
                painter.setFont(small)
                faint = QColor(ink)
                faint.setAlphaF(0.5)
                painter.setPen(faint)
                painter.drawText(QPointF(origin.x() + 4 + fm.horizontalAdvance(cell.lead), origin.y()), "rhythm")
                painter.setFont(font)

    def _paint_overlay_legend(self, painter: QPainter, cell: Cell, p: Palette, fm: QFontMetrics) -> None:
        x, y = cell.frame.left() + 9, cell.frame.top() + 7
        for name, color in zip(self._order, _overlay_colors(p, len(self._order)), strict=False):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(QRectF(x, y + fm.ascent() / 2 - 1, 10, 3), 1.5, 1.5)
            painter.setPen(QColor(p.ecg_trace))
            painter.drawText(QPointF(x + 14, y + fm.ascent()), name)
            x += 14 + fm.horizontalAdvance(name) + 14

    # ------------------------------------------------------------------ hover
    def hover_info(self, pos: QPointF) -> tuple[Cell, str, float, float] | None:
        """``(cell, lead, t, mV)`` under ``pos``, or ``None`` outside the traces."""
        for cell in self.cells():
            if cell.frame.contains(pos):
                t = min(max(cell.t_at(pos.x()), cell.t0, self.full_window[0]), cell.t1, self.full_window[1])
                if cell.lead != "*":
                    return cell, cell.lead, t, float(np.interp(t, self.ecg.t, self.lead(cell.lead)))
                best = min(self._order, key=lambda n: abs(cell.y(float(np.interp(t, self.ecg.t, self.lead(n))))
                                                          - pos.y()))
                return cell, best, t, float(np.interp(t, self.ecg.t, self.lead(best)))
        return None

    def _paint_hover(self, painter: QPainter, pos: QPointF) -> None:
        info = self.hover_info(pos)
        if info is None:
            return
        hovered, lead, t, value = info
        p = theme().palette
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        line = QColor(p.accent_text if p.dark else p.accent)
        line.setAlphaF(0.55)
        painter.setPen(QPen(line, 1))
        for cell in self.cells():
            if cell.t0 <= t <= cell.t1:
                x = cell.x(t)
                painter.drawLine(QPointF(x, cell.frame.top() + 4), QPointF(x, cell.frame.bottom() - 4))
        x, y = hovered.x(t), hovered.y(value)
        accent = QColor(p.accent)
        painter.setPen(QPen(QColor(p.ecg_paper), 2))
        painter.setBrush(accent)
        painter.drawEllipse(QPointF(x, y), 4.5, 4.5)
        text = f"{lead}  ·  {t:.0f} ms  ·  {_signed(value, 2)} mV"
        font = QFont()
        font.setPixelSize(12)
        font.setWeight(QFont.Weight.Medium)
        fm = QFontMetrics(font)
        box = QRectF(0, 0, fm.horizontalAdvance(text) + 18, fm.height() + 10)
        box.moveTopLeft(QPointF(x + 10, y - box.height() - 8))
        if box.right() > self.width() - 8:
            box.moveRight(x - 10)
        if box.top() < 6:
            box.moveTop(y + 10)
        painter.setPen(QPen(QColor(p.line2), 1))
        fill = QColor(p.bg3)
        fill.setAlphaF(0.96)
        painter.setBrush(fill)
        painter.drawRoundedRect(box, 7, 7)
        painter.setFont(font)
        painter.setPen(QColor(p.text))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if self._press is not None:
            start, window, px_per_ms = self._press
            shift = (pos.x() - start.x()) / px_per_ms
            self.set_window(window[0] - shift, window[1] - shift)
        self._hover = pos
        info = self.hover_info(pos)
        self.hovered.emit("" if info is None else
                          f"{info[1]}  ·  t = {info[2]:.0f} ms  ·  {_signed(info[3], 3)} mV")
        self.update()

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._hover = None
        self.hovered.emit("")
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        info = self.hover_info(event.position())
        if event.button() == Qt.MouseButton.LeftButton and info is not None:
            self._press = (event.position(), self.time_window, info[0].px_per_ms)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        self._press = None
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseDoubleClickEvent(self, _event) -> None:  # noqa: N802
        self.reset_window()

    def wheelEvent(self, event) -> None:  # noqa: N802
        info = self.hover_info(event.position())
        t0, t1 = self.time_window
        anchor = info[2] if info is not None else (t0 + t1) / 2
        factor = 0.85 ** (event.angleDelta().y() / 120.0)
        span = (t1 - t0) * factor
        start = anchor - (anchor - t0) * factor
        self.set_window(start, start + span)


def _overlay_colors(p: Palette, n: int) -> list[str]:
    """Twelve distinct trace colours: the series palette, then lighter/darker variants."""
    colors = []
    for i in range(n):
        base = QColor(p.series[i % len(p.series)])
        if i >= len(p.series):
            base = base.lighter(135) if p.dark else base.darker(130)
        colors.append(base.name())
    return colors


class ECGView(QWidget):
    """ECG bundle view; adds a Heart page (an EP view) when the bundle has the heart's fields."""

    def __init__(self, bundle: ResultBundle, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bundle = bundle
        self.ecg = bundle.ecg
        self.has_heart = bundle.has("surface") and bundle.has("ep")
        self.heart: AsyncPage | None = None
        self._active = True
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.page_control = SegmentedControl([("ecg", "12-lead ECG", "ecg"), ("heart", "Heart", "heart")],
                                             value="ecg")
        self.page_control.changed.connect(self.show_page)
        if not self.has_heart:
            self.page_control.hide()
        self.layout_control = SegmentedControl([("grid", "Grid"), ("stacked", "Stacked"), ("overlay", "Overlay")],
                                               value="grid")
        self.layout_control.changed.connect(self._set_mode)
        self.clinical_holder, self.clinical_switch = switch("Clinical scale", False, self._set_clinical,
                                                            "Paper proportions: 25 mm/s and 10 mm/mV")
        self.png_button = Button("PNG", "download", size="sm", on_click=self._ask_png)
        self.png_button.setToolTip("Export the ECG as an image")
        self.csv_button = Button("CSV", "download", size="sm", on_click=self._ask_csv)
        self.csv_button.setToolTip("Export the leads as CSV")
        self._ecg_tools = [self.layout_control, self.clinical_holder, self.png_button, self.csv_button]

        self.canvas = ECGCanvas(self.ecg)
        self.readout = label("", "mono")
        self.scale_note = label("", "caption")
        self.canvas.hovered.connect(self._hovered)
        self.canvas.window_changed.connect(lambda *_: self._update_scale_note())
        self.canvas.gain_changed.connect(lambda *_: self._update_scale_note())

        ecg_page = QWidget()
        page = QVBoxLayout(ecg_page)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(12)
        headline = _ecg_headline(bundle.summary)
        self.stats = StatGrid(headline, columns=max(1, len(headline)))
        page.addWidget(self.stats)
        page.addWidget(self.canvas, 1)
        foot = QHBoxLayout()
        foot.setContentsMargins(4, 0, 4, 0)
        foot.addWidget(self.readout, 1)
        foot.addWidget(self.scale_note)
        page.addLayout(foot)

        self.pages = QStackedWidget()
        self.pages.addWidget(ecg_page)

        root = QVBoxLayout(self)
        root.setContentsMargins(*VIEW_MARGINS)
        root.setSpacing(14)
        root.addLayout(toolbar(self.page_control, *self._ecg_tools[:2], None, *self._ecg_tools[2:], spacing=12))
        root.addWidget(self.pages, 1)
        self._hovered("")
        self._update_scale_note()

    # ------------------------------------------------------------------ pages
    def show_page(self, name: str) -> None:
        """``"ecg"`` or ``"heart"`` (the latter loads the heart's fields on first use)."""
        if name == "heart" and self.has_heart:
            if self.heart is None:
                bundle = self.bundle

                def work() -> ResultBundle:
                    bundle.preload("surface", "ep")
                    import pyvista  # noqa: F401 - warm the import off the UI thread

                    return bundle

                from torchcor_studio.ui.results.ep_view import EPView

                self.heart = AsyncPage(
                    work, lambda b: EPView(b, show_stats=False, title="Simulated heart", margins=(0, 0, 0, 0)),
                    "Loading the heart…")
                self.heart.set_active(False)
                self.pages.addWidget(self.heart)
            self.pages.setCurrentWidget(self.heart)
        else:
            self.pages.setCurrentIndex(0)
        on_ecg = self.pages.currentIndex() == 0
        self.page_control.set_value("ecg" if on_ecg else "heart", animate=False)
        for widget in self._ecg_tools:
            widget.setVisible(on_ecg)
        if self.heart is not None:
            self.heart.set_active(self._active and not on_ecg)

    def _set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self._update_scale_note()

    def _set_clinical(self, on: bool) -> None:
        self.canvas.set_clinical(on)
        self._update_scale_note()

    def _hovered(self, text: str) -> None:
        self.readout.setText(text or "Hover a lead to read values · scroll to zoom · drag to pan")
        self.readout.setProperty("role", "mono" if text else "caption")
        repolish(self.readout)

    def _update_scale_note(self) -> None:
        self.canvas.cells()  # settles the fitted gain for the current size
        t0, t1 = self.canvas.time_window
        zoom = "" if (t0, t1) == self.canvas.full_window else f"{t0:.0f}–{t1:.0f} ms  ·  "
        fitted = "" if self.canvas.clinical else "  ·  fitted"
        self.scale_note.setText(f"{zoom}25 mm/s  ·  {self.canvas.gain:g} mm/mV{fitted}")

    # ------------------------------------------------------------------ export
    def export_png(self, path: Path | str, scale: float = 2.0) -> Path:
        """Save the ECG paper as a PNG at ``scale`` × the on-screen size."""
        path = Path(path)
        if not self.canvas.render_image(scale).save(str(path), "PNG"):
            raise OSError(f"could not write {path}")
        return path

    def export_csv(self, path: Path | str) -> Path:
        return write_ecg_csv(path, self.ecg)

    def _default_path(self, suffix: str) -> Path:
        folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation) or str(Path.home())
        return Path(folder) / f"torchcor-ecg-{datetime.now():%Y%m%d-%H%M%S}.{suffix}"

    def _ask_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export ECG image", str(self._default_path("png")),
                                              "PNG image (*.png)")
        if path:
            self._export(self.export_png, path)

    def _ask_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export ECG data", str(self._default_path("csv")),
                                              "CSV (*.csv)")
        if path:
            self._export(self.export_csv, path)

    def _export(self, fn, path: str) -> None:
        try:
            saved = fn(path)
        except OSError as exc:
            ToastHost.show(self.window(), "Export failed", str(exc), "danger")
            return
        ToastHost.show(self.window(), "Exported", Path(saved).name, "success")

    # ------------------------------------------------------------------ lifecycle
    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if self.heart is not None:
            self.heart.set_active(self._active and self.pages.currentWidget() is self.heart)

    def release(self) -> None:
        if self.heart is not None:
            self.heart.release()


def _ecg_headline(items, limit: int = 4):  # noqa: ANN001, ANN202
    """One row of the numbers that matter on the ECG page; the heart's stay on its own tab."""

    def rank(item) -> int:  # noqa: ANN001
        text = item.label.lower()
        if any(k in text for k in ("lead", "qrs", "qt", "heart rate")):
            return 0
        if "duration" in text:
            return 1
        if "activation time" in text:
            return 2
        return 3

    ranked = sorted(items, key=rank)
    return [item for item in ranked if rank(item) < 3][:limit] or ranked[:limit]
