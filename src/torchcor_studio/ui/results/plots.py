"""pyqtgraph styled from the design tokens and kept in sync with the theme.

* :class:`ThemedPlot` / :func:`make_plot` — a plot widget with token colours, quiet
  axes, a ``plot_bg`` data area and no context menu; it restyles on ``theme().changed``.
* :func:`style_plot` — apply the styling to any ``PlotItem``.
* :class:`HoverReadout` — hairline + snapping dot + value label following the mouse.
* :class:`TimeCursor` — the vertical playback cursor (optionally draggable to seek).
* :class:`LegendChips` — a wrapping row of colour chips that toggle or remove series.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import Palette, theme

__all__ = ["HoverReadout", "LegendChips", "LegendItem", "ThemedPlot", "TimeCursor", "make_plot", "style_plot"]

TICK_PX = 11


def _font(px: int = TICK_PX, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setPixelSize(px)
    font.setWeight(weight)
    return font


def grid_ink(p: Palette) -> tuple[QColor, float]:
    """``p.grid`` as an opaque colour and the smallest alpha with which it looks the same
    over ``p.plot_bg``. The axes draw their grid lines above the data: opaque lines would cut
    every curve they cross (a dashed look); these only tint it."""
    grid, bg = QColor(p.grid), QColor(p.plot_bg)
    pairs = [(grid.red(), bg.red()), (grid.green(), bg.green()), (grid.blue(), bg.blue())]
    alpha = max((g - b) / (255 - b) if g > b else (b - g) / b if g < b else 0.0 for g, b in pairs)
    if alpha <= 0.0:
        return grid, 1.0
    return QColor(*(min(255, max(0, round(b + (g - b) / alpha))) for g, b in pairs)), alpha


def style_plot(item: pg.PlotItem, p: Palette, *, grid_x: bool = False, grid_y: bool = True,
               x_label: tuple[str, str] | None = None, y_label: tuple[str, str] | None = None) -> None:
    """Token styling for a ``PlotItem``: quiet axes, hairline grid, 11 px tick labels.

    ``x_label`` / ``y_label`` are ``(text, unit)`` pairs.
    """
    item.getViewBox().setBackgroundColor(QColor(p.plot_bg))
    ink, ink_alpha = grid_ink(p)
    for name, grid, label in (("left", grid_y, y_label), ("bottom", grid_x, x_label)):
        axis = item.getAxis(name)
        axis.setPen(pg.mkPen(QColor(p.line2), width=1))
        axis.setTextPen(pg.mkPen(QColor(p.text3)))
        axis.setTickPen(pg.mkPen(ink if grid else QColor(p.line2), width=1))
        axis.setGrid(255 if grid else False)
        axis.setStyle(tickFont=_font(), tickTextOffset=6, tickLength=0 if grid else 4,
                      tickAlpha=ink_alpha if grid else 1.0, maxTickLevel=1, autoExpandTextSpace=True)
        axis.enableAutoSIPrefix(False)
        if label:
            axis.setLabel(text=label[0], units=label[1] or None, color=p.text3, **{"font-size": f"{TICK_PX}px"})
    for name in ("top", "right"):
        item.hideAxis(name)


class ThemedPlot(pg.PlotWidget):
    """A ``PlotWidget`` that follows the theme. ``interactive`` enables x zoom/pan."""

    mouse_left = Signal()

    def __init__(self, parent: QWidget | None = None, *, x_label: str = "", x_unit: str = "",
                 y_label: str = "", y_unit: str = "", grid_x: bool = False, grid_y: bool = True,
                 interactive: bool = False, min_height: int = 160) -> None:
        super().__init__(parent, background=None)
        self.setStyleSheet("background: transparent; border: none;")
        self.setAntialiasing(True)
        self.setMinimumHeight(min_height)
        self._grid = (grid_x, grid_y)
        self._labels = ((x_label, x_unit) if x_label or x_unit else None,
                        (y_label, y_unit) if y_label or y_unit else None)
        item = self.getPlotItem()
        item.setMenuEnabled(False)
        item.hideButtons()
        item.layout.setContentsMargins(0, 4, 8, 0)
        vb = item.getViewBox()
        vb.setMouseEnabled(x=interactive, y=False)
        vb.setDefaultPadding(0.04)
        if interactive:
            self.scene().sigMouseClicked.connect(self._double_click)
        theme().changed.connect(self.restyle)
        self.restyle(theme().palette)

    def restyle(self, p: Palette) -> None:
        style_plot(self.getPlotItem(), p, grid_x=self._grid[0], grid_y=self._grid[1],
                   x_label=self._labels[0], y_label=self._labels[1])

    def leaveEvent(self, event) -> None:  # noqa: N802
        super().leaveEvent(event)
        self.mouse_left.emit()

    def _double_click(self, event) -> None:
        if event.double():
            self.getPlotItem().enableAutoRange()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(320, self.minimumHeight())


def make_plot(parent: QWidget | None = None, **kwargs) -> ThemedPlot:
    """Shorthand for :class:`ThemedPlot` (same keyword arguments)."""
    return ThemedPlot(parent, **kwargs)


def series_pen(color: str, width: float = 1.6, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    pen = pg.mkPen(QColor(color), width=width, style=style)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCosmetic(True)
    return pen


# =============================================================================== cursor
class TimeCursor(QObject):
    """A vertical line marking the playback time; ``on_drag`` makes it a scrubber.

    Parented to its plot, so its theme connection ends with the plot.
    """

    def __init__(self, plot: ThemedPlot, on_drag: Callable[[float], None] | None = None) -> None:
        super().__init__(plot)
        self._on_drag = on_drag
        self.line = pg.InfiniteLine(pos=0.0, angle=90, movable=on_drag is not None)
        self.line.setZValue(20)
        if on_drag is not None:
            self.line.sigDragged.connect(lambda line: on_drag(float(line.value())))
            self.line.setCursor(Qt.CursorShape.SizeHorCursor)
        plot.addItem(self.line, ignoreBounds=True)
        theme().changed.connect(self._restyle)
        self._restyle(theme().palette)

    def set_value(self, x: float) -> None:
        if not self.line.moving:
            self.line.setValue(float(x))

    def _restyle(self, p: Palette) -> None:
        color = QColor(p.accent_text if p.dark else p.accent)
        self.line.setPen(pg.mkPen(color, width=1.4))
        hover = QColor(color)
        hover.setAlphaF(0.9)
        self.line.setHoverPen(pg.mkPen(hover, width=3))


# =============================================================================== hover
class HoverReadout(QObject):
    """Hairline + dot snapped to the nearest curve under the mouse, with a value label.

    ``fmt(name, x, y)`` returns the label text; by default ``"name · x · y"``. Parented
    to its plot.
    """

    def __init__(self, plot: ThemedPlot, fmt: Callable[[str, float, float], str] | None = None) -> None:
        super().__init__(plot)
        self.plot = plot
        self._fmt = fmt or (lambda name, x, y: f"{name}  ·  {x:.0f}  ·  {y:.2f}")
        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.dot = pg.ScatterPlotItem(size=9)
        self.text = pg.TextItem(anchor=(0, 1))
        for item in (self.vline, self.dot, self.text):
            item.setZValue(30)
            item.hide()
            plot.addItem(item, ignoreBounds=True)
        plot.scene().sigMouseMoved.connect(self._moved)
        plot.mouse_left.connect(self.hide)
        theme().changed.connect(self._restyle)
        self._restyle(theme().palette)

    def hide(self) -> None:
        for item in (self.vline, self.dot, self.text):
            item.hide()

    def _restyle(self, p: Palette) -> None:
        self._palette = p
        line = QColor(p.text3)
        line.setAlphaF(0.6)
        self.vline.setPen(pg.mkPen(line, width=1, style=Qt.PenStyle.DashLine))
        self.text.setColor(QColor(p.text))
        fill = QColor(p.bg3)
        fill.setAlphaF(0.94)
        self.text.fill = pg.mkBrush(fill)
        self.text.border = pg.mkPen(QColor(p.line2))
        self.text.setFont(_font(11, QFont.Weight.Medium))

    def _curves(self) -> list[pg.PlotDataItem]:
        return [c for c in self.plot.getPlotItem().listDataItems()
                if isinstance(c, pg.PlotDataItem) and c.isVisible() and c.name()]

    def _moved(self, pos: QPointF) -> None:
        vb = self.plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            self.hide()
            return
        point = vb.mapSceneToView(pos)
        best, best_px = None, 1e9
        for curve in self._curves():
            x, y = curve.getData()
            if x is None or len(x) < 2:
                continue
            i = int(np.clip(np.searchsorted(x, point.x()), 1, len(x) - 1))
            i = i if abs(x[i] - point.x()) < abs(x[i - 1] - point.x()) else i - 1
            if abs(x[i] - point.x()) > (x[-1] - x[0]) * 0.05:
                continue
            scene = vb.mapViewToScene(QPointF(float(x[i]), float(y[i])))
            dist = abs(scene.y() - pos.y())
            if dist < best_px:
                best, best_px = (curve, float(x[i]), float(y[i])), dist
        if best is None:
            self.hide()
            return
        curve, x, y = best
        color = curve.opts["pen"].color() if isinstance(curve.opts["pen"], QPen) else QColor(self._palette.text)
        self.vline.setValue(x)
        self.dot.setData([x], [y], brush=pg.mkBrush(color), pen=pg.mkPen(QColor(self._palette.bg2), width=2))
        self.text.setText(self._fmt(curve.name(), x, y))
        (x0, x1), (_y0, y1) = vb.viewRange()
        right = x > x0 + (x1 - x0) * 0.62
        self.text.setAnchor((1.0, 1.0) if right else (0.0, 1.0))
        self.text.setPos(x, y1)
        for item in (self.vline, self.dot, self.text):
            item.show()


# =============================================================================== legend
@dataclass
class LegendItem:
    key: str
    label: str
    color: str
    removable: bool = False
    visible: bool = True


class LegendChips(QWidget):
    """Wrapping colour chips. Click a chip to toggle its series; × removes it."""

    toggled = Signal(str, bool)
    removed = Signal(str)

    H = 24
    GAP = 6

    def __init__(self, parent: QWidget | None = None, toggleable: bool = True) -> None:
        super().__init__(parent)
        self._items: list[LegendItem] = []
        self._hover = -1
        self._toggleable = toggleable
        self.setMouseTracking(True)
        sp = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        theme().changed.connect(self._theme_changed)

    def _theme_changed(self, _palette=None) -> None:
        self.update()

    def set_items(self, items: Sequence[LegendItem]) -> None:
        self._items = list(items)
        self.updateGeometry()
        self.update()

    def items(self) -> list[LegendItem]:
        return list(self._items)

    def _font(self) -> QFont:
        return _font(12, QFont.Weight.Medium)

    def _chip_width(self, item: LegendItem, fm: QFontMetrics) -> int:
        return 10 + 8 + 6 + fm.horizontalAdvance(item.label) + (18 if item.removable else 10)

    def _layout(self, width: int) -> list[QRectF]:
        fm = QFontMetrics(self._font())
        rects, x, y = [], 0, 0
        for item in self._items:
            w = self._chip_width(item, fm)
            if x > 0 and x + w > width:
                x, y = 0, y + self.H + self.GAP
            rects.append(QRectF(x, y, w, self.H))
            x += w + self.GAP
        return rects

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        rects = self._layout(max(1, width))
        return int(rects[-1].bottom()) if rects else 0

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(240, self.heightForWidth(max(240, self.width())))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(60, self.heightForWidth(max(60, self.width())))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.updateGeometry()

    def _hit(self, pos: QPointF) -> tuple[int, bool]:
        for i, r in enumerate(self._layout(self.width())):
            if r.contains(pos):
                on_close = self._items[i].removable and pos.x() > r.right() - 20
                return i, on_close
        return -1, False

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        index, _ = self._hit(event.position())
        if index != self._hover:
            self._hover = index
            self.setCursor(Qt.CursorShape.PointingHandCursor if index >= 0 else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._hover = -1
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        index, on_close = self._hit(event.position())
        if index < 0:
            return
        item = self._items[index]
        if on_close:
            self.removed.emit(item.key)
        elif self._toggleable:
            item.visible = not item.visible
            self.toggled.emit(item.key, item.visible)
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font())
        for i, (item, r) in enumerate(zip(self._items, self._layout(self.width()), strict=False)):
            hovered = i == self._hover
            painter.setPen(QPen(QColor(p.line2 if hovered else p.line), 1))
            painter.setBrush(QColor(p.bg4 if hovered else p.bg3))
            painter.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), r.height() / 2, r.height() / 2)
            dot = QColor(item.color)
            if not item.visible:
                dot.setAlphaF(0.25)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(dot)
            painter.drawEllipse(QPointF(r.x() + 14, r.center().y()), 4, 4)
            painter.setPen(QColor(p.text if item.visible else p.text3))
            text_rect = QRectF(r.x() + 24, r.y(), r.width() - 24, r.height())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, item.label)
            if item.removable:
                glyph = icons.pixmap("x", p.text if hovered else p.text3, 12)
                painter.drawPixmap(QPointF(r.right() - 17, r.center().y() - 6), glyph)
