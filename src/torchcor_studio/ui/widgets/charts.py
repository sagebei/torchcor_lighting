"""Tiny painted charts for dashboards: sparklines and meter bars.

They are QPainter-based (no pyqtgraph) because dozens of them update every second in
the GPU grid; painting a 60-point polyline is far cheaper than a plot widget.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets.basic import tone_color


class Sparkline(QWidget):
    """A filled area sparkline over a fixed-length rolling window.

    Values are in ``[vmin, vmax]`` (default 0–100, i.e. percent). Supports a second,
    dimmer series (e.g. memory under utilisation).
    """

    def __init__(self, capacity: int = 60, vmin: float = 0.0, vmax: float = 100.0,
                 tone: str = "accent", height: int = 36, parent: QWidget | None = None):
        super().__init__(parent)
        self.capacity = capacity
        self.vmin, self.vmax = vmin, vmax
        self.tone = tone
        self.series: deque[float] = deque(maxlen=capacity)
        self.series2: deque[float] = deque(maxlen=capacity)
        self.tone2 = "ember"
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(height)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(160, self.height())

    def push(self, value: float, value2: float | None = None) -> None:
        self.series.append(float(value))
        if value2 is not None:
            self.series2.append(float(value2))
        self.update()

    def set_values(self, values: Iterable[float], values2: Iterable[float] | None = None) -> None:
        self.series = deque((float(v) for v in values), maxlen=self.capacity)
        if values2 is not None:
            self.series2 = deque((float(v) for v in values2), maxlen=self.capacity)
        self.update()

    def clear(self) -> None:
        self.series.clear()
        self.series2.clear()
        self.update()

    def _path(self, values, rect: QRectF) -> tuple[QPainterPath, QPainterPath]:
        n = self.capacity
        span = (self.vmax - self.vmin) or 1.0
        step = rect.width() / max(1, n - 1)
        offset = n - len(values)
        line = QPainterPath()
        for i, v in enumerate(values):
            x = rect.left() + (offset + i) * step
            y = rect.bottom() - (min(max(v, self.vmin), self.vmax) - self.vmin) / span * rect.height()
            if i == 0:
                line.moveTo(x, y)
            else:
                line.lineTo(x, y)
        area = QPainterPath(line)
        if values:
            area.lineTo(rect.left() + (offset + len(values) - 1) * step, rect.bottom())
            area.lineTo(rect.left() + offset * step, rect.bottom())
            area.closeSubpath()
        return line, area

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 2, -1, -1)
        # baseline grid
        grid = QPen(QColor(p.grid))
        grid.setWidthF(1.0)
        painter.setPen(grid)
        for frac in (0.5,):
            y = rect.top() + rect.height() * frac
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        for values, tone, weight in ((self.series2, self.tone2, 0.6), (self.series, self.tone, 1.0)):
            if not values:
                continue
            color = QColor(tone_color(p, tone))
            line, area = self._path(list(values), rect)
            grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            top = QColor(color)
            top.setAlphaF(0.28 * weight)
            bottom = QColor(color)
            bottom.setAlphaF(0.0)
            grad.setColorAt(0, top)
            grad.setColorAt(1, bottom)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawPath(area)
            pen = QPen(color)
            pen.setWidthF(1.6)
            if weight < 1:
                c = QColor(color)
                c.setAlphaF(0.75)
                pen.setColor(c)
                pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(line)


class MeterBar(QWidget):
    """Thin horizontal meter: track + rounded fill. Tone shifts to warning/danger when high."""

    def __init__(self, value: float = 0.0, tone: str = "accent", auto_tone: bool = True,
                 height: int = 6, parent: QWidget | None = None):
        super().__init__(parent)
        self._value = value
        self.tone = tone
        self.auto_tone = auto_tone
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, fraction: float) -> None:
        self._value = max(0.0, min(1.0, float(fraction)))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        rad = r.height() / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(p.bg4))
        painter.drawRoundedRect(r, rad, rad)
        tone = self.tone
        if self.auto_tone:
            tone = "danger" if self._value >= 0.92 else "warning" if self._value >= 0.75 else self.tone
        if self._value > 0:
            fill = QRectF(r.x(), r.y(), max(r.height(), r.width() * self._value), r.height())
            painter.setBrush(QColor(tone_color(p, tone)))
            painter.drawRoundedRect(fill, rad, rad)
