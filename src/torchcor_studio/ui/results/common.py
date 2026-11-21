"""Layout pieces shared by the result views: stage, side panel, stat grid, toolbar bits."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.core.results import SummaryItem
from torchcor_studio.ui.results import colormaps
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import StatTile, ToggleSwitch, label

#: Outer margins of a result view, aligned with the run page header.
VIEW_MARGINS = (28, 16, 24, 20)
RADIUS = 12


# =============================================================================== numbers
def nice_step(span: float, target: int = 8) -> float:
    """A 1/2/2.5/5 × 10ⁿ step that splits ``span`` into about ``target`` intervals."""
    if not math.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(1, target)
    base = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * base:
            return m * base
    return 10 * base


def nice_bounds(lo: float, hi: float, target: int = 6) -> tuple[float, float]:
    """``(lo, hi)`` rounded outward to multiples of a nice step (a colour range)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return 0.0, 1.0
    if hi <= lo:
        pad = abs(lo) * 0.05 or 1.0
        return lo - pad, hi + pad
    step = nice_step(hi - lo, target)
    return math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step


def finite_range(values) -> tuple[float, float]:
    """Finite min/max of an array (``(0, 1)`` when nothing is finite)."""
    import numpy as np

    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)


def prettify(name: str) -> str:
    """``von_mises`` → ``Von Mises``."""
    words = name.replace("-", " ").replace("_", " ").split()
    return " ".join(w if w.isupper() else w.capitalize() for w in words) or name


def frame_text(label_text: str, unit: str, value: float) -> str:
    """Timeline text for a frame value: ``λ = 0.75``, ``t = 120 ms``, ``Step = 3``."""
    lowered = label_text.lower()
    symbol = "λ" if "λ" in label_text or "lambda" in lowered or "load" in lowered else \
        "t" if lowered.startswith("time") else label_text
    if symbol == "t" and not unit:
        unit = "ms"
    digits = 0 if unit == "ms" or abs(value) >= 100 else 2 if abs(value) < 10 else 1
    return f"{symbol} = {value:.{digits}f}{' ' + unit if unit else ''}"


# =============================================================================== stage
class _StageFrame(QWidget):
    """Paints the rounded corners and hairlines over a stage (above GL content)."""

    def __init__(self, stage: Stage) -> None:
        super().__init__(stage)
        self._stage = stage
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        theme().changed.connect(self._theme_changed)

    def _theme_changed(self, _palette=None) -> None:
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        inner = QPainterPath()
        inner.addRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS)
        outer = QPainterPath()
        outer.addRect(rect)
        painter.fillPath(outer.subtracted(inner), QColor(p.bg0))
        footer = self._stage.footer
        if footer is not None and footer.isVisible():
            y = footer.geometry().top() + 0.5
            painter.setPen(QPen(QColor(p.line), 1))
            painter.drawLine(QPointF(rect.left() + 1, y), QPointF(rect.right() - 1, y))
        painter.setPen(QPen(QColor(p.line), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS)


class Stage(QWidget):
    """Rounded container for a viewport (``body``) with an optional footer (timeline)."""

    def __init__(self, body: QWidget, footer: QWidget | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.body = body
        self.footer = footer
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(body, 1)
        if footer is not None:
            lay.addWidget(footer)
        self._frame = _StageFrame(self)
        theme().changed.connect(self._theme_changed)

    def _theme_changed(self, _palette=None) -> None:
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._frame.setGeometry(self.rect())
        self._frame.raise_()

    def set_footer_visible(self, visible: bool) -> None:
        if self.footer is not None:
            self.footer.setVisible(visible)
            self._frame.raise_()
            self._frame.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme().palette.bg2))


# =============================================================================== panels
class SidePanel(QScrollArea):
    """The scrolling right-hand column of a result view."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 12) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(320)
        content = QWidget()
        content.setObjectName("Transparent")
        self.column = QVBoxLayout(content)
        self.column.setContentsMargins(0, 0, 6, 0)
        self.column.setSpacing(spacing)
        self.column.addStretch(1)
        self.setWidget(content)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.column.insertWidget(self.column.count() - 1, widget, stretch)
        return widget


class ViewSplitter(QSplitter):
    """Viewport | side panel with a quiet gutter; the side starts at 30 % (340–460 px)."""

    SIDE = (0.3, 340, 460)

    def __init__(self, left: QWidget, right: QWidget) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self._sized = False
        self.setChildrenCollapsible(False)
        self.setHandleWidth(16)
        self.setStyleSheet("QSplitter::handle { background: transparent; }")
        self.addWidget(left)
        self.addWidget(right)
        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 0)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._sized and self.width() > 0:
            self._sized = True
            share, lo, hi = self.SIDE
            side = round(min(max(self.width() * share, lo), hi))
            self.setSizes([self.width() - side - self.handleWidth(), side])


def split(left: QWidget, right: QWidget) -> ViewSplitter:
    """Viewport | side panel (see :class:`ViewSplitter`)."""
    return ViewSplitter(left, right)


class StatGrid(QWidget):
    """Summary numbers as a grid of stat tiles (hints become tooltips to keep rows even)."""

    def __init__(self, items: Iterable[SummaryItem] = (), columns: int = 2, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._columns = columns
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(10)
        self.tiles: list[StatTile] = []
        self.set_items(items)

    def set_items(self, items: Iterable[SummaryItem]) -> None:
        while self._grid.count():
            widget = self._grid.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self.tiles = []
        for i, item in enumerate(items):
            tile = StatTile(item.label, item.text, item.unit)
            tile.setToolTip(item.hint)
            tile.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._grid.addWidget(tile, i // self._columns, i % self._columns)
            self.tiles.append(tile)
        for c in range(self._columns):
            self._grid.setColumnStretch(c, 1)
        if not self.tiles:
            self.hide()
        elif self.parentWidget() is not None:
            self.show()


# =============================================================================== toolbar
def switch(text: str, checked: bool, on_toggle: Callable[[bool], None], tooltip: str = "") -> tuple[QWidget, ToggleSwitch]:
    """A captioned toggle switch for toolbars."""
    holder = QWidget()
    lay = QHBoxLayout(holder)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    toggle = ToggleSwitch(checked)
    toggle.toggled.connect(on_toggle)
    caption = label(text, "muted")
    lay.addWidget(caption)
    lay.addWidget(toggle)
    holder.setToolTip(tooltip)
    toggle.setAccessibleName(text)
    return holder, toggle


def toolbar(*widgets: QWidget | None, spacing: int = 16) -> QHBoxLayout:
    """A toolbar row; ``None`` inserts a stretch."""
    lay = QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for widget in widgets:
        if widget is None:
            lay.addStretch(1)
        else:
            lay.addWidget(widget)
    return lay


def _swatch(name: str, w: int = 36, h: int = 10) -> QIcon:
    dpr = 2.0
    pixmap = QPixmap(round(w * dpr), round(h * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    rect = QRectF(0.5, 0.5, w - 1, h - 1)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(colormaps.gradient(name, rect.topLeft(), rect.topRight()))
    painter.drawRoundedRect(rect, h / 2, h / 2)
    painter.end()
    return QIcon(pixmap)


class ColormapCombo(QComboBox):
    """Colour-map picker with gradient swatches."""

    colormap_changed = Signal(str)

    def __init__(self, value: str = colormaps.DEFAULT, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setIconSize(QSize(36, 10))
        self.setToolTip("Colour map")
        self.setAccessibleName("Colour map")
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for name in colormaps.names():
            self.addItem(_swatch(name), colormaps.get(name).label, name)
        self.set_value(value)
        self.currentIndexChanged.connect(lambda *_: self.colormap_changed.emit(self.value()))

    def value(self) -> str:
        return self.currentData() or colormaps.DEFAULT

    def set_value(self, name: str) -> None:
        index = self.findData(name)
        if index >= 0 and index != self.currentIndex():
            self.blockSignals(True)
            self.setCurrentIndex(index)
            self.blockSignals(False)


def card_title(title: str, caption: str = "") -> QWidget:
    """Title + caption block for the top of a side-panel card."""
    holder = QWidget()
    lay = QVBoxLayout(holder)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    lay.addWidget(label(title, "h3"))
    if caption:
        lay.addWidget(label(caption, "caption", wrap=True))
    holder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    return holder


def notes_card(notes: Sequence[str]) -> QWidget:
    """The bundle's notes as a quiet card of captions."""
    from torchcor_studio.ui.widgets import Card, IconLabel

    card = Card(padding=14, spacing=8)
    for note in notes:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(IconLabel("info", 14, tone="faint"), 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(label(note, "caption", wrap=True), 1)
        card.addLayout(row)
    return card


def panel_header(eyebrow: str, title: str) -> QWidget:
    """Eyebrow + title at the top of a side panel."""
    head = QWidget()
    lay = QVBoxLayout(head)
    lay.setContentsMargins(2, 0, 0, 2)
    lay.setSpacing(2)
    lay.addWidget(label(eyebrow.upper(), "eyebrow"))
    lay.addWidget(label(title, "h3", wrap=True))
    return head
