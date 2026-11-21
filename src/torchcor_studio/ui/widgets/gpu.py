"""Compute-device telemetry widgets.

* :class:`GpuCard` — one device: load, 60 s history, memory, temperature and power.
* :class:`GpuGrid` — responsive grid of cards (1–4 columns) with loading / empty states
  and optional single selection (for picking the device a run uses).
* :class:`DeviceList` — compact rows for narrow side panels.
* :class:`GpuFeed` — Qt adapter around :class:`~torchcor_studio.core.gpu.GpuMonitor`
  that delivers samples on the UI thread.

Wiring::

    feed = GpuFeed(machine, interval=settings.telemetry_interval_s, parent=page)
    feed.samples.connect(grid.update_samples)
    feed.error.connect(grid.set_error)
    feed.start()            # feed.stop() when the page closes or the machine changes
"""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.core.gpu import NOT_SUPPORTED_HINT, GpuMonitor
from torchcor_studio.core.machines.base import Machine
from torchcor_studio.core.models import DeviceKind, GPUSample
from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets.basic import (
    Badge,
    ElidedLabel,
    EmptyState,
    Spinner,
    label,
    tone_color,
)
from torchcor_studio.ui.widgets.charts import MeterBar, Sparkline

HISTORY_S = 60.0
_ICONS = {DeviceKind.CUDA: "gpu", DeviceKind.CPU: "cpu", DeviceKind.APPLE: "chip"}


# ============================================================================ formatting
def memory_text(sample: GPUSample) -> str:
    """``"12.3 / 80.0 GB"`` (MB below 1 GB in use)."""
    total = sample.memory_total_mb / 1024
    if not sample.memory_total_mb:
        return "—"
    return f"{sample.memory_used_mb / 1024:.1f} / {total:.1f} GB"


def device_state(sample: GPUSample) -> tuple[str, str]:
    """``(label, tone)`` of the state pill: Idle, Busy, "N processes" or Not supported."""
    if not sample.runnable:
        return "Not supported", "neutral"
    if sample.processes:
        return (f"{sample.processes} process" + ("es" if sample.processes > 1 else "")), "warning"
    busy_at = 50.0 if sample.kind == DeviceKind.CPU else 10.0
    return ("Busy", "warning") if sample.utilization >= busy_at else ("Idle", "success")


def device_title(sample: GPUSample) -> str:
    """Display name: ``"A100-SXM4-80GB"``, ``"AMD EPYC 7763"`` (details go to the caption)."""
    name = sample.name.split(" · ", 1)[0].strip() or sample.name
    if sample.kind == DeviceKind.CUDA and name.startswith("NVIDIA ") and len(name) > 7:
        name = name[7:]
    return name


def device_caption(sample: GPUSample) -> str:
    """Secondary line: torch device string and core count."""
    detail = sample.name.split(" · ", 1)[1].strip() if " · " in sample.name else ""
    if sample.kind == DeviceKind.CUDA:
        return f"cuda:{sample.index}"
    if sample.kind == DeviceKind.CPU:
        return " · ".join(filter(None, ["cpu", detail]))
    return " · ".join(filter(None, ["Metal", detail]))


class _History:
    """Rolling (load %, memory %) pairs for one device."""

    def __init__(self, capacity: int) -> None:
        self.load: deque[float] = deque(maxlen=capacity)
        self.memory: deque[float] = deque(maxlen=capacity)

    def push(self, sample: GPUSample) -> None:
        self.load.append(sample.utilization)
        self.memory.append(sample.memory_fraction * 100.0)


def _capacity(interval: float) -> int:
    return max(10, round(HISTORY_S / max(0.1, interval)))


# ============================================================================ painted bits
class IconTile(QWidget):
    """Rounded square holding a glyph: device icons, dialog headers.

    ``tone`` is ``"accent"`` (tinted tile), ``"muted"`` or ``"faint"`` (neutral tile).
    """

    def __init__(self, icon: str, tone: str, size: int = 32, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icon, self._tone = icon, tone
        self.setFixedSize(size, size)
        theme().changed.connect(self.update)

    def set(self, icon: str, tone: str) -> None:
        self._icon, self._tone = icon, tone
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        accent = self._tone == "accent"
        painter.setPen(QPen(p.qcolor("accent", 0.35) if accent else QColor(p.line), 1.0))
        painter.setBrush(p.qcolor("accent", 0.12) if accent else QColor(p.bg3))
        radius = self.width() / 4
        painter.drawRoundedRect(rect, radius, radius)
        color = {"accent": p.accent_text, "muted": p.text2}.get(self._tone, p.text3)
        glyph = round(self.width() * 0.56)
        painter.drawPixmap((self.width() - glyph) // 2, (self.height() - glyph) // 2,
                           icons.pixmap(self._icon, color, glyph))


class _Chip(QWidget):
    """Small read-only pill: icon + text (temperature, power)."""

    def __init__(self, icon: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icon, self._text = icon, ""
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        theme().changed.connect(self.update)

    def set_text(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self.updateGeometry()
            self.update()

    def _font(self):
        f = self.font()
        f.setPixelSize(theme().metrics.fs_sm)
        return f

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(QFontMetrics(self._font()).horizontalAdvance(self._text) + 34, 24)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QColor(p.line))
        painter.setBrush(QColor(p.bg3))
        painter.drawRoundedRect(rect, 6, 6)
        painter.drawPixmap(8, (self.height() - 13) // 2, icons.pixmap(self._icon, p.text3, 13))
        painter.setPen(QColor(p.text2))
        painter.setFont(self._font())
        painter.drawText(rect.adjusted(26, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter, self._text)


class _Figure(QWidget):
    """A large number and its unit drawn on one baseline (``87`` ``%``)."""

    def __init__(self, unit: str = "%", parent: QWidget | None = None, *, scale: float = 1.0) -> None:
        super().__init__(parent)
        self._value, self._unit = "0", unit
        m = theme().metrics
        self._size, self._unit_size = round((m.fs_2xl + 6) * scale), round(m.fs_lg * scale)
        self.setFixedHeight(round(self._size * 1.25))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        theme().changed.connect(self.update)

    def set_value(self, value: str) -> None:
        if value != self._value:
            self._value = value
            self.update()

    def _fonts(self):
        big, small = self.font(), self.font()
        big.setPixelSize(self._size)
        big.setWeight(big.Weight.DemiBold)
        big.setLetterSpacing(big.SpacingType.AbsoluteSpacing, -0.8)
        small.setPixelSize(self._unit_size)
        small.setWeight(small.Weight.DemiBold)
        return big, small

    def sizeHint(self) -> QSize:  # noqa: N802
        big, small = self._fonts()
        width = QFontMetrics(big).horizontalAdvance("100") + QFontMetrics(small).horizontalAdvance(self._unit)
        return QSize(width + 4, self.height())

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        big, small = self._fonts()
        fm = QFontMetrics(big)
        baseline = fm.ascent() + (self.height() - fm.height()) / 2
        painter.setFont(big)
        painter.setPen(QColor(p.text))
        painter.drawText(QPointF(0, baseline), self._value)
        painter.setFont(small)
        painter.setPen(QColor(p.text3))
        painter.drawText(QPointF(fm.horizontalAdvance(self._value) + 2, baseline), self._unit)


class _KeyLabel(QWidget):
    """A caption with a coloured key dot, doubling as the sparkline's legend."""

    def __init__(self, text: str, tone: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text, self._tone = text, tone
        self.setFixedHeight(16)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        theme().changed.connect(self.update)

    def set_text(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self.updateGeometry()
            self.update()

    def _font(self):
        f = self.font()
        f.setPixelSize(theme().metrics.fs_sm)
        return f

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(QFontMetrics(self._font()).horizontalAdvance(self._text) + 14, 16)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(tone_color(p, self._tone)))
        painter.drawRoundedRect(QRectF(0, self.height() / 2 - 3, 6, 6), 3, 3)
        painter.setFont(self._font())
        painter.setPen(QColor(p.text3))
        painter.drawText(QRectF(12, 0, self.width() - 12, self.height()), Qt.AlignmentFlag.AlignVCenter,
                         self._text)


# ============================================================================ card
class GpuCard(QFrame):
    """Telemetry card for one compute device.

    Selectable cards emit :attr:`clicked` with the device key; the owner decides what
    is selected (see :class:`GpuGrid`). Devices TorchCor cannot use look disabled.
    """

    clicked = Signal(str)
    MIN_WIDTH = 250
    #: The compact tile (``compact=True``) is the same design at about 80 % scale.
    COMPACT_MIN_WIDTH = 210

    def __init__(self, sample: GPUSample, *, selectable: bool = False, interval: float = 1.0,
                 compact: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sample = sample
        self._selectable = selectable
        self._compact = compact
        self._selected = False
        self._hover = False
        self.history = _History(_capacity(interval))
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(self.COMPACT_MIN_WIDTH if compact else self.MIN_WIDTH)
        self._build()
        theme().changed.connect(self.update)
        self.set_sample(sample)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        c = self._compact
        pad, gap = (12, 10) if c else (16, 16)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(pad, pad, pad, pad)
        outer.setSpacing(0)
        self._content = QWidget(self)
        self._content.setObjectName("Transparent")
        outer.addWidget(self._content)
        body = QVBoxLayout(self._content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        header = QHBoxLayout()
        header.setSpacing(10 if c else 12)
        self._tile = IconTile("gpu", "accent", size=26 if c else 32)
        header.addWidget(self._tile, 0, Qt.AlignmentFlag.AlignTop)
        titles = QVBoxLayout()
        titles.setSpacing(3)
        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        self._name = ElidedLabel("", "h3")
        name_row.addWidget(self._name, 1)
        self._state = Badge("Idle", "success", dot=True)
        self._state_tone = "success"
        name_row.addWidget(self._state, 0, Qt.AlignmentFlag.AlignTop)
        titles.addLayout(name_row)
        sub_row = QWidget()
        sub_row.setFixedHeight(20)  # same height with or without the index badge
        sub = QHBoxLayout(sub_row)
        sub.setContentsMargins(0, 0, 0, 0)
        sub.setSpacing(6)
        self._index = Badge("", mono=True)
        sub.addWidget(self._index)
        self._caption = label("", "mono")
        sub.addWidget(self._caption)
        sub.addStretch(1)
        titles.addWidget(sub_row)
        header.addLayout(titles, 1)
        body.addLayout(header)
        body.addSpacing(gap)

        load = QHBoxLayout()
        load.setSpacing(12 if c else 16)
        figure = QVBoxLayout()
        figure.setSpacing(2)
        self._figure = _Figure("%", scale=0.8 if c else 1.0)
        figure.addWidget(self._figure)
        self._load_key = _KeyLabel("GPU load", "accent")
        figure.addWidget(self._load_key)
        load.addLayout(figure)
        self._spark = Sparkline(capacity=self.history.load.maxlen or 60, height=40 if c else 52)
        self._spark.tone2 = "neutral"
        self._spark.setToolTip("Load and memory use over the last minute")
        load.addWidget(self._spark, 1, Qt.AlignmentFlag.AlignBottom)
        body.addLayout(load)
        body.addSpacing(gap)

        mem_row = QHBoxLayout()
        mem_row.setSpacing(8)
        self._mem_key = _KeyLabel("Memory", "neutral")
        mem_row.addWidget(self._mem_key)
        mem_row.addStretch(1)
        self._mem_value = label("—", "mono")
        mem_row.addWidget(self._mem_value)
        body.addLayout(mem_row)
        body.addSpacing(6 if c else 8)
        self._meter = MeterBar(tone="neutral", height=5 if c else 6)
        body.addWidget(self._meter)

        self._chips_row = QWidget(self._content)
        chips = QHBoxLayout(self._chips_row)
        chips.setContentsMargins(0, 10 if c else 14, 0, 0)
        chips.setSpacing(6)
        self._temp = _Chip("thermometer")
        self._power = _Chip("zap")
        chips.addWidget(self._temp)
        chips.addWidget(self._power)
        chips.addStretch(1)
        body.addWidget(self._chips_row)
        body.addStretch(1)

    # ------------------------------------------------------------------ api
    @property
    def sample(self) -> GPUSample:
        return self._sample

    @property
    def key(self) -> str:
        return self._sample.key

    def set_sample(self, sample: GPUSample, record: bool = True) -> None:
        """Show a new reading (and append it to the history unless ``record`` is False)."""
        self._sample = sample
        if record:
            self.history.push(sample)
        kind = sample.kind
        self._tile.set(_ICONS.get(kind, "chip"),
                       "accent" if kind == DeviceKind.CUDA else "muted" if sample.runnable else "faint")
        self._name.setText(device_title(sample))
        self._name.setToolTip(sample.name)
        self._caption.setText(device_caption(sample))
        self._index.setVisible(kind == DeviceKind.CUDA)
        if self._index.text() != f"GPU {sample.index}":
            self._index.set(f"GPU {sample.index}")
        state = device_state(sample)
        if state != (self._state.text(), self._state_tone):
            self._state_tone = state[1]
            self._state.set(*state)
        self._figure.set_value(f"{sample.utilization:.0f}")
        self._load_key.set_text("CPU load" if kind == DeviceKind.CPU else "GPU load")
        self._mem_key.set_text({DeviceKind.CUDA: "Memory", DeviceKind.CPU: "RAM"}.get(kind, "Unified memory"))
        self._mem_value.setText(memory_text(sample))
        self._meter.set_value(sample.memory_fraction)
        self._spark.set_values(self.history.load, self.history.memory)
        has_temp, has_power = sample.temperature_c is not None, sample.power_w is not None
        self._temp.setVisible(has_temp)
        self._power.setVisible(has_power)
        self._chips_row.setVisible(has_temp or has_power)
        if has_temp:
            self._temp.set_text(f"{sample.temperature_c:.0f} °C")
        if has_power:
            limit = f" / {sample.power_limit_w:.0f}" if sample.power_limit_w else ""
            self._power.set_text(f"{sample.power_w:.0f}{limit} W")
        self._apply_enabled(sample.runnable)

    def _apply_enabled(self, runnable: bool) -> None:
        effect = self._content.graphicsEffect()
        if runnable and effect is not None:
            self._content.setGraphicsEffect(None)
        elif not runnable and effect is None:
            fx = QGraphicsOpacityEffect(self._content)
            fx.setOpacity(0.5)
            self._content.setGraphicsEffect(fx)
        self.setToolTip("" if runnable else NOT_SUPPORTED_HINT)
        can_select = self._selectable and runnable
        self.setCursor(Qt.CursorShape.PointingHandCursor if can_select else Qt.CursorShape.ArrowCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus if can_select else Qt.FocusPolicy.NoFocus)

    def isSelected(self) -> bool:  # noqa: N802 - Qt naming
        return self._selected

    def setSelected(self, selected: bool) -> None:  # noqa: N802
        if selected != self._selected:
            self._selected = selected
            self.update()

    # ------------------------------------------------------------------ events
    def event(self, e: QEvent) -> bool:
        if e.type() in (QEvent.Type.HoverEnter, QEvent.Type.HoverLeave):
            self._hover = e.type() == QEvent.Type.HoverEnter
            self.update()
        return super().event(e)

    def _activate(self) -> None:
        if self._selectable and self._sample.runnable:
            self.clicked.emit(self.key)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self._activate()
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._activate()
        else:
            super().keyPressEvent(e)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        radius = theme().metrics.r_lg
        if self._selected:
            painter.setPen(QPen(p.qcolor("accent", 0.22), 3.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, radius, radius)
        interactive = self._selectable and self._sample.runnable
        fill = QColor(p.bg3 if (self._hover and interactive and not self._selected) else p.bg2)
        border = QColor(p.accent) if self._selected else QColor(
            p.line2 if (self._hover and interactive) or self.hasFocus() else p.line)
        painter.setPen(QPen(border, 1.0))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius - 1, radius - 1)


# ============================================================================ grid
class GpuGrid(QWidget):
    """Responsive grid of :class:`GpuCard` (1–4 columns, by width).

    Call :meth:`update_samples` with every telemetry reading; cards are created,
    updated and removed by :attr:`GPUSample.key`, and each keeps a 60 s history.
    With ``selectable=True`` one runnable device can be picked (:attr:`selection_changed`
    emits the selected :class:`GPUSample`, or ``None``).
    """

    selection_changed = Signal(object)

    MIN_CARD_WIDTH = 280
    COMPACT_CARD_WIDTH = 230
    SPACING = 12

    def __init__(self, *, selectable: bool = False, interval: float = 1.0, max_columns: int = 4,
                 compact: bool = False, parent: QWidget | None = None) -> None:
        """``compact`` uses the smaller tiles on a fixed column grid (they do not stretch to
        fill the row), matching a :class:`ResponsiveGrid` with ``fill=False`` above it."""
        super().__init__(parent)
        self.selectable = selectable
        self.interval = interval
        self.max_columns = max_columns
        self.compact = compact
        self.min_card_width = self.COMPACT_CARD_WIDTH if compact else self.MIN_CARD_WIDTH
        self._cards: dict[str, GpuCard] = {}
        self._order: list[str] = []
        self._selected: str | None = None
        self._columns = 0
        self._has_data = False

        self._stack = QStackedLayout(self)
        self._loading = QWidget()
        lay = QHBoxLayout(self._loading)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(10)
        lay.addWidget(Spinner(16))
        self._loading_text = label("Reading device telemetry…", "muted")
        lay.addWidget(self._loading_text)
        self._empty = EmptyState("gpu", "No compute devices",
                                 "No NVIDIA GPU or CPU telemetry is available for this machine.")
        self._page = QWidget()
        # Columns follow the width (see _relayout), so the grid must never force its parent
        # to be wider than one card: an explicit minimum overrides the layout's.
        self._page.setMinimumWidth(GpuCard.MIN_WIDTH)
        self._grid = QGridLayout(self._page)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(self.SPACING)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        for page in (self._loading, self._empty, self._page):
            self._stack.addWidget(page)
        self._stack.setCurrentWidget(self._loading)

    # ------------------------------------------------------------------ state pages
    def set_loading(self, text: str = "Reading device telemetry…") -> None:
        """Show the spinner page (until the next :meth:`update_samples`)."""
        self._loading_text.setText(text)
        self._has_data = False
        self._stack.setCurrentWidget(self._loading)

    def set_error(self, message: str | None) -> None:
        """Explain why there is no telemetry (ignored while cards are showing)."""
        if message and not self._cards:
            self._empty.title.setText("Telemetry unavailable")
            self._empty.message.setText(message)
            self._stack.setCurrentWidget(self._empty)

    # ------------------------------------------------------------------ data
    def update_samples(self, samples: list[GPUSample]) -> None:
        """Show a reading of every device (keeps order: CUDA, CPU, Apple)."""
        self._has_data = True
        keys = [s.key for s in samples]
        for key in [k for k in self._cards if k not in keys]:
            card = self._cards.pop(key)
            self._grid.removeWidget(card)
            card.deleteLater()
            if key == self._selected:
                self._selected = None
                self.selection_changed.emit(None)
        for sample in samples:
            card = self._cards.get(sample.key)
            if card is None:
                card = GpuCard(sample, selectable=self.selectable, interval=self.interval,
                               compact=self.compact, parent=self._page)
                card.clicked.connect(self.select)
                self._cards[sample.key] = card
                card.setSelected(sample.key == self._selected)
            else:
                card.set_sample(sample)
        if keys != self._order:
            self._order = keys
            self._relayout(force=True)
        if samples:
            self._stack.setCurrentWidget(self._page)
        else:
            self._empty.title.setText("No compute devices")
            self._empty.message.setText("No NVIDIA GPU or CPU telemetry is available for this machine.")
            self._stack.setCurrentWidget(self._empty)

    def cards(self) -> list[GpuCard]:
        return [self._cards[k] for k in self._order if k in self._cards]

    # ------------------------------------------------------------------ selection
    def select(self, key: str | None) -> None:
        """Select a device by key (``None`` clears); non-runnable devices are ignored."""
        card = self._cards.get(key) if key else None
        if key and (card is None or not card.sample.runnable):
            return
        if key == self._selected:
            return
        self._selected = key
        for k, c in self._cards.items():
            c.setSelected(k == key)
        self.selection_changed.emit(card.sample if card else None)

    def selected(self) -> GPUSample | None:
        card = self._cards.get(self._selected) if self._selected else None
        return card.sample if card else None

    # ------------------------------------------------------------------ layout
    def columns_for(self, width: int) -> int:
        """How many columns fit ``width``. A few cards share the row, like the machine
        cards above them, but there are at least two columns: one card never spans it."""
        fit = (width + self.SPACING) // (self.min_card_width + self.SPACING)
        if self.compact:
            return max(1, min(self.max_columns, fit))
        return max(1, min(self.max_columns, fit, max(2, len(self._order))))

    def _relayout(self, force: bool = False) -> None:
        # Until the grid is on screen its width is meaningless: assume the widest layout,
        # so the size hint does not make the parent window grow.
        width = self.width() if self.isVisible() else self.max_columns * (self.min_card_width + self.SPACING)
        columns = self.columns_for(width)
        if columns == self._columns and not force:
            return
        self._columns = columns
        for card in self._cards.values():
            self._grid.removeWidget(card)
        for i, key in enumerate(self._order):
            self._grid.addWidget(self._cards[key], i // columns, i % columns)
        for c in range(self.max_columns):
            self._grid.setColumnStretch(c, 1 if c < columns else 0)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._relayout()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._relayout()


# ============================================================================ compact list
class _DeviceRow(QFrame):
    clicked = Signal(str)

    def __init__(self, sample: GPUSample, selectable: bool, interval: float,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sample = sample
        self._selectable = selectable
        self._selected = False
        self._hover = False
        self.history = _History(_capacity(interval))
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(10)
        self._icon = IconTile("gpu", "accent", size=28)
        lay.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(8)
        self._name = ElidedLabel("")
        self._name.setStyleSheet("font-weight: 500;")
        top.addWidget(self._name, 1)
        self._value = label("", "mono")
        top.addWidget(self._value)
        col.addLayout(top)
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        self._meter = MeterBar(tone="neutral", height=4)
        self._meter.setFixedWidth(56)
        bottom.addWidget(self._meter, 0, Qt.AlignmentFlag.AlignVCenter)
        self._spark = Sparkline(capacity=self.history.load.maxlen or 60, height=18)
        self._spark.tone2 = "neutral"
        bottom.addWidget(self._spark, 1)
        col.addLayout(bottom)
        lay.addLayout(col, 1)
        theme().changed.connect(self.update)
        self.set_sample(sample)

    @property
    def sample(self) -> GPUSample:
        return self._sample

    def set_sample(self, sample: GPUSample, record: bool = True) -> None:
        self._sample = sample
        if record:
            self.history.push(sample)
        self._icon.set(_ICONS.get(sample.kind, "chip"),
                       "accent" if sample.kind == DeviceKind.CUDA else "muted" if sample.runnable else "faint")
        prefix = f"{sample.index} · " if sample.kind == DeviceKind.CUDA else ""
        self._name.setText(prefix + device_title(sample))
        self._value.setText(f"{sample.utilization:.0f}%")
        self._meter.set_value(sample.memory_fraction)
        self._meter.setToolTip(f"Memory {memory_text(sample)}")
        self._spark.set_values(self.history.load, self.history.memory)
        self.setEnabled(sample.runnable)
        self.setToolTip(sample.name if sample.runnable else NOT_SUPPORTED_HINT)
        can_select = self._selectable and sample.runnable
        self.setCursor(Qt.CursorShape.PointingHandCursor if can_select else Qt.CursorShape.ArrowCursor)

    def setSelected(self, selected: bool) -> None:  # noqa: N802
        self._selected = selected
        self.update()

    def event(self, e: QEvent) -> bool:
        if e.type() in (QEvent.Type.HoverEnter, QEvent.Type.HoverLeave):
            self._hover = e.type() == QEvent.Type.HoverEnter
            self.update()
        return super().event(e)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if (e.button() == Qt.MouseButton.LeftButton and self._selectable and self._sample.runnable
                and self.rect().contains(e.position().toPoint())):
            self.clicked.emit(self._sample.key)
        super().mouseReleaseEvent(e)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        if not (self._selected or (self._hover and self._selectable and self._sample.runnable)):
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(p.qcolor("accent", 0.16 if p.dark else 0.10) if self._selected else QColor(p.bg4))
        painter.drawRoundedRect(rect, 8, 8)
        if self._selected:
            painter.setBrush(QColor(p.accent))
            painter.drawRoundedRect(QRectF(rect.x(), rect.y() + 10, 3, rect.height() - 20), 1.5, 1.5)


class DeviceList(QWidget):
    """Compact, optionally selectable device rows for narrow side panels.

    Same data API as :class:`GpuGrid`: :meth:`update_samples`, :meth:`select`,
    :meth:`selected` and :attr:`selection_changed`.
    """

    selection_changed = Signal(object)

    def __init__(self, *, selectable: bool = True, interval: float = 1.0,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.selectable = selectable
        self.interval = interval
        self._rows: dict[str, _DeviceRow] = {}
        self._selected: str | None = None
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(2)
        self._placeholder = QWidget()
        ph = QHBoxLayout(self._placeholder)
        ph.setContentsMargins(10, 8, 10, 8)
        ph.setSpacing(8)
        ph.addWidget(Spinner(14))
        self._placeholder_text = label("Reading devices…", "caption")
        ph.addWidget(self._placeholder_text, 1)
        self._lay.addWidget(self._placeholder)

    def set_error(self, message: str | None) -> None:
        if message and not self._rows:
            self._placeholder_text.setText(message)

    def update_samples(self, samples: list[GPUSample]) -> None:
        keys = [s.key for s in samples]
        for key in [k for k in self._rows if k not in keys]:
            row = self._rows.pop(key)
            self._lay.removeWidget(row)
            row.deleteLater()
            if key == self._selected:
                self._selected = None
                self.selection_changed.emit(None)
        for i, sample in enumerate(samples):
            row = self._rows.get(sample.key)
            if row is None:
                row = _DeviceRow(sample, self.selectable, self.interval, self)
                row.clicked.connect(self.select)
                self._rows[sample.key] = row
                row.setSelected(sample.key == self._selected)
            else:
                row.set_sample(sample)
            if self._lay.indexOf(row) != i + 1:
                self._lay.insertWidget(i + 1, row)
        self._placeholder.setVisible(not samples)
        if not samples:
            self._placeholder_text.setText("No compute devices found")

    def select(self, key: str | None) -> None:
        row = self._rows.get(key) if key else None
        if key and (row is None or not row.sample.runnable):
            return
        if key == self._selected:
            return
        self._selected = key
        for k, r in self._rows.items():
            r.setSelected(k == key)
        self.selection_changed.emit(row.sample if row else None)

    def selected(self) -> GPUSample | None:
        row = self._rows.get(self._selected) if self._selected else None
        return row.sample if row else None


# ============================================================================ feed
class GpuFeed(QObject):
    """Runs a :class:`GpuMonitor` and re-emits its readings on the UI thread.

    ``samples`` carries ``list[GPUSample]``; ``error`` carries a message, or ``None``
    once telemetry recovers.
    """

    samples = Signal(object)
    error = Signal(object)

    def __init__(self, machine: Machine, interval: float = 1.0, parent: QObject | None = None) -> None:
        super().__init__(parent)
        monitor = self._monitor = GpuMonitor(machine, interval, callback=self._emit_samples,
                                             on_error=self._emit_error)
        self.destroyed.connect(lambda *_: monitor.stop(timeout=0))

    def _emit_samples(self, samples: list[GPUSample]) -> None:
        try:
            self.samples.emit(samples)
        except RuntimeError:  # the feed was deleted while the monitor was sampling
            self._monitor.stop(timeout=0)

    def _emit_error(self, message: str | None) -> None:
        try:
            self.error.emit(message)
        except RuntimeError:
            self._monitor.stop(timeout=0)

    @property
    def machine(self) -> Machine:
        return self._monitor.machine

    @property
    def monitor(self) -> GpuMonitor:
        return self._monitor

    def start(self) -> None:
        self._monitor.start()

    def stop(self) -> None:
        """Ask the monitor to stop without blocking the UI (threads exit on their own)."""
        self._monitor.stop(timeout=0)
