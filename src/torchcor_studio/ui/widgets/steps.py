"""Progress indicators: wizard step header, pipeline stage tracker, selectable cards."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets.basic import label, tone_color


class StepIndicator(QWidget):
    """``1 Compute — 2 Simulation — 3 Run — 4 Result``: done steps get a green tick, and the
    step shown is underlined.

    Steps up to ``reachable`` can be clicked (:attr:`step_clicked`), unless the indicator
    is not ``interactive``.
    """

    step_clicked = Signal(int)

    def __init__(self, steps: Sequence[str], *, interactive: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._steps = list(steps)
        self._current = 0
        self._reachable = 0
        self._selected = 0
        self._tone = "accent"
        self._mark: str | None = None
        self._interactive = interactive
        self.setMouseTracking(interactive)
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_current(self, index: int, reachable: int | None = None, *, tone: str = "accent",
                    mark: str | None = None, selected: int | None = None) -> None:
        """Step ``index`` is under way (``len(steps)``: all are done). Its disc takes the
        ``tone`` colour and shows the ``mark`` icon, if given, instead of its number.
        ``selected`` is the step shown (by default the one under way)."""
        reachable = max(index, reachable if reachable is not None else index)
        selected = index if selected is None else selected
        state = (index, reachable, selected, tone, mark)
        if state == (self._current, self._reachable, self._selected, self._tone, self._mark):
            return
        self._current, self._reachable, self._selected, self._tone, self._mark = state
        self.update()

    def _font(self) -> QFont:
        f = self.font()
        f.setPixelSize(13)
        f.setWeight(QFont.Weight.Medium)
        return f

    def _items(self) -> list[QRectF]:
        fm = QFontMetrics(self._font())
        x, out = 0.0, []
        for text in self._steps:
            w = 22 + 8 + fm.horizontalAdvance(text)
            out.append(QRectF(x, 0, w, self.height()))
            x += w + 44
        return out

    def sizeHint(self) -> QSize:  # noqa: N802
        items = self._items()
        return QSize(int(items[-1].right()) + 2 if items else 0, 32)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if not self._interactive:
            return
        for i, r in enumerate(self._items()):
            if r.contains(e.position()) and i <= self._reachable and i != self._selected:
                self.step_clicked.emit(i)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if not self._interactive:
            return
        over = any(r.contains(e.position()) and i <= self._reachable and i != self._selected
                   for i, r in enumerate(self._items()))
        self.setCursor(Qt.CursorShape.PointingHandCursor if over else Qt.CursorShape.ArrowCursor)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font())
        items = self._items()
        for i, (r, text) in enumerate(zip(items, self._steps, strict=False)):
            done = i < self._current
            active = i == self._current
            circle = QRectF(r.x(), r.center().y() - 11, 22, 22)
            if active:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(tone_color(p, self._tone)))
            elif done:  # done: a green disc with a white tick
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(p.success))
            else:
                painter.setPen(QPen(QColor(p.line2), 1.2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(circle)
            if done or (active and self._mark):
                mark = self._mark if active else "check"
                painter.drawPixmap(int(circle.x() + 4), int(circle.y() + 4), icons.pixmap(mark, "#FFFFFF", 14, 2.6))
            else:
                painter.setPen(QColor(p.on_accent if active else p.text3))
                f = self._font()
                f.setPixelSize(12)
                f.setWeight(QFont.Weight.Bold)
                painter.setFont(f)
                painter.drawText(circle, Qt.AlignmentFlag.AlignCenter, str(i + 1))
                painter.setFont(self._font())
            shown = i == self._selected
            painter.setPen(QColor(p.text if shown or active else (p.text2 if i <= self._reachable or done
                                                                   else p.text3)))
            text_rect = QRectF(circle.right() + 8, r.y(), r.width() - 30, r.height())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, text)
            if shown:  # underlined, like a selected tab
                painter.fillRect(QRectF(text_rect.x(), r.bottom() - 2, text_rect.width(), 2), QColor(p.accent))
            if i < len(items) - 1:
                y = r.center().y()
                pen = QPen(QColor(p.success if done else p.line2), 1.2)
                painter.setPen(pen)
                painter.drawLine(int(r.right() + 10), int(y), int(items[i + 1].left() - 10), int(y))


class StageTracker(QWidget):
    """Horizontal pipeline: done stages get a check, the live one pulses.

    Labels alternate below and above the line, so each has two stages' width (and is
    shortened if even that is too narrow): they never overlap.
    """

    LINE_Y = 28  # the line through the stage dots
    LABEL_H = 16

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stages: list[tuple[str, str]] = []
        self._current = ""
        self._state = "running"
        self._phase = 0.0
        self.setFixedHeight(2 * self.LINE_Y)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

    def set_stages(self, stages: Sequence[tuple[str, str]]) -> None:
        self._stages = list(stages)
        self.update()

    def set_state(self, current_stage: str, run_state: str) -> None:
        self._current = current_stage
        self._state = run_state
        if run_state == "running":
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.04) % 1.0
        self.update()

    def _font(self) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(11)
        return font

    def labels(self) -> list[tuple[QRectF, str]]:
        """Each stage's label: where it goes, below the line for even stages and above it for
        odd ones (up to the next label on the same side), and its text as shown there."""
        n, w = len(self._stages), self.width()
        step = w / max(1, n)
        fm = QFontMetrics(self._font())
        out = []
        for i, (_sid, text) in enumerate(self._stages):
            cx = step * i + step / 2
            left, right = max(0.0, cx - step + 4), min(float(w), cx + step - 4)
            top = self.LINE_Y - 12 - self.LABEL_H if i % 2 else self.LINE_Y + 12
            out.append((QRectF(left, top, right - left, self.LABEL_H),
                        fm.elidedText(text, Qt.TextElideMode.ElideRight, int(right - left))))
        return out

    def _index(self) -> int:
        if self._state == "completed":
            return len(self._stages)
        ids = [s for s, _ in self._stages]
        return ids.index(self._current) if self._current in ids else (0 if self._state != "queued" else -1)

    def paintEvent(self, _e) -> None:  # noqa: N802
        if not self._stages:
            return
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(self._stages)
        cur = self._index()
        step = self.width() / n
        painter.setFont(self._font())
        failed = self._state in ("failed", "lost")
        cy = self.LINE_Y
        for i, (rect, shown) in enumerate(self.labels()):
            cx = step * i + step / 2
            if i < n - 1:
                pen = QPen(QColor(p.success if i < cur else p.line2), 2)
                painter.setPen(pen)
                painter.drawLine(int(cx + 9), cy, int(cx + step - 9), cy)
            done = i < cur
            live = i == cur and self._state == "running"
            broken = i == cur and failed
            painter.setPen(Qt.PenStyle.NoPen)
            if live:
                halo = QColor(p.accent)
                halo.setAlphaF(0.35 * (1 - self._phase))
                painter.setBrush(halo)
                rad = 7 + 6 * self._phase
                painter.drawEllipse(QRectF(cx - rad, cy - rad, 2 * rad, 2 * rad))
            if done:
                painter.setBrush(QColor(p.success))
                painter.drawEllipse(QRectF(cx - 7, cy - 7, 14, 14))
                painter.drawPixmap(int(cx - 5), int(cy - 5), icons.pixmap("check", p.on_accent, 10, 3))
            elif broken:
                painter.setBrush(QColor(p.danger))
                painter.drawEllipse(QRectF(cx - 7, cy - 7, 14, 14))
                painter.drawPixmap(int(cx - 5), int(cy - 5), icons.pixmap("x", p.on_accent, 10, 3))
            elif live:
                painter.setBrush(QColor(p.accent))
                painter.drawEllipse(QRectF(cx - 6, cy - 6, 12, 12))
            else:
                painter.setPen(QPen(QColor(p.line2), 1.5))
                painter.setBrush(QColor(p.bg2))
                painter.drawEllipse(QRectF(cx - 6, cy - 6, 12, 12))
            painter.setPen(QColor(p.text if (live or broken) else (p.text2 if done else p.text3)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, shown)


class SelectCard(QFrame):
    """A clickable card with an icon disc (none when ``icon`` is ``None``), title, caption and
    a row of chips (:attr:`chips`); ``checked`` shows the accent ring, and :meth:`set_mark`
    paints a small icon at the top right."""

    clicked = Signal()

    def __init__(self, icon: str | None, title: str, caption: str = "", *, tone: str = "accent",
                 footer: str = "", compact: bool = False, parent: QWidget | None = None) -> None:
        """``compact``: the same card at about 80 % scale."""
        super().__init__(parent)
        self._checked = False
        self._hover = False
        self._icon = icon
        self._tone = tone
        self._mark: tuple[str, str] | None = None
        self._pad = 11 if compact else 14
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMinimumHeight(68 if compact else 84)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(self._pad, self._pad, self._pad, self._pad)
        lay.setSpacing(10 if compact else 12)
        self._disc = _Disc(icon, tone, 30 if compact else 36) if icon else None
        if self._disc is not None:
            lay.addWidget(self._disc, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(3)
        self.title = label(title, wrap=True)
        self.title.setStyleSheet(f"font-weight: 600; font-size: {13 if compact else 14}px;")
        col.addWidget(self.title)
        self.caption = label(caption, "caption", wrap=True)
        self.caption.setVisible(bool(caption))
        col.addWidget(self.caption)
        self.footer = label(footer, "faint", wrap=True)
        self.footer.setStyleSheet("font-size: 11px;")
        self.footer.setVisible(bool(footer))
        col.addWidget(self.footer)
        col.addStretch(1)
        self.chips = QHBoxLayout()
        self.chips.setSpacing(8)
        # aligned: an empty row would otherwise cap the column's width (cards without chips)
        self.chips.setAlignment(Qt.AlignmentFlag.AlignLeft)
        col.addLayout(self.chips)
        lay.addLayout(col, 1)
        self.trailing = QHBoxLayout()
        lay.addLayout(self.trailing)

    def set_checked(self, on: bool) -> None:
        self._checked = on
        self.update()

    def is_checked(self) -> bool:
        return self._checked

    def set_icon(self, icon: str, tone: str | None = None) -> None:
        if self._disc is not None:
            self._disc.set(icon, tone or self._tone)

    def set_mark(self, icon: str | None, tone: str = "accent") -> None:
        """A 16 px icon at the top right (``None``: none)."""
        mark = (icon, tone) if icon else None
        if mark != self._mark:
            self._mark = mark
            self.update()

    def enterEvent(self, e) -> None:  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setBrush(QColor(p.bg3 if (self._hover and not self._checked) else p.bg2))
        if self._checked:
            painter.setPen(QPen(QColor(p.accent), 1.6))
        else:
            painter.setPen(QPen(QColor(p.line2 if self._hover else p.line), 1))
        painter.drawRoundedRect(r, 12, 12)
        if self._checked:
            glow = QColor(p.accent)
            glow.setAlphaF(0.07)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawRoundedRect(r, 12, 12)
        if self._mark is not None:
            icon, tone = self._mark
            color = p.accent_text if tone == "accent_text" else tone_color(p, tone)
            painter.drawPixmap(self.width() - self._pad - 16, self._pad, icons.pixmap(icon, color, 16))


class _Disc(QWidget):
    def __init__(self, icon: str, tone: str, size: int = 36) -> None:
        super().__init__()
        self._icon, self._tone = icon, tone
        self.setFixedSize(size, size)

    def set(self, icon: str, tone: str) -> None:
        self._icon, self._tone = icon, tone
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(tone_color(p, self._tone))
        fill = QColor(color)
        fill.setAlphaF(0.14)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(QRectF(self.rect()), 10, 10)
        painter.drawPixmap(9, 9, icons.pixmap(self._icon, color.name(), 18))
