"""Custom-painted controls: segmented control and toggle switch.

Both draw a validation ``issue`` ("error" or "warning") and a ``flash`` (a field just
revealed) as a coloured outline, from the dynamic properties the stylesheet uses on inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QEvent,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import QAbstractButton, QSizePolicy, QToolTip, QWidget

from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import Palette, theme
from torchcor_studio.ui.widgets.basic import tone_color


def ring_color(widget: QWidget, p: Palette) -> QColor | None:
    """The outline a control shows for its ``flash`` or ``issue`` property (``None``: none)."""
    if widget.property("flash"):
        return QColor(p.accent)
    return {"error": QColor(p.danger), "warning": QColor(p.warning)}.get(widget.property("issue"))


class SegmentedControl(QWidget):
    """A row of mutually exclusive options with a sliding selection pill.

    ``options`` are ``(value, label)`` or ``(value, label, icon)`` tuples. Emits
    :attr:`changed` with the selected value. ``height`` overrides the control's height,
    ``tones`` maps a value to the tone of its icon while it is selected, and ``tips`` to
    the tooltip of its segment.
    """

    changed = Signal(object)

    def __init__(self, options: Sequence[tuple], value=None, *, compact: bool = False,
                 expand: bool = False, height: int | None = None, tones: dict | None = None,
                 tips: dict | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._opts = [(o[0], o[1], o[2] if len(o) > 2 else None) for o in options]
        self._index = 0
        self._hover = -1
        self._pos = 0.0
        self._compact = compact
        self._expand = expand
        self._height = height or (28 if compact else 32)
        self._tones = tones or {}
        self._tips = tips or {}
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding if expand else QSizePolicy.Policy.Fixed,
                           QSizePolicy.Policy.Fixed)
        self._anim = QPropertyAnimation(self, b"slide", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        if value is not None:
            self.set_value(value, animate=False)

    # ------------------------------------------------------------------ api
    def value(self):
        return self._opts[self._index][0] if self._opts else None

    def set_value(self, value, animate: bool = True) -> None:
        for i, (v, *_rest) in enumerate(self._opts):
            if v == value:
                self._select(i, animate=animate, emit=False)
                return

    def set_options(self, options: Sequence[tuple], value=None) -> None:
        self._opts = [(o[0], o[1], o[2] if len(o) > 2 else None) for o in options]
        self._index = 0
        self._pos = 0.0
        if value is not None:
            self.set_value(value, animate=False)
        self.updateGeometry()
        self.update()

    # ------------------------------------------------------------------ geometry
    def _font(self):
        f = self.font()
        f.setPixelSize(12 if self._compact else 13)
        f.setWeight(f.Weight.Medium)
        return f

    def _segment_widths(self) -> list[float]:
        fm = QFontMetrics(self._font())
        pad = 20 if self._compact else 28
        widths = [fm.horizontalAdvance(lbl) + pad + (18 if ic else 0) for _v, lbl, ic in self._opts]
        if self._expand and self.width() > 0 and widths:
            total = self.width() - 4
            return [total / len(widths)] * len(widths)
        return widths

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(int(sum(self._segment_widths())) + 4, self._height)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint() if not self._expand else QSize(60 * max(1, len(self._opts)), self.sizeHint().height())

    def _rects(self) -> list[QRectF]:
        x, rects = 2.0, []
        for w in self._segment_widths():
            rects.append(QRectF(x, 2, w, self.height() - 4))
            x += w
        return rects

    # ------------------------------------------------------------------ animation
    def _get_pos(self) -> float:
        return self._pos

    def _set_pos(self, v: float) -> None:
        self._pos = v
        self.update()

    slide = Property(float, _get_pos, _set_pos)

    def _select(self, index: int, animate: bool = True, emit: bool = True) -> None:
        if not (0 <= index < len(self._opts)):
            return
        changed = index != self._index
        self._index = index
        if animate:
            self._anim.stop()
            self._anim.setStartValue(self._pos)
            self._anim.setEndValue(float(index))
            self._anim.start()
        else:
            self._pos = float(index)
            self.update()
        if emit and changed:
            self.changed.emit(self.value())

    # ------------------------------------------------------------------ events
    def mousePressEvent(self, e) -> None:  # noqa: N802
        for i, r in enumerate(self._rects()):
            if r.contains(e.position()):
                self._select(i)
                return

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        hover = next((i for i, r in enumerate(self._rects()) if r.contains(e.position())), -1)
        if hover != self._hover:
            self._hover = hover
            self.update()

    def leaveEvent(self, _e) -> None:  # noqa: N802
        self._hover = -1
        self.update()

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.ToolTip and self._tips:
            index = next((i for i, r in enumerate(self._rects()) if r.contains(QPointF(e.pos()))), -1)
            tip = self._tips.get(self._opts[index][0], "") if index >= 0 else ""
            if tip:
                QToolTip.showText(e.globalPos(), tip, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(e)

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self._select(max(0, self._index - 1))
        elif e.key() in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self._select(min(len(self._opts) - 1, self._index + 1))
        else:
            super().keyPressEvent(e)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QColor(p.line))
        painter.setBrush(QColor(p.bg1 if p.dark else p.bg4))
        painter.drawRoundedRect(outer, 8, 8)
        rects = self._rects()
        if not rects:
            return
        # sliding pill
        lo, hi = int(self._pos), min(len(rects) - 1, int(self._pos) + 1)
        t = self._pos - lo
        a, b = rects[lo], rects[hi]
        pill = QRectF(a.x() + (b.x() - a.x()) * t, a.y(), a.width() + (b.width() - a.width()) * t, a.height())
        painter.setPen(QColor(p.line2) if p.dark else QColor(p.line))
        painter.setBrush(QColor(p.bg4 if p.dark else p.bg2))
        painter.drawRoundedRect(pill.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
        painter.setFont(self._font())
        fm = QFontMetrics(self._font())
        for i, (r, (value, text, ic)) in enumerate(zip(rects, self._opts, strict=False)):
            active = i == self._index
            color = QColor(p.text if active else (p.text if i == self._hover else p.text2))
            tw = fm.horizontalAdvance(text) + (18 if ic else 0)
            x = r.x() + (r.width() - tw) / 2
            if ic:
                tone = self._tones.get(value)
                tint = (p.text if tone == "neutral" else tone_color(p, tone)) if tone else p.accent_text
                painter.drawPixmap(int(x), int(r.center().y() - 7),
                                   icons.pixmap(ic, tint if active else color.name(), 14))
                x += 18
            painter.setPen(color)
            painter.drawText(QRectF(x, r.y(), r.width(), r.height()), Qt.AlignmentFlag.AlignVCenter, text)
        ring = ring_color(self, p) or (QColor(p.accent) if self.hasFocus() else None)
        if ring is not None:
            painter.setPen(ring)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(outer, 8, 8)


class ToggleSwitch(QAbstractButton):
    """An animated on/off switch (checkable)."""

    def __init__(self, checked: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(36, 20)
        self._offset = 1.0 if checked else 0.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _get(self) -> float:
        return self._offset

    def _set(self, v: float) -> None:
        self._offset = v
        self.update()

    offset = Property(float, _get, _set)

    def _animate(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        super().setChecked(checked)
        if self._anim_ready():
            self._offset = 1.0 if checked else 0.0
            self.update()

    def _anim_ready(self) -> bool:
        return hasattr(self, "_anim") and self._anim.state() != QPropertyAnimation.State.Running

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(36, 20)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        off, on = QColor(p.line2), QColor(p.accent)
        t = self._offset
        track = QColor(
            round(off.red() + (on.red() - off.red()) * t),
            round(off.green() + (on.green() - off.green()) * t),
            round(off.blue() + (on.blue() - off.blue()) * t),
        )
        if not self.isEnabled():
            track.setAlphaF(0.45)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = r.height() - 4
        x = r.x() + 2 + (r.width() - d - 4) * t
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawEllipse(QRectF(x, r.y() + 2, d, d))
        ring = ring_color(self, p)
        if ring is not None:
            painter.setPen(ring)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            painter.drawRoundedRect(outer, outer.height() / 2, outer.height() / 2)
