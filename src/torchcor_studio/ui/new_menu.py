"""The New menu: the kinds of simulation in a list, each with its icon, its name and a line
on what it simulates. It opens beside the sidebar's New button when the pointer rests on
it, on a click, and from Ctrl+N; choosing a kind starts a simulation of that kind."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QCursor, QFont, QFontMetrics, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QWidget

from torchcor_studio.core.simulations import specs
from torchcor_studio.ui import kinds
from torchcor_studio.ui.theme import theme

OPEN_MS = 120  # how long the pointer rests on the button before the menu opens
CLOSE_MS = 300  # how long it is away from the button and the menu before the menu closes


def _font(px: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setPixelSize(px)
    font.setWeight(weight)
    return font


class NewMenu(QWidget):
    """The popup list, painted (one widget). Up and Down move between the kinds, Enter or a
    click chooses one, Escape or a click elsewhere closes it. Opened by the pointer resting
    on its button, it also closes once the pointer has left the button and the menu."""

    chosen = Signal(str)  # kind
    closed = Signal()

    PAD, HEAD, ROW, WIDTH = 6, 30, 54, 456  # wide enough for every kind's line

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setAccessibleName("New simulation")
        self.kinds = [(str(s.kind), s.title, f"{s.tagline}.", s.badge) for s in specs()]
        self._current = -1  # the row under the pointer or chosen with the arrow keys
        self._trigger: QWidget | None = None
        self._sticky = True
        self._leave = QTimer(self)
        self._leave.setSingleShot(True)
        self._leave.setInterval(CLOSE_MS)
        self._leave.timeout.connect(self.close)
        self.setFixedSize(self.WIDTH, 2 * self.PAD + self.HEAD + self.ROW * len(self.kinds))

    def popup(self, trigger: QWidget, at: QPoint, *, keyboard: bool = False, sticky: bool = True) -> None:
        """Open with its top-left at ``at`` (kept on the screen) for ``trigger``, a press on
        which leaves it open. ``keyboard``: the first kind is current; ``sticky``: it stays
        open until a choice, Escape or a click elsewhere (else it also closes once the
        pointer has left both)."""
        self._trigger, self._sticky = trigger, sticky
        self._current = 0 if keyboard else -1
        screen = QGuiApplication.screenAt(at) or trigger.screen()
        area = screen.availableGeometry()
        self.move(max(area.left(), min(at.x(), area.right() - self.width() + 1)),
                  max(area.top(), min(at.y(), area.bottom() - self.height() + 1)))
        self.show()
        self.setFocus(Qt.FocusReason.PopupFocusReason)

    def current(self) -> str | None:
        """The kind under the pointer or chosen with the arrow keys."""
        return self.kinds[self._current][0] if self._current >= 0 else None

    # ------------------------------------------------------------------ geometry
    def _row(self, i: int) -> QRectF:
        return QRectF(self.PAD, self.PAD + self.HEAD + i * self.ROW, self.width() - 2 * self.PAD, self.ROW)

    def _index_at(self, pos: QPointF) -> int:
        return next((i for i in range(len(self.kinds)) if self._row(i).contains(pos)), -1)

    def _on_trigger(self, pos: QPoint) -> bool:
        trigger = self._trigger
        return trigger is not None and trigger.isVisible() and trigger.rect().contains(trigger.mapFromGlobal(pos))

    def _follow(self, pos: QPoint) -> None:
        """Opened by the pointer: close soon once it is away from the button and the menu
        (the space between them included)."""
        if self._sticky or self._trigger is None:
            return
        trigger = QRect(self._trigger.mapToGlobal(QPoint(0, 0)), self._trigger.size())
        if self.geometry().united(trigger).adjusted(-8, -8, 8, 8).contains(pos):
            self._leave.stop()
        elif not self._leave.isActive():
            self._leave.start()

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(p.qcolor("line2"), 1))
        painter.setBrush(p.qcolor("bg2"))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
        eyebrow = _font(11, QFont.Weight.DemiBold)
        eyebrow.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        painter.setFont(eyebrow)
        painter.setPen(p.qcolor("text3"))
        painter.drawText(QRectF(self.PAD + 12, self.PAD, self.width(), self.HEAD),
                         Qt.AlignmentFlag.AlignVCenter, "NEW SIMULATION")
        title_font, line_font, badge_font = (_font(13, QFont.Weight.DemiBold), _font(12),
                                             _font(11, QFont.Weight.DemiBold))
        for i, (kind, title, line, badge) in enumerate(self.kinds):
            row = self._row(i)
            if i == self._current:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(p.qcolor("bg4"))
                painter.drawRoundedRect(row, 8, 8)
            tile = QRectF(row.x() + 10, row.center().y() - 17, 34, 34)
            kinds.paint_tile(painter, tile, kind)
            fm = QFontMetrics(badge_font)
            pill = QRectF(row.right() - 12 - fm.horizontalAdvance(badge) - 12, row.center().y() - 9,
                          fm.horizontalAdvance(badge) + 12, 18)
            painter.setPen(QPen(p.qcolor("line2"), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(pill.adjusted(0.5, 0.5, -0.5, -0.5), 9, 9)
            painter.setFont(badge_font)
            painter.setPen(p.qcolor("text3"))
            painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, badge)
            left = tile.right() + 12
            room = int(pill.left() - 10 - left)
            painter.setFont(title_font)
            painter.setPen(p.qcolor("text"))
            painter.drawText(QRectF(left, row.y() + 8, room, 19), Qt.AlignmentFlag.AlignVCenter,
                             QFontMetrics(title_font).elidedText(title, Qt.TextElideMode.ElideRight, room))
            painter.setFont(line_font)
            painter.setPen(p.qcolor("text2"))
            painter.drawText(QRectF(left, row.y() + 27, room, 18), Qt.AlignmentFlag.AlignVCenter,
                             QFontMetrics(line_font).elidedText(line, Qt.TextElideMode.ElideRight, room))

    # ------------------------------------------------------------------ input
    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        index = self._index_at(event.position())
        if index != self._current and (index >= 0 or self.rect().contains(event.position().toPoint())):
            self._current = index
            self.update()
        self._follow(event.globalPosition().toPoint())

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._on_trigger(event.globalPosition().toPoint()):
            self._sticky = True  # its button clicked: it stays open
            self._leave.stop()
            event.accept()
        elif not self.rect().contains(event.position().toPoint()):
            super().mousePressEvent(event)  # elsewhere: closes (and the press goes on there)
        else:
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        index = self._index_at(event.position())
        if index >= 0 and event.button() == Qt.MouseButton.LeftButton:
            self._choose(index)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key, count = event.key(), len(self.kinds)
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            step = 1 if key == Qt.Key.Key_Down else -1
            self._current = (self._current + step) % count if self._current >= 0 else (0 if step > 0 else count - 1)
            self._sticky = True  # in use from the keyboard
            self.update()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space) and self._current >= 0:
            self._choose(self._current)
        elif key == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def leaveEvent(self, _event) -> None:  # noqa: N802
        if self._current >= 0 and not self._sticky:
            self._current = -1
            self.update()

    def hideEvent(self, event) -> None:  # noqa: N802
        self._leave.stop()
        super().hideEvent(event)
        self.closed.emit()

    def _choose(self, index: int) -> None:
        kind = self.kinds[index][0]
        self.close()
        self.chosen.emit(kind)


class NewMenuButton(QObject):
    """Opens a :class:`NewMenu` from ``trigger``: on a click, from :meth:`open` (Ctrl+N)
    and, with ``hover``, when the pointer rests on it. ``place(size)``: where the menu's
    top-left goes, in global coordinates."""

    chosen = Signal(str)  # kind

    def __init__(self, trigger: QAbstractButton, place: Callable[[QSize], QPoint], *, hover: bool = False) -> None:
        super().__init__(trigger)
        self.trigger = trigger
        self.menu = NewMenu(trigger)
        self.menu.chosen.connect(self.chosen.emit)
        self.menu.closed.connect(self._menu_closed)
        self._place = place
        self._armed = True  # False while the pointer stays on the button after the menu closed
        self._wait = QTimer(self)
        self._wait.setSingleShot(True)
        self._wait.setInterval(OPEN_MS)
        self._wait.timeout.connect(self._rested)
        trigger.clicked.connect(lambda *_: self.open())
        if hover:
            trigger.installEventFilter(self)

    def open(self, *, keyboard: bool = False, sticky: bool = True) -> None:
        """Open the menu (``keyboard``: with its first kind current)."""
        self._wait.stop()
        if not self.menu.isVisible() and self.trigger.isVisible():
            self.menu.popup(self.trigger, self._place(self.menu.size()), keyboard=keyboard, sticky=sticky)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.trigger:
            if event.type() == QEvent.Type.Enter and self._armed and self.trigger.isEnabled():
                self._wait.start()
            elif event.type() == QEvent.Type.Leave:
                self._wait.stop()
                self._armed = True
        return False

    def _pointer_on_trigger(self) -> bool:
        return self.trigger.rect().contains(self.trigger.mapFromGlobal(QCursor.pos()))

    def _rested(self) -> None:
        if self._pointer_on_trigger():
            self.open(sticky=False)

    def _menu_closed(self) -> None:
        # closed with the pointer still on the button (Escape, a choice): it opens again
        # only once the pointer has left the button and come back
        self._armed = not self._pointer_on_trigger()
