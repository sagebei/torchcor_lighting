"""Transient notifications shown in the bottom-right corner of the main window."""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QVBoxLayout, QWidget

from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets.basic import IconButton, IconLabel, label, tone_color

_ICON = {"success": "check-circle", "danger": "x-circle", "warning": "alert", "info": "info", "accent": "info"}


class Toast(QWidget):
    def __init__(self, parent: QWidget, title: str, message: str = "", tone: str = "info",
                 timeout_ms: int = 4500):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setFixedWidth(360)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 12, 8, 12)
        lay.setSpacing(10)
        self._tone = tone
        lay.addWidget(IconLabel(_ICON.get(tone, "info"), 18, tone=tone), 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(2)
        t = label(title)
        t.setStyleSheet("font-weight: 600;")
        col.addWidget(t)
        if message:
            col.addWidget(label(message, "muted", wrap=True))
        lay.addLayout(col, 1)
        lay.addWidget(IconButton("x", "Dismiss", size=24, icon_size=14, on_click=self.dismiss), 0,
                      Qt.AlignmentFlag.AlignTop)
        self._fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fx)
        self._fade = QPropertyAnimation(self._fx, b"opacity", self)
        self._fade.setDuration(180)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        if timeout_ms > 0:
            QTimer.singleShot(timeout_ms, self.dismiss)

    def show_animated(self) -> None:
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self.show()
        self.raise_()
        self._fade.start()

    def dismiss(self) -> None:
        if not self.isVisible():
            return
        self._fade.stop()
        self._fade.setStartValue(self._fx.opacity())
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self._finish)
        self._fade.start()

    def _finish(self) -> None:
        self.hide()
        host = self.parentWidget()
        self.deleteLater()
        if isinstance(host, QWidget):
            QTimer.singleShot(0, lambda: ToastHost.relayout(host))

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QColor(p.line2))
        painter.setBrush(QColor(p.bg3 if p.dark else p.bg2))
        painter.drawRoundedRect(r, 10, 10)
        accent = QColor(tone_color(p, self._tone))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawRoundedRect(QRectF(r.x(), r.y() + 10, 3, r.height() - 20), 1.5, 1.5)


class ToastHost:
    """Stacks toasts in a window's bottom-right corner."""

    MARGIN = 18

    @staticmethod
    def show(window: QWidget, title: str, message: str = "", tone: str = "info",
             timeout_ms: int = 4500) -> Toast:
        toast = Toast(window, title, message, tone, timeout_ms)
        toast.adjustSize()
        toast.show_animated()
        ToastHost.relayout(window)
        return toast

    @staticmethod
    def relayout(window: QWidget) -> None:
        toasts = [w for w in window.findChildren(Toast) if w.isVisible()]
        y = window.height() - ToastHost.MARGIN
        for toast in reversed(toasts):
            toast.adjustSize()
            y -= toast.height()
            toast.move(QPoint(window.width() - toast.width() - ToastHost.MARGIN, y))
            toast.raise_()
            y -= 10
