"""A panel of the Setup step: a card whose painted header opens and closes it.

The header shows the panel's icon and title, a dot when it was edited, its one-line
summary while collapsed, its issue count, an Advanced button and a chevron. A collapsed
panel is built on its first opening.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QToolTip, QVBoxLayout, QWidget

from torchcor_studio.core.simulations.panels import ResolvedPanel
from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import Button, chain_focus, tone_text

HEADER_H = 52


class _Header(QWidget):
    """Painted header; only the Advanced button is a widget."""

    def __init__(self, card: PanelCard) -> None:
        super().__init__(card)
        self.card = card
        self.setFixedHeight(HEADER_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAccessibleName(card.resolved.title)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16 + 16 + 8, 0)  # room for the chevron
        lay.setSpacing(8)
        lay.addStretch(1)
        self.layout_ = lay
        self._keyboard = False  # focused with Tab: draw the focus ring

    def focusInEvent(self, event) -> None:  # noqa: N802
        self._keyboard = event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802
        self._keyboard = False
        super().focusOutEvent(event)
        self.update()

    def _fonts(self) -> tuple[QFont, QFont, QFont]:
        title = QFont(self.font())
        title.setPixelSize(14)
        title.setWeight(QFont.Weight.DemiBold)
        small = QFont(self.font())
        small.setPixelSize(12)
        pill = QFont(self.font())
        pill.setPixelSize(11)
        pill.setWeight(QFont.Weight.DemiBold)
        return title, small, pill

    def _right(self) -> float:
        """Where the painted part ends: the Advanced button, else the chevron."""
        button = self.card.advanced_button
        if button is not None and not button.isHidden():
            return button.x() - 8
        return self.width() - 16 - 16 - 8

    def _geometry(self) -> dict[str, QRectF]:
        card = self.card
        title_font, _small, pill_font = self._fonts()
        mid = self.height() / 2
        rects = {"icon": QRectF(16, mid - 8, 16, 16)}
        tw = QFontMetrics(title_font).horizontalAdvance(card.resolved.title)
        rects["title"] = QRectF(42, 0, tw, self.height())
        x = 42 + tw + 8
        if card.edited:
            rects["dot"] = QRectF(x, mid - 3, 6, 6)
            x += 14
        right = self._right()
        count, _tone = card.issue_badge()
        if count:
            width = 14 + 4 + QFontMetrics(pill_font).horizontalAdvance(str(count)) + 16
            rects["pill"] = QRectF(right - width, mid - 10, width, 20)
            right -= width + 8
        if not card.is_expanded() and card.summary:
            rects["summary"] = QRectF(x + 4, 0, max(0.0, right - x - 4), self.height())
        rects["chevron"] = QRectF(self.width() - 16 - 16, mid - 8, 16, 16)
        return rects

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        card = self.card
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        title_font, small, pill_font = self._fonts()
        rects = self._geometry()
        painter.drawPixmap(rects["icon"].topLeft().toPoint(), icons.pixmap(card.resolved.icon, p.text2, 16))
        painter.setFont(title_font)
        painter.setPen(p.qcolor("text"))
        painter.drawText(rects["title"], Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         card.resolved.title)
        if "dot" in rects:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(p.qcolor("accent"))
            painter.drawEllipse(rects["dot"])
        if "summary" in rects:
            painter.setFont(small)
            painter.setPen(p.qcolor("text3"))
            text = QFontMetrics(small).elidedText(card.summary, Qt.TextElideMode.ElideRight,
                                                  int(rects["summary"].width()))
            painter.drawText(rects["summary"], Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             text)
        if "pill" in rects:
            count, tone = card.issue_badge()
            color = QColor(tone_text(p, tone))
            fill = p.qcolor(tone)
            fill.setAlphaF(0.14)
            pill = rects["pill"]
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(pill, 10, 10)
            painter.drawPixmap(int(pill.x() + 7), int(pill.center().y() - 6),
                               icons.pixmap("x-circle" if tone == "danger" else "alert", color.name(), 12))
            painter.setFont(pill_font)
            painter.setPen(color)
            painter.drawText(pill.adjusted(23, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             str(count))
        chevron = "chevron-down" if card.is_expanded() else "chevron-right"
        painter.drawPixmap(rects["chevron"].topLeft().toPoint(), icons.pixmap(chevron, p.text3, 16))
        if self._keyboard and self.hasFocus():
            painter.setPen(QPen(p.qcolor("accent"), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(2, 2, -2, -2), 10, 10)

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            rects = self._geometry()
            point = QPointF(event.pos())
            card = self.card
            if "dot" in rects and rects["dot"].adjusted(-4, -4, 4, 4).contains(point):
                tip = card.edited_tip
            elif "pill" in rects and rects["pill"].contains(point):
                tip = card.issue_tip()
            elif not card.is_expanded() and card.summary:
                tip = f"{card.resolved.title}: {card.summary}"
            else:
                tip = ""
            if tip:
                QToolTip.showText(event.globalPos(), tip, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.card.toggle()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.card.toggle()
        else:
            super().keyPressEvent(event)


class PanelCard(QFrame):
    """One resolved panel: a header, and a body that ``build(card)`` makes on first need."""

    toggled = Signal(bool)  # expanded
    advanced_toggled = Signal(bool)

    def __init__(self, resolved: ResolvedPanel, build: Callable[[PanelCard], QWidget], *,
                 expanded: bool) -> None:
        super().__init__()
        self.resolved = resolved
        self.key = resolved.key
        self._build = build
        self._expanded = expanded
        self._body: QWidget | None = None
        self.summary = ""
        self.edited = False
        self.edited_tip = ""
        self.errors = self.warnings = 0
        self.advanced_count = 0
        self.show_advanced = False
        self.advanced_button: Button | None = None
        self.setProperty("card", "true")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self.header = _Header(self)
        self._lay.addWidget(self.header)
        if expanded:
            self.body()

    # ------------------------------------------------------------------ open / close
    def is_expanded(self) -> bool:
        return self._expanded

    def is_built(self) -> bool:
        return self._body is not None

    def body(self) -> QWidget:
        """The panel's body, built now if it was not yet (it stays hidden while collapsed)."""
        if self._body is None:
            self._body = self._build(self)
            # parented before it is shown: a widget shown without a parent is a window of its
            # own, which flashes on screen (and moves the cursor's focus) for a moment
            self._lay.addWidget(self._body)
            self._body.setVisible(self._expanded)
            chain_focus(self.header, self._body)  # Tab goes on into the panel, built late or not
        return self._body

    def expand(self) -> None:
        if not self._expanded:
            self._expanded = True
            self.body().show()
            self._sync_button()
            self.header.update()
            self.toggled.emit(True)

    def collapse(self) -> None:
        if self._expanded:
            self._expanded = False
            if self._body is not None:
                self._body.hide()
            self._sync_button()
            self.header.update()
            self.toggled.emit(False)

    def toggle(self) -> None:
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    # ------------------------------------------------------------------ header state
    def set_summary(self, text: str) -> None:
        if text != self.summary:
            self.summary = text
            self.header.update()

    def set_issues(self, errors: int, warnings: int) -> None:
        if (errors, warnings) != (self.errors, self.warnings):
            self.errors, self.warnings = errors, warnings
            self.header.update()

    def issue_badge(self) -> tuple[int, str]:
        """The count shown on the header and its tone (errors before warnings)."""
        if self.errors:
            return self.errors, "danger"
        return (self.warnings, "warning") if self.warnings else (0, "")

    def issue_tip(self) -> str:
        parts = [f"{n} {word}{'s' * (n != 1)}" for n, word in ((self.errors, "error"),
                                                              (self.warnings, "warning")) if n]
        return " and ".join(parts)

    def set_edited(self, edited: bool, tip: str = "") -> None:
        if (edited, tip) != (self.edited, self.edited_tip):
            self.edited, self.edited_tip = edited, tip
            self.header.update()

    def set_advanced_count(self, count: int) -> None:
        """How many advanced parameters the panel has right now (the button hides at 0)."""
        if count != self.advanced_count or (self.advanced_button is None and count):
            self.advanced_count = count
            if count and self.advanced_button is None:
                self.advanced_button = Button("Advanced", "sliders", "ghost", size="sm")
                self.advanced_button.setCheckable(True)
                self.advanced_button.setChecked(self.show_advanced)
                self.advanced_button.setToolTip("Show this panel's advanced parameters")
                self.advanced_button.toggled.connect(self._advanced_clicked)
                self.header.layout_.addWidget(self.advanced_button)
                QWidget.setTabOrder(self.header, self.advanced_button)  # made late: Tab reaches it next
            if self.advanced_button is not None:
                self.advanced_button.setText(f"Advanced {count}")
            self._sync_button()

    def _sync_button(self) -> None:
        if self.advanced_button is not None:
            self.advanced_button.setVisible(self._expanded and self.advanced_count > 0)

    def set_show_advanced(self, on: bool) -> None:
        """Show (or hide) the advanced parameters without emitting ``advanced_toggled``."""
        self.show_advanced = on
        if self.advanced_button is not None and self.advanced_button.isChecked() != on:
            self.advanced_button.blockSignals(True)
            self.advanced_button.setChecked(on)
            self.advanced_button.blockSignals(False)

    def _advanced_clicked(self, on: bool) -> None:
        self.show_advanced = on
        self.advanced_toggled.emit(on)
