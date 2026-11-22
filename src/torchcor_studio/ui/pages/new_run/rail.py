"""The Setup step's rail, beside the panels: an outline of the panels, then three cards:
the checks, the kind's demos (a list to start from) and the parameter sweep.

Beside the panels the rail sticks in view (``StickyRow``); its outline stays at the top and
only the cards below it scroll when they do not fit. On narrow pages the rail sits above
the panels: the outline becomes a row of chips that jump to the panels, the checks and the
demos follow, and the parameter sweep moves below the last panel, next to the Run button.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.core.simulations import Issue, Severity
from torchcor_studio.ui import icons
from torchcor_studio.ui.forms.form import OutlineEntry
from torchcor_studio.ui.pages.new_run.demo_picker import DemoPicker
from torchcor_studio.ui.pages.new_run.sweep_panel import SweepPanel
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import Badge, Card, IconLabel, discard, label, tone_text

ICON = {Severity.ERROR: ("x-circle", "danger"), Severity.WARNING: ("alert", "warning"),
        Severity.INFO: ("info", "info")}


#: How far a group's panels sit in from its heading in the outline.
INDENT = 14


class OutlineView(QWidget):
    """The panels in page order, painted (no widget per row): the current one highlighted,
    error and warning counts, an edited dot and a chevron for a collapsed panel; a group's
    heading, then its panels indented along a guide line. A click, or Enter on the focused
    row, asks to :attr:`jump` to that panel (or group). In ``flow`` mode (a narrow page) the
    panels are chips that wrap, with no current one and no group headings."""

    jump = Signal(str)  # panel key
    HEAD, GAP, ROW = 16, 6, 24
    CHIP, CHIP_GAP = 26, 6

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[OutlineEntry] = []
        self._current = ""
        self._hover = -1
        self._focus = 0
        self._keyboard = False
        self._flow = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_entries(self, entries: Sequence[OutlineEntry]) -> None:
        entries = list(entries)
        if entries != self._entries:
            resized = len(entries) != len(self._entries) or self._flow
            self._entries = entries
            if resized:
                self.updateGeometry()
            self.update()

    def entries(self) -> list[OutlineEntry]:
        return list(self._entries)

    def _rows(self) -> list[OutlineEntry]:
        """The entries shown: all of them, or in flow mode the panels alone."""
        return [e for e in self._entries if not (self._flow and e.group)]

    def set_current(self, key: str) -> None:
        if key != self._current:
            self._current = key
            self.update()

    def current(self) -> str:
        return self._current

    def set_flow(self, flow: bool) -> None:
        """Chips that wrap (a narrow page) or one row per panel."""
        if flow != self._flow:
            self._flow = flow
            policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred if flow
                                 else QSizePolicy.Policy.Fixed)
            policy.setHeightForWidth(flow)
            self.setSizePolicy(policy)
            self.updateGeometry()
            self.update()

    # ------------------------------------------------------------------ geometry
    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return self._flow

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        chips = self._chips(width)
        return int(chips[-1].bottom()) if chips else 0

    def sizeHint(self) -> QSize:  # noqa: N802
        if self._flow:
            width = self.width() or 600
            return QSize(width, self.heightForWidth(width))
        return QSize(240, self.HEAD + self.GAP + self.ROW * len(self._rows()))

    def _fonts(self) -> tuple[QFont, QFont, QFont, QFont]:
        eyebrow = QFont(self.font())
        eyebrow.setPixelSize(11)
        eyebrow.setWeight(QFont.Weight.DemiBold)
        eyebrow.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        row = QFont(self.font())
        row.setPixelSize(12 if self._flow else 13)
        current = QFont(row)
        current.setWeight(QFont.Weight.DemiBold)
        pill = QFont(self.font())
        pill.setPixelSize(11)
        pill.setWeight(QFont.Weight.DemiBold)
        return eyebrow, row, current, pill

    def _markers(self, entry: OutlineEntry, pill: QFont) -> float:
        """The width of a row's or chip's counts."""
        fm = QFontMetrics(pill)
        return sum(fm.horizontalAdvance(str(n)) + 12 + 4 for n in (entry.errors, entry.warnings) if n)

    def _chips(self, width: int) -> list[QRectF]:
        _eyebrow, row, _current, pill = self._fonts()
        fm = QFontMetrics(row)
        out, x, y = [], 0.0, 0.0
        for entry in self._rows():
            w = 20 + fm.horizontalAdvance(entry.title) + (10 if entry.edited else 0) + self._markers(entry, pill)
            if x and x + w > width:
                x, y = 0.0, y + self.CHIP + self.CHIP_GAP
            out.append(QRectF(x, y, w, self.CHIP))
            x += w + self.CHIP_GAP
        return out

    def _rects(self) -> list[QRectF]:
        if self._flow:
            return self._chips(self.width())
        return [QRectF(0, self.HEAD + self.GAP + i * self.ROW, self.width(), self.ROW)
                for i in range(len(self._rows()))]

    def _index_at(self, point: QPointF) -> int:
        return next((i for i, rect in enumerate(self._rects()) if rect.contains(point)), -1)

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        eyebrow, row_font, current_font, pill_font = self._fonts()
        if not self._flow:
            painter.setFont(eyebrow)
            painter.setPen(p.qcolor("text3"))
            painter.drawText(QRectF(8, 0, self.width() - 8, self.HEAD), Qt.AlignmentFlag.AlignVCenter, "OUTLINE")
        for i, (entry, rect) in enumerate(zip(self._rows(), self._rects(), strict=True)):
            current = entry.key == self._current and not self._flow
            radius = rect.height() / 2 if self._flow else 6
            indent = 0.0 if self._flow else INDENT * entry.depth
            if indent:  # the guide line of a group's panels, beside their rows
                painter.setPen(QPen(p.qcolor("line2"), 1))
                painter.drawLine(QPointF(rect.x() + 12.5, rect.top()), QPointF(rect.x() + 12.5, rect.bottom()))
                rect = rect.adjusted(indent, 0, 0, 0)
            if current or i == self._hover or self._flow:
                fill = QColor(p.accent) if current else QColor(p.bg3 if i == self._hover or not self._flow
                                                               else p.bg2)
                if current:
                    fill.setAlphaF(0.12)
                painter.setPen(QPen(p.qcolor("line"), 1) if self._flow else Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
            if self._keyboard and self.hasFocus() and i == self._focus:
                painter.setPen(QPen(p.qcolor("accent"), 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
            right = rect.right() - (6 if self._flow else 8)
            if entry.collapsed and not self._flow:
                painter.drawPixmap(int(right - 12), int(rect.center().y() - 6),
                                   icons.pixmap("chevron-right", p.text3, 12))
                right -= 18
            for count, tone in ((entry.warnings, "warning"), (entry.errors, "danger")):
                if count:
                    right = self._pill(painter, pill_font, right, rect, str(count), p.qcolor(tone),
                                       QColor(tone_text(p, tone))) - 4
            font = current_font if current else row_font
            painter.setFont(font)
            painter.setPen(p.qcolor("text" if current else "text2"))
            fm = QFontMetrics(font)
            left = rect.x() + (10 if self._flow else 8)
            room = right - left - (14 if entry.edited else 0)
            text = fm.elidedText(entry.title, Qt.TextElideMode.ElideRight, int(max(0.0, room)))
            painter.drawText(QRectF(left, rect.y(), rect.right() - left, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            if entry.edited:
                x = left + fm.horizontalAdvance(text) + 6
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(p.qcolor("accent"))
                painter.drawEllipse(QRectF(x, rect.center().y() - 3, 6, 6))

    @staticmethod
    def _pill(painter: QPainter, font: QFont, right: float, rect: QRectF, text: str, color: QColor,
              ink: QColor) -> float:
        width = QFontMetrics(font).horizontalAdvance(text) + 12
        pill = QRectF(right - width, rect.center().y() - 8, width, 16)
        fill = QColor(color)
        fill.setAlphaF(0.16)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(pill, 8, 8)
        painter.setFont(font)
        painter.setPen(ink)
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)
        return pill.x()

    # ------------------------------------------------------------------ input
    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        index = self._index_at(event.position())
        if index != self._hover:
            self._hover = index
            self.setCursor(Qt.CursorShape.PointingHandCursor if index >= 0 else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._hover = -1
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        index = self._index_at(event.position())
        if index >= 0 and event.button() == Qt.MouseButton.LeftButton:
            self._focus = index
            self.jump.emit(self._rows()[index].key)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key, rows = event.key(), self._rows()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right) and rows:
            step = -1 if key in (Qt.Key.Key_Up, Qt.Key.Key_Left) else 1
            self._focus = max(0, min(len(rows) - 1, self._focus + step))
            self.update()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space) and rows:
            self.jump.emit(rows[min(self._focus, len(rows) - 1)].key)
        else:
            super().keyPressEvent(event)

    def focusInEvent(self, event) -> None:  # noqa: N802
        self._keyboard = event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802
        self._keyboard = False
        super().focusOutEvent(event)
        self.update()

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            index = self._index_at(QPointF(event.pos()))
            if index >= 0:
                entry = self._rows()[index]
                QToolTip.showText(event.globalPos(), entry.title + (f": {entry.summary}" if entry.collapsed
                                                                    and entry.summary else ""), self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)


class _IssueRow(QWidget):
    """One check: its icon and its message, ending with a link to the panel it belongs to."""

    clicked = Signal()

    def __init__(self, issue: Issue, where: str) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        icon, tone = ICON.get(issue.severity, ICON[Severity.INFO])
        mark = IconLabel(icon, 14, tone=tone)
        text = html.escape(issue.message)
        if where:  # the link's colour is the palette's, so it follows the theme
            text += f' <a href="#" style="text-decoration: none">{html.escape(where)}&nbsp;›</a>'
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip(f"Show it in {where}")
        message = label(text, "check", wrap=True)  # the rail's layouts follow its height for width
        message.setTextFormat(Qt.TextFormat.RichText)
        message.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        message.linkActivated.connect(lambda *_: self.clicked.emit())
        # the icon sits on the first line of the message
        icon_col = QVBoxLayout()
        icon_col.setContentsMargins(0, max(0, (message.fontMetrics().height() - 14) // 2), 0, 0)
        icon_col.addWidget(mark)
        icon_col.addStretch(1)
        lay.addLayout(icon_col)
        lay.addWidget(message, 1, Qt.AlignmentFlag.AlignTop)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class ChecksCard(Card):
    """The validation issues, errors first, each one revealing its field when clicked."""

    reveal = Signal(str)  # parameter path

    def __init__(self) -> None:
        super().__init__("Checks", icon="check-circle", padding=14)
        self.status = Badge("Ready", "success", icon="check")
        self.add_action(self.status)
        self._rows = QVBoxLayout()
        self._rows.setSpacing(12)
        self.body.addLayout(self._rows)
        self._key: tuple | None = None
        self.set_issues([], lambda _path: None)

    def set_issues(self, issues: Sequence[Issue], where: Callable[[str], str | None]) -> None:
        """Show ``issues`` (``where(path)``: the title of the panel a path belongs to). The
        rows are rebuilt only when the issues change."""
        order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        issues = sorted(issues, key=lambda i: order.get(i.severity, 3))
        key = tuple((i.severity, i.message, i.field) for i in issues)
        if key == self._key:
            return
        self._key = key
        while self._rows.count():
            discard(self._rows.takeAt(0).widget())
        errors = sum(i.severity == Severity.ERROR for i in issues)
        warnings = sum(i.severity == Severity.WARNING for i in issues)
        if errors:
            self.status.set(f"{errors} error{'s' * (errors != 1)}", "danger")
            self.status.set_icon("x-circle")
        elif warnings:
            self.status.set(f"{warnings} warning{'s' * (warnings != 1)}", "warning")
            self.status.set_icon("alert")
        else:
            self.status.set("Ready", "success")
            self.status.set_icon("check")
        if not issues:
            self._rows.addWidget(label("Everything looks good.", "muted"))
            return
        for issue in issues:
            row = _IssueRow(issue, (where(issue.field) or "") if issue.field else "")
            if issue.field:
                row.clicked.connect(lambda path=issue.field: self.reveal.emit(path))
            self._rows.addWidget(row)


class SetupRail(QWidget):
    """The outline (kept at the top), then the Checks, demo and sweep cards, which scroll
    on their own only when they do not fit. ``tail``: where the sweep goes on a narrow page
    (below the panels)."""

    jump = Signal(str)  # panel key
    reveal = Signal(str)  # parameter path

    def __init__(self, demos: DemoPicker, sweep: SweepPanel, tail: QWidget) -> None:
        super().__init__()
        self.outline = OutlineView()
        self.outline.jump.connect(self.jump.emit)
        self.checks = ChecksCard()
        self.checks.reveal.connect(self.reveal.emit)
        self.demos = demos
        self.sweep = sweep
        self._tail = tail
        QVBoxLayout(tail).setContentsMargins(0, 0, 0, 0)
        tail.hide()
        self.cards = QWidget()
        self.cards.setObjectName("Transparent")
        # as narrow as the scroll area's viewport, so a scroll bar never covers the cards
        self.cards.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._cards = QVBoxLayout(self.cards)
        self._cards.setContentsMargins(0, 0, 0, 0)
        self._cards.setSpacing(12)
        # no alignment and no stretch: each card gets the height its width needs (the rail is
        # as tall as its cards, so there is never room left over)
        for card in (self.checks, demos, sweep):
            self._cards.addWidget(card)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("Transparent")
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.cards)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        lay.addWidget(self.outline)
        lay.addWidget(self.scroll, 1)
        self._beside = True
        self._outlined = False
        self.cards.installEventFilter(self)

    def set_outline(self, entries: Sequence[OutlineEntry]) -> None:
        """The panels of the kind; the outline shows for three or more."""
        self.outline.set_entries(entries)
        self._outlined = len(entries) >= 3
        self.outline.setVisible(self._outlined)

    def _cards_height(self, width: int) -> int:
        cards = self._cards
        return cards.totalHeightForWidth(width) if cards.hasHeightForWidth() else cards.totalSizeHint().height()

    def natural_height(self, width: int) -> int:
        """The rail's height with nothing scrolled away."""
        height = self._cards_height(width)
        if not self.outline.isHidden():
            height += self.outline.sizeHint().height() + self.layout().spacing()
        return height

    def set_mode(self, beside: bool) -> None:
        """Beside the panels (sticky, an outline list, scrolling cards) or above them (outline
        chips, the checks and the demos; the sweep goes below the panels)."""
        if beside == self._beside:
            return
        self._beside = beside
        self.outline.set_flow(not beside)
        if beside:
            self._tail.layout().removeWidget(self.sweep)
            self._cards.addWidget(self.sweep)
            self._tail.hide()
        else:
            self._cards.removeWidget(self.sweep)
            self._tail.layout().addWidget(self.sweep)
            self._tail.show()
        self._fit()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.cards and event.type() == QEvent.Type.LayoutRequest and not self._beside:
            self._fit()
        return False

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._beside:
            self._fit()

    def _fit(self) -> None:
        if self._beside:
            self.scroll.setMinimumHeight(0)
            self.scroll.setMaximumHeight(16_777_215)
            return
        height = self._cards_height(max(1, self.width()))
        if self.scroll.height() != height or self.scroll.minimumHeight() != height:
            self.scroll.setFixedHeight(height)
