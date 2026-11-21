"""Small building blocks: labels, buttons, cards, badges, status dots, empty states."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import Palette, theme

# Tones understood by Badge / StatusDot / MeterBar
TONES = ("neutral", "accent", "success", "warning", "danger", "info", "ember")


def tone_color(p: Palette, tone: str) -> str:
    return {
        "accent": p.accent,
        "success": p.success,
        "warning": p.warning,
        "danger": p.danger,
        "info": p.info,
        "ember": p.ember,
    }.get(tone, p.text3)


def tone_text(p: Palette, tone: str) -> str:
    """A tone as small text (or a small icon) on its own soft tint: legible in both themes."""
    return {
        "accent": p.accent_text,
        "success": p.success_text,
        "warning": p.warning_text,
        "danger": p.danger_text,
        "info": p.info_text,
        "ember": p.ember_text,
    }.get(tone, p.text2)


def discard(widget: QWidget | None) -> None:
    """Remove a widget now: hide it immediately (so it never paints again) and delete it later."""
    if widget is not None:
        widget.hide()
        widget.deleteLater()


# ------------------------------------------------------------------------- labels
def label(text: str = "", role: str | None = None, *, wrap: bool = False,
          selectable: bool = False, parent: QWidget | None = None) -> QLabel:
    lbl = QLabel(text, parent)
    if role:
        lbl.setProperty("role", role)
    lbl.setWordWrap(wrap)
    if selectable:
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lbl


_NBSP = re.compile(r"(?<=\d) | (?=\()|(?<=×) (?=\d)")


def keep_together(text: str) -> str:
    """Non-breaking spaces after numbers, before brackets and inside "2 × 3", so a line
    never ends on a lone number or starts with a lone unit ("4,305 nodes", "(CPU)")."""
    return _NBSP.sub("\u00a0", text)


def elide_lines(text: str, font: QFont, width: int, lines: int) -> str:
    """``text`` wrapped at ``width`` px into at most ``lines`` lines (joined with newlines),
    the last one ending in "…" when the text does not fit. Only plain spaces break."""
    fm = QFontMetrics(font)
    words = [w for w in text.split(" ") if w]
    out: list[str] = []
    current = ""
    for i, word in enumerate(words):
        trial = f"{current} {word}" if current else word
        if not current or fm.horizontalAdvance(trial) <= width:
            current = trial
            continue
        out.append(current)
        if len(out) == lines - 1:
            rest = " ".join(words[i:])
            return "\n".join([*out, fm.elidedText(rest, Qt.TextElideMode.ElideRight, width)])
        current = word
    if current:
        out.append(current if len(out) < lines - 1 or fm.horizontalAdvance(current) <= width
                   else fm.elidedText(current, Qt.TextElideMode.ElideRight, width))
    return "\n".join(out)


class WrapLabel(QLabel):
    """A word-wrapped label that always reserves its full wrapped height.

    Plain wrapped labels inside nested layouts can be laid out one line short, because
    Qt does not propagate height-for-width through every layout; this one fixes its
    minimum height to the wrapped height of its current width.
    """

    def __init__(self, text: str = "", role: str | None = None, parent: QWidget | None = None):
        super().__init__(text, parent)
        if role:
            self.setProperty("role", role)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        super().setText(text)
        self._fit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if self.width() > 0:
            height = self.heightForWidth(self.width())
            if height > 0 and height != self.minimumHeight():
                self.setMinimumHeight(height)


class ElidedLabel(QLabel):
    """A single-line label that elides with "…" instead of growing."""

    def __init__(self, text: str = "", role: str | None = None,
                 mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight, parent: QWidget | None = None):
        super().__init__(parent)
        self._full = text
        self._mode = mode
        if role:
            self.setProperty("role", role)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(10)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._full = text or ""
        self.setToolTip(self._full if len(self._full) > 24 else "")
        self._elide()

    def fullText(self) -> str:  # noqa: N802
        return self._full

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        fm = QFontMetrics(self.font())
        super().setText(fm.elidedText(self._full, self._mode, max(10, self.width())))


# ------------------------------------------------------------------------- buttons
class Button(QPushButton):
    """Push button with a theme-tinted icon. ``variant``: primary, secondary, ghost, danger, subtle."""

    def __init__(self, text: str = "", icon: str | None = None, variant: str = "secondary",
                 size: str | None = None, parent: QWidget | None = None,
                 on_click: Callable[[], None] | None = None):
        super().__init__(text, parent)
        self._icon_name = icon
        self.setProperty("variant", variant)
        if size:
            self.setProperty("size", size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIconSize(QSize(15, 15) if size != "lg" else QSize(17, 17))
        theme().changed.connect(self._retint)
        self._retint(theme().palette)
        if on_click:
            self.clicked.connect(lambda *_: on_click())

    def set_icon_name(self, name: str | None) -> None:
        self._icon_name = name
        self._retint(theme().palette)

    def set_variant(self, variant: str) -> None:
        self.setProperty("variant", variant)
        from torchcor_studio.ui.theme import repolish

        repolish(self)
        self._retint(theme().palette)

    def _retint(self, p: Palette) -> None:
        if not self._icon_name:
            self.setIcon(QIcon())
            return
        variant = self.property("variant")
        color = {
            "primary": p.on_accent,
            "danger": p.danger,
            "ghost": p.text2,
            "subtle": p.accent_text,
            "link": p.accent_text,
        }.get(variant, p.text)
        self.setIcon(icons.icon(self._icon_name, color, 16, disabled=p.text3))


class IconButton(QToolButton):
    """Square, icon-only ghost button with a tooltip."""

    def __init__(self, icon: str, tooltip: str = "", size: int = 30, icon_size: int = 16,
                 checkable: bool = False, parent: QWidget | None = None,
                 on_click: Callable[[], None] | None = None, tone: str | None = None):
        super().__init__(parent)
        self._icon_name = icon
        self._tone = tone
        self.setProperty("variant", "ghost")
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip or icon)
        self.setCheckable(checkable)
        self.setFixedSize(size, size)
        self.setIconSize(QSize(icon_size, icon_size))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        theme().changed.connect(self._retint)
        self._retint(theme().palette)
        if on_click:
            self.clicked.connect(lambda *_: on_click())

    def set_icon_name(self, name: str) -> None:
        self._icon_name = name
        self._retint(theme().palette)

    def _retint(self, p: Palette) -> None:
        base = tone_color(p, self._tone) if self._tone else p.text2
        self.setIcon(icons.icon(self._icon_name, base, self.iconSize().width(), disabled=p.text3,
                                active=p.accent_text if self.isCheckable() else p.text))


# ------------------------------------------------------------------------- containers
class Card(QFrame):
    """A rounded surface. Optional header with title, subtitle and trailing actions."""

    def __init__(self, title: str = "", subtitle: str = "", *, icon: str | None = None,
                 padding: int = 16, spacing: int = 12, variant: str = "true",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("card", variant)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(padding, padding, padding, padding)
        self._outer.setSpacing(spacing)
        self.header: QHBoxLayout | None = None
        self.title_label: QLabel | None = None
        self.subtitle_label: QLabel | None = None
        if title or subtitle:
            self.header = QHBoxLayout()
            self.header.setSpacing(10)
            if icon:
                ic = IconLabel(icon, size=16, tone="muted")
                self.header.addWidget(ic, 0, Qt.AlignmentFlag.AlignTop)
            col = QVBoxLayout()
            col.setSpacing(2)
            self.title_label = label(title, "h3")
            col.addWidget(self.title_label)
            if subtitle:
                self.subtitle_label = label(subtitle, "caption", wrap=True)
                col.addWidget(self.subtitle_label)
            self.header.addLayout(col, 1)
            self._outer.addLayout(self.header)
        self.body = QVBoxLayout()
        self.body.setSpacing(spacing)
        self.body.setContentsMargins(0, 0, 0, 0)
        self._outer.addLayout(self.body, 1)

    def add_action(self, widget: QWidget) -> None:
        if self.header is None:
            self.header = QHBoxLayout()
            self.header.addStretch(1)
            self._outer.insertLayout(0, self.header)
        self.header.addWidget(widget, 0, Qt.AlignmentFlag.AlignTop)

    def addWidget(self, widget: QWidget, stretch: int = 0) -> None:  # noqa: N802
        self.body.addWidget(widget, stretch)

    def addLayout(self, layout, stretch: int = 0) -> None:  # noqa: N802
        self.body.addLayout(layout, stretch)


def divider(vertical: bool = False) -> QFrame:
    line = QFrame()
    line.setProperty("vdivider" if vertical else "divider", True)
    return line


def hairline() -> QFrame:
    """A 1 px rule (under table headers, between the rail's blocks)."""
    line = QFrame()
    line.setProperty("hairline", True)
    return line


def chain_focus(anchor: QWidget, container: QWidget) -> QWidget:
    """Put the widgets of ``container`` that Tab reaches right after ``anchor`` in the focus
    chain, in the order they were laid out, and return the last of them (``anchor`` when
    none). Widgets built after their neighbours (a lazily built panel) otherwise come last."""
    window = anchor.window()
    for w in container.findChildren(QWidget):
        parent = w.parentWidget()
        if (w.focusPolicy() & Qt.FocusPolicy.TabFocus and w.window() is window  # not a popup's
                and not (parent is not None and parent.focusProxy() is w)):
            QWidget.setTabOrder(anchor, w)
            anchor = w
    return anchor


def hbox(*widgets, spacing: int = 8, margins: tuple[int, int, int, int] = (0, 0, 0, 0),
         stretch_at: int | None = None) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setSpacing(spacing)
    lay.setContentsMargins(*margins)
    for i, w in enumerate(widgets):
        if stretch_at == i:
            lay.addStretch(1)
        if w is None:
            lay.addStretch(1)
        elif isinstance(w, QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    return lay


def vbox(*widgets, spacing: int = 8, margins: tuple[int, int, int, int] = (0, 0, 0, 0)) -> QVBoxLayout:
    lay = QVBoxLayout()
    lay.setSpacing(spacing)
    lay.setContentsMargins(*margins)
    for w in widgets:
        if w is None:
            lay.addStretch(1)
        elif isinstance(w, QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    return lay


# ------------------------------------------------------------------------- painted
class IconLabel(QWidget):
    """A theme-aware icon glyph (no button behaviour)."""

    def __init__(self, icon: str, size: int = 16, tone: str = "muted", color: str | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._icon, self._size, self._tone, self._color = icon, size, tone, color
        self.setFixedSize(size, size)

    def set_icon(self, icon: str, tone: str | None = None, color: str | None = None) -> None:
        self._icon = icon
        if tone:
            self._tone = tone
        self._color = color
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        color = self._color or {"muted": p.text2, "faint": p.text3, "text": p.text}.get(
            self._tone, tone_color(p, self._tone))
        painter = QPainter(self)
        painter.drawPixmap(0, 0, icons.pixmap(self._icon, color, self._size))


class Badge(QWidget):
    """Compact pill: ``Badge("Running", tone="accent")``. Optional leading dot or icon;
    ``outline``: a quiet outlined pill instead of a tinted one."""

    def __init__(self, text: str = "", tone: str = "neutral", *, dot: bool = False,
                 icon: str | None = None, mono: bool = False, outline: bool = False,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._text, self._tone, self._dot, self._icon, self._mono = text, tone, dot, icon, mono
        self._outline = outline
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set(self, text: str | None = None, tone: str | None = None, *,
            outline: bool | None = None) -> None:
        if text is not None:
            self._text = text
        if tone is not None:
            self._tone = tone
        if outline is not None:
            self._outline = outline
        self.updateGeometry()
        self.update()

    def set_icon(self, icon: str | None, dot: bool = False) -> None:
        """A leading icon, or a dot, or neither."""
        if (icon, dot) != (self._icon, self._dot):
            self._icon, self._dot = icon, dot
            self.updateGeometry()
            self.update()

    def text(self) -> str:
        return self._text

    def _font(self):
        f = self.font()
        f.setPixelSize(11)
        f.setWeight(f.Weight.DemiBold)
        return f

    def sizeHint(self) -> QSize:  # noqa: N802
        fm = QFontMetrics(self._font())
        extra = 14 if (self._dot or self._icon) else 0
        return QSize(fm.horizontalAdvance(self._text) + 16 + extra, 20)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        neutral = self._tone == "neutral"
        color = QColor(p.text2 if neutral else tone_color(p, self._tone))
        text = QColor(p.text2 if neutral else tone_text(p, self._tone))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._outline:
            painter.setPen(QColor(p.line2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        else:
            fill = QColor(color)
            fill.setAlphaF(0.14 if not neutral else (0.10 if p.dark else 0.08))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
        painter.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        x = 8
        if self._dot:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QRectF(x, r.center().y() - 3, 6, 6))
            x += 12
        elif self._icon:
            painter.drawPixmap(int(x - 1), int(r.center().y() - 6), icons.pixmap(self._icon, text.name(), 12))
            x += 14
        painter.setPen(text)
        painter.setFont(self._font())
        painter.drawText(QRectF(x, 0, r.width() - x, r.height()), Qt.AlignmentFlag.AlignVCenter, self._text)


class StatusDot(QWidget):
    """A coloured dot; ``pulse=True`` adds a breathing halo (used for live runs)."""

    def __init__(self, tone: str = "neutral", size: int = 8, pulse: bool = False,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._tone, self._size = tone, size
        self._phase = 0.0
        self.setFixedSize(size + 8, size + 8)
        self._anim = QPropertyAnimation(self, b"phase", self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(1600)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.set(tone, pulse)

    def set(self, tone: str, pulse: bool = False) -> None:
        self._tone = tone
        if pulse and self._anim.state() != QPropertyAnimation.State.Running:
            self._anim.start()
        elif not pulse:
            self._anim.stop()
            self._phase = 0.0
        self.update()

    def _get_phase(self) -> float:
        return self._phase

    def _set_phase(self, v: float) -> None:
        self._phase = v
        self.update()

    phase = Property(float, _get_phase, _set_phase)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        color = QColor(p.state_color(self._tone) if self._tone in
                       ("running", "queued", "completed", "failed", "cancelled", "lost")
                       else tone_color(p, self._tone))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QRectF(self.rect()).center()
        if self._anim.state() == QPropertyAnimation.State.Running:
            halo = QColor(color)
            halo.setAlphaF(0.45 * (1 - self._phase))
            rad = self._size / 2 + 4 * self._phase
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(c, rad, rad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(c, self._size / 2, self._size / 2)


# ------------------------------------------------------------------------- composites
class EmptyState(QWidget):
    """Centered icon + title + message + optional action — for empty lists and pages."""

    def __init__(self, icon: str, title: str, message: str = "", action: QWidget | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(10)
        badge = _IconDisc(icon)
        lay.addWidget(badge, 0, Qt.AlignmentFlag.AlignHCenter)
        self.title = label(title, "h3")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.title)
        self.message = label(message, "muted", wrap=True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setMaximumWidth(420)
        lay.addWidget(self.message, 0, Qt.AlignmentFlag.AlignHCenter)
        if action is not None:
            lay.addSpacing(6)
            lay.addWidget(action, 0, Qt.AlignmentFlag.AlignHCenter)


class _IconDisc(QWidget):
    def __init__(self, icon: str, size: int = 48):
        super().__init__()
        self._icon = icon
        self.setFixedSize(size, size)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(p.line2))
        painter.setBrush(QColor(p.bg3))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
        s = 22
        painter.drawPixmap((self.width() - s) // 2, (self.height() - s) // 2, icons.pixmap(self._icon, p.text2, s))


class StatTile(QFrame):
    """Metric tile: small label, large value with unit, optional hint underneath."""

    def __init__(self, label_text: str, value: str = "—", unit: str = "", hint: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("card", "true")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(4)
        self.label = label(label_text.upper(), "eyebrow")
        row = QHBoxLayout()
        row.setSpacing(4)
        self.value = label(value, "value")
        self.unit = label(unit, "faint")
        row.addWidget(self.value)
        row.addWidget(self.unit, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        self.hint = label(hint, "caption", wrap=True)
        self.hint.setVisible(bool(hint))
        lay.addWidget(self.label)
        lay.addLayout(row)
        lay.addWidget(self.hint)

    def set(self, value: str, unit: str | None = None, hint: str | None = None) -> None:
        self.value.setText(value)
        if unit is not None:
            self.unit.setText(unit)
        if hint is not None:
            self.hint.setText(hint)
            self.hint.setVisible(bool(hint))


class KeyValueGrid(QWidget):
    """Two-column "label — value" list with muted keys."""

    def __init__(self, rows: Iterable[tuple[str, str]] = (), parent: QWidget | None = None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(16)
        self._grid.setVerticalSpacing(8)
        self._grid.setColumnStretch(1, 1)
        self.set_rows(rows)

    def set_rows(self, rows: Iterable[tuple[str, str]]) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                discard(item.widget())
        for r, (k, v) in enumerate(rows):
            key = label(k, "faint")
            val = label(v, selectable=True)
            val.setWordWrap(True)
            self._grid.addWidget(key, r, 0, Qt.AlignmentFlag.AlignTop)
            self._grid.addWidget(val, r, 1, Qt.AlignmentFlag.AlignTop)


class SectionHeader(QWidget):
    """Heading row for page sections: title (+ optional caption) on the left, actions right."""

    def __init__(self, title: str, caption: str = "", role: str = "h3", parent: QWidget | None = None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        col = QVBoxLayout()
        col.setSpacing(2)
        self.title = label(title, role)
        col.addWidget(self.title)
        self.caption = label(caption, "caption", wrap=True)
        self.caption.setVisible(bool(caption))
        col.addWidget(self.caption)
        lay.addLayout(col, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(6)
        lay.addLayout(self.actions)

    def add_action(self, widget: QWidget) -> None:
        self.actions.addWidget(widget)


class SearchField(QLineEdit):
    """Line edit with a leading magnifier glyph and a clear button."""

    def __init__(self, placeholder: str = "Search", parent: QWidget | None = None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setProperty("search", True)
        self.setClearButtonEnabled(True)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        p = theme().palette
        painter = QPainter(self)
        painter.drawPixmap(10, (self.height() - 14) // 2, icons.pixmap("search", p.text3, 14))


class Spinner(QWidget):
    """Indeterminate busy indicator (rotating arc)."""

    def __init__(self, size: int = 16, parent: QWidget | None = None):
        super().__init__(parent)
        self._angle = 0
        self.setFixedSize(size, size)
        self._anim = QPropertyAnimation(self, b"angle", self)
        self._anim.setStartValue(0)
        self._anim.setEndValue(360)
        self._anim.setDuration(900)
        self._anim.setLoopCount(-1)  # runs only while shown (see showEvent / hideEvent)

    def _get(self) -> int:
        return self._angle

    def _set(self, v: int) -> None:
        self._angle = v
        self.update()

    angle = Property(int, _get, _set)

    def showEvent(self, e) -> None:  # noqa: N802
        self._anim.start()
        super().showEvent(e)

    def hideEvent(self, e) -> None:  # noqa: N802
        self._anim.stop()
        super().hideEvent(e)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        pen = painter.pen()
        pen.setWidthF(2)
        pen.setColor(QColor(p.line2))
        painter.setPen(pen)
        painter.drawEllipse(r)
        pen.setColor(QColor(p.accent))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(r, int(-self._angle * 16), 100 * 16)


def rounded_path(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path
