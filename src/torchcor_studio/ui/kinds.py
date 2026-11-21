"""The kinds of simulation as the UI shows them: each one's icon (its spec's) on a tile of
its tone, its abbreviation and the names its new simulations get (EP, EP_1, EP_2, …)."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from torchcor_studio.core.simulations import get_spec
from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import tone_color, tone_text

#: A kind's tone: the colour of its icon wherever the kind is named.
TONE = {"ep": "ember", "ecg": "danger", "mechanics": "info", "em": "accent", "script": "neutral"}
#: The kind of a simulation started without choosing one (at start-up, or when the last
#: tab closes).
DEFAULT = "ep"


def icon(kind: str) -> str:
    try:
        return get_spec(kind).icon
    except KeyError:
        return "zap"


def tone(kind: str) -> str:
    return TONE.get(kind, "neutral")


def badge(kind: str) -> str:
    """The kind's abbreviation: "EP", "ECG", "MECH", "EM", "PY"."""
    try:
        return get_spec(kind).badge or kind.upper()
    except KeyError:
        return kind.upper()


def stage_label(kind: str, stage: str, fallback: str = "") -> str:
    """A stage's short name, as the Run page's tracker shows it ("Inflating"), else
    ``fallback``: the label the run's script gave it (a custom script's own stages)."""
    try:
        return next(s.label for s in get_spec(kind).stages if s.id == stage)
    except (KeyError, StopIteration):
        return fallback


def next_name(kind: str, taken: Iterable[str]) -> str:
    """The first free name for a new simulation of ``kind``: its abbreviation ("EP"), then
    "EP_1", "EP_2", … A name is taken by a simulation of that name, whatever its case, and
    by the runs of a sweep started from it ("EP · Duration = 200 ms")."""
    base = badge(kind)
    used = {name.strip().casefold() for name in taken}
    stems = {name.split(" · ")[0] for name in used}

    def free(name: str) -> bool:
        return name.casefold() not in used and name.casefold() not in stems

    if free(base):
        return base
    n = 1
    while not free(f"{base}_{n}"):
        n += 1
    return f"{base}_{n}"


def paint_tile(painter: QPainter, rect: QRectF, kind: str) -> None:
    """The kind's icon on a soft square of its tone (the New menu, the Setup step's title)."""
    p = theme().palette
    shade = tone(kind)
    tint = QColor(tone_color(p, shade))
    tint.setAlphaF(0.16)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(tint)
    radius = rect.width() * 0.26
    painter.drawRoundedRect(rect, radius, radius)
    size = round(rect.width() * 0.53)
    painter.drawPixmap(round(rect.x() + (rect.width() - size) / 2), round(rect.y() + (rect.height() - size) / 2),
                       icons.pixmap(icon(kind), tone_text(p, shade), size))
    painter.restore()


class KindTile(QWidget):
    """A kind's tile as a widget."""

    def __init__(self, kind: str, size: int = 40) -> None:
        super().__init__()
        self._kind = kind
        self.setFixedSize(size, size)

    def set_kind(self, kind: str) -> None:
        if kind != self._kind:
            self._kind = kind
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        paint_tile(QPainter(self), QRectF(self.rect()), self._kind)
