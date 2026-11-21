"""The Setup rail's demos: the kind's ready-to-run set-ups in a list, what the chosen one is
and where its data stands, and, when there is something to say, where the parameters come
from (with Reset and Undo)."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.core.data_folders import DataStatus, DemoData
from torchcor_studio.core.simulations import Cost, Demo, SimulationSpec
from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import Badge, Button, Card, WrapLabel, keep_together, tone_color

COST = {Cost.SECONDS: "Seconds", Cost.MINUTES: "Minutes", Cost.HOURS: "Hours"}
#: The icon and tone of a demo in the list, by where its data stands.
LIST_ICON = {DataStatus.BUILT_IN: ("box", "neutral"), DataStatus.FOUND: ("folder", "success"),
             DataStatus.DOWNLOAD: ("download", "info"), DataStatus.MISSING: ("alert", "warning"),
             DataStatus.UNKNOWN: ("folder", "neutral")}
_STEPS = (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown, Qt.Key.Key_Home,
          Qt.Key.Key_End)
_CHORDS = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.AltModifier


def data_chip(demo: Demo, data: DemoData, machine: str) -> tuple[str, str, str | None, bool, str]:
    """``(text, tone, icon, dot, tooltip)`` of a demo's data chip; a missing data set is a
    quiet "Needs data" (the input folder says what to do once the demo is chosen)."""
    where = machine or "the machine"
    case = demo.case or ""
    if data.status == DataStatus.BUILT_IN:
        return "Built-in", "neutral", "box", False, "Needs no input folder."
    if data.status == DataStatus.FOUND:
        return case, "success", None, True, data.path
    if data.status == DataStatus.DOWNLOAD:
        return ("Auto-download", "info", "download", False,
                f"{demo.data or 'Its data'} is downloaded on the first run, then reused.")
    if data.status == DataStatus.MISSING:
        return ("Needs data", "neutral", None, False,
                f"This demo reads {data.path}, which is not on {where}. Copy TorchCor's demo data there.")
    return case, "neutral", None, False, f"Checked when {where} connects."


class DemoList(QComboBox):
    """The list of demos. Choosing one applies it, which replaces the parameters, so only a
    choice in the open list does: the wheel scrolls the page, and the arrow keys and letters
    open the list instead of stepping through the demos. The closed box shows the title
    alone (the data chip below says what the list's icon does), ending with "…" when it is
    too long (the whole title is then the tooltip)."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in _STEPS or (event.text().strip() and not event.modifiers() & _CHORDS):
            self.showPopup()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        opt.currentIcon = QIcon()
        field = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt,
                                            QStyle.SubControl.SC_ComboBoxEditField, self)
        room = field.width() - 2
        full = opt.currentText
        opt.currentText = QFontMetrics(self.font()).elidedText(full, Qt.TextElideMode.ElideRight, max(0, room))
        self.setToolTip(full if opt.currentText != full else "")
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)


class DemoPicker(Card):
    """The rail's START FROM A DEMO card: the kind's demos in a list (each marked by where
    its data stands), the chosen one's summary and chips, and a status line with Reset or
    Undo when there is something to say. Choosing the demo shown again puts its parameters
    back."""

    apply_requested = Signal(str)  # demo key
    reset_requested = Signal()
    undo_requested = Signal()

    def __init__(self) -> None:
        super().__init__("Start from a demo", icon="sparkles", padding=14, spacing=10)
        self.list = DemoList()
        self.list.setPlaceholderText("Choose a demo")
        self.list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.list.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.list.setMinimumContentsLength(14)
        self.list.view().setMinimumWidth(360)  # the open list shows whole titles
        self.list.activated.connect(self._chosen)
        self.body.addWidget(self.list)
        self.about = QWidget()  # the chosen demo: its summary and chips
        about = QVBoxLayout(self.about)
        about.setContentsMargins(0, 0, 0, 0)
        about.setSpacing(8)
        self.summary = WrapLabel("", "caption")
        about.addWidget(self.summary)
        chips = QHBoxLayout()
        chips.setSpacing(6)
        self.data_chip = Badge("", "neutral")
        self.cost_chip = Badge("", "neutral", icon="clock")
        self.cost_chip.setToolTip("Rough time on a current GPU.")
        chips.addWidget(self.data_chip)
        chips.addWidget(self.cost_chip)
        chips.addStretch(1)
        about.addLayout(chips)
        self.body.addWidget(self.about)
        self.status_row = QWidget()
        row = QHBoxLayout(self.status_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.status = WrapLabel("", "caption")
        row.addWidget(self.status, 1)
        self.reset = Button("Reset", "reset", "link", size="sm", on_click=self.reset_requested.emit)
        self.undo = Button("Undo", "reset", "link", size="sm", on_click=self.undo_requested.emit)
        self.undo.setToolTip("Put back the parameters and folders the demo replaced.")
        for button in (self.reset, self.undo):
            button.hide()
            row.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)
        self.status_row.hide()
        self.body.addWidget(self.status_row)
        self._demos: tuple[Demo, ...] = ()
        self._data: dict[str, DemoData] = {}
        self._machine = ""
        self._show_chosen()
        theme().changed.connect(self._paint_icons)

    # ------------------------------------------------------------------ public
    def demos(self) -> tuple[Demo, ...]:
        return self._demos

    def chosen(self) -> str | None:
        """The key of the demo the list shows (``None``: none)."""
        return self.list.currentData()

    def set_spec(self, spec: SimulationSpec) -> None:
        """List ``spec``'s demos, none chosen."""
        if spec.demos == self._demos:
            return
        self._demos = spec.demos
        self._data.clear()
        self.list.blockSignals(True)
        self.list.clear()
        for demo in spec.demos:
            self.list.addItem(demo.title, demo.key)
        self.list.setCurrentIndex(-1)
        self.list.blockSignals(False)
        self._paint_icons()
        self._show_chosen()

    def set_data(self, status: Callable[[Demo], DemoData], machine: str) -> None:
        """Where each demo's data stands on ``machine``."""
        self._data = {demo.key: status(demo) for demo in self._demos}
        self._machine = machine
        self._paint_icons()
        self._show_chosen()

    def set_selected(self, key: str | None) -> None:
        """Show the demo the parameters come from (``None``: the list says "Choose a demo")."""
        index = self.list.findData(key) if key else -1
        if index != self.list.currentIndex():
            self.list.setCurrentIndex(index)
            self._show_chosen()

    def set_status(self, text: str, *, tip: str = "", reset: str = "", undo: bool = False) -> None:
        """The status line (none when ``text`` is empty); ``reset``: the Reset button's
        tooltip (no button when empty)."""
        self.status.setText(text)
        self.status.setToolTip(tip)
        self.reset.setVisible(bool(reset) and not undo)
        self.reset.setToolTip(reset)
        self.undo.setVisible(undo)
        self.status_row.setVisible(bool(text))

    # ------------------------------------------------------------------ internals
    def _chosen(self, index: int) -> None:
        key = self.list.itemData(index)
        if key:
            self._show_chosen()
            self.apply_requested.emit(key)

    def _paint_icons(self, *_args) -> None:
        """Each demo's icon and tooltip in the list (again when the theme changes)."""
        p = theme().palette
        for i, demo in enumerate(self._demos):
            data = self._data.get(demo.key)
            name, tone = LIST_ICON[data.status if data else DataStatus.UNKNOWN]
            self.list.setItemIcon(i, icons.icon(name, tone_color(p, tone)))
            tip = data_chip(demo, data, self._machine)[4] if data else ""
            self.list.setItemData(i, "\n".join(filter(None, (demo.summary, tip))), Qt.ItemDataRole.ToolTipRole)

    def _show_chosen(self) -> None:
        """The chosen demo's summary and chips (nothing below the list when none is)."""
        demo = next((d for d in self._demos if d.key == self.list.currentData()), None)
        self.about.setVisible(demo is not None)
        if demo is None:
            return
        self.summary.setText(keep_together(demo.summary))
        data = self._data.get(demo.key)
        text, tone, icon, dot, tip = (data_chip(demo, data, self._machine) if data
                                      else (demo.case or "", "neutral", None, False, ""))
        self.data_chip.set_icon(icon, dot)
        self.data_chip.set(text, tone, outline=bool(data) and data.status == DataStatus.MISSING)
        self.data_chip.setToolTip(tip)
        self.data_chip.setVisible(bool(text))
        self.cost_chip.set(COST[demo.cost])
