"""Page scaffolding shared by every page: header bar and scrollable body."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from torchcor_studio.ui.widgets import label

#: The steps of a simulation, at the top of its tab: set up on the "New simulation" page,
#: then run and explored on the run's page (the tab turns into it when the run starts).
SIMULATION_STEPS = ("Compute", "Setup", "Run", "Result")


class PageHeader(QWidget):
    """Title (+ optional subtitle) on the left, actions on the right, hairline below."""

    def __init__(self, title: str = "", subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PageHeader")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(28, 18, 24, 16)
        lay.setSpacing(12)
        self.leading = QHBoxLayout()
        self.leading.setSpacing(12)
        lay.addLayout(self.leading)
        col = QVBoxLayout()
        col.setSpacing(2)
        self.title = label(title, "h2")
        self.subtitle = label(subtitle, "caption")
        self.subtitle.setVisible(bool(subtitle))
        col.addWidget(self.title)
        col.addWidget(self.subtitle)
        lay.addLayout(col, 1)
        self.center = QHBoxLayout()
        lay.addLayout(self.center)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        lay.addLayout(self.actions)

    def set_title(self, title: str, subtitle: str | None = None) -> None:
        self.title.setText(title)
        if subtitle is not None:
            self.subtitle.setText(subtitle)
            self.subtitle.setVisible(bool(subtitle))


def scroll(content: QWidget, max_width: int | None = 1180, margins=(28, 24, 28, 32)) -> QScrollArea:
    """Wrap ``content`` in a scroll area that centres it with a comfortable max width."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    holder = QWidget()
    outer = QHBoxLayout(holder)
    outer.setContentsMargins(*margins)
    outer.addStretch(0)
    if max_width:
        content.setMaximumWidth(max_width)
    outer.addWidget(content, 1)
    outer.addStretch(0)
    area.setWidget(holder)
    return area
