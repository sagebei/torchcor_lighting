"""Responsive containers: a row whose side panel sticks in view, card grids that reflow,
side panels the user can resize."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QBoxLayout,
    QGridLayout,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QWidget,
)

_MAX = 16_777_215


class StickyRow(QWidget):
    """A main column with a side panel beside it that stays in view while the page scrolls.

    The row's layout holds the main column and a fixed-width placeholder; the side panel
    is a free child of the row, moved on every scroll so that it sits ``top`` px below the
    top of the viewport, clamped to the row (one ``move``, no layout pass). Its height is
    its natural height (``side.natural_height(width)`` when it has one), at most the
    viewport's height less ``2 * top``. Below ``breakpoint`` px the side panel goes above
    the main column, in the layout, and nothing sticks. ``watch(widget)`` re-measures the
    side panel when that widget's layout changes (content inside a scroll area).
    """

    scrolled = Signal(int)  # the enclosing scroll area's value
    mode_changed = Signal(bool)  # beside

    def __init__(self, main: QWidget, side: QWidget, *, side_width: int | Callable[[int], int] = 300,
                 breakpoint: int = 920, gap: int = 20, top: int = 16, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.main, self.side = main, side
        self._side_width = side_width
        self.breakpoint, self.top = breakpoint, top
        self._lay = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(gap)
        self.slot = QWidget()
        self._lay.addWidget(main, 1)
        self._lay.addWidget(self.slot, 0, Qt.AlignmentFlag.AlignTop)
        side.setParent(self)
        self._beside = True
        self._area: QScrollArea | None = None
        self.watch(side)

    # ------------------------------------------------------------------ public
    def scroll_area(self) -> QScrollArea | None:
        """The scroll area the row lives in (found when the row is first shown)."""
        return self._area

    def is_beside(self) -> bool:
        return self._beside

    def watch(self, widget: QWidget) -> None:
        widget.installEventFilter(self)

    def width_of_side(self) -> int:
        width = self.width() or self.breakpoint
        return self._side_width(width) if callable(self._side_width) else self._side_width

    # ------------------------------------------------------------------ events
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        viewport = self._area.viewport() if self._area is not None else None
        if (kind == QEvent.Type.LayoutRequest and obj is not self) or (
                kind == QEvent.Type.Resize and obj is viewport):
            self._measure()
        return False

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._area is None:
            parent = self.parentWidget()
            while parent is not None and not isinstance(parent, QScrollArea):
                parent = parent.parentWidget()
            if parent is not None:
                self._area = parent
                parent.verticalScrollBar().valueChanged.connect(self._scrolled)
                parent.viewport().installEventFilter(self)
        self._relayout()
        self._measure()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._relayout()
        self._measure()

    def _scrolled(self, value: int) -> None:
        self.follow()
        self.scrolled.emit(value)

    # ------------------------------------------------------------------ geometry
    def _relayout(self) -> None:
        beside = self.width() >= self.breakpoint or self.width() == 0
        width = self.width_of_side()
        if beside and self.slot.width() != width:
            self.slot.setFixedWidth(width)
        if beside == self._beside:
            return
        self._beside = beside
        for widget in (self.main, self.slot, self.side):
            self._lay.removeWidget(widget)  # never a bare takeAt: it keeps a stale item
        if beside:
            self._lay.setDirection(QBoxLayout.Direction.LeftToRight)
            self._lay.addWidget(self.main, 1)
            self._lay.addWidget(self.slot, 0, Qt.AlignmentFlag.AlignTop)
            self.slot.show()
            self.side.setFixedWidth(width)
        else:
            self._lay.setDirection(QBoxLayout.Direction.TopToBottom)
            self.slot.hide()
            self.side.setMinimumSize(0, 0)
            self.side.setMaximumSize(_MAX, _MAX)
            self._lay.addWidget(self.side)
            self._lay.addWidget(self.main, 1)
        self.mode_changed.emit(beside)
        self._measure()

    def natural_height(self, width: int) -> int:
        natural = getattr(self.side, "natural_height", None)
        if natural is not None:
            return natural(width)
        layout = self.side.layout()
        if layout is not None and layout.hasHeightForWidth():
            return layout.totalHeightForWidth(width)
        return self.side.sizeHint().height()

    def _measure(self) -> None:
        """Size the side panel (beside: natural height, capped by the viewport) and follow."""
        if not self._beside:
            return
        width = self.width_of_side()
        height = self.natural_height(width)
        if self._area is not None:
            height = min(height, max(120, self._area.viewport().height() - 2 * self.top))
        self.side.setFixedSize(width, height)
        if self.slot.minimumHeight() != height:
            self.slot.setMinimumHeight(height)
        self.follow()

    def follow(self) -> None:
        """Keep the side panel ``top`` px below the top of the viewport, within the row."""
        if not self._beside:
            return
        y = 0
        if self._area is not None and self._area.widget() is not None:
            row_top = self.mapTo(self._area.widget(), QPoint(0, 0)).y()
            y = self._area.verticalScrollBar().value() - row_top + self.top
        y = max(0, min(y, self.height() - self.side.height()))
        self.side.move(self.slot.x(), y)
        self.side.raise_()


class ResponsiveGrid(QWidget):
    """Equal-width cards that reflow into as many columns as fit (``min_item``…``max_columns``)."""

    def __init__(self, min_item: int = 210, max_columns: int = 5, spacing: int = 12, *, fill: bool = True,
                 max_item: int | None = None, parent: QWidget | None = None) -> None:
        """``fill``: a few cards share the row; without it the columns stay fixed (cards
        keep the column width, which lines grids up with each other). ``max_item``: the
        cards grow no wider, and the rest of the row stays empty."""
        super().__init__(parent)
        self.min_item = min_item
        self.fill = fill
        self.max_columns = max_columns
        self.max_item = max_item
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(spacing)
        self._grid.setVerticalSpacing(spacing)
        if max_item is not None:  # a last column takes what the capped cards leave
            self._grid.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum),
                               0, max_columns)
            self._grid.setColumnStretch(max_columns, 1)
        self._widgets: list[QWidget] = []
        self._columns = 0
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def set_widgets(self, widgets: list[QWidget]) -> None:
        self.clear()
        self._widgets = list(widgets)
        self._relayout(force=True)

    def add(self, widget: QWidget) -> None:
        if self.max_item is not None:
            widget.setMaximumWidth(self.max_item)
        self._widgets.append(widget)
        self._relayout(force=True)

    def clear(self) -> None:
        for w in self._widgets:
            self._grid.removeWidget(w)
        for w in self._widgets:
            w.hide()
            w.deleteLater()
        self._widgets = []
        self._columns = 0

    def widgets(self) -> list[QWidget]:
        return list(self._widgets)

    def columns_for(self, width: int) -> int:
        spacing = self._grid.horizontalSpacing()
        fit = max(1, (width + spacing) // (self.min_item + spacing))
        if not self.fill:
            return max(1, min(self.max_columns, fit))
        return max(1, min(self.max_columns, fit, len(self._widgets) or 1))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self, force: bool = False) -> None:
        cols = self.columns_for(self.width() or self.min_item * self.max_columns)
        if not force and cols == self._columns:
            return
        self._columns = cols
        for w in self._widgets:
            self._grid.removeWidget(w)  # not takeAt: see StickyRow
        for c in range(min(self._grid.columnCount(), self.max_columns)):  # not the spare column
            self._grid.setColumnStretch(c, 0)
        for i, w in enumerate(self._widgets):
            self._grid.addWidget(w, i // cols, i % cols)
            w.show()
        for c in range(cols):  # far above the spare column's 1: it only takes what caps leave
            self._grid.setColumnStretch(c, 100)


class PanelSplitter(QSplitter):
    """A side panel beside the main area whose width the user sets by dragging its edge.

    ``width`` is the initial width (kept within ``minimum``…``maximum``); ``moved`` reports
    the panel's width while it is dragged, so the caller can remember it.
    """

    moved = Signal(int)

    def __init__(self, side: QWidget, main: QWidget, *, width: int, minimum: int, maximum: int,
                 parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setObjectName("PanelSplitter")
        self.setChildrenCollapsible(False)
        self.setHandleWidth(5)
        side.setMinimumWidth(minimum)
        side.setMaximumWidth(maximum)
        self.addWidget(side)
        self.addWidget(main)
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 1)
        self.handle(1).setToolTip("Drag to resize")
        self._width = max(minimum, min(maximum, width))
        self._sized = False
        self.splitterMoved.connect(lambda *_: self.moved.emit(self.side_width()))

    def side_width(self) -> int:
        return self.sizes()[0] if self._sized else self._width

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._sized and self.width() > 0:  # the first real width: place the edge
            self._sized = True
            self.setSizes([self._width, max(1, self.width() - self._width - self.handleWidth())])
