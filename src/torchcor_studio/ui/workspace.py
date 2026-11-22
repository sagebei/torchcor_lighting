"""The right panel: a tab per simulation (from set-up to results), plus Machines and Settings.

Tabs carry a *kind* (``"draft"``, ``"run"``, ``"machines"``, ``"settings"``) and, for runs,
the run's key. With no tab open, the panel offers to start a new simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QLineEdit,
    QMenu,
    QStyle,
    QStyleOptionTab,
    QTabBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio.ui import icons
from torchcor_studio.ui.new_menu import NewMenuButton
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import Button, EmptyState, IconButton
from torchcor_studio.ui.widgets.titlebar import UNIFIED, WindowDrag

#: Longer tab titles are shortened (the whole title is in the tab's tooltip).
MAX_TITLE = 30
#: Height of the tab strip (the title bar on macOS): the stylesheet's tab height + padding +
#: border + top margin, so that the sidebar's title-bar space can match it exactly.
STRIP_HEIGHT = 28


def _short(title: str) -> str:
    return title if len(title) <= MAX_TITLE else title[: MAX_TITLE - 1].rstrip() + "…"


@dataclass
class _Tab:
    kind: str
    icon: str
    key: str | None = None
    title: str = ""  # in full (the tab shows it shortened)


class _CloseButton(QToolButton):
    """The small × on a tab (a plain icon button would pad the glyph away at this size)."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("TabClose")
        self.setFixedSize(18, 18)
        self.setIconSize(QSize(12, 12))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Close tab  (Ctrl+W)")
        self.tint()

    def tint(self) -> None:
        self.setIcon(icons.icon("x", theme().palette.text2, 12))


class _TitleEditor(QLineEdit):
    """Edits a tab's title in place; Enter or focus loss commits, Esc cancels."""

    def __init__(self, parent: QWidget, text: str) -> None:
        super().__init__(text, parent)
        self.setObjectName("TabTitleEdit")
        self.cancelled = False

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Escape:
            self.cancelled = True
            self.clearFocus()
            return
        super().keyPressEvent(e)


class Workspace(QWidget):
    """Tabs on a strip above the pages; the "+" at its end opens the New menu."""

    new_requested = Signal(str)  # kind, chosen in the New menu
    current_changed = Signal(object)  # the page now shown (or None)
    close_requested = Signal(object)  # a page whose tab's × was clicked
    title_edited = Signal(object, str)  # page, new title (double-click on its tab)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("Workspace")
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.tabs.setUsesScrollButtons(True)
        bar = self.tabs.tabBar()
        bar.setObjectName("WorkspaceTabBar")
        bar.setExpanding(False)
        bar.setDrawBase(False)
        bar.setElideMode(Qt.TextElideMode.ElideRight)
        bar.tabBarDoubleClicked.connect(self._edit_title)
        bar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._tab_menu)
        self.new_button = IconButton("plus", "New simulation  (Ctrl+N)", size=24, icon_size=14)
        self.new_menu = NewMenuButton(self.new_button, self._new_menu_place)  # below the "+"
        self.new_menu.chosen.connect(self.new_requested.emit)
        corner = QWidget()
        corner.setObjectName("WorkspaceCorner")
        corner_lay = QVBoxLayout(corner)
        corner_lay.setContentsMargins(4, 2, 6, 2)
        corner_lay.addWidget(self.new_button)
        self.tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)
        self.tabs.currentChanged.connect(lambda i: self.current_changed.emit(self.tabs.widget(i)))
        if UNIFIED:  # the strip is the title bar: its empty parts move the window
            bar.installEventFilter(WindowDrag(self, lambda w, pos: w.tabAt(pos) < 0))
            # (the strip's height, not the bar's: with no tab open the bar has none)
            self.tabs.installEventFilter(WindowDrag(self, lambda w, pos: pos.y() < STRIP_HEIGHT))
            corner.installEventFilter(WindowDrag(self))
        lay.addWidget(self.tabs)
        self._info: dict[int, _Tab] = {}
        self._editor: _TitleEditor | None = None
        self.empty = self._empty_page()
        theme().changed.connect(self._restyle)

    def _new_menu_place(self, size: QSize) -> QPoint:
        """The New menu opens below the "+", its right edge under the button's."""
        button = self.new_button
        return button.mapToGlobal(QPoint(button.width() - size.width(), button.height() + 6))

    # ------------------------------------------------------------------ no tab open
    def _empty_page(self) -> EmptyState:
        """Shown under the strip while no tab is open: a New simulation button (its menu
        opens below it, centred)."""
        button = Button("New simulation", "plus", variant="primary")
        self.empty_menu = NewMenuButton(button, lambda size: button.mapToGlobal(
            QPoint((button.width() - size.width()) // 2, button.height() + 6)))
        self.empty_menu.chosen.connect(self.new_requested.emit)
        page = EmptyState("heart", "No simulation open",
                          "Open a run from the sidebar, or start a new simulation.", button)
        page.setParent(self.tabs)
        page.hide()
        self.tabs.installEventFilter(self)  # resizes it with the panel
        return page

    def _sync_empty(self) -> None:
        empty = self.tabs.count() == 0
        if empty:
            self.empty.setGeometry(QRect(0, STRIP_HEIGHT, self.tabs.width(), max(0, self.tabs.height() - STRIP_HEIGHT)))
            self.empty.raise_()
        self.empty.setVisible(empty)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802, ANN001
        if obj is self.tabs and event.type() == QEvent.Type.Resize:
            self._sync_empty()
        return False

    # ------------------------------------------------------------------ tabs
    def add(self, page: QWidget, kind: str, title: str, icon: str, *, key: str | None = None,
            focus: bool = True, index: int | None = None) -> None:
        if index is None:
            index = self.tabs.count()
        index = self.tabs.insertTab(index, page, _short(title))
        self._info[id(page)] = _Tab(kind, icon, key, title)
        close = _CloseButton()
        close.clicked.connect(lambda: self.close_requested.emit(page))
        self.tabs.tabBar().setTabButton(index, QTabBar.ButtonPosition.RightSide, close)
        self._style(index)
        if focus:
            self.tabs.setCurrentIndex(index)
        self._sync_empty()

    def replace(self, old: QWidget, new: QWidget, kind: str, title: str, icon: str, *,
                key: str | None = None) -> None:
        """Put ``new`` in the place of ``old`` (whose widget is not deleted here)."""
        index = self.tabs.indexOf(old)
        self.remove(old)
        self.add(new, kind, title, icon, key=key, index=max(0, index))

    def remove(self, page: QWidget) -> None:
        index = self.tabs.indexOf(page)
        if index >= 0:
            self.tabs.removeTab(index)
        self._info.pop(id(page), None)
        self._sync_empty()

    def focus(self, page: QWidget) -> None:
        index = self.tabs.indexOf(page)
        if index >= 0:
            self.tabs.setCurrentIndex(index)

    def current(self) -> QWidget | None:
        return self.tabs.currentWidget()

    def pages(self, kind: str | None = None) -> list[QWidget]:
        out = [self.tabs.widget(i) for i in range(self.tabs.count())]
        return [p for p in out if kind is None or self.kind(p) == kind]

    def find(self, kind: str, key: str | None = None) -> QWidget | None:
        return next((p for p in self.pages(kind) if key is None or self.key(p) == key), None)

    def kind(self, page: QWidget | None) -> str | None:
        info = self._info.get(id(page)) if page is not None else None
        return info.kind if info else None

    def key(self, page: QWidget | None) -> str | None:
        info = self._info.get(id(page)) if page is not None else None
        return info.key if info else None

    def set_title(self, page: QWidget, title: str, icon: str | None = None) -> None:
        index = self.tabs.indexOf(page)
        if index < 0:
            return
        info = self._info[id(page)]
        if icon is not None:
            info.icon = icon
        info.title = title
        if self.tabs.tabText(index) != _short(title):
            self.tabs.setTabText(index, _short(title))
        self._style(index)

    def title(self, page: QWidget) -> str:
        info = self._info.get(id(page))
        return info.title if info else ""

    def set_key(self, page: QWidget, key: str | None) -> None:
        if id(page) in self._info:
            self._info[id(page)].key = key

    # ------------------------------------------------------------------ look
    def _style(self, index: int) -> None:
        if index < 0:
            return
        page = self.tabs.widget(index)
        info = self._info.get(id(page))
        if info is None:
            return
        self.tabs.setTabIcon(index, icons.icon(info.icon, theme().palette.text2, 14))
        self.tabs.tabBar().setTabToolTip(index, info.title)

    def _restyle(self, *_args) -> None:
        bar = self.tabs.tabBar()
        for i in range(self.tabs.count()):
            self._style(i)
            button = bar.tabButton(i, QTabBar.ButtonPosition.RightSide)
            if isinstance(button, _CloseButton):
                button.tint()

    # ------------------------------------------------------------------ tab menu
    def _tab_menu(self, pos) -> None:  # noqa: ANN001
        bar = self.tabs.tabBar()
        index = bar.tabAt(pos)
        if index < 0:
            return
        page = self.tabs.widget(index)
        p = theme().palette
        menu = QMenu(self)
        if self.kind(page) in ("draft", "run"):
            menu.addAction(icons.icon("edit", p.text2), "Rename", lambda: self._edit_title(index))
        menu.addSeparator()
        menu.addAction(icons.icon("x", p.text2), "Close", lambda: self.close_requested.emit(page))
        others = [w for w in self.pages() if w is not page]
        close_others = menu.addAction("Close other tabs", lambda: [self.close_requested.emit(w) for w in others])
        close_others.setEnabled(bool(others))
        menu.exec(bar.mapToGlobal(pos))

    # ------------------------------------------------------------------ rename in place
    def _edit_title(self, index: int) -> None:
        if index < 0:
            return
        page = self.tabs.widget(index)
        if self.kind(page) not in ("draft", "run"):
            return
        bar = self.tabs.tabBar()
        editor = _TitleEditor(bar, self.title(page))
        opt = QStyleOptionTab()
        bar.initStyleOption(opt, index)
        text = bar.style().subElementRect(QStyle.SubElement.SE_TabBarTabText, opt, bar)
        # over the title, centred on it, its text where the tab's was (frame + padding: 5 px)
        height = editor.fontMetrics().height() + 4
        editor.setGeometry(text.left() - 5, text.center().y() + 1 - height // 2, text.width() + 5, height)
        editor.selectAll()
        editor.show()
        editor.setFocus()
        self._editor = editor

        def done() -> None:
            if self._editor is not editor:
                return
            self._editor = None
            text = editor.text().strip()
            if not editor.cancelled and text and text != self.title(page):
                self.title_edited.emit(page, text)
            editor.deleteLater()

        editor.editingFinished.connect(done)
        editor.installEventFilter(_FocusOut(editor, done))


class _FocusOut(QObject):
    """Commit an in-place editor when it loses focus (editingFinished covers Enter)."""

    def __init__(self, parent: QObject, callback) -> None:  # noqa: ANN001
        super().__init__(parent)
        self._callback = callback

    def eventFilter(self, obj, event) -> bool:  # noqa: N802, ANN001
        if event.type() == QEvent.Type.FocusOut:
            self._callback()
        return False
