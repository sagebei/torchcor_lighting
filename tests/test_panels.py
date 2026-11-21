"""Left panels the user can widen: the run list and the Machines page list."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gui


def test_panel_splitter_clamps_and_reports_its_width(themed, qtbot) -> None:
    from PySide6.QtWidgets import QWidget

    from torchcor_studio.ui.widgets.layout import PanelSplitter

    split = PanelSplitter(QWidget(), QWidget(), width=900, minimum=220, maximum=560)
    qtbot.addWidget(split)
    split.resize(1400, 600)
    split.show()
    qtbot.wait(20)
    assert split.side_width() == 560  # the remembered width is kept within the limits
    seen: list[int] = []
    split.moved.connect(seen.append)
    split.moveSplitter(300, 1)  # as when the edge is dragged
    assert split.side_width() == 300 and seen[-1] == 300
    split.moveSplitter(100, 1)
    assert split.side_width() == 220


def test_the_run_list_width_is_remembered(themed, qtbot) -> None:
    from torchcor_studio.core.settings import Store
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.main_window import MainWindow

    store = Store()
    store.settings.sidebar_width = 400
    ctx = AppContext(store)
    win = MainWindow(ctx)
    qtbot.addWidget(win)
    win.resize(1480, 920)
    win.show()
    qtbot.waitUntil(lambda: win.sidebar.width() == 400, timeout=3000)
    assert win.sidebar.minimumWidth() < win.sidebar.maximumWidth()  # no longer a fixed width
    from torchcor_studio.ui.widgets.titlebar import UNIFIED

    assert win.windowTitle() == ("" if UNIFIED else "TorchCor Lighting")  # macOS: tabs fill the title bar
    win.split.moveSplitter(330, 1)
    assert ctx.settings.sidebar_width == 330
    win.close()  # saves the settings
    assert Store().settings.sidebar_width == 330


def test_empty_title_bar_areas_move_and_zoom_the_window(themed, qtbot) -> None:
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QWidget

    from torchcor_studio.ui.widgets.titlebar import WindowDrag
    from torchcor_studio.ui.workspace import STRIP_HEIGHT, Workspace

    ws = Workspace()
    qtbot.addWidget(ws)
    ws.add(QWidget(), "draft", "New simulation", "sliders")
    ws.resize(900, 600)
    ws.show()
    bar = ws.tabs.tabBar()
    assert bar.height() == STRIP_HEIGHT  # the sidebar's title space matches it exactly

    def click(widget, pos, kind):  # noqa: ANN001, ANN202
        return QMouseEvent(kind, QPointF(pos), QPointF(pos), Qt.MouseButton.LeftButton,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)

    drag = WindowDrag(ws, lambda w, pos: w.tabAt(pos) < 0)
    on_tab = bar.tabRect(0).center()
    assert not drag.eventFilter(bar, click(bar, on_tab, QEvent.Type.MouseButtonPress))  # tabs keep their clicks
    empty = QPoint(bar.tabRect(0).right() + 40, on_tab.y())  # past the last tab
    window = ws.window()
    was_maximized = window.isMaximized()
    assert drag.eventFilter(bar, click(bar, empty, QEvent.Type.MouseButtonDblClick))  # zoom
    assert window.isMaximized() != was_maximized
