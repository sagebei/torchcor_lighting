"""A unified title bar (macOS): the window content fills the title bar, the native window
buttons float over the sidebar, and empty parts of the top row move the window.

Only Qt's own window hints are used (no native code), and the drag handling is a plain
event filter: nothing runs unless the mouse is pressed on an empty title area.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtWidgets import QWidget

#: Content under the title bar on macOS; other platforms keep their native title bar.
UNIFIED = sys.platform == "darwin"


def unify(window: QWidget, content: QWidget) -> None:
    """Let ``window`` draw ``content`` under its title bar (the native buttons stay on top).

    Qt keeps a window's contents out of the title bar (its "safe area") by default; here
    the content fills it, and the sidebar leaves room for the buttons itself.
    """
    if UNIFIED:
        with warnings.catch_warnings():  # the hint reuses a deprecated alias's value
            warnings.simplefilter("ignore", DeprecationWarning)
            window.setWindowFlag(Qt.WindowType.ExpandedClientAreaHint, True)
        window.setWindowFlag(Qt.WindowType.NoTitleBarBackgroundHint, True)
        for widget in (window, content):
            widget.setAttribute(Qt.WidgetAttribute.WA_ContentsMarginsRespectsSafeArea, False)


def _zoom(window: QWidget) -> None:
    """What a double-click on a title bar does: zoom, or back to the previous size."""
    if window.isMaximized():
        window.showNormal()
    else:
        window.showMaximized()


class WindowDrag(QObject):
    """Event filter: pressing an empty title area moves the window; double-click zooms.

    ``is_empty(widget, pos)`` says whether a point is empty title bar (e.g. not on a tab);
    by default the whole widget is.
    """

    def __init__(self, parent: QObject, is_empty: Callable[[QWidget, QPoint], bool] | None = None) -> None:
        super().__init__(parent)
        self._is_empty = is_empty

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind not in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            return False
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if self._is_empty is not None and not self._is_empty(obj, event.position().toPoint()):
            return False
        window = obj.window()
        if kind == QEvent.Type.MouseButtonDblClick:
            _zoom(window)
        elif window.windowHandle() is not None:
            window.windowHandle().startSystemMove()
        return True


class TitleBarSpace(QWidget):
    """Empty room at the top of the sidebar for the window buttons; it moves the window."""

    def __init__(self, height: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(height)
        self.installEventFilter(WindowDrag(self))
