"""Run Python's cyclic garbage collector on the GUI thread only.

Python may start a collection in whatever thread happens to allocate; if that is a worker
thread and the cycle holds Qt wrappers, their C++ objects are destroyed off the GUI thread,
which crashes Qt. The app disables automatic collection and collects from a timer here.
"""

from __future__ import annotations

import gc

from PySide6.QtCore import QObject, QTimer


class MainThreadGC(QObject):
    def __init__(self, interval_ms: int = 1000, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._threshold = gc.get_threshold()
        gc.disable()
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.check)
        self._timer.start()

    def check(self) -> None:
        g0, g1, g2 = gc.get_count()
        t0, t1, t2 = self._threshold
        if g0 > t0:
            generation = 2 if g2 > t2 else 1 if g1 > t1 else 0
            gc.collect(generation)

    def stop(self) -> None:
        self._timer.stop()
        gc.enable()
