"""Run blocking work off the UI thread and get the result back on it.

    from torchcor_studio.ui.tasks import run_async

    run_async(machine.listdir, path,
              on_done=self._show_listing,           # called on the UI thread
              on_error=self._show_error,            # receives the exception
              owner=self)                           # dropped silently if owner is deleted

Every SSH/file operation in the UI goes through here so the window never freezes.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import shiboken6
from PySide6.QtCore import QObject, Qt, Signal, Slot

log = logging.getLogger(__name__)

_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tcs-worker")


class _Bridge(QObject):
    """Lives on the UI thread; marshals completions back to it."""

    finished = Signal(object, object, object, object)  # callback, value, error, owner

    def __init__(self) -> None:
        super().__init__()
        self.finished.connect(self._deliver, Qt.ConnectionType.QueuedConnection)

    @Slot(object, object, object, object)
    def _deliver(self, callback, value, error, owner) -> None:
        if owner is not None and not shiboken6.isValid(owner):
            return
        try:
            callback(error if error is not None else value)
        except Exception:  # noqa: BLE001 - never let a UI callback kill the event loop
            log.exception("task callback failed")


_bridge: _Bridge | None = None
_bridge_lock = threading.Lock()


def _get_bridge() -> _Bridge:
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = _Bridge()
        return _bridge


def run_async(
    fn: Callable[..., Any],
    *args: Any,
    on_done: Callable[[Any], None] | None = None,
    on_error: Callable[[BaseException], None] | None = None,
    owner: QObject | None = None,
    **kwargs: Any,
) -> Future:
    """Execute ``fn(*args, **kwargs)`` on the worker pool.

    Must be called from the UI thread (the bridge is created there on first use).
    """
    bridge = _get_bridge()

    def job() -> Any:
        try:
            value = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            if on_error is not None:
                bridge.finished.emit(on_error, None, exc, owner)
            else:
                log.warning("background task %s failed: %s", getattr(fn, "__name__", fn), exc)
            raise
        if on_done is not None:
            bridge.finished.emit(on_done, value, None, owner)
        return value

    return _POOL.submit(job)


def shutdown() -> None:
    _POOL.shutdown(wait=False, cancel_futures=True)
