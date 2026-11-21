"""Load a bundle off the UI thread and show the view for its kind (or a friendly empty state)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QStackedLayout, QVBoxLayout, QWidget

from torchcor_studio.core.results import BundleError, BundleNotFound, ResultBundle, load_bundle
from torchcor_studio.ui.tasks import run_async
from torchcor_studio.ui.widgets import EmptyState, Spinner, label

__all__ = ["AsyncPage", "ResultsView", "build_view", "create_results_view"]

log = logging.getLogger(__name__)


def _prepare(viz_dir: Path) -> ResultBundle:
    """Worker-thread part: validate, decompress the datasets, warm the VTK import."""
    bundle = load_bundle(viz_dir)
    bundle.preload()
    if bundle.kind != "ecg":
        import pyvista  # noqa: F401 - the first import costs ~1 s; pay it here, not on the UI thread
    return bundle


def build_view(bundle: ResultBundle, parent: QWidget | None = None) -> QWidget:
    """The view for a loaded bundle (its datasets must be preloaded)."""
    if bundle.kind == "ep":
        from torchcor_studio.ui.results.ep_view import EPView

        return EPView(bundle, parent)
    if bundle.kind == "ecg":
        from torchcor_studio.ui.results.ecg_view import ECGView

        return ECGView(bundle, parent)
    if bundle.kind == "mechanics":
        from torchcor_studio.ui.results.mechanics_view import MechanicsView

        return MechanicsView(bundle, parent)
    if bundle.kind == "em":
        from torchcor_studio.ui.results.em_view import EMView

        return EMView(bundle, parent)
    raise BundleError(f"There is no view for “{bundle.kind}” results.")


def _loading_page(text: str) -> QWidget:
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.setSpacing(12)
    lay.addWidget(Spinner(24), 0, Qt.AlignmentFlag.AlignHCenter)
    lay.addWidget(label(text, "muted"), 0, Qt.AlignmentFlag.AlignHCenter)
    return page


class AsyncPage(QWidget):
    """Runs ``work`` on the worker pool, then shows ``build(result)``; errors become an empty state.

    The built widget may implement ``set_active(bool)`` and ``release()``; both are forwarded.
    """

    ready = Signal()
    failed = Signal(str)

    def __init__(self, work: Callable[[], object], build: Callable[[object], QWidget],
                 loading_text: str = "Loading results…", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build = build
        self.content: QWidget | None = None
        self.error: str | None = None
        self._active = True
        self._released = False
        self._stack = QStackedLayout(self)
        self._stack.addWidget(_loading_page(loading_text))
        run_async(work, on_done=self._done, on_error=self._failed, owner=self)

    @property
    def loading(self) -> bool:
        return self.content is None and self.error is None

    def _done(self, result) -> None:
        if self._released:
            return
        try:
            content = self._build(result)
        except BundleError as exc:
            self._show_error("Couldn't open these results", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - a broken bundle must not take the page down
            log.exception("building the results view failed")
            self._show_error("Couldn't display these results", f"{type(exc).__name__}: {exc}")
            return
        self.content = content
        self._stack.addWidget(content)
        self._stack.setCurrentWidget(content)
        if hasattr(content, "set_active"):
            content.set_active(self._active)
        self.ready.emit()

    def _failed(self, exc: BaseException) -> None:
        if self._released:
            return
        if isinstance(exc, BundleNotFound):
            self._show_error("No results to show", "This run did not produce a visualisation bundle.", "layers")
        elif isinstance(exc, BundleError):
            self._show_error("Couldn't open these results", str(exc))
        else:
            log.warning("loading results failed: %s", exc)
            self._show_error("Couldn't open these results", f"{type(exc).__name__}: {exc}")

    def _show_error(self, title: str, message: str, icon: str = "alert") -> None:
        self.error = message
        page = EmptyState(icon, title, message)
        self._stack.addWidget(page)
        self._stack.setCurrentWidget(page)
        self.failed.emit(message)

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if self.content is not None and hasattr(self.content, "set_active"):
            self.content.set_active(self._active)

    def release(self) -> None:
        self._released = True
        if self.content is not None and hasattr(self.content, "release"):
            self.content.release()


class ResultsView(AsyncPage):
    """The results of one run: loads ``viz_dir`` in the background, then shows its view."""

    def __init__(self, viz_dir: Path | str, parent: QWidget | None = None) -> None:
        self.viz_dir = Path(viz_dir)
        self.bundle: ResultBundle | None = None
        super().__init__(lambda: _prepare(self.viz_dir), self._make, "Loading results…", parent)

    def _make(self, bundle: ResultBundle) -> QWidget:
        self.bundle = bundle
        return build_view(bundle)

    @property
    def view(self) -> QWidget | None:
        """The kind-specific view once loaded (``EPView``, ``ECGView``, …)."""
        return self.content


def create_results_view(viz_dir: Path | str, parent: QWidget | None = None) -> ResultsView:
    """A results widget for a ``viz/`` folder (or the run folder containing it).

    It loads in the background and shows a polished empty state when the bundle is
    missing or invalid. Call ``set_active(False)`` while its tab is hidden and
    ``release()`` before deleting it.
    """
    return ResultsView(viz_dir, parent)
