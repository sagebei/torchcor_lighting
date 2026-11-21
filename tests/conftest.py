"""Shared test setup: headless Qt and throwaway per-user folders.

pytest-qt provides the ``qapp`` and ``qtbot`` fixtures. Do not add fixtures here from
feature work; put them in the feature's own test module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_EXIT_STATUS = pytest.StashKey[int]()


@pytest.fixture(autouse=True)
def _isolated_user_dirs(tmp_path_factory, monkeypatch):
    """Never touch the real settings, cache or machine list during tests."""
    base = tmp_path_factory.mktemp("user")
    monkeypatch.setenv("TCS_CONFIG_DIR", str(base / "config"))
    monkeypatch.setenv("TCS_DATA_DIR", str(base / "data"))
    monkeypatch.setenv("TCS_CACHE_DIR", str(base / "cache"))
    yield


@pytest.fixture(autouse=True, scope="session")
def _gui_thread_gc(request):
    """Like the app: never let a worker thread collect Qt objects (that segfaults)."""
    if "qapp" not in request.fixturenames and not any(
            "qtbot" in item.fixturenames or "qapp" in item.fixturenames or "themed" in item.fixturenames
            for item in request.session.items):
        yield
        return
    import gc

    qapp = request.getfixturevalue("qapp")
    from torchcor_studio.ui.gc import MainThreadGC

    collector = MainThreadGC(interval_ms=200)
    yield
    collector.stop()
    collector.deleteLater()
    del collector
    gc.collect()
    _ = qapp


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001
    session.config.stash[_EXIT_STATUS] = int(exitstatus)


def pytest_unconfigure(config) -> None:  # noqa: ANN001
    """Leave without the interpreter's teardown once a Qt application exists, as the app
    does on quit (``torchcor_studio.app.main``): PySide would destroy the Qt objects still
    alive, pyqtgraph's context menus among them, in no safe order and segfault, turning
    a passing run into exit code 139."""
    from PySide6.QtWidgets import QApplication

    status = config.stash.get(_EXIT_STATUS, None)
    if status is not None and QApplication.instance() is not None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)


@pytest.fixture
def themed(qapp):
    """A QApplication with the studio theme applied."""
    from torchcor_studio.ui.theme import theme

    theme().set_mode("dark")
    return qapp
