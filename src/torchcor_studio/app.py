"""Application entry point: ``torchcor-lighting`` or ``python -m torchcor_studio``."""

from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from typing import NoReturn


def _configure_logging() -> None:
    from torchcor_studio.core.paths import data_dir

    log_dir = data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "studio.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.getLogger("paramiko").setLevel(logging.WARNING)


def _warm_caches() -> None:
    """Build matplotlib's font cache off the UI thread; pyvista imports matplotlib on the
    first 3D view, and a cold cache would otherwise stall that view for many seconds."""
    try:
        from matplotlib import font_manager

        font_manager.findfont("DejaVu Sans")
    except Exception:  # noqa: BLE001 - purely an optimisation
        logging.getLogger(__name__).debug("cache warm-up failed", exc_info=True)


def main(argv: list[str] | None = None) -> NoReturn:
    os.environ.setdefault("QT_API", "pyside6")  # pyvistaqt/qtpy must bind to PySide6
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from torchcor_studio import APP_NAME, ORG_DOMAIN, ORG_NAME, __version__

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setOrganizationDomain(ORG_DOMAIN)
    app.setApplicationVersion(__version__)
    app.setDesktopFileName("torchcor-lighting")
    _configure_logging()

    from torchcor_studio.core.settings import Store
    from torchcor_studio.ui import tasks
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.gc import MainThreadGC
    from torchcor_studio.ui.main_window import MainWindow
    from torchcor_studio.ui.resources import app_icon
    from torchcor_studio.ui.theme import theme

    app.setWindowIcon(app_icon())
    threading.Thread(target=_warm_caches, name="tcs-warmup", daemon=True).start()
    collector = MainThreadGC()  # Qt objects must be collected on this thread
    store = Store()
    theme().set_mode(store.settings.theme)
    ctx = AppContext(store)
    window = MainWindow(ctx)
    window.show()
    ctx.reattach()
    code = app.exec()
    collector.stop()
    tasks.shutdown()
    # Everything is saved by now (MainWindow.closeEvent). Leave without the interpreter's
    # teardown: PySide would destroy the Qt objects still alive, pyqtgraph's context menus
    # among them, in no safe order and segfault ("quit unexpectedly" on macOS). Leaving at
    # once is also the fastest quit.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
