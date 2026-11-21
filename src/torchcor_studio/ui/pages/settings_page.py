"""Settings: appearance, TorchCor source, telemetry, storage, about."""

from __future__ import annotations

import shutil

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from torchcor_studio import APP_NAME, TORCHCOR_GIT_URL, __version__
from torchcor_studio.core.paths import cache_dir, config_dir
from torchcor_studio.ui.context import AppContext
from torchcor_studio.ui.format import size
from torchcor_studio.ui.pages.common import PageHeader, scroll
from torchcor_studio.ui.tasks import run_async
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import Button, Card, SegmentedControl, label


def _row(grid: QGridLayout, r: int, title: str, widget: QWidget, caption: str = "") -> None:
    col = QVBoxLayout()
    col.setSpacing(2)
    col.addWidget(label(title))
    if caption:
        col.addWidget(label(caption, "caption", wrap=True))
    grid.addLayout(col, r, 0)
    grid.addWidget(widget, r, 1)


class SettingsPage(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        s = ctx.settings
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(PageHeader("Settings", "Preferences are saved automatically."))
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(16)

        # appearance ---------------------------------------------------------------
        appearance = Card("Appearance", icon="sun")
        grid = _grid()
        self.theme_ctl = SegmentedControl([("system", "System", "monitor"), ("light", "Light", "sun"),
                                           ("dark", "Dark", "moon")], value=s.theme)
        self.theme_ctl.changed.connect(self._theme)
        _row(grid, 0, "Theme", self.theme_ctl, "Follow the operating system or pick one.")
        appearance.addLayout(grid)
        col.addWidget(appearance)

        # torchcor source ----------------------------------------------------------
        source = Card("TorchCor", "Where “Install / Update TorchCor” gets the library from.", icon="package")
        grid = _grid()
        self.source_ctl = SegmentedControl([("git", "GitHub"), ("pypi", "PyPI"), ("path", "Local folder")],
                                           value=s.torchcor_source)
        self.source_ctl.changed.connect(self._source)
        _row(grid, 0, "Source", self.source_ctl)
        self.git_url = QLineEdit(s.torchcor_git_url or TORCHCOR_GIT_URL)
        self.git_url.editingFinished.connect(self._save_source)
        _row(grid, 1, "Repository", self.git_url, "Any pip-compatible git URL.")
        self.git_ref = QLineEdit(s.torchcor_git_ref)
        self.git_ref.setPlaceholderText("default branch")
        self.git_ref.editingFinished.connect(self._save_source)
        _row(grid, 2, "Branch, tag or commit", self.git_ref)
        self.local_path = QLineEdit(s.torchcor_local_path)
        self.local_path.setPlaceholderText("/path/to/torchcor checkout (on the compute machine)")
        self.local_path.editingFinished.connect(self._save_source)
        _row(grid, 3, "Local checkout", self.local_path, "Installed in editable mode is not required; pip installs it.")
        self.requirement = label("", "mono")
        _row(grid, 4, "pip requirement", self.requirement)
        source.addLayout(grid)
        col.addWidget(source)
        self._source_rows = {"git": [1, 2], "path": [3]}
        self._grid_source = grid

        # telemetry ----------------------------------------------------------------
        tele = Card("Telemetry", icon="activity")
        grid = _grid()
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.5, 10)
        self.interval.setSingleStep(0.5)
        self.interval.setSuffix("  s")
        self.interval.setValue(s.telemetry_interval_s)
        self.interval.setFixedWidth(140)
        self.interval.valueChanged.connect(self._interval)
        _row(grid, 0, "GPU refresh interval", self.interval, "How often device usage is sampled while visible.")
        tele.addLayout(grid)
        col.addWidget(tele)

        # storage ------------------------------------------------------------------
        storage = Card("Storage", icon="memory")
        grid = _grid()
        cache_row = QHBoxLayout()
        self.cache_label = label("…", "muted")
        cache_row.addWidget(self.cache_label, 1)
        cache_row.addWidget(Button("Open", "folder-open", "ghost", size="sm",
                                   on_click=lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(cache_dir())))))
        cache_row.addWidget(Button("Clear", "trash", "ghost", size="sm", on_click=self._clear_cache))
        holder = QWidget()
        holder.setLayout(cache_row)
        _row(grid, 0, "Downloaded results", holder, "Local copies of visualisation bundles from remote runs.")
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(label(str(config_dir()), "mono"), 1)
        cfg_row.addWidget(Button("Open", "folder-open", "ghost", size="sm",
                                 on_click=lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(config_dir())))))
        holder2 = QWidget()
        holder2.setLayout(cfg_row)
        _row(grid, 1, "Settings folder", holder2, "Machines and preferences. Passwords live in the OS keychain.")
        storage.addLayout(grid)
        col.addWidget(storage)

        # about --------------------------------------------------------------------
        about = Card("About", icon="info")
        about.addWidget(label(f"{APP_NAME} {__version__}", "h3"))
        about.addWidget(label("A desktop studio for TorchCor — GPU-accelerated cardiac electrophysiology, ECG, "
                              "mechanics and electromechanics. Simulations run as plain Python scripts you can "
                              "inspect and re-run.", "muted", wrap=True))
        links = QHBoxLayout()
        links.addWidget(Button("TorchCor on GitHub", "external", "ghost", size="sm",
                               on_click=lambda: QDesktopServices.openUrl(QUrl(TORCHCOR_GIT_URL.removesuffix(".git")))))
        links.addWidget(Button("Documentation", "book", "ghost", size="sm",
                               on_click=lambda: QDesktopServices.openUrl(QUrl("https://torchcor.readthedocs.io"))))
        links.addStretch(1)
        about.addLayout(links)
        col.addWidget(about)
        col.addStretch(1)
        lay.addWidget(scroll(body, max_width=860))
        self._update_source_rows()

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        self._measure_cache()

    # ------------------------------------------------------------------ handlers
    def _theme(self, mode: str) -> None:
        self.ctx.settings.theme = mode
        self.ctx.store.save_settings()
        theme().set_mode(mode)

    def _source(self, value: str) -> None:
        self.ctx.settings.torchcor_source = value
        self._save_source()

    def _save_source(self) -> None:
        s = self.ctx.settings
        s.torchcor_git_url = self.git_url.text().strip() or TORCHCOR_GIT_URL
        s.torchcor_git_ref = self.git_ref.text().strip()
        s.torchcor_local_path = self.local_path.text().strip()
        self.ctx.store.save_settings()
        self._update_source_rows()

    def _update_source_rows(self) -> None:
        current = self.ctx.settings.torchcor_source
        for key, rows in self._source_rows.items():
            for r in rows:
                for c in range(2):
                    item = self._grid_source.itemAtPosition(r, c)
                    if item is None:
                        continue
                    if item.widget():
                        item.widget().setVisible(key == current)
                    elif item.layout():
                        for i in range(item.layout().count()):
                            w = item.layout().itemAt(i).widget()
                            if w:
                                w.setVisible(key == current)
        self.requirement.setText(self.ctx.settings.torchcor_requirement())

    def _interval(self, value: float) -> None:
        self.ctx.settings.telemetry_interval_s = float(value)
        self.ctx.store.save_settings()

    def _measure_cache(self) -> None:
        def work() -> int:
            root = cache_dir() / "runs"
            return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0

        run_async(work, on_done=lambda n: self.cache_label.setText(size(n) if n else "Empty"), owner=self)

    def _clear_cache(self) -> None:
        shutil.rmtree(cache_dir() / "runs", ignore_errors=True)
        self._measure_cache()
        self.ctx.notify.emit("Cache cleared", "Remote results will be downloaded again when opened.", "success")


def _grid() -> QGridLayout:
    grid = QGridLayout()
    grid.setHorizontalSpacing(24)
    grid.setVerticalSpacing(16)
    grid.setColumnMinimumWidth(0, 240)
    grid.setColumnStretch(1, 1)
    return grid
