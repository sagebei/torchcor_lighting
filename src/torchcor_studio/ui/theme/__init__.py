"""Theme manager: applies the palette + stylesheet and follows the OS light/dark setting.

Usage::

    from torchcor_studio.ui.theme import theme
    theme().palette.accent           # current token values
    theme().changed.connect(widget.on_theme_changed)

Widgets that paint themselves read ``theme().palette`` in ``paintEvent`` and repaint on
``changed``; stylesheet-driven widgets update automatically.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication, QWidget

from torchcor_studio.core.paths import cache_dir
from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import qss
from torchcor_studio.ui.theme.palette import DARK, LIGHT, METRICS, MONO_FAMILIES, Metrics, Palette

__all__ = ["DARK", "LIGHT", "METRICS", "Metrics", "Palette", "ThemeManager", "mono_font", "repolish", "theme"]


class ThemeManager(QObject):
    changed = Signal(object)  # Palette

    def __init__(self) -> None:
        super().__init__()
        self.mode = "system"
        self.palette: Palette = DARK
        self.metrics: Metrics = METRICS
        hints = QGuiApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(lambda *_: self._follow_system())

    # ------------------------------------------------------------------ public
    def set_mode(self, mode: str) -> None:
        self.mode = mode if mode in ("system", "dark", "light") else "system"
        self.apply(self._resolve())

    def apply(self, palette: Palette) -> None:
        app = QApplication.instance()
        if app is None:
            return
        self.palette = palette
        app.setStyle("Fusion")
        app.setPalette(self._qpalette(palette))
        tint_dir = cache_dir() / "theme"
        files = icons.write_svg_files(tint_dir, palette.text2, ("chevron-down", "chevron-up"))
        files.update(icons.write_svg_files(tint_dir, palette.on_accent, ("check",)))
        app.setStyleSheet(qss.build(palette, files, self.metrics))
        self.changed.emit(palette)
        # Custom-painted widgets read the palette in paintEvent; repaint them all here so
        # they need no signal connection (a lambda would outlive a deleted widget).
        for widget in app.allWidgets():
            widget.update()

    @property
    def is_dark(self) -> bool:
        return self.palette.dark

    # ------------------------------------------------------------------ internals
    def _resolve(self) -> Palette:
        if self.mode == "dark":
            return DARK
        if self.mode == "light":
            return LIGHT
        hints = QGuiApplication.styleHints()
        scheme = hints.colorScheme() if hasattr(hints, "colorScheme") else Qt.ColorScheme.Dark
        return LIGHT if scheme == Qt.ColorScheme.Light else DARK

    def _follow_system(self) -> None:
        if self.mode == "system":
            resolved = self._resolve()
            if resolved is not self.palette:
                self.apply(resolved)

    @staticmethod
    def _qpalette(p: Palette) -> QPalette:
        pal = QPalette()
        role = QPalette.ColorRole
        pal.setColor(role.Window, QColor(p.bg0))
        pal.setColor(role.WindowText, QColor(p.text))
        pal.setColor(role.Base, QColor(p.bg3))
        pal.setColor(role.AlternateBase, QColor(p.bg2))
        pal.setColor(role.ToolTipBase, QColor(p.bg3))
        pal.setColor(role.ToolTipText, QColor(p.text))
        pal.setColor(role.PlaceholderText, QColor(p.text3))
        pal.setColor(role.Text, QColor(p.text))
        pal.setColor(role.Button, QColor(p.bg3))
        pal.setColor(role.ButtonText, QColor(p.text))
        pal.setColor(role.BrightText, QColor(p.danger))
        pal.setColor(role.Highlight, QColor(p.accent))
        pal.setColor(role.HighlightedText, QColor(p.on_accent))
        pal.setColor(role.Link, QColor(p.accent_text))
        pal.setColor(role.Mid, QColor(p.line2))
        pal.setColor(role.Dark, QColor(p.bg1))
        pal.setColor(role.Light, QColor(p.bg4))
        pal.setColor(role.Shadow, QColor(0, 0, 0, 120))
        for r in (role.Text, role.WindowText, role.ButtonText):
            pal.setColor(QPalette.ColorGroup.Disabled, r, QColor(p.text3))
        return pal


_THEME: ThemeManager | None = None


def theme() -> ThemeManager:
    global _THEME
    if _THEME is None:
        _THEME = ThemeManager()
    return _THEME


def mono_font(size: int | None = None) -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    available = set(QFontDatabase.families())
    for family in MONO_FAMILIES:
        if family in available:
            font = QFont(family)
            break
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPixelSize(size or METRICS.fs_sm)
    return font


def repolish(widget: QWidget) -> None:
    """Re-apply the stylesheet after changing a dynamic property."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
