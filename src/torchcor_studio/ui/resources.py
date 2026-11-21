"""Bundled images (logo, app icon)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap

RESOURCES = Path(__file__).resolve().parent.parent / "resources"


def resource(name: str) -> Path:
    return RESOURCES / name


@lru_cache(maxsize=8)
def logo_pixmap(size: int = 64) -> QPixmap:
    px = QPixmap(str(resource("logo_mark.png")))
    if px.isNull():
        return px
    return px.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


def app_icon() -> QIcon:
    icon = QIcon()
    for s in (16, 32, 64, 128, 256, 512, 1024):
        px = QPixmap(str(resource("app_icon.png")))
        if not px.isNull():
            icon.addPixmap(px.scaled(s, s, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))
    return icon
