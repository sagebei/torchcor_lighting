"""What the machine running the tests has: optional packages, and the Qt platform."""

from __future__ import annotations

import importlib.metadata
import os
import sys

import pytest


def installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


#: torch and TorchCor in this Python: a ready simulation environment (a developer's;
#: the app's own CI installs neither, since simulations run in a separate Python).
TORCHCOR_READY = installed("torch") and installed("torchcor")

#: Qt's offscreen platform on Windows (GitHub's runners) lays text out unlike a Windows
#: desktop: labels elide and wrapped text keeps its height. Layout tests that measure text
#: are expected to fail there; they hold on macOS and Linux.
offscreen_windows_text = pytest.mark.xfail(
    sys.platform == "win32" and os.environ.get("QT_QPA_PLATFORM") == "offscreen",
    reason="text metrics of Qt's offscreen platform on Windows", strict=False)
