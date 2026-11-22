"""Which optional packages the Python running the tests has."""

from __future__ import annotations

import importlib.metadata


def installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


#: torch and TorchCor in this Python: a ready simulation environment (a developer's;
#: the app's own CI installs neither, since simulations run in a separate Python).
TORCHCOR_READY = installed("torch") and installed("torchcor")
