"""Per-user folders (OS conventions via platformdirs)."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import PlatformDirs

from torchcor_studio import APP_ID, ORG_NAME

_dirs = PlatformDirs(APP_ID, ORG_NAME, roaming=True)


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_dir() -> Path:
    override = os.environ.get("TCS_CONFIG_DIR")
    return _ensure(Path(override) if override else Path(_dirs.user_config_dir))


def data_dir() -> Path:
    override = os.environ.get("TCS_DATA_DIR")
    return _ensure(Path(override) if override else Path(_dirs.user_data_dir))


def cache_dir() -> Path:
    override = os.environ.get("TCS_CACHE_DIR")
    return _ensure(Path(override) if override else Path(_dirs.user_cache_dir))


def default_local_output_dir() -> Path:
    """Where local runs go unless the user picks another folder."""
    return Path.home() / "TorchCorLighting" / "runs"


def run_cache_dir(machine_id: str, run_id: str) -> Path:
    """Local mirror of a (remote) run's visualisation bundle."""
    return _ensure(cache_dir() / "runs" / machine_id / run_id)
