"""The run-folder protocol shared with the compute-side runtime.

The constants are imported from :mod:`torchcor_studio.runtime.tcs_runtime` so both
sides can never drift apart.
"""

from __future__ import annotations

import re
import secrets
import time
from datetime import datetime
from typing import Any

from torchcor_studio import __version__
from torchcor_studio.runtime import tcs_runtime as _rt

MANIFEST = _rt.MANIFEST
STATUS = _rt.STATUS
LOG = _rt.LOG
EVENTS = _rt.EVENTS
SCRIPT = _rt.SCRIPT
CANCEL = _rt.CANCEL
RESULTS_DIR = _rt.RESULTS_DIR
VIZ_DIR = _rt.VIZ_DIR
STALE_AFTER_S = _rt.STALE_AFTER_S
MANIFEST_SCHEMA = 1


def slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].strip("-") or "run"


def new_run_id(name: str, now: float | None = None) -> str:
    """Sortable, human-readable and collision-resistant: ``20261121-143012-ventricle-pacing-3f9a``."""
    stamp = datetime.fromtimestamp(now or time.time()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slugify(name, 32)}-{secrets.token_hex(2)}"


def default_run_name(title: str, variant: str = "") -> str:
    stamp = datetime.now().strftime("%b %d, %H:%M")
    return f"{variant or title} · {stamp}"


def build_manifest(
    *,
    run_id: str,
    name: str,
    kind: str,
    variant: str,
    params: dict[str, Any],
    machine: dict[str, Any],
    device: str,
    device_name: str,
    python: str,
    input_dir: str,
    output_dir: str,
    torchcor_version: str | None,
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "id": run_id,
        "name": name,
        "kind": kind,
        "variant": variant,
        "created_at": time.time(),
        "machine": machine,
        "device": device,
        "device_name": device_name,
        "python": python,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "params": params,
        "torchcor_version": torchcor_version,
        "studio_version": __version__,
    }
