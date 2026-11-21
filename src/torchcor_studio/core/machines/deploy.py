"""Deploying the studio runtime (:mod:`torchcor_studio.runtime`) to machines.

The runtime files are copied into a folder named after the SHA-256 of their contents,
``<base>/<sha12>/``. A folder therefore never changes once written: upgrading the app
simply deploys a new folder next to the old one, and runs that are still using the old
copy keep working.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from torchcor_studio.runtime import RUNTIME_FILES, runtime_path

if TYPE_CHECKING:
    from torchcor_studio.core.machines.base import Machine

#: Where SSH hosts keep deployed runtimes (expanded on the host).
REMOTE_RUNTIME_BASE = "~/.cache/torchcor-studio/runtime"


@lru_cache(maxsize=1)
def runtime_bundle() -> tuple[str, dict[str, bytes]]:
    """``(digest, {file name: contents})`` of the bundled runtime files.

    ``digest`` is the first 12 hex characters of a SHA-256 over names and contents.
    """
    files = {name: runtime_path(name).read_bytes() for name in RUNTIME_FILES}
    sha = hashlib.sha256()
    for name in sorted(files):
        sha.update(name.encode())
        sha.update(b"\0")
        sha.update(files[name])
        sha.update(b"\0")
    return sha.hexdigest()[:12], files


def runtime_digest() -> str:
    """Content hash identifying this build's runtime (the deployment folder name)."""
    return runtime_bundle()[0]


def deploy_local(base: Path) -> Path:
    """Copy the runtime into ``base/<digest>/`` on this computer and return that folder.

    The folder is assembled under a temporary name and renamed into place, so
    concurrent callers (or a crash half-way) never leave a partial copy behind.
    """
    digest, files = runtime_bundle()
    target = base / digest
    if all((target / name).is_file() for name in files):
        return target
    base.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{digest}-", dir=base))
    try:
        for name, data in files.items():
            (staging / name).write_bytes(data)
        try:
            os.replace(staging, target)
        except OSError:
            # Another process won the race (target exists and is not empty).
            if not all((target / name).is_file() for name in files):
                shutil.rmtree(target, ignore_errors=True)
                os.replace(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def deploy_remote(machine: Machine, base: str = REMOTE_RUNTIME_BASE) -> str:
    """Upload the runtime to ``base/<digest>/`` on ``machine`` unless it is already there.

    Uses only the generic :class:`Machine` file API (one listing when the runtime is
    present; atomic writes otherwise) and returns the folder path on the machine.
    """
    digest, files = runtime_bundle()
    target = machine.join(machine.expand(base), digest)
    try:
        present = {e.name for e in machine.listdir(target) if not e.is_dir}
    except OSError:
        present = set()
    missing = [name for name in files if name not in present]
    if missing:
        machine.mkdir(target, parents=True)
        for name in missing:
            machine.write_bytes(machine.join(target, name), files[name])
    return target
