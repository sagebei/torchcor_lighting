"""Files deployed to compute machines.

* :mod:`tcs_runtime` — stdlib-only agent: environment probe, input inspection, run
  discovery and the detached run supervisor.
* :mod:`tcs_studio` — helpers imported by generated ``run.py`` scripts: progress
  markers and the export of the lightweight visualisation bundle (needs numpy/torch).

Both are plain files, copied verbatim; they must not import :mod:`torchcor_studio`.
"""

from pathlib import Path

RUNTIME_FILES = ("tcs_runtime.py", "tcs_studio.py")


def runtime_path(name: str) -> Path:
    """Absolute path of a bundled runtime file (works from source and frozen builds)."""
    return Path(__file__).with_name(name)
