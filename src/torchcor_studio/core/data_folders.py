"""The TorchCor data folder of a machine: ``~/torchcor_data/<kind>/<case>`` by convention.

A data folder holds one sub-folder per simulation kind (``ep``, ``ecg``, ``mechanics``,
``em``), each with case folders (``ep/atrium``, ``ep/ventricle``, ...). New runs start
their input folder in the sub-folder of their kind, and a demo reads its case
(:func:`demo_data`). The machine's own input folder (``MachineProfile.input_dir``) is
tried first, so a data folder elsewhere works too. Everything is listed once, when the
machine connects, so looking a case up costs nothing.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from torchcor_studio.core.machines.base import Machine, MachineError
from torchcor_studio.core.simulations.base import Demo, SimulationKind

#: Where TorchCor's data sets are kept when the machine names no other folder.
DATA_FOLDER = "~/torchcor_data"


@dataclass(frozen=True)
class DataFolders:
    """A data folder found on a machine (absolute path), its kind sub-folders and the case
    folders inside them."""

    root: str
    kinds: frozenset[str]
    cases: frozenset[str] = frozenset()  # "ep/atrium", "ep/ventricle", ...: one level per kind
    windows: bool = False
    home: str = ""  # to read "~/..." paths

    def has_case(self, case: str, requires: Sequence[str] = ()) -> bool:
        """Is the case ("ep/ventricle", or a kind folder itself: "ecg") on the machine, with
        the sub-folders ``requires`` that show its data is there (a kind folder's
        sub-folders are listed, so this costs no call to the machine)?"""
        return (case in self.kinds or case in self.cases) and all(
            f"{case}/{sub}" in self.cases for sub in requires)

    def case_path(self, case: str) -> str:
        """Where the case is (or would be) in this data folder."""
        return (ntpath if self.windows else posixpath).join(self.root, *case.split("/"))

    def folder(self, kind: str) -> str | None:
        """The sub-folder for ``kind``, if the data folder has one."""
        if kind not in self.kinds:
            return None
        return (ntpath if self.windows else posixpath).join(self.root, kind)

    def contains(self, folder: str, path: str) -> bool:
        """Is ``path`` ``folder`` itself or inside it?"""
        paths = ntpath if self.windows else posixpath
        if self.home and (path == "~" or path.startswith(("~/", "~\\"))):
            path = self.home + path[1:]
        norm = (lambda p: paths.normcase(paths.normpath(p))) if self.windows else posixpath.normpath
        folder, path = norm(folder), norm(path)
        return path == folder or path.startswith(folder.rstrip("/\\") + ("\\" if self.windows else "/"))


def find_data_folders(machine: Machine) -> DataFolders | None:
    """The machine's data folder: its profile's input folder, else :data:`DATA_FOLDER`,
    whichever first holds a sub-folder named after a simulation kind, with the case
    folders of each kind (one listing per kind folder; a failed one is skipped)."""
    kinds = {str(k) for k in SimulationKind}
    candidates = [machine.profile.input_dir, DATA_FOLDER]
    for candidate in dict.fromkeys(c for c in candidates if c):
        try:
            root = machine.expand(candidate)
            if not machine.is_dir(root):
                continue
            found = frozenset(e.name for e in machine.listdir(root) if e.is_dir and e.name in kinds)
        except (MachineError, OSError):
            continue
        if found:
            windows = machine.profile.is_local and os.name == "nt"
            join = (ntpath if windows else posixpath).join
            cases = set()
            for kind in sorted(found):
                try:
                    cases |= {f"{kind}/{e.name}" for e in machine.listdir(join(root, kind)) if e.is_dir}
                except (MachineError, OSError):
                    continue
            return DataFolders(root, found, frozenset(cases), windows=windows,
                               home=machine.expand("~"))
    return None


class DataStatus(StrEnum):
    """Where a demo's data stands on the selected machine."""

    BUILT_IN = "built_in"  # needs no input folder
    FOUND = "found"
    DOWNLOAD = "download"  # missing, and the run downloads it
    MISSING = "missing"
    UNKNOWN = "unknown"  # the machine has not connected yet


@dataclass(frozen=True)
class DemoData:
    status: DataStatus
    path: str  # where the case is or would be; "" for built-in, or a download with no data folder


def demo_data(demo: Demo, data: DataFolders | None, connected: bool) -> DemoData:
    """BUILT_IN when the demo has no case; UNKNOWN until the machine has connected; FOUND when
    the data folder lists the case (and the sub-folders it ``requires``); else DOWNLOAD for a
    demo whose run downloads its data (its path in the data folder, or "" when the machine
    has none), or MISSING (its expected path: ``~/torchcor_data/<case>`` when the machine
    has no data folder)."""
    if demo.case is None:
        return DemoData(DataStatus.BUILT_IN, "")
    if not connected:
        return DemoData(DataStatus.UNKNOWN, f"{DATA_FOLDER}/{demo.case}")
    if data is not None and data.has_case(demo.case, demo.requires):
        return DemoData(DataStatus.FOUND, data.case_path(demo.case))
    if demo.download:
        return DemoData(DataStatus.DOWNLOAD, data.case_path(demo.case) if data else "")
    return DemoData(DataStatus.MISSING, data.case_path(demo.case) if data else f"{DATA_FOLDER}/{demo.case}")
