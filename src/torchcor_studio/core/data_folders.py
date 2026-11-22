"""The demo data folder of a machine: ``~/torchcor_data``, as the Compute step downloads it.

It holds ``input/<kind>/<case>`` (``input/ep/ventricle``, or a kind folder itself:
``input/ecg``) and, for each, ``output/<kind>/<case>`` for the runs made from it. New runs
start their input in the input folder of their kind; a demo reads its case and writes to
the matching output folder (:func:`demo_data`). The machine's own data folder
(``MachineProfile.data_dir``) is tried first, so a data folder elsewhere works too.
Everything is listed once, when the machine connects, so looking a case up costs nothing.
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

#: Where the demo data is kept when the machine names no other folder.
DATA_FOLDER = "~/torchcor_data"
INPUT, OUTPUT = "input", "output"
#: Written into the data folder by the runtime's ``fetch-data``: ``{"version": "data-v1.0"}``.
VERSION_FILE = ".version"


@dataclass(frozen=True)
class DataFolders:
    """A data folder found on a machine (absolute path), the kind folders of its ``input``
    and the case folders inside them."""

    root: str
    kinds: frozenset[str]
    cases: frozenset[str] = frozenset()  # "ep/atrium", "ep/ventricle", ...: one level per kind
    windows: bool = False
    home: str = ""  # to read "~/..." paths
    version: str = ""  # "data-v1.0" when it was downloaded by the app

    def _join(self, *parts: str) -> str:
        return (ntpath if self.windows else posixpath).join(self.root, *parts)

    def has_case(self, case: str, requires: Sequence[str] = ()) -> bool:
        """Is the case ("ep/ventricle", or a kind folder itself: "ecg") on the machine, with
        the sub-folders ``requires`` that show its data is there (a kind folder's
        sub-folders are listed, so this costs no call to the machine)?"""
        return (case in self.kinds or case in self.cases) and all(
            f"{case}/{sub}" in self.cases for sub in requires)

    def case_path(self, case: str) -> str:
        """Where the case's input is (or would be) in this data folder."""
        return self._join(INPUT, *case.split("/"))

    def output_path(self, case: str) -> str:
        """Where the runs of the case go: its folder under ``output``."""
        return self._join(OUTPUT, *case.split("/"))

    def folder(self, kind: str) -> str | None:
        """The input folder for ``kind``, if the data folder has one."""
        return self.case_path(kind) if kind in self.kinds else None

    def paired_output(self, path: str) -> str | None:
        """The output folder that matches an input folder of this data folder (``None`` for
        a folder elsewhere): ``.../input/ep/atrium`` -> ``.../output/ep/atrium``."""
        inputs = self._join(INPUT)
        if not self.contains(inputs, path):
            return None
        paths = ntpath if self.windows else posixpath
        rel = paths.relpath(self._expand(path), inputs)
        return self._join(OUTPUT) if rel == "." else self._join(OUTPUT, rel)

    def contains(self, folder: str, path: str) -> bool:
        """Is ``path`` ``folder`` itself or inside it?"""
        paths = ntpath if self.windows else posixpath
        norm = (lambda p: paths.normcase(paths.normpath(p))) if self.windows else posixpath.normpath
        folder, path = norm(folder), norm(self._expand(path))
        return path == folder or path.startswith(folder.rstrip("/\\") + ("\\" if self.windows else "/"))

    def _expand(self, path: str) -> str:
        if self.home and (path == "~" or path.startswith(("~/", "~\\"))):
            return self.home + path[1:]
        return path


def find_data_folders(machine: Machine) -> DataFolders | None:
    """The machine's data folder: its profile's, else :data:`DATA_FOLDER`, whichever first
    has an ``input`` folder with a sub-folder named after a simulation kind; with the case
    folders of each kind (one listing per kind folder; a failed one is skipped) and the
    version it was downloaded as."""
    kinds = {str(k) for k in SimulationKind}
    windows = machine.profile.is_local and os.name == "nt"
    join = (ntpath if windows else posixpath).join
    for candidate in dict.fromkeys(c for c in (machine.profile.data_dir, DATA_FOLDER) if c):
        try:
            root = machine.expand(candidate)
            inputs = join(root, INPUT)
            if not machine.is_dir(inputs):
                continue
            found = frozenset(e.name for e in machine.listdir(inputs) if e.is_dir and e.name in kinds)
        except (MachineError, OSError):
            continue
        if not found:
            continue
        cases = set()
        for kind in sorted(found):
            try:
                cases |= {f"{kind}/{e.name}" for e in machine.listdir(join(inputs, kind)) if e.is_dir}
            except (MachineError, OSError):
                continue
        version = ""
        try:
            if machine.exists(join(root, VERSION_FILE)):
                version = str(machine.read_json(join(root, VERSION_FILE)).get("version", ""))
        except (MachineError, OSError, ValueError, AttributeError):
            pass
        return DataFolders(root, found, frozenset(cases), windows=windows, home=machine.expand("~"),
                           version=version)
    return None


class DataStatus(StrEnum):
    """Where a demo's data stands on the selected machine."""

    BUILT_IN = "built_in"  # needs no input folder
    FOUND = "found"
    MISSING = "missing"
    UNKNOWN = "unknown"  # the machine has not connected yet


@dataclass(frozen=True)
class DemoData:
    status: DataStatus
    path: str  # where the case's input is or would be; "" for built-in
    output: str = ""  # where its runs go, when the data is there (or may be: UNKNOWN)


def demo_data(demo: Demo, data: DataFolders | None, connected: bool) -> DemoData:
    """BUILT_IN when the demo has no case; UNKNOWN until the machine has connected; FOUND when
    the data folder lists the case (and the sub-folders it ``requires``); else MISSING (its
    expected path: ``~/torchcor_data/input/<case>`` when the machine has no data folder)."""
    if demo.case is None:
        return DemoData(DataStatus.BUILT_IN, "")
    if not connected:
        return DemoData(DataStatus.UNKNOWN, f"{DATA_FOLDER}/{INPUT}/{demo.case}",
                        f"{DATA_FOLDER}/{OUTPUT}/{demo.case}")
    if data is not None and data.has_case(demo.case, demo.requires):
        return DemoData(DataStatus.FOUND, data.case_path(demo.case), data.output_path(demo.case))
    return DemoData(DataStatus.MISSING, data.case_path(demo.case) if data else f"{DATA_FOLDER}/{INPUT}/{demo.case}")
