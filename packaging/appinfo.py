"""Facts about the app shared by ``torchcor_studio.spec`` and ``build.py``.

Build-time only (never bundled). Stdlib only, and nothing here imports
``torchcor_studio``: the version and module list are read from the source tree.
"""

from __future__ import annotations

import ast
import platform
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # packaging/
ROOT = HERE.parent
SRC = ROOT / "src"
PKG = SRC / "torchcor_studio"
RESOURCES = PKG / "resources"
RUNTIME = PKG / "runtime"

APP_NAME = "TorchCor Lighting"
BUNDLE_ID = "org.torchcor.lighting"
#: Linux convention: a lower-case, space-free executable, matching the pip console
#: script and ``torchcor-lighting.desktop`` (also the Wayland app id set by the app).
LINUX_NAME = "torchcor-lighting"
PUBLISHER = "TorchCor Lighting contributors"
COPYRIGHT = f"Copyright © 2026 {PUBLISHER}. MIT License."
#: The Debian package's one-line synopsis (its long description is pyproject's).
SYNOPSIS = "desktop app for TorchCor cardiac simulations"
#: What the Linux bundle loads from the system instead of carrying: PyInstaller leaves out
#: glibc (``libc6 (>= <the build machine's glibc>)`` is added by build.py), the OpenGL / EGL
#: driver stack, libdrm, libxcb and Wayland, which must match the machine's drivers and
#: display server. build.py checks the bundle against this list before making the .deb.
DEB_DEPENDS = ("libdrm2", "libegl1", "libgl1", "libglvnd0", "libglx0", "libwayland-client0",
               "libwayland-cursor0", "libwayland-egl1", "libxcb1")
#: The oldest macOS the bundle may claim; raised automatically to what the bundled
#: binaries (Qt, numpy, VTK ...) actually require.
MACOS_MIN_FLOOR = (11, 0)

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

#: Modules the frozen app must never contain. Simulations run in a separate Python
#: environment on the compute machine, so the deep-learning / scientific stacks that
#: are often installed next to the app (and that ``tcs_studio.py`` imports *on the
#: compute machine*) stay out, as do GUI toolkits and dev tools.
EXCLUDES = [
    # compute-side stack (imported lazily by runtime/tcs_studio.py on the compute machine)
    "torch", "torchvision", "torchaudio", "torchcor", "triton", "functorch",
    "scipy", "pandas", "sympy", "sklearn", "skimage", "numba", "llvmlite", "cupy", "jax",
    "networkx", "gmsh", "pygmsh", "meshio", "wfdb", "soundfile", "seaborn", "statsmodels",
    "h5py", "tables", "dask", "xarray", "pyarrow", "imageio", "imageio_ffmpeg",
    # other GUI toolkits / notebook stacks
    "tkinter", "_tkinter", "turtle", "turtledemo", "idlelib",
    "PyQt5", "PyQt6", "PySide2", "wx", "gi",
    "IPython", "ipykernel", "ipywidgets", "jupyter_client", "jupyter_core", "notebook", "nbformat",
    "trame", "trame_client", "trame_server", "trame_vtk", "trame_vuetify",
    # pyvista / VTK extras the app does not use. `vtk` is the eager shim that imports
    # every VTK module (vtkmodules.all); pyvista only references it for type checkers.
    "vtk", "vtkmodules.all", "vtkmodules.test", "vtkmodules.tk", "vtkmodules.gtk",
    "vtkmodules.wx", "vtkmodules.web", "vtkmodules.generate_pyi",
    "pyvista.ext", "pyvista.trame", "pyvista.conftest", "pyvista.examples", "pyvista.demos",
    # optional extras of the dependencies (remote files, pretty printing, plugins ...)
    "fsspec", "aiohttp", "jinja2", "rich", "pygments", "invoke", "filelock", "tqdm",
    "pyparsing.diagram", "pydantic.mypy", "pydantic.v1.mypy", "numpy.f2py", "numpy.distutils",
    # (not "distutils": from Python 3.12 PyInstaller aliases it to setuptools._distutils
    # and fails on an excluded alias; with setuptools excluded it stays out all the same)
    "setuptools", "_distutils_hack", "pkg_resources", "wheel", "tomli",
    # Qt modules pulled in by optional imports only (qtpy's compatibility alias; PySide6's
    # Windows OpenSSL probe, guarded by try/except)
    "PySide6.QtDataVisualization", "qtpy.QtDataVisualization", "PySide6.QtNetwork",
    # matplotlib backends that are not Agg (pyvista needs matplotlib, not a GUI backend)
    "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_qt",
    "matplotlib.backends.backend_qt5agg", "matplotlib.backends.backend_qt5",
    "matplotlib.backends.backend_tkagg", "matplotlib.backends.backend_tkcairo",
    "matplotlib.backends.backend_wx", "matplotlib.backends.backend_wxagg",
    "matplotlib.backends.backend_gtk3agg", "matplotlib.backends.backend_gtk4agg",
    "matplotlib.backends.backend_webagg", "matplotlib.backends.backend_nbagg",
    "matplotlib.backends.backend_macosx",
    # dev / build tools
    "pytest", "_pytest", "pytestqt", "py", "ruff", "pip", "PyInstaller", "sphinx", "docutils",
    "setuptools._distutils.tests", "lib2to3", "pydoc_data",
]

#: Must be absent from the finished bundle (checked by build.py and the self-test).
FORBIDDEN_IN_BUNDLE = ("torch", "torchcor", "scipy", "pandas", "tkinter")

#: VTK modules that pyvista maps but the app never needs: readers, importers and
#: exporters of file formats it does not open (results come as the app's own .npz
#: bundles). Leaving them out drops HDF5, NetCDF, CGNS, Exodus, XDMF, libxml2 and gl2ps.
#: A module another kept module depends on is still collected by its hook.
VTK_SKIP = frozenset({
    "vtkIOCGNSReader", "vtkIOChemistry", "vtkDomainsChemistry", "vtkIOEnSight", "vtkIOExodus",
    "vtkIOParallelExodus", "vtkIOFLUENTCFF", "vtkIOHDF", "vtkIOMINC", "vtkIOSegY", "vtkIOXdmf2",
    "vtkIOInfovis", "vtkIOExportGL2PS", "vtkIOImport", "vtkIOParallel",
})

#: Qt plugins the app never uses, and the Qt libraries that only they need (~20 MB):
#: the on-screen virtual keyboard input method (pulls in QtQuick and the QML engine) and
#: the PDF image format (QtPdf). Matched against bundle paths on every platform.
QT_SKIP = re.compile(
    r"""(?:^|/)plugins/platforminputcontexts/[^/]*virtualkeyboard[^/]*$
      | (?:^|/)plugins/imageformats/[^/]*qpdf[^/]*$
      | (?:^|/)(?:lib)?Qt6?(?:Pdf|Quick|Qml|QmlModels|QmlMeta|QmlWorkerScript
                           |VirtualKeyboard|VirtualKeyboardQml|VirtualKeyboardSettings)
        (?:\.framework(?:/|$)|\.dll$|\.so(?:\.|$)|\.dylib$|$)""",
    re.X | re.I,
)


def version() -> str:
    text = (PKG / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not match:
        raise RuntimeError("__version__ not found in src/torchcor_studio/__init__.py")
    return match.group(1)


def description() -> str:
    """The one-sentence description in ``pyproject.toml``."""
    import tomllib

    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["description"]


def version_tuple(value: str | None = None) -> tuple[int, int, int, int]:
    """``"1.2.3rc1"`` -> ``(1, 2, 3, 0)`` (Windows file versions are four integers)."""
    numbers = [int(n) for n in re.findall(r"\d+", value or version())[:4]]
    return tuple([*numbers, 0, 0, 0, 0][:4])  # type: ignore[return-value]


def exe_name() -> str:
    return LINUX_NAME if IS_LINUX else APP_NAME


def arch() -> str:
    machine = platform.machine().lower()
    if IS_WIN:
        return {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)
    return {"amd64": "x86_64", "arm64": "arm64" if IS_MAC else "aarch64"}.get(machine, machine)


def os_tag() -> str:
    return "macos" if IS_MAC else "windows" if IS_WIN else "linux"


def artifact_stem() -> str:
    """``TorchCor-Lighting-0.1.0-macos-arm64`` (extension added by the archiver)."""
    return f"TorchCor-Lighting-{version()}-{os_tag()}-{arch()}"


def bundle_layout(dist: Path) -> dict[str, Path]:
    """Where PyInstaller puts things inside ``dist``.

    ``bundle``  what gets shipped (the .app on macOS, the folder elsewhere)
    ``exe``     the program to launch
    ``data``    the folder that holds ``torchcor_studio/`` data files
    """
    if IS_MAC:
        app = dist / f"{APP_NAME}.app"
        return {"bundle": app, "exe": app / "Contents" / "MacOS" / APP_NAME,
                "data": app / "Contents" / "Resources", "folder": dist / APP_NAME}
    folder = dist / exe_name()
    exe = folder / (f"{APP_NAME}.exe" if IS_WIN else LINUX_NAME)
    return {"bundle": folder, "exe": exe, "data": folder / "_internal", "folder": folder}


def runtime_files() -> list[Path]:
    """The plain files copied to compute machines (``runtime/*.py`` except ``__init__``)."""
    names = _runtime_names()
    return [RUNTIME / name for name in names]


def _runtime_names() -> list[str]:
    """``RUNTIME_FILES`` from ``runtime/__init__.py`` (parsed, not imported), plus any
    other ``runtime/*.py`` so a newly added runtime file is never left behind."""
    names: list[str] = []
    init = RUNTIME / "__init__.py"
    tree = ast.parse(init.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(isinstance(t, ast.Name) and t.id == "RUNTIME_FILES" for t in targets):
            value = ast.literal_eval(node.value)
            names.extend(value)
    names.extend(p.name for p in sorted(RUNTIME.glob("*.py")) if p.name != "__init__.py")
    return list(dict.fromkeys(names))


#: Package files that are not needed at run time (platform packaging assets).
_SKIP_DATA_SUFFIXES = {".py", ".pyc", ".pyo", ".pyi", ".pyd", ".so", ".icns", ".ico", ".desktop"}
_SKIP_DATA_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini", "py.typed"}


def package_datas() -> list[tuple[str, str]]:
    """``(source, dest_dir)`` pairs: every non-Python file of the package (resources,
    presets, ...) plus the runtime scripts, which must exist as real files on disk."""
    datas: list[tuple[str, str]] = []
    for path in sorted(PKG.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in _SKIP_DATA_SUFFIXES or path.name in _SKIP_DATA_NAMES:
            continue
        rel = path.relative_to(SRC).parent
        datas.append((str(path), rel.as_posix()))
    for path in runtime_files():
        datas.append((str(path), path.relative_to(SRC).parent.as_posix()))
    return datas


def app_modules() -> list[str]:
    """Every module of ``torchcor_studio``, found by walking the source tree.

    Unlike ``collect_submodules`` this also finds modules in folders without an
    ``__init__.py`` (implicit namespace packages) and needs no import.
    """
    names: set[str] = set()
    for path in sorted(PKG.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__main__":
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.add(".".join(parts))
        # parents of a module in a namespace folder are importable packages too
        for i in range(1, len(parts)):
            names.add(".".join(parts[:i]))
    return sorted(names)


def _package_dir(name: str) -> Path | None:
    """Folder of an installed top-level package, without importing it."""
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def pyvista_vtk_modules() -> list[str]:
    """``vtkmodules.*`` modules that pyvista / pyvistaqt load *by name*.

    pyvista >= 0.48 imports VTK lazily (``importlib.import_module`` driven by a name
    table in ``pyvista/_vtk.py`` and f-strings in ``plotting/_rendering_imports.py``),
    which PyInstaller cannot see. Every VTK module mentioned anywhere in their sources
    is collected; the per-module ``vtkmodules`` hooks then add each module's compiled
    dependencies. Over-inclusion is limited to modules pyvista itself references,
    instead of the whole of VTK (``vtkmodules.all``, ~500 MB).
    """
    vtk_dir = _package_dir("vtkmodules")
    if vtk_dir is None:
        return []
    compiled = {f.name.split(".", 1)[0] for f in vtk_dir.iterdir()
                if f.suffix in {".so", ".pyd"} and f.name.startswith("vtk")}
    sources: list[Path] = []
    for pkg in ("pyvista", "pyvistaqt"):
        pkg_dir = _package_dir(pkg)
        if pkg_dir is not None:
            sources += [p for p in pkg_dir.rglob("*.py")
                        if not ({"examples", "demos", "tests", "ext", "trame"} & set(p.parts))
                        and p.name != "conftest.py"]
    module_name = re.compile(r"\bvtk[A-Z][A-Za-z0-9]+")
    helper_name = re.compile(r"""["'](numpy_interface|util)\.([A-Za-z_][A-Za-z0-9_]*)["']""")
    wanted: set[str] = set()
    helpers: set[str] = {"util.numpy_support", "util.vtkAlgorithm"}
    for src in sources:
        text = src.read_text(encoding="utf-8", errors="ignore")
        wanted.update(module_name.findall(text))
        helpers.update(f"{a}.{b}" for a, b in helper_name.findall(text))
    modules = [f"vtkmodules.{m}" for m in sorted((wanted & compiled) - VTK_SKIP)]
    # pure-Python helpers mapped by dotted name (numpy_interface.*, util.*), if present
    modules += [f"vtkmodules.{h}" for h in sorted(helpers)
                if (vtk_dir / (h.replace(".", "/") + ".py")).is_file()]
    return modules


def _wheel_macos_version(dist, arch: str) -> tuple[int, int] | None:
    """The macOS version in a distribution's wheel tag for ``arch`` (None: not a wheel)."""
    text = dist.read_text("WHEEL") or ""
    found = []
    for tag in re.findall(r"^Tag:\s*(\S+)", text, re.M):
        for plat in tag.split("-")[-1].split("."):
            m = re.fullmatch(r"macosx_(\d+)_(\d+)_(\w+)", plat)
            if m and m.group(3) in {arch, "universal2", "universal"}:
                found.append((int(m.group(1)), int(m.group(2))))
    return min(found) if found else None


def macos_minimum(binaries, floor: tuple[int, int] = MACOS_MIN_FLOOR) -> tuple[tuple[int, int], list[str]]:
    """``LSMinimumSystemVersion`` for the collected ``binaries`` (PyInstaller TOC).

    The newest requirement among them, where a binary that comes from a wheel counts
    with that wheel's macOS platform tag -- the publisher's declared compatibility,
    which is what pip checked -- and any other binary with its Mach-O ``minos``.
    (dyld does not enforce a library's ``minos``, and some wheels, e.g. PySide6 6.12,
    ship bindings whose ``minos`` is newer than their tag.) Returns the version and
    notes about binaries whose ``minos`` exceeds their wheel tag.
    """
    import importlib.metadata
    import os

    from PyInstaller.utils import osx

    arch = platform.machine()
    owner: dict[str, tuple[int, int] | None] = {}
    for dist in importlib.metadata.distributions():
        version_ = _wheel_macos_version(dist, arch)
        if version_ is None:
            continue
        for f in dist.files or ():
            if f.suffix in {".so", ".dylib", ""}:
                owner[os.path.realpath(dist.locate_file(f))] = version_
    best, notes = tuple(floor), []
    for _dest, src, _kind in binaries:
        real = os.path.realpath(src)
        try:
            minos = tuple(osx.macosx_version_min(real)[:2])
        except Exception:
            continue
        declared = owner.get(real)
        value = min(minos, declared) if declared else minos
        if value > best:
            best = value
        if declared and minos > declared:
            notes.append(f"{Path(src).name}: minos {minos[0]}.{minos[1]} > wheel tag {declared[0]}.{declared[1]}")
    return best, notes  # type: ignore[return-value]
