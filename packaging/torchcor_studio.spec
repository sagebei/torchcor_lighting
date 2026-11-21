# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for TorchCor Lighting: one folder, windowed.

Use ``python packaging/build.py`` (icons, smoke test and release archive included),
or run PyInstaller directly from the repository root:

    pyinstaller --noconfirm packaging/torchcor_studio.spec

Output in ``dist/``:

* macOS    ``TorchCor Lighting.app`` (bundle id ``org.torchcor.studio``)
* Windows  ``TorchCor Lighting/TorchCor Lighting.exe``
* Linux    ``torchcor-lighting/torchcor-lighting``

Notes
-----
* torch / torchcor are never bundled: simulations run in a separate Python
  environment on the compute machine (see ``appinfo.EXCLUDES``).
* ``runtime/tcs_runtime.py`` and ``runtime/tcs_studio.py`` are copied to compute
  machines by the app, so they ship as real files (data), not only as compiled modules.
* pyvista >= 0.48 imports VTK modules by name; ``appinfo.pyvista_vtk_modules()`` lists
  them so only the needed part of VTK is collected (instead of ``vtkmodules.all``),
  minus readers of file formats the app never opens (``appinfo.VTK_SKIP``).
* Qt plugins the app cannot use (virtual keyboard, PDF images) and the Qt libraries
  only they need are left out (``appinfo.QT_SKIP``); build.py verifies on macOS that no
  bundled binary links to anything missing.
* Optional macOS signing: set ``TCS_CODESIGN_IDENTITY`` (and ``TCS_ENTITLEMENTS``);
  otherwise binaries are ad-hoc signed.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

sys.path.insert(0, SPECPATH)  # noqa: F821 - SPECPATH is injected by PyInstaller
import appinfo  # noqa: E402

VERSION = appinfo.version()
EXE_NAME = appinfo.exe_name()
ICNS = appinfo.RESOURCES / "app_icon.icns"
ICO = appinfo.RESOURCES / "app_icon.ico"

# ----------------------------------------------------------------------------- inputs
datas = appinfo.package_datas()
# keyring finds its OS backends through entry points, i.e. its dist-info metadata
# (PyInstaller collects metadata read with importlib.metadata.version("x") by itself).
datas += copy_metadata("keyring")

hiddenimports = appinfo.app_modules()  # everything, incl. modules imported lazily in functions
try:  # plus PyInstaller's own submodule discovery (duplicates are harmless)
    hiddenimports += collect_submodules("torchcor_studio", filter=lambda name: not name.endswith("__main__"))
except Exception as exc:  # pragma: no cover - the package is not importable in the build env
    print(f"collect_submodules(torchcor_studio) failed: {exc}")
hiddenimports += appinfo.pyvista_vtk_modules()
hiddenimports += [
    "pyvistaqt",
    "qtpy.QtOpenGLWidgets",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtSvg",
]
hiddenimports += collect_submodules("paramiko")
hiddenimports += collect_submodules("keyring.backends")
if appinfo.IS_WIN:
    hiddenimports += collect_submodules("win32ctypes")
if appinfo.IS_LINUX:
    for pkg in ("secretstorage", "jeepney"):
        try:
            hiddenimports += collect_submodules(pkg, filter=lambda n: not n.startswith(("jeepney.io.trio", "jeepney.tests")))
        except Exception:
            print(f"warning: {pkg} not installed; the Secret Service keyring backend will be unavailable")
try:
    import importlib.util

    if importlib.util.find_spec("pynvml"):  # optional extra [nvidia]: local NVIDIA telemetry
        hiddenimports.append("pynvml")
except Exception:
    pass

a = Analysis(
    [str(appinfo.HERE / "launcher.py")],
    pathex=[str(appinfo.SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        # pyvista imports matplotlib (colormaps, charts); the app draws with Qt, so the
        # headless Agg backend is all that is needed.
        "matplotlib": {"backends": ["Agg"]},
    },
    runtime_hooks=[],
    excludes=appinfo.EXCLUDES,
    noarchive=False,
    optimize=0,
)

# ----------------------------------------------------------------------------- trim & check
_forbidden = [name for name, *_ in a.pure if name.split(".")[0] in appinfo.FORBIDDEN_IN_BUNDLE]
if _forbidden:
    raise SystemExit(f"refusing to bundle {sorted({n.split('.')[0] for n in _forbidden})}: "
                     "check appinfo.EXCLUDES")


def _drop(toc, label):
    """Remove the Qt plugins/libraries matched by ``appinfo.QT_SKIP`` from a TOC."""
    kept, dropped = [], []
    for entry in toc:
        dest = entry[0].replace("\\", "/")
        if appinfo.QT_SKIP.search(dest) or Path(dest).name in {".DS_Store", "Thumbs.db"}:
            dropped.append(Path(dest).name)
            continue
        kept.append(entry)
    if dropped:
        print(f"torchcor_studio.spec: left out {len(dropped)} unused Qt {label}: "
              f"{', '.join(sorted(set(dropped))[:12])}{' ...' if len(set(dropped)) > 12 else ''}")
    return kept


a.binaries = _drop(a.binaries, "plugins/libraries")
a.datas = _drop(a.datas, "data files")


# ----------------------------------------------------------------------------- build
pyz = PYZ(a.pure)

exe_options = {}
if appinfo.IS_WIN:
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo,
        StringFileInfo,
        StringStruct,
        StringTable,
        VarFileInfo,
        VarStruct,
        VSVersionInfo,
    )

    numbers = appinfo.version_tuple(VERSION)
    exe_options["version"] = VSVersionInfo(
        ffi=FixedFileInfo(filevers=numbers, prodvers=numbers, mask=0x3F, flags=0x0, OS=0x40004,
                          fileType=0x1, subtype=0x0, date=(0, 0)),
        kids=[
            StringFileInfo([StringTable("040904B0", [
                StringStruct("CompanyName", "TorchCor"),
                StringStruct("FileDescription", appinfo.APP_NAME),
                StringStruct("FileVersion", VERSION),
                StringStruct("InternalName", "torchcor-lighting"),
                StringStruct("LegalCopyright", appinfo.COPYRIGHT),
                StringStruct("OriginalFilename", f"{EXE_NAME}.exe"),
                StringStruct("ProductName", appinfo.APP_NAME),
                StringStruct("ProductVersion", VERSION),
            ])]),
            VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
        ],
    )
    exe_options["icon"] = str(ICO)
elif appinfo.IS_MAC:
    exe_options["icon"] = str(ICNS)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=EXE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=os.environ.get("TCS_CODESIGN_IDENTITY") or None,
    entitlements_file=os.environ.get("TCS_ENTITLEMENTS") or None,
    contents_directory="_internal",
    **exe_options,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=EXE_NAME,
)

if appinfo.IS_MAC:
    minimum, notes = appinfo.macos_minimum(a.binaries)
    print(f"torchcor_studio.spec: LSMinimumSystemVersion {minimum[0]}.{minimum[1]}")
    if notes:
        print(f"torchcor_studio.spec: {len(notes)} binaries declare a newer minos than their wheel tag, "
              f"e.g. {notes[0]} (dyld does not enforce it; the wheel tags are used)")
    app = BUNDLE(
        coll,
        name=f"{appinfo.APP_NAME}.app",
        icon=str(ICNS),
        bundle_identifier=appinfo.BUNDLE_ID,
        version=VERSION,
        info_plist={
            "CFBundleName": appinfo.APP_NAME,
            "CFBundleDisplayName": appinfo.APP_NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "CFBundleDevelopmentRegion": "en",
            "LSMinimumSystemVersion": "{}.{}".format(*minimum),
            "LSApplicationCategoryType": "public.app-category.education",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,  # follow the system dark mode
            "NSHumanReadableCopyright": appinfo.COPYRIGHT,
            # macOS 15+ asks before an app talks to machines on the local network (SSH).
            "NSLocalNetworkUsageDescription":
                "TorchCor Lighting connects to your compute machines over SSH to run simulations.",
        },
    )
