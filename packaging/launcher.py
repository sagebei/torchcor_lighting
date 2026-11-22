"""Entry script of the frozen TorchCor Lighting app (PyInstaller).

Starts ``torchcor_studio.app.main()`` like the ``torchcor-lighting`` console script,
with three additions that only matter in a frozen build:

* **Child-process hygiene.** PyInstaller may put the bundle's private library folder
  on ``LD_LIBRARY_PATH`` / ``DYLD_LIBRARY_PATH`` so that *this* process finds its
  libraries. Inherited by the Python environments and tools the app launches (local
  simulations, ``ssh``, ``xdg-open``), it would make them load our libssl / libstdc++
  / libpython. The dynamic loader read those variables when this process started, so
  restoring them here only affects children.
* **A persistent matplotlib cache.** PyInstaller points ``MPLCONFIGDIR`` at a new
  temporary folder on every launch, so matplotlib (which pyvista imports) would rescan
  the system fonts each time a results view opens. It goes to the app's cache instead.
* **A self-test** (``TCS_SMOKE_TEST=1``), used by ``packaging/build.py`` and CI to check
  a bundle without a display: it imports every module of the app and exercises what
  PyInstaller is most likely to miss -- VTK modules that pyvista loads by name, keyring
  backends found through entry points, Qt image/SVG plugins, cryptography/OpenSSL,
  the runtime files deployed to compute machines -- then prints a report and exits
  0 (all good) or 1. Options (environment variables):

  ``TCS_SMOKE_RENDER=0``            skip the off-screen VTK render (CI without OpenGL)
  ``TCS_SMOKE_INTERACTIVE=1``       also render in a pyvistaqt ``QtInteractor`` window
                                    (needs a real display platform, not ``offscreen``)
  ``TCS_SMOKE_MODULES=<file>``      app modules to import, one per line (build.py)
  ``TCS_SMOKE_VTK_MODULES=<file>``  VTK modules the spec collected on purpose (build.py)
  ``TCS_SMOKE_BUNDLES=<dir>``       open a results view for every bundle in <dir>
                                    (build.py writes the test suite's synthetic ones)
  ``TCS_SMOKE_REPORT=<file>``       also write the report to a file

  ``TCS_SMOKE_GUI=1`` (used by build.py's launch check) starts the normal app but
  prints start-up exceptions and exits 3, instead of PyInstaller's error dialog.
"""

from __future__ import annotations

import os
import sys

FROZEN = bool(getattr(sys, "frozen", False))


# --------------------------------------------------------------------------- environment
def _inside(path: str, root: str) -> bool:
    try:
        path, root = os.path.realpath(path), os.path.realpath(root)
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def _restore_library_paths() -> None:
    bundle = getattr(sys, "_MEIPASS", None)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LIBPATH"):
        original = os.environ.pop(f"{var}_ORIG", None)
        if original is not None:
            os.environ[var] = original
            continue
        value = os.environ.get(var)
        if not value or not bundle:
            continue
        keep = [p for p in value.split(os.pathsep) if p and not _inside(p, bundle)]
        if keep:
            os.environ[var] = os.pathsep.join(keep)
        else:
            del os.environ[var]


def _matplotlib_cache() -> None:
    try:
        from torchcor_studio.core.paths import cache_dir

        folder = cache_dir() / "matplotlib"
        folder.mkdir(parents=True, exist_ok=True)
    except Exception:  # keep PyInstaller's temporary folder
        return
    os.environ["MPLCONFIGDIR"] = str(folder)


def _ensure_stdio() -> None:
    """Windowed Windows builds have no console: ``sys.stdout``/``sys.stderr`` are None,
    which breaks libraries that write or flush them. Send them to a log file instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        from torchcor_studio.core.paths import data_dir

        log_dir = data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stream = open(log_dir / "console.log", "w", encoding="utf-8", errors="replace", buffering=1)  # noqa: SIM115
    except Exception:
        stream = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def prepare_environment() -> None:
    os.environ.setdefault("QT_API", "pyside6")  # pyvistaqt/qtpy bind to PySide6
    os.environ.setdefault("MPLBACKEND", "Agg")  # the only matplotlib backend bundled
    if FROZEN:
        _ensure_stdio()
        _restore_library_paths()
        _matplotlib_cache()


# --------------------------------------------------------------------------- self-test
def _smoke_test() -> int:  # noqa: C901 - a flat list of independent checks
    import importlib
    import importlib.util
    import platform
    import time
    import traceback
    from pathlib import Path

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    results: list[tuple[str, bool, str, float]] = []
    state: dict = {}

    def check(name):
        def run(fn):
            started = time.perf_counter()
            try:
                detail = fn() or ""
                ok = True
            except Exception as exc:  # report every failure, keep going
                ok, detail = False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            results.append((name, ok, str(detail), time.perf_counter() - started))
            return fn
        return run

    @check("app modules")
    def _modules():
        listed = os.environ.get("TCS_SMOKE_MODULES")
        if listed and Path(listed).is_file():
            names = [n.strip() for n in Path(listed).read_text().splitlines() if n.strip()]
        else:
            import pkgutil

            import torchcor_studio

            names = ["torchcor_studio"] + [m.name for m in pkgutil.walk_packages(
                torchcor_studio.__path__, "torchcor_studio.", onerror=lambda _name: None)]
        names = [n for n in names if n.rpartition(".")[2] != "__main__"]  # would start the app
        failed = []
        for name in names:
            try:
                importlib.import_module(name)
            except Exception as exc:
                failed.append(f"{name}: {type(exc).__name__}: {exc}")
        if failed:
            raise ImportError(f"{len(failed)}/{len(names)} modules failed:\n  " + "\n  ".join(failed))
        return f"{len(names)} modules imported"

    @check("runtime files")
    def _runtime():
        from torchcor_studio.runtime import RUNTIME_FILES, runtime_path

        out = []
        for name in RUNTIME_FILES:
            path = runtime_path(name)
            data = path.read_bytes()  # must be a real file on disk
            compile(data, str(path), "exec")
            out.append(f"{name} ({len(data) // 1024} KB)")
        from torchcor_studio.core.machines.deploy import deploy_local
        from torchcor_studio.core.paths import cache_dir

        target = deploy_local(cache_dir() / "runtime-selftest")
        copied = sorted(p.name for p in target.iterdir())
        if any(name not in copied for name in RUNTIME_FILES):
            raise FileNotFoundError(f"deploy_local copied {copied}")
        return ", ".join(out) + f"; deploy_local -> {target.name}/"

    @check("resources")
    def _resources():
        from torchcor_studio.ui.resources import resource

        names = ("app_icon.png", "logo_mark.png", "logo_full.png")
        missing = [n for n in names if not resource(n).is_file()]
        if missing:
            raise FileNotFoundError(f"missing {missing} in {resource('')}")
        return f"{len(names)} images in {resource('').parent.name}/{resource('').name}"

    @check("qt")
    def _qt():
        from PySide6.QtCore import QByteArray
        from PySide6.QtGui import QImageReader, QPixmap
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtWidgets import QApplication

        from torchcor_studio.ui.resources import app_icon, resource

        app = QApplication.instance() or QApplication([sys.argv[0]])
        state["app"] = app
        if QPixmap(str(resource("app_icon.png"))).isNull():
            raise RuntimeError("app_icon.png did not load (PNG image plugin missing?)")
        if app_icon().isNull():
            raise RuntimeError("app icon is empty")
        svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8"/></svg>'
        if not QSvgRenderer(QByteArray(svg)).isValid():
            raise RuntimeError("QtSvg cannot render")
        formats = {bytes(f).decode() for f in QImageReader.supportedImageFormats()}
        if not {"png", "svg", "jpg"} <= formats:
            raise RuntimeError(f"image formats: {sorted(formats)}")
        from torchcor_studio.ui.theme import theme

        theme().set_mode("dark")  # writes tinted SVGs into the cache, builds the QSS
        return f"platform={app.platformName()}, image formats: {len(formats)}"

    @check("qt opengl")
    def _qt_opengl():
        from PySide6.QtGui import QOffscreenSurface, QOpenGLContext
        from PySide6.QtOpenGLWidgets import QOpenGLWidget

        app = state["app"]
        QOpenGLWidget().deleteLater()  # the bindings pyvistaqt's QtInteractor is built on
        context = QOpenGLContext()
        if not context.create():
            if app.platformName() in {"offscreen", "minimal"}:
                return f"no OpenGL on the {app.platformName()} platform (the viewer falls back to off-screen VTK)"
            raise RuntimeError(f"cannot create an OpenGL context on the {app.platformName()} platform")
        surface = QOffscreenSurface()
        surface.setFormat(context.format())
        surface.create()
        if not context.makeCurrent(surface):
            raise RuntimeError("cannot make the OpenGL context current")
        fmt = context.format()
        detail = f"OpenGL {fmt.majorVersion()}.{fmt.minorVersion()} on {app.platformName()}"
        context.doneCurrent()
        if os.environ.get("TCS_SMOKE_INTERACTIVE") == "1":  # opens a small window briefly
            import numpy as np
            import pyvista as pv
            from pyvistaqt import QtInteractor

            widget = QtInteractor(None, auto_update=False)
            widget.add_mesh(pv.Sphere(), color="orange")
            widget.resize(240, 180)
            widget.show()
            for _ in range(30):
                app.processEvents()
                time.sleep(0.03)
            widget.render()
            image = widget.screenshot(return_img=True)
            widget.close()
            if image is None or int(np.ptp(image)) == 0:
                raise RuntimeError("QtInteractor rendered a blank image")
            detail += f"; QtInteractor rendered {image.shape[1]}x{image.shape[0]} px"
        return detail

    @check("pyqtgraph")
    def _pyqtgraph():
        import pyqtgraph as pg

        from torchcor_studio.ui.results.colormaps import pg_colormap

        widget = pg.PlotWidget()
        widget.plot([0, 1, 2], [0, 1, 0])
        pg_colormap("turbo")
        widget.deleteLater()
        return f"pyqtgraph {pg.__version__}"

    @check("vtk modules")
    def _vtk():
        import pyvista as pv

        # Strict: what the spec collected on purpose (build.py passes the list), else the
        # modules any rendering needs. Informational: the rest of pyvista's name table
        # (file readers for formats the app never opens are left out on purpose).
        essential = {"vtkCommonCore", "vtkCommonDataModel", "vtkCommonExecutionModel", "vtkFiltersCore",
                     "vtkFiltersGeneral", "vtkFiltersGeometry", "vtkRenderingCore", "vtkRenderingOpenGL2",
                     "vtkRenderingUI", "vtkInteractionStyle", "vtkRenderingFreeType",
                     "vtkRenderingContextOpenGL2", "vtkRenderingAnnotation", "vtkIOImage", "vtkIOXML"}
        listed = os.environ.get("TCS_SMOKE_VTK_MODULES")
        if listed and Path(listed).is_file():
            essential |= {n.strip().removeprefix("vtkmodules.") for n in Path(listed).read_text().splitlines()
                          if n.strip()}
        mapped: set[str] = set()
        try:
            from pyvista import _vtk as table  # pyvista >= 0.48: name-driven lazy imports

            for attr in dir(table):
                value = getattr(table, attr, None) if attr.endswith("_MODULES") else None
                if isinstance(value, dict):
                    mapped.update(value)
        except ImportError:
            pass

        def importable(name: str) -> bool:
            try:
                importlib.import_module(f"vtkmodules.{name}")
            except ImportError:
                return False
            return True

        missing = sorted(n for n in essential if not importable(n))
        optional = sorted(n for n in mapped - essential if not importable(n))
        import pyvista.plotting  # noqa: F401 - registers the rendering factories
        import pyvistaqt  # noqa: F401

        if missing:
            raise ImportError("VTK modules missing from the bundle: " + ", ".join(missing))
        note = f"; {len(optional)} optional pyvista readers not bundled" if optional else ""
        return f"pyvista {pv.__version__}, VTK {pv.vtk_version_info}, {len(essential)} modules{note}"

    @check("vtk render")
    def _render():
        if os.environ.get("TCS_SMOKE_RENDER", "1") == "0":
            return "skipped (TCS_SMOKE_RENDER=0)"
        import numpy as np
        import pyvista as pv

        from torchcor_studio.ui.results.colormaps import vtk_lookup_table

        mesh = pv.Sphere(theta_resolution=24, phi_resolution=24)
        mesh["vm"] = mesh.points[:, 2]
        plotter = pv.Plotter(off_screen=True, window_size=(160, 120))
        plotter.add_mesh(mesh, scalars="vm", cmap=vtk_lookup_table("torchcor", (-0.5, 0.5)),
                         show_scalar_bar=False)
        plotter.add_text("TorchCor", font_size=10)  # FreeType + bundled fonts
        contours = mesh.contour(isosurfaces=5, scalars="vm")
        image = plotter.screenshot(return_img=True)
        plotter.close()
        if image is None or image.shape[:2] != (120, 160) or int(np.ptp(image)) == 0:
            raise RuntimeError(f"blank render: {None if image is None else image.shape}")
        return f"{image.shape[1]}x{image.shape[0]} px, {contours.n_points} isoline points"

    @check("results views")
    def _results():
        folder = os.environ.get("TCS_SMOKE_BUNDLES")
        if not folder or not Path(folder).is_dir():
            return "skipped (no TCS_SMOKE_BUNDLES)"
        if os.environ.get("TCS_SMOKE_RENDER", "1") == "0":
            os.environ["TCS_DISABLE_3D"] = "1"  # the viewer's own switch: no OpenGL here
        from torchcor_studio.ui.results import create_results_view

        app = state["app"]
        errors: list[str] = []
        opened: list[str] = []
        previous = sys.excepthook  # exceptions raised in Qt slots end up here
        sys.excepthook = lambda kind, exc, tb: errors.append("".join(traceback.format_exception(kind, exc, tb)))
        try:
            for bundle in sorted(p for p in Path(folder).iterdir() if (p / "index.json").is_file()):
                view = create_results_view(bundle)
                view.resize(1100, 720)
                view.show()
                deadline = time.monotonic() + 120
                while view.loading and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.01)
                for _ in range(30):  # let the first frames paint
                    app.processEvents()
                    time.sleep(0.01)
                if view.loading:
                    errors.append(f"{bundle.name}: still loading after 120 s")
                elif view.error:
                    errors.append(f"{bundle.name}: {view.error}")
                else:
                    opened.append(f"{bundle.name} ({type(getattr(view, 'view', None)).__name__})")
                view.release()
                view.close()
                view.deleteLater()
                app.processEvents()
        finally:
            sys.excepthook = previous
        if errors:
            raise RuntimeError("\n".join(errors))
        return ", ".join(opened)

    @check("keyring")
    def _keyring():
        import keyring
        from keyring.backends import fail

        backend = keyring.get_keyring()
        name = f"{type(backend).__module__}.{type(backend).__name__}"
        expected = {"darwin": "keyring.backends.macOS", "win32": "keyring.backends.Windows"}.get(sys.platform)
        if expected and not name.startswith(expected):
            raise RuntimeError(f"expected the {expected} backend, got {name}")
        if sys.platform.startswith("linux"):
            importlib.import_module("keyring.backends.SecretService")
            importlib.import_module("secretstorage")
        if isinstance(backend, fail.Keyring) and expected:
            raise RuntimeError("no usable keyring backend")
        return name

    @check("ssh crypto")
    def _ssh():
        import io

        import bcrypt
        import paramiko
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519

        # an encrypted OpenSSH ed25519 key: cryptography (OpenSSL), bcrypt KDF, PyNaCl
        pem = ed25519.Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH,
            serialization.BestAvailableEncryption(b"selftest"))
        key = paramiko.Ed25519Key.from_private_key(io.StringIO(pem.decode()), password="selftest")
        key.sign_ssh_data(b"torchcor")
        rsa = paramiko.RSAKey.generate(2048)
        rsa.sign_ssh_data(b"torchcor", "rsa-sha2-256")
        bcrypt.kdf(b"x", b"salt", 16, 4, ignore_few_rounds=True)
        from cryptography.hazmat.backends.openssl.backend import backend as ossl

        return f"paramiko {paramiko.__version__}, {ossl.openssl_version_text()}"

    @check("support libraries")
    def _support():
        import numpy as np
        import platformdirs
        import psutil
        import pydantic

        class Probe(pydantic.BaseModel):
            dt: float = 0.01

        Probe.model_validate({"dt": "0.02"})
        psutil.virtual_memory()
        np.linalg.norm(np.ones(4))
        platformdirs.user_cache_dir("probe")
        nvml = "bundled" if importlib.util.find_spec("pynvml") else "absent"
        return f"numpy {np.__version__}, pydantic {pydantic.VERSION}, psutil {psutil.__version__}, pynvml {nvml}"

    @check("excluded")
    def _excluded():
        if not FROZEN:
            return "skipped (not frozen)"
        present = [n for n in ("torch", "torchcor", "scipy", "pandas", "tkinter")
                   if importlib.util.find_spec(n) is not None]
        if present:
            raise RuntimeError(f"must not be bundled: {present}")
        return "torch, torchcor, scipy, pandas, tkinter absent"

    from torchcor_studio import __version__

    lines = [f"TorchCor Lighting {__version__} self-test ({'frozen' if FROZEN else 'source'}, "
             f"{sys.platform} {platform.machine()}, Python {platform.python_version()})"]
    for name, ok, detail, seconds in results:
        first, *rest = detail.splitlines() or [""]
        lines.append(f"  {'ok  ' if ok else 'FAIL'}  {name:<18} {first} [{seconds:.1f}s]")
        lines.extend(f"        {line}" for line in rest)
    failed = [name for name, ok, _, _ in results if not ok]
    lines.append("self-test passed" if not failed else f"self-test FAILED: {', '.join(failed)}")
    report = "\n".join(lines)
    print(report, flush=True)
    target = os.environ.get("TCS_SMOKE_REPORT")
    if target:
        Path(target).write_text(report + "\n", encoding="utf-8")
    return 1 if failed else 0


# --------------------------------------------------------------------------- main
def main() -> int:
    import multiprocessing

    multiprocessing.freeze_support()  # no-op unless a library spawns worker processes
    prepare_environment()
    from torchcor_studio.app import leave
    from torchcor_studio.app import main as app_main

    if os.environ.get("TCS_SMOKE_TEST"):
        leave(_smoke_test())  # no interpreter teardown, as the app quits

    if os.environ.get("TCS_SMOKE_GUI"):  # build.py: report start-up errors instead of a dialog
        try:
            return app_main()
        except BaseException:
            import traceback

            traceback.print_exc()
            return 3
    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
