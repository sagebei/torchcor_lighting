#!/usr/bin/env python3
"""Build the TorchCor Lighting desktop app (PyInstaller, one folder) and check it.

    python packaging/build.py                # icons if missing -> PyInstaller -> checks
    python packaging/build.py --clean        # from scratch (PyInstaller cache, build/, old bundle)
    python packaging/build.py --archive      # ... and the release artifact in dist/

Run it with the Python environment that has the app's dependencies and PyInstaller
(``pip install -e ".[dev]"``) -- never one you would mind seeing frozen: torch and
torchcor are refused, everything else that the app imports is bundled.

Release artifacts (``dist/``, each with a ``.sha256`` file):

* macOS    ``TorchCor-Lighting-<version>-macos-<arch>.dmg``  (zip when hdiutil fails)
* Windows  ``TorchCor-Lighting-<version>-windows-<arch>.zip``
* Linux    ``TorchCor-Lighting-<version>-linux-<arch>.tar.gz`` (with the .desktop file,
  the icon and ``install.sh``, which adds a menu entry for the current user)

Checks after building:

1. layout -- the runtime files that the app copies to compute machines exist in the
   bundle as real files identical to the sources; images are there; torch & co. are not.
2. self-test (``--no-run`` skips 2 and 3) -- the frozen app runs with
   ``TCS_SMOKE_TEST=1`` on the offscreen Qt platform with throwaway settings: it
   imports every module of the app, renders with VTK off-screen (``--no-render`` skips
   that on machines without OpenGL) and exercises keyring, SSH crypto and Qt plugins.
3. launch -- the real GUI starts offscreen and must still be running, without
   errors, after ``--launch-seconds``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import plistlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import appinfo  # noqa: E402

SPEC = appinfo.HERE / "torchcor_studio.spec"
DIST = appinfo.ROOT / "dist"
WORK = appinfo.ROOT / "build" / "pyinstaller"
LINUX_DIR = appinfo.HERE / "linux"
ERROR_MARKERS = ("Traceback (most recent call last)", "ModuleNotFoundError", "ImportError:",
                 "FileNotFoundError", "Failed to execute script", "Unhandled exception")


class BuildError(RuntimeError):
    pass


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def rel(path: Path) -> Path:
    """``path`` relative to the repository when it is inside it (for messages)."""
    try:
        return path.relative_to(appinfo.ROOT)
    except ValueError:
        return path


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def tree_size(path: Path) -> int:
    """Bytes of all regular files under ``path`` (symlinks are not followed)."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            p = os.path.join(root, name)
            if not os.path.islink(p):
                total += os.lstat(p).st_size
    return total


def layout() -> dict[str, Path]:
    """``bundle`` (shipped), ``exe``, ``meipass`` (= sys._MEIPASS) and ``folder``."""
    paths = appinfo.bundle_layout(DIST)
    if appinfo.IS_MAC:
        paths["meipass"] = paths["bundle"] / "Contents" / "Frameworks"
    else:
        paths["meipass"] = paths["data"]
    return paths


# ----------------------------------------------------------------------------- build
def ensure_icons() -> None:
    import make_icons

    missing = [p for p in (make_icons.ICNS, make_icons.ICO) if not p.is_file()]
    if missing:
        step("Icons")
        if make_icons.main([]) != 0:
            raise BuildError("icon generation failed")


def clean(paths: dict[str, Path]) -> None:
    step("Clean")
    for path in {WORK, paths["bundle"], paths["folder"]}:
        if path.exists():
            print(f"removing {rel(path)}")
            shutil.rmtree(path)


def run_pyinstaller(clean_cache: bool) -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:
        raise BuildError('PyInstaller is not installed: pip install -e ".[dev]" (or pip install pyinstaller)') from exc
    step("PyInstaller")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm",
           "--distpath", str(DIST), "--workpath", str(WORK), str(SPEC)]
    if clean_cache:
        cmd.insert(4, "--clean")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    started = time.monotonic()
    result = subprocess.run(cmd, cwd=appinfo.ROOT, check=False)
    if result.returncode != 0:
        raise BuildError(f"PyInstaller failed (exit code {result.returncode})")
    print(f"PyInstaller finished in {time.monotonic() - started:.0f} s")


# ----------------------------------------------------------------------------- checks
def check_layout(paths: dict[str, Path], fresh: bool = True) -> None:
    step("Check bundle layout")
    exe, meipass = paths["exe"], paths["meipass"]
    if not exe.is_file():
        raise BuildError(f"executable missing: {exe}")
    problems = []
    for source in appinfo.runtime_files():
        bundled = meipass / "torchcor_studio" / "runtime" / source.name  # what runtime_path() opens
        if not bundled.is_file():
            problems.append(f"runtime file missing: {bundled}")
        elif bundled.read_bytes() != source.read_bytes():
            message = f"runtime file differs from source: {bundled}"
            if fresh:
                problems.append(message)
            else:
                print(f"WARN  {message} (source edited since the bundle was built)")
        else:
            real = bundled.resolve()
            where = real.relative_to(paths["bundle"]) if paths["bundle"] in real.parents else real
            print(f"ok    {source.name} -> {where}")
    for name in ("app_icon.png", "logo_mark.png", "logo_full.png"):
        if not (meipass / "torchcor_studio" / "resources" / name).is_file():
            problems.append(f"resource missing: torchcor_studio/resources/{name}")
    for name in appinfo.FORBIDDEN_IN_BUNDLE:
        if (meipass / name).exists():
            problems.append(f"{name}/ must not be bundled")
    if appinfo.IS_MAC:
        plist = plistlib.loads((paths["bundle"] / "Contents" / "Info.plist").read_bytes())
        print(f"ok    Info.plist: {plist['CFBundleIdentifier']} {plist['CFBundleShortVersionString']}, "
              f"LSMinimumSystemVersion {plist.get('LSMinimumSystemVersion')}")
        if shutil.which("codesign"):
            verify = subprocess.run(["codesign", "--verify", "--deep", "--strict", str(paths["bundle"])],
                                    capture_output=True, text=True, check=False)
            print("ok    code signature (ad-hoc unless TCS_CODESIGN_IDENTITY was set)" if verify.returncode == 0
                  else f"WARN  codesign --verify: {verify.stderr.strip()}")
        problems += _dangling_macos_links(paths)
    if problems:
        raise BuildError("bundle layout:\n  " + "\n  ".join(problems))
    print("ok    images present; torch, torchcor, scipy, pandas, tkinter absent")


def _dangling_macos_links(paths: dict[str, Path]) -> list[str]:
    """Every ``@rpath/...`` library a bundled binary links against must be in the bundle
    (PyInstaller points ``@rpath`` at Contents/Frameworks). Catches over-trimming."""
    frameworks = paths["meipass"]
    binaries = [p for p in frameworks.rglob("*") if p.is_file() and not p.is_symlink()
                and (p.suffix in {".so", ".dylib"} or (p.parent.name == "A" and p.parent.parent.name == "Versions"))]
    missing: dict[str, set[str]] = {}
    for chunk in range(0, len(binaries), 200):
        group = [str(p) for p in binaries[chunk:chunk + 200]]
        own_ids: dict[str, str] = {}  # a dylib's install name is listed by otool -L too
        current = ""
        for line in subprocess.run(["otool", "-D", *group], capture_output=True, text=True, check=False).stdout.splitlines():
            if line.endswith(":") and line[:-1] in group:
                current = line[:-1]
            elif current:
                own_ids[current] = line.strip()
        out = subprocess.run(["otool", "-L", *group], capture_output=True, text=True, check=False).stdout
        for line in out.splitlines():
            if not line.startswith("\t"):
                current = line.rstrip(":")
                continue
            dep = line.strip().split(" (")[0]
            if dep == own_ids.get(current):
                continue
            if dep.startswith("@rpath/") and not (frameworks / dep[len("@rpath/"):]).exists():
                missing.setdefault(dep, set()).add(Path(current).name)
    if not missing:
        print(f"ok    all @rpath libraries of {len(binaries)} binaries are bundled")
    return [f"{dep} (needed by {', '.join(sorted(users)[:3])})" for dep, users in sorted(missing.items())]


def _isolated_env(base: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "TCS_", "VIRTUAL_ENV"))}
    env.update({
        "TCS_CONFIG_DIR": str(base / "config"),
        "TCS_DATA_DIR": str(base / "data"),
        "TCS_CACHE_DIR": str(base / "cache"),
        "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", "offscreen"),
        "PYVISTA_OFF_SCREEN": "true",
        "PYTHONIOENCODING": "utf-8",
    })
    return env


def synthetic_bundles(base: Path) -> Path | None:
    """Small EP / ECG / mechanics / EM result bundles (the test suite's fixtures), so the
    self-test can open every results view inside the frozen app."""
    saved = list(sys.path)
    sys.path[:0] = [str(appinfo.SRC), str(appinfo.ROOT / "tests")]
    try:
        from fixtures import synthetic_bundles as fixtures

        folder = base / "bundles"
        for kind in ("ep", "ecg", "mechanics", "em"):
            getattr(fixtures, f"write_{kind}_bundle")(folder / kind)
        return folder
    except Exception as exc:  # the fixtures are a convenience; never block a build on them
        print(f"WARN  results views not checked: no synthetic bundles ({type(exc).__name__}: {exc})")
        return None
    finally:
        sys.path[:] = saved


def self_test(paths: dict[str, Path], base: Path, render: bool) -> None:
    step("Self-test of the frozen app")
    bundles = synthetic_bundles(base)
    modules = base / "modules.txt"
    modules.write_text("\n".join(appinfo.app_modules()) + "\n", encoding="utf-8")
    vtk_modules = base / "vtk-modules.txt"
    vtk_modules.write_text("\n".join(appinfo.pyvista_vtk_modules()) + "\n", encoding="utf-8")
    report = base / "self-test.txt"
    env = _isolated_env(base)
    env.update({"TCS_SMOKE_TEST": "1", "TCS_SMOKE_MODULES": str(modules), "TCS_SMOKE_REPORT": str(report),
                "TCS_SMOKE_VTK_MODULES": str(vtk_modules), "TCS_SMOKE_RENDER": "1" if render else "0"})
    if bundles:
        env["TCS_SMOKE_BUNDLES"] = str(bundles)
    try:
        result = subprocess.run([str(paths["exe"])], env=env, cwd=base, capture_output=True,
                                timeout=600, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BuildError("self-test timed out after 600 s") from exc
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    text = report.read_text(encoding="utf-8") if report.is_file() else ""
    print(text or output)
    if result.returncode != 0:
        if text:  # the report is above; show whatever else the process printed
            extra = output.replace(text.strip(), "").strip()
            if extra:
                print(extra[-4000:])
        raise BuildError(f"self-test failed (exit code {result.returncode})")


def launch_test(paths: dict[str, Path], base: Path, seconds: float) -> None:
    step(f"Launch the GUI for {seconds:.0f} s (QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', 'offscreen')})")
    env = _isolated_env(base)
    env["TCS_SMOKE_GUI"] = "1"
    log_path = base / "launch-output.txt"
    with open(log_path, "wb") as out:
        proc = subprocess.Popen([str(paths["exe"])], env=env, cwd=base, stdout=out, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.25)
        alive = proc.poll() is None
        if alive:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    output = log_path.read_text(encoding="utf-8", errors="replace")
    app_log = base / "data" / "logs" / "studio.log"
    log_text = app_log.read_text(encoding="utf-8", errors="replace") if app_log.is_file() else ""
    if not alive:
        print(output[-6000:])
        raise BuildError(f"the app exited after start-up (exit code {proc.returncode})")
    errors = [line for line in output.splitlines() if any(m in line for m in ERROR_MARKERS)]
    if errors:
        print(output[-6000:])
        raise BuildError("errors while the app was running:\n  " + "\n  ".join(errors[:20]))
    noisy = [line for line in output.splitlines() if line.strip()]
    print(f"ok    still running after {seconds:.0f} s; terminated")
    if noisy:
        print(f"      console output ({len(noisy)} lines):")
        for line in noisy[:15]:
            print(f"        {line[:200]}")
    log_errors = [line for line in log_text.splitlines() if " ERROR " in line or "Traceback" in line]
    if log_errors:
        print("WARN  errors in the app log (studio.log):")
        for line in log_errors[:10]:
            print(f"        {line[:200]}")
    else:
        print(f"ok    app log clean ({len(log_text.splitlines())} lines)")


def report_size(paths: dict[str, Path]) -> int:
    step("Bundle size")
    size = tree_size(paths["bundle"])
    print(f"{rel(paths['bundle'])}: {human(size)}")
    root = paths["meipass"]
    entries = []
    for child in root.iterdir():
        real = child.resolve()  # macOS: data folders are symlinked from Frameworks to Resources
        if child.is_symlink() and root in real.parents:
            continue  # an alias of something listed already (e.g. libvtk*.dylib, QtCore)
        entries.append((tree_size(real), child.name))
    for size_, name in sorted(entries, reverse=True)[:12]:
        print(f"  {human(size_):>9}  {name}")
    return size


# ----------------------------------------------------------------------------- archive
def _write_checksum(artifact: Path) -> None:
    digest = hashlib.sha256()
    with open(artifact, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    Path(f"{artifact}.sha256").write_text(f"{digest.hexdigest()}  {artifact.name}\n", encoding="utf-8")


def archive_macos(paths: dict[str, Path]) -> Path:
    stem = appinfo.artifact_stem()
    app = paths["bundle"]
    dmg = DIST / f"{stem}.dmg"
    staging = WORK / "dmg"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    subprocess.run(["ditto", str(app), str(staging / app.name)], check=True)
    (staging / "Applications").symlink_to("/Applications")
    dmg.unlink(missing_ok=True)
    cmd = ["hdiutil", "create", "-volname", f"{appinfo.APP_NAME} {appinfo.version()}", "-srcfolder",
           str(staging), "-fs", "HFS+", "-format", "UDZO", "-imagekey", "zlib-level=9", "-ov", str(dmg)]
    for attempt in (1, 2, 3):  # hdiutil is occasionally "Resource busy" on CI machines
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and dmg.is_file():
            shutil.rmtree(staging, ignore_errors=True)
            return dmg
        print(f"hdiutil attempt {attempt} failed: {(result.stderr or result.stdout).strip()}")
        time.sleep(5)
    shutil.rmtree(staging, ignore_errors=True)
    print("falling back to a zip archive")
    archive = DIST / f"{stem}.zip"
    archive.unlink(missing_ok=True)
    # ditto keeps the symlinks and metadata that a signed .app needs (zipfile would not)
    subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(app), str(archive)], check=True)
    return archive


def archive_windows(paths: dict[str, Path]) -> Path:
    base = DIST / appinfo.artifact_stem()
    Path(f"{base}.zip").unlink(missing_ok=True)
    return Path(shutil.make_archive(str(base), "zip", root_dir=DIST, base_dir=paths["bundle"].name))


def archive_linux(paths: dict[str, Path]) -> Path:
    stem = appinfo.artifact_stem()
    top = f"{appinfo.LINUX_NAME}-{appinfo.version()}"
    archive = DIST / f"{stem}.tar.gz"
    archive.unlink(missing_ok=True)
    extras = {
        "torchcor-lighting.desktop": appinfo.RESOURCES / "torchcor-lighting.desktop",
        "torchcor-lighting.png": appinfo.RESOURCES / "app_icon.png",
        "install.sh": LINUX_DIR / "install.sh",
        "LICENSE": appinfo.ROOT / "LICENSE",
    }

    def executable(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.mode = 0o755
        return info

    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        tar.add(paths["bundle"], arcname=top)
        for name, source in extras.items():
            tar.add(source, arcname=f"{top}/{name}", filter=executable if name == "install.sh" else None)
    return archive


def make_archive(paths: dict[str, Path]) -> Path:
    step("Release archive")
    archive = (archive_macos if appinfo.IS_MAC else archive_windows if appinfo.IS_WIN else archive_linux)(paths)
    _write_checksum(archive)
    print(f"{rel(archive)}: {human(archive.stat().st_size)} (+ .sha256)")
    return archive


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and check the TorchCor Lighting desktop app.")
    parser.add_argument("--clean", action="store_true", help="remove previous build output and PyInstaller's cache first")
    parser.add_argument("--archive", action="store_true", help="create the release artifact (dmg / zip / tar.gz)")
    parser.add_argument("--skip-build", action="store_true", help="check (and archive) the existing bundle in dist/")
    parser.add_argument("--no-run", action="store_true", help="do not run the frozen app (layout checks only)")
    parser.add_argument("--no-render", action="store_true", help="self-test without the off-screen VTK render")
    parser.add_argument("--launch-seconds", type=float, default=8.0, help="how long the GUI must stay up (default 8)")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(line_buffering=True)  # keep our output in order with the subprocesses'
    if sys.version_info < (3, 11):  # noqa: UP036 - build.py may be started with any python
        print("error: TorchCor Lighting needs Python 3.11 or newer", file=sys.stderr)
        return 2
    paths = layout()
    print(f"TorchCor Lighting {appinfo.version()} -- {appinfo.os_tag()} {appinfo.arch()}, Python {sys.version.split()[0]}")
    try:
        if not args.skip_build:
            if args.clean:
                clean(paths)
            ensure_icons()
            run_pyinstaller(clean_cache=args.clean)
        check_layout(paths, fresh=not args.skip_build)
        if not args.no_run:
            with tempfile.TemporaryDirectory(prefix="tcs-smoke-") as tmp:
                base = Path(tmp)
                self_test(paths, base, render=not args.no_render)
                launch_test(paths, base, args.launch_seconds)
        size = report_size(paths)
        archive = make_archive(paths) if args.archive else None
    except BuildError as exc:
        print(f"\nBUILD FAILED: {exc}", file=sys.stderr)
        return 1
    step("Done")
    print(f"bundle   {paths['bundle']}  ({human(size)})")
    print(f"run      {paths['exe']}")
    if archive:
        print(f"artifact {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
