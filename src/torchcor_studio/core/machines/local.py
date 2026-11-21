"""This computer as a :class:`Machine`: subprocesses and the native filesystem."""

from __future__ import annotations

import glob
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from functools import cache
from importlib import metadata
from pathlib import Path

import psutil

from torchcor_studio.core.machines.base import Machine, MachineError
from torchcor_studio.core.machines.deploy import deploy_local
from torchcor_studio.core.models import CommandResult, FileEntry
from torchcor_studio.core.paths import cache_dir, data_dir

IS_WINDOWS = os.name == "nt"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if IS_WINDOWS else 0
_CHUNK = 1 << 20


# ============================================================================ errors
@cache
def _error_class(base: type[OSError]) -> type[MachineError]:
    return type(f"Machine{base.__name__}", (MachineError, base), {"__module__": __name__})


def machine_os_error(exc: OSError, path: str = "") -> MachineError:
    """Wrap a filesystem error as a :class:`MachineError`.

    The result is *also* an instance of the original ``OSError`` subclass, so callers can
    catch ``MachineError`` or ``FileNotFoundError`` / ``PermissionError`` / ``OSError``.
    """
    if isinstance(exc, MachineError):
        return exc
    base = type(exc) if isinstance(exc, OSError) else OSError
    reason = exc.strerror or (exc.args[0] if exc.args and isinstance(exc.args[0], str) else "")
    reason = reason or str(exc) or base.__name__
    err = _error_class(base)(f"{reason}: {path}" if path else reason)
    err.errno, err.strerror, err.filename = exc.errno, exc.strerror, path or exc.filename
    return err


@contextmanager
def _fs(path: str) -> Iterator[None]:
    try:
        yield
    except MachineError:
        raise
    except OSError as exc:
        raise machine_os_error(exc, path) from exc


# ============================================================================ processes
def kill_tree(pid: int, timeout: float = 3.0) -> None:
    """Terminate ``pid`` and all its descendants (SIGTERM, then SIGKILL after ``timeout``)."""
    try:
        root = psutil.Process(pid)
        procs = [*root.children(recursive=True), root]
    except psutil.Error:
        return
    for proc in procs:
        with suppress(psutil.Error):
            proc.terminate()
    _gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for proc in alive:
        with suppress(psutil.Error):
            proc.kill()


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


def _program(argv: Sequence[str] | str) -> str:
    return argv.split()[0] if isinstance(argv, str) and argv.split() else str(argv[0] if argv else "")


# ============================================================================ pythons
def _has_torchcor_here() -> bool:
    """Is TorchCor installed for the running interpreter (not just a checkout in the cwd)?"""
    try:
        metadata.version("torchcor")
        return True
    except metadata.PackageNotFoundError:
        pass
    try:
        spec = importlib.util.find_spec("torchcor")
    except (ImportError, ValueError):
        return False
    return spec is not None and spec.origin not in (None, "namespace")


def _conda_roots(home: Path) -> list[Path]:
    names = ("miniconda3", "anaconda3", "miniforge3", "mambaforge", "micromamba", ".conda")
    roots = [home / n for n in names]
    if IS_WINDOWS:
        for env_var in ("LOCALAPPDATA", "PROGRAMDATA"):
            base = os.environ.get(env_var)
            if base:
                roots += [Path(base) / n for n in ("miniconda3", "anaconda3", "miniforge3")]
    else:
        roots += [Path("/opt/conda"), Path("/opt/miniconda3"), Path("/opt/anaconda3"),
                  Path("/opt/homebrew/Caskroom/miniforge/base")]
    return roots


#: macOS asks the user before an app may look inside these folders: never probe there.
_MAC_PRIVATE = ("Desktop", "Documents", "Downloads")


def _conda_recorded_envs(home: Path) -> list[Path]:
    """Every environment conda created, wherever it lives (``~/.conda/environments.txt``)."""
    try:
        lines = (home / ".conda" / "environments.txt").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    private = [home / name for name in _MAC_PRIVATE] if sys.platform == "darwin" else []
    envs = [Path(line.strip()) for line in lines if line.strip()]
    return [env for env in envs if not any(env.is_relative_to(folder) for folder in private) and env.is_dir()]


def _env_roots() -> list[Path]:
    """Python environments worth checking, most likely first (no subprocesses)."""
    home = Path.home()
    roots: list[Path] = [data_dir() / "venv"]
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if os.environ.get(var):
            roots.append(Path(os.environ[var]))
    for conda in _conda_roots(home):
        if conda.is_dir():
            roots += sorted(p for p in (conda / "envs").glob("*") if p.is_dir())
            roots.append(conda)
    roots += _conda_recorded_envs(home)
    roots += [Path.cwd() / ".venv", home / ".venv"]
    roots += sorted(p for p in home.glob("venv*") if p.is_dir())
    roots += sorted(p for p in (home / ".virtualenvs").glob("*") if p.is_dir())
    seen: set[str] = set()
    unique = []
    for root in roots:
        key = os.path.normcase(str(root))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def env_interpreter(root: Path) -> Path | None:
    """The interpreter of an environment folder (conda env or venv), if any."""
    if IS_WINDOWS:
        options = (root / "python.exe", root / "Scripts" / "python.exe")
    else:
        options = (root / "bin" / "python", root / "bin" / "python3")
    return next((p for p in options if p.is_file()), None)


def env_has_torchcor(root: Path) -> bool:
    """Cheap check (a glob, no subprocess): is TorchCor installed in this environment?"""
    pattern = "Lib/site-packages/torchcor*" if IS_WINDOWS else "lib/python3*/site-packages/torchcor*"
    return bool(glob.glob(str(root / pattern)))


def _system_pythons() -> list[str]:
    """Interpreters on PATH and in well-known install locations."""
    found: list[str] = []
    if IS_WINDOWS:
        launcher = shutil.which("py")
        if launcher:
            try:
                out = subprocess.run([launcher, "-3", "-c", "import sys; print(sys.executable)"],
                                     capture_output=True, text=True, timeout=5,
                                     creationflags=_NO_WINDOW)
                if out.returncode == 0 and out.stdout.strip():
                    found.append(out.stdout.strip())
            except (OSError, subprocess.SubprocessError):
                pass
        exe = shutil.which("python")
        if exe and "WindowsApps" not in exe:  # skip the Microsoft Store stub
            found.append(exe)
        local = os.environ.get("LOCALAPPDATA")
        if local:
            found += sorted(glob.glob(os.path.join(local, "Programs", "Python", "Python3*", "python.exe")),
                            reverse=True)
        return found
    for name in ("python3", "python"):
        exe = shutil.which(name)
        if exe and not _is_clt_stub(exe):
            found.append(exe)
    found += [p for p in ("/opt/homebrew/bin/python3", "/usr/local/bin/python3") if os.path.isfile(p)]
    found += sorted(glob.glob("/Library/Frameworks/Python.framework/Versions/3.*/bin/python3"), reverse=True)
    found += sorted(glob.glob(str(Path.home() / ".pyenv" / "versions" / "*" / "bin" / "python")), reverse=True)
    if os.path.isfile("/usr/bin/python3") and not _is_clt_stub("/usr/bin/python3"):
        found.append("/usr/bin/python3")
    return found


def _is_clt_stub(exe: str) -> bool:
    """macOS ``/usr/bin/python3`` without the Command Line Tools pops up an installer."""
    if sys.platform != "darwin" or os.path.realpath(exe) != "/usr/bin/python3":
        return False
    return not (os.path.isdir("/Library/Developer/CommandLineTools")
                or os.path.isdir("/Applications/Xcode.app"))


def python_candidates() -> list[str]:
    """Every Python interpreter found on this computer, most promising first.

    Environments with TorchCor come first; nothing is executed except, on Windows, the
    ``py`` launcher to resolve its default interpreter.
    """
    with_tc: list[str] = []
    without: list[str] = []
    if not getattr(sys, "frozen", False):
        (with_tc if _has_torchcor_here() else without).append(sys.executable)
    for root in _env_roots():
        exe = env_interpreter(root)
        if exe is not None:
            (with_tc if env_has_torchcor(root) else without).append(str(exe))
    ordered = [*with_tc, *without, *_system_pythons()]
    seen: set[str] = set()
    unique = []
    for exe in ordered:
        # python / python3 / python3.11 in one folder are the same environment
        key = os.path.normcase(os.path.dirname(os.path.abspath(exe)))
        if key not in seen:
            seen.add(key)
            unique.append(exe)
    return unique


def studio_venv_path() -> str:
    """Where the studio creates its own virtual environment on this computer."""
    return str(data_dir() / "venv")


# ============================================================================ machine
class LocalMachine(Machine):
    """Runs commands with :mod:`subprocess` and uses native paths."""

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        """Nothing to do: this computer is always reachable."""

    def close(self) -> None:
        """Nothing to release."""

    @property
    def is_connected(self) -> bool:
        return True

    # ------------------------------------------------------------------ commands
    @staticmethod
    def _env(extra: dict[str, str] | None) -> dict[str, str]:
        env = dict(os.environ)
        if getattr(sys, "frozen", False):  # the bundled app's own Qt/matplotlib settings
            for var in ("QT_PLUGIN_PATH", "QML2_IMPORT_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "MPLCONFIGDIR"):
                env.pop(var, None)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        if extra:
            env.update({k: str(v) for k, v in extra.items()})
        return env

    def run(
        self,
        argv: Sequence[str] | str,
        *,
        timeout: float | None = 60.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
    ) -> CommandResult:
        if cwd and not os.path.isdir(os.path.expanduser(cwd)):
            return CommandResult(2, "", f"cd: {cwd}: No such file or directory")
        try:
            proc = subprocess.Popen(
                argv if isinstance(argv, str) else list(argv),
                shell=isinstance(argv, str),
                cwd=os.path.expanduser(cwd) if cwd else None,
                env=self._env(env),
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=_NO_WINDOW,
            )
        except FileNotFoundError:
            return CommandResult(127, "", f"{_program(argv)}: command not found")
        except OSError as exc:
            return CommandResult(126, "", f"{_program(argv)}: {exc.strerror or exc}")
        try:
            out, err = proc.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(proc.pid, timeout=1.0)
            proc.communicate()
            raise MachineError(f"{_program(argv)} timed out after {timeout:g} s") from None
        return CommandResult(proc.returncode, _decode(out), _decode(err))

    def stream(
        self,
        argv: Sequence[str] | str,
        on_line: Callable[[str], None],
        stop: threading.Event,
        *,
        env: dict[str, str] | None = None,
    ) -> int:
        """Run ``argv`` and call ``on_line`` per output line (stderr is merged into stdout)."""
        if stop.is_set():
            return -1
        try:
            proc = subprocess.Popen(
                argv if isinstance(argv, str) else list(argv),
                shell=isinstance(argv, str),
                env=self._env(env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=_NO_WINDOW,
            )
        except FileNotFoundError:
            on_line(f"{_program(argv)}: command not found")
            return 127
        except OSError as exc:
            on_line(f"{_program(argv)}: {exc.strerror or exc}")
            return 126

        finished = threading.Event()

        def watch() -> None:
            while not finished.is_set():
                if stop.wait(0.1):
                    kill_tree(proc.pid, timeout=2.0)
                    return

        watcher = threading.Thread(target=watch, name="tcs-stream-stop", daemon=True)
        watcher.start()
        drained = False
        try:
            assert proc.stdout is not None
            for raw in iter(proc.stdout.readline, b""):
                on_line(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            drained = True
        finally:
            if not drained and proc.poll() is None:
                kill_tree(proc.pid, timeout=2.0)  # ``on_line`` raised
            finished.set()
            watcher.join()
            proc.stdout.close()
            code = proc.wait()
        return -1 if stop.is_set() else code

    # ------------------------------------------------------------------ filesystem
    def home(self) -> str:
        return str(Path.home())

    def join(self, *parts: str) -> str:
        return os.path.join(*parts)

    def parent(self, path: str) -> str:
        return str(Path(path).parent)

    def basename(self, path: str) -> str:
        return Path(path).name

    def expand(self, path: str) -> str:
        return os.path.abspath(os.path.expanduser(path))

    def exists(self, path: str) -> bool:
        return os.path.exists(os.path.expanduser(path))

    def is_dir(self, path: str) -> bool:
        return os.path.isdir(os.path.expanduser(path))

    def listdir(self, path: str) -> list[FileEntry]:
        folder = os.path.expanduser(path)
        entries: list[FileEntry] = []
        with _fs(path), os.scandir(folder) as it:
            for entry in it:
                try:
                    st = entry.stat()
                    is_dir = stat.S_ISDIR(st.st_mode)
                except OSError:  # dangling symlink
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    is_dir = False
                entries.append(FileEntry(entry.name, os.path.join(folder, entry.name), is_dir,
                                         0 if is_dir else st.st_size, st.st_mtime))
        entries.sort(key=lambda e: (not e.is_dir, e.name.lower(), e.name))
        return entries

    def mkdir(self, path: str, parents: bool = True) -> None:
        with _fs(path):
            Path(os.path.expanduser(path)).mkdir(parents=parents, exist_ok=True)

    def read_bytes(self, path: str, offset: int = 0, length: int | None = None) -> bytes:
        with _fs(path), open(os.path.expanduser(path), "rb") as fh:
            if offset:
                fh.seek(offset)
            return fh.read() if length is None else fh.read(length)

    def write_bytes(self, path: str, data: bytes) -> None:
        target = Path(os.path.expanduser(path))
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
        with _fs(path):
            try:
                fh = open(tmp, "xb")  # noqa: SIM115 - closed below, before the rename
            except FileNotFoundError:
                target.parent.mkdir(parents=True, exist_ok=True)
                fh = open(tmp, "xb")  # noqa: SIM115
            try:
                with fh:
                    fh.write(data)
                _replace(tmp, target)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise

    def size(self, path: str) -> int:
        with _fs(path):
            return os.path.getsize(os.path.expanduser(path))

    def remove(self, path: str, recursive: bool = False) -> None:
        """Delete a file or folder (folders need ``recursive`` unless empty); missing paths are ignored."""
        target = Path(os.path.expanduser(path))
        with _fs(path):
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                if recursive:
                    shutil.rmtree(target)
                else:
                    target.rmdir()

    def download(
        self,
        remote_path: str,
        local_path: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        _copy(os.path.expanduser(remote_path), local_path, progress)

    def upload(self, local_path: str, remote_path: str) -> None:
        _copy(local_path, os.path.expanduser(remote_path), None)

    # ------------------------------------------------------------------ python & runtime
    fallback_python = "python" if IS_WINDOWS else "python3"

    def detect_python(self) -> str:
        """Like :meth:`Machine.detect_python`, with a shortcut: the app's own interpreter
        (running from source) when it has TorchCor and there is no NVIDIA GPU to prefer."""
        if (not self.profile.python and not getattr(sys, "frozen", False)
                and not shutil.which("nvidia-smi") and _has_torchcor_here()):
            return sys.executable
        return super().detect_python()

    def runtime_dir(self) -> str:
        with self._lock:
            if self._runtime_dir is None:
                self._runtime_dir = str(deploy_local(cache_dir() / "runtime"))
            return self._runtime_dir

    # ------------------------------------------------------------------ processes
    def is_alive(self, pid: int) -> bool:
        try:
            return psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied:
            return True

    def terminate(self, pid: int) -> None:
        kill_tree(int(pid))


def _replace(src: Path, dst: Path) -> None:
    """``os.replace`` with retries: on Windows a reader may briefly hold ``dst`` open."""
    for attempt in range(5):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if not IS_WINDOWS or attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _copy(src: str, dst: str, progress: Callable[[int, int], None] | None) -> None:
    """Chunked copy through a temporary file, reporting ``(bytes_done, total)``."""
    target = Path(dst)
    with _fs(src):
        total = os.path.getsize(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")
        try:
            done = 0
            with open(src, "rb") as fi, open(tmp, "wb") as fo:
                while chunk := fi.read(_CHUNK):
                    fo.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
            if progress is not None and total == 0:
                progress(0, 0)
            _replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
