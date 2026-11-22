"""Application context: the one object every page talks to.

It owns the settings store, the machine sessions (connection, environment, live
telemetry), the run history and a poller that keeps active runs up to date. All slow
work runs on the worker pool; results come back as Qt signals on the UI thread.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QTimer, Signal

from torchcor_studio.core.data_folders import find_data_folders
from torchcor_studio.core.machines import Machine, MachineError, create_machine
from torchcor_studio.core.models import (
    EnvironmentReport,
    GPUSample,
    InputInfo,
    MachineProfile,
    Project,
    RunRecord,
    RunState,
)
from torchcor_studio.core.runs import LaunchRequest, RunHistory, RunService
from torchcor_studio.core.settings import LOCAL_MACHINE_ID, Store
from torchcor_studio.ui.tasks import run_async

if TYPE_CHECKING:
    from torchcor_studio.core.data_folders import DataFolders
    from torchcor_studio.core.environment import PythonCandidate

log = logging.getLogger(__name__)


class MachineState(StrEnum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class DataDownload:
    """The demo data being downloaded onto a machine (the runtime's ``fetch-data``)."""

    dest: str  # the folder it goes into, as <dest>/torchcor_data
    stage: str = "starting"  # then "download", "unzip"; and at last "done", "failed" or "cancelled"
    done: int = 0  # bytes downloaded or unzipped
    total: int = 0
    version: str = ""  # "data-v1.0"
    path: str = ""  # the data folder, once done
    error: str = ""
    stop: threading.Event = field(default_factory=threading.Event)

    @property
    def active(self) -> bool:
        return self.stage in ("starting", "download", "unzip")


@dataclass
class MachineSession:
    profile: MachineProfile
    machine: Machine | None = None
    state: MachineState = MachineState.DISCONNECTED
    error: str = ""
    env: EnvironmentReport | None = None
    #: A Python on the machine better suited than the one chosen in the profile.
    suggestion: PythonCandidate | None = None
    #: The machine's TorchCor data folder (``~/torchcor_data``), if it has one.
    data_folders: DataFolders | None = None
    #: The demo data download started from the Compute step, while it runs and after.
    download: DataDownload | None = None
    samples: list[GPUSample] = field(default_factory=list)
    monitor: Any = None
    watchers: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def connected(self) -> bool:
        return self.state == MachineState.CONNECTED and self.machine is not None


def _suggest(machine: Machine, env: EnvironmentReport) -> PythonCandidate | None:
    """A better Python than the one pinned in the profile (auto-detection already picks
    the best one). Runs on a worker thread."""
    if not machine.profile.python:
        return None
    from torchcor_studio.core.environment import better_python  # loads the SSH stack

    try:
        return better_python(machine, env)
    except MachineError as exc:
        log.info("%s: could not look for other Pythons: %s", machine.name, exc)
        return None


class AppContext(QObject):
    machines_changed = Signal()
    machine_state_changed = Signal(str)  # machine id
    environment_changed = Signal(str)  # machine id
    telemetry = Signal(str, object)  # machine id, list[GPUSample]
    runs_changed = Signal()
    run_updated = Signal(object)  # RunRecord
    projects_changed = Signal()
    notify = Signal(str, str, str)  # title, message, tone
    demo_data_changed = Signal(str)  # machine id: a download moved on, or ended

    POLL_MS = 2000

    def __init__(self, store: Store | None = None) -> None:
        super().__init__()
        self.store = store or Store()
        self.history = RunHistory()
        self.runs = RunService()
        self._sessions: dict[str, MachineSession] = {}
        self._polling: set[str] = set()
        self._poller = QTimer(self)
        self._poller.setInterval(self.POLL_MS)
        self._poller.timeout.connect(self._poll_active_runs)
        self._poller.start()

    @property
    def settings(self):
        return self.store.settings

    # ================================================================== machines
    def profiles(self) -> list[MachineProfile]:
        return list(self.store.machines)

    def profile(self, machine_id: str) -> MachineProfile | None:
        return self.store.machine(machine_id)

    def session(self, machine_id: str) -> MachineSession:
        sess = self._sessions.get(machine_id)
        profile = self.store.machine(machine_id)
        if sess is None:
            if profile is None:
                raise KeyError(machine_id)
            sess = MachineSession(profile=profile)
            self._sessions[machine_id] = sess
        elif profile is not None and profile is not sess.profile:
            sess.profile = profile
        return sess

    def machine(self, machine_id: str) -> Machine | None:
        sess = self._sessions.get(machine_id)
        return sess.machine if sess and sess.connected else None

    def save_machine(self, profile: MachineProfile) -> None:
        existing = self._sessions.get(profile.id)
        self.store.upsert_machine(profile)
        if existing is not None:
            self.disconnect(profile.id)
            self._sessions.pop(profile.id, None)
        self.machines_changed.emit()

    def remove_machine(self, machine_id: str) -> None:
        self.disconnect(machine_id)
        self._sessions.pop(machine_id, None)
        self.store.remove_machine(machine_id)
        self.machines_changed.emit()

    def connect_machine(self, machine_id: str, on_ready: Callable[[MachineSession], None] | None = None,
                        reprobe: bool = False) -> None:
        """Connect (if needed) and probe the Python environment, asynchronously."""
        sess = self.session(machine_id)
        if sess.connected and sess.env is not None and not reprobe:
            if on_ready:
                on_ready(sess)
            return
        if sess.state == MachineState.CONNECTING:
            if on_ready:
                self._after_connect(machine_id, on_ready)
            return
        sess.state = MachineState.CONNECTING
        sess.error = ""
        self.machine_state_changed.emit(machine_id)
        if on_ready:
            self._after_connect(machine_id, on_ready)

        def work() -> tuple[Machine, EnvironmentReport, PythonCandidate | None]:
            machine = sess.machine or create_machine(sess.profile)
            machine.connect()
            if reprobe and not sess.profile.python:
                machine.python = ""  # auto-detect again: an environment may have appeared
            env = machine.probe_environment()
            if env.python:
                machine.python = env.python
            return machine, env, _suggest(machine, env), find_data_folders(machine)

        def done(result) -> None:
            machine, env, sess.suggestion, sess.data_folders = result
            sess.machine, sess.env = machine, env
            sess.state = MachineState.CONNECTED
            self.machine_state_changed.emit(machine_id)
            self.environment_changed.emit(machine_id)
            if sess.watchers and sess.monitor is None:
                self._start_monitor(sess)
            self.refresh_runs(machine_id)

        def failed(exc: BaseException) -> None:
            sess.state = MachineState.ERROR
            sess.error = str(exc) or type(exc).__name__
            self.machine_state_changed.emit(machine_id)

        run_async(work, on_done=done, on_error=failed, owner=self)

    # ================================================================== demo data
    def fetch_demo_data(self, machine_id: str, dest: str) -> None:
        """Download the newest demo data release into ``dest`` on the machine (``fetch-data``,
        on the worker pool), then make ``<dest>/torchcor_data`` its data folder. Progress
        and the end come as :attr:`demo_data_changed`; one download at a time per machine."""
        sess = self.session(machine_id)
        machine = self.machine(machine_id)
        if machine is None or (sess.download is not None and sess.download.active):
            return
        job = sess.download = DataDownload(dest)
        last_line = [""]  # a failure's message: the runtime's last plain line

        def on_line(line: str) -> None:  # on the worker: the signal reaches the UI thread queued
            try:
                event = json.loads(line)
            except ValueError:
                if line.strip():
                    last_line[0] = line.strip()
                return
            if not isinstance(event, dict):
                return
            if "progress" in event:
                job.stage, job.version = str(event["progress"]), str(event.get("version", ""))
                job.done, job.total = int(event.get("done") or 0), int(event.get("total") or 0)
            elif event.get("ok"):
                job.path, job.version = str(event.get("path", "")), str(event.get("version", job.version))
            elif event.get("error"):
                job.error = str(event["error"])
            self.demo_data_changed.emit(machine_id)

        def work() -> DataFolders | None:
            script = machine.join(machine.runtime_dir(), "tcs_runtime.py")
            code = machine.stream([machine.python, script, "fetch-data", dest], on_line, job.stop)
            if code != 0 or not job.path:
                job.error = job.error or last_line[0] or f"fetch-data ended with code {code}"
                return None
            machine.profile.data_dir = job.path
            return find_data_folders(machine)

        def done(found: DataFolders | None) -> None:
            if job.path and not job.error:
                job.stage = "done"
                profile = self.store.machine(machine_id)
                if profile is not None:
                    profile.data_dir = job.path
                    self.store.upsert_machine(profile)
                sess.data_folders = found
                self.machine_state_changed.emit(machine_id)  # the Setup step's folders follow
            else:
                job.stage = "cancelled" if job.stop.is_set() else "failed"
            self.demo_data_changed.emit(machine_id)

        def failed(exc: BaseException) -> None:
            job.stage, job.error = "failed", str(exc) or type(exc).__name__
            self.demo_data_changed.emit(machine_id)

        self.demo_data_changed.emit(machine_id)
        run_async(work, on_done=done, on_error=failed, owner=self)

    def cancel_demo_data(self, machine_id: str) -> None:
        sess = self._sessions.get(machine_id)
        if sess is not None and sess.download is not None and sess.download.active:
            sess.download.stop.set()

    def _after_connect(self, machine_id: str, callback: Callable[[MachineSession], None]) -> None:
        def handler(mid: str) -> None:
            if mid != machine_id:
                return
            sess = self.session(machine_id)
            if sess.state in (MachineState.CONNECTED, MachineState.ERROR):
                self.machine_state_changed.disconnect(handler)
                callback(sess)

        self.machine_state_changed.connect(handler)

    def reprobe(self, machine_id: str, python: str | None = None) -> None:
        """Re-check the environment (e.g. after installing TorchCor or picking a Python)."""
        sess = self.session(machine_id)
        if not sess.connected:
            self.connect_machine(machine_id)
            return
        machine = sess.machine

        def work() -> tuple[EnvironmentReport, PythonCandidate | None]:
            if python:
                machine.python = python
            env = machine.probe_environment(python or machine.python)
            return env, _suggest(machine, env)

        def done(result) -> None:
            env, sess.suggestion = result
            sess.env = env
            if env.python and not env.error:
                machine.python = env.python
            self.environment_changed.emit(machine_id)

        run_async(work, on_done=done, on_error=lambda e: self._warn("Environment check failed", e), owner=self)

    def disconnect(self, machine_id: str) -> None:
        sess = self._sessions.get(machine_id)
        if sess is None:
            return
        self._stop_monitor(sess)
        machine, sess.machine = sess.machine, None
        sess.state = MachineState.DISCONNECTED
        sess.env = None
        if machine is not None:
            run_async(machine.close)
        self.machine_state_changed.emit(machine_id)

    def shutdown(self) -> None:
        self._poller.stop()
        for sess in self._sessions.values():
            if sess.download is not None:
                sess.download.stop.set()  # a download in flight ends with the app
            self._stop_monitor(sess)
            if sess.machine is not None:
                try:
                    sess.machine.close()
                except Exception:  # noqa: BLE001
                    log.debug("close failed", exc_info=True)

    # ================================================================== telemetry
    def watch_telemetry(self, machine_id: str) -> None:
        sess = self.session(machine_id)
        sess.watchers += 1
        if sess.connected and sess.monitor is None:
            self._start_monitor(sess)
        elif not sess.connected and sess.state != MachineState.CONNECTING:
            self.connect_machine(machine_id)

    def unwatch_telemetry(self, machine_id: str) -> None:
        sess = self._sessions.get(machine_id)
        if sess is None:
            return
        sess.watchers = max(0, sess.watchers - 1)
        if sess.watchers == 0:
            self._stop_monitor(sess)

    def _start_monitor(self, sess: MachineSession) -> None:
        from torchcor_studio.core.gpu import GpuMonitor

        machine_id = sess.profile.id
        monitor: GpuMonitor | None = None

        def on_samples(samples: list[GPUSample]) -> None:
            # Monitors stop on a background thread, so a last reading can arrive after
            # unwatch / shutdown, when this context may already be deleted: drop it.
            if sess.monitor is not monitor:
                return
            sess.samples = samples
            try:
                self.telemetry.emit(machine_id, samples)  # queued to the UI thread
            except RuntimeError:  # deleted while this reading was taken
                log.debug("telemetry for %s arrived after shutdown", machine_id)

        try:
            monitor = GpuMonitor(sess.machine, interval=self.settings.telemetry_interval_s,
                                 callback=on_samples)
            sess.monitor = monitor
            monitor.start()
        except Exception as exc:  # noqa: BLE001
            log.warning("telemetry unavailable for %s: %s", machine_id, exc)
            sess.monitor = None

    @staticmethod
    def _stop_monitor(sess: MachineSession) -> None:
        monitor, sess.monitor = sess.monitor, None
        if monitor is not None:
            threading.Thread(target=monitor.stop, daemon=True).start()

    # ================================================================== inputs
    def inspect_input(self, machine_id: str, path: str, on_done: Callable[[InputInfo], None],
                      owner: QObject | None = None) -> None:
        """Inspect an input folder off the UI thread; ``on_done`` is dropped if ``owner`` (the
        widget asking, say) is deleted meanwhile."""
        machine = self.machine(machine_id)
        if machine is None or not path:
            on_done(InputInfo(path=path, error="machine not connected" if path else None))
            return

        def work() -> InputInfo:
            return InputInfo.from_dict(machine.runtime("inspect", machine.expand(path), timeout=300))

        run_async(work, on_done=on_done,
                  on_error=lambda e: on_done(InputInfo(path=path, error=str(e))), owner=owner or self)

    # ================================================================== runs
    def records(self) -> list[RunRecord]:
        return self.history.all()

    def record(self, key: str) -> RunRecord | None:
        return self.history.get(key)

    def refresh_runs(self, machine_id: str | None = None) -> None:
        ids = [machine_id] if machine_id else [p.id for p in self.profiles()]
        for mid in ids:
            machine = self.machine(mid)
            if machine is None:
                continue
            profile = self.store.machine(mid)
            roots = [r for r in [profile.output_dir if profile else "", *self.history.roots(mid)] if r]

            def work(machine=machine, roots=roots) -> tuple[list[RunRecord], list[str]]:
                expanded = [machine.expand(r) for r in roots]
                return self.runs.scan(machine, expanded), expanded

            def done(result, mid=mid) -> None:
                records, scanned = result
                self.history.replace_machine(mid, records, scanned)
                self.runs_changed.emit()

            run_async(work, on_done=done, on_error=lambda e: log.info("scan failed: %s", e), owner=self)

    def launch(self, machine_id: str, request: LaunchRequest,
               on_done: Callable[[RunRecord], None] | None = None,
               on_error: Callable[[BaseException], None] | None = None) -> None:
        """Create and start one run."""
        self.launch_batch(machine_id, [(request, None)],
                          on_done=(lambda records: on_done(records[0])) if on_done else None,
                          on_error=on_error)

    def launch_batch(self, machine_id: str, items: list[tuple[LaunchRequest, int | None]],
                     on_done: Callable[[list[RunRecord]], None] | None = None,
                     on_error: Callable[[BaseException], None] | None = None) -> None:
        """Create and start several runs in order.

        Each item is ``(request, after_index)``: with an index, that run queues behind the
        run created for ``items[after_index]`` (same machine), forming a per-GPU queue.
        """
        machine = self.machine(machine_id)
        if machine is None:
            if on_error:
                on_error(MachineError("The machine is not connected."))
            return

        def work() -> list[RunRecord]:
            records: list[RunRecord] = []
            for request, after_index in items:
                if after_index is not None:
                    request = replace(request, after=(*request.after, records[after_index].run_dir))
                record = self.runs.create(machine, request)
                records.append(self.runs.launch(machine, record))
            return records

        def done(records: list[RunRecord]) -> None:
            for record in records:
                self.history.upsert(record, save=False)
            self.history.save()
            self.runs_changed.emit()
            if len(records) == 1:
                self.notify.emit("Simulation started", f"{records[0].name} is running on {machine.name}.", "success")
            else:
                self.notify.emit("Sweep started", f"{len(records)} runs queued on {machine.name}.", "success")
            if on_done:
                on_done(records)

        def failed(exc: BaseException) -> None:
            self.notify.emit("Could not start the simulation", str(exc), "danger")
            self.refresh_runs(machine_id)  # some runs of a batch may have started
            if on_error:
                on_error(exc)

        run_async(work, on_done=done, on_error=failed, owner=self)

    def cancel(self, record: RunRecord) -> None:
        machine = self.machine(record.machine_id)
        if machine is None:
            self.notify.emit("Machine offline", "Reconnect to stop this run.", "warning")
            return
        run_async(self.runs.cancel, machine, record,
                  on_done=lambda _r: self._poll_one(record),
                  on_error=lambda e: self._warn("Could not stop the run", e), owner=self)

    def delete(self, record: RunRecord, on_done: Callable[[], None] | None = None) -> None:
        machine = self.machine(record.machine_id)

        def finished(_result=None) -> None:
            self.history.remove(record.key)
            self.runs_changed.emit()
            if on_done:
                on_done()

        if machine is None:
            finished()  # forget an unreachable run locally
            return
        run_async(self.runs.delete, machine, record, on_done=finished,
                  on_error=lambda e: self._warn("Could not delete the run", e), owner=self)

    def rename(self, record: RunRecord, name: str) -> None:
        machine = self.machine(record.machine_id)
        record.name = name
        record.manifest["name"] = name
        self.history.upsert(record)
        self.run_updated.emit(record)
        if machine is not None:
            from torchcor_studio.core.runs import protocol as P

            run_async(machine.write_json, machine.join(record.run_dir, P.MANIFEST), record.manifest,
                      on_error=lambda e: self._warn("Could not rename the run", e), owner=self)

    # ------------------------------------------------------------------ projects
    def projects(self) -> list[Project]:
        return self.history.projects()

    def project(self, project_id: str | None) -> Project | None:
        return self.history.project(project_id)

    def create_project(self, name: str) -> Project:
        """A new project, or the existing one of that name (names are unique, ignoring case)."""
        existing = next((p for p in self.history.projects() if p.name.casefold() == name.strip().casefold()), None)
        if existing is not None:
            return existing
        project = self.history.add_project(name)
        self.projects_changed.emit()
        return project

    def rename_project(self, project_id: str, name: str) -> None:
        project = self.history.project(project_id)
        if project is None or not name.strip():
            return
        self.history.rename_project(project_id, name)
        members = [r for r in self.history.all() if r.project_id == project_id]
        self.move_to_project(members, project_id)  # their manifests carry the new name too
        self.projects_changed.emit()

    def delete_project(self, project_id: str) -> None:
        """Remove the project; its runs are kept and become unfiled."""
        members = [r for r in self.history.all() if r.project_id == project_id]
        self.move_to_project(members, None)
        self.history.remove_project(project_id)
        self.projects_changed.emit()

    def move_to_project(self, records: list[RunRecord], project_id: str | None) -> None:
        """File runs in a project (``None``: take them out). The project is written into
        each run's manifest, so it travels with the run folder."""
        project = self.history.project(project_id)
        for record in records:
            if project is not None:
                record.manifest["project"] = project.ref()
            else:
                record.manifest.pop("project", None)
            self.history.upsert(record, save=False)
            self.run_updated.emit(record)
            self._write_manifest(record, "Could not file the run in the project")
        self.history.save()
        self.runs_changed.emit()

    def _write_manifest(self, record: RunRecord, failure: str) -> None:
        """Save ``record.manifest`` on its machine (connecting first if needed, so that the
        next scan of the machine does not bring back the old manifest)."""
        from torchcor_studio.core.runs import protocol as P

        def write(sess: MachineSession) -> None:
            if sess.machine is None or not sess.connected:
                self._warn(failure, MachineError(f"{sess.profile.name} is not reachable"))
                return
            run_async(sess.machine.write_json, sess.machine.join(record.run_dir, P.MANIFEST),
                      dict(record.manifest), on_error=lambda e: self._warn(failure, e), owner=self)

        if self.store.machine(record.machine_id) is None:
            return
        sess = self.session(record.machine_id)
        if sess.connected:
            write(sess)
        else:
            self.connect_machine(record.machine_id, on_ready=write)

    # ------------------------------------------------------------------ polling
    def _poll_active_runs(self) -> None:
        for record in self.history.all():
            if record.status.state.is_active and record.key not in self._polling:
                self._poll_one(record)

    def _poll_one(self, record: RunRecord) -> None:
        machine = self.machine(record.machine_id)
        if machine is None:
            return
        self._polling.add(record.key)

        def done(status) -> None:
            self._polling.discard(record.key)
            changed_state = status.state != record.status.state
            record.status = status
            self.history.upsert(record, save=changed_state)
            self.run_updated.emit(record)
            if changed_state and status.state.is_terminal:
                self._announce(record)

        def failed(_exc) -> None:
            self._polling.discard(record.key)

        run_async(self.runs.read_status, machine, record, on_done=done, on_error=failed, owner=self)

    def _announce(self, record: RunRecord) -> None:
        state = record.status.state
        if state == RunState.COMPLETED:
            self.notify.emit("Simulation finished", f"{record.name} completed.", "success")
        elif state == RunState.FAILED:
            self.notify.emit("Simulation failed", record.status.error or record.name, "danger")
        elif state == RunState.CANCELLED:
            self.notify.emit("Simulation stopped", record.name, "info")
        elif state == RunState.LOST:
            self.notify.emit("Lost contact with a run", record.name, "warning")
        self.runs_changed.emit()

    def _warn(self, title: str, exc: BaseException) -> None:
        self.notify.emit(title, str(exc), "danger")

    # ------------------------------------------------------------------ helpers
    def reattach(self) -> None:
        """Connect to every machine that still has active runs, so they resume updating."""
        machine_ids = {r.machine_id for r in self.history.all() if r.status.state.is_active}
        machine_ids.add(LOCAL_MACHINE_ID)
        for machine_id in machine_ids:
            if self.store.machine(machine_id) is not None:
                self.connect_machine(machine_id)

    @staticmethod
    def default_machine_id() -> str:
        return LOCAL_MACHINE_ID

    @staticmethod
    def now() -> float:
        return time.time()
