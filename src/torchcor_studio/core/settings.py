"""Application settings and machine profiles, persisted as JSON in the config folder."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from torchcor_studio import TORCHCOR_GIT_URL
from torchcor_studio.core.models import MachineKind, MachineProfile
from torchcor_studio.core.paths import config_dir, default_local_output_dir

LOCAL_MACHINE_ID = "local"


@dataclass
class Settings:
    theme: str = "system"  # "system" | "dark" | "light"
    #: How TorchCor is installed: "git" (GitHub), "pypi", or "path" (a local checkout).
    torchcor_source: str = "git"
    torchcor_git_url: str = TORCHCOR_GIT_URL
    torchcor_git_ref: str = ""  # branch / tag / commit; "" = default branch
    torchcor_local_path: str = ""
    telemetry_interval_s: float = 1.0
    confirm_delete: bool = True
    last_machine_id: str = LOCAL_MACHINE_ID
    last_kind: str = "ep"
    window_geometry: str = ""  # base64 QByteArray
    #: Widths of the resizable left panels (px): the run list, the Machines page list.
    sidebar_width: int = 272
    machine_list_width: int = 320
    #: Projects shown open in the sidebar.
    expanded_projects: list[str] = field(default_factory=list)
    #: Machine id → folders chosen on the New run page, most recent first.
    recent_inputs: dict[str, list[str]] = field(default_factory=dict)
    recent_outputs: dict[str, list[str]] = field(default_factory=dict)
    #: Machine id → simulation kind → the input folder last chosen for that kind.
    kind_inputs: dict[str, dict[str, str]] = field(default_factory=dict)

    def torchcor_requirement(self) -> str:
        """The pip requirement that installs TorchCor from the configured source."""
        if self.torchcor_source == "pypi":
            return "torchcor"
        if self.torchcor_source == "path" and self.torchcor_local_path:
            return self.torchcor_local_path
        ref = f"@{self.torchcor_git_ref}" if self.torchcor_git_ref else ""
        return f"torchcor @ git+{self.torchcor_git_url}{ref}"

    def remember_input(self, machine_id: str, path: str, kind: str | None = None, keep: int = 8) -> None:
        _remember(self.recent_inputs, machine_id, path, keep)
        if kind and path:
            self.kind_inputs.setdefault(machine_id, {})[kind] = path

    def remember_output(self, machine_id: str, path: str, keep: int = 8) -> None:
        _remember(self.recent_outputs, machine_id, path, keep)


def _remember(recent: dict[str, list[str]], machine_id: str, path: str, keep: int) -> None:
    """Move ``path`` to the front of a machine's most-recent-first folder list."""
    if not path:
        return
    items = [p for p in recent.get(machine_id, []) if p != path]
    recent[machine_id] = [path, *items][:keep]


def _local_profile() -> MachineProfile:
    return MachineProfile(
        id=LOCAL_MACHINE_ID,
        name="This computer",
        kind=MachineKind.LOCAL,
        output_dir=str(default_local_output_dir()),
    )


class Store:
    """Thread-safe JSON persistence for :class:`Settings` and machine profiles."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or config_dir()
        self._lock = threading.RLock()
        self.settings = self._load_settings()
        self.machines = self._load_machines()

    # ---------------------------------------------------------------- settings
    @property
    def _settings_path(self) -> Path:
        return self.dir / "settings.json"

    def _load_settings(self) -> Settings:
        data = _read(self._settings_path)
        known = {f for f in Settings.__dataclass_fields__}
        return Settings(**{k: v for k, v in (data or {}).items() if k in known})

    def save_settings(self) -> None:
        with self._lock:
            _write(self._settings_path, asdict(self.settings))

    # ---------------------------------------------------------------- machines
    @property
    def _machines_path(self) -> Path:
        return self.dir / "machines.json"

    def _load_machines(self) -> list[MachineProfile]:
        raw = _read(self._machines_path) or []
        profiles = []
        for item in raw:
            try:
                profiles.append(MachineProfile.from_dict(item))
            except (TypeError, ValueError):
                continue
        if not any(p.id == LOCAL_MACHINE_ID for p in profiles):
            profiles.insert(0, _local_profile())
        return profiles

    def save_machines(self) -> None:
        with self._lock:
            _write(self._machines_path, [p.to_dict() for p in self.machines])

    def machine(self, machine_id: str) -> MachineProfile | None:
        return next((m for m in self.machines if m.id == machine_id), None)

    def upsert_machine(self, profile: MachineProfile) -> None:
        with self._lock:
            for i, m in enumerate(self.machines):
                if m.id == profile.id:
                    self.machines[i] = profile
                    break
            else:
                self.machines.append(profile)
            self.save_machines()

    def remove_machine(self, machine_id: str) -> None:
        if machine_id == LOCAL_MACHINE_ID:
            raise ValueError("the local machine cannot be removed")
        with self._lock:
            self.machines = [m for m in self.machines if m.id != machine_id]
            self.save_machines()


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
