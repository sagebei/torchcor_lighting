"""Local index of known runs and projects, so history shows instantly (even offline)."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable
from pathlib import Path

from torchcor_studio.core.models import Project, RunRecord
from torchcor_studio.core.paths import data_dir


class RunHistory:
    """Thread-safe, JSON-backed ``{key: RunRecord}`` map with per-machine output roots
    and the projects runs are organised in."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (data_dir() / "history.json")
        self._lock = threading.RLock()
        self._records: dict[str, RunRecord] = {}
        self._roots: dict[str, list[str]] = {}
        self._projects: dict[str, Project] = {}
        #: Deleted projects: runs on offline machines may still name them in their manifest.
        self._deleted_projects: set[str] = set()
        self._load()

    # ------------------------------------------------------------------ records
    def all(self) -> list[RunRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def get(self, key: str) -> RunRecord | None:
        with self._lock:
            return self._records.get(key)

    def upsert(self, record: RunRecord, save: bool = True) -> None:
        with self._lock:
            self._records[record.key] = record
            self.add_root(record.machine_id, _parent(record.run_dir), save=False)
            self._adopt_projects([record])
            if save:
                self.save()

    def remove(self, key: str) -> None:
        with self._lock:
            self._records.pop(key, None)
            self.save()

    def replace_machine(self, machine_id: str, records: Iterable[RunRecord], roots: Iterable[str]) -> None:
        """Make ``records`` the complete set for ``machine_id`` under the scanned ``roots``."""
        roots = {r.rstrip("/\\") for r in roots}
        with self._lock:
            for key, rec in list(self._records.items()):
                if rec.machine_id == machine_id and _parent(rec.run_dir).rstrip("/\\") in roots:
                    del self._records[key]
            records = list(records)
            for rec in records:
                self._records[rec.key] = rec
            self._adopt_projects(records)
            self.save()

    # ------------------------------------------------------------------ projects
    def projects(self) -> list[Project]:
        with self._lock:
            return sorted(self._projects.values(), key=lambda p: (p.name.casefold(), p.created_at))

    def project(self, project_id: str | None) -> Project | None:
        with self._lock:
            return self._projects.get(project_id) if project_id else None

    def add_project(self, name: str) -> Project:
        project = Project.new(name)
        with self._lock:
            self._projects[project.id] = project
            self.save()
        return project

    def rename_project(self, project_id: str, name: str) -> None:
        with self._lock:
            if project_id in self._projects:
                self._projects[project_id].name = name.strip()
                self.save()

    def remove_project(self, project_id: str) -> None:
        with self._lock:
            self._projects.pop(project_id, None)
            self._deleted_projects.add(project_id)
            self.save()

    def project_of(self, record: RunRecord) -> Project | None:
        return self.project(record.project_id)

    def _adopt_projects(self, records: Iterable[RunRecord]) -> None:
        """Projects named by runs' manifests (e.g. after a reinstall) join the list."""
        for rec in records:
            ref = (rec.manifest or {}).get("project")
            pid = rec.project_id
            if pid and pid not in self._projects and pid not in self._deleted_projects:
                self._projects[pid] = Project(id=pid, name=str(ref.get("name") or "Project"),
                                              created_at=rec.created_at or 0.0)

    # ------------------------------------------------------------------ roots
    def roots(self, machine_id: str) -> list[str]:
        with self._lock:
            return list(self._roots.get(machine_id, []))

    def add_root(self, machine_id: str, root: str, save: bool = True) -> None:
        if not root:
            return
        with self._lock:
            roots = self._roots.setdefault(machine_id, [])
            if root not in roots:
                roots.insert(0, root)
                del roots[12:]
            if save:
                self.save()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for item in data.get("records", []):
            try:
                rec = RunRecord.from_dict(item)
            except (TypeError, ValueError, KeyError):
                continue
            self._records[rec.key] = rec
        self._roots = {k: list(v) for k, v in (data.get("roots") or {}).items()}
        for item in data.get("projects", []):
            try:
                project = Project(**{k: item[k] for k in ("id", "name", "created_at") if k in item})
            except (TypeError, KeyError):
                continue
            self._projects[project.id] = project
        self._deleted_projects = set(data.get("deleted_projects") or [])

    def save(self) -> None:
        with self._lock:
            payload = {
                "records": [r.to_dict() for r in self._records.values()],
                "roots": self._roots,
                "projects": [{"id": p.id, "name": p.name, "created_at": p.created_at}
                             for p in self._projects.values()],
                "deleted_projects": sorted(self._deleted_projects),
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
            os.replace(tmp, self.path)


def _parent(path: str) -> str:
    path = path.rstrip("/\\")
    cut = max(path.rfind("/"), path.rfind("\\"))
    return path[:cut] if cut > 0 else path
