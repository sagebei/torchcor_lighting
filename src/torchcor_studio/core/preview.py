"""Mesh previews: a light boundary surface + pacing sites of an input folder, cached locally."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from torchcor_studio.core.machines.base import Machine
from torchcor_studio.core.paths import cache_dir


@dataclass
class MeshPreview:
    points: np.ndarray  # (N, 3) float32, mm
    triangles: np.ndarray  # (M, 3) int32
    regions: np.ndarray  # (M,) int32 region tag per triangle
    sites: dict[str, np.ndarray] = field(default_factory=dict)  # vtx name -> (k, 3) points

    @property
    def region_ids(self) -> list[int]:
        return sorted(int(r) for r in np.unique(self.regions))

    @classmethod
    def load(cls, path: Path) -> MeshPreview:
        with np.load(path, allow_pickle=False) as data:
            offsets = data["vtx_offsets"]
            names = [str(n) for n in data["vtx_names"]]
            pts = data["vtx_points"]
            sites = {name: pts[offsets[i]:offsets[i + 1]] for i, name in enumerate(names)}
            return cls(points=data["points"].astype(np.float32), triangles=data["triangles"].astype(np.int32),
                       regions=data["regions"].astype(np.int32), sites=sites)


def build_preview(machine: Machine, mesh_dir: str) -> MeshPreview:
    """Extract the preview on the machine (where the mesh lives) and bring it here."""
    mesh_dir = machine.expand(mesh_dir)
    key = hashlib.sha1(f"{machine.id}:{mesh_dir}".encode()).hexdigest()[:16]
    local = cache_dir() / "previews" / f"{key}.npz"
    local.parent.mkdir(parents=True, exist_ok=True)
    if machine.profile.is_local:
        machine.runtime("preview", mesh_dir, str(local), timeout=900)
    else:
        remote = machine.join(machine.runtime_dir(), f"preview-{key}.npz")
        machine.runtime("preview", mesh_dir, remote, timeout=900)
        try:
            machine.download(remote, str(local))
        finally:
            machine.remove(remote)
    return MeshPreview.load(local)
