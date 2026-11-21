"""Look at an input mesh before running: regions in colour, pacing sites marked."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QDialog, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from torchcor_studio.core.preview import MeshPreview, build_preview
from torchcor_studio.ui.format import number
from torchcor_studio.ui.results import colormaps
from torchcor_studio.ui.tasks import run_async
from torchcor_studio.ui.theme import theme
from torchcor_studio.ui.widgets import Button, Card, ElidedLabel, EmptyState, Spinner, label

REGION_MAP = "turbo"


class _Swatch(QWidget):
    def __init__(self, color: str, round_: bool = False) -> None:
        super().__init__()
        self._color, self._round = color, round_
        self.setFixedSize(12, 12)

    def paintEvent(self, _e) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._color))
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if self._round:
            painter.drawEllipse(r)
        else:
            painter.drawRoundedRect(r, 3, 3)


def _legend_row(color: str, text: str, detail: str, round_: bool = False) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(8)
    row.addWidget(_Swatch(color, round_))
    row.addWidget(label(text), 1)
    row.addWidget(label(detail, "faint"))
    return row


class MeshPreviewDialog(QDialog):
    def __init__(self, machine, mesh_dir: str, parent: QWidget | None = None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("Mesh preview")
        self.resize(1180, 760)
        self.machine = machine
        self.mesh_dir = mesh_dir
        self.viewer = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.stage = QVBoxLayout()
        self.stage.setContentsMargins(0, 0, 0, 0)
        holder = QWidget()
        holder.setLayout(self.stage)
        lay.addWidget(holder, 1)
        side = QWidget()
        side.setFixedWidth(300)
        side.setObjectName("Sidebar")
        side.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.side = QVBoxLayout(side)
        self.side.setContentsMargins(18, 18, 18, 18)
        self.side.setSpacing(14)
        self.side.addWidget(label("Mesh preview", "h3"))
        self.path_label = ElidedLabel(mesh_dir, "mono", Qt.TextElideMode.ElideMiddle)
        self.side.addWidget(self.path_label)
        self.details = QVBoxLayout()
        self.details.setSpacing(14)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setLayout(self.details)
        scroll.setWidget(inner)
        self.side.addWidget(scroll, 1)
        self.side.addWidget(Button("Close", variant="primary", on_click=self.accept))
        lay.addWidget(side)
        self._show_loading()
        run_async(build_preview, machine, mesh_dir, on_done=self._show, on_error=self._failed, owner=self)

    # ------------------------------------------------------------------ states
    def _show_loading(self) -> None:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(Spinner(24), 0, Qt.AlignmentFlag.AlignHCenter)
        v.addWidget(label("Extracting the surface on the compute machine…", "muted"), 0, Qt.AlignmentFlag.AlignHCenter)
        self.stage.addWidget(box)
        self._loading = box

    def _failed(self, exc) -> None:  # noqa: ANN001
        self._loading.deleteLater()
        self.stage.addWidget(EmptyState("alert", "Couldn't preview this mesh", str(exc)))

    def _show(self, preview: MeshPreview) -> None:
        from torchcor_studio.ui.results.viewer3d import Viewer3D

        self._loading.deleteLater()
        self.viewer = Viewer3D(self)
        self.stage.addWidget(self.viewer)
        if not self.viewer.available:
            return
        regions = preview.region_ids
        index_of = {r: i for i, r in enumerate(regions)}
        point_region = np.zeros(len(preview.points), np.float32)
        tri_index = np.vectorize(index_of.get)(preview.regions) if len(preview.regions) else np.zeros(0)
        for k in range(3):
            point_region[preview.triangles[:, k]] = tri_index
        self.viewer.set_mesh(preview.points, preview.triangles)
        k = max(1, len(regions))
        if len(regions) > 1:  # categorical: each region at the centre of its colour band
            self.viewer.set_scalars(point_region, clim=(-0.5, k - 0.5), cmap=REGION_MAP, title="Region", unit="")
            self.viewer.colorbar.hide()

        p = theme().palette
        cmap = colormaps.get(REGION_MAP)
        stats = Card("Geometry", padding=14)
        stats.addWidget(label(f"{number(len(preview.points))} surface points · "
                              f"{number(len(preview.triangles))} triangles", "muted", wrap=True))
        self.details.addWidget(stats)

        reg_card = Card("Regions", f"{len(regions)} tag{'s' if len(regions) != 1 else ''}", padding=14)
        counts = {r: int(np.count_nonzero(preview.regions == r)) for r in regions}
        total = max(1, len(preview.regions))
        for i, r in enumerate(regions):
            color = cmap.color((i + 0.5) / k).name() if len(regions) > 1 else p.text2
            reg_card.addLayout(_legend_row(color, f"Region {r}", f"{100 * counts[r] / total:.0f}%"))
        self.details.addWidget(reg_card)

        sites = [(name, pts) for name, pts in preview.sites.items() if len(pts)]
        site_card = Card("Pacing sites", f"{len(sites)} .vtx file{'s' if len(sites) != 1 else ''}", padding=14)
        indices, colors = [], []
        for i, (name, pts) in enumerate(sites):
            color = p.series[i % len(p.series)]
            centroid = pts.mean(axis=0)
            nearest = int(np.argmin(((preview.points - centroid) ** 2).sum(axis=1)))
            indices.append(nearest)
            colors.append(color)
            site_card.addLayout(_legend_row(color, name, f"{number(len(pts))} nodes", round_=True))
        if not sites:
            site_card.addWidget(label("No .vtx files in this folder.", "caption"))
        self.details.addWidget(site_card)
        self.details.addStretch(1)
        if indices:
            self.viewer.set_markers(indices, colors)
        self.viewer.set_active(True)
        QTimer.singleShot(60, self.viewer.reset_camera)  # after the dialog has its final size

    def done(self, result: int) -> None:  # noqa: D401 - Qt override
        if self.viewer is not None:
            self.viewer.release()
        super().done(result)
