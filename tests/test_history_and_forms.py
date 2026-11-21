"""Run history persistence, run ids, and the schema-driven parameter form."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from torchcor_studio.core.models import InputInfo, RunRecord, RunState, RunStatus, VtxInfo
from torchcor_studio.core.runs import RunHistory
from torchcor_studio.core.runs.protocol import new_run_id, slugify
from torchcor_studio.core.simulations.fields import Section, field_meta, param


def _record(machine: str, run_dir: str, created: float, state: RunState = RunState.COMPLETED) -> RunRecord:
    return RunRecord(id=Path(run_dir).name, name=Path(run_dir).name, kind="ep", machine_id=machine,
                     run_dir=run_dir, created_at=created, status=RunStatus(state=state))


def test_history_roundtrip_and_ordering(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    h = RunHistory(path)
    h.upsert(_record("local", "/runs/a", 1.0))
    h.upsert(_record("local", "/runs/b", 2.0, RunState.RUNNING))
    again = RunHistory(path)
    assert [r.id for r in again.all()] == ["b", "a"]
    assert again.all()[0].status.state == RunState.RUNNING
    assert again.roots("local") == ["/runs"]


def test_history_replace_machine_only_touches_scanned_roots(tmp_path: Path) -> None:
    h = RunHistory(tmp_path / "h.json")
    h.upsert(_record("gpu", "/data/runs/old", 1.0))
    h.upsert(_record("gpu", "/elsewhere/keep", 1.0))
    h.upsert(_record("local", "/data/runs/local", 1.0))
    h.replace_machine("gpu", [_record("gpu", "/data/runs/new", 3.0)], ["/data/runs"])
    ids = sorted(r.id for r in h.all())
    assert ids == ["keep", "local", "new"]


def test_run_ids_are_sortable_and_safe() -> None:
    a = new_run_id("Ventricle — pacing (TTP)!", now=1_700_000_000)
    assert a.startswith("2023") and "ventricle-pacing-ttp" in a
    assert slugify("   ") == "run"


class Inner(Section):
    value: float = param(1.5, "Value", unit="ms", min=0)
    kind: Literal["a", "b"] = param("a", "Kind", choices={"a": "Alpha", "b": "Beta"})
    extra: float = param(2.0, "Extra", visible_if={"kind": "b"})


class Row(Section):
    site: str = param("x.vtx", "Site", widget="vtx")
    start: float = param(0.0, "Start", unit="ms")


class Model(Section):
    inner: Inner = param(default_factory=Inner, title="Inner")
    rows: list[Row] = param(default_factory=lambda: [Row()], title="Rows")
    regions: list[int] = param([], "Regions", widget="regions")


def test_field_meta_exposes_ui_hints() -> None:
    meta = field_meta(Inner, "value")
    assert meta["unit"] == "ms" and meta["min"] == 0 and meta["default"] == 1.5
    assert field_meta(Inner, "extra")["visible_if"] == {"kind": "b"}


def test_param_form_validates_and_tracks_visibility(themed, qtbot) -> None:
    from torchcor_studio.ui.forms import ParamForm, ParamSummary

    form = ParamForm(Model)
    qtbot.addWidget(form)
    form.show()
    seen: list = []
    form.changed.connect(seen.append)
    errors: list = []
    form.errors.connect(errors.append)

    extra = form.editor(("inner", "extra"))
    assert not extra.widget.isVisible()
    kind = form.editor(("inner", "kind"))
    kind.set("b")
    kind.emit()
    assert extra.widget.isVisible()
    assert seen and seen[-1].inner.kind == "b"

    value = form.editor(("inner", "value"))
    value.focus_target().setValue(-1)  # below min=0 → spin box clamps, still valid
    assert form.value() is not None and form.value().inner.value == 0

    form.set_input_info(InputInfo(path="/m", vtx=[VtxInfo("LV.vtx", 3)], regions=[1, 2]))
    rows = form.table(("rows",))
    rows.add_row()
    assert len(form.data()["rows"]) == 2 and rows.row_count() == 2
    rows.remove_row(0)
    assert len(form.data()["rows"]) == 1 and rows.row_count() == 1

    summary = ParamSummary(Model, Model().model_dump(mode="json"))
    qtbot.addWidget(summary)


class Cond(Section):
    il: float = param(0.17, "il")
    el: float = param(0.62, "el", visible_if={"../mode": "bidomain"})


class Owner(Section):
    mode: Literal["mono", "bidomain"] = param("bidomain", "Mode", choices={"mono": "Mono", "bidomain": "Bi"})
    rows: list[Cond] = param(default_factory=lambda: [Cond()], title="Rows")


class Top(Section):
    owner: Owner = param(default_factory=Owner, title="Owner")


def test_row_fields_can_depend_on_the_owning_section(themed, qtbot) -> None:
    from torchcor_studio.ui.forms import ParamForm

    form = ParamForm(Top)
    qtbot.addWidget(form)
    form.show()
    rows = form.table(("owner", "rows"))
    el_editor = rows.cell(0, "el")
    assert el_editor.widget.isVisible()
    mode = form.editor(("owner", "mode"))
    mode.set("mono")
    mode.emit()
    assert not el_editor.widget.isVisible()
    assert not rows.header("el").isVisible()


def test_segmented_control_click_slides_without_moving(themed, qtbot) -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    from torchcor_studio.ui.widgets import SegmentedControl

    host = QWidget()
    lay = QVBoxLayout(host)
    lay.setContentsMargins(40, 30, 0, 0)
    ctl = SegmentedControl([("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")], value="a")
    lay.addWidget(ctl)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    before = ctl.pos()
    seen = []
    ctl.changed.connect(seen.append)
    third = ctl._rects()[2].center()
    qtbot.mouseClick(ctl, Qt.MouseButton.LeftButton, pos=QPoint(int(third.x()), int(third.y())))
    qtbot.wait(300)  # let the slide animation finish
    assert seen == ["c"] and ctl.value() == "c"
    assert ctl.pos() == before  # the animation must not move the widget itself


def test_theme_switch_after_widgets_are_deleted(qapp, qtbot) -> None:
    from torchcor_studio.ui.theme import theme
    from torchcor_studio.ui.widgets import Badge, MeterBar, SegmentedControl, Sparkline, StatusDot

    widgets = [Badge("x"), MeterBar(0.3), Sparkline(), StatusDot("running"), SegmentedControl([("a", "A")])]
    for w in widgets:
        w.deleteLater()
    from PySide6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    theme().set_mode("light")  # must not touch deleted widgets
    theme().set_mode("dark")
