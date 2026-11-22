"""Demos and panels of every simulation kind: what the Setup step is built from (no Qt)."""

from __future__ import annotations

import re

import pytest

from torchcor_studio.core.simulations import SimulationKind, ecg, ep, get_spec, specs
from torchcor_studio.core.simulations.fields import field_meta
from torchcor_studio.core.simulations.panels import (
    INPUT,
    OUTPUT,
    SLOTS,
    changes,
    leaves,
    node,
    owner,
    resolve,
    summary,
)
from torchcor_studio.core.simulations.types import resolve as field_type

KINDS = [str(k) for k in SimulationKind]
OURS = ("ep", "ecg", "mechanics", "em")


def _panels(kind: str):  # noqa: ANN202
    spec = get_spec(kind)
    return {p.key: p for p in resolve(spec.params_model, spec.panels)}


@pytest.mark.parametrize("kind", KINDS)
def test_first_demo_is_the_defaults(kind: str) -> None:
    spec = get_spec(kind)
    assert spec.demos[0].params() == spec.defaults()
    assert spec.demos[0].params() is not spec.demos[0].params()  # a fresh model every call


def test_demo_keys_unique_titles_short_summaries_one_sentence() -> None:
    for spec in specs():
        keys = [d.key for d in spec.demos]
        assert len(set(keys)) == len(keys) and all(re.fullmatch(r"[a-z]+(-[a-z]+)*", k) for k in keys)
        for demo in spec.demos:
            assert spec.demo(demo.key) is demo
            assert demo.title and len(demo.title) <= 60 and demo.title[0].isupper()
            assert demo.summary.endswith(".") and len(demo.summary) <= 120
            assert not re.search(r"(?<! al)\. [A-Z]", demo.summary), f"{demo.key}: one sentence"
    with pytest.raises(KeyError):
        get_spec("ep").demo("nothing")


def test_demo_cases_name_a_kind_folder() -> None:
    for spec in specs():
        for demo in spec.demos:
            if demo.case is None:
                continue
            kind, *rest = demo.case.split("/")
            assert kind == str(spec.kind) and len(rest) <= 1, demo.case  # "ep/ventricle" or "ecg"
    assert get_spec("ecg").demo("healthy").case == "ecg"  # the kind folder is the prepared data
    assert get_spec("em").demo("five-beats").case == "em"  # its mesh files sit in input/em


@pytest.mark.parametrize("kind", KINDS)
def test_panels_claim_every_leaf_once(kind: str) -> None:
    spec = get_spec(kind)
    resolved = resolve(spec.params_model, spec.panels)
    assert "other" not in [p.key for p in resolved], "a leaf is in no panel"
    claimed = [leaf for p in resolved for leaf in p.leaves]
    assert sorted(claimed) == sorted(leaves(spec.params_model))  # every leaf, once
    for panel, declared in zip(resolved, spec.panels, strict=True):
        assert panel.blocks and panel.title == declared.title
        if any(b.kind in ("input", "output") for b in panel.blocks):
            assert not panel.collapsed  # the folders are never hidden
        for path in declared.paths:
            raw = path[0] if isinstance(path, tuple) else path
            if raw not in SLOTS:
                node(spec.params_model, raw.split("."))  # resolves (KeyError otherwise)
    assert len({p.key for p in resolved}) == len(resolved)


def test_eyebrow_rules() -> None:
    cells = _panels("ep")["cells"]
    assert [(b.kind, b.eyebrow) for b in cells.blocks] == [("table", ""), ("fields", "Pre-pacing")]
    geometry = _panels("ep")["geometry"]  # the input folder, how it is read, the output folder
    assert [(b.kind, b.eyebrow) for b in geometry.blocks] == [("input", ""), ("fields", ""), ("output", "")]
    solver = _panels("ecg")["solver"]
    assert [b.eyebrow for b in solver.blocks] == ["Heart solve", "Lead fields"]
    models = _panels("ecg")["models"]
    assert [(b.kind, b.eyebrow) for b in models.blocks] == [
        ("fields", ""), ("table", "Conductivities"), ("table", "Conduction velocities"), ("fields", "")]
    assert models.blocks[1].caption.startswith("Heart conductivity per region")
    em_cells = _panels("em")["cells-and-contraction"]  # cell models, then two forced eyebrows
    assert [(b.kind, b.eyebrow) for b in em_cells.blocks] == [
        ("fields", ""), ("fields", "Pre-pacing"), ("fields", "Land contraction")]
    assert em_cells.blocks[1].leaves[-1] == ("heart", "reuse_dir")  # explicit claims join their block
    assert em_cells.blocks[2].leaves[0] == ("contraction", "lv_tags")  # the section claim's
    circulation = _panels("em")["circulation"]
    assert [b.eyebrow for b in circulation.blocks] == ["", "Tubes", "Valves", "Chambers", "Peripheral beds"]
    assert not circulation.blocks[1].caption and circulation.blocks[1].tip  # fixed: help as tooltip


def test_input_role_per_demo() -> None:
    def role(kind: str, params) -> str:  # noqa: ANN001
        return get_spec(kind).input_role(params)[0]

    assert all(role("ep", d.params()) == "required" for d in get_spec("ep").demos)  # a mesh folder
    assert role("ecg", ecg.ECGParams()) == "required"
    assert all(role("mechanics", d.params()) == "unused" for d in get_spec("mechanics").demos)
    assert all(role("em", d.params()) == "required" for d in get_spec("em").demos)
    assert role("script", get_spec("script").defaults()) == "optional"
    assert all(get_spec(k).input_role(get_spec(k).defaults())[1] or k in ("ep", "mechanics") for k in KINDS)


def test_every_first_panel_ends_with_the_output_folder() -> None:
    for spec in specs():
        first, *rest = resolve(spec.params_model, spec.panels)
        assert first.blocks[-1].kind == "output", spec.kind
        assert not any(b.kind == "output" for p in rest for b in p.blocks), spec.kind
        assert owner([first, *rest], OUTPUT) is first, spec.kind


def test_collapsed_panel_summaries() -> None:
    for spec in specs():
        params = spec.defaults()
        for panel in resolve(spec.params_model, spec.panels):
            if panel.collapsed:
                assert summary(spec.params_model, panel, params), (spec.kind, panel.key)
    em = get_spec("em")
    circulation = _panels("em")["circulation"]
    text = summary(em.params_model, circulation, em.defaults())
    assert "4 tubes" in text and "6 valves" in text and text.startswith("Reference flow 85 mL/s")
    support = summary(em.params_model, _panels("em")["supports-and-start-pressures"], em.defaults())
    assert support == "3 springs · Start pressures LV 9.538, RV 5.937, LA 9.538, RA 5.937 mmHg"
    solver = summary(ep.EPParams, _panels("ep")["solver"], ep.EPParams())
    assert solver == "Single (float32) · CG tolerance 1e-5 · Iteration limit 100"  # short exponents


def _sections(cls, seen: set) -> list:  # noqa: ANN001
    """Every section class of a parameter tree, table rows included."""
    if cls in seen:
        return []
    seen.add(cls)
    out = [cls]
    for info in cls.model_fields.values():
        ftype = field_type(info.annotation)
        if ftype.kind in ("section", "rows"):
            out += _sections(ftype.inner, seen)
    return out


def test_field_metadata_sanity() -> None:
    for spec in specs():
        for cls in _sections(spec.params_model, set()):
            runs: list[tuple[str, str]] = []  # (group, visible_if) of each run of a group
            previous = ""
            for name, info in cls.model_fields.items():
                meta, ftype = field_meta(cls, name), field_type(info.annotation)
                where = f"{cls.__name__}.{name}"
                assert not meta.get("fixed") or ftype.kind == "rows", f"{where}: fixed on a non-list"
                assert meta.get("span") is None or 0 <= meta["span"] <= 4, where
                group = meta.get("group", "")
                if group and group != previous:
                    runs.append((group, str(meta.get("visible_if"))))
                previous = group
                if ftype.kind == "rows":  # table columns with long titles have a short one
                    for column in ftype.inner.model_fields:
                        cmeta = field_meta(ftype.inner, column)
                        assert len(cmeta["title"]) <= 20 or cmeta.get("short"), \
                            f"{ftype.inner.__name__}.{column} needs a short title"
            assert len(runs) == len(set(runs)), f"{cls.__name__}: a group is split: {runs}"


def test_numbers_read_plainly() -> None:
    from torchcor_studio.core.simulations.panels import number

    assert [number(v) for v in (1000.0, 85.0, 9.53762, 0.059, 12345.6)] == ["1000", "85", "9.538", "0.059", "12346"]
    assert [number(v) for v in (1e-5, 1e-10, 2.5e7)] == ["1e-5", "1e-10", "2.5e7"]


def test_input_folder_issues_point_at_the_slot() -> None:
    em, ecg_spec = get_spec("em"), get_spec("ecg")
    assert any(i.field == INPUT and i.severity == "error" for i in em.validate(em.defaults(), None))
    assert any(i.field == INPUT for i in ecg_spec.validate(ecg.ECGParams(), None))
    assert all(i.field == INPUT for i in get_spec("ep").validate(ep.ventricle(), None)
               if "mesh (.pts" in i.message)


def test_ecg_heart_ep_groups_three_panels() -> None:
    from torchcor_studio.core.simulations.panels import Group, Panel, resolve

    panels = _panels("ecg")
    grouped = [key for key, panel in panels.items() if panel.group is not None]
    assert grouped == ["cells", "models", "pacing-sites"]
    assert {panels[key].group.title for key in grouped} == {"Heart EP"}
    assert panels["models"].blocks[0].leaves[0] == ("heart", "solver")
    spec = get_spec("ecg")
    split = Group("Split", "heart")
    with pytest.raises(ValueError, match="consecutive"):
        resolve(spec.params_model, (Panel("A", "x", ("data",), group=split), Panel("B", "x", ("heart",)),
                                    Panel("C", "x", ("conductor",), group=split)))


def test_ecg_heart_and_torso_is_the_prepared_folder() -> None:
    """No choice of model: the panel is the input folder, its parts and the output folder."""
    spec = get_spec("ecg")
    panel = next(p for p in spec.panels if p.title == "Heart and torso")
    assert panel.paths[:2] == (INPUT, "data") and "data.source" not in panel.paths
    assert "source" not in type(spec.defaults().data).model_fields
    assert not any(demo.case is None for demo in spec.demos)  # each reads the prepared folder


def test_owner_and_changes() -> None:
    spec = get_spec("ep")
    resolved = resolve(spec.params_model, spec.panels)
    assert owner(resolved, "stimuli.0.site").key == "pacing"
    assert owner(resolved, "model.conductivities").key == "conduction"
    assert owner(resolved, "timing").key == "time-and-outputs"  # a section: its first leaf's panel
    assert owner(resolved, "nothing") is None
    assert owner(resolved, INPUT).key == "geometry"  # input-folder issues: the panel with the slot
    mechanics = get_spec("mechanics")
    assert owner(resolve(mechanics.params_model, mechanics.panels), INPUT) is None  # no slot
    before = ep.ventricle()
    after = before.model_copy(deep=True)
    after.stimuli[3].start = 8.0
    after.timing.duration = 600.0
    after.cells.append(ep.CellGroup())
    labels = [c.label for c in changes(spec.params_model, resolved, before.model_dump(), after.model_dump())]
    assert labels == ["Cells › Cell models: 1 → 2 cell models",
                      "Pacing › Stimulus 4 › Start: 5 → 8 ms",
                      "Time and outputs › Duration: 500 → 600 ms"]
