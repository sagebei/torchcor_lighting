"""EP stimuli point at files that exist: the slab's, or the input folder's first .vtx."""

from __future__ import annotations

from dataclasses import replace

import pytest

from torchcor_studio.core.models import InputInfo, VtxInfo
from torchcor_studio.core.simulations import ep
from torchcor_studio.core.simulations.common import Stimulus

FOLDER = InputInfo(path="/data/ep/ventricle", exists=True, mesh_name="heart",
                   vtx=[VtxInfo("LV_af.vtx", 0), VtxInfo("LV_pf.vtx", 12), VtxInfo("RV_sf.vtx", 9)])


def _params(source: str, *sites: str) -> ep.EPParams:
    return ep.EPParams(geometry=ep.EPGeometry(source=source), stimuli=[Stimulus(site=s) for s in sites])


def test_missing_sites_become_the_first_vtx_with_nodes() -> None:
    fitted = ep.fit_input(_params("input", "stimulus.vtx", "RV_sf.vtx", ""), FOLDER)
    assert [s.site for s in fitted.stimuli] == ["LV_pf.vtx", "RV_sf.vtx", "LV_pf.vtx"]  # LV_af lists no nodes
    only_empty = InputInfo(path="/x", exists=True, mesh_name="m", vtx=[VtxInfo("a.vtx", 0)])
    assert ep.fit_input(_params("input", ""), only_empty).stimuli[0].site == "a.vtx"


def test_slab_sites_and_unknown_folders() -> None:
    assert ep.fit_input(_params("slab", "LV_pf.vtx"), FOLDER).stimuli[0].site == "stimulus.vtx"
    assert ep.fit_input(_params("slab", "left_face.vtx"), FOLDER) is None  # already valid
    assert ep.fit_input(_params("input", "x.vtx"), None) is None  # folder not inspected yet
    no_mesh = InputInfo(path="/data/ep", exists=True, subdirs=["atrium"])
    assert ep.fit_input(_params("input", "x.vtx"), no_mesh) is None


@pytest.mark.parametrize(("before", "after", "fits"), [
    (_params("input", "RV_sf.vtx"), _params("input", "RV_"), False),  # typing a site: left alone
    (_params("slab", "stimulus.vtx"), _params("input", "stimulus.vtx"), True),  # geometry switched
    (_params("input", "RV_sf.vtx"), _params("input", "RV_sf.vtx", "stimulus.vtx"), True),  # row added
])
def test_only_edits_that_can_break_sites_are_fitted(before, after, fits) -> None:  # noqa: ANN001
    assert (ep.fit_input(after, FOLDER, before) is not None) is fits


def _step(qtbot):  # noqa: ANN001, ANN202
    from torchcor_studio.ui.pages.new_run.setup_step import SetupStep

    step = SetupStep(lambda *_: None)
    qtbot.addWidget(step)
    return step


def test_the_setup_step_applies_it(themed, qtbot) -> None:
    step = _step(qtbot)
    assert step.params().geometry.source == "slab" and step.params().stimuli[0].site == "stimulus.vtx"
    step.set_input(FOLDER)  # a new folder with a mesh: the untouched draft defaults to it
    assert step.params().geometry.source == "input" and step.params().stimuli[0].site == "LV_pf.vtx"
    assert step.demo() is None and step.status_text() == "Defaults, fitted to ventricle"
    geometry = step.form().editor(("geometry", "source"))
    geometry.set("slab")  # the user picks the built-in slab: kept, sites follow
    geometry.emit()
    qtbot.waitUntil(lambda: step.params().stimuli[0].site == "stimulus.vtx", timeout=3000)
    assert step.params().geometry.source == "slab"
    geometry.set("input")  # "Mesh in the input folder" again
    geometry.emit()
    qtbot.waitUntil(lambda: step.params().stimuli[0].site == "LV_pf.vtx", timeout=3000)
    step.apply_demo("atrium")  # its site is left blank: the folder's first .vtx with nodes
    assert step.params().stimuli[0].site == "LV_pf.vtx"
    assert step.status_text() == "From Atrium (Courtemanche)"  # the fit moved the baseline too


def test_a_new_folder_switches_a_pristine_slab_to_its_mesh_only(themed, qtbot) -> None:
    step = _step(qtbot)
    step.set_input(FOLDER)
    assert step.params().geometry.source == "input"
    step.load("ep", ep.EPParams().model_dump())  # back on the slab, as a run would be
    step.set_input(FOLDER)  # the same folder inspected again (a reconnect): left alone
    assert step.params().geometry.source == "slab"

    edited = _step(qtbot)  # an edited set-up: left alone
    form = edited.form()
    duration = form.editor(("timing", "duration"))
    duration.set(250.0)
    duration.emit()
    edited.set_input(FOLDER)
    assert edited.params().geometry.source == "slab" and edited.params().timing.duration == 250

    chosen = _step(qtbot)  # a demo the user chose, even the slab: left alone
    chosen.apply_demo("slab")
    chosen.set_input(FOLDER)
    assert chosen.params().geometry.source == "slab" and chosen.demo().key == "slab"

    again = _step(qtbot)  # a folder inspected again (it now holds a mesh): left alone
    again.set_input(InputInfo(path=FOLDER.path, exists=True))
    again.set_input(FOLDER)
    assert again.params().geometry.source == "slab"

    moved = _step(qtbot)  # fitted to one folder, then to another: the status names the last
    moved.set_input(FOLDER)
    moved.set_input(replace(FOLDER, path="/data/ep/atrium"))
    assert moved.status_text() == "Defaults, fitted to atrium"

    # the core still switches when asked, and only for a mesh
    fitted = ep.fit_input(ep.EPParams(), FOLDER, None, new_input=True)
    assert fitted.geometry.source == "input" and fitted.stimuli[0].site == "LV_pf.vtx"
    assert ep.fit_input(ep.EPParams(), FOLDER, None) is None
    no_mesh = InputInfo(path="/data/ep", exists=True, subdirs=["atrium"])
    assert ep.fit_input(ep.EPParams(), no_mesh, None, new_input=True) is None
