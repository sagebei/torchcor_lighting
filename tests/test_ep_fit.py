"""EP stimuli point at files that exist: a site the input folder lacks becomes its first .vtx."""

from __future__ import annotations

import pytest

from torchcor_studio.core.models import InputInfo, VtxInfo
from torchcor_studio.core.simulations import ep
from torchcor_studio.core.simulations.common import Stimulus

FOLDER = InputInfo(path="/data/ep/ventricle", exists=True, mesh_name="heart",
                   vtx=[VtxInfo("LV_af.vtx", 0), VtxInfo("LV_pf.vtx", 12), VtxInfo("RV_sf.vtx", 9)])


def _params(*sites: str) -> ep.EPParams:
    return ep.EPParams(stimuli=[Stimulus(site=s) for s in sites])


def test_missing_sites_become_the_first_vtx_with_nodes() -> None:
    fitted = ep.fit_input(_params("stimulus.vtx", "RV_sf.vtx", ""), FOLDER)
    assert [s.site for s in fitted.stimuli] == ["LV_pf.vtx", "RV_sf.vtx", "LV_pf.vtx"]  # LV_af lists no nodes
    only_empty = InputInfo(path="/x", exists=True, mesh_name="m", vtx=[VtxInfo("a.vtx", 0)])
    assert ep.fit_input(_params(""), only_empty).stimuli[0].site == "a.vtx"


def test_valid_sites_and_unknown_folders_are_left_alone() -> None:
    assert ep.fit_input(_params("RV_sf.vtx", "LV_af.vtx"), FOLDER) is None  # already valid
    assert ep.fit_input(_params("x.vtx"), None) is None  # folder not inspected yet
    no_mesh = InputInfo(path="/data/ep", exists=True, subdirs=["atrium"])
    assert ep.fit_input(_params("x.vtx"), no_mesh) is None


@pytest.mark.parametrize(("before", "after", "fits"), [
    (_params("RV_sf.vtx"), _params("RV_"), False),  # typing a site: left alone
    (_params("RV_sf.vtx"), _params("RV_sf.vtx", "x.vtx"), True),  # row added
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
    assert step.params() == ep.EPParams() and step.demo().key == "ventricle"  # EP starts as the ventricle demo
    step.set_input(FOLDER)  # a folder lacking some of its sites: those become its first .vtx with nodes
    assert [s.site for s in step.params().stimuli] == ["LV_pf.vtx", "LV_pf.vtx", "LV_af.vtx", "RV_sf.vtx",
                                                       "LV_pf.vtx"]
    assert step.status_text() == "From Biventricle, five pacing sites"  # the fit moved the baseline
    step.apply_demo("atrium")  # its site is left blank: the folder's first .vtx with nodes
    assert step.params().stimuli[0].site == "LV_pf.vtx"
    assert step.status_text() == "From Atrium (Courtemanche)"
