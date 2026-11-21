"""Results visualisation: the bundle loader, playback, and every view (offscreen)."""

from __future__ import annotations

import json
import shutil
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from fixtures.synthetic_bundles import (
    write_ecg_bundle,
    write_em_bundle,
    write_ep_bundle,
    write_mechanics_bundle,
)
from torchcor_studio.core.results import (
    BundleError,
    BundleNotFound,
    Circulation,
    FrameStack,
    SummaryItem,
    cardiac_summary,
    format_value,
    load_bundle,
    write_ecg_csv,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

#: Finished views are parked, not deleted: several shared widgets (SegmentedControl,
#: ToggleSwitch, …) connect lambdas to the global theme signal, which raise in the Qt
#: event loop once their widget is gone and the next test switches the theme.
_PARKED: list = []


def _park(widget) -> None:
    if hasattr(widget, "release"):
        widget.release()
    widget.hide()
    _PARKED.append(widget)


def _apply_theme(qtbot, mode: str) -> None:
    """Switch the theme, tolerating only stale slots of widgets that other tests deleted."""
    from torchcor_studio.ui.theme import theme

    with qtbot.capture_exceptions() as caught:
        theme().set_mode(mode)
    for _type, error, _tb in caught:
        if not (isinstance(error, RuntimeError) and "already deleted" in str(error)):
            raise error


@pytest.fixture
def studio(qapp, qtbot):
    """The studio theme (dark), applied like the shared ``themed`` fixture."""
    _apply_theme(qtbot, "dark")
    return qapp


@pytest.fixture(scope="module")
def bundles(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("bundles")
    return {
        "ep": write_ep_bundle(root / "ep"),
        "ecg": write_ecg_bundle(root / "ecg"),
        "mechanics": write_mechanics_bundle(root / "mechanics"),
        "em": write_em_bundle(root / "em"),
    }


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _rewrite(path: Path, **changes) -> None:
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays.update(changes)
    np.savez_compressed(path, **arrays)


# =============================================================================== loader
def test_index_is_read_and_datasets_load_lazily(bundles):
    bundle = load_bundle(bundles["ep"])
    assert bundle.kind == "ep"
    assert bundle.title == "Biventricular activation"
    assert [item.label for item in bundle.summary][:2] == ["Total activation", "Mean APD₉₀"]
    assert bundle.notes
    assert not bundle.loaded("surface") and not bundle.loaded("ep")
    ep = bundle.ep
    assert bundle.loaded("surface") and bundle.loaded("ep")
    n = bundle.surface.n_points
    assert bundle.surface.points.shape == (n, 3) and bundle.surface.triangles.dtype == np.int32
    assert ep.vm.n_points == n and ep.vm.n_frames == len(ep.times)


def test_frames_stay_float16_until_asked(bundles):
    vm = load_bundle(bundles["ep"]).ep.vm
    assert vm.nbytes == vm.n_frames * vm.n_points * 2  # still float16 in memory
    frame = vm.frame(3)
    assert frame.dtype == np.float32 and frame.shape == (vm.n_points,)
    out = np.empty(vm.n_points, np.float32)
    assert vm.frame(4, out) is out
    np.testing.assert_allclose(vm.blend(3.5), 0.5 * (vm.frame(3) + vm.frame(4)), atol=1e-3)
    assert vm.column(0).shape == (vm.n_frames,)
    lo, hi = vm.range
    assert -90 < lo < -80 and 20 < hi < 40


def test_run_folder_is_accepted(bundles, tmp_path):
    run = tmp_path / "run"
    _copy(bundles["ep"], run / "viz")
    assert load_bundle(run).root == run / "viz"


def test_probe_nodes_map_to_surface_points(bundles):
    bundle = load_bundle(bundles["ep"])
    local = bundle.surface.local_index(bundle.ep.traces.node_ids)
    assert (local >= 0).all()
    np.testing.assert_array_equal(bundle.surface.node_ids[local], bundle.ep.traces.node_ids)
    assert bundle.surface.local_index([-5])[0] == -1


def test_other_kinds_parse(bundles):
    ecg = load_bundle(bundles["ecg"])
    assert ecg.ecg.lead_names[:3] == ("I", "II", "III") and ecg.ecg.leads.shape == (12, 601)
    assert ecg.ecg.lead("avr") is not None and ecg.has("ep") and ecg.has("surface")
    mech = load_bundle(bundles["mechanics"]).mechanics
    assert mech.frame_label == "Load factor λ" and mech.frame_unit == ""
    assert set(mech.fields) == {"strain", "von_mises"}
    assert mech.convergence is not None and mech.convergence.newton.sum() == 45
    assert mech.max_displacement[-1] == pytest.approx(np.linalg.norm(mech.displacement[-1], axis=1).max())
    em = load_bundle(bundles["em"]).em
    assert em.circulation.chambers == ("lv", "rv", "la", "ra")
    lv = em.circulation.metrics("lv")
    assert lv.edv == pytest.approx(120, abs=1) and lv.esv == pytest.approx(50, abs=1)
    assert lv.ejection_fraction == pytest.approx(58.3, abs=0.5) and lv.peak_pressure == pytest.approx(120, abs=3)


def test_missing_bundle_is_reported_as_not_found(tmp_path):
    with pytest.raises(BundleNotFound):
        load_bundle(tmp_path)


@pytest.mark.parametrize(("breakage", "message"), [
    (lambda d: (d / "index.json").write_text("{not json"), "not valid JSON"),
    (lambda d: _edit_index(d, kind="cfd"), "Unknown result kind"),
    (lambda d: _edit_index(d, format=99), "format 99"),
    (lambda d: (d / "ep.npz").unlink(), "missing"),
    (lambda d: _edit_index(d, datasets={"surface": "surface.npz", "ep": "../ep.npz"}), "outside the bundle"),
    (lambda d: _edit_index(d, datasets={"surface": "surface.npz"}), "no “ep” dataset"),
])
def test_broken_index_is_rejected_on_load(bundles, tmp_path, breakage, message):
    folder = _copy(bundles["ep"], tmp_path / "viz")
    breakage(folder)
    with pytest.raises(BundleError, match=message):
        load_bundle(folder)


@pytest.mark.parametrize(("breakage", "message"), [
    (lambda d: (d / "ep.npz").write_bytes(b"\x00garbage"), "damaged"),
    (lambda d: _rewrite(d / "ep.npz", vm=np.zeros((3, 7), np.float16), times=np.arange(3.0)), "surface has"),
    (lambda d: _rewrite(d / "ep.npz", times=np.arange(2.0)), "frames but times"),
    (lambda d: _rewrite(d / "surface.npz", triangles=np.array([[0, 1, 10**7]], np.int32)), "outside"),
    (lambda d: _rewrite(d / "surface.npz", points=np.zeros((4, 2), np.float32)), r"\(N, 3\)"),
])
def test_corrupt_datasets_fail_with_clear_errors(bundles, tmp_path, breakage, message):
    folder = _copy(bundles["ep"], tmp_path / "viz")
    breakage(folder)
    bundle = load_bundle(folder)  # the index itself is fine
    with pytest.raises(BundleError, match=message):
        bundle.preload()


def _edit_index(folder: Path, **changes) -> None:
    path = folder / "index.json"
    index = json.loads(path.read_text())
    index.update(changes)
    path.write_text(json.dumps(index))


def test_cardiac_summary_prefers_bundle_values():
    circ = Circulation(np.arange(3.0), {"lv": np.array([5.0, 118.0, 9.0])}, {"lv": np.array([120.0, 60.0, 50.0])})
    items = cardiac_summary([SummaryItem("LV ejection fraction", 61.0, "%"), SummaryItem("Simulated", 800.0, "ms")],
                            circ)
    assert [i.label for i in items] == ["EDV", "ESV", "Stroke volume", "LV ejection fraction", "Peak LV pressure",
                                        "Simulated"]
    assert items[3].value == 61.0 and items[2].value == 70.0


def test_format_value():
    assert format_value(12345) == "12,345"
    assert format_value(98.0) == "98"
    assert format_value(41.67) == "41.7"
    assert format_value(1.5) == "1.5"
    assert format_value(0.9985641972225969) == "0.999"  # Min J must not round to 1
    assert format_value(0.00421) == "0.00421"
    assert format_value(-90.03) == "\u221290"
    assert format_value(float("nan")) == "—"
    assert format_value(True) == "Yes" and format_value("yes") == "Yes"
    assert format_value("cuda:0") == "cuda:0"


def test_frame_stack_rejects_bad_shapes():
    with pytest.raises(BundleError):
        FrameStack(np.zeros(5, np.float16), "vm")


def test_ecg_csv_round_trip(bundles, tmp_path):
    ecg = load_bundle(bundles["ecg"]).ecg
    path = write_ecg_csv(tmp_path / "ecg.csv", ecg)
    header = path.read_text().splitlines()[0].split(",")
    assert header[:3] == ["t_ms", "I_mV", "II_mV"] and len(header) == 13
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(table[:, 2], ecg.leads[1], atol=1e-4)


# =============================================================================== playback
def test_controller_plays_loops_and_stops(qtbot, studio):
    from torchcor_studio.ui.results.playback import PlaybackController

    controller = PlaybackController(10, fps=60.0)
    frames: list[int] = []
    controller.frame_changed.connect(frames.append)
    controller.play()
    qtbot.waitUntil(lambda: any(b < a for a, b in pairwise(frames)), timeout=3000)
    assert controller.playing  # wrapped past the last frame and kept going
    controller.pause()
    controller.set_loop(False)
    controller.seek(7)
    controller.play()
    qtbot.waitUntil(lambda: not controller.playing, timeout=3000)
    assert controller.frame == 9
    controller.seek(99)
    assert controller.frame == 9
    controller.set_loop(True)
    controller.step(1)
    assert controller.frame == 0 and not controller.playing


def test_timeline_follows_controller_and_space_toggles(qtbot, studio):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    from torchcor_studio.ui.results.playback import PlaybackController, TimelineBar

    host = QWidget()
    controller = PlaybackController(20, parent=host)
    bar = TimelineBar(controller, lambda i: f"t = {i * 5} ms", parent=host)
    host.show()
    host.activateWindow()
    qtbot.waitUntil(host.isActiveWindow, timeout=3000)
    controller.seek(6)
    assert bar.slider.value() == 6 and bar.time_label.text() == "t = 30 ms"
    bar.slider.setValue(11)
    assert controller.frame == 11
    bar.speed.set_value(4.0, animate=False)
    bar.speed.changed.emit(4.0)
    assert controller.speed == 4.0
    bar.slider.setFocus()
    qtbot.keyClick(bar.slider, Qt.Key.Key_Space)
    assert controller.playing
    qtbot.keyClick(bar.slider, Qt.Key.Key_Space)
    assert not controller.playing
    _park(host)


# =============================================================================== views
def _open(qtbot, path: Path):
    from torchcor_studio.ui.results import create_results_view

    view = create_results_view(path)
    view.resize(1280, 800)
    view.show()
    qtbot.waitUntil(lambda: not view.loading, timeout=60000)
    return view


def _cycle_theme_and_release(qtbot, view) -> None:
    _apply_theme(qtbot, "light")
    _apply_theme(qtbot, "dark")
    view.set_active(False)
    view.set_active(True)
    view.release()
    _park(view)  # releases again: must be idempotent


def test_ep_view_steps_switches_fields_and_probes(qtbot, studio, bundles):
    from torchcor_studio.ui.results.ep_view import EPView

    view = _open(qtbot, bundles["ep"])
    assert view.error is None
    ep = view.view
    assert isinstance(ep, EPView)
    assert [f.key for f in ep.fields] == ["voltage", "activation", "repolarisation", "apd"]
    ep.controller.seek(0)
    ep.controller.step(5)
    assert ep.controller.frame == 5
    if ep.viewer.available:
        np.testing.assert_allclose(ep.viewer.scalars(), ep.ep.vm.frame(5), atol=1e-3)
    assert ep.ap_cursor.line.value() == pytest.approx(float(ep.ep.times[5]))
    ep.set_field("activation")
    assert not ep.timeline.isVisible()
    ep.iso_switch.setChecked(True)
    if ep.viewer.available:
        np.testing.assert_allclose(ep.viewer.scalars(), ep.ep.lat, atol=1e-3)
        assert "isolines" in ep.viewer._extra
    ep.set_field("voltage")
    assert ep.timeline.isVisible()
    for point in range(0, 80, 10):
        ep.add_pick(point)
    assert len(ep.picks) == 6 and ep.picks[0] == 20  # only the latest six are kept
    assert len(ep.legend.items()) == len(ep.ep.traces) + 6
    ep.clear_picks()
    assert len(ep.legend.items()) == len(ep.ep.traces)
    ep.controller.play()
    ep.set_active(False)
    assert not ep.controller.playing
    ep.set_active(True)
    assert ep.controller.playing
    ep.controller.pause()
    _cycle_theme_and_release(qtbot, view)


def test_viewer_picks_the_surface_under_the_cursor(qtbot, studio, bundles):
    view = _open(qtbot, bundles["ep"])
    ep = view.view
    if not ep.viewer.available:
        pytest.skip("no 3D backend on this display")
    with qtbot.waitSignal(ep.viewer.point_picked, timeout=5000) as picked:
        ep.viewer.pick_at(ep.viewer.width() / 2, ep.viewer.height() / 2)
    assert 0 <= picked.args[0] < ep.surface.n_points
    assert picked.args[0] in ep.picks
    _park(view)


def test_viewer_screenshot_includes_the_colour_bar(qtbot, studio, bundles, tmp_path):
    from PySide6.QtGui import QImage

    view = _open(qtbot, bundles["ep"])
    viewer = view.view.viewer
    if not viewer.available:
        pytest.skip("no 3D backend on this display")
    path = viewer.save_screenshot(tmp_path / "shot.png")
    image = QImage(str(path))
    assert not image.isNull() and image.width() >= viewer.width()
    _park(view)


def _pixels(image) -> np.ndarray:
    from PySide6.QtGui import QImage

    image = image.convertToFormat(QImage.Format.Format_RGB888)
    h, w = image.height(), image.width()
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(h, image.bytesPerLine())
    return rows[:, : w * 3].reshape(h, w, 3).astype(np.int16)


def test_seeking_after_the_first_paint_repaints_the_viewport(qtbot, studio, bundles, tmp_path):
    from PySide6.QtGui import QImage

    view = _open(qtbot, bundles["mechanics"])
    mech = view.view
    viewer = mech.viewer
    if not viewer.available:
        pytest.skip("no 3D backend on this display")
    qtbot.wait(100)
    before = _pixels(viewer.grab().toImage())  # opens at full load: the beam bends
    mech.controller.seek(0)
    qtbot.wait(100)  # the coalesced render runs
    after = _pixels(viewer.grab().toImage())
    h, w = after.shape[:2]
    middle = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    assert np.abs(after[middle] - before[middle]).mean() > 1.0
    fresh = _pixels(QImage(str(viewer.save_screenshot(tmp_path / "now.png"))))
    assert fresh.shape == after.shape
    assert np.abs(fresh[middle] - after[middle]).mean() < 1.0  # what is shown is the current scene
    _park(view)


def test_ecg_view_layouts_exports_and_heart(qtbot, studio, bundles, tmp_path):
    from torchcor_studio.ui.results.ecg_view import ECGView

    view = _open(qtbot, bundles["ecg"])
    ecg = view.view
    assert isinstance(ecg, ECGView) and ecg.has_heart
    canvas = ecg.canvas
    assert len(canvas.cells()) == 12 and not canvas.rhythm_strip()  # 3 × 4; one beat: no rhythm strip
    for mode, count in (("stacked", 12), ("overlay", 1), ("grid", 12)):
        ecg._set_mode(mode)
        cells = canvas.cells()
        assert len(cells) == count
        frames = [c.frame for c in cells]  # every lead on a panel of its own, apart from the others
        assert not any(a.adjusted(-2, -2, 2, 2).intersects(b) for i, a in enumerate(frames) for b in frames[i + 1:])
        assert all(c.frame.contains(c.rect) for c in cells)
    ecg.clinical_switch.setChecked(True)
    assert canvas.time_window == (0.0, 2500.0)
    assert len(canvas.cells()) == 13 and canvas.cells()[-1].rhythm  # the clinical page's rhythm strip
    assert "10 mm/mV" in ecg.scale_note.text()
    ecg.clinical_switch.setChecked(False)
    canvas.set_window(100.0, 300.0)
    assert canvas.zoomed and canvas.time_window == (100.0, 300.0)
    canvas.reset_window()
    assert canvas.time_window == canvas.full_window
    _cell, lead, t, value = canvas.hover_info(canvas.cells()[0].rect.center())
    assert lead == "I" and value == pytest.approx(float(np.interp(t, ecg.ecg.t, ecg.ecg.leads[0])))
    png = ecg.export_png(tmp_path / "ecg.png")
    csv = ecg.export_csv(tmp_path / "ecg.csv")
    assert png.stat().st_size > 10_000 and csv.read_text().startswith("t_ms,I_mV")
    ecg.show_page("heart")
    qtbot.waitUntil(lambda: not ecg.heart.loading, timeout=60000)
    assert ecg.heart.error is None and not ecg.layout_control.isVisible()
    heart = ecg.heart.content
    before = heart.controller.frame
    heart.controller.step(3)
    assert heart.controller.frame == (before + 3) % heart.controller.n_frames
    ecg.show_page("ecg")
    assert ecg.layout_control.isVisible()
    _cycle_theme_and_release(qtbot, view)


def test_mechanics_view_warps_and_steps(qtbot, studio, bundles):
    from torchcor_studio.ui.results.mechanics_view import MechanicsView

    view = _open(qtbot, bundles["mechanics"])
    mech = view.view
    assert isinstance(mech, MechanicsView)
    assert mech.controller.frame == mech.mech.n_frames - 1  # opens at full load
    assert mech.timeline.time_label.text() == "λ = 1.00"
    mech.controller.seek(2)
    if mech.viewer.available:
        expected = mech.surface.points + mech.mech.displacement[2]
        np.testing.assert_allclose(mech.viewer.mesh.points, expected, atol=1e-4)
        mech.set_warp(3.0)
        np.testing.assert_allclose(mech.viewer.mesh.points, mech.surface.points + 3 * mech.mech.displacement[2],
                                   atol=1e-4)
        mech.set_field("von_mises")
        np.testing.assert_allclose(mech.viewer.scalars(), mech.mech.fields["von_mises"].frame(2), atol=1e-2)
    mech.set_reference_visible(True)
    mech.controller.seek_position(4.5)  # interpolated between load steps
    assert mech.controller.frame == 4
    _cycle_theme_and_release(qtbot, view)


def test_em_view_beats_with_pv_loops(qtbot, studio, bundles):
    from torchcor_studio.ui.results.em_view import EMView

    view = _open(qtbot, bundles["em"])
    em = view.view
    assert isinstance(em, EMView)
    labels = [tile.label.text() for tile in em.stats.tiles]
    assert labels[:5] == ["EDV", "ESV", "STROKE VOLUME", "EJECTION FRACTION", "PEAK LV PRESSURE"]
    assert em.stats.tiles[3].value.text() == "58.3"
    em.controller.seek(10)
    xs, _ys = em.pv_markers.getData()
    assert len(xs) == 4
    t = float(em.em.times[10])
    assert xs[0] == pytest.approx(float(np.interp(t, em.circulation.t, em.circulation.volume["lv"])))
    em.legend.toggled.emit("ra", False)
    assert len(em.pv_markers.getData()[0]) == 3
    for key in ("tension", "displacement", "voltage"):
        em.set_field(key)
        assert em.field == key
    _cycle_theme_and_release(qtbot, view)


def test_missing_or_broken_bundles_show_an_empty_state(qtbot, studio, bundles, tmp_path):
    from torchcor_studio.ui.widgets import EmptyState

    missing = _open(qtbot, tmp_path / "nothing")
    assert missing.view is None and missing.error
    assert missing.findChild(EmptyState) is not None
    folder = _copy(bundles["mechanics"], tmp_path / "broken")
    (folder / "mechanics.npz").write_bytes(b"truncated")
    broken = _open(qtbot, folder)
    assert broken.view is None and "damaged" in broken.error
    _park(missing)
    _park(broken)
