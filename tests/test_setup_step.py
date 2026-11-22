"""The Setup step: kinds and demos, where the parameters come from (changes, Reset, Undo,
duplicates), the folders in the first panel, the sticky rail with its outline, checks,
demo list and sweep, and demos pointing the input at their data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from torchcor_studio.core.models import InputInfo, VtxInfo

pytestmark = pytest.mark.gui

FOLDER = InputInfo(path="/data/ep/ventricle", exists=True, mesh_name="heart",
                   vtx=[VtxInfo("LV_af.vtx", 0), VtxInfo("LV_pf.vtx", 12), VtxInfo("RV_sf.vtx", 9)])


def _step(qtbot, folders=None):  # noqa: ANN001, ANN202
    from torchcor_studio.ui.pages.new_run.setup_step import SetupStep

    step = SetupStep(lambda *_: None, folders)
    qtbot.addWidget(step)
    return step


def _edit(step, path: tuple, value) -> None:  # noqa: ANN001
    editor = step.form().editor(path)
    editor.set(value)
    editor.emit()


def test_kinds_build_lazily_on_their_first_demo(themed, qtbot) -> None:
    from torchcor_studio.core.simulations import get_spec

    step = _step(qtbot)
    assert list(step._states) == ["ep"]  # only the kind shown
    demos = step.demos
    for kind in ("ep", "ecg", "mechanics", "em", "script"):
        step.set_kind(kind)
        spec = get_spec(kind)
        first = spec.demos[0]
        assert step.params() == first.params() and step.demo().key == first.key
        assert [demos.list.itemData(i) for i in range(demos.list.count())] == [d.key for d in spec.demos]
        assert demos.chosen() == first.key and demos.list.currentText() == first.title
        assert demos.summary.text().startswith(first.summary.split(" ")[0])
        assert not demos.status_row.isVisibleTo(step)  # the list names the demo: nothing to add
        assert step.status_text() == f"From {first.title}"
        assert step.kind_title.text() == spec.title and step.tagline.text() == f"{spec.tagline}."
    assert list(step._states) == ["ep", "ecg", "mechanics", "em", "script"]
    form = step.form("ep")
    step.set_kind("ep")
    assert step.form() is form  # each kind keeps its form


def test_a_simulation_builds_its_own_kind_only(themed, qtbot) -> None:
    from torchcor_studio.core.simulations import get_spec
    from torchcor_studio.ui.pages.new_run.setup_step import SetupStep

    step = SetupStep(lambda *_: None, kind="em")
    qtbot.addWidget(step)
    assert step.kind == "em" and list(step._states) == ["em"]  # no other kind's form
    assert step.kind_title.text() == get_spec("em").title and step.demos.chosen() == "five-beats"


def test_nothing_flashes_as_a_window_of_its_own(themed, qtbot) -> None:
    """A widget shown before it has a parent is a window of its own for a moment: on macOS
    it flashes on the screen and the cursor jitters. Building a kind, opening its panels and
    showing their advanced fields never does it."""
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QApplication, QWidget

    step = _step(qtbot)
    area = _host(qtbot, step, 1400, 800)
    windows: list[str] = []

    class Watch(QObject):
        def eventFilter(self, obj, event) -> bool:  # noqa: N802, ANN001
            if event.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow() \
                    and obj is not area:
                windows.append(type(obj).__name__)
            return False

    watch = Watch()
    QApplication.instance().installEventFilter(watch)
    try:
        for kind in ("ecg", "mechanics", "em", "script", "ep"):
            step.set_kind(kind)
            for card in step.form().cards():
                card.expand()
            step.set_show_advanced(True)
            qtbot.wait(5)
    finally:
        QApplication.instance().removeEventFilter(watch)
    assert windows == []


def test_a_demo_applies_in_place(themed, qtbot) -> None:
    from PySide6.QtWidgets import QWidget

    from torchcor_studio.core.simulations import ep

    step = _step(qtbot)
    form = step.form()
    cards, tables = list(form.cards()), [form.table(p) for p in (("stimuli",), ("model", "velocities"))]
    applied = []
    step.demo_applied.connect(lambda kind, demo: applied.append((kind, demo.key)))
    step.apply_demo("ventricle-eikonal")
    assert step.params() == ep.ventricle_eikonal() and step.form() is form
    assert applied == [("ep", "ventricle-eikonal")] and step.status_text() == "From Biventricle, reaction–eikonal"
    assert form.cards() == cards and [form.table(p) for p in (("stimuli",), ("model", "velocities"))] == tables
    assert all(isinstance(card, QWidget) for card in cards)  # the same panels and tables: no rebuild
    assert step.demos.chosen() == "ventricle-eikonal"


def test_only_a_choice_in_the_open_list_applies_a_demo(themed, qtbot) -> None:
    """Choosing a demo replaces the parameters: the wheel and the arrow keys never do it,
    a choice in the open list does, and choosing the shown demo again puts it back."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication

    from torchcor_studio.core.simulations import ep

    step = _step(qtbot)
    _host(qtbot, step, 1400, 800)
    box = step.demos.list
    centre = QPointF(box.rect().center())
    wheel = QWheelEvent(centre, QPointF(box.mapToGlobal(centre.toPoint())), QPoint(0, 0), QPoint(0, -120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
                        False)
    QApplication.sendEvent(box, wheel)
    assert step.demos.chosen() == "ventricle" and step.params() == ep.ventricle()
    box.setFocus()
    qtbot.keyClick(box, Qt.Key.Key_Down)  # opens the list instead of stepping through it
    qtbot.waitUntil(box.view().isVisible, timeout=3000)
    assert step.demos.chosen() == "ventricle" and step.params() == ep.ventricle()
    qtbot.keyClick(box.view(), Qt.Key.Key_Down)
    qtbot.keyClick(box.view(), Qt.Key.Key_Return)  # chosen in the open list: applied
    qtbot.waitUntil(lambda: step.params() == ep.ventricle_eikonal(), timeout=3000)
    assert step.demos.chosen() == "ventricle-eikonal" and step.demo().key == "ventricle-eikonal"
    _edit(step, ("timing", "duration"), 250.0)
    assert step.demos.status.text() == "1 change"
    box.activated.emit(box.currentIndex())  # the shown demo chosen again: its parameters back
    assert step.params() == ep.ventricle_eikonal() and step.demos.status.text() == "Applied over your edits"


def test_changes_reset_and_undo(themed, qtbot) -> None:
    from torchcor_studio.core.settings import Store
    from torchcor_studio.core.simulations import ep
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.pages.new_run.folders import Folders

    ctx = AppContext(Store())
    folders = Folders(ctx)
    step = _step(qtbot, folders)
    demos = step.demos
    _edit(step, ("timing", "duration"), 250.0)
    _edit(step, ("timing", "dt"), 0.02)
    assert step.status_text() == "From Biventricle, five pacing sites · 2 changes" and demos.status.text() == "2 changes"
    assert demos.status.toolTip().splitlines() == ["Time and outputs › Duration: 500 → 250 ms",
                                                   "Time and outputs › Time step: 0.01 → 0.02 ms"]
    assert demos.reset.isVisibleTo(step) and "Biventricle, five pacing sites" in demos.reset.toolTip()
    assert step.form().panel("time-and-outputs").edited
    step.reset()
    assert step.params() == ep.ventricle() and step.status_text() == "From Biventricle, five pacing sites"
    assert not demos.reset.isVisibleTo(step) and not demos.status_row.isVisibleTo(step)

    _edit(step, ("timing", "duration"), 300.0)
    folders.set_input("/mine", "user")
    step.apply_demo("ventricle-eikonal")  # replaces edited parameters: one level of Undo
    assert step.status_text() == "Applied Biventricle, reaction–eikonal" and demos.undo.isVisibleTo(step)
    assert demos.chosen() == "ventricle-eikonal" and demos.status.text() == "Applied over your edits"
    folders.set_input("/elsewhere", "demo")  # as a demo's data would
    step.undo()
    assert step.params().timing.duration == 300.0 and step.demo().key == "ventricle"
    assert demos.chosen() == "ventricle" and folders.input_dir() == "/mine" and folders.source == "user"
    assert step.status_text() == "From Biventricle, five pacing sites · 1 change" and not demos.undo.isVisibleTo(step)
    assert demos.status.text() == "1 change" and demos.reset.isVisibleTo(step)
    ctx.shutdown()


def test_a_duplicate_shows_where_it_came_from(themed, qtbot) -> None:
    from torchcor_studio.core.simulations import ep, get_spec

    step = _step(qtbot)
    step.load("ep", ep.atrium().model_dump(), origin="Atrium pacing")
    assert step.status_text() == "From run “Atrium pacing”"
    assert step.demos.chosen() == "atrium"  # equal to a demo: shown in the list...
    assert step.demos.status.text() == "From run “Atrium pacing”"  # ...and the run it came from
    step.load("em", {"rhythm": {"beats": 2}}, origin="Two beats")
    assert step.kind == "em" and step.demo() is None
    assert step.demos.chosen() is None and step.demos.list.currentIndex() == -1  # "Choose a demo"
    assert not step.demos.about.isVisibleTo(step)
    assert step.params().rhythm.beats == 2 and step.status_text() == "From run “Two beats”"
    _edit(step, ("rhythm", "beats"), 3)
    assert step.status_text() == "From run “Two beats” · 1 change"
    step.reset()
    assert step.params() == get_spec("em").params_model(rhythm={"beats": 2})


def test_a_fit_after_a_demo_moves_the_baseline(themed, qtbot) -> None:
    step = _step(qtbot)
    step.set_input(FOLDER)
    step.apply_demo("atrium")  # its blank site comes from the folder...
    assert step.params().stimuli[0].site == "LV_pf.vtx"
    assert step.status_text() == "From Atrium (Courtemanche)"  # ...and is not a change
    assert step.demos.chosen() == "atrium" and not step.form().changes()


def test_undo_survives_the_script_editors_pause(themed, qtbot) -> None:
    """Setting the code from a demo, Reset or Undo is no edit, even after the editor's
    typing pause: Undo stays and brings the user's script back."""
    step = _step(qtbot)
    step.set_kind("script")
    editor = step.form().editor(("code",))
    mine = "print('mine')\n"
    editor.widget.setPlainText(mine)  # typed: an edit once the typing pauses
    qtbot.waitUntil(lambda: step.params().code == mine, timeout=3000)
    step.apply_demo("starter")
    assert step.status_text() == "Applied Starter script" and step.demos.undo.isVisibleTo(step)
    qtbot.wait(600)  # longer than the pause
    assert step.status_text() == "Applied Starter script" and step.demos.undo.isVisibleTo(step)
    step.undo()
    qtbot.wait(600)
    assert step.params().code == mine and step.status_text() == "From Starter script · 1 change"


def test_tab_goes_through_the_panels_then_the_rail(themed, qtbot) -> None:
    """The panels are built after the rail, yet Tab reaches them first; in the rail, the
    demo list comes before the sweep."""
    from PySide6.QtCore import Qt

    def after(widget):  # noqa: ANN001, ANN202
        widget = widget.nextInFocusChain()
        while not (widget.isVisibleTo(step) and widget.focusPolicy() & Qt.FocusPolicy.TabFocus):
            widget = widget.nextInFocusChain()
        return widget

    step = _step(qtbot)
    for kind in ("ep", "em"):
        step.set_kind(kind)
        order, widget = [], step.header  # the kind's title: nothing to focus
        while (widget := after(widget)) not in order and len(order) < 500:
            order.append(widget)
        assert order[0] is step.form().cards()[0].header, kind  # the first panel first
        last_field = max(i for i, w in enumerate(order) if step.form().isAncestorOf(w))
        assert last_field < order.index(step.demos.list) < order.index(step.sweep.toggle), kind


def test_the_first_panel_holds_the_folders(themed, qtbot) -> None:
    """Every kind's first panel ends with the output folder; the input folder sits in it
    too, except for mechanics, whose meshes are built in (its slot waits off the page)."""
    from torchcor_studio.core.settings import Store
    from torchcor_studio.core.simulations.panels import INPUT, OUTPUT
    from torchcor_studio.ui.context import AppContext
    from torchcor_studio.ui.pages.new_run.folders import Folders

    ctx = AppContext(Store())
    folders = Folders(ctx)
    step = _step(qtbot, folders)
    for kind in ("ep", "ecg", "mechanics", "em", "script"):
        step.set_kind(kind)
        first = step.form().cards()[0]
        assert step.form().has_slot(OUTPUT) and first.isAncestorOf(folders.output_slot), kind
        assert first.isAncestorOf(folders.input_slot) == (kind != "mechanics"), kind
        assert step._parking.isAncestorOf(folders.input_slot) == (kind == "mechanics"), kind
        assert step.form().has_slot(INPUT) == (kind != "mechanics"), kind
    ctx.shutdown()


def _host(qtbot, step, width: int, height: int):  # noqa: ANN001, ANN202
    from torchcor_studio.ui.pages.common import scroll

    area = scroll(step)
    qtbot.addWidget(area)
    area.resize(width, height)
    area.show()
    qtbot.waitExposed(area)
    qtbot.wait(50)
    return area


def test_the_rail_sticks_beside_and_stacks_when_narrow(themed, qtbot) -> None:
    from PySide6.QtCore import QPoint

    step = _step(qtbot)
    step.set_kind("em")
    area = _host(qtbot, step, 1400, 800)
    rail, bar = step.rail, area.verticalScrollBar()
    assert step.body.is_beside() and rail.width() == 300 and rail.outline.isVisibleTo(step)
    tops = []
    for value in range(0, bar.maximum() + 1, 200):
        bar.setValue(value)
        qtbot.wait(5)
        top = rail.mapTo(area.viewport(), QPoint(0, 0)).y()
        if top <= 16:  # stuck, or clamped at the end of the body: wholly in view
            assert top >= 0 and top + rail.height() <= area.viewport().height(), value
        tops.append(top)
    assert 16 in tops  # stuck 16 px below the top once scrolled
    bar.setValue(bar.maximum())
    qtbot.wait(5)
    assert rail.geometry().bottom() <= step.body.height()  # clamped to the body
    area.resize(860, 800)  # under 920 px of content: above the panels, the outline as chips
    qtbot.waitUntil(lambda: not step.body.is_beside(), timeout=3000)
    qtbot.wait(50)
    assert rail.outline.isVisibleTo(step) and rail.outline.hasHeightForWidth()
    assert rail.mapTo(step, QPoint(0, 0)).y() < step.form().mapTo(step, QPoint(0, 0)).y()
    form_top = step.form().mapTo(step, QPoint(0, 0)).y()
    form_bottom = step.form().mapTo(step, QPoint(0, step.form().height())).y()
    assert step.demos.mapTo(step, QPoint(0, 0)).y() < form_top  # the demos above the panels
    assert step.sweep.mapTo(step, QPoint(0, 0)).y() >= form_bottom  # the sweep below them
    assert area.widget().width() <= area.viewport().width()  # nothing overflows
    area.resize(1400, 800)  # beside again: the sweep is back in the rail, under the demos
    qtbot.waitUntil(step.body.is_beside, timeout=3000)
    assert rail.isAncestorOf(step.sweep) and not rail.outline.hasHeightForWidth()

    def tops() -> list[int]:
        return [card.mapTo(rail, QPoint(0, 0)).y() for card in (rail.checks, step.demos, step.sweep)]

    qtbot.waitUntil(lambda: tops() == sorted(set(tops())), timeout=3000)  # Checks, the demos, the sweep


def test_the_outline_follows_the_scroll_and_opens_panels(themed, qtbot) -> None:
    from PySide6.QtCore import QPoint, Qt

    step = _step(qtbot)
    step.set_kind("em")
    area = _host(qtbot, step, 1400, 800)
    outline, bar = step.rail.outline, area.verticalScrollBar()
    assert outline.current() == "heart"
    conduction = step.form().panel("conduction")
    bar.setValue(conduction.mapTo(area.widget(), QPoint(0, 0)).y() - 40)
    qtbot.waitUntil(lambda: outline.current() == "conduction", timeout=3000)
    keys = [e.key for e in outline.entries()]
    circulation = step.form().panel("circulation")
    assert not circulation.is_expanded()
    y = outline.HEAD + outline.GAP + keys.index("circulation") * outline.ROW + outline.ROW // 2
    qtbot.mouseClick(outline, Qt.MouseButton.LeftButton, pos=QPoint(40, y))
    qtbot.waitUntil(lambda: circulation.mapTo(area.viewport(), QPoint(0, 0)).y() == 12, timeout=3000)
    assert circulation.is_expanded() and circulation.is_built()
    entry = next(e for e in outline.entries() if e.key == "supports-and-start-pressures")
    assert entry.collapsed and entry.summary.startswith("3 springs")


def test_script_hides_the_outline_and_the_sweep(themed, qtbot) -> None:
    step = _step(qtbot)
    _host(qtbot, step, 1400, 800)
    assert step.rail.outline.isVisibleTo(step) and step.sweep.isVisibleTo(step)
    step.set_kind("script")
    assert not step.rail.outline.isVisibleTo(step) and not step.sweep.isVisibleTo(step)
    assert step.demos.isVisibleTo(step) and step.demos.list.count() == 1  # its starter
    step.set_kind("mechanics")
    assert step.rail.outline.isVisibleTo(step) and step.sweep.isVisibleTo(step)


def test_widget_budgets(themed, qtbot) -> None:
    from PySide6.QtWidgets import QWidget

    step = _step(qtbot)
    first = len(step.form("ep").findChildren(QWidget)) + 1
    step.set_kind("em")
    assert len(step.form().findChildren(QWidget)) <= 300
    assert len(step.findChildren(QWidget)) - first <= 450  # everything but the EP set-up built first


def _context(root: Path):  # noqa: ANN202
    from torchcor_studio.core.settings import LOCAL_MACHINE_ID, Store
    from torchcor_studio.ui.context import AppContext

    store = Store()
    local = store.machine(LOCAL_MACHINE_ID)
    local.python = sys.executable
    local.data_dir = str(root)  # the machine's demo data folder
    store.upsert_machine(local)
    return AppContext(store)


def test_demos_point_the_input_at_their_data(themed, qtbot, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt

    from torchcor_studio.core.settings import LOCAL_MACHINE_ID
    from torchcor_studio.ui.pages.new_run.page import NewRunPage

    root = tmp_path / "torchcor_data"
    inputs, outputs = root / "input", root / "output"
    for sub in ("ep/ventricle", "em"):
        (inputs / sub).mkdir(parents=True)
    ctx = _context(root)
    page = NewRunPage(ctx)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    ctx.connect_machine(LOCAL_MACHINE_ID)
    page.compute.set_machine(LOCAL_MACHINE_ID)
    qtbot.waitUntil(lambda: ctx.session(LOCAL_MACHINE_ID).data_folders is not None, timeout=60000)
    page.go(1)
    setup, folders = page.setup, page.folders
    # EP starts as the ventricle demo: its input folder, and the matching output folder
    qtbot.waitUntil(lambda: folders.input_dir() == str(inputs / "ep" / "ventricle"), timeout=60000)
    assert folders.output_dir() == str(outputs / "ep" / "ventricle")
    demos = setup.demos

    def tip(key: str) -> str:  # a demo's tooltip in the list: its summary, then its data
        return demos.list.itemData(demos.list.findData(key), Qt.ItemDataRole.ToolTipRole)

    qtbot.waitUntil(lambda: tip("ventricle").endswith(str(inputs / "ep" / "ventricle")), timeout=10000)
    assert "is not on this computer" in tip("atrium") and "Download the demo data" in tip("atrium")
    assert demos.data_chip.text() == "ep/ventricle"

    setup.apply_demo("ventricle")  # found: the input is its folder, inspected, not remembered
    assert folders.input_dir() == str(inputs / "ep" / "ventricle") and folders.source == "demo"
    assert folders.output_dir() == str(outputs / "ep" / "ventricle")
    qtbot.waitUntil(lambda: folders.input_info is not None, timeout=60000)
    assert folders.input_summary.text.text().startswith("No mesh")  # what was found
    assert LOCAL_MACHINE_ID not in ctx.settings.kind_inputs
    setup.apply_demo("atrium")  # missing: the input stays, and the slot says why
    assert folders.input_dir() == str(inputs / "ep" / "ventricle")
    assert "download the demo data in the Compute step" in folders.input_slot.warning_text.text()
    assert folders.input_slot.warning.isVisibleTo(page)

    setup.set_kind("ecg")  # no prepared ecg folder: the demo says where its data goes
    assert demos.data_chip.text() == "Needs data"
    assert f"This demo reads {inputs / 'ecg'}," in folders.input_slot.warning_text.text()
    requires = setup.spec.demo("healthy").requires
    (inputs / "ecg" / "heart").mkdir(parents=True)  # an ecg folder that is not KCL torso 3...
    ctx.connect_machine(LOCAL_MACHINE_ID, reprobe=True)
    session = ctx.session(LOCAL_MACHINE_ID)
    qtbot.waitUntil(lambda: session.connected and session.data_folders.has_case("ecg"), timeout=60000)
    setup.apply_demo("healthy")
    assert folders.input_dir() != str(inputs / "ecg") and demos.data_chip.text() == "Needs data"  # ...is not it
    for sub in requires:  # prepared there: the demo reads it (found on the next probe)
        (inputs / "ecg" / sub).mkdir(exist_ok=True)
    ctx.connect_machine(LOCAL_MACHINE_ID, reprobe=True)
    qtbot.waitUntil(lambda: session.connected and session.data_folders.has_case("ecg", requires), timeout=60000)
    assert demos.data_chip.toolTip() == str(inputs / "ecg")
    setup.apply_demo("healthy")
    assert (folders.input_dir(), folders.output_dir()) == (str(inputs / "ecg"), str(outputs / "ecg"))

    setup.set_kind("em")
    assert page.issues_btn.isVisible() and page.issues_btn.text().endswith("to fix")
    target = folders.input.edit
    assert target.property("issue") == "error"  # the input folder is what is missing
    page.issues_btn.click()  # shows the first error: the heart's mesh folder, i.e. the input
    qtbot.waitUntil(lambda: page.window().focusWidget() is target and target.property("flash"), timeout=3000)
    qtbot.waitUntil(lambda: not target.property("flash"), timeout=3000)  # a short flash
    ctx.shutdown()


def test_ecg_heart_ep_is_one_box_with_its_panels_as_sections(themed, qtbot) -> None:
    from PySide6.QtCore import QPoint, Qt

    from torchcor_studio.ui.forms.panel import PanelGroup
    from torchcor_studio.ui.pages.new_run.setup_step import JUMP_OFFSET

    step = _step(qtbot)
    step.set_kind("ecg")
    area = _host(qtbot, step, 1400, 800)
    form = step.form()
    box = form.anchor("group-heart-ep")
    assert isinstance(box, PanelGroup) and [card.key for card in box.cards] == ["cells", "models", "pacing-sites"]
    assert all(card.nested and box.isAncestorOf(card) for card in box.cards)
    assert not form.panel("heart-and-torso").nested and not box.isAncestorOf(form.panel("heart-and-torso"))
    outline = step.rail.outline
    rows = [(e.key, e.title, e.depth, e.group) for e in outline.entries()]
    i = rows.index(("group-heart-ep", "Heart EP", 0, True))  # the heading, then its panels indented
    assert rows[i + 1:i + 4] == [("cells", "Cells", 1, False), ("models", "Models", 1, False),
                                 ("pacing-sites", "Pacing sites", 1, False)]
    y = outline.HEAD + outline.GAP + i * outline.ROW + outline.ROW // 2
    qtbot.mouseClick(outline, Qt.MouseButton.LeftButton, pos=QPoint(40, y))  # the heading: to the box
    qtbot.waitUntil(lambda: box.mapTo(area.viewport(), QPoint(0, 0)).y() == JUMP_OFFSET, timeout=3000)
    conductivities, velocities = form.table(("heart", "conductivities")), form.table(("heart", "velocities"))
    assert conductivities.isVisibleTo(form) and not velocities.isVisibleTo(form)
    _edit(step, ("heart", "solver"), "eikonal")  # Models: the table of the model chosen
    assert velocities.isVisibleTo(form) and not conductivities.isVisibleTo(form)
    _edit(step, ("heart", "solver"), "re_plus")  # with diffusion: both
    assert velocities.isVisibleTo(form) and conductivities.isVisibleTo(form)
