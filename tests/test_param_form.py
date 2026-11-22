"""The parameter form: panels built on demand, values updated in place, tables, the field
grid, issue tints and the editors' own formats."""

from __future__ import annotations

import pytest

from torchcor_studio.core.simulations import Issue, Severity, get_spec

pytestmark = pytest.mark.gui


def _form(qtbot, kind: str, value=None):  # noqa: ANN001, ANN202
    from torchcor_studio.ui.forms import ParamForm

    spec = get_spec(kind)
    form = ParamForm(spec.params_model, value or spec.defaults(), panels=spec.panels)
    qtbot.addWidget(form)
    form.resize(820, 900)
    form.show()
    return form


def test_collapsed_panels_are_built_when_first_opened(themed, qtbot) -> None:
    from PySide6.QtWidgets import QWidget

    from torchcor_studio.ui.forms.table import TableEditor

    form = _form(qtbot, "em")
    circulation = form.panel("circulation")
    assert not circulation.is_built() and not circulation.is_expanded()
    assert not any(t.path[0] == "circulation" for t in form.findChildren(TableEditor))
    assert len(form.findChildren(QWidget)) <= 300  # the widget budget of the default EM form
    circulation.expand()
    assert circulation.is_built() and circulation.resolved.key == "circulation"
    assert {t.path for t in form.findChildren(TableEditor)} >= {("circulation", "tubes"),
                                                                ("circulation", "valves")}


def test_values_update_in_place(themed, qtbot) -> None:
    from torchcor_studio.core.simulations import ep

    form = _form(qtbot, "ep")
    duration = form.editor(("timing", "duration"))
    cells, stimuli = form.table(("cells",)), form.table(("stimuli",))
    cell_model, site = cells.cell(0, "model"), stimuli.cell(0, "site")
    form.set_value(ep.atrium())  # from the defaults (the ventricle) to the atrium
    assert form.editor(("timing", "duration")) is duration and duration.get() == 500.0
    assert cells.cell(0, "model") is cell_model  # same row count: the cells stay
    assert stimuli.cell(0, "site") is not site and stimuli.row_count() == 1  # one row now, not five
    assert form.value() == ep.atrium() and form.data()["stimuli"][0]["site"] == ""


def test_fixed_tables_have_row_headers_and_no_add_or_remove(themed, qtbot) -> None:
    from PySide6.QtWidgets import QLabel, QPushButton, QToolButton

    form = _form(qtbot, "em")
    tubes, pacing = form.table(("circulation", "tubes")), form.table(("rhythm", "pacing"))
    heads = [w.text() for w in tubes.findChildren(QLabel) if w.property("role") == "rowhead"]
    assert heads == ["aorta", "pulmonary artery", "venae cavae", "pulmonary veins"]
    assert tubes.cell(0, "name") is None and tubes.cell(0, "length") is not None
    assert not tubes.findChildren(QPushButton) and not tubes.findChildren(QToolButton)
    assert any(b.text() == "Add site" for b in pacing.findChildren(QPushButton))
    assert tubes.header("reference_pressure").title == "Ref. pressure"
    assert pacing.is_half() and not tubes.is_half()


def test_column_groups_show_with_two_groups_only(themed, qtbot) -> None:
    from torchcor_studio.ui.forms.table import GroupHead

    form = _form(qtbot, "ep")
    table = form.table(("model", "conductivities"))
    groups = {g.title: g for g in table.findChildren(GroupHead)}
    assert set(groups) == {"INTRACELLULAR", "EXTRACELLULAR"} and all(g.isVisible() for g in groups.values())
    assert table.header("il").title == "Along fibres" and table.header("il").unit == "S/m"
    mode = form.editor(("model", "conductivity_mode"))
    mode.set("monodomain")  # effective values: one group left, so no group headers
    mode.emit()
    assert not any(g.isVisible() for g in groups.values()) and not table.header("el").isVisible()


def test_field_grid_packing(themed, qtbot) -> None:
    from PySide6.QtWidgets import QWidget

    from torchcor_studio.ui.forms.grid import FieldGrid, GroupTile

    def box(height: int = 54) -> QWidget:
        widget = QWidget()
        widget.setFixedHeight(height)
        return widget

    host = QWidget()
    grid = FieldGrid(host)
    qtbot.addWidget(host)
    numbers = [box() for _ in range(5)]
    for widget in numbers[:4]:
        grid.add(widget)
    tiles = []
    for _ in range(2):
        tile = GroupTile("Group")
        for _ in range(2):
            tile.add(box(), 1)
        tiles.append(tile)
        grid.add(tile, span=tile.span)
    grid.add(numbers[4])
    table, half_a, half_b = box(100), box(80), box(80)
    grid.add(table, span=0, role="block")
    grid.add(half_a, span=0, role="block", half=lambda: True)
    grid.add(half_b, span=0, role="block", half=lambda: True, natural=lambda: 300)
    host.resize(777, 2000)
    host.show()
    qtbot.wait(10)
    assert grid.columns(777) == 4
    assert len({w.y() for w in numbers[:4]}) == 1 and numbers[4].y() > numbers[0].y()  # 4 a row
    assert numbers[0].width() == round((777 - 3 * 16) / 4)
    assert tiles[0].y() == tiles[1].y() and tiles[1].x() > tiles[0].x()  # two 2-wide tiles share
    assert numbers[4].y() > tiles[0].geometry().bottom()
    assert table.y() > numbers[4].geometry().bottom() and table.width() == 777  # its own row
    assert half_a.y() == half_b.y() and abs(half_a.width() - half_b.width()) <= 1 and half_a.width() < 777 / 2
    numbers[1].hide()  # a hidden field leaves no gap
    qtbot.wait(10)
    assert numbers[2].x() == numbers[0].x() + numbers[0].width() + 16
    assert grid.columns(500) == 2 and grid.columns(100) == 2
    host.resize(560, 2000)  # half of 560 is narrower than half_b's own 300 px: it takes the row
    qtbot.wait(10)
    assert half_b.y() > half_a.y() and half_b.width() == 560


def test_reveal_opens_the_panel_and_its_advanced(themed, qtbot) -> None:
    form = _form(qtbot, "em")
    heart = form.panel("heart")
    assert not form.advanced_shown("heart")
    widget = form.reveal("heart.lv_endo")
    assert form.advanced_shown("heart") and heart.show_advanced and widget is not None
    assert widget is form.editor(("heart", "lv_endo")).focus_target() and widget.isVisibleTo(form)
    cell = form.reveal("circulation.valves.2.open_area")  # a cell of a collapsed panel's table
    assert form.panel("circulation").is_expanded()
    assert cell is form.table(("circulation", "valves")).cell(2, "open_area").focus_target()
    assert form.reveal("rhythm.pacing") is form.table(("rhythm", "pacing")).first_cell()


def test_issues_tint_their_editors_and_count_on_panels(themed, qtbot) -> None:
    form = _form(qtbot, "ep")
    form.mark_issues([Issue(Severity.WARNING, "Long.", "timing.duration"),
                      Issue(Severity.ERROR, "Bad site.", "stimuli.0.site"),
                      Issue(Severity.INFO, "Note.", "timing.dt")])
    duration = form.editor(("timing", "duration"))
    assert duration.focus_target().property("issue") == "warning"
    assert duration.focus_target().toolTip() == "Long."
    assert form.table(("stimuli",)).cell(0, "site").issue() == "error"
    assert form.issue_counts()["time-and-outputs"] == (0, 1) and form.issue_counts()["pacing"] == (1, 0)
    assert form.panel("pacing").issue_badge() == (1, "danger")
    form.mark_issues([])
    assert not duration.focus_target().property("issue")


def test_number_editors_draw_their_unit_inside(themed, qtbot) -> None:
    from PySide6.QtWidgets import QLabel

    form = _form(qtbot, "ep")
    duration = form.editor(("timing", "duration")).widget
    assert duration.suffix() == ""
    units = [w for w in duration.findChildren(QLabel) if w.property("role") == "unit"]
    assert [u.text() for u in units] == ["ms"]
    cell = form.table(("stimuli",)).cell(0, "start").widget
    assert cell.property("cell") and not [w for w in cell.findChildren(QLabel) if w.property("role") == "unit"]


def test_region_ranges() -> None:
    from torchcor_studio.ui.forms.editors import MAX_RANGE, format_regions, parse_regions

    assert format_regions([1, 2, 5, 6, 7, 8, 12]) == "1, 2, 5–8, 12"
    assert format_regions([5, 6, *range(7, 25), 27]) == "5–24, 27"  # runs of three or more
    assert parse_regions("1-3, 7") == [1, 2, 3, 7] and parse_regions("7–9") == [7, 8, 9]
    assert parse_regions("7-") == "7-"  # left for pydantic to report
    assert parse_regions("36-34") == "36-34"  # backwards: not "no region", which means all
    assert parse_regions(f"1-{MAX_RANGE + 1}") == f"1-{MAX_RANGE + 1}"  # a slip of the keys


def test_the_regions_button_sits_on_the_fields_middle(themed, qtbot) -> None:
    """The regions button inside a field (a table cell, or a field of its own) is centred on
    it, not stretched to a button's height by the buttons' style."""
    from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget

    from torchcor_studio.core.simulations.types import FieldType
    from torchcor_studio.ui.forms.editors import ListEditor

    host = QWidget()
    lay = QVBoxLayout(host)
    editors = [ListEditor({}, [], FieldType(kind="list", inner=int, optional=False), lambda *_: None,
                          regions=True, cell=cell) for cell in (True, False)]
    for editor in editors:
        lay.addWidget(editor.widget)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    for editor in editors:
        field = editor.widget
        button = field.findChild(QToolButton)
        middle = button.geometry().top() + button.height() / 2
        assert button.height() < field.height() and abs(middle - field.height() / 2) <= 1


def test_dotted_visible_if(themed, qtbot) -> None:
    from torchcor_studio.core.simulations import mechanics

    form = _form(qtbot, "mechanics")
    frame = form.editor(("outputs", "frame_interval"))
    assert not frame.widget.isVisibleTo(form)  # "../loading.mode": a quasi-static run has no frames
    form.set_value(mechanics.lv_beat())
    assert frame.widget.isVisibleTo(form)


def test_controlling_choice_ends_its_row(themed, qtbot) -> None:
    form = _form(qtbot, "mechanics")
    law = form.editor(("material", "law")).widget
    stiffness = form.editor(("material", "C")).widget
    b_fs = form.editor(("material", "bfs")).widget
    assert stiffness.mapTo(form, law.rect().topLeft()).y() > law.mapTo(form, law.rect().topLeft()).y()
    assert stiffness.mapTo(form, stiffness.rect().topLeft()).y() == b_fs.mapTo(form, b_fs.rect().topLeft()).y()


def test_row_breaks_hold_past_hidden_fields(themed, qtbot) -> None:
    """A choice that later fields depend on ends its row even when the field after it is
    hidden, and group tiles never share a row with loose fields."""
    from torchcor_studio.core.simulations import mechanics

    form = _form(qtbot, "mechanics", mechanics.lv_beat())
    qtbot.wait(20)

    def top(path):  # noqa: ANN001, ANN202
        widget = form.editor(path).widget
        return widget.mapTo(form, widget.rect().topLeft()).y()

    geometry = ("geometry", "shape")
    endo, epi = ("geometry", "endo_radius"), ("geometry", "epi_radius")
    base, rings = ("geometry", "base_z"), ("geometry", "transmural")
    assert top(geometry) < top(endo) == top(epi) < top(base) == top(rings)  # ENDO beside EPI
    law, matrix, fibre = ("material", "law"), ("material", "a"), ("material", "a_f")
    sheet, shear = ("material", "a_s"), ("material", "a_fs")
    assert top(law) < top(matrix) == top(fibre) < top(sheet) == top(shear)  # MATRIX beside FIBRE


def test_tables_fit_their_width_and_never_drop_units(themed, qtbot) -> None:
    """At the main column of a 1280 px window every table fits (narrow cells elide), half
    tables pair or take the row, and every unit shows under its column's title."""
    from torchcor_studio.ui.forms.table import TableEditor

    form = _form(qtbot, "em")
    form.resize(657, 900)  # the main column at 937 px of content
    form.expand_all()
    qtbot.wait(50)
    for table in form.findChildren(TableEditor):
        if not table.isVisibleTo(form):
            continue
        for (_row, _name), editor in table.editors():
            cell = editor.widget
            if cell.isVisibleTo(form):
                assert cell.geometry().right() <= table.width(), (table.path, _name)
        for col in table._columns:
            head = table.header(col.name)
            if head.isVisibleTo(form) and col.meta.get("unit"):
                assert head.lines() == 2 and head.unit == col.meta["unit"], (table.path, col.name)
    valves, chambers = form.table(("circulation", "valves")), form.table(("circulation", "chambers"))
    assert valves.is_half() and chambers.is_half()


def test_number_boxes_read_plainly(themed, qtbot) -> None:
    form = _form(qtbot, "em")
    form.expand_all()
    assert form.editor(("circulation", "blood_density")).widget.text() == "1050"  # no "1,050.0"
    assert form.table(("circulation", "tubes")).cell(1, "reference_area").widget.text() == "500"
    assert form.editor(("rhythm", "ep_dt")).widget.text() == "0.02"
    pacing = form.table(("rhythm", "pacing"))
    assert pacing.cell(0, "first").widget.text() == "713"


def test_switches_and_segmented_controls_show_their_issues(themed, qtbot) -> None:
    form = _form(qtbot, "mechanics")
    form.mark_issues([Issue(Severity.ERROR, "Helical fibres need the ventricle.", "material.fibres"),
                      Issue(Severity.WARNING, "Many DOFs.", "geometry.shape")])
    fibres = form.editor(("material", "fibres")).widget
    assert fibres.property("issue") == "error" and fibres.toolTip() == "Helical fibres need the ventricle."
    assert form.reveal("geometry.shape") is form.editor(("geometry", "shape")).widget  # focusable
    vtk = form.editor(("outputs", "vtk"))
    assert vtk.focus_target() is vtk.widget


def test_input_folder_issues_go_to_the_slot(themed, qtbot) -> None:
    from PySide6.QtWidgets import QLineEdit, QVBoxLayout, QWidget

    from torchcor_studio.core.simulations.panels import INPUT

    class Slot(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.edit = QLineEdit()
            QVBoxLayout(self).addWidget(self.edit)
            self.issue = None

        def focus_target(self):  # noqa: ANN202
            return self.edit

        def set_issue(self, severity, message=""):  # noqa: ANN001
            self.issue = severity

    form = _form(qtbot, "em")
    slot = Slot()
    form.set_slot(INPUT, slot)
    form.mark_issues([Issue(Severity.ERROR, "Choose the heart's mesh folder.", INPUT)])
    assert slot.issue == "error" and form.issue_counts()["heart"] == (1, 0)
    assert not form.editor(("heart", "lv_endo")).issue()  # not a field of the folder
    assert form.reveal(INPUT) is slot.edit
    form.mark_issues([])
    assert slot.issue is None


def test_a_section_switch_keeps_its_row_and_its_settings_follow_below(themed, qtbot) -> None:
    """Pre-pacing's switch ends its row: turning it on shows its settings side by side on
    the next row, and the switch does not move. A switch that gates one field of several
    (VTK frames: its frame step) does not end its row: the step follows it where it fits."""
    from PySide6.QtCore import QPoint

    form = _form(qtbot, "ep")
    form.panel("time-and-outputs").expand()

    def cell(*path: str):  # noqa: ANN202
        return form._fields[path].cell

    def top(*path: str) -> int:
        return cell(*path).mapTo(form, QPoint(0, 0)).y()

    def ends_row(*path: str) -> bool:  # the grid's own rule, whatever the fonts' widths
        grid = cell(*path).parentWidget().layout()
        return next(entry.end for entry in grid._items if entry.item.widget() is cell(*path))

    def turn_on(path: tuple[str, str], shows: tuple[str, str]) -> None:
        editor = form.editor(path)
        editor.set(True)
        editor.emit()
        qtbot.waitUntil(cell(*shows).isVisible, timeout=3000)
        qtbot.wait(20)  # the grid lays the new fields out

    switch = top("prepace", "enabled")
    turn_on(("prepace", "enabled"), ("prepace", "cycle_length"))
    assert top("prepace", "enabled") == switch  # the switch stays put
    assert top("prepace", "cycle_length") >= switch + cell("prepace", "enabled").height()  # below it
    assert top("prepace", "beats") == top("prepace", "cycle_length")  # side by side
    assert cell("prepace", "beats").x() > cell("prepace", "cycle_length").x()
    assert ends_row("prepace", "enabled") and not ends_row("outputs", "vtk")
