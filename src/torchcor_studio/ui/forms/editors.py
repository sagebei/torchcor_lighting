"""One editor widget per field kind, all sharing the small :class:`Editor` protocol.

:func:`make_editor` picks the editor of a field. Number boxes draw their unit faint inside
the box, use tabular figures and drop trailing zeros. ``cell=True`` gives the borderless
28 px variant used in table cells: numbers right-aligned without a unit, choices always as
combo boxes. Combo boxes end a text that does not fit with "…" (the whole text is then
their tooltip), and text boxes show the start of a long value.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEvent, QLocale, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QAction, QDoubleValidator, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSpinBox,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QWidget,
)

from torchcor_studio.core.models import InputInfo
from torchcor_studio.core.simulations.panels import number
from torchcor_studio.core.simulations.types import FieldType
from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import repolish, theme
from torchcor_studio.ui.widgets import Button, SegmentedControl, ToggleSwitch

BrowseFn = Callable[[str, str], str | None]
CONTROL_H = 32
CELL_H = 28
_TNUM = QFont.Tag("tnum")


class _WheelGuard(QObject):
    """Ignore wheel events on unfocused inputs so scrolling a form never edits it."""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QWidget) and not obj.hasFocus():
            event.ignore()
            return True
        return False


_GUARD = _WheelGuard()


def guard_wheel(widget: QWidget) -> None:
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    widget.installEventFilter(_GUARD)
    if isinstance(widget, QAbstractSpinBox):
        inner = widget.findChild(QLineEdit)
        if inner is not None:  # Qt ignores stylesheet padding for the spin box text field
            inner.setTextMargins(8, 0, 8, 0)


def tabular(widget: QWidget) -> QWidget:
    """Tabular (equal-width) figures, so numbers line up in columns."""
    font = widget.font()
    font.setFeature(_TNUM, 1)
    widget.setFont(font)
    return widget


def _cell(widget: QWidget, height: int = CELL_H) -> None:
    widget.setProperty("cell", True)
    widget.setFixedHeight(height)


def _plain(value: float, decimals: int, point: str) -> str:
    """``value`` with at most ``decimals`` decimals, no trailing zeros and no thousands
    separator: "0.408", "1000", "157.376262"."""
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text.replace(".", point)


class _Spin(QDoubleSpinBox):
    """A number box that drops trailing zeros ("0.408", "1000"); it keeps its decimals."""

    def textFromValue(self, value: float) -> str:  # noqa: N802
        return _plain(value, self.decimals(), self.locale().decimalPoint())


class _Combo(QComboBox):
    """A combo box whose text ends with "…" when it does not fit (the whole text is then
    its tooltip), instead of being cut."""

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        field = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt,
                                            QStyle.SubControl.SC_ComboBoxEditField, self)
        full = opt.currentText
        opt.currentText = QFontMetrics(self.font()).elidedText(full, Qt.TextElideMode.ElideRight,
                                                               max(0, field.width() - 2))
        tip = full if opt.currentText != full else ""
        if self.toolTip() != tip and not self.property("issue"):
            self.setToolTip(tip)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)


class _Unit(QObject):
    """A faint unit drawn inside the right edge of a number box (no suffix in the text);
    it hides while the box shows its special text ("Whole beats" for 0)."""

    def __init__(self, box: QWidget, unit: str) -> None:
        super().__init__(box)
        self._box = box
        self.label = QLabel(unit, box)
        self.label.setProperty("role", "unit")
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.label.ensurePolished()
        self.label.adjustSize()
        if isinstance(box, QLineEdit):  # its padding already insets the text by 10 px
            box.setTextMargins(0, 0, self.label.width() + 4, 0)
        else:  # the spin box's inner line edit has no padding of its own
            box.findChild(QLineEdit).setTextMargins(8, 0, self.label.width() + 14, 0)
        box.installEventFilter(self)
        if isinstance(box, QAbstractSpinBox) and box.specialValueText():
            box.valueChanged.connect(self._special)

    def _special(self, value: float) -> None:
        self.label.setVisible(value != self._box.minimum())

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self._box and event.type() == QEvent.Type.Resize:
            self.label.move(self._box.width() - self.label.width() - 10,
                            (self._box.height() - self.label.height()) // 2)
            self.label.raise_()
        return False


class Editor:
    """Common interface: ``widget``, ``get()``, ``set(value)``, ``on_change`` and
    ``set_issue`` (an error or warning tint with its message as tooltip)."""

    widget: QWidget

    def __init__(self, on_change: Callable[[], None]) -> None:
        self._on_change = on_change
        self._muted = False
        self._issue: tuple[str, str] = ("", "")

    def emit(self) -> None:
        if not self._muted:
            self._on_change()

    def get(self) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def set(self, value: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def set_quiet(self, value: Any) -> None:
        self._muted = True
        try:
            self.set(value)
        finally:
            self._muted = False

    def set_issue(self, severity: str | None, message: str = "") -> None:
        """Tint the editor ``"error"`` or ``"warning"`` (``None`` clears it)."""
        target = self.focus_target()
        state = (severity or "", message if severity else "")
        if target is None or state == self._issue:
            return
        self._issue = state
        target.setProperty("issue", state[0])
        target.setToolTip(state[1])
        repolish(target)

    def issue(self) -> str:
        return self._issue[0]

    def focus_target(self) -> QWidget | None:
        return self.widget

    def update_context(self, info: InputInfo | None) -> None:
        """Refresh dynamic choices (vtx files, region ids) from the input folder."""

    def set_enabled(self, enabled: bool) -> None:
        self.widget.setEnabled(enabled)


# ---------------------------------------------------------------------------- numbers
def _auto_decimals(value: float, step: float | None) -> int:
    """Decimals that show both the step and the value whole (1 to 6)."""
    places = [len(f"{abs(v):.10f}".rstrip("0").split(".")[1]) for v in (step, value) if v]
    return min(6, max(1, *places)) if places else 2


def _number_box(box: QWidget, meta: dict, cell: bool) -> None:
    """Height, alignment, figures and the unit of a number box; a ``placeholder`` is the
    text shown for the minimum ("Whole beats" for a duration override of 0)."""
    if cell:
        _cell(box)
        box.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    else:
        box.setFixedHeight(CONTROL_H)
    tabular(box)
    if isinstance(box, QAbstractSpinBox):
        box.setKeyboardTracking(False)
        box.setSpecialValueText(meta.get("placeholder", ""))
        guard_wheel(box)
    if meta.get("unit") and not cell:
        _Unit(box, meta["unit"])


class FloatEditor(Editor):
    def __init__(self, meta: dict, default: float, on_change, optional: bool = False,
                 cell: bool = False) -> None:
        super().__init__(on_change)
        lo, hi = meta.get("min"), meta.get("max")
        ref = meta.get("default", default)
        ref = ref if isinstance(ref, int | float) else default
        scientific = bool(meta.get("log")) or (ref not in (None, 0) and abs(ref) < 1e-3)
        if scientific:
            self._edit: QLineEdit | None = QLineEdit()
            v = QDoubleValidator(lo if lo is not None else -1e300, hi if hi is not None else 1e300, 12)
            v.setNotation(QDoubleValidator.Notation.ScientificNotation)
            self._edit.setValidator(v)
            self._edit.editingFinished.connect(self.emit)
            self._spin: QDoubleSpinBox | None = None
            box: QWidget = self._edit
        else:
            self._edit = None
            self._spin = _Spin()
            self._spin.setDecimals(meta.get("decimals") or _auto_decimals(ref or 0.0, meta.get("step")))
            self._spin.setRange(lo if lo is not None else -1e9, hi if hi is not None else 1e9)
            if meta.get("step"):
                self._spin.setSingleStep(meta["step"])
            else:
                self._spin.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
            self._spin.valueChanged.connect(lambda *_: self.emit())
            box = self._spin
        box.setMinimumWidth(56 if cell else 72)
        _number_box(box, meta, cell)
        self._toggle: ToggleSwitch | None = None
        if optional:
            self.widget = QWidget()
            lay = QHBoxLayout(self.widget)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(8)
            self._toggle = ToggleSwitch(default is not None)
            self._toggle.setToolTip("Enable to set a value")
            self._toggle.toggled.connect(self._toggled)
            lay.addWidget(self._toggle)
            lay.addWidget(box, 1)
        else:
            self.widget = box
        self.set_quiet(default)

    def _toggled(self, on: bool) -> None:
        self.focus_target().setEnabled(on)
        self.emit()

    def focus_target(self) -> QWidget:
        return self._spin if self._spin is not None else self._edit

    def get(self) -> float | None:
        if self._toggle is not None and not self._toggle.isChecked():
            return None
        if self._spin is not None:
            return float(self._spin.value())
        text = self._edit.text().strip()
        try:
            return float(text)
        except ValueError:
            return text or None  # let pydantic report it

    def set(self, value: float | None) -> None:
        if self._toggle is not None:
            self._toggle.setChecked(value is not None)
            self.focus_target().setEnabled(value is not None)
        if value is None:
            return
        if self._spin is not None:
            self._spin.setValue(float(value))
        else:
            self._edit.setText(number(float(value), 6))  # "1000", "1e-5"
            self._edit.setCursorPosition(0)


class NumberCellEditor(Editor):
    """A number in a table cell: a light line edit rather than a spin box (a table has
    dozens), right-aligned in tabular figures and without trailing zeros. Up and Down step
    it; a value that is not a number, or out of range, is left for validation to report."""

    def __init__(self, meta: dict, default: float | int | None, on_change, integer: bool) -> None:
        super().__init__(on_change)
        ref = meta.get("default", default)
        ref = ref if isinstance(ref, int | float) else (default or 0)
        self._int = integer
        self._log = bool(meta.get("log")) or (not integer and ref not in (None, 0) and abs(ref) < 1e-3)
        self._decimals = 0 if integer else meta.get("decimals") or _auto_decimals(float(ref or 0), meta.get("step"))
        self._step = meta.get("step") or (1 if integer else 10.0 ** -min(self._decimals, 2))
        self._range = (meta.get("min"), meta.get("max"))
        self._point = QLocale().decimalPoint()
        self.widget = _CellEdit(self.step)
        _number_box(self.widget, meta, cell=True)
        self.widget.editingFinished.connect(self._finished)
        self._shown: str | None = None
        self.set_quiet(default)

    def _text(self, value: float | int) -> str:
        if self._int:
            return str(int(value))
        return number(float(value), 6) if self._log else _plain(float(value), self._decimals, self._point)

    def _finished(self) -> None:
        text = self.widget.text()
        if text != self._shown:  # typed, not just left
            self._shown = text
            self.emit()

    def step(self, up: bool) -> None:
        """Up or Down: one step, within the field's range."""
        value = self.get()
        if not isinstance(value, int | float):
            return
        value += self._step if up else -self._step
        lo, hi = self._range
        value = max(lo, value) if lo is not None else value
        value = min(hi, value) if hi is not None else value
        self.set(int(value) if self._int else round(value, self._decimals))
        self.emit()

    def get(self) -> float | int | str:
        text = self.widget.text().strip().replace(self._point, ".")
        try:
            return int(text) if self._int else float(text)
        except ValueError:
            return text  # let pydantic report it

    def set(self, value) -> None:
        text = "" if value is None else self._text(value)
        self._shown = text
        if text != self.widget.text():
            self.widget.setText(text)
            self.widget.setCursorPosition(0)


class _CellEdit(QLineEdit):
    """A number cell's line edit: Up and Down step its number."""

    def __init__(self, step: Callable[[bool], None]) -> None:
        super().__init__()
        self._step = step

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            self._step(event.key() == Qt.Key.Key_Up)
        else:
            super().keyPressEvent(event)


class IntEditor(Editor):
    def __init__(self, meta: dict, default: int, on_change, cell: bool = False) -> None:
        super().__init__(on_change)
        self.widget = QSpinBox()
        lo, hi = meta.get("min"), meta.get("max")
        self.widget.setRange(int(lo) if lo is not None else -2_000_000_000,
                             int(hi) if hi is not None else 2_000_000_000)
        self.widget.setMinimumWidth(48 if cell else 72)
        if meta.get("step"):
            self.widget.setSingleStep(int(meta["step"]))
        self.widget.valueChanged.connect(lambda *_: self.emit())
        _number_box(self.widget, meta, cell)
        self.set_quiet(default if default is not None else 0)

    def get(self) -> int:
        return int(self.widget.value())

    def set(self, value: int) -> None:
        self.widget.setValue(int(value or 0))


class VecEditor(Editor):
    """Fixed-length numeric tuple, e.g. slab dimensions (x, y, z)."""

    def __init__(self, meta: dict, default, ftype: FieldType, on_change) -> None:
        super().__init__(on_change)
        self.widget = QWidget()
        lay = QHBoxLayout(self.widget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._spins: list[QDoubleSpinBox] = []
        names = ("x", "y", "z", "w")
        default = tuple(default or (0,) * ftype.length)
        for i in range(ftype.length):
            spin = QDoubleSpinBox()
            spin.setDecimals(meta.get("decimals") or _auto_decimals(float(default[i]), meta.get("step")))
            spin.setRange(meta.get("min", -1e9) if meta.get("min") is not None else -1e9,
                          meta.get("max", 1e9) if meta.get("max") is not None else 1e9)
            spin.setPrefix(f"{names[i]}  ")
            if meta.get("unit"):
                spin.setSuffix(f"  {meta['unit']}")
            spin.setMinimumWidth(64)
            spin.setFixedHeight(CONTROL_H)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda *_: self.emit())
            guard_wheel(spin)
            self._spins.append(spin)
            lay.addWidget(spin, 1)
        self._int = ftype.inner is int
        self.set_quiet(default)

    def focus_target(self) -> QWidget:
        return self._spins[0]

    def get(self):
        vals = [s.value() for s in self._spins]
        return tuple(int(v) for v in vals) if self._int else tuple(float(v) for v in vals)

    def set(self, value) -> None:
        for spin, v in zip(self._spins, value or (), strict=False):
            spin.setValue(float(v))


# ---------------------------------------------------------------------------- choices
class BoolEditor(Editor):
    def __init__(self, default: bool, on_change) -> None:
        super().__init__(on_change)
        self.widget = ToggleSwitch(bool(default))
        self.widget.toggled.connect(lambda *_: self.emit())

    def get(self) -> bool:
        return self.widget.isChecked()

    def set(self, value: bool) -> None:
        self.widget.setChecked(bool(value))


def _choice_labels(meta: dict, values: tuple) -> list[tuple[Any, str]]:
    labels = meta.get("choices") or {}
    out = []
    for v in values:
        key = v.value if hasattr(v, "value") else v
        out.append((key, labels.get(key, labels.get(str(key), str(key).replace("_", " ").capitalize()))))
    return out


def is_segmented(meta: dict, ftype: FieldType) -> bool:
    """A few short options get a segmented control, others a combo box."""
    values = ftype.choices or tuple((meta.get("choices") or {}).keys())
    options = _choice_labels(meta, values)
    hint = meta.get("widget")
    short = len(options) <= 4 and sum(len(lbl) for _v, lbl in options) <= 44
    return hint == "segmented" or (short and hint != "combo")


class ChoiceEditor(Editor):
    """Segmented control for a few short options, a combo box otherwise (always in cells)."""

    def __init__(self, meta: dict, default, ftype: FieldType, on_change, cell: bool = False) -> None:
        super().__init__(on_change)
        values = ftype.choices or tuple((meta.get("choices") or {}).keys())
        options = _choice_labels(meta, values)
        self._segmented = not cell and is_segmented(meta, ftype)
        if self._segmented:
            self.widget = SegmentedControl(options, value=default)
            self.widget.changed.connect(lambda *_: self.emit())
        else:
            self.widget = _Combo()
            for v, lbl in options:
                self.widget.addItem(lbl, v)
            self.widget.setMinimumContentsLength(6)
            self.widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            if cell:
                _cell(self.widget)
            else:
                self.widget.setFixedHeight(CONTROL_H)
            guard_wheel(self.widget)
            self.widget.currentIndexChanged.connect(lambda *_: self.emit())
        self.set_quiet(default)

    def labels(self) -> list[str]:
        """Every option's label (a table sizes its column for the longest)."""
        return ([self.widget.itemText(i) for i in range(self.widget.count())]
                if not self._segmented else [])

    def get(self):
        if self._segmented:
            return self.widget.value()
        return self.widget.currentData()

    def set(self, value) -> None:
        value = value.value if hasattr(value, "value") else value
        if self._segmented:
            self.widget.set_value(value, animate=False)
        else:
            idx = self.widget.findData(value)
            if idx >= 0:
                self.widget.setCurrentIndex(idx)


# ---------------------------------------------------------------------------- text & paths
class TextEditor(Editor):
    def __init__(self, meta: dict, default: str, on_change, optional: bool = False,
                 cell: bool = False) -> None:
        super().__init__(on_change)
        self._optional = optional
        self.widget = QLineEdit()
        self.widget.setPlaceholderText(meta.get("placeholder", ""))
        if meta.get("widget") == "mono":
            self.widget.setProperty("mono", True)
        if cell:
            _cell(self.widget)
        else:
            self.widget.setFixedHeight(CONTROL_H)
        self.widget.editingFinished.connect(self.emit)
        self.set_quiet(default)

    def get(self):
        text = self.widget.text().strip()
        return None if (self._optional and not text) else text

    def set(self, value) -> None:
        text = "" if value is None else str(value)
        if text != self.widget.text():
            self.widget.setText(text)
            self.widget.setCursorPosition(0)  # long values show their start


class PathEditor(Editor):
    """Path on the target machine with a Browse… button (local dialog or remote browser)."""

    def __init__(self, meta: dict, default: str, on_change, mode: str, browse: BrowseFn | None,
                 optional: bool = False) -> None:
        super().__init__(on_change)
        self._optional = optional
        self._mode = mode
        self._browse = browse
        self.widget = QWidget()
        lay = QHBoxLayout(self.widget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._edit = QLineEdit()
        self._edit.setFixedHeight(CONTROL_H)
        self._edit.setPlaceholderText(meta.get("placeholder") or ("Folder on the compute machine" if mode == "dir"
                                                                  else "File on the compute machine"))
        self._edit.editingFinished.connect(self.emit)
        lay.addWidget(self._edit, 1)
        btn = Button("Browse…", "folder-open" if mode == "dir" else "file", size="sm")
        btn.setEnabled(browse is not None)
        btn.clicked.connect(self._pick)
        lay.addWidget(btn)
        self.set_quiet(default)

    def focus_target(self) -> QWidget:
        return self._edit

    def _pick(self) -> None:
        if self._browse is None:
            return
        path = self._browse(self._mode, self._edit.text())
        if path:
            self._edit.setText(path)
            self.emit()

    def get(self):
        text = self._edit.text().strip()
        return None if (self._optional and not text) else text

    def set(self, value) -> None:
        self._edit.setText("" if value is None else str(value))


class VtxEditor(Editor):
    """Stimulus site: a ``.vtx`` file from the input folder (editable for custom names)."""

    def __init__(self, meta: dict, default: str, on_change, cell: bool = False) -> None:
        super().__init__(on_change)
        self.widget = QComboBox()
        self.widget.setMinimumWidth(110)
        self.widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.widget.setMinimumContentsLength(8)
        self.widget.setEditable(True)
        self.widget.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.widget.lineEdit().setPlaceholderText(meta.get("placeholder") or "Choose a .vtx site")
        if cell:
            _cell(self.widget)
        guard_wheel(self.widget)
        self.widget.currentTextChanged.connect(lambda *_: self.emit())
        self._fixed = list((meta.get("choices") or {}).items())
        self._names: list[tuple[str, int]] | None = None
        self._populate(None)
        self.set_quiet(default)

    def _populate(self, info: InputInfo | None) -> None:
        names = [(v.name, v.count) for v in info.vtx] if info else []
        if names == self._names:
            return  # the same folder again: nothing to refresh
        self._names = names
        current = self.widget.currentText()
        self.widget.blockSignals(True)
        self.widget.clear()
        for value, text in self._fixed:
            self.widget.addItem(text, value)
        for name, count in names:
            self.widget.addItem(name, name)
            self.widget.setItemData(self.widget.count() - 1, f"{count:,} nodes", Qt.ItemDataRole.ToolTipRole)
        self._show(current)
        self.widget.blockSignals(False)

    def _show(self, text: str) -> None:
        """Show ``text`` from its start ("RV_moderator.vtx", not "_moderator.vtx")."""
        self.widget.setEditText(text)
        self.widget.lineEdit().setCursorPosition(0)

    def labels(self) -> list[str]:
        """The sites offered and the one shown (a table sizes its column for the longest)."""
        return [self.widget.itemText(i) for i in range(self.widget.count())] + [self.widget.currentText()]

    def update_context(self, info: InputInfo | None) -> None:
        self._populate(info)

    def get(self) -> str:
        text = self.widget.currentText().strip()
        for value, label_text in self._fixed:
            if text == label_text:
                return value
        return text

    def set(self, value) -> None:
        value = "" if value is None else str(value)
        for value_key, label_text in self._fixed:
            if value == value_key:
                self._show(label_text)
                return
        if value != self.widget.currentText():
            self._show(value)


_LIST_SPLIT = re.compile(r"[,\s;]+")
_RANGE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")
MAX_RANGE = 10_000  # region ids a single range may expand to


def format_regions(values: list[int]) -> str:
    """``[5, 6, 7, ..., 24, 27]`` -> ``"5, 6, 7–24, 27"``: runs of three or more as ranges."""
    parts: list[str] = []
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"{values[i]}–{values[j]}")
        else:
            parts.extend(str(v) for v in values[i:j + 1])
        i = j + 1
    return ", ".join(parts)


def parse_regions(text: str) -> list[int] | str:
    """``"1-3, 7"`` (or "1–3") -> ``[1, 2, 3, 7]``. Text that is not that comes back as it
    is, for pydantic to report: a backwards range ("24-7", which would read as no region,
    i.e. all of them) or one of more than :data:`MAX_RANGE` regions."""
    ranges = [(int(a), int(b)) for a, b in _RANGE.findall(text)]
    if any(b < a or b - a >= MAX_RANGE for a, b in ranges):
        return text
    expanded = _RANGE.sub(lambda m: " ".join(str(v) for v in range(int(m.group(1)), int(m.group(2)) + 1)),
                          text)
    try:
        return [int(p) for p in _LIST_SPLIT.split(expanded.strip()) if p]
    except ValueError:
        return text


class ListEditor(Editor):
    """Comma-separated numbers or strings. With ``widget="regions"`` it reads and writes
    ranges ("7–24"), shows an empty list as "All regions", and offers the mesh's region ids
    from a button inside the box."""

    def __init__(self, meta: dict, default, ftype: FieldType, on_change, regions: bool = False,
                 cell: bool = False) -> None:
        super().__init__(on_change)
        self._type = ftype.inner or str
        self._optional = ftype.optional
        self._regions = regions
        self.widget = self._edit = QLineEdit()
        self._edit.setPlaceholderText(meta.get("placeholder") or ("All regions" if regions else "comma-separated"))
        if cell:
            _cell(self._edit)
        else:
            self._edit.setFixedHeight(CONTROL_H)
        tabular(self._edit)
        self._edit.editingFinished.connect(self.emit)
        self._pick: QAction | None = None
        self._known: list[int] | None = None
        if regions:
            self._edit.setProperty("regions", True)
            self._pick = self._edit.addAction(icons.icon("list", theme().palette.text2, 16,
                                                         disabled=theme().palette.text3),
                                              QLineEdit.ActionPosition.TrailingPosition)
            self._pick.triggered.connect(self._show_menu)
            self.update_context(None)
        self.set_quiet(default)

    def update_context(self, info: InputInfo | None) -> None:
        if self._pick is not None:
            self._known = list(info.regions) if info and info.regions else None
            self._pick.setEnabled(bool(self._known))
            self._pick.setToolTip("Choose regions" if self._known else
                                  "Choose regions: listed once the input folder's mesh is read")

    def _show_menu(self) -> None:
        """The mesh's regions, ticked when listed."""
        menu = QMenu(self._edit)
        values = self.get()
        current = set(values) if isinstance(values, list) else set()
        for r in self._known or []:
            act = menu.addAction(f"Region {r}")
            act.setCheckable(True)
            act.setChecked(r in current)
            act.toggled.connect(lambda on, r=r: self._toggle(r, on))
        menu.exec(self._edit.mapToGlobal(QPoint(self._edit.width() - menu.sizeHint().width(),
                                                self._edit.height())))

    def _toggle(self, region: int, on: bool) -> None:
        current = self.get()
        values = list(current) if isinstance(current, list) else []
        if on and region not in values:
            values.append(region)
        if not on and region in values:
            values.remove(region)
        self.set(sorted(values))
        self.emit()

    def get(self):
        text = self._edit.text().strip()
        if not text:
            return None if self._optional else []
        if self._regions:
            return parse_regions(text)
        parts = [p for p in _LIST_SPLIT.split(text) if p]
        try:
            return [self._type(p) for p in parts]
        except ValueError:
            return text  # let pydantic report it

    def set(self, value) -> None:
        if not value:
            text = ""
        elif self._regions and all(isinstance(v, int) for v in value):
            text = format_regions(list(value))
        else:
            text = ", ".join(_fmt_num(v) for v in value)
        if text != self._edit.text():
            self._edit.setText(text)
            self._edit.setCursorPosition(0)


def _fmt_num(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


class CodeFieldEditor(Editor):
    """Multi-line Python source (``widget="code"``), validated after a short pause in typing."""

    def __init__(self, meta: dict, default: str, on_change) -> None:
        super().__init__(on_change)
        from torchcor_studio.ui.widgets.code import CodeEditor

        self.widget = CodeEditor(default or "")
        self.widget.setMinimumHeight(int(meta.get("height", 460)))
        self._debounce = QTimer(self.widget)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(400)
        self._debounce.timeout.connect(self.emit)
        self.widget.textChanged.connect(self._typed)

    def _typed(self) -> None:
        if not self._muted:  # text set from code (a demo, Undo) is no edit
            self._debounce.start()

    def get(self) -> str:
        return self.widget.toPlainText()

    def set(self, value) -> None:
        self._debounce.stop()
        if (value or "") != self.widget.toPlainText():
            self.widget.setPlainText(value or "")


def make_editor(ftype: FieldType, meta: dict, default: Any, on_change: Callable[[], None],
                browse: BrowseFn | None, *, cell: bool = False) -> Editor:
    """The editor of one field (``cell``: the compact variant for table cells)."""
    widget = meta.get("widget")
    kind = ftype.kind
    if cell and kind in ("float", "int") and not ftype.optional:
        return NumberCellEditor(meta, default, on_change, integer=kind == "int")
    if kind == "float":
        return FloatEditor(meta, default, on_change, optional=ftype.optional, cell=cell)
    if kind == "int":
        return IntEditor(meta, default, on_change, cell=cell)
    if kind == "bool":
        return BoolEditor(default, on_change)
    if kind == "choice":
        return ChoiceEditor(meta, default, ftype, on_change, cell=cell)
    if kind == "vec":
        return VecEditor(meta, default, ftype, on_change)
    if kind in ("int_list", "float_list", "str_list"):
        return ListEditor(meta, default, ftype, on_change, regions=widget == "regions", cell=cell)
    if widget == "vtx":
        return VtxEditor(meta, default, on_change, cell=cell)
    if widget == "code":
        return CodeFieldEditor(meta, default, on_change)
    if widget in ("dir", "file", "path"):
        return PathEditor(meta, default, on_change, "file" if widget == "file" else "dir", browse,
                          optional=ftype.optional)
    if meta.get("choices"):
        return ChoiceEditor(meta, default, ftype, on_change, cell=cell)
    return TextEditor(meta, default, on_change, optional=ftype.optional, cell=cell)
