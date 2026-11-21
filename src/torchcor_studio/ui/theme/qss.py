"""Qt stylesheet generated from design tokens.

Widgets opt into variants with dynamic properties instead of object names, e.g.
``button.setProperty("variant", "primary")`` or ``label.setProperty("role", "muted")``.
Supported properties:

* ``QPushButton[variant=primary|secondary|ghost|danger|subtle|link]``, ``[size=sm|lg]``,
  ``[inline=true]`` (a ghost button whose icon lines up with the text above it)
* ``QLabel[role=title|h2|h3|muted|faint|caption|mono|eyebrow|value|error|link]``, and for
  the Setup step ``[role=field|unit|rowhead|cardtitle|check]`` (field labels, units inside
  number boxes, the row names of fixed tables, demo card titles, the checks' messages)
* ``QFrame[card=true]``, ``QFrame[card=inset]``, ``QFrame[divider=true]``,
  ``QFrame[hairline=true]`` (rules under table headers and between rail blocks)
* inputs: ``[cell=true]`` (borderless, filled table cells), ``[issue=error|warning]`` (a
  validation issue; ``[invalid=true]`` is the same as an error), ``[flash=true]`` (a field
  just revealed), ``QLineEdit[mono=true]`` (parameter strings, paths),
  ``QLineEdit[regions=true]`` (its placeholder, "All regions", is a value: text2). Custom-
  painted controls (switches, segmented controls) draw ``issue`` and ``flash`` themselves.
"""

from __future__ import annotations

from torchcor_studio.ui.theme.palette import METRICS, MONO_FAMILIES, Metrics, Palette


def mono_family() -> str:
    """The first installed monospace family from :data:`MONO_FAMILIES`."""
    from PySide6.QtGui import QFontDatabase

    available = set(QFontDatabase.families())
    for family in MONO_FAMILIES:
        if family in available:
            return family
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()


def _rgba(color: str, alpha: float) -> str:
    """``#RRGGBB`` → ``rgba(r, g, b, a)`` (QSS has no alpha suffix for hex colours)."""
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)}, {int(c[2:4], 16)}, {int(c[4:6], 16)}, {round(alpha * 255)})"


def build(p: Palette, icons: dict[str, str], m: Metrics = METRICS) -> str:  # noqa: PLR0915
    mono = f'"{mono_family()}", monospace'
    sel = p.soft("accent", 0.20 if p.dark else 0.12)
    sel_hover = p.soft("accent", 0.10 if p.dark else 0.07)
    focus = p.soft("accent", 0.45)
    return f"""
/* ------------------------------------------------------------------ base */
QWidget {{
    color: {p.text};
    font-size: {m.fs_base}px;
    selection-background-color: {sel};
    selection-color: {p.text};
    outline: none;
}}
QMainWindow, QDialog, #AppRoot {{ background: {p.bg0}; }}
QWidget#Sidebar {{ background: {p.bg1}; border-right: 1px solid {p.line}; }}
QWidget#PageHeader {{ background: {p.bg0}; border-bottom: 1px solid {p.line}; }}
QWidget#PageFooter {{ background: {p.bg0}; border-top: 1px solid {p.line}; }}
QWidget#Transparent, QScrollArea#Transparent > QWidget > QWidget {{ background: transparent; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QToolTip {{
    background: {p.bg3}; color: {p.text}; border: 1px solid {p.line2};
    border-radius: {m.r_sm}px; padding: 6px 8px; font-size: {m.fs_sm}px;
}}

/* ------------------------------------------------------------------ type */
QLabel {{ background: transparent; }}
QLabel[role="title"] {{ font-size: {m.fs_2xl}px; font-weight: 650; letter-spacing: -0.3px; }}
QLabel[role="h2"] {{ font-size: {m.fs_xl}px; font-weight: 600; letter-spacing: -0.2px; }}
QLabel[role="h3"] {{ font-size: {m.fs_md}px; font-weight: 600; }}
QLabel[role="muted"] {{ color: {p.text2}; }}
QLabel[role="faint"] {{ color: {p.text3}; }}
QLabel[role="caption"] {{ color: {p.text3}; font-size: {m.fs_sm}px; }}
QLabel[role="eyebrow"] {{ color: {p.text3}; font-size: {m.fs_xs}px; font-weight: 600; letter-spacing: 0.8px; }}
QLabel[role="mono"] {{ font-family: {mono}; font-size: {m.fs_sm}px; color: {p.text2}; }}
QLabel[role="value"] {{ font-size: {m.fs_xl}px; font-weight: 600; letter-spacing: -0.2px; }}
QLabel[role="error"] {{ color: {p.danger}; font-size: {m.fs_sm}px; }}
QLabel[role="link"] {{ color: {p.accent_text}; }}
QLabel[role="field"] {{ color: {p.text2}; font-size: {m.fs_sm}px; font-weight: 500; }}
QLabel[role="unit"] {{ color: {p.text3}; font-size: {m.fs_sm}px; }}
QLabel[role="rowhead"] {{ color: {p.text}; font-size: {m.fs_base}px; font-weight: 500; }}
QLabel[role="cardtitle"] {{ color: {p.text}; font-size: {m.fs_base}px; font-weight: 600; }}
QLabel[role="check"] {{ color: {p.text2}; font-size: {m.fs_sm}px; }}
QLabel:disabled {{ color: {p.text3}; }}

/* ------------------------------------------------------------------ cards */
QFrame[card="true"] {{
    background: {p.bg2}; border: 1px solid {p.line}; border-radius: {m.r_lg}px;
}}
QFrame[card="inset"] {{
    background: {p.bg3}; border: 1px solid {p.line}; border-radius: {m.r_md}px;
}}
QFrame[card="hover"] {{
    background: {p.bg2}; border: 1px solid {p.line}; border-radius: {m.r_lg}px;
}}
QFrame[card="hover"]:hover {{ border-color: {p.line2}; background: {p.bg3}; }}
QFrame[divider="true"] {{ background: {p.line}; border: none; max-height: 1px; min-height: 1px; }}
QFrame[vdivider="true"] {{ background: {p.line}; border: none; max-width: 1px; min-width: 1px; }}
QFrame[hairline="true"] {{ background: {p.line}; border: none; max-height: 1px; min-height: 1px; }}

/* ------------------------------------------------------------------ buttons */
QPushButton, QToolButton {{
    background: {p.bg3}; color: {p.text}; border: 1px solid {p.line2};
    border-radius: {m.r_md}px; padding: 0 12px; min-height: {m.control_h - 2}px;
    font-weight: 500;
}}
QToolButton {{ padding: 0 6px; }}
QPushButton:hover, QToolButton:hover {{ background: {p.bg4}; border-color: {p.line2}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {p.line}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {p.text3}; background: {p.bg2}; border-color: {p.line}; }}
QPushButton:focus, QToolButton:focus {{ border-color: {focus}; }}
QPushButton[variant="primary"] {{
    background: {p.accent}; color: {p.on_accent}; border: 1px solid {p.accent}; font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{ background: {p.accent_hover}; border-color: {p.accent_hover}; }}
QPushButton[variant="primary"]:pressed {{ background: {p.accent_pressed}; border-color: {p.accent_pressed}; }}
QPushButton[variant="primary"]:disabled {{ background: {p.soft("accent", 0.35)}; border-color: transparent; color: {p.soft("on_accent", 0.6)}; }}
QPushButton[variant="danger"] {{ background: transparent; color: {p.danger}; border: 1px solid {p.soft("danger", 0.45)}; }}
QPushButton[variant="danger"]:hover {{ background: {p.soft("danger", 0.10)}; }}
QPushButton[variant="ghost"], QToolButton[variant="ghost"] {{ background: transparent; border-color: transparent; color: {p.text2}; }}
QPushButton[variant="ghost"]:hover, QToolButton[variant="ghost"]:hover {{ background: {p.bg4}; color: {p.text}; }}
QPushButton[variant="ghost"]:checked, QToolButton[variant="ghost"]:checked {{ background: {sel}; color: {p.accent_text}; }}
QPushButton[variant="subtle"] {{ background: {p.soft("accent", 0.12)}; color: {p.accent_text}; border-color: transparent; }}
QPushButton[variant="subtle"]:hover {{ background: {p.soft("accent", 0.20)}; }}
QPushButton[variant="link"] {{ background: transparent; border-color: transparent; color: {p.accent_text}; }}
QPushButton[variant="link"]:hover {{ background: {p.soft("accent", 0.12)}; }}
QPushButton[size="sm"], QToolButton[size="sm"] {{ min-height: 24px; max-height: 26px; padding: 0 8px; font-size: {m.fs_sm}px; border-radius: {m.r_sm}px; }}
QPushButton[size="lg"] {{ min-height: 40px; padding: 0 20px; font-size: {m.fs_md}px; border-radius: {m.r_md + 2}px; }}
QPushButton[inline="true"] {{ padding-left: 2px; }}
QPushButton::menu-indicator {{ image: none; width: 0; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}

/* ------------------------------------------------------------------ inputs */
QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit[editor="true"], QTextEdit[editor="true"] {{
    background: {p.bg3}; color: {p.text}; border: 1px solid {p.line2};
    border-radius: {m.r_md}px; padding: 0 10px; min-height: {m.control_h - 2}px;
    selection-background-color: {p.soft("accent", 0.35)};
}}
QPlainTextEdit[editor="true"], QTextEdit[editor="true"] {{ padding: 6px 8px; }}
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover {{ border-color: {p.text3}; }}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QComboBox:on {{ border: 1px solid {p.accent}; background: {p.bg2}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{ color: {p.text3}; background: {p.bg2}; border-color: {p.line}; }}
QLineEdit[cell="true"], QAbstractSpinBox[cell="true"], QComboBox[cell="true"] {{
    background: {p.bg3 if p.dark else p.bg1}; border: 1px solid transparent; border-radius: {m.r_sm}px;
    min-height: 26px;
}}
QLineEdit[cell="true"]:hover, QAbstractSpinBox[cell="true"]:hover, QComboBox[cell="true"]:hover {{ border-color: {p.line2}; }}
QLineEdit[cell="true"]:focus, QAbstractSpinBox[cell="true"]:focus, QComboBox[cell="true"]:focus,
QComboBox[cell="true"]:on {{ border-color: {p.accent}; background: {p.bg2}; }}
QLineEdit[issue="warning"], QAbstractSpinBox[issue="warning"], QComboBox[issue="warning"],
QLineEdit[issue="warning"]:hover, QAbstractSpinBox[issue="warning"]:hover, QComboBox[issue="warning"]:hover {{
    border-color: {p.warning};
}}
QLineEdit[issue="error"], QAbstractSpinBox[issue="error"], QComboBox[issue="error"],
QLineEdit[issue="error"]:hover, QAbstractSpinBox[issue="error"]:hover, QComboBox[issue="error"]:hover,
QLineEdit[invalid="true"], QAbstractSpinBox[invalid="true"] {{ border-color: {p.danger}; }}
QLineEdit[flash="true"], QAbstractSpinBox[flash="true"], QComboBox[flash="true"] {{
    border-color: {p.accent}; background: {p.soft("accent", 0.16)};
}}
QLineEdit[mono="true"] {{ font-family: {mono}; font-size: {m.fs_sm}px; }}
QLineEdit[regions="true"] {{ placeholder-text-color: {p.text2}; }}
/* the icon button Qt puts inside a field (QLineEdit.addAction): none of the buttons' sizing,
   which would stretch it to a button's height and draw its icon below the field's middle */
QLineEdit QToolButton {{ min-height: 0; min-width: 0; padding: 0; margin: 0; border: none; background: transparent; }}
QLineEdit[flat="true"] {{ background: transparent; border-color: transparent; padding: 0 4px; }}
QLineEdit[flat="true"]:hover {{ background: {p.bg3}; }}
QLineEdit[search="true"] {{ padding-left: 30px; background: {p.bg2}; border-color: {p.line}; }}
QAbstractSpinBox {{ padding: 0 2px; }}
QAbstractSpinBox QLineEdit {{ background: transparent; border: none; padding: 0; margin: 0; min-height: 0; }}
QComboBox QLineEdit {{ background: transparent; border: none; padding: 0; margin: 0; min-height: 0; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0; border: none; }}
QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {{ image: none; width: 0; height: 0; }}
QComboBox {{ padding-right: 28px; combobox-popup: 0; }}
QComboBox::drop-down {{ border: none; width: 26px; subcontrol-origin: padding; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url("{icons.get("chevron-down", "")}"); width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{
    background: {p.bg2}; border: 1px solid {p.line2}; border-radius: {m.r_md}px;
    padding: 4px; outline: none; selection-background-color: {sel};
}}
QComboBox QAbstractItemView::item {{ min-height: 28px; padding: 0 8px; border-radius: {m.r_sm}px; }}
QComboBox QAbstractItemView::item:hover {{ background: {p.bg4}; }}

QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px; border: 1px solid {p.line2}; background: {p.bg3};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {p.text3}; }}
QCheckBox::indicator:checked {{ background: {p.accent}; border-color: {p.accent}; image: url("{icons.get("check", "")}"); }}
QRadioButton::indicator:checked {{ background: {p.bg3}; border: 5px solid {p.accent}; }}

/* ------------------------------------------------------------------ lists & tables */
QListView, QTreeView, QTableView, QListWidget, QTreeWidget, QTableWidget {{
    background: transparent; border: none; outline: none; alternate-background-color: {p.bg3};
}}
QListView::item, QTreeView::item {{ border-radius: {m.r_sm}px; padding: 4px 6px; }}
QListView::item:hover, QTreeView::item:hover {{ background: {sel_hover}; }}
QListView::item:selected, QTreeView::item:selected {{ background: {sel}; color: {p.text}; }}
QTableView {{ gridline-color: {p.line}; }}
QTableView::item {{ padding: 4px 8px; border: none; }}
QTableView::item:selected {{ background: {sel}; color: {p.text}; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background: transparent; color: {p.text3}; border: none; border-bottom: 1px solid {p.line};
    padding: 6px 8px; font-size: {m.fs_xs}px; font-weight: 600; text-transform: uppercase;
}}
QTableCornerButton::section {{ background: transparent; border: none; }}

/* ------------------------------------------------------------------ tabs */
QTabWidget::pane {{ border: none; }}
QTabBar {{ background: transparent; qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent; color: {p.text2}; padding: 8px 2px; margin-right: 18px;
    border: none; border-bottom: 2px solid transparent; font-weight: 500;
}}
QTabBar::tab:hover {{ color: {p.text}; }}
QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {p.accent}; }}
QTabBar::tab:disabled {{ color: {p.text3}; }}

/* ------------------------------------------------------------------ workspace tabs */
QTabWidget#Workspace::pane {{ border: none; }}
QTabWidget#Workspace, QWidget#WorkspaceCorner, QTabBar#WorkspaceTabBar {{ background: {p.bg1}; }}
QTabBar#WorkspaceTabBar::tab {{
    background: transparent; color: {p.text2}; font-weight: 500;
    height: 20px; padding: 2px 4px 2px 10px; margin: 3px 0 0 4px; min-width: 96px;
    border: 1px solid transparent; border-bottom: none;
    border-top-left-radius: {m.r_md}px; border-top-right-radius: {m.r_md}px;
}}
QTabBar#WorkspaceTabBar::tab:hover {{ background: {p.bg2}; color: {p.text}; }}
QTabBar#WorkspaceTabBar::tab:selected {{ background: {p.bg0}; color: {p.text}; border-color: {p.line}; }}
/* in-place renaming: exactly over the title (no control height), in the title's font */
QLineEdit#TabTitleEdit, QLineEdit#RowTitleEdit {{
    min-height: 0; padding: 0 4px; border: 1px solid {p.accent}; border-radius: 4px; background: {p.bg3};
    font-weight: 500;
}}
QLineEdit#RowTitleEdit {{ font-size: 13px; }}
QToolButton#TabClose {{ padding: 0; margin: 0 0 0 4px; border: none; border-radius: 4px; background: transparent; }}
QToolButton#TabClose:hover {{ background: {p.bg3}; }}

/* ------------------------------------------------------------------ scrollbars */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p.line2}; border-radius: 3px; min-height: 32px; margin: 0 2px; }}
QScrollBar::handle:horizontal {{ background: {p.line2}; border-radius: 3px; min-width: 32px; margin: 2px 0; }}
QScrollBar::handle:hover {{ background: {p.text3}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: none; background: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* ------------------------------------------------------------------ misc */
QProgressBar {{
    background: {p.bg4}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px;
    text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {p.accent}; border-radius: 3px; }}
QProgressBar[tone="success"]::chunk {{ background: {p.success}; }}
QProgressBar[tone="danger"]::chunk {{ background: {p.danger}; }}
QProgressBar[tone="muted"]::chunk {{ background: {p.text3}; }}
QSlider::groove:horizontal {{ height: 4px; background: {p.bg4}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
    background: {p.text}; border: 2px solid {p.accent};
}}
QSlider::handle:horizontal:hover {{ background: {p.on_accent}; }}
QSplitter::handle {{ background: {p.line}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter#PanelSplitter::handle:horizontal {{ width: 5px; background: transparent; }}
QSplitter#PanelSplitter::handle:horizontal:hover,
QSplitter#PanelSplitter::handle:horizontal:pressed {{ background: {_rgba(p.accent, 0.35)}; }}
QMenu {{
    background: {p.bg2}; border: 1px solid {p.line2}; border-radius: {m.r_md}px; padding: 4px;
}}
QMenu::item {{ padding: 6px 24px 6px 10px; border-radius: {m.r_sm}px; }}
QMenu::item:selected {{ background: {sel}; }}
QMenu::item:disabled {{ color: {p.text3}; }}
QMenu::separator {{ height: 1px; background: {p.line}; margin: 4px 6px; }}
QMenu::icon {{ padding-left: 8px; }}
QPlainTextEdit[console="true"] {{
    background: {p.plot_bg}; color: {p.text2}; border: 1px solid {p.line};
    border-radius: {m.r_md}px; padding: 8px 10px; font-family: {mono}; font-size: {m.fs_sm}px;
    selection-background-color: {p.soft("accent", 0.35)};
}}
QPlainTextEdit[code="true"] {{
    background: {p.plot_bg}; color: {p.text}; border: 1px solid {p.line};
    border-radius: {m.r_md}px; padding: 10px 12px; font-family: {mono}; font-size: {m.fs_sm}px;
}}
QDialog QLabel[role="title"] {{ font-size: {m.fs_xl}px; }}
QMessageBox {{ background: {p.bg2}; }}
QStatusBar {{ background: {p.bg1}; color: {p.text3}; border-top: 1px solid {p.line}; }}
"""
