"""Panels: how a kind's parameters are grouped on the Setup step.

A kind declares ``PANELS``, titled groups in physiology order. A path names a section, a
single field (a scalar or a list), :data:`INPUT` (the input-folder slot) or :data:`OUTPUT`
(the output-folder slot, which every kind's first panel ends with);
``("path", "Eyebrow")`` forces an eyebrow above that block. :func:`resolve` turns them
into blocks the form renders:

1. a path naming a scalar or a list is an explicit claim, wherever it appears;
2. a section path claims that section's leaves in model order, minus explicit claims
   anywhere and leaves claimed earlier; nested sub-sections start their own blocks;
3. consecutive scalars form one fields block (a section claim's scalars join the block
   just before it when that block holds leaves of the same section); a list of sections
   is a table block, ``INPUT`` an input block and ``OUTPUT`` an output block;
4. eyebrows: a forced one always shows; the panel's first block has none; a later table
   shows its list title (and its help as caption unless it is ``fixed``); a later block
   started by a section claim shows the section's title unless an earlier block of the
   panel holds leaves of that section; blocks of explicit scalars show none;
5. leaves no panel claims go to a trailing "Other" panel; no ``PANELS`` at all gives one
   panel per top-level section or list, after one for the top-level scalars.

Consecutive panels can share a :class:`Group`: one box headed by the group's title, with
the panels as its sections (each still opens, closes and counts its issues on its own).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from torchcor_studio.core.simulations.fields import field_meta
from torchcor_studio.core.simulations.types import resolve as field_type

INPUT = "@input"
OUTPUT = "@output"
SLOTS = {INPUT: "input", OUTPUT: "output"}  # slot path → its block kind
Path = tuple[str, ...]


@dataclass(frozen=True)
class Group:
    """A box around consecutive panels, headed by its title and icon."""

    title: str
    icon: str

    @property
    def key(self) -> str:
        return f"group-{slug(self.title)}"


@dataclass(frozen=True)
class Panel:
    """A titled group of parameters (see the module docstring)."""

    title: str
    icon: str
    paths: tuple[str | tuple[str, str], ...]
    collapsed: bool = False
    group: Group | None = None


@dataclass(frozen=True)
class Block:
    """Part of a panel: a folder slot, a run of fields, or one table."""

    kind: Literal["input", "output", "fields", "table"]
    leaves: tuple[Path, ...] = ()
    eyebrow: str = ""
    caption: str = ""
    tip: str = ""  # the eyebrow's tooltip (a fixed table's help)


@dataclass(frozen=True)
class ResolvedPanel:
    key: str  # slug of the title: "time-and-outputs"
    title: str
    icon: str
    collapsed: bool
    blocks: tuple[Block, ...]
    leaves: tuple[Path, ...]  # every claimed leaf, in block order
    group: Group | None = None


@dataclass(frozen=True)
class Change:
    """One parameter that differs from the baseline."""

    path: tuple[str | int, ...]  # ("stimuli", 3, "start"); a list whose length changed: ("stimuli",)
    panel: str  # key of the owning panel
    label: str  # "Pacing › Stimulus 4 › Start: 5 → 8 ms"


# ------------------------------------------------------------------------------- model walk
def node(model_cls: type[BaseModel], path: Sequence[str]) -> type[BaseModel] | None:
    """The section class at ``path`` (``None`` for a scalar or a list); ``KeyError`` if unknown."""
    cls: type[BaseModel] | None = model_cls
    for name in path:
        if cls is None or name not in cls.model_fields:
            raise KeyError(".".join(path))
        ftype = field_type(cls.model_fields[name].annotation)
        cls = ftype.inner if ftype.kind == "section" else None
    return cls


def leaves(model_cls: type[BaseModel], prefix: Path = ()) -> list[Path]:
    """Every leaf below ``prefix`` in model order: scalars and lists (a list of sections is
    one leaf, rendered as a table)."""
    out: list[Path] = []
    for name, info in model_cls.model_fields.items():
        ftype = field_type(info.annotation)
        if ftype.kind == "section":
            out += leaves(ftype.inner, (*prefix, name))
        else:
            out.append((*prefix, name))
    return out


def leaf_meta(model_cls: type[BaseModel], leaf: Sequence[str | int]) -> tuple[dict, Any]:
    """``(metadata, field type)`` of a leaf, or of a column of a table (``("stimuli", 0,
    "start")``: row indices are skipped)."""
    names = [p for p in leaf if not isinstance(p, int)]
    cls = model_cls
    for name in names[:-1]:
        ftype = field_type(cls.model_fields[name].annotation)
        cls = ftype.inner
    return field_meta(cls, names[-1]), field_type(cls.model_fields[names[-1]].annotation)


def nouns(meta: dict) -> tuple[str, str]:
    """``(one, several)`` for the rows of a list: ``("stimulus", "stimuli")``."""
    one = meta.get("item") or "row"
    return one, meta.get("items") or f"{one}s"


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


# ------------------------------------------------------------------------------- resolution
@dataclass
class _Draft:
    kind: Literal["input", "output", "fields", "table"]
    leaves: list[Path] = field(default_factory=list)
    eyebrow: str = ""
    caption: str = ""
    tip: str = ""
    explicit: bool = False  # started by an explicit scalar claim: the next one joins it


def default_panels(model_cls: type[BaseModel]) -> tuple[Panel, ...]:
    """One panel per top-level section or list, after one for the top-level scalars."""
    scalars, groups = [], []
    for name, info in model_cls.model_fields.items():
        kind = field_type(info.annotation).kind
        if kind in ("section", "rows"):
            groups.append(Panel(field_meta(model_cls, name)["title"],
                                "layers" if kind == "section" else "rows", (name,)))
        else:
            scalars.append(name)
    lead = (Panel(model_cls.__name__, "sliders", tuple(scalars)),) if scalars else ()
    return lead + tuple(groups)


def resolve(model_cls: type[BaseModel], panels: Sequence[Panel] = ()) -> tuple[ResolvedPanel, ...]:
    """The blocks of every panel (see the module docstring); ``KeyError`` for an unknown path."""
    panels = tuple(panels) or default_panels(model_cls)
    runs = [p.group for i, p in enumerate(panels)
            if p.group is not None and (i == 0 or panels[i - 1].group != p.group)]
    if len(runs) != len(set(runs)):
        raise ValueError("the panels of a group must be consecutive")
    specs = [[(s, None) if isinstance(s, str) else s for s in p.paths] for p in panels]
    explicit = {tuple(raw.split(".")) for paths in specs for raw, _eyebrow in paths
                if raw not in SLOTS and node(model_cls, raw.split(".")) is None}
    claimed: set[Path] = set()
    out: list[ResolvedPanel] = []
    for panel, paths in zip(panels, specs, strict=True):
        blocks: list[_Draft] = []
        for raw, forced in paths:
            if raw in SLOTS:
                blocks.append(_Draft(SLOTS[raw], eyebrow=forced or ""))
                continue
            path = tuple(raw.split("."))
            section = node(model_cls, path)
            if section is None:  # an explicit scalar or list
                if path in claimed:
                    raise ValueError(f"{raw} is claimed twice")
                claimed.add(path)
                _place(model_cls, blocks, path, forced, section=None, parent_changed=False)
                continue
            parent = path
            for i, leaf in enumerate(lf for lf in leaves(section, path)
                                     if lf not in explicit and lf not in claimed):
                claimed.add(leaf)
                _place(model_cls, blocks, leaf, forced if i == 0 else None, section=path,
                       parent_changed=leaf[:-1] != parent)
                parent = leaf[:-1]
        out.append(_finish(_unique(slug(panel.title), out), panel.title, panel.icon,
                           panel.collapsed, blocks, panel.group))
    rest = [leaf for leaf in leaves(model_cls) if leaf not in claimed]
    if rest:
        blocks = []
        for leaf in rest:
            _place(model_cls, blocks, leaf, None, section=None, parent_changed=False)
        out.append(_finish(_unique("other", out), "Other", "more", False, blocks))
    return tuple(out)


def _unique(key: str, panels: Sequence[ResolvedPanel]) -> str:
    taken = {p.key for p in panels}
    n = 1
    while (candidate := key if n == 1 else f"{key}-{n}") in taken:
        n += 1
    return candidate


def _holds(blocks: Sequence[_Draft], section: Path) -> bool:
    return any(leaf[:len(section)] == section for b in blocks for leaf in b.leaves)


def _place(model_cls: type[BaseModel], blocks: list[_Draft], leaf: Path, forced: str | None, *,
           section: Path | None, parent_changed: bool) -> None:
    """Add one claimed leaf to a panel's blocks (``section``: the section claim it comes
    from, ``None`` for an explicit claim), starting a block and its eyebrow per rules 3-4."""
    meta, ftype = leaf_meta(model_cls, leaf)
    if ftype.kind == "rows":
        draft = _Draft("table", [leaf])
        if forced or blocks:
            draft.eyebrow = forced or meta["title"]
            if meta.get("fixed"):
                draft.tip = meta.get("help", "")
            else:
                draft.caption = meta.get("help", "")
        blocks.append(draft)
        return
    last = blocks[-1] if blocks else None
    if (forced is None and not parent_changed and last is not None and last.kind == "fields"
            and (last.explicit if section is None else _holds([last], section))):
        last.leaves.append(leaf)
        return
    draft = _Draft("fields", [leaf], explicit=section is None)
    if forced:
        draft.eyebrow = forced
    elif section is not None and len(leaf) - 1 > len(section):  # a nested sub-section
        draft.eyebrow = leaf_meta(model_cls, leaf[:-1])[0]["title"]
    elif section is not None and blocks and not _holds(blocks, section):
        draft.eyebrow = leaf_meta(model_cls, section)[0]["title"]
    blocks.append(draft)


def _finish(key: str, title: str, icon: str, collapsed: bool, blocks: list[_Draft],
            group: Group | None = None) -> ResolvedPanel:
    frozen = tuple(Block(b.kind, tuple(b.leaves), b.eyebrow, b.caption, b.tip) for b in blocks)
    return ResolvedPanel(key, title, icon, collapsed, frozen,
                         tuple(leaf for b in frozen for leaf in b.leaves), group)


# ------------------------------------------------------------------------------- queries
def owner(resolved: Sequence[ResolvedPanel], dotted: str) -> ResolvedPanel | None:
    """The panel owning a parameter path ("stimuli.0.site"): the one whose claimed leaf is
    the longest prefix of it, else the one holding the first leaf below it (a section).
    :data:`INPUT` or :data:`OUTPUT` (an issue about a folder) belongs to the panel holding
    that slot."""
    if dotted in SLOTS:
        return next((p for p in resolved if any(b.kind == SLOTS[dotted] for b in p.blocks)), None)
    parts = tuple(p for p in dotted.split(".") if p)
    if not parts:
        return None
    best, length = None, 0
    for panel in resolved:
        for leaf in panel.leaves:
            if len(leaf) > length and parts[:len(leaf)] == leaf:
                best, length = panel, len(leaf)
    if best is not None:
        return best
    return next((p for p in resolved for leaf in p.leaves if leaf[:len(parts)] == parts), None)


def value_at(data: Any, path: Sequence[str | int]) -> Any:
    """The value at ``path`` in a model or in its dump."""
    for key in path:
        data = data[key] if isinstance(data, dict | list) else getattr(data, key)
    return data


_EXPONENT = re.compile(r"e([+-])0*(\d)")


def number(value: float, digits: int = 4) -> str:
    """``value`` with ``digits`` significant digits: plain from 0.001 to 100,000 ("1000",
    not "1e+03"), else with a short exponent ("1e-5", not "1e-05")."""
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    if 1e-3 <= abs(value) < 1e5:
        return f"{value:.{max(digits, len(str(int(abs(value)))))}g}"
    return _EXPONENT.sub(lambda m: "e" + m[2] if m[1] == "+" else "e-" + m[2], f"{value:.{digits}g}")


def summary(model_cls: type[BaseModel], panel: ResolvedPanel, params: BaseModel) -> str:
    """The one-line summary of a collapsed panel, from its ``summary=True`` leaves: choices
    by label, true switches by name, numbers as "short value unit" (a run of one group and
    unit as "Group LV 9.538, RV 5.937 mmHg") and lists as "4 tubes"."""
    parts: list[str] = []
    run: tuple[str, str, list[str]] | None = None  # group, unit, "LV 9.538" items

    def flush() -> None:
        nonlocal run
        if run is not None:
            group, unit, items = run
            parts.append(f"{group} {', '.join(items)} {unit}".strip())
            run = None

    for leaf in panel.leaves:
        meta, ftype = leaf_meta(model_cls, leaf)
        if not meta.get("summary"):
            continue
        value = value_at(params, leaf)
        name = meta.get("short") or meta["title"]
        if ftype.kind == "rows" or isinstance(value, list):
            flush()
            one, several = nouns(meta)
            parts.append(f"{len(value)} {one if len(value) == 1 else several}")
        elif isinstance(value, bool):
            flush()
            if value:
                parts.append(name)
        elif meta.get("choices") or ftype.kind == "choice":
            flush()
            parts.append(str((meta.get("choices") or {}).get(value, value)))
        elif isinstance(value, int | float):
            group, unit = meta.get("group", ""), meta.get("unit", "")
            if group:
                if run is None or run[:2] != (group, unit):
                    flush()
                    run = (group, unit, [])
                run[2].append(f"{name} {number(value)}")
                continue
            flush()
            parts.append(f"{name} {number(value)} {unit}".strip())
        else:
            flush()
            parts.append(f"{name} {value}")
    flush()
    return " · ".join(parts)


def _shown(value: Any, meta: dict) -> str:
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, float):
        return number(value, 6)
    if isinstance(value, list):
        return ", ".join(_shown(v, {}) for v in value) or "none"
    choices = meta.get("choices") or {}
    return str(choices.get(value, value)) if value != "" else "empty"


def _change(label: str, before: Any, after: Any, meta: dict) -> str:
    unit = meta.get("unit", "")
    return f"{label}: {_shown(before, meta)} → {_shown(after, meta)}" + (f" {unit}" if unit else "")


def changes(model_cls: type[BaseModel], resolved: Sequence[ResolvedPanel], baseline: dict,
            params: dict) -> list[Change]:
    """What differs between two dumps of the parameters (``model_dump()``), leaf by leaf;
    a list whose length changed counts as one change."""
    out: list[Change] = []
    for panel in resolved:
        for leaf in panel.leaves:
            before, after = value_at(baseline, leaf), value_at(params, leaf)
            if before == after:
                continue
            meta, ftype = leaf_meta(model_cls, leaf)
            title = f"{panel.title} › {meta['title']}"
            if ftype.kind != "rows":
                out.append(Change(leaf, panel.key, _change(title, before, after, meta)))
            elif len(before) != len(after):
                one, several = nouns(meta)
                out.append(Change(leaf, panel.key,
                                  f"{title}: {len(before)} → {len(after)} "
                                  f"{one if len(after) == 1 else several}"))
            else:
                noun = nouns(meta)[0].capitalize()
                for i, (old, new) in enumerate(zip(before, after, strict=True)):
                    for name in old:
                        if old[name] != new.get(name):
                            column = leaf_meta(model_cls, (*leaf, i, name))[0]
                            out.append(Change((*leaf, i, name), panel.key, _change(
                                f"{panel.title} › {noun} {i + 1} › {column['title']}",
                                old[name], new.get(name), column)))
    return out
