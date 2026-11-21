"""Declarative parameter fields.

Simulation parameters are pydantic models whose fields carry UI metadata, so a single
definition drives validation, defaults, (de)serialisation, the generated form and the
generated script. Always declare fields with :func:`param`::

    class Timing(Section):
        duration: float = param(500.0, "Duration", unit="ms", min=1, step=50,
                                help="Simulated time.")

Metadata keys (stored in ``json_schema_extra``) understood by the form builder:

``unit``         unit, drawn faint inside number boxes and after table column titles
``help``         one-sentence explanation (tooltip and caption)
``min``/``max``  numeric bounds (also enforced by pydantic via ``ge``/``le``)
``step``         spin-box step; ``decimals`` for float display; ``log`` for log-scale steps
``widget``       override: "spin", "slider", "segmented", "combo", "switch", "text",
                 "mono" (monospace text), "path", "dir", "file", "vtx", "regions", "code",
                 "tags"
``choices``      {value: label} for str fields rendered as segmented/combo controls
``advanced``     hidden until the panel's Advanced button is on
``visible_if``   {"field": value | [values]} — sibling fields that must match for the
                 field to be shown; "../field" looks one section up (from a row of a
                 ``list[Section]``, that is the section owning the list), and the key may
                 go on down with dots ("../loading.mode")
``placeholder``  hint text for empty text inputs
``group``        consecutive fields sharing a group form a tile under that eyebrow; table
                 columns sharing one get a column-group header
``short``        short label: group members, table columns and collapsed-panel summaries
``span``         grid columns the field takes (0: the whole row) instead of its type's
``summary``      shown in the one-line summary of its panel while collapsed
``fixed``        on a ``list[Section]``: a table whose rows are fixed (no add or remove)
``item``/``items`` the noun of one row and of several ("stimulus", "stimuli"); ``items``
                 defaults to ``item`` + "s"
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    """Base class for every parameter (sub)model."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True, use_enum_values=True)


def param(
    default: Any = ...,
    title: str = "",
    *,
    unit: str = "",
    help: str = "",  # noqa: A002 - reads naturally at call sites
    min: float | None = None,  # noqa: A002
    max: float | None = None,  # noqa: A002
    step: float | None = None,
    decimals: int | None = None,
    log: bool = False,
    widget: str | None = None,
    choices: dict[str, str] | None = None,
    advanced: bool = False,
    visible_if: dict[str, Any] | None = None,
    placeholder: str = "",
    group: str = "",
    short: str = "",
    span: int | None = None,
    summary: bool = False,
    fixed: bool = False,
    item: str = "",
    items: str = "",
    default_factory: Any = None,
) -> Any:
    """A pydantic ``Field`` carrying the studio's UI metadata."""
    extra: dict[str, Any] = {
        k: v
        for k, v in {
            "unit": unit,
            "help": help,
            "min": min,
            "max": max,
            "step": step,
            "decimals": decimals,
            "log": log or None,
            "widget": widget,
            "choices": choices,
            "advanced": advanced or None,
            "visible_if": visible_if,
            "placeholder": placeholder,
            "group": group,
            "short": short,
            "span": span,
            "summary": summary or None,
            "fixed": fixed or None,
            "item": item,
            "items": items,
        }.items()
        if v not in (None, "", {})
    }
    kwargs: dict[str, Any] = {"title": title, "description": help, "json_schema_extra": extra}
    if min is not None:
        kwargs["ge"] = min
    if max is not None:
        kwargs["le"] = max
    if default_factory is not None:
        return Field(default_factory=default_factory, **kwargs)
    return Field(default, **kwargs)


def field_meta(model: type[BaseModel] | BaseModel, name: str) -> dict[str, Any]:
    """UI metadata of one field (empty dict when none)."""
    cls = model if isinstance(model, type) else type(model)
    info = cls.model_fields[name]
    extra = info.json_schema_extra
    meta = dict(extra) if isinstance(extra, dict) else {}
    meta.setdefault("title", info.title or name.replace("_", " ").capitalize())
    meta.setdefault("help", info.description or "")
    if not info.is_required() and info.default_factory is None:
        meta.setdefault("default", info.default)
    return meta
