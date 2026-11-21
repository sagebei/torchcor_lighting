"""Map pydantic field annotations to editor kinds."""

from __future__ import annotations

import enum
import types
import typing
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel


@dataclass(frozen=True)
class FieldType:
    kind: str  # float int bool str choice section rows int_list float_list str_list vec
    optional: bool = False
    inner: Any = None  # nested model class, enum class, element type
    choices: tuple[Any, ...] = ()
    length: int = 0  # for fixed tuples (vec)


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        optional = len(args) != len(get_args(annotation))
        if len(args) == 1:
            return args[0], optional
        return args[0], optional  # first alternative wins (e.g. float | str)
    return annotation, False


def resolve(annotation: Any) -> FieldType:
    ann, optional = _strip_optional(annotation)
    origin = get_origin(ann)
    if origin is Literal:
        return FieldType("choice", optional, choices=tuple(get_args(ann)))
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        return FieldType("choice", optional, inner=ann, choices=tuple(m.value for m in ann))
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return FieldType("section", optional, inner=ann)
    if origin in (list, tuple, set, frozenset) or ann in (list, tuple):
        args = get_args(ann)
        elem = args[0] if args else Any
        if origin is tuple and args and args[-1] is not Ellipsis and all(a in (float, int) for a in args):
            return FieldType("vec", optional, inner=args[0], length=len(args))
        elem, _ = _strip_optional(elem)
        if isinstance(elem, type) and issubclass(elem, BaseModel):
            return FieldType("rows", optional, inner=elem)
        if elem is int:
            return FieldType("int_list", optional, inner=int)
        if elem is float:
            return FieldType("float_list", optional, inner=float)
        return FieldType("str_list", optional, inner=str)
    if ann is bool:
        return FieldType("bool", optional)
    if ann is int:
        return FieldType("int", optional)
    if ann is float:
        return FieldType("float", optional)
    return FieldType("str", optional)
