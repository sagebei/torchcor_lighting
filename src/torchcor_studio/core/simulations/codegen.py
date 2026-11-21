"""Building blocks for the generated ``run.py`` scripts.

Generated scripts double as documentation of what a run does, so they are laid out the
way a scientist would write one by hand: a docstring saying what the run is, the imports,
one block of settings with aligned comments, then numbered steps that each announce
themselves to the studio with ``studio.stage(...)``. This module renders Python literals
and assembles that layout; the simulation modules decide what goes into it.
"""

from __future__ import annotations

import json
import math
import textwrap
from collections.abc import Iterable, Sequence
from datetime import datetime

from torchcor_studio.core.simulations.base import ScriptContext

#: Target line length of generated code.
WIDTH = 96


# =============================================================================== literals
def lit(value) -> str:
    """Python source for a plain value: numbers, strings, bools, None, lists, tuples, dicts."""
    if value is None or isinstance(value, bool):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return 'float("nan")'
        if math.isinf(value):
            return 'float("inf")' if value > 0 else '-float("inf")'
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, tuple):
        inner = ", ".join(lit(v) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(lit(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{lit(k)}: {lit(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"no literal for {type(value).__name__}")


def regions(ids: Sequence[int]) -> str:
    """Region-id argument: ``None`` (TorchCor's "all regions") for an empty list."""
    return "None" if not ids else lit([int(i) for i in ids])


def path(value: str, windows: bool = False) -> str:
    """A ``Path(...)`` expression for a folder or file on the target machine."""
    if not value:
        raise ValueError("empty path")
    if value.startswith("~"):
        return f"Path({lit(value)}).expanduser()"
    if windows and "\\" in value and not value.endswith("\\") and '"' not in value:
        return f'Path(r"{value}")'
    return f"Path({lit(value)})"


def call(name: str, *args: str, prefix: str = "", width: int = WIDTH, **kwargs: str) -> str:
    """``prefix + name(args, key=value)`` on one line, or wrapped under the open bracket.

    Arguments are source text already (use :func:`lit` for values).
    """
    parts = [*args, *(f"{key}={value}" for key, value in kwargs.items())]
    one = f"{prefix}{name}({', '.join(parts)})"
    if len(one) <= width or not parts:
        return one
    pad = " " * (len(prefix) + len(name) + 1)
    lines, current = [], f"{prefix}{name}("
    for i, part in enumerate(parts):
        piece = part + ("," if i < len(parts) - 1 else ")")
        if current.endswith("("):
            current += piece
        elif len(current) + 1 + len(piece) <= width:
            current += " " + piece
        else:
            lines.append(current)
            current = pad + piece
    lines.append(current)
    return "\n".join(lines)


def identifier(text: str, fallback: str = "item") -> str:
    """A lowercase Python identifier made of the words of ``text``."""
    word = "".join(c if c.isalnum() else "_" for c in text.strip().lower())
    word = "_".join(filter(None, word.split("_")))
    if not word or word[0].isdigit():
        word = f"{fallback}_{word}" if word else fallback
    return word


# =============================================================================== layout
class Imports:
    """Import lines, grouped (stdlib, third party, local) and de-duplicated."""

    def __init__(self) -> None:
        self._groups: dict[str, list[str]] = {"stdlib": [], "third": [], "local": []}

    def add(self, line: str, group: str = "third") -> None:
        if line not in self._groups[group]:
            self._groups[group].append(line)

    def render(self) -> str:
        blocks = []
        for lines in self._groups.values():
            if lines:
                plain = sorted(x for x in lines if x.startswith("import "))
                froms = sorted(x for x in lines if x.startswith("from "))
                blocks.append("\n".join(_wrap_import(x) for x in plain + froms))
        return "\n\n".join(blocks)


def _wrap_import(line: str) -> str:
    """``from a import (b, c, ...)`` wrapped under the bracket when the line is too long."""
    if len(line) <= WIDTH or " import " not in line or not line.startswith("from "):
        return line
    head, names = line.split(" import ", 1)
    return call(f"{head} import ", *(n.strip() for n in names.split(",")))


class Settings:
    """``NAME = value  # comment`` rows rendered with the comments aligned."""

    def __init__(self) -> None:
        self._rows: list[tuple[str, str, str]] = []

    def add(self, name: str, expr: str, comment: str = "") -> None:
        self._rows.append((name, expr, comment))

    def render(self) -> str:
        if not self._rows:
            return ""
        heads = [f"{name} = {expr}" for name, expr, _ in self._rows]
        fitting = [len(h) for h in heads if len(h) <= 44] or [44]
        column = max(fitting) + 2
        lines = []
        for head, (_, _, comment) in zip(heads, self._rows, strict=True):
            if comment and len(head) <= 44 and column + 2 + len(comment) <= WIDTH:
                lines.append(f"{head.ljust(column)}# {comment}")
            elif comment:
                lines.append(f"# {comment}\n{head}")
            else:
                lines.append(head)
        return "\n".join(lines)


class Script:
    """A generated script: docstring, imports, settings, then numbered steps."""

    def __init__(self, docstring: str) -> None:
        self.docstring = docstring
        self.imports = Imports()
        self.settings = Settings()
        self._body: list[str] = []
        self._step = 0
        self.imports.add("from pathlib import Path", "stdlib")
        self.imports.add("import tcs_studio as studio", "local")

    # ---------------------------------------------------------------- body
    def step(self, title: str, stage_id: str | None = None, label: str | None = None,
             note: str = "") -> None:
        """Start a numbered step; with ``stage_id`` it also tells the studio about it."""
        self._step += 1
        head = f"# {self._step}. {title} "
        self._body.append("")
        self._body.append("")
        self._body.append(head + "-" * max(4, WIDTH - len(head)))
        if note:
            self.comment(note)
        if stage_id:
            self.code(f"studio.stage({lit(stage_id)}, {lit(label or title)})")

    def code(self, text: str, indent: str = "") -> None:
        """Append source: dedented, then indented by ``indent``; outer blank lines dropped."""
        block = textwrap.dedent(text).strip("\n")
        self._body.extend((indent + line).rstrip() for line in block.split("\n"))

    def definitions(self, text: str, comment: str = "") -> None:
        """Append top-level ``def``/``class`` blocks with two blank lines around them."""
        self._body += ["", ""]
        if comment:
            self.comment(comment)
        self.code(text)
        self._body += ["", ""]

    def comment(self, text: str) -> None:
        for line in textwrap.wrap(text, width=WIDTH - 2):
            self._body.append(f"# {line}")

    def blank(self) -> None:
        self._body.append("")

    # ---------------------------------------------------------------- output
    def render(self) -> str:
        parts = [f'"""{self.docstring}\n"""', self.imports.render()]
        settings = self.settings.render()
        if settings:
            head = "# Settings "
            parts.append(head + "-" * (WIDTH - len(head)) + "\n" + settings)
        body: list[str] = []
        for line in self._body:  # at most two blank lines in a row
            if line or len(body) < 2 or body[-1] or body[-2]:
                body.append(line)
        return "\n\n".join(parts) + "\n" + "\n".join(body).rstrip() + "\n"


def docstring(title: str, kind: str, ctx: ScriptContext, details: Iterable[str] = (),
              outputs: str = "") -> str:
    """The module docstring of a generated script."""
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    machine = ctx.machine_name or "this computer"
    lines = [f"{title} — TorchCor Lighting run", "", kind]
    lines += list(details)
    lines += ["", f"Generated by TorchCor Lighting on {when} for {machine} (device {ctx.device})."]
    if ctx.run_id:
        lines.append(f"Run id: {ctx.run_id}.")
    lines += ["", outputs or ("TorchCor's outputs go to results/ and the studio's visualisation "
                              "bundle to viz/."),
              "Needs torch, torchcor and numpy, plus tcs_studio.py (next to this file) for "
              "progress reporting.", "", "Re-run with:  python run.py"]
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=WIDTH - 4) or [""])
    return "\n".join(wrapped)


def device_setting(script: Script, ctx: ScriptContext) -> None:
    """``DEVICE = torch.device(...)``: the one device every tensor of the run lives on."""
    script.imports.add("import torch")
    comment = ("GPU chosen in the studio (PCI bus order)" if ctx.device.startswith("cuda")
               else "compute device chosen in the studio")
    script.settings.add("DEVICE", f"torch.device({lit(ctx.device)})", comment)


def output_settings(script: Script) -> None:
    """``HERE`` and ``RESULTS``: outputs live next to the script, wherever it is run."""
    script.settings.add("HERE", "Path(__file__).resolve().parent", "the run folder")
    script.settings.add("RESULTS", 'HERE / "results"', "TorchCor's own outputs")
