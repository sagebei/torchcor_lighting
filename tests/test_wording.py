"""Wording rules for everything that makes up the app: its source, tests, packaging and docs."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Other cardiac simulators are not named anywhere (written so that this file names none).
OTHER_SIMULATORS = re.compile(r"open[\W_]*carp|carp[\W_]*entry", re.IGNORECASE)
TEXT_SUFFIXES = {".py", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".json",
                 ".spec", ".sh", ".iss", ".desktop", ".html", ".css", ".js", ".svg"}


def _files() -> Iterator[Path]:
    for top in ("src", "tests", "packaging", "docs", ".github"):
        folder = ROOT / top
        if folder.is_dir():
            yield from (p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    yield from (p for p in (ROOT / "README.md", ROOT / "pyproject.toml") if p.is_file())


def test_no_other_simulator_is_named() -> None:
    found = []
    for path in _files():
        name = path.relative_to(ROOT).as_posix()
        if OTHER_SIMULATORS.search(name):
            found.append(name)
        if path.suffix.lower() in TEXT_SUFFIXES:
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if OTHER_SIMULATORS.search(line):
                    found.append(f"{name}:{number}: {line.strip()[:100]}")
    assert not found, "\n".join(found)
