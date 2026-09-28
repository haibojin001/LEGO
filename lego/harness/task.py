"""LEGO-REPO task definitions.

A task is one line of ``benchmark/tasks.jsonl`` plus its frozen directory
``benchmark/tasks/<name>/``:

    task.json              pinned commit, package prefix, test path, environment
    task.md                natural-language task statement (API surface from tests)
    tests/                 the native suite (input)
    original/              the original sources (answer; never shown to a system)
    expected/ceiling.txt   node ids passed by the original sources  -> c(T)
    expected/floor.txt     node ids passed by an empty package      -> f(T)
    env/requirements.lock  the dependency set resolved at freeze time

``tasks.jsonl`` is written by ``benchmark/build_index.py``; the split files in
``benchmark/splits/`` are plain lists of task names.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

BENCH_DIR = os.environ.get("LEGO_BENCH", "benchmark")
BAND_CUTOFFS = (4.0, 5.0, 6.0, 7.0)


def difficulty_score(n_modules: int, ceiling: int) -> float:
    """d(T) = 2 log10 m(T) + log10 c(T)  (Eq. 6)."""
    return 2 * math.log10(max(n_modules, 1)) + math.log10(max(ceiling, 1))


def band_of(n_modules: int, ceiling: int, cutoffs=BAND_CUTOFFS) -> int:
    d = difficulty_score(n_modules, ceiling)
    return 1 + sum(d >= t for t in cutoffs)


@dataclass
class Task:
    name: str
    clone: str
    commit: str
    package: str
    src_prefix: str = "."
    project_dir: str = "."
    tests: str = "tests"
    domain: str = ""
    track: str = ""
    stars: int = 0
    n_modules: int = 0
    ceiling: int = 0
    floor: int = 0
    band: int = 0
    note: str = ""
    kloc: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def dir(self) -> str:
        return os.path.join(BENCH_DIR, "tasks", self.name)

    def ceiling_ids(self) -> set[str]:
        return _read_ids(os.path.join(self.dir, "expected", "ceiling.txt"))

    def floor_ids(self) -> set[str]:
        return _read_ids(os.path.join(self.dir, "expected", "floor.txt"))

    def statement(self) -> str:
        p = os.path.join(self.dir, "task.md")
        if not os.path.exists(p):
            return ""
        # The human-readable card also contains the grading formula, frozen
        # ceiling/floor counts, answer location and provenance. Those sections
        # are for auditing, never for a constructor or coverage-model prompt.
        sections = re.split(r"(?=^## )", open(p).read(), flags=re.M)
        public = [sections[0].strip()]
        for section in sections[1:]:
            heading = section.splitlines()[0].removeprefix("## ").strip().lower()
            if heading in ("what to build", "api surface",
                           "api surface the tests require"):
                public.append(section.strip())
        return "\n\n".join(part for part in public if part)


def _read_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    return {l.strip() for l in open(path) if l.strip()}


_FIELDS = set(Task.__dataclass_fields__) - {"extra"}


def load_index(path: str | None = None) -> dict[str, Task]:
    path = path or os.path.join(BENCH_DIR, "tasks.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"task index missing: {path}; run benchmark/build_index.py")
    out = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            known = {k: v for k, v in row.items() if k in _FIELDS}
            rest = {k: v for k, v in row.items() if k not in _FIELDS}
            t = Task(**known, extra=rest)
            out[t.name] = t
    return out


def load_split(name_or_path: str) -> list[str]:
    """A split is a text file of task names, one per line (``#`` comments ok)."""
    p = name_or_path
    if not os.path.exists(p):
        p = os.path.join(BENCH_DIR, "splits", name_or_path)
        if not p.endswith(".txt"):
            p += ".txt"
    if not os.path.isfile(p):
        raise SystemExit(
            f"split missing: {p}; run benchmark/build_index.py and set "
            "LEGO_SPLIT=lego_repo_available for the currently frozen tasks")
    return [l.split("#")[0].strip() for l in open(p)
            if l.split("#")[0].strip()]


def select(index: dict[str, Task], split: str | None = None,
           only: list[str] | None = None) -> list[Task]:
    names = only or (load_split(split) if split else sorted(index))
    missing = [n for n in names if n not in index]
    if missing:
        raise SystemExit(f"{len(missing)} task(s) not in index: {missing[:5]}")
    return [index[n] for n in names]


def band_interleave(tasks: list[Task]) -> list[Task]:
    """Order tasks D1,D2,D3,D4,D5,D1,... so that any prefix of a sweep (and any
    strided shard of it) keeps the band mix of the whole split."""
    by = {}
    for t in sorted(tasks, key=lambda t: (t.band, t.n_modules, t.name)):
        by.setdefault(t.band, []).append(t)
    out, i = [], 0
    while any(by.values()):
        for b in sorted(by):
            if i < len(by[b]):
                out.append(by[b][i])
        i += 1
        if all(i >= len(v) for v in by.values()):
            break
    return out
