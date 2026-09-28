"""Step 3 of mining: segment the dependency graph into candidate components.

One candidate is seeded per *public* module: a module that defines public
functions or classes and is not a helper. Each seed is grown to a component and
its boundary refined, deterministically:

  1. closure      every package module the seed needs at import time (the
                  module-level dependency closure, re-exports resolved) is
                  internalized. A component whose closure breaks the bound is
                  rejected rather than truncated: a truncated closure cannot
                  import.
  2. lazy helpers modules imported only inside functions are internalized when
                  they are private (underscore / utility names, or imported by
                  nothing outside the component) and still fit.
  3. tests        a test file that exercises the seed together with other
                  package modules pulls those modules (and their closure) in,
                  so the file can be carried whole, when the result still fits.
  4. dedup        seeds that end with identical module sets collapse into one.

Helpers do not seed their own candidate: underscore modules, utility-named
modules (``utils``, ``compat``, ``constants``, ...), package hubs (an
``__init__`` that only re-exports), and private helpers (imported by exactly
one other module). They still appear inside the components that need them. A
helper does seed when a test targets it directly or when a package
``__init__`` re-exports its public names (``from ._parser import loads``
makes ``_parser`` the implementation of public API).

Bounds (``max_files``, ``max_lines``) count the package modules of a component;
the generated ``__init__`` of a primitive is not counted.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from lego.mining.graph import RepoGraph

MAX_FILES = 6
MAX_LINES = 1500

HELPER_NAME = re.compile(
    r"^(_.*|utils?|util_.*|helpers?|common|compat|py\d?compat|constants?|consts|"
    r"exceptions?|errors?|types|typing|typedefs|globals|defaults|version|about|"
    r"config|settings|conf|__main__|main|cli|setup|conftest|log|logs)$", re.I)


@dataclass
class Component:
    seed: str
    modules: list                      # package module paths, seed first
    lines: int
    notes: list = field(default_factory=list)
    seeds: list = field(default_factory=list)   # all seeds that collapsed here


@dataclass
class Rejected:
    seed: str
    reason: str


def stem(path: str, pkg: str) -> str:
    base = os.path.basename(path)[:-3]
    if base == "__init__":
        d = os.path.dirname(path)
        return os.path.basename(d) if d else pkg
    return base


def seed_problem(g: RepoGraph, path: str) -> str | None:
    """Why ``path`` does not seed a candidate (None = it does)."""
    m = g.modules[path]
    if m.error:
        return f"parse error: {m.error[:120]}"
    defs = [s for s in m.symbols.values()
            if s.kind in ("function", "class") and not s.name.startswith("_")]
    if not defs:
        return ("package hub (re-exports only)" if g.is_package(path)
                else "no public functions or classes")
    if g.is_package(path):
        return None
    tested = any(path in t.targets for t in g.tests.values())
    # ``from ._parser import loads`` in a package __init__ makes the private
    # module the implementation of public API
    reexported = any(p == path and not n.startswith("_") and
                     any(g.is_package(u) for u in users)
                     for (p, n), users in g.uses.items())
    if HELPER_NAME.match(stem(path, g.pkg)) and not (tested or reexported):
        return "helper module"
    users = {u for u in g.importers(path) if not g.is_package(u)}
    if len(users) == 1 and not (tested or reexported):
        return f"private helper of {next(iter(users))}"
    return None


def _fits(g: RepoGraph, mods, max_files: int, max_lines: int) -> bool:
    return len(mods) <= max_files and g.lines(mods) <= max_lines


def _private(g: RepoGraph, path: str, mods: set) -> bool:
    if HELPER_NAME.match(stem(path, g.pkg)):
        return True
    users = {u for u, ds in list(g.deps.items()) + list(g.lazy.items())
             if path in ds and u != path}
    return users <= mods


def _lazy_helpers(g: RepoGraph, mods: set, max_files: int, max_lines: int,
                  notes: list) -> set:
    for m in sorted(mods):
        for d in sorted(g.lazy.get(m, ())):
            if d in mods or not _private(g, d, mods):
                continue
            ext = mods | g.closure([d])
            if _fits(g, ext, max_files, max_lines):
                mods = ext
                notes.append(f"internalized lazy helper {d}")
    return mods


def test_files(g: RepoGraph) -> list:
    """Test modules (not conftest) with a clean parse and resolved imports."""
    return [t for p, t in sorted(g.tests.items())
            if os.path.basename(p) != "conftest.py" and not t.error
            and not t.unresolved and t.targets]


def segment(g: RepoGraph, max_files: int = MAX_FILES,
            max_lines: int = MAX_LINES) -> tuple[list, list]:
    """-> (components, rejected). Deterministic for a given graph."""
    comps, rejected = [], []
    tests = test_files(g)
    for seed in sorted(g.modules):
        why = seed_problem(g, seed)
        if why:
            rejected.append(Rejected(seed, why))
            continue
        mods = g.closure([seed])
        if not _fits(g, mods, max_files, max_lines):
            rejected.append(Rejected(seed, f"closure exceeds bound: {len(mods)} "
                                           f"files, {g.lines(mods)} lines"))
            continue
        notes = []
        # 2. private lazy helpers
        mods = _lazy_helpers(g, mods, max_files, max_lines, notes)
        # 3. refine against tests that exercise the seed with other modules
        touching = sorted((t for t in tests if seed in t.targets
                           and not t.targets <= mods),
                          key=lambda t: (len(t.targets - mods), t.path))
        for t in touching:
            if t.targets <= mods:
                continue
            ext = mods | g.closure(t.targets - mods)
            if _fits(g, ext, max_files, max_lines):
                mods = ext
                notes.append(f"expanded for {t.path}")
                mods = _lazy_helpers(g, mods, max_files, max_lines, notes)
        order = [seed] + sorted(mods - {seed})
        comps.append(Component(seed, order, g.lines(mods), notes, [seed]))
    # 4. dedup identical module sets (first seed in path order wins)
    out, by_set = [], {}
    for c in comps:
        key = frozenset(c.modules)
        if key in by_set:
            by_set[key].seeds.append(c.seed)
            by_set[key].notes.append(f"merged seed {c.seed}")
            continue
        by_set[key] = c
        out.append(c)
    return out, rejected


def carried(g: RepoGraph, modules) -> list:
    """Test files every package reference of which lands inside ``modules``,
    ordered: files that exercise the seed first."""
    mods = set(modules)
    seed = modules[0] if modules else None
    keep = [t for t in test_files(g) if t.targets <= mods]
    return sorted(keep, key=lambda t: (seed not in t.targets, t.path))
