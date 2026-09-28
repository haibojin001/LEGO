"""Step 1-2 of mining: program entities, relations and the dependency graph.

``build_graph(repo_dir)`` parses the repository's package (as located by
``core.detect_package``) with ``ast`` and returns a ``RepoGraph``:

  modules      one ``Module`` per package file (path relative to the package
               directory, e.g. ``sub/mod.py``), with its top-level symbols, the
               names each symbol references, and its import bindings
  deps         module-level import graph restricted to the package. Edges point
               at the module that *defines* an imported name: re-exports through
               ``__init__`` are followed, so ``from pkg import f`` depends on the
               file where ``f`` is written, not on the package hub
  lazy         the same for imports inside function bodies / optional-import
               blocks (needed only when that code path runs)
  uses         symbol-level references: ``(module, name) -> {importing modules}``
  external     top-level third-party imports per module (stdlib removed)
  tests        every test file of the repository, with the package modules and
               symbols it reaches and the test-local helpers it imports

Nothing here executes repository code.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass, field

from lego.harness import core

STDLIB = frozenset(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
MAX_MODULES = 2000
MAX_TEST_FILES = 3000
_SKIP_DIRS = {".git", "__pycache__", ".tox", ".venv", "venv", "node_modules",
              "build", "dist", ".eggs", "site-packages", ".mypy_cache"}


@dataclass
class Binding:
    """One name bound by an import statement."""
    local: str                  # name bound in the importing module
    target: str | None          # package module path it refers to (None = external)
    name: str | None            # imported attribute (None = the module itself)
    lazy: bool = False          # inside a function body / try / TYPE_CHECKING
    type_only: bool = False     # under ``if TYPE_CHECKING``
    obj: bool = False           # ``import pkg`` / ``import pkg.a as x``: a module
                                # object whose attribute uses are tracked instead


@dataclass
class Symbol:
    name: str
    kind: str                   # function | class | constant
    lineno: int
    end_lineno: int
    refs: set = field(default_factory=set)       # free names used in the body
    literal: str | None = None                   # source of a simple literal value
    doc: str = ""
    conditional: bool = False                    # defined inside a module-level if/try


@dataclass
class Module:
    path: str
    source: str
    lines: int
    doc: str = ""
    symbols: dict = field(default_factory=dict)  # name -> Symbol
    bindings: list = field(default_factory=list)  # [Binding]
    stars: list = field(default_factory=list)     # [(target path, lazy)]
    exports: list = field(default_factory=list)   # public names (``__all__`` first)
    external: set = field(default_factory=set)    # required third-party tops
    optional: set = field(default_factory=set)    # imported only in try/lazy
    attr_refs: set = field(default_factory=set)   # (target path, attr) via module objects
    error: str = ""

    def binding(self, local: str) -> Binding | None:
        for b in self.bindings:
            if b.local == local and not b.type_only:
                return b
        return None


@dataclass
class TestFile:
    path: str                   # relative to repo_dir
    source: str
    targets: set = field(default_factory=set)    # package modules reached
    names: set = field(default_factory=set)      # (module, symbol) reached
    helpers: set = field(default_factory=set)    # repo-relative helper files
    external: set = field(default_factory=set)
    unresolved: list = field(default_factory=list)
    error: str = ""


@dataclass
class RepoGraph:
    repo_dir: str
    src_prefix: str
    pkg: str
    pkg_dir: str
    single: bool                                  # a one-file distribution (pkg.py)
    modules: dict = field(default_factory=dict)   # path -> Module
    deps: dict = field(default_factory=dict)      # path -> {path} (module level)
    lazy: dict = field(default_factory=dict)      # path -> {path} (lazy)
    uses: dict = field(default_factory=dict)      # (path, name) -> {path}
    tests: dict = field(default_factory=dict)     # repo-rel path -> TestFile
    aux: dict = field(default_factory=dict)       # analysed test-local helpers
    test_root: str = ""                           # repo-relative test dir/file

    # ---------------------------------------------------------------- names
    def parts(self, path: str) -> list[str]:
        """Dotted parts of a module *below* the package root."""
        if self.single:
            return []
        p = path[:-3].split("/")
        return p[:-1] if p[-1] == "__init__" else p

    def impl_parts(self, path: str) -> list[str]:
        """Dotted parts of a module below the root of a primitive's impl/."""
        p = path[:-3].split("/")
        return p[:-1] if p[-1] == "__init__" else p

    def dotted(self, path: str) -> str:
        return ".".join([self.pkg] + self.parts(path))

    def lookup(self, parts: list[str]) -> str | None:
        """Package module for dotted parts below the root, or None."""
        if self.single:
            return f"{self.pkg}.py" if not parts else None
        base = "/".join(parts)
        for cand in ((base + ".py") if base else None,
                     (base + "/__init__.py") if base else "__init__.py"):
            if cand and cand in self.modules:
                return cand
        return None

    def is_package(self, path: str) -> bool:
        return path == "__init__.py" or path.endswith("/__init__.py")

    def lines(self, paths) -> int:
        return sum(self.modules[p].lines for p in paths if p in self.modules)

    # ---------------------------------------------------------- resolution
    def resolve(self, path: str, name: str, _depth: int = 0):
        """Follow re-exports: ``(path, name)`` -> ``(defining path, name)``.
        Returns ``("module", sub_path)`` when ``name`` is a submodule, and None
        when the name cannot be found in the package."""
        m = self.modules.get(path)
        if m is None or _depth > 12:
            return None
        if self.is_package(path):
            sub = self.lookup(self.parts(path) + [name])
            if sub and sub != path and name not in m.symbols and not m.binding(name):
                return ("module", sub)
        if name in m.symbols:
            return (path, name)
        b = m.binding(name)
        if b is not None:
            if b.target is None:
                return None
            if b.name is None:
                return ("module", b.target)
            return self.resolve(b.target, b.name, _depth + 1)
        for tgt, _lazy in m.stars:
            t = self.modules.get(tgt)
            if t and (name in t.exports or name in t.symbols or t.binding(name)):
                r = self.resolve(tgt, name, _depth + 1)
                if r:
                    return r
        return None

    def inlined(self, res) -> bool:
        """Is ``res`` a simple literal defined in a package ``__init__``?
        Those (``__version__`` and friends) are copied into the primitive's
        generated ``__init__`` rather than pulling in the package hub."""
        if not res or res[0] == "module" or not self.is_package(res[0]):
            return False
        sym = self.modules[res[0]].symbols.get(res[1])
        return sym is not None and sym.kind == "constant" and sym.literal is not None

    def closure(self, paths, lazy: bool = False) -> set:
        out, todo = set(), list(paths)
        while todo:
            p = todo.pop()
            if p in out or p not in self.modules:
                continue
            out.add(p)
            todo += sorted(self.deps.get(p, ()))
            if lazy:
                todo += sorted(self.lazy.get(p, ()))
        return out

    def importers(self, path: str) -> set:
        return {m for m, ds in self.deps.items() if path in ds and m != path}

    def analysis(self, rel: str) -> "TestFile | None":
        """Analysis of a test file or test-local helper (cached)."""
        if rel in self.tests:
            return self.tests[rel]
        if rel not in self.aux:
            try:
                src = open(os.path.join(self.repo_dir, rel), errors="ignore").read()
            except OSError:
                return None
            self.aux[rel] = analyze_test(self, rel, src)
        return self.aux[rel]

    def definers(self, name: str) -> list[str]:
        """Modules that define a top-level ``name`` (for NameError repair)."""
        return sorted(p for p, m in self.modules.items() if name in m.symbols)


# ------------------------------------------------------------------ parsing
def _literal(node) -> str | None:
    try:
        ast.literal_eval(node)
    except Exception:
        return None
    text = ast.unparse(node)
    return text if len(text) <= 200 else None


def bound_names(node) -> set:
    """Names bound *inside* a definition (arguments, assignments, loop and
    ``with`` targets, nested defs, handler and import aliases)."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and n is not node:
            out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name.split(".")[0]) for a in n.names)
    return out


class Refs(ast.NodeVisitor):
    """Names and ``name.attr`` chains used inside one definition."""

    def __init__(self):
        self.names, self.chains = set(), set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)

    def visit_Attribute(self, node):
        chain, cur = [], node
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            self.chains.add((cur.id, tuple(reversed(chain))))
        self.generic_visit(node)


def _type_checking(node) -> bool:
    t = node.test
    return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
        isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING")


def iter_imports(tree):
    """(node, lazy, type_only) for every import statement in a module."""
    def walk(nodes, lazy, tonly):
        for n in nodes:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                yield n, lazy, tonly
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                yield from walk(ast.iter_child_nodes(n), True, tonly)
            elif isinstance(n, ast.If):
                tc = _type_checking(n)
                yield from walk(n.body, lazy or tc, tonly or tc)
                yield from walk(n.orelse, lazy, tonly)
            elif isinstance(n, ast.Try) or type(n).__name__ == "TryStar":
                yield from walk(n.body, True, tonly)
                for h in n.handlers:
                    yield from walk(h.body, True, tonly)
                yield from walk(n.orelse, True, tonly)
                yield from walk(n.finalbody, lazy, tonly)
            else:
                yield from walk(ast.iter_child_nodes(n), lazy, tonly)
    yield from walk(tree.body, False, False)


def _top_bindings(tree):
    """(statement, conditional) for top-level statements, descending into
    module-level if/try blocks. ``if TYPE_CHECKING`` bodies define nothing at
    run time and are skipped."""
    def walk(nodes, cond):
        for n in nodes:
            if isinstance(n, ast.If):
                if not _type_checking(n):
                    yield from walk(n.body, True)
                yield from walk(n.orelse, True)
            elif isinstance(n, ast.Try) or type(n).__name__ == "TryStar":
                yield from walk(n.body, True)
                for h in n.handlers:
                    yield from walk(h.body, True)
                yield from walk(n.orelse, True)
                yield from walk(n.finalbody, cond)
            else:
                yield n, cond
    yield from walk(tree.body, False)


def _all_names(tree) -> list[str] | None:
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in tgts):
                try:
                    return [str(x) for x in ast.literal_eval(n.value)]
                except Exception:
                    return None
    return None


def import_base(g: RepoGraph, path: str, node: ast.ImportFrom):
    """Dotted parts (below the root) an ImportFrom refers to, or None when it
    leaves the package."""
    if node.level:
        here = path[:-3].split("/")[:-1] if not g.single else []
        if node.level - 1 > len(here):
            return None
        base = here[: len(here) - (node.level - 1)]
        return base + (node.module.split(".") if node.module else [])
    if node.module and (node.module == g.pkg or node.module.startswith(g.pkg + ".")):
        return node.module.split(".")[1:]
    return None


def _parse_module(g: RepoGraph, path: str, source: str) -> Module:
    m = Module(path=path, source=source, lines=source.count("\n") + 1)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as e:
        m.error = f"{type(e).__name__}: {e}"
        return m
    m.doc = ast.get_docstring(tree) or ""
    for n, cond in _top_bindings(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            r = Refs()
            r.visit(n)
            kind = "class" if isinstance(n, ast.ClassDef) else "function"
            m.symbols.setdefault(n.name, Symbol(
                n.name, kind, n.lineno, n.end_lineno or n.lineno, r.names,
                doc=ast.get_docstring(n) or "", conditional=cond))
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            value = n.value
            for t in tgts:
                for nm in ([t] if isinstance(t, ast.Name) else
                           [e for e in getattr(t, "elts", []) if isinstance(e, ast.Name)]):
                    r = Refs()
                    if value is not None:
                        r.visit(value)
                    m.symbols.setdefault(nm.id, Symbol(
                        nm.id, "constant", n.lineno, n.end_lineno or n.lineno,
                        r.names, literal=_literal(value) if value is not None
                        and isinstance(t, ast.Name) else None, conditional=cond))
    for node, lazy, tonly in iter_imports(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top == g.pkg:
                    tgt = g.lookup(a.name.split(".")[1:])
                    root = g.lookup([])
                    if a.asname:
                        m.bindings.append(Binding(a.asname, tgt, None, lazy, tonly,
                                                  obj=tgt == root))
                    else:
                        m.bindings.append(Binding(g.pkg, root, None, lazy, tonly,
                                                  obj=True))
                        if tgt and tgt != root:
                            m.bindings.append(Binding("", tgt, None, lazy, tonly))
                elif not tonly:
                    _external(m, top, lazy)
            continue
        base = import_base(g, path, node)
        if base is None:
            if not node.level and node.module and not tonly:
                _external(m, node.module.split(".")[0], lazy)
            continue
        tgt = g.lookup(base)
        for a in node.names:
            if a.name == "*":
                if tgt:
                    m.stars.append((tgt, lazy))
                continue
            sub = g.lookup(base + [a.name])
            if sub and (tgt is None or g.is_package(tgt)):
                m.bindings.append(Binding(a.asname or a.name, sub, None, lazy, tonly))
            else:
                m.bindings.append(Binding(a.asname or a.name, tgt, a.name, lazy, tonly))
    r = Refs()
    r.visit(tree)
    for root, chain in r.chains:
        b = m.binding(root)
        if b is not None and b.target and b.name is None and chain:
            m.attr_refs.add((b.target, chain))
    names = _all_names(tree)
    if names is None:
        names = [x for x in m.symbols if not x.startswith("_")]
        names += [b.local for b in m.bindings
                  if b.target and b.name and not b.local.startswith("_")
                  and not b.lazy and g.is_package(path)]
    m.exports = list(dict.fromkeys(names))
    m.optional -= m.external
    return m


def _external(m: Module, top: str, lazy: bool):
    if not top or top in STDLIB:
        return
    (m.optional if lazy else m.external).add(top)


def _chain_target(g: RepoGraph, tgt: str, chain) -> tuple | None:
    """Resolve ``mod.a.b`` where ``mod`` is a package module object. An
    attribute that the module binds from outside the package (``nap.time``)
    resolves to the module itself."""
    cur = tgt
    for attr in chain:
        sub = g.lookup(g.parts(cur) + [attr]) if g.is_package(cur) else None
        if sub and attr not in g.modules[cur].symbols:
            cur = sub
            continue
        r = g.resolve(cur, attr)
        if r is None and g.modules[cur].binding(attr) is not None:
            return (cur, attr)
        return r
    return ("module", cur)


def _dest(g: RepoGraph, path: str, res) -> set:
    """Module(s) a resolved reference makes ``path`` depend on."""
    if res[0] == "module":
        return {res[1]}
    if g.inlined(res):
        return set()                # literal in a package __init__: inlined
    g.uses.setdefault(res, set()).add(path)
    return {res[0]}


def _edges(g: RepoGraph):
    for path, m in g.modules.items():
        hard, soft = set(), set()
        for b in m.bindings:
            if b.target is None or b.type_only or b.obj:
                continue
            if b.name is None:
                dest = {b.target}
            else:
                r = g.resolve(b.target, b.name)
                dest = {b.target} if r is None else _dest(g, path, r)
            (soft if b.lazy else hard).update(dest)
        for tgt, lazy in m.stars:
            (soft if lazy else hard).add(tgt)
        for tgt, chain in sorted(m.attr_refs):
            r = _chain_target(g, tgt, chain)
            if r is None:
                continue
            lazy = all(b.lazy for b in m.bindings if b.target == tgt)
            (soft if lazy else hard).update(_dest(g, path, r))
        hard.discard(path)
        soft.discard(path)
        g.deps[path] = hard
        g.lazy[path] = soft - hard


# -------------------------------------------------------------------- tests
def _test_files(g: RepoGraph) -> list[str]:
    pats = core.pytest_python_files(g.repo_dir)
    pkg_abs = os.path.realpath(g.pkg_dir)
    out = []
    for root, dirs, files in os.walk(g.repo_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS
                         and not d.startswith(".") and d not in ("docs", "doc"))
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, g.repo_dir)
            in_pkg = os.path.realpath(full).startswith(pkg_abs + os.sep)
            if in_pkg and os.path.relpath(full, g.pkg_dir) in g.modules:
                continue
            if core.is_test_file(f, pats) or f == "conftest.py":
                out.append(rel)
        if len(out) >= MAX_TEST_FILES:
            break
    return out


def local_helper(g: RepoGraph, test_rel: str, parts: list[str], level: int):
    """Repo-relative path of a test-local helper module, or None."""
    here = os.path.dirname(test_rel)
    cands = []
    if level:
        base = here
        for _ in range(level - 1):
            base = os.path.dirname(base)
        cands.append(os.path.join(base, *parts))
    else:
        cands.append(os.path.join(here, *parts))
        cands.append(os.path.join(*parts))
    for c in cands:
        for f in (c + ".py", os.path.join(c, "__init__.py")):
            f = os.path.normpath(f)
            if os.path.isfile(os.path.join(g.repo_dir, f)):
                return f
    return None


def test_import_base(g: RepoGraph, rel: str, node: ast.ImportFrom):
    """Package parts an ImportFrom in a test file refers to, or None. Handles
    absolute ``from pkg.x import`` and relative imports from tests that live
    inside the package directory (``pkg/tests/test_x.py``)."""
    if not node.level:
        if node.module and (node.module == g.pkg or
                            node.module.startswith(g.pkg + ".")):
            return node.module.split(".")[1:]
        return None
    if g.single:
        return None
    full = os.path.realpath(os.path.join(g.repo_dir, rel))
    root = os.path.realpath(g.pkg_dir)
    if not full.startswith(root + os.sep):
        return None
    here = os.path.relpath(os.path.dirname(full), root)
    here = [] if here == "." else here.split(os.sep)
    if node.level - 1 > len(here):
        return None
    base = here[: len(here) - (node.level - 1)] + (
        node.module.split(".") if node.module else [])
    if g.lookup(base) or any(g.lookup(base + [a.name]) for a in node.names):
        return base
    return None


def analyze_test(g: RepoGraph, rel: str, source: str) -> TestFile:
    t = TestFile(path=rel, source=source)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as e:
        t.error = f"{type(e).__name__}: {e}"
        return t
    root = g.lookup([])
    bound = {}                                   # local name -> module path

    def reach(res):
        if res is None:
            return False
        if res[0] == "module":
            t.targets.add(res[1])
        else:
            t.names.add(res)
            if not g.inlined(res):
                t.targets.add(res[0])
        return True

    for node, _lazy, tonly in iter_imports(tree):
        if tonly:
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top == g.pkg:
                    tgt = g.lookup(a.name.split(".")[1:])
                    if tgt is None:
                        t.unresolved.append(a.name)
                        continue
                    if tgt != root:
                        t.targets.add(tgt)
                    bound[a.asname or g.pkg] = tgt if a.asname else root
                else:
                    h = local_helper(g, rel, a.name.split("."), 0)
                    if h:
                        t.helpers.add(h)
                    elif top not in STDLIB:
                        t.external.add(top)
            continue
        base = test_import_base(g, rel, node)
        if base is not None:
            tgt = g.lookup(base)
            label = node.module or "." * node.level
            if tgt is None:
                t.unresolved.append(label)
                continue
            for a in node.names:
                if a.name == "*":
                    t.targets.add(tgt)
                    continue
                sub = g.lookup(base + [a.name])
                if sub and g.is_package(tgt) and a.name not in g.modules[tgt].symbols:
                    t.targets.add(sub)
                    bound[a.asname or a.name] = sub
                elif not reach(g.resolve(tgt, a.name)):
                    t.unresolved.append(f"{label}.{a.name}")
            continue
        parts = node.module.split(".") if node.module else []
        if node.level and not parts:           # from . import sibling
            for a in node.names:
                h2 = local_helper(g, rel, [a.name], node.level)
                if h2:
                    t.helpers.add(h2)
                else:
                    t.unresolved.append(f"{'.' * node.level}{a.name}")
            continue
        h = local_helper(g, rel, parts, node.level)
        if h:
            t.helpers.add(h)
        elif not node.level and parts and parts[0] not in STDLIB:
            t.external.add(parts[0])
    r = Refs()
    r.visit(tree)
    for name, chain in r.chains:
        tgt = bound.get(name)
        if tgt and chain:
            res = _chain_target(g, tgt, chain)
            if not reach(res):
                t.unresolved.append(f"{name}.{'.'.join(chain)}")
    return t


# -------------------------------------------------------------------- build
def locate_package(repo_dir: str) -> tuple:
    """(src_prefix, package). The package the project declares for itself wins
    when it exists on disk (``src/<name>`` included); otherwise the harness's
    detection (``core.detect_package``)."""
    decl = (core.declared_package(repo_dir) or "").replace("-", "_").lower()
    if decl:
        for prefix in (".", "src", "lib", "python"):
            root = repo_dir if prefix == "." else os.path.join(repo_dir, prefix)
            try:
                entries = sorted(os.listdir(root))
            except OSError:
                continue
            for e in entries:
                full = os.path.join(root, e)
                if e.lower() == decl and os.path.isfile(
                        os.path.join(full, "__init__.py")):
                    return prefix, e
                if e.lower() == decl + ".py" and os.path.isfile(full):
                    return prefix, e[:-3]
    return core.detect_package(repo_dir)


def build_graph(repo_dir: str, src_prefix: str | None = None,
                pkg: str | None = None, with_tests: bool = True) -> RepoGraph:
    if pkg is None:
        src_prefix, pkg = locate_package(repo_dir)
        if not pkg:
            raise ValueError("no package found")
    src_prefix = src_prefix or "."
    src_root = repo_dir if src_prefix == "." else os.path.join(repo_dir, src_prefix)
    pkg_dir = os.path.join(src_root, pkg)
    single = False
    if not os.path.isdir(pkg_dir) and os.path.isfile(pkg_dir + ".py"):
        pkg_dir, single = src_root, True
    g = RepoGraph(repo_dir=repo_dir, src_prefix=src_prefix, pkg=pkg,
                  pkg_dir=pkg_dir, single=single)
    paths = ([f"{pkg}.py"] if single else
             [p.replace(os.sep, "/") for p in core.package_modules(pkg_dir)])
    sources = {}
    for p in paths[:MAX_MODULES]:
        try:
            sources[p] = open(os.path.join(pkg_dir, p), errors="ignore").read()
        except OSError:
            continue
    # lookup() needs the full module set before any import is resolved
    g.modules = {p: Module(path=p, source=s, lines=s.count("\n") + 1)
                 for p, s in sources.items()}
    for p, src in sources.items():
        g.modules[p] = _parse_module(g, p, src)
    _edges(g)
    if with_tests:
        tp = core.detect_tests(repo_dir, os.path.relpath(pkg_dir, repo_dir))
        g.test_root = tp or ""
        for rel in _test_files(g):
            try:
                src = open(os.path.join(repo_dir, rel), errors="ignore").read()
            except OSError:
                continue
            g.tests[rel] = analyze_test(g, rel, src)
    return g
