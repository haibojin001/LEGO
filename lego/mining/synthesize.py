"""Step 6 of mining: synthesize a Code Primitive and validate it in isolation.

Layout. ``impl/`` *is* the primitive's package: the component's modules keep
their paths relative to the source package directory, and every
package-internal import is regenerated as a RELATIVE import to the module that
defines the imported name (re-exports followed). ``impl/`` therefore imports
under any package name, which is what the Import+Call control relies on when it
vendors a primitive as ``_lego_vendor.p_<pid>`` (``lego.pipeline.construct``).
Package ``__init__`` files that are not part of the component are generated:
the root one re-exports the component's public names and carries the literal
constants (``__version__`` ...) of the original ``__init__``; nested ones are
empty apart from such literals. Carried tests import the primitive as
``p_<pid>`` (the same name it is vendored under), recorded as
``interface.package``.

Validation. A candidate is written to a scratch directory as ``p_<pid>/`` +
``tests/`` and run with only that directory on ``PYTHONPATH``, in a venv that
holds D_i (and pytest) but never the source package. The loop:

  import check      import the package and every component module
  carried tests     ``pytest tests`` with a small plugin that records per-test
                    outcomes and whether the *source* package got imported
  refinement        a failure naming a package-internal module or symbol that
                    the component lacks (ModuleNotFoundError, "cannot import
                    name", NameError, missing module attribute) adds the
                    defining module and its closure and retries, up to
                    ``refine`` times and within the component bound; a missing
                    third-party import is installed and retried
  pruning           otherwise uncollectable files and failing test functions
                    are removed from V_i and the rest re-run

A candidate is admitted when every retained test passes and at least one does
(``mode="tests"``). ``mode="import"`` is reserved for exploratory recovery and
does not produce a paper-eligible primitive. ``synth_tests`` (optional) asks a
model for tests when none could be carried; such primitives are marked
``tests_synthesized``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field

from lego.harness import core
from lego.library import primitive as primitive_io
from lego.mining import collect
from lego.mining.graph import STDLIB, RepoGraph, import_base
from lego.mining.segment import MAX_FILES, MAX_LINES, Component
from lego.pipeline.construct import VENDOR_PKG, _safe

IMPORT_TIMEOUT = int(os.environ.get("LEGO_MINE_IMPORT_TIMEOUT", 120))
TEST_TIMEOUT = int(os.environ.get("LEGO_MINE_TEST_TIMEOUT", 300))
REFINE_ROUNDS = 3

_PLUGIN_HINTS = ((re.compile(r"mark\.asyncio"), "pytest_asyncio"),
                 (re.compile(r"\bmocker\b"), "pytest_mock"))


def package_name(pid: str) -> str:
    """Import name of a primitive's package. Equal to the sub-package the
    Import+Call control vendors it as (``_lego_vendor.p_<pid>``)."""
    return "p_" + _safe(pid)


def vendored_name(pid: str) -> str:
    return f"{VENDOR_PKG}.{package_name(pid)}"


# ------------------------------------------------------------------ impl
def _rel(here: list, target: list) -> str:
    """Relative module reference from package ``here`` to dotted ``target``."""
    common = 0
    while (common < len(here) and common < len(target)
           and here[common] == target[common]):
        common += 1
    return "." * (len(here) - common + 1) + ".".join(target[common:])


def impl_rewriter(g: RepoGraph, path: str):
    """Rewrite package-internal imports of module ``path`` to relative ones."""
    here = [] if g.single else path.split("/")[:-1]
    depth = len(g.impl_parts(path))
    root = g.lookup([])
    root_expr = f'_lego_sys.modules[__name__.rsplit(".", {depth})[0]]'
    counter = [0]

    def mod_ref(p):
        parts = g.impl_parts(p)
        return parts[:-1], parts[-1]

    def fn(node):
        out = []
        if isinstance(node, ast.Import):
            changed = False
            for a in node.names:
                if a.name.split(".")[0] != g.pkg:
                    out.append(f"import {collect._alias(a.name, a.asname)}")
                    continue
                tgt = g.lookup(a.name.split(".")[1:])
                if tgt is None:
                    out.append(f"import {collect._alias(a.name, a.asname)}")
                    continue
                changed = True
                if tgt == root and g.single is False:
                    out.append(f"import sys as _lego_sys; "
                               f"{a.asname or g.pkg} = {root_expr}")
                    continue
                pkg_parts, last = mod_ref(tgt)
                if a.asname:
                    out.append(f"from {_rel(here, pkg_parts)} import {last} as "
                               f"{a.asname}")
                else:
                    counter[0] += 1
                    out.append(f"from {_rel(here, pkg_parts)} import {last} as "
                               f"_lego_m{counter[0]}; import sys as _lego_sys; "
                               f"{g.pkg} = {root_expr}")
            return "; ".join(out) if changed else None
        base = import_base(g, path, node)
        if base is None:
            return None
        tgt = g.lookup(base)
        for a in node.names:
            local = a.asname or a.name
            if a.name == "*":
                if tgt is None:
                    return None
                out.append((_rel(here, g.impl_parts(tgt)), "*"))
                continue
            sub = g.lookup(base + [a.name])
            if sub and (tgt is None or (g.is_package(tgt) and
                                        a.name not in g.modules[tgt].symbols)):
                pkg_parts, last = mod_ref(sub)
                out.append((_rel(here, pkg_parts), collect._alias(last, local)))
                continue
            if tgt is None:
                return None
            res = g.resolve(tgt, a.name)
            if res is None:
                out.append((_rel(here, g.impl_parts(tgt)), collect._alias(a.name, local)))
            elif res[0] == "module":
                pkg_parts, last = mod_ref(res[1])
                out.append((_rel(here, pkg_parts), collect._alias(last, local)))
            else:
                out.append((_rel(here, g.impl_parts(res[0])), collect._alias(res[1], local)))
        return collect.merge_from(out)

    return fn


def _package_dirs(g: RepoGraph, modules) -> list:
    """Impl-relative package directories a component needs ('' = root)."""
    dirs = {""}
    for p in modules:
        d = os.path.dirname(p) if not g.single else ""
        while d:
            dirs.add(d)
            d = os.path.dirname(d)
    return sorted(dirs)


def _literals(g: RepoGraph, init_path: str) -> list:
    m = g.modules.get(init_path)
    if m is None:
        return []
    return [f"{s.name} = {s.literal}" for s in m.symbols.values()
            if s.kind == "constant" and s.literal is not None
            and s.name != "__all__"]


def generated_init(g: RepoGraph, comp: Component, pkg_dir: str,
                   summary: str = "") -> str:
    """``__init__`` for a package directory whose original is not carried."""
    orig = (f"{pkg_dir}/__init__.py" if pkg_dir else "__init__.py")
    lines = []
    if not pkg_dir:
        lines.append(repr(summary or f"Code Primitive recovered from {g.pkg}."))
    lines += _literals(g, orig) if not g.single else []
    if pkg_dir:
        return "\n".join(lines) + ("\n" if lines else "")
    taken, names = set(), []
    for p in comp.modules:
        if g.is_package(p) and not g.single:
            continue
        m = g.modules[p]
        mine = [n for n in m.exports if n in m.symbols and not n.startswith("_")
                and not m.symbols[n].conditional and n not in taken]
        if not mine:
            continue
        taken.update(mine)
        names += mine
        ref = ".".join(g.impl_parts(p))
        lines.append(f"from .{ref} import {', '.join(mine)}")
    if names:
        lines.append("")
        lines.append("__all__ = [%s]" % ", ".join(repr(n) for n in names))
    return "\n".join(lines) + "\n"


def build_impl(g: RepoGraph, comp: Component, summary: str = "") -> dict:
    """{impl-relative path: text}: rewritten modules, generated package
    ``__init__`` files, and text data files the modules name."""
    impl = {}
    for p in comp.modules:
        src = g.modules[p].source
        try:
            impl[p] = collect.replace_imports(src, impl_rewriter(g, p))
        except SyntaxError:
            impl[p] = src
    for d in _package_dirs(g, comp.modules):
        rel = f"{d}/__init__.py" if d else "__init__.py"
        if rel not in impl:
            impl[rel] = generated_init(g, comp, d, summary)
    for p in comp.modules:
        base = os.path.join(g.pkg_dir, os.path.dirname(p))
        for sub, full in collect.data_refs(base, g.modules[p].source,
                                           g.pkg_dir, top_dirs=True).items():
            rel = os.path.relpath(full, g.pkg_dir).replace(os.sep, "/")
            text = collect._read_text(full)
            if text is not None and rel not in impl:
                impl[rel] = text
    return impl


# ------------------------------------------------------------------ env
class Env:
    """The validation interpreter: a fresh venv via ``core`` (shared by all
    candidates of one source repository), or an existing interpreter given as
    ``python`` (no installs are attempted then)."""

    def __init__(self, proj_dir: str | None = None, python: str | None = None):
        self.owned = python is None
        self.installed: dict = {}
        if python:
            self.python = python
            r = core.sh([python, "-c", "import sys;print('%d.%d'%sys.version_info[:2])"],
                        60)
            self.version = r.stdout.strip() if r and r.returncode == 0 else ""
        else:
            ver, _why = core.fresh_venv(proj_dir)
            self.python = core.py()
            r = core.sh([self.python, "-c",
                         "import sys;print('%d.%d'%sys.version_info[:2])"], 60)
            self.version = r.stdout.strip() if r and r.returncode == 0 else ver
        self.has_pytest = (core.ensure_pytest() if self.owned
                           else self.importable("pytest"))

    def importable(self, top: str) -> bool:
        r = core.sh([self.python, "-c", f"import {top}"], 60,
                    env=core.clean_env(PYTHONPATH="", PYTHONNOUSERSITE="1"),
                    cwd=tempfile.gettempdir())
        return bool(r and r.returncode == 0)

    def ensure(self, tops) -> dict:
        """{import name: distribution installed, or None when it failed}."""
        for top in tops:
            if top in self.installed:
                continue
            if self.importable(top):
                self.installed[top] = "present"
            elif self.owned:
                self.installed[top] = core.install_for_import(top)
            else:
                self.installed[top] = None
        return {t: self.installed.get(t) for t in tops}

    def close(self):
        if self.owned:
            core.rm_rf(core.VENV)


# ------------------------------------------------------------ validation
_GUARD = '''"""Validation plugin written by lego.mining: per-test outcomes and a
check that the *source* package was never imported."""
import json
import os
import sys

_OUT = {"tests": {}, "errors": {}}


def pytest_runtest_logreport(report):
    if report.when == "call" or report.outcome != "passed":
        if report.when == "call":
            out = report.outcome
        else:
            out = "error" if report.failed else report.outcome
        prev = _OUT["tests"].get(report.nodeid)
        if prev not in ("failed", "error"):
            _OUT["tests"][report.nodeid] = out
        if report.failed:
            _OUT["errors"][report.nodeid] = str(report.longrepr)[-3000:]


def pytest_collectreport(report):
    if report.failed:
        key = report.nodeid or "<collect>"
        _OUT["tests"][key] = "collect-error"
        _OUT["errors"][key] = str(report.longrepr)[-3000:]


def pytest_sessionfinish(session, exitstatus):
    pkg = os.environ.get("LEGO_GUARD_PKG", "")
    _OUT["leaked"] = sorted(m for m in sys.modules
                            if pkg and (m == pkg or m.startswith(pkg + ".")))
    with open(os.environ["LEGO_GUARD_OUT"], "w") as fh:
        json.dump(_OUT, fh)
'''


@dataclass
class RunResult:
    ok: bool
    stage: str                          # import | tests
    log: str = ""
    tests: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    leaked: list = field(default_factory=list)
    timeout: bool = False

    @property
    def passed(self) -> int:
        return sum(1 for v in self.tests.values() if v == "passed")

    @property
    def failing(self) -> dict:
        return {k: self.errors.get(k, "") for k, v in self.tests.items()
                if v in ("failed", "error", "collect-error")}


def _write(root: str, files: dict):
    for rel, text in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)


def run_isolated(env: Env, ppkg: str, impl: dict, tests: dict, src_pkg: str,
                 import_mods: list, mode: str = "tests") -> RunResult:
    scratch = tempfile.mkdtemp(prefix="lego_mine_")
    try:
        _write(os.path.join(scratch, ppkg), impl)
        if tests:
            _write(os.path.join(scratch, "tests"), tests)
        for rel, text in impl.items():
            if rel.endswith(".py"):
                try:
                    compile(text, rel, "exec")
                except (SyntaxError, ValueError) as exc:
                    return RunResult(False, "build", f"{rel}: {exc}")
        _write(scratch, {"_lego_guard.py": _GUARD, "pytest.ini": "[pytest]\n"})
        guard_out = os.path.join(scratch, "_guard.json")
        envv = core.clean_env(PYTHONPATH=scratch, PYTHONNOUSERSITE="1",
                              PYTHONDONTWRITEBYTECODE="1",
                              LEGO_GUARD_OUT=guard_out, LEGO_GUARD_PKG=src_pkg)
        code = ("import importlib\nfor m in %r:\n    importlib.import_module(m)\n"
                % ([ppkg] + import_mods))
        r = core.sh([env.python, "-c", code], IMPORT_TIMEOUT, env=envv, cwd=scratch)
        if r is None:
            return RunResult(False, "import", "import timed out", timeout=True)
        if r.returncode != 0:
            return RunResult(False, "import", (r.stdout + r.stderr)[-4000:])
        if mode == "import" or not tests:
            return RunResult(True, "import", "")
        r = core.sh([env.python, "-m", "pytest", "tests", "-q", "--tb=short",
                     "-c", "pytest.ini", "--rootdir", scratch,
                     "-p", "no:cacheprovider", "-p", "_lego_guard"],
                    TEST_TIMEOUT, env=envv, cwd=scratch)
        if r is None:
            return RunResult(False, "tests", "carried tests timed out",
                             timeout=True)
        log = (r.stdout + r.stderr)[-6000:]
        res = RunResult(False, "tests", log)
        if os.path.exists(guard_out):
            data = json.load(open(guard_out))
            res.tests, res.errors = data.get("tests", {}), data.get("errors", {})
            res.leaked = data.get("leaked", [])
        elif "conftest" in log:
            res.tests["tests/conftest.py"] = "collect-error"
            res.errors["tests/conftest.py"] = log
        res.ok = (r.returncode == 0 and res.passed > 0 and not res.failing)
        return res
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ------------------------------------------------------------- refinement
_NO_MOD = re.compile(r"No module named ['\"]([\w.]+)['\"]")
_NO_NAME = re.compile(r"cannot import name ['\"](\w+)['\"] from ['\"]?([\w.]+)")
_NAME_ERR = re.compile(r"NameError: name ['\"](\w+)['\"] is not defined")
_NO_ATTR = re.compile(r"module ['\"]([\w.]+)['\"] has no attribute ['\"](\w+)['\"]")


def _internal(g: RepoGraph, ppkg: str, dotted: str):
    """Package module for a dotted name inside the primitive, or None."""
    parts = dotted.split(".")
    if VENDOR_PKG in parts:
        parts = parts[parts.index(VENDOR_PKG) + 1:]
    if not parts or parts[0] != ppkg:
        return None
    rest = parts[1:]
    if g.single:
        return g.lookup([]) if rest in ([], [g.pkg]) else None
    return g.lookup(rest)


def repair(g: RepoGraph, ppkg: str, text: str, mods: set):
    """-> ("add", path) | ("install", top) | ("leak", name) | None."""
    for m in _NO_MOD.finditer(text):
        name = m.group(1)
        top = name.split(".")[0]
        if top == ppkg:
            parts = name.split(".")[1:]
            for k in range(len(parts), 0, -1):
                p = g.lookup(parts[:k])
                if p and p not in mods:
                    return ("add", p)
            continue
        if top == g.pkg:
            return ("leak", name)
        if top not in STDLIB and top not in ("tests", "conftest"):
            return ("install", top)
    for m in _NO_NAME.finditer(text):
        p = _internal(g, ppkg, m.group(2))
        if p:
            res = g.resolve(p, m.group(1))
            dest = (res[1] if res and res[0] == "module" else
                    res[0] if res else None)
            if dest and dest not in mods:
                return ("add", dest)
    for m in _NO_ATTR.finditer(text):
        p = _internal(g, ppkg, m.group(1))
        if p:
            res = g.resolve(p, m.group(2))
            dest = (res[1] if res and res[0] == "module" else
                    res[0] if res else None)
            if dest and dest not in mods:
                return ("add", dest)
    for m in _NAME_ERR.finditer(text):
        cands = [p for p in g.definers(m.group(1)) if p not in mods]
        if len(cands) == 1:
            return ("add", cands[0])
    return None


def _node_key(nodeid: str):
    """'tests/test_x.py::C::test_y[1]' -> ('test_x.py', ['C', 'test_y'])."""
    parts = nodeid.split("::")
    f = parts[0]
    f = f[len("tests/"):] if f.startswith("tests/") else f
    return f, [re.sub(r"\[.*$", "", x) for x in parts[1:]]


def _remove_defs(source: str, paths: list) -> str | None:
    """Delete test functions / methods (``[name]`` or ``[Class, name]``) by
    line span, keeping the rest verbatim. None if nothing test-like remains."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines = source.splitlines(keepends=True)
    drop = []

    def span(n):
        start = min([n.lineno] + [d.lineno for d in n.decorator_list])
        return start, n.end_lineno

    want = {tuple(p) for p in paths}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and (n.name,) in want:
            drop.append(span(n))
        elif isinstance(n, ast.ClassDef):
            if (n.name,) in want:
                drop.append(span(n))
                continue
            meths = [x for x in n.body
                     if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]
            gone = [x for x in meths if (n.name, x.name) in want]
            tests_left = [x for x in meths if x.name.startswith("test")
                          and x not in gone]
            if gone and not tests_left:
                drop.append(span(n))
            else:
                drop += [span(x) for x in gone]
    for a, b in sorted(drop, reverse=True):
        del lines[a - 1:b]
    out = "".join(lines)
    try:
        compile(out, "<pruned>", "exec")
    except SyntaxError:
        return None
    return out if re.search(r"^\s*(async\s+)?def test", out, re.M) else None


def prune(tests: dict, failing: dict) -> tuple[dict, list]:
    """Remove failing tests from V_i. -> (new tests, removed node ids)."""
    tests = dict(tests)
    by_file, removed = {}, []
    for nodeid, _txt in sorted(failing.items()):
        f, path = _node_key(nodeid)
        if f in ("tests", "", "<collect>"):
            f = "conftest.py"
        if not path:                       # collection error: drop the file
            if f in tests:
                tests.pop(f)
                removed.append(nodeid)
            continue
        by_file.setdefault(f, []).append(path)
    for f, paths in by_file.items():
        if f not in tests:
            continue
        new = _remove_defs(tests[f], paths)
        removed += [f"{f}::{'::'.join(p)}" for p in paths]
        if new is None:
            tests.pop(f)
        else:
            tests[f] = new
    return tests, removed


def _n_test_files(tests: dict) -> int:
    return sum(1 for k in tests if re.match(r"^(test_.*|.*_test)\.py$",
                                            os.path.basename(k)))


# ----------------------------------------------------------- description
_ID = re.compile(r"[^A-Za-z]+|(?<=[a-z0-9])(?=[A-Z])")


def derive_description(g: RepoGraph, comp: Component, iface: dict) -> dict:
    seed = g.modules[comp.seed]
    name = g.dotted(comp.seed)
    docs = []
    for e in iface["exports"]:
        p = iface["owner"].get(e)
        s = g.modules[p].symbols.get(e) if p else None
        line = collect._first_line(s.doc) if s else ""
        if line and line not in docs:
            docs.append(line)
    summary = collect._first_line(seed.doc) or (docs[0] if docs else "") or (
        f"{', '.join(iface['exports'][:4])} from {g.pkg}")
    caps = docs[:8]
    if not caps:
        words = []
        for e in iface["exports"][:8]:
            words.append(" ".join(w.lower() for w in _ID.split(e) if w))
        caps = [w for w in words if w]
    return {"name": name, "summary": summary[:300], "capabilities": caps}


def describe(llm, g: RepoGraph, comp: Component, iface: dict, impl: dict,
             repo: str) -> dict:
    """Model-written name/summary/capabilities; falls back to derived."""
    base = derive_description(g, comp, iface)
    if llm is None:
        return base
    code = "\n\n".join(f"# --- {p}\n{c}" for p, c in sorted(impl.items())
                       if p.endswith(".py"))[:7000]
    obj = llm.json(
        f"You are cataloguing a reusable component recovered from the "
        f"repository {repo}.\n\nExports: {', '.join(iface['exports'][:30])}\n"
        f"Contract:\n{iface['contract'][:2000]}\n\nImplementation:\n{code}\n\n"
        "Return a JSON object with keys: \"name\" (a short descriptive name, at "
        "most five words), \"summary\" (one sentence: what it does), "
        "\"capabilities\" (3 to 8 short phrases, each a need a caller could "
        "have that this component meets).", max_tokens=800)
    if not isinstance(obj, dict):
        return base
    caps = [str(c)[:160] for c in (obj.get("capabilities") or [])
            if str(c).strip()][:8]
    return {"name": str(obj.get("name") or base["name"])[:80],
            "summary": str(obj.get("summary") or base["summary"])[:400],
            "capabilities": caps or base["capabilities"], "described_by": llm.alias}


def synth_tests(llm, ppkg: str, impl: dict, iface: dict) -> dict:
    code = "\n\n".join(f"# --- {ppkg}/{p}\n{c}" for p, c in sorted(impl.items())
                       if p.endswith(".py"))[:9000]
    text = llm.code(
        f"Write a pytest test module for the Python package `{ppkg}` below. "
        f"Import only from `{ppkg}` (for example `from {ppkg}.<module> import "
        f"<name>`), the standard library and pytest. Test the observable "
        f"behaviour of these exports: {', '.join(iface['exports'][:20])}. Write "
        f"4 to 10 small deterministic tests; no network, no files outside "
        f"tmp_path.\n\n{code}", max_tokens=4000)
    try:
        compile(text, "test_synthesized.py", "exec")
    except SyntaxError:
        return {}
    return {"test_synthesized.py": text} if "def test" in text else {}


# ------------------------------------------------------------ synthesize
@dataclass
class Options:
    kind: str = "mined"
    mode: str = "tests"                  # tests | import
    refine: int = REFINE_ROUNDS
    max_files: int = MAX_FILES
    max_lines: int = MAX_LINES
    describe_llm: object = None
    tests_llm: object = None
    keep_excluded: bool = False


def _plugin_imports(tests: dict) -> set:
    text = "\n".join(tests.values())
    return {mod for pat, mod in _PLUGIN_HINTS if pat.search(text)}


def synthesize(g: RepoGraph, comp: Component, out_root: str, pid: str,
               source: dict, env: Env, opts: Options | None = None,
               context: dict | None = None) -> dict:
    """Build, validate and (if admitted) write one primitive. Returns a log row.
    ``context`` holds extra X_i files (the license); doc excerpts that mention
    an export are added here."""
    opts = opts or Options()
    ppkg = package_name(pid)
    mods = list(comp.modules)
    row = {"pid": pid, "seed": comp.seed, "status": "excluded", "reason": "",
           "added_modules": [], "pruned": [], "installed": {}}
    empty = {"origin": {}, "external": [], "dropped": {}, "exercising": []}
    tests, tinfo, synth = {}, dict(empty), {}
    need_collect, adds, rounds = True, 0, 0
    iface = impl = deps = None
    res = RunResult(False, "import")
    while rounds < opts.refine + 8:
        rounds += 1
        cur = Component(comp.seed, mods, g.lines(mods), comp.notes, comp.seeds)
        if need_collect:
            tests, tinfo = ({}, dict(empty)) if opts.mode == "import" else \
                collect.collect_tests(g, cur, ppkg)
            tests.update(synth)
            exercising = set(tinfo["exercising"]) | set(synth)
            need_collect = False
        iface = collect.infer_interface(g, cur, tinfo["origin"], ppkg)
        impl = build_impl(g, cur)
        if (opts.mode == "tests" and not exercising and opts.tests_llm is not None
                and not synth):
            synth = synth_tests(opts.tests_llm, ppkg, impl, iface)
            tests.update(synth)
            exercising |= set(synth)
        if opts.mode == "tests" and not exercising:
            row["reason"] = ("no carried tests" if not tests else
                             "no carried test exercises the seed")
            break
        deps = collect.external_deps(g, mods, list(tinfo["external"]) +
                                     sorted(_plugin_imports(tests)))
        got = env.ensure(deps["required_imports"])
        missing = [t for t, d in got.items() if d is None]
        if missing:
            row["reason"] = f"missing dependency: {', '.join(missing[:5])}"
            break
        if tests:
            env.ensure(deps["test_imports"])
        import_mods = [collect.prim_dotted(g, ppkg, p) for p in mods]
        res = run_isolated(env, ppkg, impl, tests, g.pkg, import_mods, opts.mode)
        if res.ok:
            if res.leaked:
                row["reason"] = f"source package imported: {res.leaked[:3]}"
                break
            seed_pass = sum(1 for k, v in res.tests.items()
                            if v == "passed" and _node_key(k)[0] in exercising)
            if opts.mode == "tests" and seed_pass == 0:
                row["reason"] = "no passing test exercises the seed"
                break
            row["status"], row["reason"] = "admitted", (
                "import ok" if opts.mode == "import" else
                f"{res.passed} carried tests pass ({seed_pass} on the seed)")
            break
        if res.timeout:
            row["reason"] = f"{res.stage} timed out"
            break
        texts = res.log if res.stage in ("import", "build") else "\n".join(
            [res.log] + list(res.failing.values()))
        act = repair(g, ppkg, texts, set(mods))
        if act and act[0] == "add" and adds < opts.refine:
            ext = set(mods) | g.closure([act[1]])
            if len(ext) > opts.max_files or g.lines(ext) > opts.max_lines:
                row["reason"] = f"refinement exceeds bound (needs {act[1]})"
                break
            adds += 1
            row["added_modules"].append(act[1])
            mods = mods + sorted(ext - set(mods))
            need_collect = True
            continue
        if act and act[0] == "install" and act[1] not in row["installed"]:
            row["installed"][act[1]] = env.ensure([act[1]])[act[1]]
            if row["installed"][act[1]]:
                continue
            if res.stage in ("import", "build"):
                row["reason"] = f"missing dependency: {act[1]}"
                break
        if act and act[0] == "leak" and res.stage in ("import", "build"):
            row["reason"] = f"unrewritable reference to source package: {act[1]}"
            break
        if res.stage in ("import", "build"):
            last = [l for l in res.log.strip().splitlines() if l.strip()]
            row["reason"] = f"{res.stage} failed: "
            row["reason"] += (last[-1] if last else "?")[:200]
            break
        if not tests:
            row["reason"] = "no carried tests"
            break
        new, removed = prune(tests, res.failing)
        if not removed or new == tests:
            row["reason"] = (f"carried tests fail: "
                             f"{(res.log.strip().splitlines() or ['?'])[-1][:200]}")
            break
        row["pruned"] += removed
        tests = new
        exercising &= set(tests)
        if not _n_test_files(tests) or not exercising:
            row["reason"] = "no passing test exercises the seed after pruning"
            break
    else:
        row["reason"] = row["reason"] or "refinement budget exhausted"

    admitted = row["status"] == "admitted"
    synthesized = bool(synth) and any(k in tests for k in synth)
    row.update(modules=mods, n_passed=res.passed, rounds=rounds,
               tests_synthesized=synthesized)
    if not admitted and not opts.keep_excluded:
        return row
    desc = describe(opts.describe_llm, g, comp, iface, impl,
                    source.get("repo", ""))
    impl = build_impl(g, Component(comp.seed, mods, 0), desc["summary"])
    # prefer the distribution that actually made the import work
    ext = sorted({env.installed.get(t) if env.installed.get(t) not in
                  (None, "present", "already") else deps["imports"][t]
                  for t in deps["required_imports"]})
    src = dict(source)
    if not src.get("paths"):             # web sourcing records its own
        src["paths"] = [os.path.relpath(os.path.join(g.pkg_dir, p), g.repo_dir)
                        .replace(os.sep, "/") for p in mods]
        src["test_paths"] = sorted(set(tinfo["origin"].values()))
    meta = {
        "pid": pid, "name": desc["name"], "summary": desc["summary"],
        "capabilities": desc["capabilities"], "kind": opts.kind, "source": src,
        "interface": {"exports": iface["exports"],
                      "signatures": iface["signatures"],
                      "contract": iface["contract"], "package": ppkg,
                      "modules": mods, "seed": comp.seed},
        "dependencies": {"external": ext, "python": env.version,
                         "optional": deps["optional"], "test": deps["test"]},
        "validated": admitted,
        "validation": {"mode": opts.mode, "reason": row["reason"],
                       "passed": res.passed, "pruned": row["pruned"],
                       "seed_tests": sorted(exercising),
                       "added_modules": row["added_modules"], "rounds": rounds},
        "tests_synthesized": synthesized,
        "mining": {"seeds": comp.seeds, "notes": comp.notes,
                   "test_origin": tinfo["origin"],
                   "dropped_tests": tinfo.get("dropped", {})},
    }
    if desc.get("described_by"):
        meta["described_by"] = desc["described_by"]
    ctx = dict(context or {})
    ctx.update(collect.docs_context(g, iface["exports"]))
    root = os.path.join(out_root, pid)
    core.rm_rf(root)
    primitive_io.write(root, meta, impl, tests, ctx)
    row["written"] = True
    return row
