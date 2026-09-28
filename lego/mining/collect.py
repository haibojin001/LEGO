"""Steps 4-5 of mining: interface I_i, reachable tests V_i, context X_i, deps D_i.

  interface   exports are the component's public names that are *observed in
              use*: imported by package modules outside the component, reached
              by a carried test, or re-exported by the package hub. Signatures
              are ``interface.stub`` of the component modules filtered to those
              names; the contract lists each export's signature and first
              docstring line.
  tests       whole test files from the repository's suite whose every package
              reference resolves inside the component (``segment.carried``),
              plus the conftest chain and test-local helper modules when those
              resolve too, plus data files the tests name. Imports of the
              source package are rewritten to the primitive's package name, so
              ``from pkg import f`` becomes ``from p_<pid>.mod import f`` with
              re-exports followed to the defining module. Files are laid out
              flat under ``tests/`` (pytest ``prepend`` import mode puts that
              directory on ``sys.path``, which is how carried helpers import).
  deps        third-party top-level imports of the component, mapped to
              distribution names (the repository's declared dependencies win
              over the generic import->dist table); test-only imports separately.
  context     README / docs paragraphs that mention an export, and the license.

The import rewriter (``replace_imports``) edits statement spans in place, so
comments and formatting of the carried files survive.
"""

from __future__ import annotations

import ast
import os
import re

from lego.harness import core
from lego.harness.interface import stub
from lego.mining.graph import (STDLIB, RepoGraph, iter_imports, local_helper,
                               test_import_base)
from lego.mining.segment import Component, carried

MAX_TEST_FILES = 8
MAX_TEST_CHARS = 400_000
MAX_DATA_BYTES = 2_000_000
MAX_DATA_FILES = 200
_TEST_TOOLS = {"pytest", "_pytest", "pluggy", "py"}


# ------------------------------------------------------------ span rewriting
def replace_imports(source: str, fn, tree=None) -> str:
    """Replace every import statement for which ``fn(node)`` returns a string.
    Replacements of several statements are joined with ``; `` so they stay
    valid wherever a simple statement is (``if x: import y`` included)."""
    tree = tree or ast.parse(source)
    lines = source.splitlines(keepends=True)
    starts, acc = [], 0
    for ln in lines:
        starts.append(acc)
        acc += len(ln)

    def off(lineno, col):
        line = lines[lineno - 1]
        return starts[lineno - 1] + len(
            line.encode("utf-8")[:col].decode("utf-8", "ignore"))

    edits = []
    for node, _lazy, _tonly in iter_imports(tree):
        new = fn(node)
        if new is None:
            continue
        edits.append((off(node.lineno, node.col_offset),
                      off(node.end_lineno, node.end_col_offset), new))
    out = source
    for a, b, new in sorted(edits, reverse=True):
        out = out[:a] + new + out[b:]
    return out


def _alias(name: str, asname: str | None) -> str:
    return f"{name} as {asname}" if asname and asname != name else name


def merge_from(pairs) -> str:
    """[(module, alias text)] -> from-imports, one per module, first-seen order."""
    by = {}
    for mod, al in pairs:
        by.setdefault(mod, [])
        if al not in by[mod]:
            by[mod].append(al)
    return "; ".join(f"from {m} import {', '.join(a)}" for m, a in by.items())


def prim_dotted(g: RepoGraph, ppkg: str, path: str) -> str:
    """Absolute dotted name of package module ``path`` inside the primitive."""
    return ".".join([ppkg] + g.impl_parts(path))


def _root_dotted(g: RepoGraph, ppkg: str) -> str:
    return f"{ppkg}.{g.pkg}" if g.single else ppkg


def _str_refs(g: RepoGraph, ppkg: str, source: str) -> str:
    """``"pkg.mod.attr"`` string literals (``mock.patch`` targets,
    ``importorskip``) follow the package rename."""
    root = _root_dotted(g, ppkg)
    return re.sub(r"(['\"])%s(?=[.'\"])" % re.escape(g.pkg),
                  lambda m: m.group(1) + root, source)


class Unresolvable(Exception):
    pass


def test_rewriter(g: RepoGraph, ppkg: str, rel: str, helper_names: dict):
    """Import rewriter for one carried test/helper file. ``helper_names`` maps
    repo-relative helper paths to their flat module names under tests/."""
    root = g.lookup([])

    def helper_for(parts, level):
        h = local_helper(g, rel, parts, level)
        if h is None:
            return None
        if h not in helper_names:
            raise Unresolvable(f"helper {h} not carried")
        return helper_names[h]

    def fn(node):
        out = []
        if isinstance(node, ast.Import):
            changed = False
            for a in node.names:
                top = a.name.split(".")[0]
                if top == g.pkg:
                    changed = True
                    tgt = g.lookup(a.name.split(".")[1:])
                    if tgt is None:
                        raise Unresolvable(a.name)
                    dot = prim_dotted(g, ppkg, tgt)
                    if a.asname:
                        out.append(f"import {dot} as {a.asname}")
                    elif tgt == root:
                        out.append(f"import {dot} as {g.pkg}")
                    else:
                        out.append(f"import {dot}; import "
                                   f"{_root_dotted(g, ppkg)} as {g.pkg}")
                    continue
                flat = helper_for(a.name.split("."), 0)
                if flat:
                    changed = True
                    if "." in a.name and not a.asname:
                        raise Unresolvable(f"dotted helper import {a.name}")
                    local = a.asname or a.name
                    out.append(f"import {_alias(flat, local)}")
                    continue
                out.append(f"import {_alias(a.name, a.asname)}")
            return "; ".join(out) if changed else None
        base = test_import_base(g, rel, node)
        if base is not None:
            tgt = g.lookup(base)
            if tgt is None:
                raise Unresolvable(node.module or ".")
            for a in node.names:
                if a.name == "*":
                    out.append((prim_dotted(g, ppkg, tgt), "*"))
                    continue
                sub = g.lookup(base + [a.name])
                if sub and g.is_package(tgt) and a.name not in g.modules[tgt].symbols:
                    out.append((prim_dotted(g, ppkg, tgt), _alias(a.name, a.asname)))
                    continue
                res = g.resolve(tgt, a.name)
                if res is None:
                    raise Unresolvable(f"{node.module}.{a.name}")
                if res[0] == "module":
                    d = prim_dotted(g, ppkg, res[1])
                    head, _, last = d.rpartition(".")
                    out.append((head, _alias(last, a.asname or a.name)))
                else:
                    out.append((prim_dotted(g, ppkg, res[0]),
                                _alias(res[1], a.asname or a.name)))
            return merge_from(out)
        parts = node.module.split(".") if node.module else []
        if node.level and not parts:
            names = []
            for a in node.names:
                flat = helper_for([a.name], node.level)
                if flat is None:
                    raise Unresolvable(f"relative import {a.name}")
                names.append(f"import {flat} as {a.asname or a.name}")
            return "; ".join(names)
        flat = helper_for(parts, node.level)
        if flat:
            return f"from {flat} import " + ", ".join(
                _alias(a.name, a.asname) for a in node.names)
        if node.level:
            raise Unresolvable(f"relative import {'.' * node.level}{node.module}")
        return None

    return fn


def rewrite_test(g: RepoGraph, ppkg: str, rel: str, source: str,
                 helper_names: dict) -> str:
    out = replace_imports(source, test_rewriter(g, ppkg, rel, helper_names))
    return _str_refs(g, ppkg, out)


# ------------------------------------------------------------------ tests
def _flat_name(rel: str, taken: set, test: bool = True) -> str:
    base = os.path.basename(rel)
    if test and not re.match(r"^(test_.*|.*_test)\.py$", base):
        base = "test_" + base
    parent = os.path.basename(os.path.dirname(rel)) or "root"
    cand = base
    if cand in taken:
        cand = (f"test_{parent}_{base[5:]}" if base.startswith("test_")
                else f"{parent}_{base}")
    n = 2
    while cand in taken:
        cand = f"{cand[:-3]}_{n}.py"
        n += 1
    taken.add(cand)
    return cand


def _string_constants(source: str) -> list:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    return sorted({n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and 0 < len(n.value) < 200 and "\n" not in n.value})


def data_refs(base_dir: str, source: str, stop: str,
              top_dirs: bool = False) -> dict:
    """Files/dirs next to a source file that its string literals name.
    -> {path relative to base_dir: absolute path}. Bounded in size. Directories
    directly under ``stop`` count only with ``top_dirs`` (a repo-root directory
    name in a test is rarely test data; one in a package usually is)."""
    out, total = {}, 0
    base_real = os.path.realpath(base_dir)
    at_top = base_real == os.path.realpath(stop) and not top_dirs
    for s in _string_constants(source):
        s = s.strip()
        s = s[2:] if s.startswith("./") else s
        if not s or s.startswith(("/", "~")) or ".." in s or s.endswith(".py"):
            continue
        full = os.path.realpath(os.path.join(base_dir, s))
        if not full.startswith(base_real + os.sep) or not os.path.exists(full):
            continue
        if os.path.isdir(full):
            if at_top:
                continue            # a repo-root dir name is not test data
            files = [os.path.join(r, f) for r, _d, fs in os.walk(full) for f in fs]
            if any(f.endswith(".py") for f in files):
                continue            # a code directory, not data
        else:
            files = [full]
        files = [f for f in files if not f.endswith(".pyc")]
        if len(files) > MAX_DATA_FILES:
            continue
        size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        if total + size > MAX_DATA_BYTES:
            continue
        total += size
        for f in files:
            out[os.path.relpath(f, base_real)] = f
    return out


def _read_text(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        return data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _conftest_chain(g: RepoGraph, rel: str) -> list:
    """Repo-relative conftest.py files that apply to a test file, root first."""
    parts = os.path.dirname(rel).split(os.sep) if os.path.dirname(rel) else []
    chain = []
    for i in range(len(parts) + 1):
        c = os.path.join(*(parts[:i] + ["conftest.py"])) if parts[:i] else "conftest.py"
        if c in g.tests:
            chain.append(c)
    return chain


def _helper_ok(g: RepoGraph, rel: str, mods: set, seen: set, depth: int = 0) -> bool:
    t = g.analysis(rel)
    if t is None or t.error or t.unresolved or not t.targets <= mods or depth > 2:
        return False
    seen.add(rel)
    return all(h in seen or _helper_ok(g, h, mods, seen, depth + 1)
               for h in sorted(t.helpers))


def _split_future(text: str) -> tuple[list, str]:
    fut, rest = [], []
    for line in text.splitlines(keepends=True):
        if re.match(r"^from __future__ import ", line):
            fut.append(line.strip())
        else:
            rest.append(line)
    return fut, "".join(rest)


def collect_tests(g: RepoGraph, comp: Component, ppkg: str,
                  max_files: int = MAX_TEST_FILES) -> tuple[dict, dict]:
    """-> (tests {name under tests/: text}, info). ``info`` records the source
    file of every carried name, helpers, conftests, data files, test-only
    third-party imports and the files that were considered but dropped."""
    mods = set(comp.modules)
    info = {"origin": {}, "helpers": [], "conftest": [], "data": [],
            "external": set(), "dropped": {}, "exercising": []}
    chosen, helpers, confs = [], set(), []
    for t in carried(g, comp.modules):
        if len(chosen) >= max_files:
            info["dropped"][t.path] = "test-file cap"
            continue
        seen = set()
        if not all(_helper_ok(g, h, mods, seen) for h in sorted(t.helpers)):
            info["dropped"][t.path] = "helper imports leave the component"
            continue
        chosen.append(t)
        helpers |= seen
        for c in _conftest_chain(g, t.path):
            if c not in confs:
                confs.append(c)
    conf_ok = []
    for c in confs:
        seen = set()
        ct = g.tests[c]
        if (not ct.error and not ct.unresolved and ct.targets <= mods
                and all(_helper_ok(g, h, mods, seen) for h in sorted(ct.helpers))):
            conf_ok.append(c)
            helpers |= seen
        else:
            info["dropped"][c] = "conftest reaches outside the component"
    taken = {"conftest.py"}
    names = {h: _flat_name(h, taken, test=False)[:-3] for h in sorted(helpers)}
    out, total = {}, 0
    for t in chosen:
        try:
            text = rewrite_test(g, ppkg, t.path, t.source, names)
        except (Unresolvable, SyntaxError) as e:
            info["dropped"][t.path] = f"unresolvable import: {e}"
            continue
        if total + len(text) > MAX_TEST_CHARS:
            info["dropped"][t.path] = "size cap"
            continue
        total += len(text)
        name = _flat_name(t.path, taken)
        out[name] = text
        info["origin"][name] = t.path
        info["external"] |= t.external
    if not out:
        info["external"] = []
        return {}, info
    for h, flat in names.items():
        src = _read_text(os.path.join(g.repo_dir, h))
        try:
            out[flat + ".py"] = rewrite_test(g, ppkg, h, src or "", names)
            info["origin"][flat + ".py"] = h
            info["helpers"].append(h)
            info["external"] |= g.analysis(h).external
        except (Unresolvable, SyntaxError) as e:
            info["dropped"][h] = f"helper: {e}"
    if conf_ok:
        futs, parts = [], []
        for c in conf_ok:
            try:
                text = rewrite_test(g, ppkg, c, g.tests[c].source, names)
            except (Unresolvable, SyntaxError) as e:
                info["dropped"][c] = f"conftest: {e}"
                continue
            f, body = _split_future(text)
            futs += [x for x in f if x not in futs]
            parts.append(f"# --- carried from {c}\n{body}")
            info["conftest"].append(c)
            info["external"] |= g.tests[c].external
        if parts:
            out["conftest.py"] = "\n".join(futs + [""] if futs else []) + \
                "\n\n".join(parts)
    for name, rel in list(info["origin"].items()):
        d = os.path.join(g.repo_dir, os.path.dirname(rel))
        for sub, full in data_refs(d, g.analysis(rel).source, g.repo_dir).items():
            if sub in out:
                continue
            text = _read_text(full)
            if text is not None:
                out[sub] = text
                info["data"].append(os.path.relpath(full, g.repo_dir))
    info["external"] = sorted(info["external"] - _TEST_TOOLS - {g.pkg})
    # files that reach the seed itself, directly or through a component module
    # that depends on it (tests of an internalized helper alone do not count)
    reach = {p for p in comp.modules if comp.seed in g.closure([p])}
    info["exercising"] = sorted(
        n for n, rel in info["origin"].items()
        if n != "conftest.py" and rel in g.tests and g.tests[rel].targets & reach)
    return out, info


# -------------------------------------------------------------- interface
def _filtered_stub(source: str, names: set) -> str:
    try:
        tree = ast.parse(stub(source))
    except SyntaxError:
        return ""
    keep = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name in names:
                keep.append(n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id in names for t in tg):
                keep.append(n)
    tree.body = keep
    return ast.unparse(tree) if keep else ""


def _first_line(doc: str) -> str:
    return (doc or "").strip().split("\n")[0].strip()


def _contract_line(g: RepoGraph, path: str, name: str) -> str:
    m = g.modules[path]
    try:
        tree = ast.parse(m.source)
    except SyntaxError:
        return name
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            ret = f" -> {ast.unparse(n.returns)}" if n.returns else ""
            return (f"{name}({ast.unparse(n.args)}){ret}: "
                    f"{_first_line(ast.get_docstring(n) or '')}").rstrip(": ")
        if isinstance(n, ast.ClassDef) and n.name == name:
            bases = ", ".join(ast.unparse(b) for b in n.bases)
            meths = [x.name for x in n.body
                     if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and (not x.name.startswith("_") or x.name == "__init__")]
            return (f"class {name}({bases}): "
                    f"{_first_line(ast.get_docstring(n) or '')}"
                    + (f" [methods: {', '.join(meths[:12])}]" if meths else ""))
    sym = m.symbols.get(name)
    if sym is not None and sym.kind == "constant":
        return f"{name} = {sym.literal or '...'}"
    return name


def infer_interface(g: RepoGraph, comp: Component, test_origin: dict,
                    ppkg: str) -> dict:
    """Exports observed in use + filtered signatures + contract."""
    mods = set(comp.modules)
    tested = set()
    for rel in test_origin.values():
        t = g.tests.get(rel)
        if t is not None:
            tested |= t.names
    owner, exports = {}, []
    for p in comp.modules:
        m = g.modules[p]
        for name in m.exports:
            if name.startswith("__") or name in owner:
                continue
            if name not in m.symbols:
                continue                       # re-export; the definer owns it
            users = g.uses.get((p, name), set())
            if (users - mods) or (p, name) in tested:
                owner[name] = p
                exports.append(name)
    if not any(owner.get(e) == comp.seed for e in exports):
        seed = g.modules[comp.seed]
        for name in seed.exports:
            if name in seed.symbols and not name.startswith("__") \
                    and name not in owner:
                owner[name] = comp.seed
                exports.append(name)
    sigs = []
    for p in comp.modules:
        names = {e for e in exports if owner[e] == p}
        if names:
            s = _filtered_stub(g.modules[p].source, names)
            if s:
                sigs.append(f"# {prim_dotted(g, ppkg, p)}\n{s}")
    contract = "\n".join(_contract_line(g, owner[e], e) for e in exports)
    return {"exports": exports, "owner": owner,
            "signatures": "\n\n".join(sigs)[:8000], "contract": contract[:4000]}


# -------------------------------------------------------------- dependencies
def dist_for(top: str, declared: set) -> str:
    """Distribution for an import name; the repo's declared deps win."""
    cands = core.dist_candidates(top)
    for c in cands:
        if c.replace("-", "_").replace(".", "_").lower() in declared:
            return c
    return cands[0]


def external_deps(g: RepoGraph, modules, test_tops=()) -> dict:
    mods = [g.modules[p] for p in modules]
    req = set().union(*(m.external for m in mods)) if mods else set()
    opt = set().union(*(m.optional for m in mods)) - req if mods else set()
    local = {g.pkg} | _repo_top_levels(g)
    req, opt = req - local, opt - local
    declared = core.declared_dep_names(g.repo_dir)
    test = set(test_tops) - req - local - STDLIB
    return {"imports": {t: dist_for(t, declared) for t in sorted(req | opt | test)},
            "external": sorted(dist_for(t, declared) for t in req),
            "optional": sorted(dist_for(t, declared) for t in opt),
            "test": sorted(dist_for(t, declared) for t in test),
            "required_imports": sorted(req), "test_imports": sorted(test)}


def _repo_top_levels(g: RepoGraph) -> set:
    """Top-level importable names that live in the repo itself (test helpers,
    sibling packages): never third-party distributions."""
    out = set()
    for d in (g.repo_dir, os.path.join(g.repo_dir, g.src_prefix)):
        try:
            for f in os.listdir(d):
                if f.endswith(".py"):
                    out.add(f[:-3])
                elif os.path.isfile(os.path.join(d, f, "__init__.py")):
                    out.add(f)
        except OSError:
            pass
    return out


# --------------------------------------------------------------- context
_SPDX = [
    (r"Apache License", r"Version 2\.0", "Apache-2.0"),
    (r"GNU AFFERO GENERAL PUBLIC", r"Version 3", "AGPL-3.0"),
    (r"GNU LESSER GENERAL PUBLIC", r"Version 3", "LGPL-3.0"),
    (r"GNU LESSER GENERAL PUBLIC", r"Version 2", "LGPL-2.1"),
    (r"GNU (LIBRARY )?GENERAL PUBLIC", r"Version 3", "GPL-3.0"),
    (r"GNU (LIBRARY )?GENERAL PUBLIC", r"Version 2", "GPL-2.0"),
    (r"Mozilla Public License", r"2\.0", "MPL-2.0"),
    (r"Eclipse Public License", r"2\.0", "EPL-2.0"),
    (r"BSD 3-Clause|Neither the name", "", "BSD-3-Clause"),
    (r"BSD 2-Clause|Redistribution and use in source and binary", "", "BSD-2-Clause"),
    (r"MIT License|Permission is hereby granted, free of charge", "", "MIT"),
    (r"ISC License|Permission to use, copy, modify, and(/or)? distribute", "", "ISC"),
    (r"This is free and unencumbered software", "", "Unlicense"),
    (r"PYTHON SOFTWARE FOUNDATION", "", "PSF-2.0"),
    (r"Boost Software License", "", "BSL-1.0"),
    (r"zlib License|This software is provided 'as-is'", "", "Zlib"),
]


def license_info(repo_dir: str) -> tuple[str, str]:
    """-> (SPDX-ish identifier, license text). The identifier is matched from
    the head of the LICENSE file; failing that its first line; failing that the
    ``license`` field of the build metadata."""
    try:
        names = sorted(os.listdir(repo_dir))
    except OSError:
        names = []
    for fn in names:
        if not re.match(r"^(licen[cs]e|copying|unlicense)([._-].*)?$", fn, re.I):
            continue
        p = os.path.join(repo_dir, fn)
        if not os.path.isfile(p):
            continue
        text = _read_text(p) or ""
        head = text[:3000]
        for a, b, spdx in _SPDX:
            if re.search(a, head, re.I) and (not b or re.search(b, head, re.I)):
                return spdx, text[:20000]
        first = next((l.strip() for l in text.splitlines() if l.strip()), "")
        return first[:80], text[:20000]
    for fn in ("pyproject.toml", "setup.cfg", "setup.py"):
        p = os.path.join(repo_dir, fn)
        txt = _read_text(p) if os.path.exists(p) else None
        if not txt:
            continue
        m = re.search(r"""(?m)^\s*license\s*=\s*(?:\{\s*text\s*=\s*)?["']([^"'\n]{1,60})""",
                      txt)
        if m:
            return m.group(1).strip(), ""
    return "", ""


def docs_context(g: RepoGraph, exports, max_chars: int = 6000) -> dict:
    """Paragraphs of README/docs that mention an export."""
    if not exports:
        return {}
    pat = re.compile(r"\b(%s)\b" % "|".join(re.escape(e) for e in exports[:40]))
    files = [os.path.join(g.repo_dir, f) for f in sorted(os.listdir(g.repo_dir))
             if re.match(r"^readme", f, re.I)]
    ddir = next((os.path.join(g.repo_dir, d) for d in ("docs", "doc")
                 if os.path.isdir(os.path.join(g.repo_dir, d))), None)
    if ddir:
        for r, dirs, fs in os.walk(ddir):
            dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
            files += [os.path.join(r, f) for f in sorted(fs)
                      if f.endswith((".md", ".rst", ".txt"))]
            if len(files) > 300:
                break
    out, total = [], 0
    for f in files[:300]:
        text = _read_text(f)
        if not text:
            continue
        for para in re.split(r"\n\s*\n", text):
            if total >= max_chars:
                break
            if pat.search(para) and len(para) < 3000:
                chunk = f"<!-- {os.path.relpath(f, g.repo_dir)} -->\n{para.strip()}"
                out.append(chunk)
                total += len(chunk)
    return {"docs.md": "\n\n".join(out)} if out else {}
