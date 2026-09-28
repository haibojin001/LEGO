"""Adapter layer for the transfer benchmarks (``external/benchmarks.yaml``).

An external task reaches LEGO -- and the matched feedback baseline -- through
the same two objects a LEGO-REPO task does, so ``lego.pipeline.construct`` runs
unchanged:

  ExternalTask        name, NL spec, starting tree, target package / modules,
                      upstream repository, grader and feedback commands; it
                      exposes ``clone``, ``extra`` and ``statement()`` like
                      ``lego.harness.task.Task``
  ExternalWorkspace   duck-types ``lego.harness.prepare.Workspace``: task, pkg,
                      modules, orig_src, record, proj_dir, execute(files),
                      write_vendor(tree), python()

``ExternalWorkspace.orig_src`` holds only what a system may see: interface
stubs of the benchmark's own interface or skeleton files, or -- when the
benchmark gives only a natural-language spec -- the interfaces the backbone
planned from it (``plan`` + ``apply_plan``, the requirements step of the
activation prompt). Construction's DECOMPOSE then builds its ``Requirement``
list from that text exactly as it does from a LEGO-REPO stub, so no pipeline
stage needs to know the task is external. The upstream sources
(``ExternalTask.original_sources``) are read only by the library filter and the
coverage audit, never by a system.

``load_tasks`` reads a benchmark's task list with the loader named in
benchmarks.yaml. The ``generic`` loader is a template (one JSON/YAML record per
task, keys mapped through ``fields``); it has to be pointed at -- or replaced
for -- each benchmark's actual release format.
"""

from __future__ import annotations

import ast
import glob
import importlib
import json
import os
import re
import shlex
import shutil
from dataclasses import dataclass, field

import yaml

from lego.harness import core
from lego.harness.interface import internal_imports, stub
from lego.harness.prepare import ExecResult
from lego.pipeline.decompose import (Requirement, build_requirements,
                                     heuristic_query)

BENCH_FILE = os.environ.get("LEGO_EXTERNAL", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmarks.yaml"))
_ACT_PROMPT = open(os.path.join(os.path.dirname(os.path.abspath(core.__file__)),
                                "..", "prompts", "activation.md")).read()
MAX_PLAN_MODULES = int(os.environ.get("LEGO_MAX_PLAN_MODULES", 80))
SPEC_CHARS = int(os.environ.get("LEGO_SPEC_CHARS", 40000))
_PLACEHOLDER = re.compile(r"^<[^<>]*>$")
_COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", "*.pyc",
                                      ".pytest_cache")
_TASK_KEYS = ("name", "spec", "spec_file", "workdir", "target_package",
              "target_files", "stubs", "upstream_repo", "src_prefix", "tests",
              "original_sources", "canonical_name", "forks", "grader_cmd",
              "feedback_cmd")


@dataclass
class ExternalTask:
    name: str
    spec_text: str = ""
    workdir: str = ""                 # starting tree (may be empty / absent)
    target_files: list = field(default_factory=list)   # package-relative .py
    target_package: str = ""          # import name; planned if empty
    upstream_repo: str = ""           # URL or org/name, for library filtering
    grader_cmd: str = ""              # the benchmark's own grader
    benchmark: str = ""
    src_prefix: str = "."
    tests: str = ""                   # tests visible under the protocol
    feedback_cmd: str = ""            # test command allowed during construction
    stubs: dict = field(default_factory=dict)   # module -> interface text
    original_sources: str = ""        # upstream sources: filter/coverage only
    extra: dict = field(default_factory=dict)
    band: int = 0
    domain: str = ""
    track: str = ""

    @property
    def clone(self) -> str:           # what CodeFace.view reads as the repo
        return self.upstream_repo

    def statement(self) -> str:
        return self.spec_text


# ---------------------------------------------------------------- loading
def is_placeholder(v) -> bool:
    return isinstance(v, str) and bool(_PLACEHOLDER.match(v.strip()))


def load_benchmark(name: str, path: str | None = None) -> dict:
    path = path or BENCH_FILE
    with open(path) as fh:
        allb = yaml.safe_load(fh) or {}
    if name not in allb:
        raise SystemExit(f"benchmark {name!r} not in {path}; known: "
                         f"{sorted(allb)}")
    spec = dict(allb[name])
    spec["name"], spec["_file"] = name, path
    return spec


def require(spec: dict, *keys: str) -> None:
    """Exit with a clear message if a field is still a placeholder."""
    bad = [k for k in keys if not spec.get(k) or is_placeholder(spec.get(k))]
    if bad:
        raise SystemExit(
            f"benchmark {spec['name']}: {', '.join(bad)} not set in "
            f"{spec.get('_file')}; wire it to the benchmark's release first "
            f"(see external/README.md)")


def _get(row: dict, dotted: str):
    cur = row
    for part in str(dotted).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _read_records(path: str) -> list[dict]:
    text = open(path, errors="ignore").read()
    if path.endswith(".jsonl"):
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    obj = json.loads(text) if path.endswith(".json") else yaml.safe_load(text)
    return [x for x in (obj if isinstance(obj, list) else [obj])
            if isinstance(x, dict)]


def _generic_rows(spec: dict):
    root = spec["release_path"]
    for p in sorted(glob.glob(os.path.join(root, spec["task_glob"]),
                              recursive=True)):
        if os.path.isdir(p):
            f = next((os.path.join(p, n) for n in
                      ("task.json", "task.yaml", "task.yml")
                      if os.path.exists(os.path.join(p, n))), None)
            if f is None:
                yield {"_base": p, "name": os.path.basename(p)}
                continue
            base = p
        else:
            f, base = p, os.path.dirname(p)
        for row in _read_records(f):
            row.setdefault("_base", base)
            yield row


def _task_from_row(row: dict, spec: dict) -> ExternalTask:
    fm = {k: k for k in _TASK_KEYS}
    fm.update(spec.get("fields") or {})

    def g(k):
        v = _get(row, fm.get(k, k))
        return None if is_placeholder(v) else v
    base = row.get("_base") or spec.get("release_path") or "."

    def path(v):
        return (v if not v or os.path.isabs(str(v))
                else os.path.normpath(os.path.join(base, str(v))))
    text = str(g("spec") or "")
    if g("spec_file"):
        text = open(path(g("spec_file")), errors="ignore").read()
    targets = g("target_files") or []
    extra = {k: v for k, v in row.items() if k not in fm.values()
             and not k.startswith("_")}
    if g("canonical_name"):
        extra["canonical_name"] = str(g("canonical_name"))
    if g("forks"):
        extra["forks"] = list(g("forks"))
    return ExternalTask(
        name=str(g("name") or os.path.basename(base)), spec_text=text,
        workdir=path(g("workdir") or ""),
        target_files=[targets] if isinstance(targets, str) else list(targets),
        target_package=str(g("target_package") or ""),
        upstream_repo=str(g("upstream_repo") or ""),
        src_prefix=str(g("src_prefix") or "."), tests=str(g("tests") or ""),
        stubs=dict(g("stubs") or {}),
        original_sources=path(g("original_sources") or ""),
        grader_cmd=str(g("grader_cmd") or ""),
        feedback_cmd=str(g("feedback_cmd") or ""), extra=extra)


def read_pairs(path: str | None) -> list[tuple[str, str]]:
    """``a<TAB>b`` (or whitespace-separated) pairs, ``#`` comments allowed."""
    if not path or is_placeholder(path) or not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        parts = line.split("#")[0].split()
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))
    return out


def load_tasks(spec: dict) -> list[ExternalTask]:
    loader = spec.get("loader") or "generic"
    if loader == "generic":
        require(spec, "release_path", "task_glob")
        rows = list(_generic_rows(spec))
    else:
        mod, _, fn = loader.partition(":")
        rows = list(getattr(importlib.import_module(mod), fn or "load")(spec))
    ups = dict(read_pairs(spec.get("upstream_repos_file")))
    osrc = spec.get("original_sources")
    out = []
    for row in rows:
        t = row if isinstance(row, ExternalTask) else _task_from_row(row, spec)
        t.benchmark = spec["name"]
        t.upstream_repo = t.upstream_repo or ups.get(t.name, "")
        for k in ("grader_cmd", "feedback_cmd"):
            v = getattr(t, k) or spec.get(k) or ""
            setattr(t, k, "" if is_placeholder(v) else str(v))
        if not t.original_sources and osrc and not is_placeholder(osrc):
            t.original_sources = str(osrc).format(
                name=t.name, release_path=spec.get("release_path", ""))
        out.append(t)
    return out


def target_sources(task: ExternalTask) -> dict[str, str]:
    """The upstream (original) sources of an external target, keyed by path
    relative to the package directory (or to the sources root). Used by the
    library filter and the coverage audit only."""
    root = task.original_sources
    if not root or not os.path.exists(root):
        return {}
    if os.path.isfile(root):
        return {os.path.basename(root): open(root, errors="ignore").read()}
    base = root
    if task.target_package:
        cand = os.path.join(root, task.src_prefix or ".", task.target_package)
        if os.path.isdir(cand):
            base = cand
    return {m: open(os.path.join(base, m), errors="ignore").read()
            for m in core.package_modules(base)}


# ---------------------------------------------------------------- workspace
def render(template: str, **fields) -> list[str]:
    """Split a command template after shell-quoting every field."""
    q = {k: shlex.quote(str(v)) for k, v in fields.items()}
    return shlex.split(template.format(**q))


def _count(pat: str, text: str) -> int:
    m = re.findall(pat, text)
    return int(m[-1]) if m else 0


class ExternalWorkspace:
    """One external task prepared for construction (see module docstring)."""

    def __init__(self, task: ExternalTask, tdir: str | None = None):
        self.task, self.tdir = task, tdir
        # without a starting tree, point at a path under which nothing exists
        self.base = (os.path.join(tdir, "base") if tdir
                     else task.workdir or os.devnull)
        self.repo_dir = self.proj_dir = self.base
        self.src_prefix = task.src_prefix or "."
        self.pkg = task.target_package
        self.test_path = task.tests
        self.orig_src: dict[str, str] = {}
        self.modules: list[str] = []
        self.record = {"benchmark": task.benchmark, "deps": [],
                       "single_module": False, "interface_source": None}
        self.plan: dict | None = None
        self.own_env = False
        self._load_interface(task.workdir)

    @property
    def pkg_dir(self) -> str:
        return os.path.normpath(os.path.join(self.base, self.src_prefix,
                                             self.pkg))

    def _load_interface(self, root: str):
        """Stubs shipped by the benchmark, else stubs of its skeleton files."""
        pkg_dir = (os.path.join(root, self.src_prefix, self.pkg)
                   if root and self.pkg else "")
        mods = list(self.task.target_files) or sorted(self.task.stubs)
        if not mods and pkg_dir and os.path.isdir(pkg_dir):
            mods = core.package_modules(pkg_dir)
        for m in mods:
            src = self.task.stubs.get(m)
            p = os.path.join(pkg_dir, m) if pkg_dir else ""
            if src is None and p and os.path.isfile(p):
                src = open(p, errors="ignore").read()
            self.orig_src[m] = stub(src or "")
        self.modules = sorted(mods)
        if mods:
            self.record["interface_source"] = ("benchmark" if self.task.stubs
                                               else "skeleton")

    def prepare(self, env: dict | None = None) -> "ExternalWorkspace":
        """Copy the starting tree and build the task environment (as
        ``lego.harness.prepare`` does, unless the benchmark provides one)."""
        env = env or {}
        core.rm_rf(self.tdir)
        wd = self.task.workdir
        if wd and os.path.isdir(wd):
            shutil.copytree(wd, self.base, symlinks=True, ignore=_COPY_IGNORE)
        else:
            os.makedirs(self.base)
        self.record["deps"] = sorted(core.detect_deps(self.base))
        if env.get("venv") and not is_placeholder(env["venv"]):
            core.VENV = env["venv"]
            self.record["env_setup"] = "provided"
        else:
            self.own_env = True
            self.record["python"], self.record["python_why"] = \
                core.fresh_venv(self.base)
            _ok, self.record["env_setup"] = core.setup_env(
                self.base, exclude_pkg=self.pkg or None)
            if self.pkg:
                core.uninstall_target(self.base, self.pkg)
        for req in env.get("requirements") or []:
            if not is_placeholder(req):
                core.sh([core.py(), "-m", "pip", "install", "-q", "-r", req],
                        core.PIP_TIMEOUT)
        return self

    def materialize(self, files: dict[str, str], dst: str) -> str:
        """The starting tree with ``files`` written into the target package."""
        core.rm_rf(dst)
        shutil.copytree(self.base, dst, symlinks=True, ignore=_COPY_IGNORE)
        zsrc = dst if self.src_prefix == "." else os.path.join(dst,
                                                               self.src_prefix)
        zpkg = os.path.join(zsrc, self.pkg) if self.pkg else zsrc
        for m, code in files.items():
            p = os.path.join(zpkg, m)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write((code or "").rstrip() + "\n")
        return dst

    def execute(self, files: dict[str, str]) -> ExecResult:
        zero = self.materialize(files, os.path.join(self.tdir, "zero"))
        zsrc = zero if self.src_prefix == "." else os.path.join(zero,
                                                                self.src_prefix)
        zpkg = os.path.join(zsrc, self.pkg) if self.pkg else zsrc
        gen = {os.path.realpath(os.path.join(zpkg, m)) for m in files}
        prov = os.path.join(self.tdir, "prov")
        os.makedirs(prov, exist_ok=True)
        with open(os.path.join(prov, "cf_prov.py"), "w") as fh:
            fh.write(core._PROV_PLUGIN)
        seen_p, tests_p = (os.path.join(prov, "seen.json"),
                           os.path.join(prov, "tests.json"))
        for p in (seen_p, tests_p):
            if os.path.exists(p):
                os.remove(p)
        env = core.clean_env(
            PYTHONPATH=os.pathsep.join([zsrc, *core.EXTRA_PYTHONPATH, prov]),
            CF_PROV_PKG=self.pkg, CF_PROV_OUT=seen_p, CF_TESTS_OUT=tests_p,
            PYTEST_ADDOPTS="-p cf_prov -p no:cacheprovider")
        if self.pkg:
            imp = core.sh([core.py(), "-c", f"import {self.pkg}"], 25, env=env,
                          cwd=zero)
            if not imp or imp.returncode != 0:
                err = (imp.stderr if imp else "import timeout") or ""
                return ExecResult(False, 0, 0, err[-600:], "import-failed")
        if not self.task.feedback_cmd:
            return ExecResult(True, 0, 0, "(no test feedback is available "
                              "during construction under this benchmark's "
                              "protocol)", "ok")
        argv = render(self.task.feedback_cmd, workdir=zero, python=core.py(),
                      tests=self.test_path or ".")
        r = core.sh(argv, core.TEST_TIMEOUT, env=env, cwd=zero)
        if r is None:
            return ExecResult(True, 0, 0, "test command timeout", "timeout")
        out = (r.stdout + r.stderr).strip()
        tests = {}
        if os.path.exists(tests_p):
            try:
                tests = json.load(open(tests_p))
            except (OSError, json.JSONDecodeError):
                tests = {}
        verdict, exercised = "ok", 0
        if os.path.exists(seen_p):       # a pytest run: same provenance rule
            try:
                seen = json.load(open(seen_p))
            except (OSError, json.JSONDecodeError):
                seen = {}
            pv = core.check_provenance(seen, zsrc, gen, os.path.join(
                zero, self.test_path) if self.test_path else None)
            verdict, exercised = pv["verdict"], pv["exercised"]
        return ExecResult(True, _count(r"(\d+) passed", out),
                          _count(r"(\d+) failed", out)
                          + _count(r"(\d+) errors?\b", out),
                          out[-1500:], verdict, exercised, tests)

    def write_vendor(self, vendor: dict[str, str]) -> str:
        vdir = os.path.join(self.tdir, "vendor_root")
        core.rm_rf(vdir)
        for rel, code in vendor.items():
            p = os.path.join(vdir, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(code)
        return vdir

    def python(self) -> str:
        return core.py()

    def cleanup(self):
        if self.tdir:
            core.rm_rf(self.tdir)
        if self.own_env:
            core.rm_rf(core.VENV)


def feedback_score(res: ExecResult) -> dict:
    """Round selection inside Construction for external tasks: the fraction of
    feedback tests passing. Not a benchmark metric; that comes from the
    benchmark's own grader."""
    tot = res.passed + res.failed
    raw = res.passed / tot if tot else 0.0
    return {"passed": res.passed, "score": round(raw, 4), "raw": round(raw, 4)}


feedback_score.bounds = {}


# ---------------------------------------------------------------- planning
_IDENT = re.compile(r"[^A-Za-z0-9_]")


def _module_path(target: str, pkg: str) -> str:
    t = str(target).strip().strip("`").replace("\\", "/")
    while t.startswith(("./", "/")):
        t = t[1:] if t.startswith("/") else t[2:]
    if pkg and t.startswith(pkg + "/"):
        t = t[len(pkg) + 1:]
    if not t.endswith(".py"):
        if pkg and (t == pkg or t.startswith(pkg + ".")):
            t = t[len(pkg) + 1:] or "__init__"
        t = t.replace(".", "/") + ".py"
    return t


def plan_prompt(task: ExternalTask, pkg: str = "") -> str:
    want = f"`{pkg}`" if pkg else "(choose its import name)"
    return (
        f"{_ACT_PROMPT}\n\n---\n\n# Construction Request\n"
        f"Construct the Python package {want} for `{task.name}` from the "
        "specification below. No target interface is given: planning the "
        "package's modules and their public interfaces is part of this step."
        f"\n\n{task.spec_text[:SPEC_CHARS]}\n\n"
        "For THIS STEP return only the `requirements` list of the output "
        "format, plus a top-level `package` field with the import name. One "
        f"entry per module of the package (at most {MAX_PLAN_MODULES}; include "
        "`__init__.py`): `id` r1, r2, ...; `target` the module path relative "
        "to the package directory (e.g. `io/readers.py`); `interface` the "
        "module's public interface as Python source -- imports, docstrings, "
        "class and function signatures with docstrings, every body `...`; "
        "`dependencies` the other planned modules it imports (by `target`) "
        "and the third-party distributions it may use; `capability` one "
        "sentence describing the reusable functionality it needs (not its "
        "name); `retrieval_request` a short query for CodeFace.")


def plan(llm, task: ExternalTask, pkg: str = "") -> dict:
    """Spec-driven decomposition: the backbone plans modules + interfaces."""
    obj = llm.json(plan_prompt(task, pkg), max_tokens=16000) or {}
    return normalize_plan(obj, pkg)


def normalize_plan(obj, pkg: str = "") -> dict:
    obj = obj if isinstance(obj, dict) else {}
    pkg = pkg or _IDENT.sub("_", str(obj.get("package") or "").strip()) or "pkg"
    if pkg[0].isdigit():
        pkg = "_" + pkg
    mods = {}
    for x in obj.get("requirements") or []:
        if not isinstance(x, dict) or not x.get("target"):
            continue
        m = _module_path(x["target"], pkg)
        if m in mods or len(mods) >= MAX_PLAN_MODULES:
            continue
        mods[m] = {"interface": stub(str(x.get("interface") or "")),
                   "dependencies": [str(d) for d in x.get("dependencies") or []
                                    if d][:40],
                   "capability": str(x.get("capability") or "")[:400],
                   "retrieval_request": str(x.get("retrieval_request") or "")[:300]}
    if mods and "__init__.py" not in mods:
        mods["__init__.py"] = {"interface": "", "dependencies": [],
                               "capability": "", "retrieval_request": ""}
    return {"package": pkg, "modules": mods}


def _find(dep: str, names: set, pkg: str) -> str | None:
    m = _module_path(dep, pkg)
    for cand in (m, m[:-3] + "/__init__.py"):
        if cand in names:
            return cand
    return None


def _insert_after_docstring(src: str, lines: list[str]) -> str:
    if not lines:
        return src
    at = 0
    try:
        body = ast.parse(src).body
        if body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant) and isinstance(
                body[0].value.value, str):
            at = body[0].end_lineno
    except SyntaxError:
        pass
    ls = src.splitlines()
    return "\n".join(ls[:at] + lines + ls[at:]) + "\n"


def apply_plan(ws: ExternalWorkspace, p: dict) -> None:
    """Make a plan the workspace's visible interface. Planned dependencies on
    sibling modules become import lines, so build_requirements derives the
    same dependency order it derives from a LEGO-REPO stub."""
    ws.pkg = ws.pkg or p["package"]
    mods = p["modules"]
    names = set(mods)
    ext = set(ws.record.get("deps") or [])
    ws.orig_src, ws.modules = {}, sorted(names)
    for m, x in mods.items():
        iface = x.get("interface") or ""
        have = {r for d in internal_imports(iface, ws.pkg, m)
                if (r := _find(d, names, ws.pkg))}
        add = []
        for d in x.get("dependencies") or []:
            r = _find(d, names, ws.pkg)
            if r is None:
                top = re.split(r"[^A-Za-z0-9_\-]", d.strip())[0]
                if top and top != ws.pkg:
                    ext.add(top.lower().replace("-", "_"))
            elif r != m and r != "__init__.py" and r not in have:
                have.add(r)
                dotted = r[:-3].replace("/", ".")
                dotted = dotted[:-len(".__init__")] if dotted.endswith(
                    ".__init__") else dotted
                add.append(f"import {ws.pkg}.{dotted}")
        ws.orig_src[m] = _insert_after_docstring(iface, add)
    ws.record["deps"] = sorted(ext)
    ws.record["interface_source"] = "planned"
    ws.plan = p


def requirements(ws: ExternalWorkspace) -> list[Requirement]:
    """The Requirement list DECOMPOSE derives from the workspace's interface;
    capability text from the plan when there is one, else from identifiers."""
    reqs = build_requirements(ws, "stub")
    mods = (ws.plan or {}).get("modules") or {}
    for r in reqs:
        x = mods.get(r.module) or {}
        r.capability = x.get("capability") or heuristic_query(r, ws.pkg)
        r.retrieval_request = x.get("retrieval_request") or r.capability
    return reqs
