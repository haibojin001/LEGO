"""Execution harness: clone, environment, suite execution, provenance.

Everything a configuration needs to *measure* a reconstruction lives here; nothing
in this module calls a model. The construction pipeline (``lego.pipeline``) writes
a dict ``{module_path: source}`` and hands it to ``write_tree`` / ``run_suite``.

Module-level settings (``VENV``, ``CLONE_BASE``, timeouts) are assigned by
``lego.run.configure_harness`` before any task runs.
"""

import argparse
import ast
import collections
import concurrent.futures as cf
import fnmatch
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

from lego.benchmark_meta import HELD_REPOS

BASE = os.environ.get("LEGO_RUNS", "runs")

PINS_PATH = os.environ.get("LEGO_PINS", "benchmark/pins.json")


def _load_pins(path=None):
    try:
        with open(path or PINS_PATH) as fh:
            return json.load(fh).get("pins", {})
    except Exception:
        return {}


PINS = _load_pins()

BENCH_OUT = os.environ.get("LEGO_BENCH_OUT") or ""
VENV = "/tmp/repo_bench_venv"
CLONE_BASE = BASE            # per-run clone dir; retagged in main() for parallel arms
CLONE_TIMEOUT = int(os.environ.get("LEGO_CLONE_TIMEOUT", 900))
PIP_TIMEOUT = int(os.environ.get("LEGO_PIP_TIMEOUT", 600))
HEAVY_PIP_TIMEOUT = int(os.environ.get("LEGO_HEAVY_PIP_TIMEOUT", 2400))
TEST_TIMEOUT = int(os.environ.get("LEGO_TEST_TIMEOUT", 900))

MAX_MODULES = int(os.environ.get("LEGO_MAX_MODULES", 300))

MIN_SPAN = int(os.environ.get("LEGO_MIN_SPAN", 5))

GEN_WORKERS = int(os.environ.get("LEGO_GEN_WORKERS", 10))

HARNESS_REV = "r3"

HOLD_REPOS = {r for r in os.environ.get(
    "LEGO_HOLD_REPOS", ",".join(sorted(HELD_REPOS))).split(",")
    if r.strip()}

SKIP_HEAVY = False
GATE_ONLY = False
TIMEOUT_ARGS = []
# Directories placed on PYTHONPATH after the generated source root (used by the
# Import+Call control, whose vendored primitives live outside the target prefix).
EXTRA_PYTHONPATH: list = []

HARVEST_MIN_RATE = 0.5

_NON_TRANSFERABLE = re.compile(
    r"^(__init__|__main__|__version__|_?version|_?about|setup|conftest|"
    r"cli|main|app|app_factory|desktop_main|settings|config|constants|"
    r"globals|compat|_compat|py3?compat)\.py$", re.I)
_MIN_HARVEST_LINES = 15

HEAVY = re.compile(r"(torch|pytorch|tensorflow|transformers|vllm|cuda|jax|"
                   r"paddle|onnxruntime|diffusers|flash.?attn|deepspeed|"
                   r"whisper|llama.?cpp|bitsandbytes|xformers|triton)", re.I)


def sh(cmd, timeout, env=None, cwd=None):
    """Run a command with a credential-free environment by default."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=clean_env() if env is None else env, cwd=cwd)
    except subprocess.TimeoutExpired:
        return None


_SECRET_ENV = re.compile(
    r"^(OPENAI_|ANTHROPIC_|AWS_|LEGO_LLM_|AZURE_|GOOGLE_|GEMINI_|"
    r"HF_TOKEN|HUGGING|COHERE_|MISTRAL_|GROQ_|TOGETHER_|REPLICATE_|"
    r"SLURM_.*KEY|.*_API_KEY$|.*_SECRET.*|.*_TOKEN$|.*PASSWORD.*)", re.I)


def clean_env(**overrides):
    """os.environ minus credentials and runner-only pytest flags."""
    env = {k: v for k, v in os.environ.items()
           if not _SECRET_ENV.match(k)
           and k != "PYTEST_DISABLE_PLUGIN_AUTOLOAD"}
    env.update(overrides)
    return env


def rm_rf(path):
    """Delete a tree, never raising."""
    if not os.path.exists(path):
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass
    if os.path.exists(path):
        subprocess.run(["rm", "-rf", path], check=False)


_PY_CANDIDATES = ("3.13", "3.12", "3.11", "3.10", "3.9", "3.8")
_PY_DEFAULT = "3.12"

_REQ_PY_RE = re.compile(
    r"""(?:requires[-_]python|python_requires)\s*[=:]\s*["']?"""
    r"""([<>=!~,.\s\d*]+?)["']?\s*(?:\n|$|,\s*\n)""")
_CLASSIFIER_TRUST_MAX = "3.10"
_CLASSIFIER_RE = re.compile(
    r"Programming Language :: Python :: (\d+\.\d+)")


def _vt(v):
    return tuple(int(x) for x in v.split(".")[:2])


def _satisfies(ver, spec):
    """Does Python `ver` ("3.8") satisfy a PEP 440 spec (">=3.7,<3.9")?"""
    for clause in (c.strip() for c in spec.split(",")):
        m = re.match(r"(==|!=|>=|<=|~=|>|<)?\s*v?([0-9][0-9.*]*)", clause)
        if not m:
            continue
        op, raw = m.group(1) or "==", m.group(2).rstrip(".*")
        if not raw:
            continue
        try:
            want = _vt(raw)
        except ValueError:
            continue
        got = _vt(ver)
        if len(want) < 2:
            if op in (">=", ">", "~=") and got[0] < want[0]:
                return False
            if op in ("<=", "<") and got[0] > want[0]:
                return False
            continue
        if op == "==" and got != want: return False
        if op == "!=" and got == want: return False
        if op == ">=" and got < want: return False
        if op == ">" and got <= want: return False
        if op == "<=" and got > want: return False
        if op == "<" and got >= want: return False
        if op == "~=" and (got < want or got[0] != want[0]): return False
    return True


def pick_python(repo_dir):
    """The newest interpreter this repo says it supports."""
    spec, classifiers = None, []
    for fn in ("pyproject.toml", "setup.py", "setup.cfg"):
        fp = os.path.join(repo_dir, fn)
        if not os.path.exists(fp):
            continue
        txt = open(fp, errors="ignore").read()
        if spec is None:
            m = _REQ_PY_RE.search(txt)
            if m:
                spec = m.group(1).strip()
        classifiers += _CLASSIFIER_RE.findall(txt)
    if spec:
        for v in _PY_CANDIDATES:
            if _satisfies(v, spec):
                return v, f"requires-python {spec}"
        return _PY_DEFAULT, f"requires-python {spec} unsatisfiable"
    if classifiers:
        allowed = {c for c in classifiers if c in _PY_CANDIDATES}
        if allowed:
            best = max(allowed, key=_vt)
            if _vt(best) <= _vt(_CLASSIFIER_TRUST_MAX):
                return best, f"classifiers<={best}"
    return _PY_DEFAULT, "default"


def fresh_venv(repo_dir=None):
    """A CLEAN venv per repo — bounds memory/disk to one repo's deps and stops
    cross-repo dependency pollution (a correctness win too)."""
    rm_rf(VENV)
    ver, why = pick_python(repo_dir) if repo_dir else (_PY_DEFAULT, "no-repo")
    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    if os.path.exists(uv):
        for v in (ver, _PY_DEFAULT):
            sh([uv, "python", "install", v], 420)
            r = sh([uv, "venv", "--seed", "--python", v, VENV], 240)
            if r and r.returncode == 0 and os.path.exists(f"{VENV}/bin/pip"):
                sh([f"{VENV}/bin/pip", "install", "-q", "--upgrade",
                    "pip", "setuptools", "wheel"], 180)
                return v, why if v == ver else f"{why}->fallback{v}"
            rm_rf(VENV)
    rm_rf(VENV)
    sh([sys.executable, "-m", "venv", VENV], 120)
    if not os.path.exists(f"{VENV}/bin/pip"):
        sh([f"{VENV}/bin/python", "-m", "ensurepip", "--upgrade"], 120)
    if not os.path.exists(f"{VENV}/bin/pip"):
        raise RuntimeError(f"no usable venv at {VENV}: uv and stdlib venv both "
                           f"failed to produce bin/pip")
    sh([f"{VENV}/bin/pip", "install", "-q", "--upgrade",
        "pip", "setuptools", "wheel"], 120)
    return "system", "no-uv"


def py():
    return f"{VENV}/bin/python"


_NOT_PKG = {"docs", "doc", "examples", "example", "scripts", "benchmarks",
            "bench", "tools", "build", "dist", "data", "assets", "images",
            "notebooks", "site", "www", "static", "ci", "contrib",
            "requirements", "benchmark", "stubs", "typings", "third_party",
            "vendor", "vendored", "misc", "utils_scripts", "devtools",
            "tutorials", "demo", "demos", "samples", "conda", "packaging",
            "python", "py", "lib", "libs", "packages", "projects",
            "components", "bindings", "src"}


def _pkg_dirs_under(root):
    """Candidate package dirs under `root`, ranked by .py count."""
    out = []
    try:
        entries = os.listdir(root)
    except OSError:
        return out
    for d in sorted(entries):
        full = os.path.join(root, d)
        if (not os.path.isdir(full) or d.startswith((".", "test"))
                or d.lower() in _NOT_PKG):
            continue
        try:
            files = os.listdir(full)
        except OSError:
            continue
        n_py = sum(1 for f in files if f.endswith(".py"))
        if n_py == 0:
            continue
        has_init = "__init__.py" in files
        out.append(((1 if has_init else 0), n_py, d))
    return [(n_py, d) for _init, n_py, d in sorted(out, reverse=True)]


def declared_package(proj_dir):
    """The package/dist name the project declares for itself, if any."""
    for fn, pats in (("pyproject.toml",
                      (r'(?m)^\s*name\s*=\s*["\']([A-Za-z0-9_.\-]+)',
                       r'packages\s*=\s*\[\s*["\']([A-Za-z0-9_.]+)')),
                     ("setup.cfg", (r'(?m)^\s*name\s*=\s*([A-Za-z0-9_.\-]+)',)),
                     ("setup.py", (r'(?m)^\s*name\s*=\s*["\']([A-Za-z0-9_.\-]+)',))):
        p = os.path.join(proj_dir, fn)
        if not os.path.exists(p):
            continue
        txt = open(p, errors="ignore").read()
        for pat in pats:
            m = re.search(pat, txt)
            if m:
                return m.group(1)
    return None


_DEP_NAME_RE = re.compile(r"^\s*([A-Za-z][\w.-]+)\s*(?:[<>=!~\[;].*)?$", re.M)


def declared_dep_names(repo_dir):
    """Import-style names of the repo's own DECLARED dependencies."""
    names = set()
    texts = []
    for fn in ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"):
        p = os.path.join(repo_dir, fn)
        if os.path.exists(p):
            texts.append(open(p, errors="ignore").read())
    for pat in ("requirements/*.txt", "requirements-*.txt"):
        for p in glob.glob(os.path.join(repo_dir, pat))[:12]:
            texts.append(open(p, errors="ignore").read())
    for txt in texts:
        for m in _DEP_NAME_RE.finditer(txt):
            raw = m.group(1).strip().strip("\"',")
            if raw and not raw.startswith("-"):
                names.add(raw.replace("-", "_").replace(".", "_").lower())
    return names


def detect_package(repo_dir):
    """Find (src_prefix, pkg). Handles flat, src-layout, AND monorepos
    (libs/core/pkg, packages/x/pkg, python/pkg) by searching prefixes up to
    two dirs deep for a package dir whose tree also carries a pyproject/setup."""
    prefixes = [".", "src", "lib", "python", "Python", "py", "bindings/python"]
    for container in ("libs", "packages", "projects", "components", "bindings"):
        c = os.path.join(repo_dir, container)
        if os.path.isdir(c):
            for sub in sorted(os.listdir(c)):
                if os.path.isdir(os.path.join(c, sub)):
                    prefixes += [os.path.join(container, sub),
                                 os.path.join(container, sub, "src")]
    try:
        for d in sorted(os.listdir(repo_dir)):
            full = os.path.join(repo_dir, d)
            if (os.path.isdir(full) and not d.startswith(".")
                    and any(os.path.exists(os.path.join(full, b)) for b in
                            ("pyproject.toml", "setup.py", "setup.cfg"))):
                prefixes += [d, os.path.join(d, "src")]
    except OSError:
        pass
    prefixes = list(dict.fromkeys(prefixes))
    deps = declared_dep_names(repo_dir)
    best = None  # ((has_build, declared, own, n_py), prefix, pkg)
    for prefix in prefixes:
        root = repo_dir if prefix == "." else os.path.join(repo_dir, prefix)
        for n_py, d in _pkg_dirs_under(root):
            proj = root
            has_build = any(os.path.exists(os.path.join(proj, b))
                            for b in ("pyproject.toml", "setup.py", "setup.cfg"))
            decl = declared_package(project_dir(repo_dir, prefix)) or ""
            is_declared = d.lower() == decl.lower().replace("-", "_")
            is_own = is_declared or d.lower() not in deps
            key = (has_build, is_declared, is_own, n_py)
            if best is None or key > best[0]:
                best = (key, prefix, d)
    if best:
        return best[1], best[2]
    decl = (declared_package(repo_dir) or "").replace("-", "_").lower()
    for prefix in (".", "src"):
        root = repo_dir if prefix == "." else os.path.join(repo_dir, prefix)
        if not os.path.isdir(root):
            continue
        mods = [f[:-3] for f in sorted(os.listdir(root))
                if f.endswith(".py") and not is_test_file(f)
                and f not in ("setup.py", "conftest.py", "__init__.py")]
        if not mods:
            continue
        return prefix, next((m for m in mods if m.lower() == decl), mods[0])
    return None, None


def project_dir(repo_dir, src_prefix):
    """The dir to run `pip install -e .` in — the nearest one (repo root or the
    monorepo subproject) that has a build file."""
    cand = repo_dir if src_prefix == "." else os.path.join(repo_dir, src_prefix)
    d = cand
    while True:
        if any(os.path.exists(os.path.join(d, b))
               for b in ("pyproject.toml", "setup.py", "setup.cfg")):
            return d
        if os.path.abspath(d) == os.path.abspath(repo_dir):
            return repo_dir
        parent = os.path.dirname(d)
        if parent == d:
            return repo_dir
        d = parent


def pytest_testpaths(repo_dir):
    """`testpaths` as declared in the repo's OWN pytest config."""
    out = []
    for fn, pat in (("pyproject.toml", r"testpaths\s*=\s*\[?([^\]\n]+)"),
                    ("setup.cfg", r"testpaths\s*=\s*(.+)"),
                    ("tox.ini", r"testpaths\s*=\s*(.+)"),
                    ("pytest.ini", r"testpaths\s*=\s*(.+)")):
        p = os.path.join(repo_dir, fn)
        if not os.path.exists(p):
            continue
        m = re.search(pat, open(p, errors="ignore").read())
        if m:
            out += [t.strip().strip("\"'") for t in
                    re.split(r"[,\s]+", m.group(1)) if t.strip()]
    return out


_TEST_FILE = re.compile(r"^(test_.*\.py|.*_tests?\.py|tests?\.py)$", re.I)


def pytest_python_files(repo_dir):
    """Extra test-file globs the repo declares via pytest `python_files`."""
    pats = []
    for fn in ("pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini"):
        p = os.path.join(repo_dir, fn)
        if not os.path.exists(p):
            continue
        m = re.search(r"python_files\s*=\s*\[?([^\]\n]+)",
                      open(p, errors="ignore").read())
        if m:
            pats += [t.strip().strip("\"'") for t in
                     re.split(r"[,\s]+", m.group(1)) if t.strip()]
    return pats


def is_test_file(fname, extra_pats=()):
    if _TEST_FILE.match(fname):
        return True
    return any(fnmatch.fnmatch(fname, p) for p in extra_pats)


def has_test_files(path, extra_pats=(), cap=4000):
    """Does this path hold anything pytest would collect as a test?"""
    if os.path.isfile(path):
        return is_test_file(os.path.basename(path), extra_pats)
    if not os.path.isdir(path):
        return False
    seen = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".tox")]
        if any(is_test_file(f, extra_pats) for f in files):
            return True
        seen += len(files)
        if seen > cap:
            break
    return False


def _walk_test_dirs(repo_dir, cap=400, extra_pats=()):
    """Every directory holding test files, relative to `repo_dir`."""
    hits = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", ".tox",
                                                ".venv", "venv", "__pycache__",
                                                "build", "dist", "docs")]
        if any(is_test_file(f, extra_pats) for f in files):
            hits.append(os.path.relpath(root, repo_dir))
        if len(hits) >= cap:
            break
    return hits


def detect_tests(repo_dir, pkg_dir):
    pats = pytest_python_files(repo_dir)
    for cand in pytest_testpaths(repo_dir):
        if has_test_files(os.path.join(repo_dir, cand), pats):
            return cand
    for cand in ("tests", "test", "Tests", os.path.join(pkg_dir, "test"),
                 os.path.join(pkg_dir, "tests"), os.path.join("src", "tests")):
        if has_test_files(os.path.join(repo_dir, cand), pats):
            return cand
    tops = sorted(f for f in os.listdir(repo_dir) if is_test_file(f, pats))
    if tops:
        return tops[0]
    found = _walk_test_dirs(repo_dir, extra_pats=pats)
    if not found:
        return None
    anc = os.path.commonpath(found) if len(found) > 1 else found[0]
    return anc if anc not in (".", "") else (pkg_dir if
                                             os.path.isdir(os.path.join(repo_dir, pkg_dir))
                                             else found[0])


_DEP_JUNK = {"python", "pytest", "setuptools", "wheel", "poetry", "hatchling",
             "coverage", "ruff", "black", "mypy", "flake8", "tox", "isort",
             "responses", "pytest-mock", "pytest-cov"}
_DEP_BAD = re.compile(r"[.\s]|shell|classifiers|source|omit|exclude|name|"
                      r"version|license|readme|homepage|repository|author|"
                      r"description|keyword|require|format|lint|test|build|"
                      r"optional|fund|url|email|depend|group|tool|project", re.I)


def detect_deps(repo_dir):
    """Runtime deps from the dependency SECTIONS only (avoid TOML-wide garbage)."""
    deps = set()
    for fn in os.listdir(repo_dir):
        if fn.startswith("requirements") and fn.endswith(".txt"):
            for line in open(os.path.join(repo_dir, fn), errors="ignore"):
                m = re.match(r"^([A-Za-z][A-Za-z0-9_\-]{1,40})", line.strip())
                if m:
                    deps.add(m.group(1).lower())
    pp = os.path.join(repo_dir, "pyproject.toml")
    if os.path.exists(pp):
        txt = open(pp, errors="ignore").read()
        for block in re.findall(r"dependencies\s*=\s*\[(.*?)\]", txt, re.S):
            for m in re.findall(r'["\']([A-Za-z][A-Za-z0-9_\-]{1,40})', block):
                deps.add(m.lower())
        pm = re.search(r"\[tool\.poetry\.dependencies\](.*?)(?:\n\[|\Z)", txt, re.S)
        if pm:
            for m in re.findall(r"^([A-Za-z][A-Za-z0-9_\-]{1,40})\s*=", pm.group(1), re.M):
                deps.add(m.lower())
    deps = {d for d in deps if d not in _DEP_JUNK and not _DEP_BAD.search(d)}
    return deps


def pip_install(pkgs):
    """Install tolerantly: one bad name must not block the rest."""
    if not pkgs:
        return
    budget = (HEAVY_PIP_TIMEOUT if any(HEAVY.search(p) for p in pkgs)
              else PIP_TIMEOUT)
    r = sh([f"{VENV}/bin/pip", "install", "-q", *pkgs], budget)
    if r and r.returncode == 0:
        return
    for p in pkgs:   # fall back to one-by-one
        sh([f"{VENV}/bin/pip", "install", "-q", p], 90)


def dist_name(repo_dir):
    """Distribution name from pyproject/setup, for pip uninstall."""
    for fn in ("pyproject.toml", "setup.cfg", "setup.py"):
        p = os.path.join(repo_dir, fn)
        if not os.path.exists(p):
            continue
        txt = open(p, errors="ignore").read()
        m = re.search(r'(?m)^\s*name\s*=\s*["\']?([A-Za-z0-9_.\-]+)', txt)
        if m:
            return m.group(1)
    return None


def site_packages():
    r = sh([f"{VENV}/bin/python", "-c",
            "import sysconfig;print(sysconfig.get_paths()['purelib'])"], 60)
    return (r.stdout.strip() if r and r.returncode == 0 else "") or ""


def uninstall_target(repo_dir, pkg):
    """Remove ONLY the target package from site-packages after baseline, so the
    reconstruction test sees the generated code (via PYTHONPATH), not the
    original. Deps stay installed."""
    sp = site_packages()
    stash = tempfile.mkdtemp(prefix="cf_distinfo_")
    saved = []
    if sp and os.path.isdir(sp):
        for nm in {dist_name(repo_dir), pkg, pkg.replace("_", "-"),
                   pkg.replace("-", "_")}:
            if not nm:
                continue
            for d in glob.glob(os.path.join(
                    sp, nm.replace("-", "_") + "-*.dist-info")):
                try:
                    shutil.copytree(d, os.path.join(stash,
                                                    os.path.basename(d)))
                    saved.append(os.path.basename(d))
                except OSError:
                    pass
    for nm in {dist_name(repo_dir), pkg, pkg.replace("_", "-")}:
        if nm:
            sh([f"{VENV}/bin/pip", "uninstall", "-y", nm], 60)
    for base in saved:
        dst = os.path.join(sp, base)
        if not os.path.exists(dst):
            try:
                shutil.copytree(os.path.join(stash, base), dst)
            except OSError:
                pass
    rm_rf(stash)


TOOLBELT = ("pytest", "responses", "pytest-mock", "pytest-asyncio",
            "pytest-cov", "pytest-timeout", "pytest-xdist", "pytest-trio",
            "pytest-tornasync", "pytest-randomly", "freezegun", "httpx",
            "trustme", "python-dotenv", "pytest-httpx", "aioresponses",
            "jsonschema", "hypothesis", "flaky", "requests-mock",
            "mock", "coverage", "trio", "anyio", "setuptools", "pip")

_REQ_GLOBS = ("requirements*test*.txt", "test*requirements*.txt",
              "requirements*dev*.txt", "dev*requirements*.txt",
              "requirements/test*.txt", "requirements/dev*.txt",
              "requirements/ci*.txt", "tests/requirements*.txt",
              "test/requirements*.txt", "requirements-testing.txt",
              "requirements.txt", "requirements/base*.txt",
              "requirements/*.txt", "requirements/main*.txt",
              "requirements/prod*.txt", "requirements/local*.txt")


def _install_req_files(repo_dir, budget=None):
    """Install every test/dev requirements file we can find — many repos put
    their REAL test deps here, not in pyproject [test] extras. Tolerant."""
    budget = budget or PIP_TIMEOUT
    seen = set()
    for pat in _REQ_GLOBS:
        for f in glob.glob(os.path.join(repo_dir, pat)):
            if f in seen:
                continue
            seen.add(f)
            sh([f"{VENV}/bin/pip", "install", "-q", "-r", f], budget)
    return len(seen)


def install_siblings(repo_root, proj_dir, exclude_pkg=None, budget=None,
                     limit=6):
    """Editable-install the sibling subprojects of a monorepo subproject."""
    budget = budget or PIP_TIMEOUT
    proj = os.path.realpath(proj_dir)
    want = exclude_pkg.replace("_", "-").lower() if exclude_pkg else None
    done = []
    for entry in ["."] + sorted(os.listdir(repo_root)):
        d = os.path.join(repo_root, entry)
        if not os.path.isdir(d) or os.path.realpath(d) == proj:
            continue
        if not any(os.path.exists(os.path.join(d, f))
                   for f in ("pyproject.toml", "setup.py")):
            continue
        if want and entry.replace("_", "-").lower() in (want, want + "-core"):
            continue
        if exclude_pkg and any(
                os.path.isdir(os.path.join(d, sub, exclude_pkg))
                for sub in (".", "src")):
            continue
        r = sh([f"{VENV}/bin/pip", "install", "-q", "-e", "."], budget, cwd=d)
        if r and r.returncode == 0:
            done.append(entry)
        if len(done) >= limit:
            break
    return done


_BUILD_BACKEND = re.compile(r"meson|scikit.build|setuptools.rust|maturin|"
                            r"pybind11|cmake|cython|Extension\(", re.I)


def needs_build(repo_dir):
    """Does this repo compile native extensions to be importable?"""
    for fn in ("pyproject.toml", "setup.py", "setup.cfg"):
        p = os.path.join(repo_dir, fn)
        if os.path.exists(p) and _BUILD_BACKEND.search(
                open(p, errors="ignore").read()):
            return True
    return False


def pip_best_effort(pkgs, budget, upgrade=True):
    """Install `pkgs`, batched, falling back to one-at-a-time. → (ok, failed)"""
    base = [f"{VENV}/bin/pip", "install", "-q"]
    r = sh(base + (["--upgrade"] if upgrade else []) + list(pkgs), budget)
    if r and r.returncode == 0:
        return list(pkgs), []
    ok, bad = [], []
    for p in pkgs:
        r = sh(base + [p], max(90, budget // 8))
        (ok if (r and r.returncode == 0) else bad).append(p)
    return ok, bad


def ensure_pytest():
    """pytest importable in the venv, whatever the interpreter. → bool"""
    def works():
        r = sh([f"{VENV}/bin/python", "-c", "import pytest"], 60)
        return bool(r and r.returncode == 0)
    if works():
        return True
    for spec in ("pytest", "pytest<8", "pytest<7", "pytest<6"):
        sh([f"{VENV}/bin/pip", "install", "-q", spec], 180)
        if works():
            return True
    return False


def setup_env(repo_dir, repo_root=None, exclude_pkg=None):
    """Build a REAL test environment (this is the true bottleneck — where the env
    is complete, reconstruction lands 49-100%; where it isn't, baseline can't
    even run). Layers, most-general last: 1. a broad common pytest + plugin
    toolbelt 2. editable install of the package with ALL known extras
    (test/dev/…) 3. every test/dev requirements*.txt file in the repo 4.
    detected deps as a fallback Returns (installed_ok, how)."""
    _ok, bad = pip_best_effort(list(TOOLBELT), 300)
    how = []
    if not ensure_pytest():
        how.append("!no-pytest")
    global TIMEOUT_ARGS
    r = sh([f"{VENV}/bin/python", "-c", "import pytest_timeout"], 60)
    TIMEOUT_ARGS = (["--timeout=180", "--timeout-method=signal"]
                    if (r and r.returncode == 0) else [])
    if bad:
        how.append("-toolbelt " + ",".join(bad[:6]))

    plugs = plugins_for_flags(_pytest_cfg_text(repo_dir))
    if plugs:
        pok, _pbad = pip_best_effort(plugs[:20], 300, upgrade=False)
        if pok:
            how.append("+plugins " + ",".join(pok))

    os.environ.pop("DJANGO_SETTINGS_MODULE", None)
    dsm = django_settings_module(repo_dir)
    if dsm:
        pip_best_effort(["django", "pytest-django"], 300, upgrade=False)
        os.environ["DJANGO_SETTINGS_MODULE"] = dsm
        how.append(f"+django({dsm})")
    heavy = (any(HEAVY.search(d) for d in detect_deps(repo_dir))
             or needs_build(repo_dir))
    budget = HEAVY_PIP_TIMEOUT if heavy else PIP_TIMEOUT
    if needs_build(repo_dir):
        sh([f"{VENV}/bin/pip", "install", "-q", "--upgrade", "ninja", "meson",
            "meson-python", "cmake", "cython", "setuptools-rust", "maturin",
            "pybind11", "scikit-build-core"], 420)
        how.append("+buildtools")
    ed_ok = False
    for spec in (".[test,tests,dev,testing,ci,all]", ".[test]", ".[tests]",
                 ".[dev]", ".[testing]", ".[all]", "."):
        r = sh([f"{VENV}/bin/pip", "install", "-q", "-e", spec], budget,
                cwd=repo_dir)
        if r and r.returncode == 0:
            ed_ok = True
            how.append(f"-e {spec}")
            break
    if not ed_ok and needs_build(repo_dir):
        r = sh([f"{VENV}/bin/pip", "install", "-q", "-e", ".",
                "--no-build-isolation"], budget, cwd=repo_dir)
        if r and r.returncode == 0:
            ed_ok = True
            how.append("-e . --no-build-isolation")
    if repo_root and os.path.realpath(repo_root) != os.path.realpath(repo_dir):
        sibs = install_siblings(repo_root, repo_dir, exclude_pkg, budget)
        if sibs:
            how.append("+siblings " + ",".join(sibs))

    nreq = _install_req_files(repo_dir, budget)
    if nreq:
        how.append(f"+{nreq} req-file(s)")
    if not ed_ok:
        deps = detect_deps(repo_dir)
        if deps:
            pip_install(list(deps)[:25])
            how.append("detected-deps")
    return ed_ok, " ".join(how) or "none"


_IMPORT_TO_DIST = {
    "yaml": "pyyaml", "cv2": "opencv-python-headless", "PIL": "pillow",
    "sklearn": "scikit-learn", "bs4": "beautifulsoup4", "dateutil": "python-dateutil",
    "attr": "attrs", "OpenSSL": "pyopenssl", "serial": "pyserial",
    "google": "protobuf", "pkg_resources": "setuptools", "zoneinfo": "backports.zoneinfo",
    "magic": "python-magic", "jwt": "pyjwt", "docx": "python-docx",
    "fitz": "pymupdf", "Crypto": "pycryptodome", "win32api": "pywin32",
    "usb": "pyusb", "gi": "pygobject", "lxml_html_clean": "lxml-html-clean",
    "snappy": "python-snappy", "redis": "redis", "psycopg2": "psycopg2-binary",
    "MySQLdb": "mysqlclient", "memcache": "python-memcached",
    "ruamel": "ruamel.yaml", "jinja2": "jinja2", "markdown": "markdown",
    "vcr": "vcrpy", "nacl": "pynacl", "zmq": "pyzmq", "git": "gitpython",
    "skimage": "scikit-image", "Levenshtein": "python-Levenshtein",
    "dotenv": "python-dotenv", "jose": "python-jose", "slugify": "python-slugify",
    "multipart": "python-multipart", "sqlalchemy": "sqlalchemy",
    "pytest_asyncio": "pytest-asyncio", "pytest_mock": "pytest-mock",
    "freezegun": "freezegun", "hypothesis": "hypothesis", "faker": "faker",
    "respx": "respx", "responses": "responses", "httpretty": "httpretty",
    "trustme": "trustme", "anyio": "anyio", "trio": "trio",
    "yarl": "yarl", "aiohttp": "aiohttp", "websockets": "websockets",
    "openid": "python3-openid", "corsheaders": "django-cors-headers",
    "pytestqt": "pytest-qt", "concurrent_log_handler": "concurrent-log-handler",
    "opentelemetry": "opentelemetry-api", "tzlocal": "tzlocal",
}

_IMPORT_EXTRA_DISTS = {
    "opentelemetry": ["opentelemetry-sdk", "opentelemetry-semantic-conventions"],
    "google": ["protobuf", "googleapis-common-protos"],
    "ruamel": ["ruamel.yaml.clib"],
}

def dist_candidates(top):
    """Plausible distribution names for the import name `top`, best first."""
    hy = top.replace("_", "-")
    cands = [_IMPORT_TO_DIST.get(top), top, hy,
             "python-" + hy, "py" + hy, hy + "-python", "python3-" + hy]
    if top.startswith("pytest"):
        cands.insert(0, "pytest-" + hy[len("pytest"):].lstrip("-"))
    out = []
    for c in cands:
        if c and c not in out:
            out.append(c)
    return out


def importable(mod):
    """Can the venv import `mod`? The only honest test of an install."""
    r = sh([f"{VENV}/bin/python", "-c", "import " + mod.split(".")[0]], 60)
    return bool(r and r.returncode == 0)


_PYPI_SEEN = {}


def pypi_exists(name, timeout=8):
    """Does PyPI have a project by this name? Cached, HEAD only."""
    if name in _PYPI_SEEN:
        return _PYPI_SEEN[name]
    ok = False
    try:
        req = urllib.request.Request(
            "https://pypi.org/simple/%s/" % urllib.parse.quote(name),
            method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ok = 200 <= r.status < 300
    except Exception:
        ok = False
    _PYPI_SEEN[name] = ok
    return ok


def install_for_import(top, budget=180):
    """Make `import top` work, trying candidate distribution names. → dist or
    None"""
    if importable(top):
        return "already"
    cands = [c for c in dist_candidates(top) if pypi_exists(c)]
    for cand in cands[:5]:
        pkgs = [cand] + _IMPORT_EXTRA_DISTS.get(top, [])
        ok, _bad = pip_best_effort(pkgs, budget, upgrade=False)
        if ok and importable(top):
            return cand
    return None
_MISSING_RE = re.compile(r"No module named ['\"]([A-Za-z0-9_.]+)['\"]")

_ALIAS_PIN = {"numpy": "numpy<2", "scipy": "scipy<1.13",
              "pandas": "pandas<2", "PIL": "pillow<10"}


def missing_mods(out, exclude=()):
    """Top-level module names pytest could not import, in first-seen order."""
    skip = {e.lower() for e in exclude if e} | {
        "tests", "test", "conftest", "__main__", "setup", "testing"}
    out_mods = []
    for mod in _MISSING_RE.findall(out):
        top = mod.split(".")[0]
        if (top and top.lower() not in skip
                and top not in out_mods
                and top not in getattr(sys, "stdlib_module_names", ())):
            out_mods.append(top)
    return out_mods


def missing_dists(out, exclude=()):
    """Distributions to try installing, from `No module named X` in pytest
    output."""
    skip = {e.lower() for e in exclude if e} | {
        "tests", "test", "conftest", "__main__", "setup"}
    want = []
    for mod in _MISSING_RE.findall(out):
        top = mod.split(".")[0]
        if not top or top.lower() in skip:
            continue
        dist = _IMPORT_TO_DIST.get(top)
        if dist is None:
            dist = top.replace("_", "-") if top.startswith("pytest_") else top
        if dist.lower() not in skip and dist not in want:
            want.append(dist)
            alt = top.replace("_", "-")
            if ("_" in top and not top.startswith("pytest_")
                    and alt.lower() not in skip and alt not in want):
                want.append(alt)
    return want


_FLAG_PLUGIN = {
    "--no-migrations": "pytest-django", "--nomigrations": "pytest-django",
    "--ds": "pytest-django", "--reuse-db": "pytest-django",
    "--create-db": "pytest-django", "--django": "pytest-django",
    "--benchmark": "pytest-benchmark", "--cov": "pytest-cov",
    "--numprocesses": "pytest-xdist", "--dist": "pytest-xdist",
    "--timeout": "pytest-timeout", "--asyncio-mode": "pytest-asyncio",
    "--snapshot": "syrupy", "--html": "pytest-html",
    "--json-report": "pytest-json-report", "--forked": "pytest-forked",
    "--reruns": "pytest-rerunfailures", "--rerun": "pytest-rerunfailures",
    "--randomly": "pytest-randomly", "--random-order": "pytest-random-order",
    "--mypy": "pytest-mypy", "--flake8": "pytest-flake8",
    "--black": "pytest-black", "--ruff": "pytest-ruff",
    "--subtests": "pytest-subtests", "--trio": "pytest-trio",
    "--order": "pytest-order", "--repeat": "pytest-repeat",
    "--profile": "pytest-profiling", "--memray": "pytest-memray",
    "--codspeed": "pytest-codspeed", "--record-mode": "pytest-recording",
    "--vcr": "pytest-recording", "--regressions": "pytest-regressions",
    "--lazy-fixture": "pytest-lazy-fixture", "--freeze": "pytest-freezegun",
    "--aiohttp": "pytest-aiohttp", "--tornado": "pytest-tornado",
    "--anyio": "anyio", "--integration": "pytest-integration",
    "--md-report": "pytest-md-report", "--split": "pytest-split",
    "--durations-path": "pytest-split", "--doctest-cython": "pytest-cython",
    "--report-log": "pytest-reportlog", "--nunit": "pytest-nunit",
    "--allure": "allure-pytest", "--alluredir": "allure-pytest",
    "--sugar": "pytest-sugar", "--instafail": "pytest-instafail",
    "--clarity": "pytest-clarity", "--icdiff": "pytest-icdiff",
    "--datadir": "pytest-datadir", "--dependency": "pytest-dependency",
    "--celery": "pytest-celery", "--flask": "pytest-flask",
    "--localserver": "pytest-localserver", "--httpserver": "pytest-httpserver",
    "--postgresql": "pytest-postgresql", "--mysql": "pytest-mysql",
    "--redis": "pytest-redis", "--mongo": "pytest-mongo",
    "--playwright": "pytest-playwright", "--selenium": "pytest-selenium",
    "--bdd": "pytest-bdd", "--gherkin": "pytest-bdd",
    "--twisted": "pytest-twisted", "--qt": "pytest-qt",
    "--check": "pytest-check", "--picked": "pytest-picked",
    "--testmon": "pytest-testmon", "--socket": "pytest-socket",
    "--disable-socket": "pytest-socket", "--env": "pytest-env",
    "--dotenv": "pytest-dotenv", "--recording": "pytest-recording",
    "--responses": "pytest-responses", "--examples": "pytest-examples",
}


def _pytest_cfg_text(repo_dir):
    out = []
    for fn in ("pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini"):
        p = os.path.join(repo_dir, fn)
        if os.path.exists(p):
            out.append(open(p, errors="ignore").read())
    return "\n".join(out)


_ADDOPTS_RE = re.compile(
    r"""addopts\s*=\s*(\[[^\]]*\]|"""            # TOML list
    r"""\"\"\"(?:.|\n)*?\"\"\"|'''(?:.|\n)*?'''"""  # triple-quoted
    r"""|"[^"]*"|'[^']*'"""                      # single-line quoted
    r"""|[^\n]*(?:\n[ \t]+[^\n]*)*)""")          # bare INI continuation lines


def plugins_for_flags(text):
    """Plugin distributions implied by pytest flags / `required_plugins`."""
    want = []
    scan = "\n".join(m.group(1) for m in _ADDOPTS_RE.finditer(text))
    for m in re.finditer(r"required_plugins\s*=\s*([^\n]+)", text):
        for tok in re.split(r"[,\s]+", m.group(1)):
            tok = tok.strip().strip("\"'[]").split(">")[0].split("=")[0]
            tok = tok.split("<")[0].strip()
            if tok and tok not in want:
                want.append(tok)
    for flag in re.findall(r"(--[A-Za-z0-9][A-Za-z0-9-]*)", scan):
        hit = max((k for k in _FLAG_PLUGIN if flag.startswith(k)),
                  key=len, default=None)
        if hit and _FLAG_PLUGIN[hit] not in want:
            want.append(_FLAG_PLUGIN[hit])
    return want


def plugins_for_unrecognized(out):
    """Plugins named by pytest's own `unrecognized arguments: ...` complaint."""
    want = []
    for m in re.finditer(r"unrecognized arguments:\s*([^\n]+)", out):
        for tok in re.split(r"[,\s]+", m.group(1)):
            tok = tok.split("=")[0].strip()
            if not tok.startswith("-"):
                continue
            hit = max((k for k in _FLAG_PLUGIN if tok.startswith(k)),
                      key=len, default=None)
            if hit and _FLAG_PLUGIN[hit] not in want:
                want.append(_FLAG_PLUGIN[hit])
    for m in re.finditer(r"Missing required plugins:\s*([^\n]+)", out):
        for tok in re.split(r"[,\s]+", m.group(1)):
            tok = tok.strip()
            if tok and tok not in want:
                want.append(tok)
    return want


def django_settings_module(repo_dir):
    """`DJANGO_SETTINGS_MODULE` for a Django repo, or None."""
    txt = _pytest_cfg_text(repo_dir)
    m = re.search(r"""DJANGO_SETTINGS_MODULE\s*[=:]\s*["']?([\w.]+)""", txt)
    if m:
        return m.group(1)
    mp = os.path.join(repo_dir, "manage.py")
    if not os.path.exists(mp):
        return None
    mm = re.search(r"""DJANGO_SETTINGS_MODULE["']\s*,\s*["']([\w.]+)""",
                   open(mp, errors="ignore").read())
    if mm:
        return mm.group(1)
    cands = (glob.glob(os.path.join(repo_dir, "*", "settings.py"))
             + glob.glob(os.path.join(repo_dir, "*", "settings", "__init__.py")))
    for cand in cands:
        rel = os.path.relpath(cand, repo_dir)
        mod = rel[:-3] if rel.endswith(".py") else rel
        return mod.replace(os.sep + "__init__", "").replace(os.sep, ".")
    return None


def cause_block(out, budget=700):
    """The FIRST error pytest reported, for a run that passed nothing."""
    lines = [l for l in out.splitlines() if l.strip()]
    start = next((i for i, l in enumerate(lines)
                  if re.match(r"=+ (ERRORS|FAILURES) =+", l)), None)
    if start is None:
        start = next((i for i, l in enumerate(lines)
                      if re.search(r"^E \s*\w|ImportError while loading|"
                                   r"INTERNALERROR|unrecognized arguments", l)),
                     None)
    seen, why = set(), []
    for l in lines[start:] if start is not None else lines:
        m = re.match(r"\s*E\s+(\w.*)|^(INTERNALERROR.*)|.*?"
                     r"((?:ModuleNotFoundError|ImportError|AttributeError|"
                     r"TypeError|SyntaxError|RuntimeError|OSError|"
                     r"ImproperlyConfigured)\b.*)", l)
        if not m:
            continue
        msg = (m.group(1) or m.group(2) or m.group(3) or "").strip()[:200]
        if msg and msg not in seen:
            seen.add(msg)
            why.append(msg)
        if len(why) >= 6:
            break
    tail = lines[start:start + 12] if start is not None else lines[-8:]
    return (" | ".join(why + tail))[:budget]


def nofail_args():
    """Undo the repo's own `-x` / `maxfail`."""
    return ["--maxfail=0"] + TIMEOUT_ARGS


NOCONFTEST = set()

BASELINE_OUT = {}


def collect_args(repo_dir):
    """`-o python_files=…` unless the repo configures it itself."""
    args = ["--noconftest"] if os.path.realpath(repo_dir) in NOCONFTEST else []
    if pytest_python_files(repo_dir):
        return args
    return args + ["-o", "python_files=test_*.py *_test.py *_tests.py tests.py"]


def _pytest_once(repo_dir, src_root, test_path, extra=(), env_extra=None,
                 budget=None):
    parts = [src_root] + ([repo_dir] if os.path.realpath(repo_dir)
                          != os.path.realpath(src_root) else [])
    env = clean_env(PYTHONPATH=os.pathsep.join(parts), **(env_extra or {}))
    extra = (*collect_args(repo_dir), *extra)
    return sh([py(), "-m", "pytest", os.path.join(repo_dir, test_path),
               "-q", "--tb=line", "-p", "no:cacheprovider",
               "--continue-on-collection-errors", *nofail_args(), *extra],
              budget or TEST_TIMEOUT, env=env)


def baseline(repo_dir, src_prefix, test_path, pkg=None, rounds=8):
    """How many tests the ORIGINAL code passes — the grading denominator."""
    src_root = repo_dir if src_prefix == "." else os.path.join(repo_dir, src_prefix)
    exclude = {pkg, dist_name(repo_dir)} if pkg else {dist_name(repo_dir)}
    out, installed, unresolved = "", [], []
    tried, n_tried_before = set(), -1
    for attempt in range(rounds):
        r = _pytest_once(repo_dir, src_root, test_path)
        if not r:
            tight = ["--timeout=45", "--timeout-method=signal"] if TIMEOUT_ARGS \
                else []
            r = _pytest_once(repo_dir, src_root, test_path, extra=tight,
                             budget=TEST_TIMEOUT * 2)
            if not r:
                return -1, "timeout", True, installed
        out = (r.stdout + r.stderr).strip()
        mods = [m for m in missing_mods(out, exclude) if m not in tried]
        for top in mods[:15]:
            tried.add(top)
            got = install_for_import(top)
            if got:
                installed.append(got if got != "already" else top)
            else:
                unresolved.append(top)
        plugs = [d for d in plugins_for_unrecognized(out)
                 if d not in installed and d not in tried]
        if plugs:
            tried.update(plugs)
            ok, _bad = pip_best_effort(plugs, 240, upgrade=False)
            installed += ok
        for dist, pin in _ALIAS_PIN.items():
            if (re.search(r"module '%s' has no attribute" % re.escape(dist), out)
                    and pin not in tried):
                tried.add(pin)
                ok, _bad = pip_best_effort([pin], 300, upgrade=False)
                installed += ok
        if not mods and not plugs and len(tried) == n_tried_before:
            break
        n_tried_before = len(tried)

    def n(word):
        m = re.search(r"(\d+) " + word, out)
        return int(m.group(1)) if m else 0

    if n("passed") == 0 and re.search(r"hookspec|hookimpl|PluginValidationError",
                                      out):
        r = _pytest_once(repo_dir, src_root, test_path,
                         env_extra={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
        if r and re.search(r"\d+ passed", r.stdout + r.stderr):
            out = (r.stdout + r.stderr).strip()

    pytest_broke = ("cannot import name" in out or "ImportError" in out) and \
        re.search(r"site-packages[/\\](_pytest|pluggy)[/\\]", out)
    if n("passed") == 0 and pytest_broke:
        pip_best_effort(["pytest", "pluggy"], 300, upgrade=True)
        r = _pytest_once(repo_dir, src_root, test_path)
        if r and re.search(r"\d+ passed", r.stdout + r.stderr):
            out = (r.stdout + r.stderr).strip()
            installed.append("[repaired pytest+pluggy]")

    if n("passed") == 0 and "while loading conftest" in out:
        NOCONFTEST.add(os.path.realpath(repo_dir))    # ceiling/floor must match
        r = _pytest_once(repo_dir, src_root, test_path)
        if r and re.search(r"\d+ passed", r.stdout + r.stderr):
            out = (r.stdout + r.stderr).strip()
            installed.append("[--noconftest]")
        else:
            NOCONFTEST.discard(os.path.realpath(repo_dir))

    last = cause_block(out) if n("passed") == 0 else \
        " | ".join(out.splitlines()[-6:])[-700:]
    if unresolved:
        last = "unresolved=" + ",".join(unresolved[:8]) + " | " + last
    dirty = n("errors?") > 0 or n("failed") > 0
    BASELINE_OUT[os.path.realpath(repo_dir)] = out
    return n("passed"), last, dirty, installed


def package_modules(pkg_dir):
    """Every module of the package, as paths relative to `pkg_dir`."""
    out = []
    for root, dirs, fs in os.walk(pkg_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")
                   and d.lower() not in ("test", "tests", "testing")]
        for f in fs:
            if not f.endswith(".py") or f == "conftest.py":
                continue
            if is_test_file(f) and not _library_module_named_like_a_test(
                    os.path.join(root, f), pkg_dir):
                continue
            out.append(os.path.relpath(os.path.join(root, f), pkg_dir))
    return sorted(out)


def _library_module_named_like_a_test(path, pkg_dir, cap=400):
    """Is this "test file" actually library code? (`jinja2/tests.py`)"""
    base = os.path.basename(path).lower()
    if base not in ("test.py", "tests.py", "testing.py"):
        return False
    mod = base[:-3]
    pat = re.compile(r"^\s*(?:from\s+[\w.]*\.?%s\s+import"
                     r"|from\s+\.+\s+import\s+[^\n]*\b%s\b"
                     r"|import\s+[\w.]+\.%s\b)" % (mod, mod, mod), re.M)
    here = os.path.realpath(path)
    n = 0
    for root, dirs, fs in os.walk(pkg_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in fs:
            if not f.endswith(".py") or is_test_file(f):
                continue                       # only LIBRARY code counts as evidence
            p = os.path.join(root, f)
            if os.path.realpath(p) == here or n >= cap:
                continue
            n += 1
            try:
                if pat.search(open(p, errors="ignore").read()):
                    return True
            except OSError:
                continue
    return False


def _exports(code):
    """Top-level names a module defines (for sibling-API feedback)."""
    names = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return names
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


_ASSET_SKIP = (".py", ".pyc", ".pyo")


_EXT_SUFFIX = (".so", ".pyd", ".dylib")


def copy_pkg_assets(pkg_dir, zpkg, gen_files=(), limit_mb=80):
    """Copy the package's non-`.py` files into the reconstruction tree. → count"""
    if not os.path.isdir(pkg_dir):
        return 0
    shadow = {os.path.basename(f).split(".")[0] for f in gen_files}
    n, total = 0, 0
    for root, dirs, fs in os.walk(pkg_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for f in fs:
            if f.endswith(_ASSET_SKIP):
                continue
            if f.endswith(_EXT_SUFFIX) and f.split(".")[0] in shadow:
                continue
            src = os.path.join(root, f)
            try:
                total += os.path.getsize(src)
            except OSError:
                continue
            if total > limit_mb * 1024 * 1024:
                return n
            dst = os.path.join(zpkg, os.path.relpath(src, pkg_dir))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
                n += 1
            except OSError:
                pass
    return n


def _write_zero(tdir, src_prefix, pkg, files, repo_dir, test_path):
    zero = os.path.join(tdir, "zero")
    rm_rf(zero)
    zsrc = zero if src_prefix == "." else os.path.join(zero, src_prefix)
    zpkg = zsrc if list(files) == [pkg + ".py"] else os.path.join(zsrc, pkg)
    os.makedirs(zpkg, exist_ok=True)

    tp = os.path.join(repo_dir, test_path)
    dst = os.path.join(zero, test_path)
    os.makedirs(os.path.dirname(dst) or zero, exist_ok=True)
    if os.path.isdir(tp):
        shutil.copytree(tp, dst, ignore=shutil.ignore_patterns("__pycache__"),
                        dirs_exist_ok=True)
    else:
        shutil.copy(tp, dst)

    if _overlaps(dst, zpkg):
        pats = pytest_python_files(repo_dir)
        counterpart = {os.path.realpath(os.path.join(zpkg, k)) for k in files}
        for root, dirs, fs in os.walk(dst):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in fs:
                if (not f.endswith(".py") or is_test_file(f, pats)
                        or f == "conftest.py"):
                    continue                      # the measuring instrument
                p = os.path.join(root, f)
                if os.path.realpath(p) in counterpart:
                    continue                      # about to be overwritten
                if f == "__init__.py" and has_test_files(root, pats):
                    continue
                os.remove(p)

    src_root = repo_dir if src_prefix == "." else os.path.join(repo_dir, src_prefix)
    opkg = os.path.join(src_root, pkg)
    if os.path.isdir(opkg) and zpkg != zsrc:
        copy_pkg_assets(opkg, zpkg, gen_files=files)

    gen_paths = set()
    for fn, code in files.items():
        p = os.path.join(zpkg, fn)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(code.rstrip() + "\n")
        gen_paths.add(os.path.realpath(p))
    for extra in ("conftest.py", "pytest.ini", "setup.cfg", "pyproject.toml"):
        p = os.path.join(repo_dir, extra)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(zero, extra))
    return zero, zsrc, gen_paths


def _overlaps(a, b):
    """Do these two paths refer to the same tree, or one inside the other?"""
    a, b = os.path.realpath(a), os.path.realpath(b)
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


_PROV_PLUGIN = '''"""Record the resolved file of every target-package module,
and the outcome of every individual test."""
import json, os, sys

_OUTCOMES = {}


def pytest_runtest_logreport(report):
    """Per-test outcomes, so a count can be checked against WHICH tests passed.

    `ceiling = 995` alone cannot distinguish "the same 995 tests passed" from "a
    different 995 passed", so a harness change that swaps which tests pass looks
    identical to no change at all. Keyed by node id; the call phase decides
    pass/fail, but a setup/teardown error is what actually happened so it wins.
    """
    if report.when == "call":
        _OUTCOMES.setdefault(report.nodeid, report.outcome)
    elif report.outcome == "failed":
        _OUTCOMES[report.nodeid] = "error"


def pytest_sessionfinish(session, exitstatus):
    pkg, out = os.environ.get("CF_PROV_PKG"), os.environ.get("CF_PROV_OUT")
    if not pkg or not out:
        return
    seen = {}
    for nm, m in list(sys.modules.items()):
        if nm != pkg and not nm.startswith(pkg + "."):
            continue
        f = getattr(m, "__file__", None)
        if f:
            seen[nm] = os.path.realpath(f)
    try:
        with open(out, "w") as fh:
            json.dump(seen, fh)
    except OSError:
        pass
    tests_out = os.environ.get("CF_TESTS_OUT")
    if tests_out:
        try:
            with open(tests_out, "w") as fh:
                json.dump(_OUTCOMES, fh)
        except OSError:
            pass
'''


def check_provenance(seen, zsrc, gen_paths, test_root):
    """Did the suite exercise the generated code, and only the generated code?"""
    root = os.path.realpath(zsrc)
    troot = os.path.realpath(test_root) if test_root else None
    impl = {nm: f for nm, f in (seen or {}).items()
            if not is_test_file(os.path.basename(f))
            and os.path.basename(f) != "conftest.py"
            and not (troot and _overlaps(f, troot) and f not in gen_paths
                     and os.path.basename(f) == "__init__.py")}
    if not impl:
        return {"verdict": "not-exercised", "exercised": 0, "leaked": []}
    leaked = sorted(nm for nm, f in impl.items()
                    if f not in gen_paths
                    and not f.startswith(root + os.sep))
    stray = sorted(nm for nm, f in impl.items()
                   if f not in gen_paths and f.startswith(root + os.sep)
                   and not f.endswith(_EXT_SUFFIX))
    return {"verdict": "leaked" if (leaked or stray) else "ok",
            "exercised": len(impl), "leaked": (leaked + stray)[:20]}


def canonical_test_id(nodeid, zero):
    """Express a pytest node id relative to the generated repository root."""
    path, marker, suffix = nodeid.partition("::")
    root = os.path.realpath(zero)
    candidates = [path] if os.path.isabs(path) else [
        os.path.abspath(path), os.path.join(root, path)]
    for candidate in candidates:
        resolved = os.path.realpath(candidate)
        if not os.path.isfile(resolved):
            continue
        try:
            if os.path.commonpath((root, resolved)) != root:
                continue
        except ValueError:
            continue
        rel = os.path.relpath(resolved, root).replace(os.sep, "/")
        return rel + (marker + suffix if marker else "")
    return nodeid.replace(os.sep, "/")


def canonical_test_outcomes(outcomes, zero):
    """Prefer an error or failure if duplicate raw IDs name the same test."""
    rank = {"passed": 0, "skipped": 1, "failed": 2, "error": 3}
    canonical = {}
    for nodeid, outcome in outcomes.items():
        key = canonical_test_id(nodeid, zero)
        if rank.get(outcome, 3) >= rank.get(canonical.get(key), -1):
            canonical[key] = outcome
    return canonical


def _test_zero(zero, zsrc, pkg, test_path, repo_dir=None, gen_paths=frozenset()):
    prov_dir = os.path.join(os.path.dirname(os.path.abspath(zero)), "prov")
    os.makedirs(prov_dir, exist_ok=True)
    open(os.path.join(prov_dir, "cf_prov.py"), "w").write(_PROV_PLUGIN)
    prov_out = os.path.join(prov_dir, "seen.json")
    tests_out = os.path.join(prov_dir, "tests.json")
    for p in (prov_out, tests_out):
        if os.path.exists(p):
            os.remove(p)
    env = clean_env(PYTHONPATH=os.pathsep.join(
                        [zsrc, *EXTRA_PYTHONPATH, prov_dir]),
                    CF_PROV_PKG=pkg, CF_PROV_OUT=prov_out,
                    CF_TESTS_OUT=tests_out)
    imp = sh([py(), "-c", f"import {pkg}"], 25, env=env)
    if not imp or imp.returncode != 0:
        err = (imp.stderr if imp else "import timeout") or ""
        return False, 0, 0, err[-600:], {"verdict": "import-failed"}
    rt = sh([py(), "-m", "pytest", os.path.join(zero, test_path),
             "-q", "--tb=line", "-p", "no:cacheprovider", "-p", "cf_prov",
             "--continue-on-collection-errors", *nofail_args(),
             *collect_args(repo_dir or zero)], TEST_TIMEOUT, env=env)
    if not rt:
        return True, 0, 0, "pytest timeout", {"verdict": "timeout",
                                              "exercised": 0, "leaked": []}
    seen = {}
    try:
        seen = json.load(open(prov_out))
    except Exception:
        pass
    prov = check_provenance(seen, zsrc, gen_paths,
                            os.path.join(zero, test_path))
    try:
        prov["tests"] = canonical_test_outcomes(json.load(open(tests_out)),
                                                zero)
    except Exception:
        prov["tests"] = {}
    out = (rt.stdout + rt.stderr).strip()
    if prov["verdict"] == "not-exercised" and re.search(
            r"no tests ran|collected 0 items", out):
        prov["verdict"] = "no-tests-collected"
    mp = re.search(r"(\d+) passed", out)
    mf = re.search(r"(\d+) failed", out)
    me = re.search(r"(\d+) errors?", out)
    failures = (int(mf.group(1)) if mf else 0) + (
        int(me.group(1)) if me else 0)
    if rt.returncode != 0 and failures == 0:
        failures = 1
    prov["pytest_exit"] = rt.returncode
    return True, (int(mp.group(1)) if mp else 0), \
        failures, out[-1500:], prov


_TB_FILE = re.compile(r'(?:^|[\s"\'(/])([A-Za-z_][\w\-]*\.py)(?=["\'):,\s]|:\d|$)',
                      re.M)


_ID_SPLIT = re.compile(r"[^A-Za-z]+|(?<=[a-z0-9])(?=[A-Z])")
_CAP_STOP = {"self", "cls", "get", "set", "the", "and", "for", "from", "with",
             "not", "add", "new", "obj", "val", "arg", "kwargs", "args", "str",
             "int", "list", "dict", "none", "true", "false", "test", "init",
             "py", "python", "module", "class", "def", "return", "type"}


def capability_terms(fn, code, pkg=""):
    """Describe a module by WHAT IT DOES, in words that transfer across repos."""
    words = []
    for sym in sorted(_exports(code)):
        words += [w.lower() for w in _ID_SPLIT.split(sym) if w]
    try:
        doc = ast.get_docstring(ast.parse(code)) or ""
    except SyntaxError:
        doc = ""
    words += [w.lower() for w in re.findall(r"[A-Za-z]{3,}", doc[:300])]
    for mod in re.findall(r"^(?:from|import)\s+([\w.]+)", code, re.M)[:12]:
        words += [w.lower() for w in mod.split(".") if w]
    words += [w.lower() for w in _ID_SPLIT.split(fn[:-3]) if w]
    seen, out = set(), []
    for w in words:
        if len(w) > 2 and w not in _CAP_STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return " ".join(out[:40])


def _transferable(fn, code):
    """Would this module plausibly help a DIFFERENT repo? Filters the two kinds
    of dead weight the reuse audit found: repo-private entrypoints, and stubs
    too small to carry a capability."""
    if not code or not code.strip() or "generation failed" in code:
        return False
    if _NON_TRANSFERABLE.match(fn):
        return False
    body = [ln for ln in code.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    if len(body) < _MIN_HARVEST_LINES:
        return False
    return bool(_exports(code))       # must define something importable


def _find_pkg_prefix(repo_dir, pkg, max_depth=4):
    """Prefix (relative to repo_dir) under which directory `pkg` lives, or None."""
    best = None
    for root, dirs, _files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", ".tox",
                                                ".venv", "venv", "__pycache__",
                                                "build", "dist")]
        rel = os.path.relpath(root, repo_dir)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirs[:] = []
            continue
        if pkg in dirs:
            cand = os.path.join(root, pkg)
            has_py = any(f.endswith(".py") for f in os.listdir(cand))
            if not has_py:
                continue
            key = (depth, 0 if os.path.exists(
                os.path.join(cand, "__init__.py")) else 1)
            if best is None or key < best[0]:
                best = (key, rel)
    return None if best is None else (best[1] if best[1] != "." else ".")


_VERDICT_STATUS = {
    "unknown": "invalid-import-failed-every-round",
    "no-tests-collected": "invalid-no-tests-collected",
}


def _status_for(verdict):
    return _VERDICT_STATUS.get(str(verdict), "invalid-" + str(verdict))


_TOPIMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)", re.M)
_STDLIB_ISH = (frozenset(getattr(sys, "stdlib_module_names", ()))
               | {"pytest", "mock", "unittest2", "hypothesis", "conftest",
                  "tests", "test", "testing", "docs", "examples", "scripts",
                  "setup", "numpy", "pandas", "pytest_asyncio", "pytest_mock",
                  "requests", "yaml", "six", "setuptools", "typing_extensions"})


def tests_top_imports(tests_path, cap=60):
    """Top-level module names the test files import, most frequent first."""
    counts = collections.Counter()
    files = []
    if os.path.isfile(tests_path):
        files = [tests_path]
    else:
        for root, dirs, fns in os.walk(tests_path):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
            files += [os.path.join(root, f) for f in fns if f.endswith(".py")]
            if len(files) >= cap:
                break
    for fp in files[:cap]:
        try:
            counts.update(set(_TOPIMPORT_RE.findall(
                open(fp, errors="ignore").read())))
        except OSError:
            continue
    return counts


def repair_pairing(repo_dir, pkg, tests_rel):
    """(pkg, src_prefix) corrected so the package is one these tests import."""
    tests_path = os.path.join(repo_dir, tests_rel)
    counts = tests_top_imports(tests_path)
    if not counts or counts.get(pkg):
        return None
    for cand, _n in counts.most_common(12):
        if cand == pkg or cand in _STDLIB_ISH:
            continue
        prefix = _find_pkg_prefix(repo_dir, cand)
        if prefix is not None:
            return cand, prefix
    return None


def resolve_target(repo_dir, task):
    """Decide what to rebuild and what to grade with: (src_prefix, package,
    tests, project_dir, meta_source)."""
    src_prefix = getattr(task, "src_prefix", ".") or "."
    pkg = getattr(task, "package", "") or ""
    if pkg:
        top = pkg.split(".")[0]
        if not os.path.isdir(os.path.join(
                repo_dir, "" if src_prefix == "." else src_prefix, top)):
            found_prefix = _find_pkg_prefix(repo_dir, top)
            if found_prefix is None:
                pkg = ""
            else:
                src_prefix = found_prefix
    declared_pkg = bool(pkg)
    if not pkg:
        src_prefix, pkg = detect_package(repo_dir)
    out = {"src_prefix": src_prefix or ".", "package": pkg or "",
           "tests": "", "project_dir": ".",
           "meta_source": "declared" if declared_pkg else "detected"}
    if not pkg:
        return out

    proj_dir = project_dir(repo_dir, out["src_prefix"])   # monorepo-aware root
    proj_rel = os.path.relpath(proj_dir, repo_dir)
    out["project_dir"] = proj_rel

    declared_tests = getattr(task, "tests", "") or ""
    if proj_rel not in (".", ""):
        found = detect_tests(proj_dir, pkg)
        if found:
            out["tests"] = os.path.normpath(os.path.join(proj_rel, found))
            return out
    if declared_tests and has_test_files(os.path.join(repo_dir, declared_tests),
                                         pytest_python_files(repo_dir)):
        out["tests"] = declared_tests
        return _repaired(repo_dir, out, declared_pkg)
    found = detect_tests(proj_dir, pkg)
    out["tests"] = (os.path.normpath(os.path.join(proj_rel, found)) if found
                    else (detect_tests(repo_dir, pkg) or ""))
    return _repaired(repo_dir, out, declared_pkg)


def _repaired(repo_dir, out, declared_pkg):
    """Last chance to fix a package/test mismatch, using the tests' own imports."""
    if declared_pkg or not out["tests"]:
        return out
    fix = repair_pairing(repo_dir, out["package"], out["tests"])
    if fix:
        out["package"], out["src_prefix"] = fix[0], fix[1] or "."
        out["project_dir"] = os.path.relpath(
            project_dir(repo_dir, out["src_prefix"]), repo_dir)
        out["meta_source"] = "detected+repaired"
    return out


def clone_repo(clone_url, repo_dir):
    """Clone the pinned commit, with retry+backoff."""
    sha = PINS.get(clone_url, {}).get("sha")

    def run(cmd, timeout):
        """True only on a clean exit — `sh` returns a truthy CompletedProcess
        even for a nonzero return code, so the steps must be checked, not just
        chained."""
        r = sh(cmd, timeout)
        return r is not None and r.returncode == 0

    for attempt in range(3):
        if sha:
            if (run(["git", "init", "-q", repo_dir], 60)
                    and run(["git", "-C", repo_dir, "remote", "add", "origin",
                             clone_url], 60)
                    and run(["git", "-C", repo_dir, "fetch", "-q", "--depth=1",
                             "origin", sha], CLONE_TIMEOUT)
                    and run(["git", "-C", repo_dir, "checkout", "-q",
                             "FETCH_HEAD"], CLONE_TIMEOUT)):
                return sha
        else:
            r = sh(["git", "clone", "--depth=1", clone_url, repo_dir],
                   CLONE_TIMEOUT)
            if r and r.returncode == 0:
                h = sh(["git", "-C", repo_dir, "rev-parse", "HEAD"], 60)
                return (h.stdout.strip() if h and h.returncode == 0
                        else "unknown")
        rm_rf(repo_dir)
        time.sleep(5 * (attempt + 1))    # backoff on throttle
    return None


def dump_task(rec, repo_dir, proj_dir, src_prefix, pkg, test_path, mods,
              orig_src):
    """Write one prepared benchmark task to disk: inputs, expected outputs, env."""
    out = os.path.join(BENCH_OUT, rec["name"])
    rm_rf(out)
    os.makedirs(out, exist_ok=True)

    src = os.path.join(repo_dir, test_path)
    dst = os.path.join(out, "tests")
    try:
        tracked = sh(["git", "-C", repo_dir, "ls-files", "-z", "--", test_path],
                     120)
        names = ([n for n in tracked.stdout.split("\0") if n]
                 if tracked and tracked.returncode == 0 else [])
        if names:
            for rel in names:
                s = os.path.join(repo_dir, rel)
                if not os.path.isfile(s) or rel.endswith(".pyc"):
                    continue
                d = os.path.join(dst, os.path.relpath(rel, test_path)
                                 if os.path.isdir(src) else os.path.basename(rel))
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
            rec["tests_files"] = len(names)
        elif os.path.isdir(src):     # not a git checkout: fall back to the tree
            shutil.copytree(src, dst, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)
    except Exception as e:
        rec["dump_warn"] = f"tests: {e}"

    orig = os.path.join(out, "original")
    os.makedirs(orig, exist_ok=True)
    for f, text in orig_src.items():
        p = os.path.join(orig, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)

    exp = os.path.join(out, "expected")
    os.makedirs(exp, exist_ok=True)
    for fname, ids in (("ceiling.txt", rec.get("ceiling_tests") or []),
                       ("floor.txt", rec.get("floor_tests") or [])):
        with open(os.path.join(exp, fname), "w") as fh:
            fh.write("\n".join(ids) + ("\n" if ids else ""))

    envd = os.path.join(out, "env")
    os.makedirs(envd, exist_ok=True)
    fr = sh([py(), "-m", "pip", "freeze"], 120)
    lock = fr.stdout if fr and fr.returncode == 0 else ""
    lines = [l for l in lock.splitlines() if l.strip() and not l.startswith("#")]
    with open(os.path.join(envd, "requirements.lock"), "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    tb = {t.lower().replace("_", "-") for t in TOOLBELT}
    belt = [l for l in lines
            if re.split(r"[=<>@\[]", l, 1)[0].strip().lower()
            .replace("_", "-") in tb]
    with open(os.path.join(envd, "toolbelt.txt"), "w") as fh:
        fh.write("\n".join(belt) + ("\n" if belt else ""))
    rec["n_locked"] = len(lines)
    rec["n_toolbelt"] = len(belt)

    with open(os.path.join(out, "task.json"), "w") as fh:
        json.dump({
            "name": rec["name"],
            "clone": rec.get("clone"),
            "commit": rec.get("commit"),
            "pinned": rec.get("pinned"),
            "package": pkg,
            "src_prefix": src_prefix,
            "project_dir": os.path.relpath(proj_dir, repo_dir),
            "test_path": test_path,
            "python": rec.get("python"),
            "n_locked": rec.get("n_locked"),
            "n_toolbelt": rec.get("n_toolbelt"),
            "python_why": rec.get("python_why"),
            "env_setup": rec.get("env_setup"),
            "deps": rec.get("deps"),
            "late_installed": rec.get("late_installed"),
            "modules": sorted(mods),
            "n_modules": len(mods),
            "grading": {
                "metric": ("(passed - floor) / (ceiling - floor), "
                           "clamped to [0,1]"),
                "suite_size": rec.get("baseline"),
                "ceiling": rec.get("ceiling"),
                "floor": rec.get("floor"),
                "headroom": rec.get("span"),
                "ceiling_prov": rec.get("ceiling_prov"),
                "floor_prov": rec.get("floor_prov"),
            },
            "prepared_at": int(time.time()),
        }, fh, indent=1)
    print(f"    [prepared] {out}  tests={len(rec.get('ceiling_tests') or [])}"
          f" floor={len(rec.get('floor_tests') or [])}"
          f" locked={rec.get('n_locked')}(+{rec.get('n_toolbelt')} toolbelt)"
          f" modules={len(mods)}", flush=True)
