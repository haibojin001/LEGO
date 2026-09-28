"""Reconstruction workspace for an external repository agent.

    python -m baselines.workspace --task NAME --out DIR [--mode instruct|issue]
        [--index PATH] [--env copy|shared] [--no-bounds]

The task is prepared through ``lego.harness.prepare.prepare`` exactly as
``lego.run`` prepares it (pinned clone, task environment with the installed
target removed, re-measured bounds). What the agent may see is then written to
DIR:

  DIR/repo/            the repository at the pinned commit, git history dropped
                       and re-initialized as a single commit, bytecode and build
                       products removed, and the target package's modules
                       replaced by interface stubs (``instruct``) or deleted
                       (``issue``). The native tests are present.
  DIR/INSTRUCTION.md   (instruct) package prefix, public interface (the
                       concatenated stubs) and how to run the native suite
  DIR/ISSUE.md         (issue) the same interface and suite as a synthetic issue
  DIR/requirements.txt distributions installed in the task environment (target
                       removed), for systems that provision their own sandbox
  DIR/env/             (--env copy) a private, relocated copy of the task
                       environment for systems that run on the host
  DIR/workspace.json   what ``baselines.grade`` needs; no grading key

No file under DIR holds a frozen node id, the floor or ceiling, a CodeFace entry
or an original implementation; ``leak_check`` verifies the last point against
the original sources before any agent starts. The host-side grading clone is
sanitized too, once the bounds are measured: git objects, bytecode, build
products, the identity-run tree and the target modules are deleted, because
grading reads only the tests, the package's non-code assets and the pytest
config. The re-measured bounds are kept next to that clone, not under DIR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

from lego.harness import core
from lego.harness.interface import stub
from lego.harness.prepare import NotGradeable, prepare, task_dir
from lego.harness.task import load_index

MODES = ("instruct", "issue")
ENV_MODES = ("copy", "shared")

_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache",
              ".ruff_cache", ".tox", ".nox", ".eggs", ".hypothesis", ".venv",
              "venv"}
_ROOT_BUILD = ("build", "dist")          # build products, at the project roots
_SKIP_SUFFIX = (".pyc", ".pyo")
# prepare() record fields that measure the original sources; never under DIR
HOST_ONLY = ("baseline", "baseline_line", "baseline_dirty", "ceiling_run",
             "ceiling_run_verdict", "floor_run", "ceiling_drift")
STUB_HEADER = ("# Interface stub: every body is `...`. Implement this module; "
               "keep its public names and signatures.\n")
MIN_LINE = 8                             # shorter lines never count as leaked


# ---------------------------------------------------------------- tree copy
def _ignore(roots, keep):
    roots = {os.path.realpath(r) for r in roots}
    keep = [os.path.realpath(k) for k in keep]

    def ign(d, names):
        at_root = os.path.realpath(d) in roots
        out = set()
        for n in names:
            if (n in _SKIP_DIRS or n.endswith(_SKIP_SUFFIX)
                    or n.endswith(".egg-info")):
                out.add(n)
            elif at_root and n in _ROOT_BUILD:
                p = os.path.realpath(os.path.join(d, n))
                if not any(core._overlaps(p, k) for k in keep):
                    out.add(n)
        return out
    return ign


def copy_clean(src: str, dst: str, roots=(), keep=()):
    """Copy a checkout without VCS data, caches or build products."""
    shutil.copytree(src, dst, symlinks=True,
                    ignore=_ignore(list(roots) or [src], keep))


def sanitize_grading_clone(ws) -> None:
    """Delete from the host clone every copy of the original sources. Grading
    (``Workspace.execute``) reads only tests, package assets and config."""
    keep = [ws.pkg_dir, os.path.join(ws.repo_dir, ws.test_path)]
    core.rm_rf(os.path.join(ws.repo_dir, ".git"))
    for root, dirs, _fs in os.walk(ws.repo_dir):
        for d in list(dirs):
            if d in ("__pycache__", ".pytest_cache", ".eggs") or \
                    d.endswith(".egg-info"):
                core.rm_rf(os.path.join(root, d))
                dirs.remove(d)
    for base in {ws.repo_dir, ws.proj_dir}:
        for d in _ROOT_BUILD:
            p = os.path.join(base, d)
            if os.path.isdir(p) and not any(core._overlaps(p, k) for k in keep):
                core.rm_rf(p)
    for m in ws.modules:
        p = os.path.join(ws.pkg_dir, m)
        if os.path.isfile(p):
            os.remove(p)
    for d in ("zero", "prov", "vendor_root"):
        core.rm_rf(os.path.join(ws.tdir, d))


# ---------------------------------------------------------------- leak check
def _norm(line: str) -> str:
    return " ".join(line.split())


def implementation_lines(orig: dict, stubs: dict) -> dict[str, set]:
    """Per module, the body lines of the original that its stub does not show
    (signatures, decorators, docstrings, imports and comments excluded)."""
    out = {}
    for m, src in orig.items():
        shown = {_norm(x) for x in (stubs.get(m) or "").splitlines()}
        lines = set()
        for x in src.splitlines():
            n = _norm(x)
            if len(n) < MIN_LINE or n in shown or n.startswith(
                    ("#", "def ", "async def ", "class ", "@", "import ",
                     "from ", '"', "'", "r\"", "r'")):
                continue
            lines.add(n)
        out[m] = lines
    return out


def copied_modules(text: str, impl: dict[str, set]) -> list[str]:
    """Original modules whose implementation this text reproduces: at least
    half of the module's body lines (and two or more) appear verbatim."""
    have = {_norm(x) for x in text.splitlines()}
    hits = []
    for m, lines in impl.items():
        if len(lines) < 2:
            continue
        k = len(lines & have)
        if k >= max(2, math.ceil(0.5 * len(lines))):
            hits.append(m)
    return hits


def _under(rel: str, base: str) -> bool:
    rel, base = os.path.normpath(rel), os.path.normpath(base)
    return rel == base or rel.startswith(base + os.sep)


def _is_native_test(rel: str, test_path: str) -> bool:
    base = os.path.basename(rel)
    return ((test_path not in ("", ".") and _under(rel, test_path))
            or core.is_test_file(base) or base == "conftest.py")


def _id_index(node_ids) -> dict[str, set]:
    """Frozen node ids grouped by their ``file::`` prefix, for a cheap scan."""
    by = {}
    for i in node_ids:
        if "::" in i:
            by.setdefault(i.split("::", 1)[0] + "::", set()).add(i)
    return by


def _text_files(root: str):
    for r, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in fs:
            p = os.path.join(r, f)
            if os.path.islink(p) or os.path.getsize(p) > 4 * 1024 * 1024:
                continue
            try:
                with open(p, encoding="utf-8") as fh:
                    yield p, fh.read()
            except (UnicodeDecodeError, OSError):
                continue


def scrub_copies(tree: str, test_path: str, impl: dict, protect=()) -> list:
    """Delete non-test files of the tree that reproduce an original module
    (left-over build copies, vendored duplicates). Returns what was dropped."""
    protect = {os.path.realpath(p) for p in protect}
    dropped = []
    for p, text in list(_text_files(tree)):
        rel = os.path.relpath(p, tree)
        if os.path.realpath(p) in protect or _is_native_test(rel, test_path):
            continue
        hit = copied_modules(text, impl)
        if hit:
            os.remove(p)
            dropped.append({"path": rel, "copies": hit[:5]})
    return dropped


def leak_check(out: str, test_path: str, impl: dict,
               node_ids=()) -> list[str]:
    """Problems found in anything under DIR the agent can read: an original
    implementation, or a frozen node id, outside the native tests."""
    problems = []
    tree = os.path.join(out, "repo")
    ids = _id_index(node_ids)
    for p, text in _text_files(out):
        rel = os.path.relpath(p, out)
        if p.startswith(os.path.join(out, "env") + os.sep):
            continue
        if p.startswith(tree + os.sep) and _is_native_test(
                os.path.relpath(p, tree), test_path):
            continue
        for m in copied_modules(text, impl):
            problems.append(f"{rel}: reproduces {m}")
        if any(i in text for pre, grp in ids.items() if pre in text
               for i in grp):
            problems.append(f"{rel}: contains a frozen node id")
    return problems


# ---------------------------------------------------------------- text
def _mod_path(pkg_rel: str, m: str) -> str:
    return os.path.normpath(os.path.join(pkg_rel, m)).replace(os.sep, "/")


def interface_listing(pkg_rel: str, stubs: dict) -> str:
    return "\n\n".join(f"# ---- {_mod_path(pkg_rel, m)}\n{stubs[m].rstrip()}"
                       for m in sorted(stubs))


def suite_command(src_prefix: str, test_path: str) -> str:
    return (f"PYTHONPATH={src_prefix or '.'} python -m pytest {test_path} -q")


def instruction_text(pkg: str, pkg_rel: str, stubs: dict, src_prefix: str,
                     test_path: str) -> str:
    mods = "\n".join(f"  - `{_mod_path(pkg_rel, m)}`" for m in sorted(stubs))
    return f"""# Task: implement the package `{pkg}`

Every module of the Python package `{pkg}` in this repository has had its
implementation removed. Each file listed below is an interface stub: it keeps
the module's imports, public names, signatures and docstrings, and every body
is `...`. Implement the package so that the repository's native test suite
passes.

## Package prefix

- import name: `{pkg}`
- source directory: `{pkg_rel}`
- modules ({len(stubs)}):
{mods}

## Native test suite

Run from the repository root:

    {suite_command(src_prefix, test_path)}

The tests are the specification. Only the `.py` modules under `{pkg_rel}` are
collected as the result; edits to tests or to files outside that directory are
not part of it. Write the implementation yourself: do not obtain the package's
original source from a package index, the network, or elsewhere on disk.

## Public interface

```python
{interface_listing(pkg_rel, stubs)}
```
"""


def issue_text(pkg: str, pkg_rel: str, stubs: dict, src_prefix: str,
               test_path: str) -> str:
    mods = "\n".join(f"- `{_mod_path(pkg_rel, m)}`" for m in sorted(stubs))
    return f"""# The `{pkg}` package is missing its implementation

The modules of the Python package `{pkg}` (source directory `{pkg_rel}`) are
absent from this repository, so `import {pkg}` fails and the native test suite
cannot pass. Implement the package: create each module below at its path, with
the public interface given in this issue, so that the tests pass.

## Modules to create ({len(stubs)})

{mods}

## How to reproduce

Run from the repository root:

    {suite_command(src_prefix, test_path)}

## Expected behaviour

The suite passes. The tests are the specification; only the `.py` modules
under `{pkg_rel}` are part of the fix. Write the implementation yourself: do not
obtain the package's original source from a package index, the network, or
elsewhere on disk.

## Public interface

```python
{interface_listing(pkg_rel, stubs)}
```
"""


# ---------------------------------------------------------------- environment
def _site_packages(venv: str) -> list[str]:
    import glob
    return sorted(glob.glob(os.path.join(venv, "lib", "python*",
                                         "site-packages")))


def env_fingerprint(venv: str) -> str:
    """Hash of every file under the environment's site-packages (path, size,
    mtime); bytecode caches excluded. Detects an agent modifying the grading
    environment."""
    h = hashlib.sha1()
    for sp in _site_packages(venv):
        for r, dirs, fs in os.walk(sp):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for f in sorted(fs):
                p = os.path.join(r, f)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                h.update(f"{os.path.relpath(p, sp)}|{st.st_size}|"
                         f"{int(st.st_mtime)}\n".encode())
    return h.hexdigest()


def freeze(exclude: set[str]) -> str:
    r = core.sh([core.py(), "-m", "pip", "freeze", "--exclude-editable"], 180)
    keep = []
    norm = {e.lower().replace("_", "-") for e in exclude if e}
    for line in (r.stdout if r and r.returncode == 0 else "").splitlines():
        name = re.split(r"[=<>@ \[;]", line.strip(), maxsplit=1)[0]
        name = name.lower().replace("_", "-")
        if not name or line.startswith("-e") or "file://" in line or \
                name in norm:
            continue
        keep.append(line.strip())
    return "\n".join(keep) + ("\n" if keep else "")


def copy_env(src: str, dst: str) -> str:
    """Relocated copy of a virtual environment: scripts and activate files that
    name the source path are rewritten, so installing into the copy never
    touches the grading environment. Returns the copy's bin directory."""
    shutil.copytree(src, dst, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    old, new = os.path.realpath(src), os.path.abspath(dst)
    bindir = os.path.join(dst, "bin")
    for f in os.listdir(bindir):
        p = os.path.join(bindir, f)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        try:
            text = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        fixed = text.replace(old, new).replace(src, new)
        if fixed != text:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(fixed)
    return bindir


def _git_init(tree: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="workspace",
               GIT_AUTHOR_EMAIL="workspace@localhost",
               GIT_COMMITTER_NAME="workspace",
               GIT_COMMITTER_EMAIL="workspace@localhost",
               GIT_AUTHOR_DATE="2000-01-01T00:00:00+0000",
               GIT_COMMITTER_DATE="2000-01-01T00:00:00+0000")
    g = ["git", "-C", tree, "-c", "commit.gpgsign=false"]
    for cmd in (["git", "init", "-q", tree], g + ["add", "-A"],
                g + ["commit", "-q", "--allow-empty", "-m",
                     "reconstruction workspace"]):
        subprocess.run(cmd, env=env, check=True, capture_output=True)
    return subprocess.run(g + ["rev-parse", "HEAD"], env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


def _prune_empty(d: str):
    for r, _dirs, _fs in sorted(os.walk(d), key=lambda x: -len(x[0])):
        if not os.listdir(r):
            os.rmdir(r)


# ---------------------------------------------------------------- build
def build(task, out: str, mode: str = "instruct", env: str = "copy",
          measure_bounds: bool = True, tdir: str | None = None,
          index: str | None = None):
    """Prepare ``task`` and write the agent workspace to ``out``.

    Returns ``(ws, meta)``: the prepared ``Workspace`` (grading side, kept in
    memory) and the dict also written to ``out/workspace.json``. Raises
    ``NotGradeable`` exactly where ``lego.run`` would."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if env not in ENV_MODES:
        raise ValueError(f"env must be one of {ENV_MODES}")
    out = os.path.abspath(out)
    tdir = tdir or task_dir(core.CLONE_BASE, task.name)
    core.rm_rf(os.path.join(tdir, "repo"))   # a sanitized clone is never reused
    ws = prepare(task, tdir, measure_bounds=measure_bounds)

    pkg_rel = os.path.relpath(ws.pkg_dir, ws.repo_dir)
    stubs = {m: stub(ws.orig_src[m]) for m in ws.modules}
    impl = implementation_lines(ws.orig_src, stubs)

    core.rm_rf(out)
    os.makedirs(out)
    tree = os.path.join(out, "repo")
    copy_clean(ws.repo_dir, tree, roots=[ws.repo_dir, ws.proj_dir],
               keep=[ws.pkg_dir, os.path.join(ws.repo_dir, ws.test_path)])
    targets = []
    for m in ws.modules:
        p = os.path.join(tree, pkg_rel, m)
        targets.append(p)
        if mode == "instruct":
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(STUB_HEADER + stubs[m].rstrip() + "\n")
        elif os.path.lexists(p):
            os.remove(p)
    dropped = scrub_copies(tree, ws.test_path, impl, protect=targets)
    if mode == "issue" and os.path.isdir(os.path.join(tree, pkg_rel)) \
            and pkg_rel not in ("", "."):
        _prune_empty(os.path.join(tree, pkg_rel))

    text_fn = instruction_text if mode == "instruct" else issue_text
    instr = os.path.join(out, "INSTRUCTION.md" if mode == "instruct"
                         else "ISSUE.md")
    with open(instr, "w") as fh:
        fh.write(text_fn(ws.pkg, pkg_rel, stubs, ws.src_prefix, ws.test_path))
    with open(os.path.join(out, "requirements.txt"), "w") as fh:
        fh.write(freeze({ws.pkg, core.dist_name(ws.proj_dir) or ""}))
    agent_bin = (copy_env(core.VENV, os.path.join(out, "env"))
                 if env == "copy" else os.path.join(core.VENV, "bin"))
    head = _git_init(tree)

    ids = task.ceiling_ids() | task.floor_ids()
    problems = leak_check(out, ws.test_path, impl, node_ids=ids)
    if problems:
        raise RuntimeError("workspace exposes the answer: "
                           + "; ".join(problems[:5]))

    with open(os.path.join(tdir, "bounds.json"), "w") as fh:
        json.dump({k: ws.record.get(k) for k in HOST_ONLY}, fh)
    sanitize_grading_clone(ws)

    meta = {
        "task": task.name, "mode": mode, "index": index, "out": out,
        "tree": tree, "instruction": instr, "tree_commit": head,
        "pkg": ws.pkg, "pkg_rel": pkg_rel, "src_prefix": ws.src_prefix,
        "test_path": ws.test_path, "modules": list(ws.modules),
        "single_module": bool(ws.record.get("single_module")),
        "grading": {"tdir": tdir, "repo_dir": ws.repo_dir,
                    "proj_dir": ws.proj_dir, "pkg_dir": ws.pkg_dir,
                    "venv": core.VENV, "clone_base": core.CLONE_BASE},
        "env": {"mode": env, "agent_bin": agent_bin,
                "fingerprint": env_fingerprint(core.VENV)},
        "record": {k: v for k, v in ws.record.items() if k not in HOST_ONLY},
        "dropped_copies": dropped, "created": time.time()}
    with open(os.path.join(out, "workspace.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return ws, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=MODES, default="instruct")
    ap.add_argument("--env", choices=ENV_MODES, default="copy")
    ap.add_argument("--index")
    ap.add_argument("--no-bounds", action="store_true",
                    help="skip re-measuring ceiling/floor (the frozen key is "
                         "still used for grading)")
    a = ap.parse_args(argv)
    from lego.run import configure_harness
    idx = load_index(a.index)
    if a.task not in idx:
        raise SystemExit(f"task {a.task!r} not in index")
    configure_harness("ws_" + re.sub(r"[^A-Za-z0-9_.-]", "_", a.task))
    try:
        _ws, meta = build(idx[a.task], a.out, a.mode, a.env,
                          measure_bounds=not a.no_bounds, index=a.index)
    except NotGradeable as e:
        print(f"[workspace] {a.task}: not gradeable ({e.args[0]})")
        return 2
    print(f"[workspace] {a.task}: {len(meta['modules'])} modules, mode "
          f"{a.mode}\n  tree        {meta['tree']}\n  instruction "
          f"{meta['instruction']}\n  meta        "
          f"{os.path.join(meta['out'], 'workspace.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
