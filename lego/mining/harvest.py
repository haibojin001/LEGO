"""Harvest Code Primitives (kind = harvested) from verified construction runs.

    python -m lego.mining.harvest --arm runs/arms/<arm_id> [--arm ...] \\
        --out codeface [--threshold 0.5] [--index benchmark/tasks.jsonl] \\
        [--python PY]

Harvesting MUST come from a construction pass that is disjoint from the
evaluation pass. The library is frozen while an evaluation runs (``lego.run``
only reads it). Run a separate construction pass, harvest from its arm
directory, then freeze the library before evaluating.

Inputs, as written by ``lego.run``: ``<arm>/records*.jsonl`` (newest record per
task wins) and ``<arm>/trees/<task>.json.gz``, the best generated tree of a task
as ``{module path relative to the package: source}``. Task metadata (clone URL,
package, pinned commit) comes from the benchmark index.

Selection: a task qualifies when ``status == "done"`` and ``score >=
threshold``. Within it, a module is harvested when it was not implicated by
the last diagnosis of the trajectory, is transferable, and its dependency
closure inside the generated tree fits the component bound. Frozen task tests
are copied beside the generated tree and recovered by the same collector used
for mined code. When they cannot be carried, ``--synth-tests --model`` requests
component tests. Admission always requires a passing test in isolation;
import-only artifacts are excluded.

Provenance: ``source.repo``/``url``/``commit`` are the task's repository and
pinned commit; ``source.run = {arm, task, score, round}``. Log:
``<out>/_harvest_log.jsonl`` (resumable per arm and task).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import shutil
import sys
import tempfile
import time

from lego.benchmark_meta import HELD_REPOS
from lego.harness import core
from lego.mining import segment, synthesize
from lego.mining.graph import STDLIB, build_graph
from lego.mining.mine import _llm, make_pid, repo_id

LOG = "_harvest_log.jsonl"


def load_records(arm_dir: str) -> dict:
    """Newest record per task (same rule as ``lego.run.load_records``)."""
    out = {}
    for f in sorted(os.listdir(arm_dir)):
        if not (f.startswith("records") and f.endswith(".jsonl")):
            continue
        for line in open(os.path.join(arm_dir, f)):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("name"):
                prev = out.get(r["name"])
                if prev is None or r.get("ts", 0) >= prev.get("ts", 0):
                    out[r["name"]] = r
    return out


def load_tree(arm_dir: str, task: str) -> dict | None:
    p = os.path.join(arm_dir, "trees", task + ".json.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt") as fh:
        return json.load(fh)


def implicated(rec: dict) -> set:
    """Modules implicated by the last diagnosis of a trajectory."""
    for step in reversed(rec.get("trajectory") or []):
        d = step.get("diagnosis")
        if d:
            return {m for m in d.get("implicated") or [] if isinstance(m, str)}
    return set()


def guess_package(files: dict, fallback: str) -> str:
    """Package name from the tree's own absolute imports (no index)."""
    tops = collections.Counter()
    for code in files.values():
        for line in (code or "").splitlines():
            s = line.strip()
            if s.startswith(("from ", "import ")):
                parts = s.split()
                if len(parts) > 1 and not parts[1].startswith("."):
                    tops[parts[1].split(".")[0].rstrip(",")] += 1
    for top, _n in tops.most_common():
        if top not in STDLIB:
            return top
    return fallback


def _task_info(index: dict, name: str, rec: dict, files: dict) -> dict:
    t = index.get(name)
    if t is not None:
        return {"clone": t.clone, "commit": rec.get("commit") or t.commit,
                "package": t.package,
                "test_source": os.path.join(t.dir, "tests")}
    return {"clone": "", "commit": rec.get("commit"),
            "package": guess_package(files, name), "test_source": ""}


def materialize(tmp: str, pkg: str, files: dict, single: bool,
                test_source: str = "") -> str:
    """Write a generated tree as a minimal repository; -> repo dir."""
    repo = os.path.join(tmp, "repo")
    core.rm_rf(repo)
    base = repo if single else os.path.join(repo, pkg)
    for rel, code in files.items():
        dst = os.path.join(repo, pkg + ".py") if single else os.path.join(base, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as fh:
            fh.write(code or "")
    if not single and not os.path.exists(os.path.join(base, "__init__.py")):
        open(os.path.join(base, "__init__.py"), "w").close()
    if os.path.isdir(test_source):
        shutil.copytree(test_source, os.path.join(repo, "tests"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return repo


def harvest_task(arm_id: str, rec: dict, files: dict, info: dict, out: str,
                 env: synthesize.Env, opts: synthesize.Options) -> list:
    name = rec["name"]
    tmp = tempfile.mkdtemp(prefix="lego_harvest_")
    rows = []
    try:
        single = bool(rec.get("single_module"))
        repo = materialize(tmp, info["package"], files, single,
                           info.get("test_source") or "")
        g = build_graph(repo, ".", info["package"], with_tests=True)
        blamed = implicated(rec)
        org, rname = repo_id(info["clone"]) if info["clone"] else ("", name)
        source = {"repo": f"{org}/{rname}" if org else rname, "org": org,
                  "url": info["clone"], "commit": info["commit"], "license": "",
                  "forks": [],
                  "run": {"arm": arm_id, "task": name, "score": rec.get("score"),
                          "round": rec.get("best_round")}}
        for m in sorted(g.modules):
            base = {"arm": arm_id, "task": name, "module": m}
            code = g.modules[m].source
            fname = os.path.basename(m) if not single else f"{info['package']}.py"
            if m in blamed or (single and files and next(iter(files)) in blamed):
                rows.append(dict(base, status="excluded",
                                 reason="implicated by last diagnosis"))
                continue
            if not core._transferable(fname, code):
                rows.append(dict(base, status="excluded", reason="not transferable"))
                continue
            mods = g.closure([m])
            if (len(mods) > opts.max_files or g.lines(mods) > opts.max_lines):
                rows.append(dict(base, status="excluded",
                                 reason="closure exceeds bound"))
                continue
            comp = segment.Component(m, [m] + sorted(mods - {m}), g.lines(mods),
                                     ["harvested"], [m])
            pid = make_pid(f"h_{name}", g.dotted(m), f"{arm_id}|{name}|{m}")
            try:
                row = synthesize.synthesize(g, comp, out, pid, source, env, opts)
            except Exception as e:  # noqa: BLE001
                row = {"pid": pid, "status": "excluded",
                       "reason": f"synthesis error: {type(e).__name__}: {e}"[:300]}
            rows.append(dict(base, **row))
            print(f"  [{row['status']}] {name}:{m} -> {pid}: {row['reason']}",
                  flush=True)
    finally:
        core.rm_rf(tmp)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="+", required=True,
                    help="arm directories (runs/arms/<arm_id>)")
    ap.add_argument("--out", default="codeface")
    ap.add_argument("--threshold", type=float, default=core.HARVEST_MIN_RATE)
    ap.add_argument("--index", default=None, help="benchmark tasks.jsonl")
    ap.add_argument("--python", default=None, help="validate with this "
                    "interpreter instead of a fresh venv (no installs)")
    ap.add_argument("--synth-tests", action="store_true",
                    help="write component tests if frozen tests cannot be carried")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-files", type=int, default=segment.MAX_FILES)
    ap.add_argument("--max-lines", type=int, default=segment.MAX_LINES)
    a = ap.parse_args(argv)
    if (a.synth_tests or a.describe) and not a.model:
        ap.error("--synth-tests/--describe need --model")
    try:
        from lego.harness.task import load_index
        index = load_index(a.index)
    except (OSError, ValueError):
        index = {}
        print("[harvest] no benchmark index: provenance limited to task names",
              flush=True)
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, LOG)
    done = set()
    if os.path.exists(log):
        for line in open(log):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "task":
                done.add((r.get("arm"), r.get("task")))
    if not a.python:
        scratch = os.environ.get("LEGO_SCRATCH") or tempfile.gettempdir()
        core.VENV = os.path.join(scratch, f"lego_harvest_venv_{os.getpid()}")
    env = synthesize.Env(python=a.python)
    llm = _llm(a.model) if a.model else None
    opts = synthesize.Options(kind="harvested", mode="tests",
                              max_files=a.max_files, max_lines=a.max_lines,
                              tests_llm=llm if a.synth_tests else None,
                              describe_llm=llm if a.describe else None)
    n = 0
    try:
        for arm_dir in a.arm:
            arm_id = os.path.basename(os.path.normpath(arm_dir))
            recs = load_records(arm_dir)
            for name, rec in sorted(recs.items()):
                if (arm_id, name) in done:
                    continue
                trow = {"event": "task", "arm": arm_id, "task": name,
                        "ts": time.time()}
                t = index.get(name)
                held = name in HELD_REPOS or (
                    t is not None and repo_id(t.clone)[1] in HELD_REPOS)
                if held:
                    trow.update(status="held")
                elif rec.get("status") != "done" or \
                        (rec.get("score") or 0) < a.threshold:
                    trow.update(status="below-threshold", score=rec.get("score"))
                else:
                    files = load_tree(arm_dir, name)
                    if not files:
                        trow.update(status="no-tree")
                    else:
                        info = _task_info(index, name, rec, files)
                        rows = harvest_task(arm_id, rec, files, info, a.out,
                                            env, opts)
                        k = sum(r.get("status") == "admitted" for r in rows)
                        n += k
                        trow.update(status="done", score=rec.get("score"),
                                    n_admitted=k)
                        with open(log, "a") as fh:
                            for r in rows:
                                fh.write(json.dumps(dict(r, event="module")) + "\n")
                with open(log, "a") as fh:
                    fh.write(json.dumps(trow) + "\n")
    finally:
        env.close()
    print(f"[harvest] {n} primitives admitted -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
