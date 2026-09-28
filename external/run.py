"""Run LEGO or the matched feedback baseline on a transfer benchmark.

    python -m external.run --benchmark repocraft --config lego|feedback \
        --model ALIAS [--library codeface_repocraft] [--shard i --nshards n] \
        [--only t1 t2] [--set k=v ...] [--benchmarks external/benchmarks.yaml] \
        [--keep-outputs] [--no-trees] [--force] [--max N] [--no-preflight]

Per task: prepare an ``ExternalWorkspace`` (starting tree + task environment);
take the benchmark's interface or, for spec-only tasks, the backbone's plan --
cached per backbone under ``runs/external/<benchmark>/plans/<model>/`` so that
the LEGO and feedback arms construct against the same plan; run
``Construction`` with the chosen configuration; materialize the best tree; and
invoke the benchmark's own grader command on it. Per task, the raw grader
output and the metric the grader reports are stored in

    runs/external/<benchmark>/<arm>/records.jsonl   (records.s<k>.jsonl if sharded)
    runs/external/<benchmark>/<arm>/grader/<task>/{stdout,stderr}.log

``metric`` is whatever the grader reports, parsed with the benchmark's
``metric_parse`` rule; no external score is mapped onto the LEGO-REPO scale.
A configuration with a library must run on a library filtered for this
benchmark by ``external.filter_library``; the run refuses otherwise. Tasks
that already have a record in the arm are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback

from lego import config as C
from lego.harness import core
from lego.harness.prepare import task_dir
from lego.llm import client
from lego.run import configure_harness, load_records, parse_set

from external.common import (ExternalWorkspace, apply_plan, feedback_score,
                             is_placeholder, load_benchmark, load_tasks, plan)

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def safe(s: str) -> str:
    return _SAFE.sub("_", str(s))


def bench_dir(bench: str) -> str:
    return os.path.join(os.environ.get("LEGO_RUNS", "runs"), "external", bench)


def check_filtered(path: str, bench: str) -> dict:
    rep = os.path.join(path, "filter_report.json")
    r = json.load(open(rep)) if os.path.exists(rep) else {}
    if r.get("benchmark") != bench:
        raise SystemExit(
            f"{path} is not a library filtered for {bench}; run\n  python -m "
            f"external.filter_library --benchmark {bench} --library <codeface> "
            f"--out {path}\nbefore any task of this benchmark")
    return {"library": path, "removed": r.get("removed"),
            "n_kept": r.get("n_kept"), "thresh": r.get("thresh")}


_LIBS: dict = {}


def library_view(cfg, task):
    """Per-task view of the (already benchmark-filtered) library. The MinHash
    rule was applied offline against every target, so no target sources are
    passed here."""
    from lego.library.codeface import CodeFace
    key = (cfg.view.path, cfg.embedder)
    if key not in _LIBS:
        _LIBS[key] = CodeFace(cfg.view.path, cfg.embedder)
    filt = {"kinds": cfg.view.kinds, "exclude": cfg.view.exclude,
            "exclude_repos": cfg.view.exclude_repos,
            "near_dup_thresh": cfg.view.near_dup_thresh,
            "subset_frac": cfg.view.subset_frac,
            "subset_seed": cfg.view.subset_seed}
    return _LIBS[key].view(task, filt)


def get_plan(cfg, task, ws, bench: str):
    """The backbone's plan for a spec-only task (cached per backbone)."""
    pdir = os.path.join(bench_dir(bench), "plans", safe(cfg.models.backbone))
    path = os.path.join(pdir, safe(task.name) + ".json")
    if os.path.exists(path):
        return json.load(open(path))["plan"], "cached"
    llm = client.LLM(cfg.models.backbone, "decompose", cfg.temperature)
    p = plan(llm, task, ws.pkg)
    if not p["modules"]:
        raise RuntimeError("the backbone's plan contained no modules")
    os.makedirs(pdir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"plan": p, "model": cfg.models.backbone, "ts": time.time(),
                   "usage": client.USAGE.snapshot()}, fh, indent=1)
    return p, "planned"


def parse_metric(rule, text: str, out_dir: str):
    """The benchmark's own metric from its grader output: ``{regex: R}`` (last
    match, first group) or ``{json: FILE, key: a.b}`` (FILE relative to the
    grader output directory)."""
    if not rule or is_placeholder(rule):
        return None
    if isinstance(rule, str):
        rule = {"regex": rule}
    if rule.get("regex") and not is_placeholder(rule["regex"]):
        m = re.findall(rule["regex"], text or "", re.M)
        if m:
            v = m[-1][0] if isinstance(m[-1], tuple) else m[-1]
            try:
                return float(v)
            except ValueError:
                return v
    if rule.get("json") and not is_placeholder(rule["json"]):
        p = os.path.join(out_dir, rule["json"])
        if os.path.exists(p):
            cur = json.load(open(p))
            for part in str(rule.get("key") or "").split("."):
                if part:
                    cur = cur.get(part) if isinstance(cur, dict) else None
            return cur
    return None


def run_grader(spec: dict, task, tree: str, gdir: str) -> dict:
    """Invoke the benchmark's official grader command, unmodified, on a tree."""
    os.makedirs(gdir, exist_ok=True)
    q = {k: shlex.quote(str(v)) for k, v in
         {"workdir": tree, "task": task.name, "out": gdir,
          "python": core.py(), "release": spec.get("release_path") or ""}.items()}
    cmd = task.grader_cmd.format(**q)
    rel = spec.get("release_path")
    cwd = rel if rel and not is_placeholder(rel) and os.path.isdir(rel) else tree
    t0, rc, timed_out = time.time(), None, False
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                           text=True,
                           timeout=int(spec.get("grader_timeout") or 7200))
        rc, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        so, se = e.stdout or "", e.stderr or ""
        so = so.decode(errors="ignore") if isinstance(so, bytes) else so
        se = se.decode(errors="ignore") if isinstance(se, bytes) else se
    paths = {"stdout": os.path.join(gdir, "stdout.log"),
             "stderr": os.path.join(gdir, "stderr.log")}
    for k, text in (("stdout", so), ("stderr", se)):
        with open(paths[k], "w") as fh:
            fh.write(text)
    return {"cmd": cmd, "cwd": cwd, "exit_code": rc, "timed_out": timed_out,
            "seconds": round(time.time() - t0, 1), **paths,
            "metric": spec.get("metric"),
            "value": parse_metric(spec.get("metric_parse"), so, gdir)}


def run_task(cfg, spec: dict, task, adir: str, keep_tree: bool,
             keep_outputs: bool) -> dict:
    from lego.pipeline.construct import Construction
    client.USAGE.reset()
    bench = spec["name"]
    tdir = task_dir(core.CLONE_BASE, f"{bench}_{task.name}")
    rec = {"name": task.name, "benchmark": bench, "arm": cfg.arm_id(),
           "config": cfg.name, "stages": cfg.stages,
           "models": cfg.models.resolved(), "harness_rev": core.HARNESS_REV,
           "host": socket.gethostname(), "ts": time.time(),
           "upstream_repo": task.upstream_repo,
           "library": cfg.view.path if cfg.library else None}
    t0 = time.time()
    ws = ExternalWorkspace(task, tdir)
    try:
        ws.prepare(spec.get("env"))
        rec.update({k: ws.record.get(k) for k in ("python", "python_why",
                                                  "env_setup")})
        rec["plan"] = None
        if not ws.modules:
            p, rec["plan"] = get_plan(cfg, task, ws, bench)
            apply_plan(ws, p)
        rec["interface_source"] = ws.record.get("interface_source")
        rec["package"], rec["n_modules"] = ws.pkg, len(ws.modules)
        view = library_view(cfg, task) if cfg.library else None
        rec["library_view_size"] = len(view) if view is not None else 0
        out = Construction(cfg, ws, view).run(feedback_score)
        best = out.pop("best") or {}
        files = best.pop("files", {})
        rec.update(out)
        rec.update({"best_round": best.get("round"),
                    "feedback_passed": best.get("passed"),
                    "verdict": best.get("verdict")})
        final = ws.materialize(files, os.path.join(adir, "outputs",
                                                   safe(task.name)))
        g = run_grader(spec, task, final,
                       os.path.join(adir, "grader", safe(task.name)))
        rec["grader"] = g
        rec["metric"] = {"name": g["metric"], "value": g["value"]}
        rec["status"] = "graded" if g["exit_code"] == 0 else "grader-failed"
        if keep_tree and files:
            os.makedirs(os.path.join(adir, "trees"), exist_ok=True)
            with gzip.open(os.path.join(adir, "trees",
                                        safe(task.name) + ".json.gz"),
                           "wt") as fh:
                json.dump(files, fh)
        if not keep_outputs:
            core.rm_rf(final)
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
        rec["traceback"] = traceback.format_exc()[-3000:]
    finally:
        core.EXTRA_PYTHONPATH[:] = []
        ws.cleanup()
    rec["usage"] = client.USAGE.snapshot()
    rec["wall_seconds"] = round(time.time() - t0, 1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--config", required=True, help="lego | feedback (any "
                    "RunConfig name is accepted)")
    ap.add_argument("--model", required=True, help="backbone alias")
    ap.add_argument("--resident")
    ap.add_argument("--diagnosis")
    ap.add_argument("--library", help="filtered library (default: the "
                    "benchmark's `library` in benchmarks.yaml)")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--benchmarks")
    ap.add_argument("--keep-outputs", action="store_true")
    ap.add_argument("--no-trees", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max", type=int, default=10 ** 6)
    ap.add_argument("--no-preflight", action="store_true")
    a = ap.parse_args(argv)

    spec = load_benchmark(a.benchmark, a.benchmarks)
    bench = spec["name"]
    ov = parse_set(a.set)
    models = ov.setdefault("models", {})
    models["backbone"] = a.model
    for pos in ("resident", "diagnosis"):
        if getattr(a, pos):
            models[pos] = getattr(a, pos)
    base = C.get(a.config)
    filt = None
    if base.library:
        if base.source != "codeface":
            raise SystemExit("external runs support CodeFace-backed configs")
        lib = a.library or spec.get("library")
        if not lib or is_placeholder(lib):
            raise SystemExit(f"config {a.config} needs --library (a library "
                             f"filtered for {bench})")
        filt = check_filtered(lib, bench)
        ov.setdefault("view", {})["path"] = lib
    cfg = C.get(a.config, ov)

    tasks = sorted(load_tasks(spec), key=lambda t: t.name)
    if a.only:
        tasks = [t for t in tasks if t.name in set(a.only)]
    missing = [t.name for t in tasks if not t.grader_cmd]
    if missing:
        raise SystemExit(f"{len(missing)} task(s) have no grader_cmd "
                         f"(set it in benchmarks.yaml): {missing[:5]}")
    tasks = tasks[a.shard::a.nshards]
    adir = os.path.join(bench_dir(bench), cfg.arm_id())
    os.makedirs(adir, exist_ok=True)
    with open(os.path.join(adir, "config.json"), "w") as fh:
        json.dump({**cfg.to_dict(), "benchmark": bench,
                   "metric": spec.get("metric"), "library_filter": filt},
                  fh, indent=1)
    done = {} if a.force else load_records(adir)
    todo = [t for t in tasks if t.name not in done][:a.max]
    print(f"[external] {bench} {cfg.arm_id()} stages={cfg.stages} "
          f"models={cfg.models.resolved()} | shard {a.shard}/{a.nshards}: "
          f"{len(tasks)} tasks, {len(todo)} to run", flush=True)
    if not todo:
        return 0
    if not a.no_preflight:
        client.preflight(cfg.models.resolved().values())
    configure_harness(f"ext_{bench}_{cfg.arm_id()}_s{a.shard}")
    out = os.path.join(adir, "records.jsonl" if a.nshards == 1
                       else f"records.s{a.shard}.jsonl")
    for i, t in enumerate(todo, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(todo)}] {bench}/{t.name}\n{'=' * 70}",
              flush=True)
        rec = run_task(cfg, spec, t, adir, not a.no_trees, a.keep_outputs)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        m = rec.get("metric") or {}
        print(f"  => {rec['status']} {m.get('name')}={m.get('value')} "
              f"({rec.get('wall_seconds', 0):.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
