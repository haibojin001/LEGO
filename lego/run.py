"""Run one configuration (arm) over a task list.

    python -m lego.run --config lego --split lego_repo_522 \
        --shard 0 --nshards 16 [--backbone M] [--resident M] [--diagnosis M] \
        [--budget B] [--set key=value ...] [--only t1 t2]

Records go to ``runs/arms/<arm_id>/records.s<shard>.jsonl`` (one JSON object per
task, appended; newest line per task wins). The arm directory also holds
``config.json`` and, unless ``--no-trees``, the best generated tree of every task
under ``trees/<task>.json.gz`` (used by the special-casing audit and case
studies).

The library is read-only during evaluation. Resuming is automatic: tasks that
already have a record in this arm are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import socket
import sys
import tempfile
import time
import traceback

from lego import config as C
from lego.harness import core, grading
from lego.harness.prepare import NotGradeable, cleanup, prepare, task_dir
from lego.harness.task import band_interleave, load_index, select
from lego.llm import client

RUNS = os.environ.get("LEGO_RUNS", "runs")


def arm_dir(cfg) -> str:
    return os.path.join(RUNS, "arms", cfg.arm_id())


def load_records(d: str) -> dict:
    out = {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.startswith("records") and f.endswith(".jsonl"):
            for line in open(os.path.join(d, f)):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("name"):
                    prev = out.get(r["name"])
                    if prev is None or r.get("ts", 0) >= prev.get("ts", 0):
                        out[r["name"]] = r
    return out


def configure_harness(tag: str):
    scratch = os.environ.get("LEGO_SCRATCH") or tempfile.gettempdir()
    os.makedirs(scratch, exist_ok=True)
    core.VENV = os.path.join(scratch, f"lego_venv_{tag}")
    tmp = os.path.join(scratch, f"lego_tmp_{tag}_{os.getpid()}")
    os.makedirs(tmp, exist_ok=True)
    os.environ["TMPDIR"] = tmp
    tempfile.tempdir = None
    core.CLONE_BASE = os.path.join(scratch, f"lego_clones_{tag}")
    os.makedirs(core.CLONE_BASE, exist_ok=True)


_VIEWS = {}


def check_runtime_assets(cfg):
    """Fail before model calls when a library arm has no usable input data."""
    if not cfg.library:
        return
    if cfg.source == "file_rag":
        root = cfg.view.path if cfg.view.path != "codeface" else "file_rag"
        index = os.path.join(root, "index.jsonl")
        if not os.path.isfile(index) or os.path.getsize(index) == 0:
            raise SystemExit(
                f"File RAG index missing or empty: {index}; run "
                "python -m lego.library.file_rag build --library codeface "
                "--out file_rag")
    else:
        if not glob.glob(os.path.join(cfg.view.path, "*", "primitive.json")):
            raise SystemExit(
                f"CodeFace missing or empty: {cfg.view.path}; supply the "
                "validated library or run python -m lego.mining.mine")
        from lego.library.codeface import CodeFace
        lib = CodeFace(cfg.view.path, cfg.embedder,
                       allow_unvalidated=cfg.view.allow_unvalidated)
        if not lib.prims:
            raise SystemExit(
                f"CodeFace {cfg.view.path} has no usable primitives; paper "
                "runs require validated entries with carried tests. For a "
                "legacy proxy use LEGO_ALLOW_UNVALIDATED=1 explicitly.")
        if cfg.view.kinds and not any(
                p.kind in cfg.view.kinds for p in lib.prims):
            raise SystemExit(
                f"CodeFace {cfg.view.path} has no entries of kinds "
                f"{cfg.view.kinds}; check the library provenance")


def library_view(cfg, task, ws):
    if not cfg.library:
        return None
    key = (cfg.source, cfg.view.path, cfg.embedder,
           cfg.view.allow_unvalidated)
    if key not in _VIEWS:
        if cfg.source == "file_rag":
            from lego.library.file_rag import FileIndex
            _VIEWS[key] = FileIndex(cfg.view.path if cfg.view.path != "codeface"
                                    else "file_rag", cfg.embedder)
        else:
            from lego.library.codeface import CodeFace
            _VIEWS[key] = CodeFace(
                cfg.view.path, cfg.embedder,
                allow_unvalidated=cfg.view.allow_unvalidated)
    lib = _VIEWS[key]
    filt = {"kinds": cfg.view.kinds, "exclude": cfg.view.exclude,
            "exclude_repos": cfg.view.exclude_repos,
            "near_dup_thresh": cfg.view.near_dup_thresh,
            "subset_frac": cfg.view.subset_frac,
            "subset_seed": cfg.view.subset_seed}
    return lib.view(task, filt, target_sources=ws.orig_src)


def run_task(cfg, task, keep_tree: bool, adir: str) -> dict:
    from lego.pipeline.construct import Construction
    client.USAGE.reset()
    tdir = task_dir(core.CLONE_BASE, task.name)
    rec = {"name": task.name, "arm": cfg.arm_id(), "config": cfg.name,
           "stages": cfg.stages, "models": cfg.models.resolved(),
           "band": task.band, "domain": task.domain, "track": task.track,
           "harness_rev": core.HARNESS_REV, "host": socket.gethostname(),
           "ts": time.time(), "score": 0.0}
    t0 = time.time()
    try:
        ws = prepare(task, tdir)
        rec.update({k: ws.record.get(k) for k in
                    ("commit", "python", "python_why", "env_setup", "baseline",
                     "ceiling_run", "floor_run", "ceiling_drift", "n_modules",
                     "single_module")})
        score_fn = grading.make_score_fn(task, ws)
        rec.update(score_fn.bounds)
        view = library_view(cfg, task, ws)
        rec["library_view_size"] = len(view) if view is not None else 0
        out = Construction(cfg, ws, view).run(score_fn)
        best = out.pop("best") or {}
        files = best.pop("files", {})
        rec.update(out)
        rec.update({"best_round": best.get("round"), "passed": best.get("passed"),
                    "score": best.get("score", 0.0),
                    "verdict": best.get("verdict"),
                    "exercised": best.get("exercised")})
        rec["status"] = "done" if best.get("verdict") == "ok" else (
            core._status_for(best.get("verdict")) if best else "error")
        if keep_tree and files:
            os.makedirs(os.path.join(adir, "trees"), exist_ok=True)
            with gzip.open(os.path.join(adir, "trees", task.name + ".json.gz"),
                           "wt") as fh:
                json.dump(files, fh)
    except NotGradeable as e:
        rec["status"] = e.args[0]
        if len(e.args) > 1:
            rec.update({k: v for k, v in e.args[1].items()
                        if k in ("commit", "python", "baseline", "n_modules")})
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
        rec["traceback"] = traceback.format_exc()[-3000:]
    finally:
        cleanup(tdir)
    rec["usage"] = client.USAGE.snapshot()
    rec["wall_seconds"] = round(time.time() - t0, 1)
    return rec


def parse_set(pairs):
    out = {}
    for kv in pairs or []:
        k, _, v = kv.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def build_config(args) -> C.RunConfig:
    ov = parse_set(args.set)
    models = ov.setdefault("models", {})
    for pos in ("backbone", "resident", "diagnosis"):
        if getattr(args, pos):
            models[pos] = getattr(args, pos)
    if args.budget:
        ov["budget"] = args.budget
    if args.replicate:
        ov["replicate"] = args.replicate
    return C.get(args.config, ov)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default=os.environ.get("LEGO_SPLIT",
                                                      "lego_repo_522"))
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--index", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--backbone")
    ap.add_argument("--resident")
    ap.add_argument("--diagnosis")
    ap.add_argument("--budget", type=int)
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--set", nargs="*", help="field=value overrides, e.g. "
                    "view.exclude='[\"same_repo\"]' top_m=4")
    ap.add_argument("--max", type=int, default=10 ** 6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-trees", action="store_true")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--print-arm", action="store_true",
                    help="print the arm id and config, then exit")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    adir = arm_dir(cfg)
    if args.print_arm:
        print(cfg.arm_id())
        print(json.dumps(cfg.to_dict(), indent=1))
        return
    if cfg.interface != "stub":
        print("[warn] interface != stub exposes original implementations; "
              "records are not valid LEGO-REPO results", flush=True)
    os.makedirs(adir, exist_ok=True)
    with open(os.path.join(adir, "config.json"), "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=1)

    tasks = band_interleave(select(load_index(args.index), args.split, args.only))
    tasks = [t for t in tasks if t.name not in core.HOLD_REPOS]
    tasks = tasks[args.shard::args.nshards]
    done = {} if args.force else load_records(adir)
    todo = [t for t in tasks if t.name not in done][:args.max]
    print(f"[arm] {cfg.arm_id()} stages={cfg.stages} models={cfg.models.resolved()}"
          f" budget={cfg.budget} | shard {args.shard}/{args.nshards}: "
          f"{len(tasks)} tasks, {len(todo)} to run", flush=True)
    if not todo:
        return
    check_runtime_assets(cfg)
    if not args.no_preflight:
        client.preflight(cfg.models.resolved().values())
    configure_harness(f"{cfg.arm_id()}_s{args.shard}")
    out = os.path.join(adir, f"records.s{args.shard}.jsonl")
    for i, t in enumerate(todo, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(todo)}] {t.name} (D{t.band}, "
              f"{t.n_modules} modules)\n{'=' * 70}", flush=True)
        rec = run_task(cfg, t, not args.no_trees, adir)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  => {rec['status']} score={rec.get('score', 0):.3f} "
              f"({rec.get('wall_seconds', 0):.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
