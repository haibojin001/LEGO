"""Offline coverage audit for an external benchmark's targets.

    python -m external.coverage --benchmark repocraft \
        --library codeface_repocraft [--config lego] [--set k=v ...] \
        [--source auto|stubs|originals|plan] [--plans DIR] \
        [--assess --model ALIAS] [--out results/external_coverage] \
        [--benchmarks external/benchmarks.yaml]

For every target module of every task: does the filtered library (the output
of ``external.filter_library``) hold at least one candidate for it? No
construction runs and no environment is built. The reported number is the
fraction of target modules with at least one candidate; with ``--assess`` a
candidate must also be judged SUITABLE by its resident model (Eq. 1), as in
``lego.analysis.coverage``.

Where the target modules come from (``--source``):

  stubs      the interface or skeleton files the benchmark ships
  originals  the upstream sources (``original_sources``), reduced to interface
             stubs exactly as the pipeline reduces a LEGO-REPO module; read
             here only, never by a system
  plan       the plans written by ``external.run`` (``--plans``; the default is
             ``runs/external/<benchmark>/plans/<model>``)
  auto       stubs, else originals, else plan (per task)

Retrieval uses the identifier-derived query of ``lego.analysis.coverage`` for
every source, so the audit needs no model unless ``--assess`` is given.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
from collections import defaultdict

from lego import config as C
from lego.pipeline.decompose import build_requirements, heuristic_query

from external.common import (ExternalWorkspace, apply_plan, load_benchmark,
                             load_tasks, target_sources)

SOURCES = ("auto", "stubs", "originals", "plan")
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


class _Offline:
    """The subset of a workspace that build_requirements reads."""

    def __init__(self, task, modules: dict, pkg: str, deps=()):
        self.task, self.pkg = task, pkg
        self.modules, self.orig_src = sorted(modules), dict(modules)
        self.proj_dir = task.workdir or os.devnull
        self.record = {"deps": list(deps)}


def offline_workspace(task, source: str, plans_dir: str | None):
    """(workspace-like object, source used) or (None, reason)."""
    order = ("stubs", "originals", "plan") if source == "auto" else (source,)
    for s in order:
        if s == "stubs":
            ws = ExternalWorkspace(task)
            if ws.modules:
                return ws, s
        elif s == "originals":
            src = target_sources(task)
            if src:
                return _Offline(task, src, task.target_package
                                or task.name), s
        elif s == "plan" and plans_dir:
            p = os.path.join(plans_dir, _SAFE.sub("_", task.name) + ".json")
            if os.path.exists(p):
                ws = ExternalWorkspace(task)
                apply_plan(ws, json.load(open(p))["plan"])
                return ws, s
    return None, "no-" + source


def audit(tasks, lib_path: str, cfg, source: str = "auto",
          plans_dir: str | None = None, assess_model: str | None = None,
          workers: int = 8) -> list[dict]:
    from lego.library import resident
    from lego.library.codeface import CodeFace
    from lego.llm.client import LLM
    lib = CodeFace(lib_path, cfg.embedder)
    filt = {"kinds": cfg.view.kinds, "exclude": cfg.view.exclude,
            "exclude_repos": cfg.view.exclude_repos,
            "near_dup_thresh": cfg.view.near_dup_thresh,
            "subset_frac": cfg.view.subset_frac,
            "subset_seed": cfg.view.subset_seed}
    rows = []
    for t in tasks:
        ws, used = offline_workspace(t, source, plans_dir)
        if ws is None:
            rows.append({"benchmark": t.benchmark, "task": t.name,
                         "module": None, "n_cand": 0, "covered": False,
                         "source": used})
            print(f"[coverage] {t.name}: skipped ({used})", flush=True)
            continue
        view = lib.view(t, filt)
        reqs = build_requirements(ws, cfg.interface)
        for r in reqs:
            r.capability = r.retrieval_request = heuristic_query(r, ws.pkg)

        def one(r):
            q = f"{r.retrieval_request}\n{' '.join(r.exports[:20])}"
            cands = view.search(q, top_m=cfg.top_m)
            ok = bool(cands)
            if ok and assess_model:
                llm = LLM(assess_model, "resident")
                ok = any(resident.assess(llm, p, r).suitable for p in cands)
            return {"benchmark": t.benchmark, "task": t.name,
                    "module": r.module, "n_cand": len(cands), "covered": ok,
                    "source": used}
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            got = list(pool.map(one, reqs))
        rows += got
        print(f"[coverage] {t.name} ({used}): "
              f"{sum(x['covered'] for x in got)}/{len(got)}", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    mods = [r for r in rows if r["module"] is not None]
    per = defaultdict(lambda: [0, 0])
    for r in mods:
        per[r["task"]][0] += r["covered"]
        per[r["task"]][1] += 1
    c = sum(r["covered"] for r in mods)
    return {"all": {"covered": c, "modules": len(mods),
                    "coverage": c / len(mods) if mods else 0.0},
            "tasks": {k: {"covered": a, "modules": n,
                          "coverage": a / n if n else 0.0}
                      for k, (a, n) in sorted(per.items())},
            "skipped": sorted({r["task"] for r in rows if r["module"] is None}),
            "sources": sorted({r["source"] for r in rows})}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--library", required=True,
                    help="filtered library (external.filter_library --out)")
    ap.add_argument("--config", default="lego")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--source", choices=SOURCES, default="auto")
    ap.add_argument("--plans")
    ap.add_argument("--assess", action="store_true")
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--out", default="results/external_coverage")
    ap.add_argument("--benchmarks")
    a = ap.parse_args(argv)
    from lego.run import parse_set
    rep = os.path.join(a.library, "filter_report.json")
    if not os.path.exists(rep) or json.load(open(rep)).get(
            "benchmark") != a.benchmark:
        raise SystemExit(f"{a.library} is not a library filtered for "
                         f"{a.benchmark} (run external.filter_library first)")
    cfg = C.get(a.config, parse_set(a.set))
    spec = load_benchmark(a.benchmark, a.benchmarks)
    tasks = load_tasks(spec)
    plans = a.plans or os.path.join(os.environ.get("LEGO_RUNS", "runs"),
                                    "external", a.benchmark, "plans",
                                    _SAFE.sub("_", cfg.models.backbone))
    rows = audit(tasks, a.library, cfg, a.source, plans,
                 a.model if a.assess else None)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.benchmark}_{a.source}{'_assess' if a.assess else ''}"
    with open(os.path.join(a.out, f"{tag}.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summ = summarize(rows)
    summ.update(benchmark=a.benchmark, library=a.library,
                assess_model=a.model if a.assess else None)
    with open(os.path.join(a.out, f"{tag}.summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    s = summ["all"]
    print(f"[coverage] {a.benchmark}: {s['covered']}/{s['modules']} target "
          f"modules have >= 1 candidate -> "
          f"{os.path.join(a.out, tag + '.summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
