"""Offline module-level reachability audit (RQ2).

For every target module of every task in a split, ask whether the library view a
configuration would use contains at least one candidate for it. No construction
is run and no repository is cloned: requirements are built from the frozen task
directory (``original/`` is read only to derive interface stubs, exactly as the
pipeline does).

    python -m lego.analysis.coverage --split lego_repo_522 --config lego \
        [--assess --model ALIAS] [--out results/coverage]

Without ``--assess`` a module is covered when retrieval returns any candidate;
with ``--assess`` it must also be judged SUITABLE by the candidate's resident
model (Eq. 1), which is the definition used for the reported coverage.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from collections import defaultdict

from lego import config as C
from lego.harness import core
from lego.harness.task import load_index, select
from lego.pipeline.decompose import decompose


class OfflineWorkspace:
    """The subset of Workspace that build_requirements reads."""

    def __init__(self, task):
        self.task = task
        self.pkg = task.package.split(".")[-1] if task.package else task.name
        root = os.path.join(task.dir, "original")
        self.modules, self.orig_src = [], {}
        for r, _d, fs in os.walk(root):
            for f in fs:
                if (f.endswith(".py") and not core.is_test_file(f)
                        and f != "conftest.py"):
                    p = os.path.join(r, f)
                    rel = os.path.relpath(p, root)
                    self.modules.append(rel)
                    self.orig_src[rel] = open(p, errors="ignore").read()
        self.modules.sort()
        self.proj_dir = root
        tj = os.path.join(task.dir, "task.json")
        deps = json.load(open(tj)).get("deps", []) if os.path.exists(tj) else []
        self.record = {"deps": deps}


def audit(split, cfg, assess_model=None, workers=8, index=None,
          decompose_model=None, skip_tasks=None, on_task=None):
    from lego.library.codeface import CodeFace
    from lego.library import resident
    from lego.llm.client import LLM
    lib = CodeFace(cfg.view.path, cfg.embedder,
                   allow_unvalidated=cfg.view.allow_unvalidated)
    filt = {"kinds": cfg.view.kinds, "exclude": cfg.view.exclude,
            "exclude_repos": cfg.view.exclude_repos,
            "near_dup_thresh": cfg.view.near_dup_thresh,
            "subset_frac": cfg.view.subset_frac, "subset_seed": cfg.view.subset_seed}
    rows = []
    for t in select(load_index(index), split):
        if t.name in (skip_tasks or ()):
            continue
        ws = OfflineWorkspace(t)
        view = lib.view(t, filt, target_sources=ws.orig_src)
        backbone = (LLM(decompose_model or cfg.models.resolved()["backbone"],
                        "backbone", cfg.temperature)
                    if cfg.decompose == "llm" else None)
        reqs = decompose(backbone, ws, cfg.decompose, cfg.interface,
                         t.statement())

        def one(r):
            q = (f"{r.retrieval_request}\n{r.capability}\n"
                 f"{' '.join(r.exports[:20])}")
            cands = view.search(q, top_m=cfg.top_m)
            ok = bool(cands)
            if ok and assess_model:
                llm = LLM(assess_model, "resident")
                ok = any(resident.assess(llm, p, r).suitable for p in cands)
            return {"task": t.name, "band": t.band, "domain": t.domain,
                    "module": r.module, "capability": r.capability,
                    "retrieval_request": r.retrieval_request,
                    "exports": r.exports[:20],
                    "n_cand": len(cands), "covered": ok}
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            task_rows = list(pool.map(one, reqs))
        rows.extend(task_rows)
        if on_task is not None:
            on_task(t.name, task_rows)
        print(f"[coverage] {t.name}: "
              f"{sum(x['covered'] for x in task_rows)}/{len(reqs)}",
              flush=True)
    return rows


def summarize(rows):
    out = {}
    for key in ("band", "domain"):
        g = defaultdict(lambda: [0, 0])
        for r in rows:
            g[r[key]][0] += r["covered"]
            g[r[key]][1] += 1
        out[key] = {k: {"covered": c, "modules": n, "coverage": c / n if n else 0}
                    for k, (c, n) in sorted(g.items())}
    c = sum(r["covered"] for r in rows)
    out["all"] = {"covered": c, "modules": len(rows),
                  "coverage": c / len(rows) if rows else 0}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=os.environ.get("LEGO_SPLIT",
                                                      "lego_repo_522"))
    ap.add_argument("--config", default="lego")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--assess", action="store_true")
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--index")
    ap.add_argument("--out", default="results/coverage")
    a = ap.parse_args(argv)
    from lego.run import parse_set
    cfg = C.get(a.config, parse_set(a.set))
    os.makedirs(a.out, exist_ok=True)
    tag = f"{cfg.arm_id()}{'_assess' if a.assess else ''}"
    row_path = os.path.join(a.out, f"{tag}.jsonl")
    progress_path = os.path.join(a.out, f"{tag}.progress")
    done = set(open(progress_path).read().splitlines()) if os.path.exists(
        progress_path) else set()
    if not os.path.exists(row_path):
        done.clear()
        open(progress_path, "w").close()
    existing = []
    if os.path.exists(row_path):
        with open(row_path) as fh:
            existing = [json.loads(line) for line in fh if line.strip()]
        existing = [r for r in existing if r.get("task") in done]
        with open(row_path, "w") as fh:
            for r in existing:
                fh.write(json.dumps(r) + "\n")

    with open(row_path, "a") as rows_fh, open(progress_path, "a") as prog_fh:
        def save_task(name, task_rows):
            rows_fh.write("".join(json.dumps(r) + "\n" for r in task_rows))
            rows_fh.flush()
            prog_fh.write(name + "\n")
            prog_fh.flush()

        new = audit(a.split, cfg, a.model if a.assess else None,
                    index=a.index, decompose_model=a.model,
                    skip_tasks=done, on_task=save_task)
    rows = existing + new
    with open(os.path.join(a.out, f"{tag}.summary.json"), "w") as fh:
        json.dump(summarize(rows), fh, indent=1)
    print(os.path.join(a.out, f"{tag}.summary.json"))


if __name__ == "__main__":
    main()
