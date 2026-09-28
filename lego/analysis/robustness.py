"""Scoring-robustness analyses (appendix "Scoring and provenance checks").

    python -m lego.analysis.robustness experiments/rq3_paradigms/experiment.yaml \
        --a LEGO --b Feedback [--out results]

For the pair (a, b) of arm display names, recomputes the paired contrast under:

  bands_default      the benchmark's difficulty bands (Eq. 6)
  bins_modules       five equal-count bins of module count only
  bins_ceiling       five equal-count bins of identity-ceiling size only
  span_weighted      task weights proportional to c(T) - f(T)
  zero_floor_only    tasks with f(T) = 0
  raw_pass_rate      Pi(G) / c(T) without floor subtraction

and reports floor statistics (tasks with non-zero floor, span distribution).
"""

from __future__ import annotations

import argparse
import json
import os

from lego import launch
from lego import run as R
from lego.analysis import metrics as M
from lego.harness.grading import admissible
from lego.harness.task import load_index, select


def _bins(tasks, key, k=5):
    s = sorted(tasks, key=key)
    return {t.name: 1 + i * k // max(len(s), 1) for i, t in enumerate(s)}


def _weighted(ra, rb, tasks, w):
    num = sum(w(t) * (M.scores(ra, [t.name])[0] - M.scores(rb, [t.name])[0])
              for t in tasks)
    den = sum(w(t) for t in tasks) or 1
    return num / den


def _raw(rec, t):
    if not rec or not admissible(rec) or not t.ceiling:
        return 0.0
    return min((rec.get("passed") or 0) / t.ceiling, 1.0)


def analyse(exp_path, a, b):
    exp = launch.load(exp_path)
    tasks = select(load_index(exp.get("index")), exp["split"])
    recs = {cfg.display: R.load_records(R.arm_dir(cfg)) for _n, _o, cfg in
            launch.arms(exp)}
    ra, rb = recs[a], recs[b]
    out = {"pair": [a, b], "floor_nonzero": sum(1 for t in tasks if t.floor > 0),
           "n": len(tasks)}
    for lab, mapping in (("bands_default", {t.name: t.band for t in tasks}),
                         ("bins_modules", _bins(tasks, lambda t: t.n_modules)),
                         ("bins_ceiling", _bins(tasks, lambda t: t.ceiling))):
        out[lab] = {}
        for g in sorted(set(mapping.values())):
            p = M.paired(ra, rb, tasks, 2000,
                         subset=lambda t, g=g, m=mapping: m[t.name] == g)
            out[lab][g] = {"n": p["n"], "a": p["a"], "b": p["b"],
                           "delta": p["delta"], "ci95": p["ci95"]}
    out["span_weighted"] = _weighted(ra, rb, tasks,
                                     lambda t: max(t.ceiling - t.floor, 0))
    zf = [t for t in tasks if t.floor == 0]
    out["zero_floor_only"] = M.paired(ra, rb, zf, 2000)
    out["raw_pass_rate"] = {
        a: sum(_raw(ra.get(t.name), t) for t in tasks) / (len(tasks) or 1),
        b: sum(_raw(rb.get(t.name), t) for t in tasks) / (len(tasks) or 1)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="results")
    x = ap.parse_args(argv)
    res = analyse(x.experiment, x.a, x.b)
    name = launch.load(x.experiment)["name"]
    os.makedirs(os.path.join(x.out, name), exist_ok=True)
    p = os.path.join(x.out, name, f"robustness_{x.a}_vs_{x.b}.json".replace(" ", "_"))
    json.dump(res, open(p, "w"), indent=1, default=str)
    print(p)


if __name__ == "__main__":
    main()
