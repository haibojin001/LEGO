"""Test-targeted special-casing audit.

Native tests are both specification and grading key, so reconstructions are
screened for code that special-cases the tests (hard-coded expected values,
branches on test names or fixture paths, reading test files at runtime).

1. Sample (band-stratified, fixed seed) and export audit packets:

       python -m lego.analysis.audit sample --arm-dir runs/arms/<arm> \
           --split lego_repo_522 --n 120 --out audit/<arm>

   Each packet ``audit/<arm>/<task>/`` holds the delivered tree, the task's
   native tests and a ``flags.txt`` of automatic heuristics hits (to direct
   attention, never to decide). ``annotations_template.csv`` has one row per
   task with columns ``task, special_casing (0/1), note``.

2. Two annotators fill copies of the template independently
   (``annotator_a.csv``, ``annotator_b.csv``); then

       python -m lego.analysis.audit score audit/<arm>/annotator_a.csv \
           audit/<arm>/annotator_b.csv

   reports each annotator's rate, the adjudicated rate (disagreements count
   as flagged unless an ``adjudicated.csv`` is supplied), Cohen's kappa, and a
   Wilson 95% interval.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import re
import shutil

from lego.harness.task import load_index, select

_HEUR = [
    ("test-name branch", re.compile(r"(PYTEST_CURRENT_TEST|['\"]test_\w+['\"])")),
    ("reads test files", re.compile(r"open\([^)]*tests?/|conftest")),
    ("inspect caller", re.compile(r"inspect\.stack|sys\._getframe")),
    ("pytest import", re.compile(r"^\s*import pytest|from _pytest", re.M)),
]


def sample(arm_dir, split, n, out, seed=0, index=None):
    tasks = select(load_index(index), split)
    trees = os.path.join(arm_dir, "trees")
    have = [t for t in tasks if os.path.exists(os.path.join(trees, t.name + ".json.gz"))]
    by = {}
    for t in have:
        by.setdefault(t.band, []).append(t)
    rng = random.Random(f"audit|{seed}")
    total = len(have) or 1
    picked = []
    for b, ts in sorted(by.items()):
        k = max(1, round(n * len(ts) / total))
        picked += rng.sample(ts, min(k, len(ts)))
    picked = picked[:n]
    os.makedirs(out, exist_ok=True)
    for t in picked:
        files = json.load(gzip.open(os.path.join(trees, t.name + ".json.gz"), "rt"))
        d = os.path.join(out, t.name)
        os.makedirs(os.path.join(d, "delivered"), exist_ok=True)
        flags = []
        for rel, code in files.items():
            p = os.path.join(d, "delivered", rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(code)
            for lab, rx in _HEUR:
                if rx.search(code):
                    flags.append(f"{rel}: {lab}")
        tsrc = os.path.join(t.dir, "tests")
        if os.path.isdir(tsrc):
            shutil.copytree(tsrc, os.path.join(d, "tests"), dirs_exist_ok=True)
        open(os.path.join(d, "flags.txt"), "w").write("\n".join(flags) + "\n")
    with open(os.path.join(out, "annotations_template.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "band", "special_casing", "note"])
        for t in picked:
            w.writerow([t.name, t.band, "", ""])
    print(f"{len(picked)} packets -> {out}")


def _read(p):
    return {r["task"]: int(r["special_casing"]) for r in csv.DictReader(open(p))
            if r.get("special_casing", "").strip() != ""}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def score(a_path, b_path, adj_path=None):
    a, b = _read(a_path), _read(b_path)
    keys = sorted(set(a) & set(b))
    n = len(keys)
    po = sum(a[k] == b[k] for k in keys) / n if n else 0
    pa, pb = (sum(a[k] for k in keys) / n, sum(b[k] for k in keys) / n) if n else (0, 0)
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    adj = _read(adj_path) if adj_path else {}
    final = {k: adj.get(k, max(a[k], b[k])) for k in keys}
    k = sum(final.values())
    lo, hi = wilson(k, n)
    res = {"n": n, "rate_a": pa, "rate_b": pb, "kappa": kappa,
           "flagged": k, "rate": k / n if n else 0, "wilson95": [lo, hi]}
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--arm-dir", required=True)
    s.add_argument("--split", default=os.environ.get("LEGO_SPLIT",
                                                     "lego_repo_522"))
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument("--index")
    c = sub.add_parser("score")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--adjudicated")
    x = ap.parse_args(argv)
    if x.cmd == "sample":
        sample(x.arm_dir, x.split, x.n, x.out, x.seed, x.index)
    else:
        score(x.a, x.b, x.adjudicated)


if __name__ == "__main__":
    main()
