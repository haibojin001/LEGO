"""Build the tables of one experiment from its arms' records.

    python -m lego.analysis.report experiments/rq3_paradigms/experiment.yaml \
        [--out results] [--boot 10000]

The experiment file's ``tables:`` list says what to emit; each entry has a
``type`` and options:

  main      rows = arms; D1..D5, All, Dead@0, Ceil@1, $/task
  paired    every arm against ``ref`` (an arm display name): delta, relative
            delta, paired bootstrap 95% CI, wins/ties/losses, sign-test p;
            ``by_band: true`` adds one row per band; ``splits: [..]`` adds one
            row per named split (e.g. the star strata)
  pairs     for sweeps: pair arm ``a`` with arm ``b`` at every sweep point
            (e.g. LEGO vs Feedback per backbone)
  domains   rows = arms, columns = domains
  funnel    retrieval/activation/adaptation counts per band for ``arm``
  ladder    attempt-budget ladder for ``arm`` (``budgets: [1,2,3,4,5]``)
  cost      $/task by role and median wall time per arm

Every table is written as Markdown, CSV and a LaTeX tabular body under
``<out>/<experiment>/``. Numbers are whatever the records contain; nothing is
filled in when an arm has not run (its coverage column says so).
"""

from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace

from lego import launch
from lego import run as R
from lego.analysis import metrics as M
from lego.harness import core
from lego.harness.task import load_index, load_split, select


def _f(x, nd=4):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


class Table:
    def __init__(self, name, header):
        self.name, self.header, self.rows = name, header, []

    def add(self, row):
        self.rows.append([("" if c is None else c) for c in row])

    def write(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, self.name)
        with open(base + ".csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.header)
            w.writerows(self.rows)
        with open(base + ".md", "w") as fh:
            fh.write("| " + " | ".join(self.header) + " |\n")
            fh.write("|" + "---|" * len(self.header) + "\n")
            for r in self.rows:
                fh.write("| " + " | ".join(map(str, r)) + " |\n")
        with open(base + ".tex", "w") as fh:
            esc = lambda s: str(s).replace("%", r"\%").replace("_", r"\_")
            fh.write(" & ".join(esc(h) for h in self.header) + r" \\" + "\n\\midrule\n")
            for r in self.rows:
                fh.write(" & ".join(esc(c) for c in r) + r" \\" + "\n")
        return base + ".md"


def prefix_records(records: dict, budget: int) -> dict:
    """Score an already executed trajectory at its first ``budget`` rounds."""
    out = {}
    for name, record in records.items():
        rounds = [r for r in record.get("trajectory") or []
                  if r.get("round", 0) <= budget]
        if not rounds:
            continue
        admissible = [r for r in rounds if r.get("verdict") == "ok"]
        best = max(admissible, key=lambda r: r.get("score", 0.0),
                   default=rounds[-1])
        derived = dict(record)
        derived.update(trajectory=rounds, best_round=best.get("round"),
                       passed=best.get("passed"), score=best.get("score", 0.0),
                       verdict=best.get("verdict"),
                       status=("done" if best.get("verdict") == "ok"
                               else core._status_for(best.get("verdict"))),
                       prefix_of=record.get("arm"))
        derived.pop("usage", None)  # full-run calls are not prefix-only costs
        out[name] = derived
    return out


def build(exp_path: str, out: str, n_boot: int) -> list[str]:
    exp = launch.load(exp_path)
    tasks = select(load_index(exp.get("index")), exp["split"])
    A = launch.arms(exp)
    recs = {cfg.display: R.load_records(R.arm_dir(cfg)) for _n, _o, cfg in A}
    by_disp = {cfg.display: cfg for _n, _o, cfg in A}
    prefix_order = []
    for pref in exp.get("prefixes") or []:
        sources = [cfg for _n, _o, cfg in A
                   if cfg.base_display == pref["of"]]
        if not sources:
            raise ValueError(
                f"invalid trajectory prefix {pref['display']!r} "
                f"of {pref['of']!r}")
        for cfg in sources:
            suffix = cfg.display[len(cfg.base_display):]
            display = pref["display"] + suffix
            if display in recs:
                raise ValueError(f"duplicate trajectory prefix {display!r}")
            recs[display] = prefix_records(
                recs[cfg.display], int(pref.get("budget", 1)))
            by_disp[display] = SimpleNamespace(
                stages=pref["stages"], prefix=True)
            prefix_order.append(display)
    if exp.get("prefixes"):
        order = prefix_order + [
            cfg.display for _n, _o, cfg in A]
        recs = {d: recs[d] for d in order}
    bands = sorted({t.band for t in tasks})
    odir = os.path.join(out, exp["name"])
    written = []
    specs = exp.get("tables") or [{"type": "main"}]
    for i, spec in enumerate(specs):
        typ = spec["type"]
        name = spec.get("name") or f"{typ}{'' if i == 0 else i}"
        if typ == "main":
            T = Table(name, ["arm", "stages"] + [f"D{b}" for b in bands] +
                      ["All", "Dead@0", "Ceil@1", "$/task", "records"])
            for d, r in recs.items():
                s = M.summary(r, tasks)
                c = None if getattr(by_disp[d], "prefix", False) else M.cost(r, tasks)
                T.add([d, by_disp[d].stages] +
                      [_f(s["band"].get(b, (None,))[0], 3) for b in bands] +
                      [_f(s["delivery"]), _pct(s["dead0"]), _pct(s["ceil1"]),
                       _f(c["usd_per_task"], 2) if c else "",
                       f"{s['records']}/{s['n']}"])
        elif typ == "paired":
            ref = spec["ref"]
            T = Table(name, ["arm", "vs", "subset", "n", "arm score", "ref score",
                             "delta", "rel", "95% CI", "W/T/L", "sign p"])
            for d, r in recs.items():
                if d == ref:
                    continue
                subsets = [("all", None)]
                for sp in spec.get("splits") or []:
                    members = set(load_split(sp))
                    subsets.append((sp, (lambda m: lambda t: t.name in m)(members)))
                if spec.get("by_band"):
                    subsets += [(f"D{b}", (lambda b: lambda t: t.band == b)(b))
                                for b in bands]
                for lab, sub in subsets:
                    p = M.paired(r, recs[ref], tasks, n_boot, subset=sub)
                    if not p.get("n"):
                        continue
                    T.add([d, ref, lab, p["n"], _f(p["a"]), _f(p["b"]),
                           _f(p["delta"]), _pct(p["rel"]),
                           f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]",
                           f"{p['wins']}/{p['ties']}/{p['losses']}",
                           _f(p["sign_p"], 3)])
        elif typ == "pairs":
            a, b = spec["a"], spec["b"]
            T = Table(name, ["sweep point", f"{b}", f"{a}", "delta", "rel",
                             "95% CI"])
            pts = {}
            for _n, _o, cfg in A:
                key = tuple(sorted(cfg.sweep_point.items()))
                pts.setdefault(key, {})[cfg.base_display] = cfg.display
            for key, m in pts.items():
                if a in m and b in m:
                    p = M.paired(recs[m[a]], recs[m[b]], tasks, n_boot)
                    T.add([", ".join(f"{k}={v}" for k, v in key), _f(p["b"]),
                           _f(p["a"]), _f(p["delta"]), _pct(p["rel"]),
                           f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"])
        elif typ == "domains":
            doms = sorted({t.domain for t in tasks})
            T = Table(name, ["arm"] + doms)
            for d, r in recs.items():
                s = M.summary(r, tasks)
                T.add([d] + [_f(s["domain"].get(x, (None,))[0], 3) for x in doms])
        elif typ == "funnel":
            arm = spec["arm"]
            f = M.funnel(recs[arm], tasks)
            keys = ["modules", "requirements", "query", "cand", "retain",
                    "activated", "adapt_ok", "adapt_rejected", "messages"]
            T = Table(name, ["band"] + keys)
            for b in bands + ["all"]:
                row = f.get(b, {})
                T.add([f"D{b}" if b != "all" else "all"] +
                      [row.get(k, 0) for k in keys])
        elif typ == "ladder":
            arm = spec["arm"]
            bs = spec.get("budgets", [1, 2, 3, 4, 5])
            lad = M.ladder(recs[arm], tasks, bs)
            T = Table(name, ["budget b", "score"])
            for k in bs:
                T.add([k, _f(lad[k])])
        elif typ == "cost":
            roles = ["backbone", "resident", "diagnosis", "retrieval"]
            T = Table(name, ["arm", "All", "$/task"] + [f"$ {r}" for r in roles]
                      + ["median wall s"])
            for d, r in recs.items():
                s = M.summary(r, tasks)
                c = None if getattr(by_disp[d], "prefix", False) else M.cost(r, tasks)
                complete_cost = c is not None and c["usd_per_task"] is not None
                T.add([d, _f(s["delivery"]),
                       _f(c["usd_per_task"], 2) if complete_cost else ""] +
                      ([_f(c["per_role"].get(x, 0.0), 2) for x in roles]
                       if complete_cost else [""] * len(roles)) +
                      [_f(c["wall_median_s"], 0) if complete_cost else ""])
        else:
            raise SystemExit(f"unknown table type {typ!r}")
        written.append(T.write(odir))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("--out", default="results")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(argv)
    for p in build(a.experiment, a.out, a.boot):
        print(p)


if __name__ == "__main__":
    main()
