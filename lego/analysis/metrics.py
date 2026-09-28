"""Benchmark metrics over task records.

All metrics are computed over the full task list of a split: a task without a
record, or with an inadmissible record, contributes a score of 0 (Eq. 8), so an
arm cannot improve its mean by failing to run hard tasks.

  delivery   S_delivery = mean_T s~_T
  dead0      Dead@0  = fraction of tasks with s~_T = 0
  ceil1      Ceil@1  = fraction of tasks with s~_T = 1
  paired     mean(s~_A - s~_B) with a paired bootstrap CI and a sign test
  ladder     score at attempt budget k, read from the per-round trajectory
  cost       manuscript accounting: construction and diagnosis calls,
             successful adaptations, and fixed retrieval charge
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from lego.harness.grading import admissible, delivery_score
from lego.llm.client import price

RETRIEVAL_USD_PER_TASK = 0.011


def scores(records: dict, names: list[str]) -> list[float]:
    return [delivery_score(records.get(n)) for n in names]


def summary(records: dict, tasks: list) -> dict:
    names = [t.name for t in tasks]
    s = scores(records, names)
    n = len(s) or 1
    out = {"n": len(s), "records": sum(1 for x in names if x in records),
           "admissible": sum(1 for x in names if admissible(records.get(x) or {})),
           "delivery": sum(s) / n,
           "dead0": sum(1 for x in s if x <= 0.0) / n,
           "ceil1": sum(1 for x in s if x >= 1.0) / n}
    for key, attr in (("band", "band"), ("domain", "domain"), ("track", "track")):
        grp = defaultdict(list)
        for t, x in zip(tasks, s):
            grp[getattr(t, attr)].append(x)
        out[key] = {k: (sum(v) / len(v), len(v)) for k, v in sorted(grp.items())}
    return out


def paired(rec_a: dict, rec_b: dict, tasks: list, n_boot: int = 10000,
           seed: int = 0, subset=None) -> dict:
    ts = [t for t in tasks if subset is None or subset(t)]
    a = scores(rec_a, [t.name for t in ts])
    b = scores(rec_b, [t.name for t in ts])
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if not n:
        return {"n": 0}
    mean = sum(d) / n
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    wins = sum(1 for x in d if x > 0)
    losses = sum(1 for x in d if x < 0)
    mb = sum(b) / n
    return {"n": n, "a": sum(a) / n, "b": mb, "delta": mean,
            "rel": (mean / mb) if mb else float("nan"), "ci95": (lo, hi),
            "wins": wins, "losses": losses, "ties": n - wins - losses,
            "sign_p": sign_test(wins, losses)}


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def ladder(records: dict, tasks: list, budgets=(1, 2, 3, 4, 5)) -> dict:
    """Score at attempt budget k: best round <= k of each trajectory (the b=k
    run is a prefix of the b=max run; see the configuration table)."""
    out = {}
    for k in budgets:
        tot = 0.0
        for t in tasks:
            r = records.get(t.name)
            if not r or not admissible(r):
                continue
            traj = [x for x in r.get("trajectory") or [] if x["round"] <= k]
            ok = [x for x in traj if x.get("verdict") == "ok"]
            tot += max((x["score"] for x in ok), default=0.0)
        out[k] = tot / (len(tasks) or 1)
    return out


def _record_role_costs(rec: dict) -> dict[str, float] | None:
    """Return paper-billable role costs, or None when attribution is absent."""
    usage = rec.get("usage") or {}
    if ("successful_adaptation_usage" not in rec
            and any(k.startswith("resident|") for k in usage)):
        # Older records combine assessment, successful and failed adaptation
        # calls. Their paper cost cannot be recovered from that aggregate.
        return None
    billed = {key: u for key, u in usage.items()
              if key.partition("|")[0] in ("backbone", "diagnosis")}
    billed.update(rec.get("successful_adaptation_usage") or {})
    out = defaultdict(float)
    for key, u in billed.items():
        role, _, alias = key.partition("|")
        pin, pout = price(alias)
        out[role] += u.get("in", 0) / 1e6 * pin + u.get("out", 0) / 1e6 * pout
    if str(rec.get("stages") or "").startswith(("R", "F")):
        out["retrieval"] += RETRIEVAL_USD_PER_TASK
    return dict(out)


def record_cost(rec: dict) -> float | None:
    roles = _record_role_costs(rec)
    return sum(roles.values()) if roles is not None else None


def cost(records: dict, tasks: list) -> dict:
    per_role = defaultdict(float)
    n = 0
    for t in tasks:
        r = records.get(t.name)
        if not r:
            continue
        role_costs = _record_role_costs(r)
        if role_costs is None:
            continue
        n += 1
        for role, usd in role_costs.items():
            per_role[role] += usd
    complete = bool(tasks) and n == len(tasks)
    walls = sorted(r.get("wall_seconds", 0)
                   for t in tasks if (r := records.get(t.name)))
    return {"usd_per_task": sum(per_role.values()) / n if complete else None,
            "tasks_costed": n,
            "per_role": ({k: v / n for k, v in per_role.items()}
                         if complete else {}),
            "wall_median_s": walls[len(walls) // 2] if walls else 0}


def funnel(records: dict, tasks: list) -> dict:
    by = defaultdict(lambda: defaultdict(int))
    for t in tasks:
        r = records.get(t.name)
        if not r:
            continue
        for k, v in (r.get("funnel") or {}).items():
            by[t.band][k] += v
            by["all"][k] += v
        by[t.band]["modules"] += t.n_modules
        by["all"]["modules"] += t.n_modules
    return {k: dict(v) for k, v in by.items()}
