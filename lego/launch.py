"""Expand an experiment file into jobs and run them.

An experiment (``experiments/<id>/experiment.yaml``) lists arms and an optional
sweep::

    name: rq4_backbones
    split: lego_repo_522
    shards: 16
    arms:
      - {config: feedback}
      - {config: lego}
    sweep:                     # cartesian product applied to every arm
      models.backbone: [gpt-5.6-terra, claude-sonnet-5]

Each (arm, sweep point) is one RunConfig; each RunConfig is split into
``shards`` jobs. Job indices are arm-major within a shard (all arms' shard 0
first), so a sweep that is cut off early leaves every arm with the same number
of completed shards.

    python -m lego.launch experiments/rq4_backbones/experiment.yaml --list
    python -m lego.launch EXP --job 17          # one job (SLURM array task)
    python -m lego.launch EXP --local           # every job, serially
    python -m lego.launch EXP --status          # records per arm
    python -m lego.launch EXP --count           # number of jobs (for --array)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import yaml

from lego import config as C
from lego import run as R


def load(path: str) -> dict:
    with open(path) as fh:
        exp = yaml.safe_load(fh)
    exp.setdefault("name", os.path.basename(os.path.dirname(os.path.abspath(path))))
    exp.setdefault("split", "lego_repo_522")
    # Keep the paper's 522-task split in the checked-in experiment definitions.
    # A partial local freeze can run via LEGO_SPLIT=lego_repo_available.
    exp["split"] = os.environ.get("LEGO_SPLIT") or exp["split"]
    exp.setdefault("shards", 1)
    return exp


def _set(d: dict, dotted: str, v):
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = v


def arms(exp: dict) -> list[tuple[str, dict, C.RunConfig]]:
    sweep = exp.get("sweep") or {}
    keys = list(sweep)
    points = list(itertools.product(*[sweep[k] for k in keys])) or [()]
    out = []
    for a in exp["arms"]:
        a = dict(a)
        name = a.pop("config")
        display = a.pop("display", None) or name
        for pt in points:
            ov = json.loads(json.dumps(a))
            for k, v in zip(keys, pt):
                _set(ov, k, v)
            cfg = C.get(name, ov)
            cfg.display = display + "".join(
                f" [{_short(v)}]" for k, v in zip(keys, pt))
            cfg.sweep_point = dict(zip(keys, pt))
            cfg.base_display = display
            out.append((name, ov, cfg))
    seen, uniq = set(), []
    for x in out:
        if x[2].arm_id() not in seen:
            seen.add(x[2].arm_id())
            uniq.append(x)
    return uniq


def _short(v) -> str:
    return str(v) if not isinstance(v, (list, dict)) else json.dumps(v)


def jobs(exp: dict) -> list[dict]:
    A = arms(exp)
    n = int(exp.get("shards", 1))
    return [{"arm": A[i][2], "name": A[i][0], "overrides": A[i][1], "shard": s,
             "nshards": n} for s in range(n) for i in range(len(A))]


def _argv(exp, j) -> list[str]:
    ov = dict(j["overrides"])
    argv = ["--config", j["name"], "--split", exp["split"],
            "--shard", str(j["shard"]), "--nshards", str(j["nshards"])]
    sets = []

    def flat(prefix, d):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                flat(key + ".", v)
            else:
                sets.append(f"{key}={json.dumps(v)}")
    flat("", ov)
    if sets:
        argv += ["--set", *sets]
    if exp.get("index"):
        argv += ["--index", exp["index"]]
    return argv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--count", action="store_true")
    g.add_argument("--job", type=int)
    g.add_argument("--local", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--max", type=int, default=None)
    args, rest = ap.parse_known_args(argv)
    exp = load(args.experiment)
    J = jobs(exp)
    if args.count:
        print(len(J))
        return
    if args.list:
        for i, j in enumerate(J):
            print(f"{i:4d}  {j['arm'].arm_id():40s} shard {j['shard']}/{j['nshards']}"
                  f"  models={j['arm'].models.resolved()}")
        return
    if args.status:
        from lego.harness.task import load_split
        n = len(load_split(exp["split"]))
        for name, _ov, cfg in arms(exp):
            recs = R.load_records(R.arm_dir(cfg))
            done = sum(1 for r in recs.values() if r.get("status") == "done")
            print(f"{cfg.arm_id():40s} records {len(recs):4d}/{n}  done {done:4d}")
        return
    extra = (["--max", str(args.max)] if args.max else []) + rest
    if args.job is not None:
        if args.job >= len(J):
            print(f"job {args.job} >= {len(J)} jobs; nothing to do")
            return
        return R.main(_argv(exp, J[args.job]) + extra)
    for j in J:
        R.main(_argv(exp, j) + extra)


if __name__ == "__main__":
    sys.exit(main())
