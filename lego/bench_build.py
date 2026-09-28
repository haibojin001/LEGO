"""Rebuild LEGO-REPO task directories from repository cards (no model calls).

    python -m lego.bench_build --cards benchmark/metadata.jsonl \
        --pins benchmark/pins.json --out benchmark/tasks [--shard i --nshards n]

For each card: clone the pinned commit, build the environment, run the native
suite installed (baseline), remove the installed target, then measure the
identity ceiling (original sources through the reconstruction path) and the
empty-package floor. A task is written only if the identity run's provenance is
``ok``, the ceiling is positive and c - f >= MIN_SPAN. Afterwards run
``benchmark/build_index.py`` and ``tools/audit_tasks_security.py``.

The frozen task directories used in the paper were produced by this procedure;
rebuilding against live PyPI can shift a few tests (dependency drift), which is
why grading always uses the frozen ``expected/*.txt`` of the released tasks.
"""

from __future__ import annotations

import argparse
import json
import os

from lego.harness import core
from lego.harness.prepare import NotGradeable, cleanup, prepare, task_dir
from lego.harness.task import Task


def build_one(card: dict, out: str) -> dict:
    t = Task(name=card["name"], clone=card["clone"],
             commit=card.get("commit") or core.PINS.get(card["clone"], {}).get("sha", ""),
             package=card.get("package", ""), src_prefix=card.get("src_prefix", "."),
             tests=card.get("tests", ""), domain=card.get("domain", ""))
    tdir = task_dir(core.CLONE_BASE, t.name)
    rec = {"name": t.name, "clone": t.clone}
    try:
        ws = prepare(t, tdir, measure_bounds=False)
        ident = ws.execute(ws.orig_src)
        empty = ws.execute({f: "" for f in ws.orig_src})
        C = sorted(ident.passed_ids())
        F = sorted(empty.passed_ids())
        rec.update(ws.record)
        rec.update(ceiling=len(C), floor=len(F), span=len(C) - len(F),
                   ceiling_prov=ident.verdict, floor_prov=empty.verdict,
                   ceiling_tests=C, floor_tests=F, commit=ws.record.get("commit")
                   or t.commit, pinned=True)
        if ident.verdict != "ok" or not C:
            rec["status"] = "invalid-ceiling"
        elif len(C) - len(F) < core.MIN_SPAN:
            rec["status"] = "invalid-no-headroom"
        else:
            core.BENCH_OUT = out
            core.dump_task(rec, ws.repo_dir, ws.proj_dir, ws.src_prefix, ws.pkg,
                           ws.test_path, ws.modules, ws.orig_src)
            rec["status"] = "gate-ok"
    except NotGradeable as e:
        rec["status"] = e.args[0]
    finally:
        cleanup(tdir)
    for k in ("ceiling_tests", "floor_tests"):
        rec.pop(k, None)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards", required=True, help="jsonl: name, clone, "
                    "[commit, package, src_prefix, tests, domain]")
    ap.add_argument("--pins", default="benchmark/pins.json")
    ap.add_argument("--out", default="benchmark/tasks")
    ap.add_argument("--log", default="runs/bench_build.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(argv)
    from lego.benchmark_meta import HELD_REPOS
    from lego.run import configure_harness
    core.PINS.update(core._load_pins(a.pins))
    cards = [json.loads(l) for l in open(a.cards) if l.strip()]
    cards = [c for c in cards if c["name"] not in HELD_REPOS][a.shard::a.nshards]
    configure_harness(f"build_s{a.shard}")
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    for c in cards:
        if os.path.exists(os.path.join(a.out, c["name"], "task.json")):
            continue
        rec = build_one(c, a.out)
        print(f"[build] {c['name']}: {rec['status']} c={rec.get('ceiling')} "
              f"f={rec.get('floor')}", flush=True)
        with open(a.log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
