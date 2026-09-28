"""Grade the tree an external agent leaves behind, through the LEGO-REPO path.

    python -m baselines.grade --workspace DIR --system openhands --model ALIAS \
        [--shard K] [--index PATH] [--no-tree]

Only the target package's ``.py`` modules are collected from ``DIR/repo``, by
the rule ``prepare`` uses to enumerate a package's modules; tests, config and
every file outside the package are ignored. The collected tree then goes
through exactly what LEGO's own arms go through: ``Workspace.execute`` (native
suite from the pinned clone, provenance plugin) and ``grading.make_score_fn``
(frozen ceiling/floor key). The record has the format of ``lego.run`` and is
appended to ``runs/arms/ext_<system>-<model>/records.s<K>.jsonl`` with
``config = ext_<system>`` and ``stages = ext``, so ``lego.analysis`` and
``grading.admissible`` read external arms unchanged; the delivered tree is kept
as ``trees/<task>.json.gz``.

Diagnostics that never change the score or admissibility:

  delivered        collected modules, target modules missing, stubs left
                   unchanged (instruct mode), modules the agent added
  env_changed      the grading environment's site-packages differ from when
                   the workspace was built
  copy_similarity  largest MinHash similarity of a delivered module to its
                   original (modules over 200 characters)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import socket
import sys
import time

from lego.harness import core, grading
from lego.harness.interface import stub
from lego.harness.prepare import Workspace
from lego.harness.task import load_index
from lego.library import minhash

from baselines.workspace import STUB_HEADER, env_fingerprint

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def arm_id(system: str, model: str) -> str:
    return f"ext_{system}-{_SAFE.sub('_', model)}"


def runs_root(runs: str | None = None) -> str:
    return runs or os.environ.get("LEGO_RUNS", "runs")


def arm_dir(system: str, model: str, runs: str | None = None) -> str:
    return os.path.join(runs_root(runs), "arms", arm_id(system, model))


def collect(meta: dict) -> dict[str, str]:
    """The target package's ``.py`` modules as the agent left them."""
    pkg_dir = os.path.join(meta["tree"], meta["pkg_rel"])
    if meta.get("single_module"):
        p = os.path.join(pkg_dir, meta["pkg"] + ".py")
        return ({meta["pkg"] + ".py": open(p, errors="ignore").read()}
                if os.path.isfile(p) else {})
    if not os.path.isdir(pkg_dir):
        return {}
    files = {}
    for m in core.package_modules(pkg_dir):
        if core.is_test_file(os.path.basename(m)) or \
                os.path.basename(m) == "conftest.py":
            continue
        try:
            files[m] = open(os.path.join(pkg_dir, m), errors="ignore").read()
        except OSError:
            continue
    return files


def load_workspace(meta: dict, index: str | None = None) -> Workspace:
    """Rebuild the grading-side Workspace from workspace.json (no originals)."""
    g = meta["grading"]
    task = load_index(index or meta.get("index"))[meta["task"]]
    rec = dict(meta.get("record") or {})
    bounds = os.path.join(g["tdir"], "bounds.json")
    if os.path.exists(bounds):
        rec.update(json.load(open(bounds)))
    return Workspace(task, g["tdir"], g["repo_dir"], g["proj_dir"],
                     meta["src_prefix"], meta["pkg"], meta["test_path"],
                     g["pkg_dir"], list(meta["modules"]), {}, rec)


def _originals(ws: Workspace) -> dict[str, str]:
    if ws.orig_src:
        return ws.orig_src
    root = os.path.join(ws.task.dir, "original")
    out = {}
    for m in ws.modules:
        p = os.path.join(root, m)
        if os.path.isfile(p):
            out[m] = open(p, errors="ignore").read()
    return out


def copy_similarity(files: dict, orig: dict) -> dict:
    sims = {}
    for m, code in files.items():
        o = orig.get(m) or ""
        if len(code) > 200 and len(o) > 200:
            sims[m] = round(minhash.similarity(minhash.signature(code),
                                               minhash.signature(o)), 3)
    return {"max": max(sims.values(), default=0.0),
            "n_ge_0.6": sum(1 for v in sims.values() if v >= 0.6),
            "compared": len(sims)}


def _unchanged_stubs(files: dict, orig: dict) -> int:
    n = 0
    for m, code in files.items():
        if m in orig and code.strip() == (STUB_HEADER + stub(orig[m])).strip():
            n += 1
    return n


def _usage(agent: dict, model: str) -> dict:
    u = (agent or {}).get("usage") or {}
    if "tokens_in" not in u and "tokens_out" not in u:
        return {}
    return {f"agent|{model}": {"calls": int(u.get("calls") or 0),
                               "in": int(u.get("tokens_in") or 0),
                               "out": int(u.get("tokens_out") or 0),
                               "failed": 0}}


def grade(meta: dict, system: str, model: str, *, ws: Workspace | None = None,
          agent: dict | None = None, shard: int = 0, runs: str | None = None,
          keep_tree: bool = True, write: bool = True,
          index: str | None = None) -> dict:
    g = meta["grading"]
    core.VENV, core.CLONE_BASE = g["venv"], g["clone_base"]
    core.EXTRA_PYTHONPATH[:] = []
    ws = ws or load_workspace(meta, index)
    task = ws.task
    adir = arm_dir(system, model, runs)
    os.makedirs(adir, exist_ok=True)
    rec = {"name": task.name, "arm": arm_id(system, model),
           "config": f"ext_{system}", "stages": "ext",
           "models": {"backbone": model}, "band": task.band,
           "domain": task.domain, "track": task.track,
           "harness_rev": core.HARNESS_REV, "host": socket.gethostname(),
           "ts": time.time(), "mode": meta["mode"]}
    rec.update({k: ws.record.get(k) for k in
                ("commit", "python", "python_why", "env_setup", "baseline",
                 "ceiling_run", "floor_run", "ceiling_drift", "n_modules",
                 "single_module")})
    score_fn = grading.make_score_fn(task, ws)
    rec.update(score_fn.bounds)
    rec["library_view_size"] = 0
    t0 = time.time()
    env_now = env_fingerprint(core.VENV)
    files = collect(meta)
    res = ws.execute(files)
    s = score_fn(res)
    orig = _originals(ws)
    targets = set(meta["modules"])
    rec["trajectory"] = [{"round": 1, "import_ok": res.import_ok,
                          "passed": s["passed"], "failed": res.failed,
                          "verdict": res.verdict, "score": s["score"],
                          "missing_exports": 0, "t": round(time.time() - t0)}]
    rec.update({"best_round": 1, "passed": s["passed"], "score": s["score"],
                "verdict": res.verdict, "exercised": res.exercised,
                "funnel": {}, "activated": {}, "activation_status": {}})
    rec["status"] = ("done" if res.verdict == "ok"
                     else core._status_for(res.verdict))
    rec["delivered"] = {"files": len(files),
                        "missing": sorted(targets - set(files)),
                        "unchanged_stubs": _unchanged_stubs(files, orig)
                        if meta["mode"] == "instruct" else 0,
                        "added": sorted(set(files) - targets)[:50]}
    rec["env_changed"] = env_now != (meta.get("env") or {}).get("fingerprint")
    rec["copy_similarity"] = copy_similarity(files, orig)
    if agent is not None:
        rec["agent"] = {k: v for k, v in agent.items() if k != "logs"}
        if (agent.get("usage") or {}).get("cost_usd") is not None:
            rec["agent_cost_usd"] = agent["usage"]["cost_usd"]
    rec["usage"] = _usage(agent, model)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["wall_seconds"] = round((agent or {}).get("wall_seconds", 0)
                                + rec["seconds"], 1)
    if keep_tree and files:
        os.makedirs(os.path.join(adir, "trees"), exist_ok=True)
        with gzip.open(os.path.join(adir, "trees", task.name + ".json.gz"),
                       "wt") as fh:
            json.dump(files, fh)
    if write:
        append(rec, adir, shard)
    return rec


def append(rec: dict, adir: str, shard: int = 0) -> str:
    os.makedirs(adir, exist_ok=True)
    path = os.path.join(adir, f"records.s{shard}.jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return path


def write_arm_config(adir: str, system: str, model: str, spec: dict) -> None:
    os.makedirs(adir, exist_ok=True)
    cfg = {"name": f"ext_{system}", "stages": "ext", "system": system,
           "models": {"backbone": model},
           "version_pin": spec.get("version_pin"), "mode": spec.get("mode"),
           "timeout": spec.get("timeout"), "runs_on": spec.get("runs_on"),
           "delivery": spec.get("delivery"), "cmd": spec.get("cmd")}
    with open(os.path.join(adir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True, help="DIR of baselines.workspace")
    ap.add_argument("--system", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--index")
    ap.add_argument("--no-tree", action="store_true")
    a = ap.parse_args(argv)
    meta = json.load(open(os.path.join(a.workspace, "workspace.json")))
    ag = os.path.join(a.workspace, "agent", "agent.json")
    agent = json.load(open(ag)) if os.path.exists(ag) else None
    rec = grade(meta, a.system, a.model, agent=agent, shard=a.shard,
                keep_tree=not a.no_tree, index=a.index)
    print(f"[grade] {rec['name']} {rec['arm']}: status={rec['status']} "
          f"score={rec['score']:.3f} admissible={grading.admissible(rec)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
