"""Run one external repository agent over a LEGO-REPO split.

    python -m baselines.run --system openhands --model ALIAS \
        --split lego_repo_522 --shard 0 --nshards 16 [--timeout S] \
        [--only t1 t2] [--index PATH] [--agents baselines/agents.yaml] \
        [--env copy|shared] [--workdir DIR] [--keep-workspace] [--no-trees] \
        [--force] [--max N]

Per task: build the workspace (``baselines.workspace``, in the input format the
system expects), run the agent in it (``baselines.agents``), grade the tree it
leaves behind (``baselines.grade``), clean up. Task order, the hold list and
sharding are those of ``lego.run``; records go to
``runs/arms/ext_<system>-<model>/records.s<shard>.jsonl`` and tasks that already
have a record are skipped unless ``--force``. A task that cannot be prepared
gets the same status ``lego.run`` would give it.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import traceback

from lego.harness import core
from lego.harness.prepare import NotGradeable, cleanup, task_dir
from lego.harness.task import band_interleave, load_index, select
from lego.run import configure_harness, load_records

from baselines import agents as A
from baselines import grade as G
from baselines import workspace as W


def _base_record(system, model, task, mode):
    return {"name": task.name, "arm": G.arm_id(system, model),
            "config": f"ext_{system}", "stages": "ext",
            "models": {"backbone": model}, "band": task.band,
            "domain": task.domain, "track": task.track,
            "harness_rev": core.HARNESS_REV, "host": socket.gethostname(),
            "ts": time.time(), "mode": mode, "score": 0.0}


def run_task(system, model, task, spec, args, adir, shard, wroot) -> dict:
    tdir = task_dir(core.CLONE_BASE, task.name)
    out = os.path.join(wroot, os.path.basename(tdir))
    mode = spec.get("mode", "instruct")
    t0 = time.time()
    rec = _base_record(system, model, task, mode)
    try:
        ws, meta = W.build(task, out, mode=mode, env=args.env, tdir=tdir,
                           index=args.index)
        print(f"  workspace {meta['tree']} ({len(meta['modules'])} modules)",
              flush=True)
        info = A.run(system, meta, model, timeout=args.timeout,
                     config=args.agents)
        print(f"  agent exit={info.get('exit_code')} timed_out="
              f"{info.get('timed_out')} {info.get('wall_seconds')}s", flush=True)
        rec = G.grade(meta, system, model, ws=ws, agent=info, shard=shard,
                      keep_tree=not args.no_trees, write=False)
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
        if not args.keep_workspace:
            core.rm_rf(out)
    rec["wall_seconds"] = round(time.time() - t0, 1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=A.SYSTEMS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default=os.environ.get("LEGO_SPLIT",
                                                      "lego_repo_522"))
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--index")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--timeout", type=int, help="wall-clock seconds per task "
                    "(default: agents.yaml)")
    ap.add_argument("--agents", default=A.DEFAULT_CONFIG)
    ap.add_argument("--env", choices=W.ENV_MODES, default="copy")
    ap.add_argument("--workdir", help="where agent workspaces are created "
                    "(default: under LEGO_SCRATCH)")
    ap.add_argument("--keep-workspace", action="store_true")
    ap.add_argument("--no-trees", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max", type=int, default=10 ** 6)
    args = ap.parse_args(argv)

    spec = A.system_spec(args.system, args.agents)
    adir = G.arm_dir(args.system, args.model)
    G.write_arm_config(adir, args.system, args.model, spec)
    tasks = band_interleave(select(load_index(args.index), args.split,
                                   args.only))
    tasks = [t for t in tasks if t.name not in core.HOLD_REPOS]
    tasks = tasks[args.shard::args.nshards]
    done = {} if args.force else load_records(adir)
    todo = [t for t in tasks if t.name not in done][:args.max]
    arm = G.arm_id(args.system, args.model)
    print(f"[ext] {arm} version={A.version(spec)} (pin {spec.get('version_pin')})"
          f" mode={spec.get('mode')} | shard {args.shard}/{args.nshards}: "
          f"{len(tasks)} tasks, {len(todo)} to run", flush=True)
    if not todo:
        return 0
    tag = f"{arm}_s{args.shard}"
    configure_harness(tag)
    wroot = args.workdir or os.path.join(
        os.environ.get("LEGO_SCRATCH") or os.path.dirname(core.CLONE_BASE),
        f"lego_ext_ws_{tag}")
    os.makedirs(wroot, exist_ok=True)
    for i, t in enumerate(todo, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(todo)}] {t.name} (D{t.band}, "
              f"{t.n_modules} modules)\n{'=' * 70}", flush=True)
        rec = run_task(args.system, args.model, t, spec, args, adir,
                       args.shard, wroot)
        G.append(rec, adir, args.shard)
        print(f"  => {rec.get('status')} score={rec.get('score', 0):.3f} "
              f"({rec.get('wall_seconds', 0):.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
