"""Launch external-agent and transfer jobs declared by experiment.yaml.

    python -m lego.special experiments/rq1_external_agents/experiment.yaml --list
    python -m lego.special experiments/rq6_transfer/experiment.yaml --validate
    python -m lego.special EXP --job 0

The ordinary LEGO arms in those manifests are launched with lego.launch.
This module launches their external rows using the same split and shard count.
"""

from __future__ import annotations

import argparse
import os
import sys

from lego import launch


def jobs(exp: dict) -> list[dict]:
    special = exp.get("special") or {}
    shards = int(exp.get("shards", 1))
    kind = special.get("kind")
    if kind == "agents":
        cases = [{"kind": kind, "system": system, "model": special["model"]}
                 for system in special["systems"]]
    elif kind == "transfer":
        cases = [{"kind": kind, "benchmark": bench, "config": config,
                  "model": special["model"]}
                 for bench in special["benchmarks"]
                 for config in special["configs"]]
    else:
        raise SystemExit(f"experiment {exp['name']} has no supported special jobs")
    return [{**case, "shard": shard, "nshards": shards}
            for shard in range(shards) for case in cases]


def validate(exp: dict) -> None:
    special = exp["special"]
    if special["kind"] == "agents":
        from baselines.agents import system_spec
        for system in special["systems"]:
            spec = system_spec(system)
            text = repr(spec)
            if "<" in text and ">" in text:
                raise SystemExit(
                    f"{system} has unresolved release or command fields in "
                    "baselines/agents.yaml; configure the pinned installation first")
    else:
        from external.common import load_benchmark, load_tasks, is_placeholder
        for name in special["benchmarks"]:
            spec = load_benchmark(name)
            release = spec.get("release_path")
            if not release or is_placeholder(release) or not os.path.isdir(release):
                raise SystemExit(
                    f"{name}: release_path is missing in external/benchmarks.yaml")
            tasks = load_tasks(spec)
            if not tasks:
                raise SystemExit(f"{name}: no tasks loaded from {release}")
            if any(not t.grader_cmd for t in tasks):
                raise SystemExit(f"{name}: official grader command is missing")
            if "lego" in special["configs"]:
                lib = spec.get("library")
                if not lib or is_placeholder(lib) or not os.path.isfile(
                        os.path.join(lib, "filter_report.json")):
                    raise SystemExit(
                        f"{name}: filtered CodeFace is missing; run "
                        "python -m external.filter_library first")


def run(job: dict, exp: dict) -> int:
    if job["kind"] == "agents":
        from baselines.run import main
        argv = ["--system", job["system"], "--model", job["model"],
                "--split", exp["split"], "--shard", str(job["shard"]),
                "--nshards", str(job["nshards"])]
    else:
        from external.run import main
        argv = ["--benchmark", job["benchmark"], "--config", job["config"],
                "--model", job["model"], "--shard", str(job["shard"]),
                "--nshards", str(job["nshards"])]
    return main(argv) or 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--count", action="store_true")
    modes.add_argument("--list", action="store_true")
    modes.add_argument("--job", type=int)
    modes.add_argument("--local", action="store_true")
    modes.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)
    exp = launch.load(args.experiment)
    all_jobs = jobs(exp)
    if args.count:
        print(len(all_jobs))
        return 0
    if args.list:
        for i, j in enumerate(all_jobs):
            fields = {k: v for k, v in j.items() if k not in ("shard", "nshards")}
            print(f"{i:4d} {fields} shard {j['shard']}/{j['nshards']}")
        return 0
    validate(exp)
    if args.validate:
        print(f"{exp['name']}: {len(all_jobs)} external jobs ready")
        return 0
    if args.job is not None:
        if not 0 <= args.job < len(all_jobs):
            raise SystemExit(f"job index {args.job} outside 0..{len(all_jobs)-1}")
        return run(all_jobs[args.job], exp)
    for job in all_jobs:
        rc = run(job, exp)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
