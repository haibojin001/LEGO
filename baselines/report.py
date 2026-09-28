"""Summarize RQ1 external-agent records alongside the matched LEGO arms."""

from __future__ import annotations

import argparse
import csv
import os

from baselines.grade import arm_dir as agent_arm_dir
from lego import launch
from lego import run
from lego.analysis.metrics import summary
from lego.harness.task import load_index, select


def build(experiment: str, out: str) -> str:
    exp = launch.load(experiment)
    tasks = select(load_index(exp.get("index")), exp["split"])
    rows = []
    for _name, _overrides, cfg in launch.arms(exp):
        recs = run.load_records(run.arm_dir(cfg))
        stats = summary(recs, tasks)
        rows.append([cfg.display, cfg.arm_id(), stats["delivery"],
                     stats["dead0"], stats["ceil1"],
                     stats["records"], stats["n"]])
    special = exp.get("special") or {}
    for system in special.get("systems", []):
        model = special["model"]
        adir = agent_arm_dir(system, model)
        recs = run.load_records(adir)
        stats = summary(recs, tasks)
        rows.append([system, os.path.basename(adir), stats["delivery"],
                     stats["dead0"], stats["ceil1"],
                     stats["records"], stats["n"]])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    header = ["system", "arm", "delivery", "dead0", "ceil1", "records", "tasks"]
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    md = os.path.splitext(out)[0] + ".md"
    with open(md, "w") as fh:
        fh.write("| " + " | ".join(header) + " |\n")
        fh.write("|" + "---|" * len(header) + "\n")
        for row in rows:
            fh.write("| " + " | ".join(map(str, row)) + " |\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", nargs="?", default=(
        "experiments/rq1_external_agents/experiment.yaml"))
    ap.add_argument("--out", default="results/rq1_external_agents/agents.csv")
    args = ap.parse_args(argv)
    print(build(args.experiment, args.out))


if __name__ == "__main__":
    main()
