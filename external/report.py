"""Report each RQ6 benchmark in its own metric, with run coverage."""

from __future__ import annotations

import argparse
import csv
import json
import os

from external.common import load_benchmark, load_tasks
from lego import launch
from lego import run


def build(experiment: str, out: str) -> str:
    exp = launch.load(experiment)
    special = exp.get("special") or {}
    root = os.environ.get("LEGO_RUNS", "runs")
    rows = []
    for bench in special.get("benchmarks", []):
        tasks = {t.name for t in load_tasks(load_benchmark(bench))}
        bdir = os.path.join(root, "external", bench)
        if not os.path.isdir(bdir):
            continue
        for arm in sorted(os.listdir(bdir)):
            adir = os.path.join(bdir, arm)
            cp = os.path.join(adir, "config.json")
            if not os.path.isfile(cp):
                continue
            config = json.load(open(cp))
            if config.get("name") not in special["configs"]:
                continue
            models = config.get("models") or {}
            if models.get("backbone") != special["model"]:
                continue
            recs = run.load_records(adir)
            values = [float(r["metric"]["value"]) for name, r in recs.items()
                      if name in tasks and r.get("status") == "graded"
                      and isinstance((r.get("metric") or {}).get("value"),
                                     (int, float))]
            metric = config.get("metric") or ""
            rows.append([bench, arm, config["name"], metric,
                         sum(values) / len(values) if values else "",
                         len(values), len(recs), len(tasks)])
    header = ["benchmark", "arm", "config", "metric", "mean_observed",
              "graded_numeric", "records", "tasks"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
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
        "experiments/rq6_transfer/experiment.yaml"))
    ap.add_argument("--out", default="results/rq6_transfer/transfer.csv")
    args = ap.parse_args(argv)
    print(build(args.experiment, args.out))


if __name__ == "__main__":
    main()
