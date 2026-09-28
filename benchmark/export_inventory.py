"""Export the frozen 522-task inventory promised in the paper appendix.

    python benchmark/export_inventory.py [--bench benchmark]

Reads the fixed split and the index built from frozen task directories.
The CSV contains no source files, native tests, or grading node IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

COLUMNS = (
    "task", "repository_url", "commit", "package", "src_prefix",
    "test_path", "n_modules", "floor", "ceiling", "span", "band",
    "domain", "track", "stars",
)
PAPER_BANDS = (117, 111, 116, 96, 82)


def export(bench: str, out: str) -> int:
    split = os.path.join(bench, "splits", "lego_repo_522.txt")
    index = os.path.join(bench, "tasks.jsonl")
    names = [line.strip() for line in open(split) if line.strip()]
    if len(names) != 522 or len(set(names)) != 522:
        raise SystemExit(f"{split}: expected 522 unique task IDs")
    by_name = {row["name"]: row for row in
               (json.loads(line) for line in open(index) if line.strip())}
    missing = sorted(set(names) - by_name.keys())
    if missing:
        raise SystemExit(
            f"{index}: {len(missing)} paper tasks lack frozen keys: {missing[:5]}")
    rows = []
    for name in names:
        r = by_name[name]
        rows.append({
            "task": name,
            "repository_url": r["clone"],
            "commit": r["commit"],
            "package": r["package"],
            "src_prefix": r["src_prefix"],
            "test_path": r["tests"],
            "n_modules": r["n_modules"],
            "floor": r["floor"],
            "ceiling": r["ceiling"],
            "span": r["ceiling"] - r["floor"],
            "band": r["band"],
            "domain": r["domain"],
            "track": r["track"],
            "stars": r["stars"],
        })
    bands = tuple(sum(r["band"] == b for r in rows) for b in range(1, 6))
    if bands != PAPER_BANDS:
        raise SystemExit(f"paper split bands changed: {bands}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[inventory] distinct repository URLs: "
          f"{len({r['repository_url'] for r in rows})}; "
          f"above 300 modules: {sum(r['n_modules'] > 300 for r in rows)}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="benchmark")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.bench, "lego_repo_inventory.csv")
    print(f"[inventory] {export(args.bench, out)} rows -> {out}")


if __name__ == "__main__":
    main()
