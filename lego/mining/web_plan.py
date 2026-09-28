"""Turn uncovered capability requirements into targeted web-sourcing requests.

Use a coverage audit made from interface-level requirements. The companion
manifest records which target modules motivated each request, so a fresh
CodeFace rebuild can be traced without copying native tests or original source.

    python -m lego.analysis.coverage --split lego_repo_522 --config lego \
        --assess --out results/coverage
    python -m lego.mining.web_plan \
        --coverage results/coverage/<arm>_assess.jsonl \
        --out corpora/web_requests.txt
    python -m lego.mining.web --requests corpora/web_requests.txt \
        --out codeface --describe --synth-tests --model gpt-5.6-terra
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def _clean(text: str) -> str:
    """Keep a single GitHub search phrase; '#' and '|' delimit request syntax."""
    return re.sub(r"\s+", " ", text.split("|", 1)[0].split("#", 1)[0]).strip()


def plan(rows: list[dict], limit: int = 0) -> list[dict]:
    """Group uncovered modules by search phrase, ordered by demand."""
    groups: dict[str, dict] = {}
    for row in rows:
        if row.get("covered"):
            continue
        query = _clean(str(row.get("retrieval_request") or
                           row.get("capability") or ""))
        if not query:
            continue
        key = query.casefold()
        g = groups.setdefault(key, {"query": query, "hints": set(),
                                    "modules": []})
        for name in row.get("exports") or []:
            name = str(name).strip()
            if (len(name) >= 4 and re.fullmatch(r"[A-Za-z_]\w*", name)
                    and not name.startswith("_")):
                g["hints"].add(name.lower())
        g["modules"].append({
            "task": str(row.get("task") or ""),
            "module": str(row.get("module") or "")})
    ordered = sorted(groups.values(),
                     key=lambda g: (-len(g["modules"]), g["query"].casefold()))
    if limit > 0:
        ordered = ordered[:limit]
    return [{"query": g["query"], "hints": sorted(g["hints"])[:8],
             "modules": g["modules"]} for g in ordered]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coverage", required=True,
                    help="per-module JSONL from lego.analysis.coverage")
    ap.add_argument("--out", default="corpora/web_requests.txt")
    ap.add_argument("--limit", type=int, default=0,
                    help="take the most requested capabilities (0 = all)")
    a = ap.parse_args(argv)
    with open(a.coverage) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    requests = plan(rows, a.limit)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(
        r["query"] + (" | " + ", ".join(r["hints"]) if r["hints"] else "")
        for r in requests) + ("\n" if requests else ""))
    manifest = out.with_suffix(".manifest.jsonl")
    manifest.write_text("\n".join(json.dumps(r, sort_keys=True)
                                  for r in requests)
                        + ("\n" if requests else ""))
    counts = collections.Counter("uncovered" if not r.get("covered") else "covered"
                                 for r in rows)
    print(json.dumps({"coverage_rows": len(rows), "uncovered": counts["uncovered"],
                      "requests": len(requests), "out": str(out),
                      "manifest": str(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
