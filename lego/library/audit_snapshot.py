"""Audit whether a CodeFace directory has the paper's required fields.

    python -m lego.library.audit_snapshot --library codeface --paper

This structural check does not execute the carried tests. Runtime validation
must still be performed in the donor environment recorded for each primitive.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from lego.library.primitive import load

PAPER_KINDS = {"mined": 871, "harvested": 343, "web": 210}


def audit(root: str, paper: bool = False) -> dict:
    paths = sorted(Path(root).glob("*/primitive.json"))
    counts = collections.Counter()
    problems = collections.Counter()
    examples = {}

    def issue(code: str, pid: str):
        problems[code] += 1
        examples.setdefault(code, pid)

    for path in paths:
        p = load(str(path.parent))
        counts[p.kind] += 1
        if p.meta.get("validated") is not True:
            issue("unvalidated", p.pid)
        if not p.impl:
            issue("missing_implementation", p.pid)
        else:
            for name, code in p.impl.items():
                if name.endswith(".py"):
                    try:
                        compile(code, name, "exec")
                    except SyntaxError:
                        issue("implementation_syntax", p.pid)
                        break
        if not p.tests:
            issue("missing_carried_tests", p.pid)
        validation = p.meta.get("validation")
        if paper and (not isinstance(validation, dict)
                      or validation.get("mode") != "tests"
                      or not isinstance(validation.get("passed"), int)
                      or validation["passed"] < 1):
            issue("missing_passing_isolation_validation", p.pid)
        iface = p.meta.get("interface")
        if not isinstance(iface, dict) or not isinstance(
                iface.get("exports"), list):
            issue("missing_interface", p.pid)
        deps = p.meta.get("dependencies")
        if not isinstance(deps, dict) or not isinstance(
                deps.get("external"), list):
            issue("missing_dependency_record", p.pid)
        source = p.meta.get("source")
        if not isinstance(source, dict) or not (
                source.get("repo") or source.get("url")):
            issue("missing_provenance", p.pid)

    if not paths:
        issue("empty_library", "")
    if paper:
        for kind, expected in PAPER_KINDS.items():
            if counts[kind] != expected:
                issue(f"paper_kind_{kind}", f"{counts[kind]}/{expected}")
        if len(paths) != sum(PAPER_KINDS.values()):
            issue("paper_total", f"{len(paths)}/{sum(PAPER_KINDS.values())}")
        for kind in counts.keys() - PAPER_KINDS.keys():
            issue("unexpected_kind", kind)
    return {"entries": len(paths), "kinds": dict(sorted(counts.items())),
            "problems": dict(sorted(problems.items())), "examples": examples,
            "structurally_ready": not problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True)
    ap.add_argument("--paper", action="store_true",
                    help="also require the paper's 1424-entry source mix")
    args = ap.parse_args(argv)
    result = audit(args.library, args.paper)
    print(json.dumps(result, indent=2))
    return 0 if result["structurally_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
