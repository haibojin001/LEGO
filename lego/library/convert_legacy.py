"""Convert the earlier flat primitive manifest into the directory schema.

    python -m lego.library.convert_legacy \
        --src /path/to/primitives_library --out codeface_legacy

These files do not carry the paper's validation tests, dependency closure,
license records, or mined/harvested/web labels. Every safe, present manifest
file is retained for exploratory construction. Entries that fail Python syntax
checks remain raw text with validation_level=raw_unparsed; entries without
public exports retain an empty interface. Neither is marked validated. The
output is not the paper's validated CodeFace snapshot.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import os
import re

from lego.benchmark_meta import HELD_REPOS
from lego.harness.interface import exports, stub
from lego.library.primitive import write


def _repos(value: str) -> list[str]:
    value = value or ""
    if value.startswith("(verified:") and value.endswith(")"):
        value = value[len("(verified:"):-1]
        value = value.removeprefix("repo_")
    return [part.strip() for part in value.split("+")
            if part.strip() and not part.strip().endswith("more")]


def _repo_name(value: str) -> str:
    return value.rstrip("/").split("/")[-1].removeprefix("repo_").lower()


def _external_imports(tree: ast.AST) -> list[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return sorted(out - set(getattr(__import__("sys"), "stdlib_module_names", ())))


def _pid(file: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_.-]", "_", os.path.splitext(file)[0])[:80]
    return stem + "_" + hashlib.sha1(file.encode()).hexdigest()[:8]


def convert(src: str, out: str) -> dict:
    with open(os.path.join(src, "manifest.json")) as fh:
        manifest = json.load(fh)
    os.makedirs(out, exist_ok=True)
    kinds = collections.Counter()
    issues = collections.Counter()
    validation = collections.Counter()
    excluded = collections.Counter()
    seen_files = set()
    for row in manifest:
        file = row.get("file", "")
        if os.path.basename(file) != file or not file.endswith(".py"):
            excluded["unsafe_path"] += 1
            continue
        if file in seen_files:
            excluded["duplicate_file"] += 1
            continue
        seen_files.add(file)
        path = os.path.join(src, file)
        if not os.path.isfile(path):
            excluded["missing_file"] += 1
            continue
        repos = _repos(row.get("from_repo", ""))
        if any(_repo_name(repo) in HELD_REPOS for repo in repos):
            excluded["held_source"] += 1
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        tree = None
        syntax_issue = ""
        try:
            tree = ast.parse(text, filename=file)
            compile(tree, file, "exec")
        except (SyntaxError, ValueError) as e:
            tree = None
            syntax_issue = f"{type(e).__name__}: {e}"[:300]
            issues["syntax_error"] += 1
        names = exports(text) if tree is not None else []
        if not names:
            if tree is not None:
                issues["no_exports"] += 1
        level = "syntax_only" if tree is not None else "raw_unparsed"
        old_kind = row.get("kind") or "unspecified"
        kind = "mined" if old_kind == "mined" else "legacy_" + old_kind
        pid = _pid(file)
        source = {"repo": repos[0] if repos else "",
                  "repos": repos, "paths": [row.get("from_path", "")],
                  "used_in_case": row.get("used_in_case", ""),
                  "license": None, "legacy_kind": old_kind}
        meta = {"pid": pid, "name": row.get("name") or file[:-3],
                "summary": row.get("summary", ""),
                "capabilities": row.get("anchor_symbols") or
                                [row.get("summary", "")[:300]],
                "kind": kind, "source": source,
                "interface": {"exports": names,
                              "signatures": stub(text)[:9000],
                              "contract": row.get("summary", "")[:2000]},
                "dependencies": {"external": _external_imports(tree)
                                 if tree is not None else [],
                                 "closure_verified": False},
                "validated": False, "validation_level": level,
                "legacy_issue": ("syntax_error" if tree is None
                                 else "no_exports" if not names else ""),
                "syntax_issue": syntax_issue,
                "carried_tests_available": False}
        write(os.path.join(out, pid), meta, {file: text},
              context={"legacy_manifest.json": json.dumps(row, indent=2)})
        kinds[kind] += 1
        validation[level] += 1
    report = {"source": os.path.abspath(src), "manifest_entries": len(manifest),
              "converted": sum(kinds.values()), "converter_version": 2,
              "counts": dict(kinds), "validation_levels": dict(validation),
              "issues": dict(issues), "excluded": dict(excluded),
              "paper_equivalent": False}
    with open(os.path.join(out, "_conversion_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="codeface_legacy")
    args = ap.parse_args(argv)
    print(json.dumps(convert(args.src, args.out), indent=2))


if __name__ == "__main__":
    main()
