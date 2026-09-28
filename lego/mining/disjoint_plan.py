"""Prepare the corpus-disjoint CodeFace control from a pinned donor corpus.

    python -m lego.mining.disjoint_plan \
        --corpus corpora/codeface_sources.txt \
        --out corpora/disjoint.txt --split lego_repo_522

Exclude every donor from an evaluated target repository, an organization that
owns an evaluated target, or a recorded fork. At construction time the
``lego_remined`` view also excludes source-code near duplicates per target.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from lego.harness.task import load_index, select
from lego.library.codeface import canonical_repo, repo_name, repo_org
from lego.mining.mine import parse_corpus


def plan(corpus: str, targets: list, out: str) -> dict:
    blocked_repos = set()
    blocked_names = set()
    blocked_orgs = set()
    for target in targets:
        repo = canonical_repo(target.clone)
        blocked_repos.add(repo)
        blocked_names.add(repo_name(repo))
        if repo_org(repo):
            blocked_orgs.add(repo_org(repo))
        forks = (getattr(target, "extra", {}) or {}).get("forks") or []
        for fork in forks if isinstance(forks, list) else [forks]:
            blocked_repos.add(canonical_repo(str(fork)))
    kept = []
    excluded = []
    for url, sha in parse_corpus(corpus):
        if not sha or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"donor is not pinned to a commit: {url}")
        repo = canonical_repo(url)
        reason = ("target_or_recorded_fork" if repo in blocked_repos else
                  "target_name" if repo_name(repo) in blocked_names else
                  "target_organization" if repo_org(repo) in blocked_orgs else
                  "")
        if reason:
            excluded.append({"url": url, "reason": reason})
        else:
            kept.append((url, sha))
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(f"{url} {sha}\n" for url, sha in sorted(kept)))
    dest.with_suffix(".excluded.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in excluded))
    report = {"input": len(kept) + len(excluded), "kept": len(kept),
              "excluded": dict(sorted(collections.Counter(
                  row["reason"] for row in excluded).items())),
              "targets": len(targets), "corpus": str(dest)}
    dest.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpora/codeface_sources.txt")
    ap.add_argument("--out", default="corpora/disjoint.txt")
    ap.add_argument("--split", default="lego_repo_522")
    ap.add_argument("--index")
    a = ap.parse_args(argv)
    targets = select(load_index(a.index), a.split)
    print(json.dumps(plan(a.corpus, targets, a.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
