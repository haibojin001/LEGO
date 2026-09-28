"""Resolve donor repositories from the legacy manifest for a fresh mining pass.

The flat manifest supplies source *hints*, not validated primitives. This
command turns resolvable hints into a pinned repository corpus. Mining, test
collection/synthesis, and isolation validation happen after this step.

    python -m lego.mining.source_plan \
      --manifest /path/to/primitives_library/manifest.json \
      --metadata benchmark/metadata.jsonl --pins benchmark/pins.json \
      --out corpora/codeface_sources.txt
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import re
import subprocess
from pathlib import Path

from lego.benchmark_meta import HELD_REPOS
from lego.harness.core import clean_env

_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_VERIFIED = re.compile(r"^\(verified:(.+)\)$")


def _metadata(path: str) -> dict[str, str]:
    rows = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row.get("name") and row.get("clone"):
                    rows[row["name"]] = row["clone"]
    return rows


def _pins(path: str) -> dict[str, str]:
    with open(path) as fh:
        data = json.load(fh).get("pins") or {}
    return {url: info["sha"] for url, info in data.items()
            if isinstance(info, dict) and info.get("sha")}


def source_urls(hint: str, metadata: dict[str, str]) -> tuple[list[str], str]:
    """Return public clone URLs and a reason if the hint is unresolved."""
    m = _VERIFIED.fullmatch(hint.strip())
    if m:
        task = m.group(1).removeprefix("repo_")
        clone = metadata.get(task)
        return ([clone], "") if clone else ([], "unknown_verified_task")
    urls = []
    for part in hint.split("+"):
        part = part.strip()
        if part.endswith("more"):
            continue
        if "__" in part and "/" not in part:
            part = part.replace("__", "/", 1)
        if not _REPO.fullmatch(part):
            return [], "invalid_repository_hint"
        urls.append(f"https://github.com/{part}.git")
    return (urls, "") if urls else ([], "missing_repository_hint")


def _head(url: str) -> str | None:
    try:
        run = subprocess.run(["git", "ls-remote", url, "HEAD"],
                             capture_output=True, text=True, timeout=30,
                             env=clean_env(GIT_TERMINAL_PROMPT="0"))
    except (OSError, subprocess.TimeoutExpired):
        return None
    if run.returncode != 0:
        return None
    sha = run.stdout.split(maxsplit=1)[0] if run.stdout.strip() else ""
    return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else None


def plan(manifest: str, metadata_path: str, pins_path: str,
         out: str, pin_heads: bool = False, workers: int = 8) -> dict:
    rows = json.loads(Path(manifest).read_text())
    metadata = _metadata(metadata_path)
    pins = _pins(pins_path)
    corpus = {}
    unresolved = []
    kinds = collections.Counter()
    for row in rows:
        kind = row.get("kind") or "unspecified"
        kinds[kind] += 1
        hint = row.get("from_repo") or ""
        urls, issue = source_urls(hint, metadata)
        if issue:
            unresolved.append({"file": row.get("file"), "hint": hint,
                               "reason": issue})
            continue
        for url in urls:
            name = url.removesuffix(".git").rstrip("/").split("/")[-1].lower()
            if name in HELD_REPOS:
                unresolved.append({"file": row.get("file"), "hint": hint,
                                   "reason": "held_repository"})
                continue
            corpus.setdefault(url, pins.get(url))
    unavailable = []
    if pin_heads:
        missing = [url for url, sha in corpus.items() if not sha]
        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for url, sha in zip(missing, pool.map(_head, missing)):
                if sha:
                    corpus[url] = sha
                else:
                    unavailable.append(url)
                    del corpus[url]
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(
        f"{url}{' ' + sha if sha else ''}\n"
        for url, sha in sorted(corpus.items())))
    with open(target.with_suffix(".unresolved.jsonl"), "w") as fh:
        for issue in unresolved:
            fh.write(json.dumps(issue) + "\n")
    target.with_suffix(".unavailable.jsonl").write_text(
        "".join(json.dumps({"url": url, "reason": "head_unavailable"}) + "\n"
                for url in unavailable))
    report = {"manifest_entries": len(rows), "manifest_kinds": dict(kinds),
              "source_repositories": len(corpus),
              "pinned_repositories": sum(bool(sha) for sha in corpus.values()),
              "unavailable_repositories": len(unavailable),
              "unresolved_entries": len(unresolved),
              "unresolved_reasons": dict(collections.Counter(
                  r["reason"] for r in unresolved)),
              "corpus": str(target)}
    target.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--metadata", default="benchmark/metadata.jsonl")
    ap.add_argument("--pins", default="benchmark/pins.json")
    ap.add_argument("--out", default="corpora/codeface_sources.txt")
    ap.add_argument("--pin-heads", action="store_true",
                    help="resolve missing commit pins from public repository HEADs")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    print(json.dumps(plan(args.manifest, args.metadata, args.pins, args.out,
                          args.pin_heads, args.workers),
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
