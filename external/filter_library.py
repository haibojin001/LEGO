"""Remove an external benchmark's targets from CodeFace before any task runs.

    python -m external.filter_library --benchmark repozero_py2py \
        --library codeface --out codeface_repozero [--forks FILE] \
        [--thresh 0.6] [--copy] [--exact] [--workers N] [--force] \
        [--benchmarks external/benchmarks.yaml]

Every task of the benchmark is resolved to its upstream repository
(``upstream_repo`` of the task, or the benchmark's ``upstream_repos_file``). A
primitive is removed when its recorded source repository matches any of them

  url        canonical URL equal (scheme, host, ``.git``, trailing slash and
             case stripped; ``org/name`` and full URLs compare equal)
  name       canonical repository name (last path component) equal, or equal
             to a task's declared canonical project name
  fork       the source, or any repository in the primitive's
             ``source.forks``, is an upstream repository or a known fork of
             one (``--forks`` and the benchmark's ``forks_file``: one
             ``repo<TAB>fork`` pair per line, read in both directions)

and, among those that remain, when it is a near duplicate of an external
target's original sources

  near_dup   MinHash similarity >= --thresh (lego.library.minhash, the
             signature CodeFace.view uses) between the primitive's
             implementation and any original source file of the targets
             longer than 200 characters

Candidate pairs for ``near_dup`` come from banding the signatures (42 bands of
3 rows); every candidate is then compared on the full signature, so a pair at
the threshold is missed with probability below 1e-4. ``--exact`` compares all
pairs instead.

Output (``--out``) is a library directory usable as ``view.path``: one symlink
(``--copy``: one copy) per kept primitive, plus

  filter_report.json  parameters, targets and the count removed by each rule
  excluded.jsonl      one {pid, rule, match} per removed primitive
  kept_pids.txt       the kept primitive ids
  view.yaml           {view: {path, exclude_repos}} for an experiment arm; the
                      exclude_repos list re-applies the repository rules at
                      run time (the MinHash rule lives in the directory)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import sys
import time

import yaml

from lego.library import minhash
from lego.library.codeface import CodeFace

from external.common import (is_placeholder, load_benchmark, load_tasks,
                             read_pairs, target_sources)

RULES = ("url", "name", "fork", "near_dup")
BANDS, ROWS = 42, 3
MIN_CHARS = 200


def canon(u: str) -> str:
    """'https://host/Org/Repo.git' | 'git@host:Org/Repo' | 'Org/Repo' -> 'org/repo'."""
    s = (u or "").strip().lower()
    s = re.sub(r"^[a-z][a-z0-9+.-]*://", "", s)
    s = re.sub(r"^[^@/]+@([^:/]+)[:/]", r"\1/", s)
    s = re.sub(r"\.git/?$", "", s).strip("/")
    parts = s.split("/")
    if len(parts) > 2 and "." in parts[0]:          # drop the host
        parts = parts[1:]
    return "/".join(parts[-2:]) if len(parts) >= 2 else s


def name_of(u: str) -> str:
    return canon(u).split("/")[-1]


def target_sets(upstreams, fork_pairs, extra_names=()):
    """(repos, names, related): the upstream repositories, their canonical
    names, and every repository in a known fork relation with one of them."""
    repos = {canon(u) for u in upstreams if u}
    rel = set()
    for a, b in fork_pairs:
        a, b = canon(a), canon(b)
        if a in repos:
            rel.add(b)
        if b in repos:
            rel.add(a)
    names = {name_of(r) for r in repos} | {n.lower() for n in extra_names if n}
    return repos, names, rel - repos


def repo_rule(p, repos: set, names: set, related: set):
    """The first repository rule that removes primitive ``p``, as
    (rule, match), or None."""
    src = p.source or {}
    cands = {canon(x) for x in (src.get("url"), src.get("repo")) if x}
    for c in sorted(cands):
        if c in repos:
            return "url", c
    for c in sorted(cands):
        if name_of(c) in names:
            return "name", name_of(c)
    forks = {canon(x) for x in src.get("forks") or [] if x}
    for c in sorted(cands | forks):
        if c in repos or c in related:
            return "fork", c
    return None


def _bands(sig):
    return [(b, tuple(sig[b * ROWS:(b + 1) * ROWS])) for b in range(BANDS)]


def near_dups(sigs: dict, targets: list, thresh: float = 0.6,
              exact: bool = False) -> dict:
    """{pid: (target label, similarity)} for every primitive signature with
    similarity >= thresh to some target signature. ``targets`` is a list of
    (label, signature)."""
    out = {}
    if not targets:
        return out
    index = {}
    if not exact:
        for j, (_lab, t) in enumerate(targets):
            for key in _bands(t):
                index.setdefault(key, []).append(j)
    for pid, s in sigs.items():
        if exact:
            cand = range(len(targets))
        else:
            cand = {j for key in _bands(s) for j in index.get(key, ())}
        best, lab = 0.0, None
        for j in cand:
            v = minhash.similarity(s, targets[j][1])
            if v > best:
                best, lab = v, targets[j][0]
        if best >= thresh:
            out[pid] = (lab, round(best, 4))
    return out


def target_signatures(tasks, workers: int = 1) -> list:
    items = []
    for t in tasks:
        for rel, text in target_sources(t).items():
            if len(text) > MIN_CHARS:
                items.append((f"{t.name}:{rel}", text))
    if workers > 1 and len(items) > 8:
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            sigs = list(pool.map(minhash.signature, [x[1] for x in items],
                                 chunksize=8))
    else:
        sigs = [minhash.signature(x[1]) for x in items]
    return [(lab, s) for (lab, _t), s in zip(items, sigs)]


def filter_library(lib: CodeFace, tasks, fork_pairs=(), thresh: float = 0.6,
                   exact: bool = False, workers: int = 1):
    """Apply the four rules. Returns (kept prims, excluded rows, report)."""
    ups = sorted({t.upstream_repo for t in tasks if t.upstream_repo})
    pairs = list(fork_pairs)
    for t in tasks:
        pairs += [(t.upstream_repo, f) for f in (t.extra or {}).get("forks", [])]
    repos, names, related = target_sets(
        ups, pairs, [(t.extra or {}).get("canonical_name") for t in tasks])
    excluded, rest = [], []
    for p in lib.prims:
        hit = repo_rule(p, repos, names, related)
        if hit:
            excluded.append({"pid": p.pid, "rule": hit[0], "match": hit[1]})
        else:
            rest.append(p)
    tsigs = target_signatures(tasks, workers)
    allsig = dict(zip((p.pid for p in lib.prims), lib.signatures()))
    dups = near_dups({p.pid: allsig[p.pid] for p in rest}, tsigs, thresh,
                     exact)
    kept = []
    for p in rest:
        if p.pid in dups:
            lab, sim = dups[p.pid]
            excluded.append({"pid": p.pid, "rule": "near_dup", "match": lab,
                             "similarity": sim})
        else:
            kept.append(p)
    counts = {r: sum(1 for x in excluded if x["rule"] == r) for r in RULES}
    report = {"n_library": len(lib.prims), "n_kept": len(kept),
              "removed": counts, "thresh": thresh, "exact": exact,
              "bands": None if exact else [BANDS, ROWS],
              "upstream_repos": sorted(repos),
              "fork_related": sorted(related),
              "canonical_names": sorted(names),
              "tasks": len(tasks),
              "tasks_without_upstream": sorted(t.name for t in tasks
                                               if not t.upstream_repo),
              "target_files": len(tsigs)}
    return kept, excluded, report


def write_view(lib: CodeFace, kept, excluded, report, out: str,
               copy: bool = False, force: bool = False) -> str:
    if os.path.exists(out):
        if not force:
            raise SystemExit(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    os.makedirs(out)
    for p in kept:
        dst = os.path.join(out, os.path.basename(os.path.normpath(p.root)))
        if copy:
            shutil.copytree(p.root, dst)
        else:
            os.symlink(os.path.abspath(p.root), dst)
    sigs = dict(zip((p.pid for p in lib.prims), lib.signatures()))
    with open(os.path.join(out, "_minhash.json"), "w") as fh:
        json.dump({p.pid: sigs[p.pid] for p in kept}, fh)
    with open(os.path.join(out, "filter_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(out, "excluded.jsonl"), "w") as fh:
        for x in excluded:
            fh.write(json.dumps(x) + "\n")
    with open(os.path.join(out, "kept_pids.txt"), "w") as fh:
        fh.write("".join(p.pid + "\n" for p in kept))
    with open(os.path.join(out, "view.yaml"), "w") as fh:
        yaml.safe_dump({"view": {"path": out, "exclude_repos": sorted(
            set(report["upstream_repos"]) | set(report["fork_related"]))}},
            fh, sort_keys=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--library", default="codeface")
    ap.add_argument("--out", required=True)
    ap.add_argument("--forks", help="extra repo<TAB>fork pairs")
    ap.add_argument("--thresh", type=float, default=0.6)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--benchmarks", help="benchmarks.yaml path")
    a = ap.parse_args(argv)
    spec = load_benchmark(a.benchmark, a.benchmarks)
    tasks = load_tasks(spec)
    if not tasks:
        raise SystemExit(f"benchmark {a.benchmark}: the loader found no tasks")
    pairs = read_pairs(a.forks) + read_pairs(spec.get("forks_file"))
    lib = CodeFace(a.library)
    t0 = time.time()
    kept, excluded, report = filter_library(lib, tasks, pairs, a.thresh,
                                            a.exact, a.workers)
    report.update(benchmark=a.benchmark, library=os.path.abspath(a.library),
                  forks_files=[x for x in (a.forks, spec.get("forks_file"))
                               if x and not is_placeholder(x)],
                  seconds=round(time.time() - t0, 1), created=time.time())
    write_view(lib, kept, excluded, report, a.out, a.copy, a.force)
    missing = report["tasks_without_upstream"]
    print(f"[filter] {a.benchmark}: {len(tasks)} tasks, "
          f"{len(report['upstream_repos'])} upstream repos, "
          f"{len(report['fork_related'])} fork-related, "
          f"{report['target_files']} target files")
    for r in RULES:
        print(f"  removed by {r:9s} {report['removed'][r]:6d}")
    print(f"  kept               {len(kept):6d} / {len(lib.prims)}  -> {a.out}")
    if missing:
        print(f"  [warn] {len(missing)} task(s) have no upstream repository: "
              f"{missing[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
