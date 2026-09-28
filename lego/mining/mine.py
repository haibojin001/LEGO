"""Mine Code Primitives (kind = mined) from a corpus of repositories.

    python -m lego.mining.mine --corpus corpus.txt --out codeface \\
        [--workers N] [--describe --model ALIAS] [--synth-tests] \\
        [--max-files 6] [--max-lines 1500] [--refine 3] [--keep-excluded]

``corpus.txt`` lists one repository per line as ``clone_url [commit]`` (``#``
starts a comment). A given commit is pinned through ``core.PINS`` and cloned by
``core.clone_repo``; without one the default branch is cloned and the resolved
sha recorded. Repositories in ``lego.benchmark_meta.HELD_REPOS`` are skipped
without being cloned.

Per repository (one worker process each): parse the package into a dependency
graph, segment it into candidates, and synthesize + validate every candidate in
a fresh venv shared by that repository's candidates (the source package itself
is never installed). Admitted primitives are written to ``<out>/<pid>/`` with
provenance ``{repo, org, url, commit, paths, license}``; with
``--keep-excluded`` excluded ones are written too, as ``validated: false``.

Every candidate and every repository gets one line in
``<out>/_mining_log.jsonl``. Runs are resumable: a repository with a
repository-level line is skipped (``--retry-failed`` retries clone/parse
failures).

``--describe`` has a model (role ``mining``) write name/summary/capabilities;
``--synth-tests`` has it write tests for candidates that carried none (marked
``tests_synthesized``). Both use ``--model``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback

from lego.benchmark_meta import HELD_REPOS
from lego.harness import core
from lego.mining import collect, segment, synthesize
from lego.mining.graph import build_graph

LOG = "_mining_log.jsonl"
MAX_CANDIDATES = 200
_RETRYABLE = {"clone-failed", "error", "no-package"}


def parse_corpus(path: str) -> list[tuple[str, str | None]]:
    out = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        out.append((parts[0], parts[1] if len(parts) > 1 else None))
    return out


def repo_id(url: str) -> tuple[str, str]:
    """(org, name) for https/ssh GitHub-style URLs, file:// URLs and paths."""
    s = url.strip().rstrip("/")
    s = re.sub(r"\.git$", "", s)
    s = re.sub(r"^[a-z+]+://", "", s)
    s = re.sub(r"^[^@/]+@([^:/]+):", r"\1/", s)       # git@host:org/name
    parts = [p for p in s.split("/") if p]
    name = parts[-1] if parts else s
    org = parts[-2] if len(parts) > 1 else ""
    return org.lower(), name.lower()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def make_pid(name: str, dotted: str, key: str) -> str:
    """Stable, short, import-safe id: <repo>__<module>_<hash>."""
    h = hashlib.sha1(key.encode()).hexdigest()[:6]
    base = f"{_slug(name)[:22]}__{_slug(dotted)[:24]}"
    return f"{base}_{h}"


def is_held(url: str) -> bool:
    _org, name = repo_id(url)
    return name in HELD_REPOS or name in core.HOLD_REPOS


def load_log(out: str) -> dict:
    """{url: last repository-level row}."""
    done = {}
    p = os.path.join(out, LOG)
    if not os.path.exists(p):
        return done
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "repo":
            done[r.get("url")] = r
    return done


def _scratch() -> str:
    base = os.environ.get("LEGO_SCRATCH") or tempfile.gettempdir()
    d = os.path.join(base, f"lego_mine_{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    return d


def _llm(alias):
    if not alias:
        return None
    from lego.llm.client import LLM
    return LLM(alias, role="mining")


def mine_repo(url: str, commit: str | None, out: str, opts: dict) -> list[dict]:
    """Mine one repository. Runs in a worker process; returns log rows."""
    t0 = time.time()
    org, name = repo_id(url)
    base = {"url": url, "repo": f"{org}/{name}" if org else name}
    rows = []
    repo_row = dict(base, event="repo", status="done", n_candidates=0,
                    n_admitted=0)
    if is_held(url):
        repo_row.update(status="held", reason="held repository", ts=time.time())
        return [repo_row]
    scratch = _scratch()
    core.VENV = os.path.join(scratch, "venv")
    repo_dir = os.path.join(scratch, "clones", _slug(name))
    core.rm_rf(repo_dir)
    os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
    env = None
    try:
        if commit:
            core.PINS[url] = {"sha": commit}
        sha = core.clone_repo(url, repo_dir)
        if not sha:
            repo_row.update(status="clone-failed", commit=commit)
            return [repo_row]
        base["commit"] = repo_row["commit"] = sha
        try:
            g = build_graph(repo_dir)
        except ValueError as e:
            repo_row.update(status="no-package", reason=str(e))
            return [repo_row]
        spdx, text = collect.license_info(repo_dir)
        repo_row.update(package=g.pkg, n_modules=len(g.modules),
                        n_test_files=len(g.tests), license=spdx)
        comps, rejected = segment.segment(g, opts["max_files"], opts["max_lines"])
        for r in rejected:
            rows.append(dict(base, event="candidate", seed=r.seed,
                             status="excluded", stage="segment", reason=r.reason))
        comps = comps[:opts.get("max_candidates") or MAX_CANDIDATES]
        repo_row["n_candidates"] = len(comps)
        if not comps:
            repo_row["reason"] = "no candidates"
            return rows + [repo_row]
        proj = core.project_dir(repo_dir, g.src_prefix)
        env = synthesize.Env(proj)
        repo_row["python"] = env.version
        if not env.has_pytest:
            repo_row.update(status="error", reason="pytest unavailable in venv")
            return rows + [repo_row]
        sopts = synthesize.Options(
            kind="mined", refine=opts["refine"], max_files=opts["max_files"],
            max_lines=opts["max_lines"],
            describe_llm=_llm(opts.get("model")) if opts.get("describe") else None,
            tests_llm=_llm(opts.get("model")) if opts.get("synth_tests") else None,
            keep_excluded=opts.get("keep_excluded", False))
        source = {"repo": base["repo"], "org": org, "url": url, "commit": sha,
                  "license": spdx, "forks": []}
        context = {"LICENSE": text} if text else {}
        for c in comps:
            pid = make_pid(name, g.dotted(c.seed), f"{url}|{c.seed}")
            try:
                row = synthesize.synthesize(g, c, out, pid, source, env, sopts,
                                            context)
            except Exception as e:  # noqa: BLE001
                row = {"pid": pid, "seed": c.seed, "status": "excluded",
                       "reason": f"synthesis error: {type(e).__name__}: {e}"[:300]}
            rows.append(dict(base, event="candidate", **row))
            repo_row["n_admitted"] += row["status"] == "admitted"
            print(f"  [{row['status']}] {name}:{c.seed} -> {pid}: "
                  f"{row['reason']}", flush=True)
        return rows + [repo_row]
    except Exception as e:  # noqa: BLE001
        repo_row.update(status="error", reason=f"{type(e).__name__}: {e}"[:300],
                        traceback=traceback.format_exc()[-2000:])
        return rows + [repo_row]
    finally:
        if env is not None and not opts.get("keep"):
            env.close()
        if not opts.get("keep"):
            core.rm_rf(repo_dir)
        repo_row["seconds"] = round(time.time() - t0, 1)
        repo_row["ts"] = time.time()


def _append(out: str, rows: list):
    with open(os.path.join(out, LOG), "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="codeface")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--synth-tests", action="store_true")
    ap.add_argument("--model", default=None, help="model alias for --describe "
                    "and --synth-tests")
    ap.add_argument("--max-files", type=int, default=segment.MAX_FILES)
    ap.add_argument("--max-lines", type=int, default=segment.MAX_LINES)
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    ap.add_argument("--refine", type=int, default=synthesize.REFINE_ROUNDS)
    ap.add_argument("--keep-excluded", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--keep", action="store_true",
                    help="keep clones and venvs (debugging)")
    a = ap.parse_args(argv)
    if (a.describe or a.synth_tests) and not a.model:
        ap.error("--describe/--synth-tests need --model")
    os.makedirs(a.out, exist_ok=True)
    done = load_log(a.out)
    todo = []
    for url, commit in parse_corpus(a.corpus):
        prev = done.get(url)
        if prev and not (a.retry_failed and prev.get("status") in _RETRYABLE):
            continue
        todo.append((url, commit))
    print(f"[mine] {len(todo)} repositories to mine ({len(done)} in log) -> "
          f"{a.out}", flush=True)
    opts = {"max_files": a.max_files, "max_lines": a.max_lines,
            "max_candidates": a.max_candidates, "refine": a.refine,
            "describe": a.describe, "synth_tests": a.synth_tests,
            "model": a.model, "keep_excluded": a.keep_excluded, "keep": a.keep}
    n_adm = 0
    if a.workers <= 1:
        for url, commit in todo:
            print(f"[repo] {url}", flush=True)
            rows = mine_repo(url, commit, a.out, opts)
            _append(a.out, rows)
            n_adm += sum(r.get("status") == "admitted" for r in rows)
    else:
        with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
            futs = {pool.submit(mine_repo, u, c, a.out, opts): u for u, c in todo}
            for f in cf.as_completed(futs):
                try:
                    rows = f.result()
                except Exception as e:  # noqa: BLE001
                    rows = [{"event": "repo", "url": futs[f], "status": "error",
                             "reason": f"worker: {type(e).__name__}: {e}"[:300]}]
                _append(a.out, rows)
                n_adm += sum(r.get("status") == "admitted" for r in rows)
                print(f"[repo] {futs[f]}: {rows[-1].get('status')} "
                      f"{rows[-1].get('n_admitted', 0)}/"
                      f"{rows[-1].get('n_candidates', 0)} admitted", flush=True)
    print(f"[mine] done: {n_adm} primitives admitted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
