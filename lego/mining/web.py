"""Targeted web sourcing (kind = web): capability requests -> GitHub -> primitives.

    python -m lego.mining.web --requests caps.txt --out codeface \\
        [--repos 5] [--files 30] [--per-request 1] \\
        [--describe --model ALIAS] [--synth-tests --model ALIAS] [--python PY]

``caps.txt`` holds one request per line, ``search words [| hint, hint]``; hints
are substrings of the function/class names wanted (default: the request's
longer words). ``#`` starts a comment.

Per request: search repositories (GitHub REST search API, Python, by stars;
``GITHUB_TOKEN`` is used when set), pin the default branch's head commit, list
its tree, and fetch up to ``--files`` source files from
raw.githubusercontent.com, paths ranked by overlap with the request. From each
file ``ast`` extracts the top-level functions/classes whose names match a hint,
with the module-level definitions and imports they need. An extraction whose
closure imports the repository's own modules is skipped: web sourcing stays
single-file. Test files of the repository that mention the source module are
fetched, and their test functions that use only extracted names are kept.

Each extraction is materialized as a one-module package plus tests and goes
through the mining pipeline (graph, segment, synthesize) under the same
admission rule: at least one carried (or, with ``--synth-tests``, synthesized)
test on the extracted code passes in isolation. Provenance: ``{repo, org, url,
commit, paths, license}`` (license = the API's SPDX id), plus ``symbols`` and the
``request``. Log: ``<out>/_web_log.jsonl`` (resumable per request). Nothing
beyond the GitHub API and raw file fetches is contacted.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from lego.benchmark_meta import HELD_REPOS
from lego.harness import core
from lego.mining import segment, synthesize
from lego.mining.graph import STDLIB, Refs, bound_names, build_graph
from lego.mining.mine import _llm, make_pid

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"
LOG = "_web_log.jsonl"
PKG = "websrc"
MAX_FILE_BYTES = 400_000
_BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
_SKIP_PATH = re.compile(r"(^|/)(tests?|testing|docs?|examples?|benchmarks?|"
                        r"scripts|setup\.py|conftest\.py)(/|$)|(^|/)test_[^/]*$|"
                        r"_test\.py$", re.I)


class RateLimited(Exception):
    pass


# ------------------------------------------------------------------ http
def _get(url: str, token: str | None = None, accept: str =
         "application/vnd.github+json", timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept,
                                               "User-Agent": "lego-mining"})
    if token and url.startswith(API):
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(MAX_FILE_BYTES + 1)
    except urllib.error.HTTPError as e:
        if e.code in (403, 429) and url.startswith(API):
            raise RateLimited(f"{e.code} from {url.split('?')[0]}") from e
        raise


def _json(url: str, token: str | None):
    return json.loads(_get(url, token).decode("utf-8", "replace"))


def search_repos(query: str, n: int, token: str | None) -> list[dict]:
    q = urllib.parse.quote(f"{query} language:python")
    data = _json(f"{API}/search/repositories?q={q}&sort=stars&order=desc"
                 f"&per_page={max(1, min(n, 30))}", token)
    return list(data.get("items") or [])[:n]


def head_commit(full: str, branch: str, token: str | None) -> str | None:
    try:
        return _json(f"{API}/repos/{full}/commits/{urllib.parse.quote(branch)}",
                     token).get("sha")
    except (urllib.error.URLError, ValueError, OSError):
        return None


def list_tree(full: str, sha: str, token: str | None) -> list[str]:
    try:
        tree = _json(f"{API}/repos/{full}/git/trees/{sha}?recursive=1", token)
    except (urllib.error.URLError, ValueError, OSError):
        return []
    return [t["path"] for t in tree.get("tree") or []
            if t.get("type") == "blob" and t["path"].endswith(".py")]


def fetch_raw(full: str, sha: str, path: str) -> str | None:
    try:
        data = _get(f"{RAW}/{full}/{sha}/{urllib.parse.quote(path)}",
                    accept="text/plain")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if len(data) > MAX_FILE_BYTES:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# ------------------------------------------------------------ extraction
def parse_request(line: str) -> tuple[str, list[str]]:
    query, _, hints = line.partition("|")
    query = query.strip()
    hs = [h.strip().lower() for h in hints.split(",") if h.strip()]
    if not hs:
        hs = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", query)]
    return query, hs


def repo_tops(paths: list[str]) -> set:
    """Top-level importable names a repository defines (its own modules),
    at the root or under a ``src``/``lib``/``python`` prefix."""
    tops = set()
    for p in paths:
        parts = p.split("/")
        head = parts[1] if parts[0] in ("src", "lib", "python") and \
            len(parts) > 1 else parts[0]
        tops.add(head[:-3] if head.endswith(".py") else head)
    return {t for t in tops if t.isidentifier()}


def _span(node) -> tuple[int, int]:
    deco = [d.lineno for d in getattr(node, "decorator_list", [])]
    return min([node.lineno] + deco), node.end_lineno


def _import_entries(tree):
    """{local name: (statement text, lineno, top module, relative)}."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Import):
            for a in n.names:
                local = a.asname or a.name.split(".")[0]
                text = f"import {a.name}" + (f" as {a.asname}" if a.asname else "")
                out[local] = (text, n.lineno, a.name.split(".")[0], False)
        elif isinstance(n, ast.ImportFrom) and n.module != "__future__":
            for a in n.names:
                if a.name == "*":
                    continue
                local = a.asname or a.name
                mod = "." * n.level + (n.module or "")
                text = f"from {mod} import {a.name}" + (
                    f" as {a.asname}" if a.asname else "")
                out[local] = (text, n.lineno, (n.module or "").split(".")[0],
                              bool(n.level))
    return out


def _top_defs(tree) -> dict:
    defs = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.setdefault(n.name, n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in tg:
                if isinstance(t, ast.Name):
                    defs.setdefault(t.id, n)
    return defs


def _refs(node) -> set:
    """Free names a definition uses."""
    r = Refs()
    r.visit(node)
    return (r.names | {root for root, _c in r.chains}) - bound_names(node)


def extract(source: str, hints: list[str], internal: set,
            max_symbols: int = 3) -> list[dict]:
    """Self-contained extractions from one file: [{symbols, text}]."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines(keepends=True)
    defs, imps = _top_defs(tree), _import_entries(tree)
    future = [f"from __future__ import {', '.join(a.name for a in n.names)}"
              for n in tree.body if isinstance(n, ast.ImportFrom)
              and n.module == "__future__"]
    wanted = [n for n, node in defs.items()
              if not n.startswith("_") and not isinstance(node, (ast.Assign,
                                                               ast.AnnAssign))
              and any(h in n.lower() for h in hints)]
    out = []
    for name in wanted[:max_symbols * 3]:
        need_defs, need_imps, todo, ok = set(), set(), [name], True
        while todo and ok:
            cur = todo.pop()
            if cur in need_defs:
                continue
            need_defs.add(cur)
            for ref in _refs(defs[cur]):
                if ref in defs and ref not in need_defs:
                    todo.append(ref)
                elif ref in imps:
                    _text, _ln, top, rel = imps[ref]
                    if rel or top in internal:
                        ok = False
                        break
                    need_imps.add(ref)
        if not ok:
            continue
        nodes = sorted({id(defs[d]): defs[d] for d in need_defs}.values(),
                       key=lambda n: n.lineno)
        body = []
        for n in nodes:
            a, b = _span(n)
            body.append("".join(lines[a - 1:b]).rstrip() + "\n")
        head = future + [imps[i][0] for i in sorted(need_imps,
                                                    key=lambda i: imps[i][1])]
        text = "\n".join(head) + ("\n\n\n" if head else "") + "\n\n".join(body)
        try:
            compile(text, "<extracted>", "exec")
        except SyntaxError:
            continue
        out.append({"symbols": sorted(need_defs), "primary": name, "text": text})
        if len(out) >= max_symbols:
            break
    return out


def extract_tests(source: str, names: set, stem: str, internal: set) -> str | None:
    """Test functions/classes of a fetched test file that use only the
    extracted ``names`` (imported from ``websrc.<stem>``), stdlib and
    third-party modules."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    lines = source.splitlines(keepends=True)
    imps = _import_entries(tree)
    avail = set(_BUILTINS) | set(names)
    keep_imps, from_src = [], set()
    for local, (text, _ln, top, rel) in sorted(imps.items(), key=lambda x: x[1][1]):
        if local in names and (rel or top in internal):
            from_src.add(local)
        elif not rel and top not in internal:
            keep_imps.append(text)
            avail.add(local)
    defs = _top_defs(tree)
    kept, fixtures = [], {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                "fixture" in ast.unparse(d) for d in n.decorator_list):
            fixtures[n.name] = n
    for n in tree.body:
        is_test = (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name.startswith("test")) or (
            isinstance(n, ast.ClassDef) and n.name.startswith("Test"))
        if not is_test:
            continue
        refs = _refs(n)
        helpers = {r for r in refs if r in defs and r not in avail}
        if not (refs & names):
            continue
        if refs - avail - helpers - {"self", "cls"}:
            continue
        if any(_refs(defs[h]) - avail - set(defs) for h in helpers):
            continue
        kept.append(n)
        for h in helpers:
            if defs[h] not in kept:
                kept.append(defs[h])
        for a in getattr(getattr(n, "args", None), "args", []):
            f = fixtures.get(a.arg)
            if f is not None and f not in kept and not (_refs(f) - avail - set(defs)):
                kept.append(f)
    if not any(isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               and k.name.lower().startswith("test") for k in kept):
        return None
    body = []
    for n in sorted(kept, key=lambda n: n.lineno):
        a, b = _span(n)
        body.append("".join(lines[a - 1:b]).rstrip() + "\n")
    head = keep_imps + ([f"from {PKG}.{stem} import {', '.join(sorted(from_src or names))}"])
    return "\n".join(head) + "\n\n\n" + "\n\n".join(body)


def materialize(tmp: str, stem: str, code: str, tests: dict) -> str:
    repo = os.path.join(tmp, "repo")
    core.rm_rf(repo)
    os.makedirs(os.path.join(repo, PKG))
    open(os.path.join(repo, PKG, "__init__.py"), "w").close()
    with open(os.path.join(repo, PKG, stem + ".py"), "w") as fh:
        fh.write(code)
    os.makedirs(os.path.join(repo, "tests"))
    for name, text in tests.items():
        with open(os.path.join(repo, "tests", name), "w") as fh:
            fh.write(text)
    return repo


def _stem(path: str, repo: str) -> str:
    s = os.path.basename(path)[:-3]
    if s == "__init__":
        s = os.path.basename(os.path.dirname(path)) or repo
    s = re.sub(r"\W", "_", s)
    return ("m_" + s) if (not s or s[0].isdigit() or s in STDLIB or s == PKG) else s


# ------------------------------------------------------------------ run
def source_request(query: str, hints: list, out: str, env, opts, token,
                   n_repos: int, n_files: int, per_request: int,
                   deadline: float) -> list[dict]:
    t0, rows, admitted = time.time(), [], 0
    for item in search_repos(query, n_repos, token):
        full = item.get("full_name") or ""
        org, name = (full.split("/") + [""])[:2]
        if not full or name.lower() in HELD_REPOS:
            continue
        if time.time() - t0 > deadline or admitted >= per_request:
            break
        sha = head_commit(full, item.get("default_branch") or "HEAD", token)
        if not sha:
            continue
        paths = list_tree(full, sha, token)
        internal = repo_tops(paths)
        words = set(hints) | {w.lower() for w in re.findall(r"[A-Za-z]{3,}", query)}
        cands = sorted((p for p in paths if not _SKIP_PATH.search(p)),
                       key=lambda p: (-sum(w in p.lower() for w in words), len(p), p))
        test_paths = [p for p in paths if re.search(r"(^|/)test_[^/]*\.py$|_test\.py$", p)]
        lic = (item.get("license") or {}).get("spdx_id") or ""
        for path in cands[:n_files]:
            if time.time() - t0 > deadline or admitted >= per_request:
                break
            src = fetch_raw(full, sha, path)
            if not src:
                continue
            for ex in extract(src, hints, internal):
                stem = _stem(path, name)
                tests = {}
                base = os.path.basename(path)[:-3]
                for tp in [t for t in test_paths if base in os.path.basename(t)][:2]:
                    tsrc = fetch_raw(full, sha, tp)
                    tt = extract_tests(tsrc or "", set(ex["symbols"]), stem, internal)
                    if tt:
                        tests[f"test_{stem}_{len(tests)}.py"] = tt
                tmp = tempfile.mkdtemp(prefix="lego_web_")
                try:
                    repo = materialize(tmp, stem, ex["text"], tests)
                    g = build_graph(repo, ".", PKG)
                    comps, _rej = segment.segment(g, opts.max_files, opts.max_lines)
                    comp = next((c for c in comps if c.seed == stem + ".py"), None)
                    if comp is None:
                        continue
                    pid = make_pid(f"web_{name}", f"{stem}.{ex['primary']}",
                                   f"{full}|{path}|{ex['primary']}")
                    source = {"repo": full.lower(), "org": org.lower(),
                              "url": f"https://github.com/{full}", "commit": sha,
                              "paths": [path], "license": lic, "forks": [],
                              "symbols": ex["symbols"], "request": query}
                    try:
                        row = synthesize.synthesize(g, comp, out, pid, source,
                                                    env, opts)
                    except Exception as e:  # noqa: BLE001
                        row = {"pid": pid, "status": "excluded", "reason":
                               f"synthesis error: {type(e).__name__}: {e}"[:300]}
                finally:
                    core.rm_rf(tmp)
                row = dict(row, event="candidate", request=query, repo=full,
                           path=path, symbol=ex["primary"])
                rows.append(row)
                print(f"  [{row['status']}] {full}:{path}:{ex['primary']} -> "
                      f"{row['reason']}", flush=True)
                if row["status"] == "admitted":
                    admitted += 1
                    break
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", default="codeface")
    ap.add_argument("--repos", type=int, default=5)
    ap.add_argument("--files", type=int, default=30)
    ap.add_argument("--per-request", type=int, default=1)
    ap.add_argument("--deadline", type=float, default=240.0,
                    help="seconds per request")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--synth-tests", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--python", default=None, help="validate with this "
                    "interpreter instead of a fresh venv (no installs)")
    a = ap.parse_args(argv)
    if (a.describe or a.synth_tests) and not a.model:
        ap.error("--describe/--synth-tests need --model")
    token = os.environ.get("GITHUB_TOKEN") or None
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, LOG)
    done = set()
    if os.path.exists(log):
        for line in open(log):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "request":
                done.add(r.get("request"))
    reqs = []
    for line in open(a.requests):
        line = line.split("#", 1)[0].strip()
        if line and parse_request(line)[0] not in done:
            reqs.append(parse_request(line))
    print(f"[web] {len(reqs)} requests (token={'yes' if token else 'no'})",
          flush=True)
    if not reqs:
        return 0
    if not a.python:
        scratch = os.environ.get("LEGO_SCRATCH") or tempfile.gettempdir()
        core.VENV = os.path.join(scratch, f"lego_web_venv_{os.getpid()}")
    env = synthesize.Env(python=a.python)
    opts = synthesize.Options(
        kind="web", describe_llm=_llm(a.model) if a.describe else None,
        tests_llm=_llm(a.model) if a.synth_tests else None)
    limited = False
    try:
        for query, hints in reqs:
            print(f"[request] {query} hints={hints}", flush=True)
            try:
                rows = source_request(query, hints, a.out, env, opts, token,
                                      a.repos, a.files, a.per_request, a.deadline)
            except RateLimited as e:
                print(f"[web] rate limited ({e}); stopping, rerun to resume",
                      flush=True)
                limited = True
                break
            except (urllib.error.URLError, OSError, ValueError) as e:
                rows = [{"event": "candidate", "request": query,
                         "status": "excluded", "reason": f"fetch: {e}"[:300]}]
            n = sum(r.get("status") == "admitted" for r in rows)
            with open(log, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r, default=str) + "\n")
                fh.write(json.dumps({"event": "request", "request": query,
                                     "hints": hints, "n_admitted": n,
                                     "ts": time.time()}) + "\n")
    finally:
        env.close()
    return 75 if limited else 0


if __name__ == "__main__":
    sys.exit(main())
