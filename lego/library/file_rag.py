"""File RAG control: raw donor files instead of Code Primitives.

Indexes the *raw source files* of the same donor repositories CodeFace was built
from, with the same embedder and retrieval depth, and returns whole files: no
recovered boundary, no internalized helpers, no dependency closure, no contract,
no carried tests.

Build once (clones each donor at the commit recorded in CodeFace provenance)::

    python -m lego.library.file_rag build --library codeface --out file_rag

The index is ``<out>/index.jsonl`` with one ``{repo, url, path, text}`` per file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

from lego.library.codeface import CodeFace, canonical_repo, repo_name, repo_org
from lego.library.embed import make_embedder

MAX_FILE_CHARS = 60000


class FileDoc:
    """Duck-types the parts of Primitive that consumption modes read."""

    def __init__(self, row: dict):
        self.row = row
        self.pid = f"file:{row['repo']}:{row['path']}"
        self.kind = "file"
        self.source = {"repo": row["repo"], "url": row.get("url", "")}
        self.name = row["path"]

    def impl_text(self, limit: int = 8000) -> str:
        return f"# --- {self.row['repo']}/{self.row['path']}\n{self.row['text']}"[:limit]

    def tests_text(self, limit: int = 0) -> str:
        return ""

    def card(self) -> str:
        return f"{self.row['repo']}/{self.row['path']}"

    signatures = contract = ""
    exports: list = []
    external_deps: list = []
    tests: dict = {}
    impl: dict = {}


class FileIndex:
    def __init__(self, root: str, embedder: str = "tfidf"):
        self.rows = [json.loads(l) for l in open(os.path.join(root, "index.jsonl"))]
        self.emb = make_embedder(embedder).fit(
            [r["path"] + "\n" + r["text"][:20000] for r in self.rows])

    def view(self, task=None, filters: dict | None = None, **_):
        f = filters or {}
        excl = set(f.get("exclude") or [])
        drop = {canonical_repo(r) for r in f.get("exclude_repos") or []}
        keep = list(range(len(self.rows)))
        if task is not None and excl & {"same_repo", "same_org", "near_dup"}:
            t = canonical_repo(task.clone)
            keep = [i for i in keep
                    if canonical_repo(self.rows[i].get("url") or
                                      self.rows[i]["repo"]) != t
                    and repo_name(self.rows[i]["repo"]) != task.name]
            if excl & {"same_org", "near_dup"}:
                org = repo_org(task.clone)
                keep = [i for i in keep
                        if repo_org(self.rows[i].get("url") or "") != org]
        if drop:
            keep = [i for i in keep
                    if canonical_repo(self.rows[i].get("url") or
                                      self.rows[i]["repo"]) not in drop]
        return _FileView(self, keep)


class _FileView:
    def __init__(self, idx: FileIndex, keep: list[int]):
        self.idx, self.keep, self._ks = idx, keep, set(keep)

    def __len__(self):
        return len(self.keep)

    def search(self, query: str, top_m: int = 2):
        hits = self.idx.emb.query(query, k=max(top_m * 20, 50))
        return [FileDoc(self.idx.rows[i]) for i, _ in hits if i in self._ks][:top_m]


def build(library: str, out: str):
    lib = CodeFace(library)
    donors = {}
    for p in lib.prims:
        url = p.source.get("url")
        if url and p.kind in ("mined", "harvested", "target"):
            donors.setdefault(url, p.source.get("commit"))
    os.makedirs(out, exist_ok=True)
    n = 0
    with open(os.path.join(out, "index.jsonl"), "w") as fh:
        for url, sha in sorted(donors.items()):
            with tempfile.TemporaryDirectory() as d:
                cmd = (["git", "clone", "-q", "--depth=1", url, d] if not sha else
                       None)
                if cmd:
                    subprocess.run(cmd, check=False, timeout=900)
                else:
                    subprocess.run(["git", "init", "-q", d], check=False)
                    subprocess.run(["git", "-C", d, "remote", "add", "origin", url],
                                   check=False)
                    subprocess.run(["git", "-C", d, "fetch", "-q", "--depth=1",
                                    "origin", sha], check=False, timeout=900)
                    subprocess.run(["git", "-C", d, "checkout", "-q", "FETCH_HEAD"],
                                   check=False, timeout=900)
                for r, dirs, fs in os.walk(d):
                    dirs[:] = [x for x in dirs if not x.startswith(".")]
                    for f in fs:
                        if not f.endswith(".py") or f.startswith("test"):
                            continue
                        p = os.path.join(r, f)
                        rel = os.path.relpath(p, d)
                        if "/test" in "/" + rel:
                            continue
                        text = open(p, errors="ignore").read()[:MAX_FILE_CHARS]
                        fh.write(json.dumps({"repo": canonical_repo(url),
                                             "url": url, "path": rel,
                                             "text": text}) + "\n")
                        n += 1
            print(f"[file_rag] {url}: indexed (total {n})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--library", default="codeface")
    b.add_argument("--out", default="file_rag")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.library, a.out)


if __name__ == "__main__":
    main()
