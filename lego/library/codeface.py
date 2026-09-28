"""CodeFace: the primitive library F = {P_i}, with per-task views.

``CodeFace(root)`` loads validated primitives with carried tests. An explicit
``allow_unvalidated`` option enables exploratory use of flat proxies. A run
never queries
the whole library directly: it asks for ``lib.view(task, filters)``, which applies
the configuration's source restrictions (Sec. 5, App. "Configuration
definitions") for that target:

  kinds            keep only these provenance kinds (mined/harvested/web/target)
  exclude          any of: same_repo, same_org, near_dup
  exclude_repos    explicit repo names or URLs (external-benchmark filtering)
  subset_frac      keep a seeded random fraction of the library
  near_dup_thresh  MinHash threshold against the target's original sources

Retrieval (``View.search``) embeds a capability requirement and returns the top-m
candidates. Relevance assessment (Eq. 1) is done afterwards by each candidate's
resident model, not here.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re

from lego.benchmark_meta import HELD_REPOS
from lego.library import minhash
from lego.library.embed import make_embedder
from lego.library.primitive import Primitive, load


def canonical_repo(url_or_name: str) -> str:
    """'https://github.com/Org/Repo.git' -> 'org/repo'; 'repo' -> 'repo'."""
    s = (url_or_name or "").strip().lower()
    s = re.sub(r"^https?://(www\.)?github\.com/", "", s)
    s = re.sub(r"\.git$", "", s).strip("/")
    return s


def repo_name(url_or_name: str) -> str:
    return canonical_repo(url_or_name).split("/")[-1]


def repo_org(url_or_name: str) -> str:
    c = canonical_repo(url_or_name)
    return c.split("/")[0] if "/" in c else ""


def primitive_repos(p: Primitive) -> set[str]:
    """All recorded donor repositories, including multi-repo legacy entries."""
    src = p.source
    values = [src.get("repo"), src.get("url"), *(src.get("repos") or [])]
    return {canonical_repo(v) for v in values if v}


class CodeFace:
    def __init__(self, root: str, embedder: str = "tfidf",
                 allow_unvalidated: bool = False):
        self.root = root
        self.embedder_spec = embedder
        self.prims: list[Primitive] = []
        for pj in sorted(glob.glob(os.path.join(root, "*", "primitive.json"))):
            p = load(os.path.dirname(pj))
            if any(repo_name(src) in HELD_REPOS for src in primitive_repos(p)):
                continue
            if allow_unvalidated:
                if (p.meta.get("validated") is False
                        and p.meta.get("validation_level")
                        not in ("syntax_only", "raw_unparsed")):
                    continue
            else:
                validation = p.meta.get("validation") or {}
                if (p.meta.get("validated") is not True or not p.tests
                        or validation.get("mode") != "tests"
                        or not isinstance(validation.get("passed"), int)
                        or validation["passed"] < 1):
                    continue
            self.prims.append(p)
        self._sigs = None

    def __len__(self):
        return len(self.prims)

    def counts(self) -> dict:
        c = {}
        for p in self.prims:
            c[p.kind] = c.get(p.kind, 0) + 1
        return c

    def signatures(self) -> list[list[int]]:
        if self._sigs is None:
            cache = os.path.join(self.root, "_minhash.json")
            if os.path.exists(cache):
                data = json.load(open(cache))
            else:
                data = {}
            changed = False
            for p in self.prims:
                if p.pid not in data:
                    data[p.pid] = minhash.signature(p.impl_text(limit=10 ** 7))
                    changed = True
            if changed:
                try:
                    with open(cache, "w") as fh:
                        json.dump(data, fh)
                except OSError:
                    pass
            self._sigs = [data[p.pid] for p in self.prims]
        return self._sigs

    def view(self, task=None, filters: dict | None = None,
             target_sources: dict | None = None) -> "View":
        f = filters or {}
        keep = list(range(len(self.prims)))
        kinds = f.get("kinds")
        if kinds:
            keep = [i for i in keep if self.prims[i].kind in set(kinds)]
        if f.get("subset_frac") is not None and f["subset_frac"] < 1.0:
            rng = random.Random(f"subset|{f.get('subset_seed', 0)}")
            keep = [i for i in keep if rng.random() < f["subset_frac"]]
        excl = set(f.get("exclude") or [])
        drop_repos = {canonical_repo(r) for r in f.get("exclude_repos") or []}
        drop_names = {repo_name(r) for r in drop_repos}
        if task is not None:
            t_repo = canonical_repo(task.clone)
            t_name, t_org = repo_name(task.clone), repo_org(task.clone)
            forks = {canonical_repo(x) for x in
                     (getattr(task, "extra", {}) or {}).get("forks", [])}
            if excl & {"same_repo", "same_org", "near_dup"}:
                drop_repos.add(t_repo)
                drop_names.add(t_name)
                drop_names.add(task.name)
            if excl & {"same_org", "near_dup"} and t_org:
                keep = [i for i in keep
                        if self.prims[i].source_org != t_org and not any(
                            repo_org(src) == t_org
                            for src in primitive_repos(self.prims[i]))]
            if "near_dup" in excl:
                drop_repos |= forks
        if drop_repos or drop_names:
            def dropped(p):
                sources = primitive_repos(p)
                pf = {canonical_repo(x) for x in p.source.get("forks", [])}
                return (bool(sources & drop_repos)
                        or any(repo_name(src) in drop_names for src in sources)
                        or bool(pf & drop_repos))
            keep = [i for i in keep if not dropped(self.prims[i])]
        if "near_dup" in excl and target_sources:
            thr = float(f.get("near_dup_thresh", 0.6))
            sigs = self.signatures()
            tsig = [minhash.signature(src) for src in target_sources.values()
                    if len(src) > 200]
            keep = [i for i in keep
                    if not any(minhash.similarity(sigs[i], t) >= thr
                               for t in tsig)]
        return View(self, keep)


class View:
    """The library as one run on one task is allowed to see it."""

    _index_cache: dict = {}

    def __init__(self, lib: CodeFace, keep: list[int]):
        self.lib = lib
        self.keep = keep
        key = (id(lib), lib.embedder_spec)
        if key not in View._index_cache:
            emb = make_embedder(lib.embedder_spec)
            emb.fit([p.search_text() for p in lib.prims])
            View._index_cache[key] = emb
        self.emb = View._index_cache[key]
        self._keep_set = set(keep)

    def __len__(self):
        return len(self.keep)

    def search(self, query: str, top_m: int = 2) -> list[Primitive]:
        if not self.keep:
            return []
        hits = self.emb.query(query, k=max(top_m * 20, 50))
        out = [self.lib.prims[i] for i, _s in hits if i in self._keep_set]
        return out[:top_m]

    def pids(self) -> list[str]:
        return [self.lib.prims[i].pid for i in self.keep]
