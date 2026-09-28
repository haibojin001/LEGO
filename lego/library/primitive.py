"""Code Primitive: P_i = (C_i, I_i, D_i, V_i, X_i, M_i).

On-disk layout of one primitive (a directory under the library root)::

    <pid>/
      primitive.json     metadata: interface I_i, dependency closure D_i,
                         provenance and context X_i, test list, stats
      impl/...           C_i  implementation files (package-relative layout)
      tests/...          V_i  carried validation tests
      context/...        X_i  supporting docs / config (optional)

The resident model M_i is not stored: it is the model assigned to the
``resident`` position of a run, instantiated per primitive with the primitive's
own state as context (``lego.library.resident``).

``primitive.json`` fields::

    pid, name, summary, capabilities: [str]
    kind: mined | harvested | web | target
    source: {repo, org, url, commit, paths: [str], license, forks: [str]}
    interface: {exports: [str], signatures: str, contract: str}
    dependencies: {external: [dist], python: str}
    tests: [str]            relative to tests/
    stats: {lines, files, n_tests}
    validated: bool
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class Primitive:
    pid: str
    root: str
    meta: dict = field(default_factory=dict)
    _impl: dict | None = None
    _tests: dict | None = None
    _context: dict | None = None

    # ---- metadata views
    @property
    def kind(self) -> str:
        return self.meta.get("kind", "mined")

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.pid

    @property
    def summary(self) -> str:
        return self.meta.get("summary", "")

    @property
    def capabilities(self) -> list[str]:
        return list(self.meta.get("capabilities") or [])

    @property
    def source(self) -> dict:
        return self.meta.get("source") or {}

    @property
    def source_repo(self) -> str:
        return (self.source.get("repo") or "").lower()

    @property
    def source_org(self) -> str:
        return (self.source.get("org") or "").lower()

    @property
    def exports(self) -> list[str]:
        return list((self.meta.get("interface") or {}).get("exports") or [])

    @property
    def signatures(self) -> str:
        return (self.meta.get("interface") or {}).get("signatures", "")

    @property
    def contract(self) -> str:
        return (self.meta.get("interface") or {}).get("contract", "")

    @property
    def external_deps(self) -> list[str]:
        return list((self.meta.get("dependencies") or {}).get("external") or [])

    # ---- file contents (lazy)
    def _read_tree(self, sub: str) -> dict:
        base = os.path.join(self.root, sub)
        out = {}
        if not os.path.isdir(base):
            return out
        for r, _d, fs in os.walk(base):
            for f in sorted(fs):
                if f.endswith((".pyc",)):
                    continue
                p = os.path.join(r, f)
                try:
                    out[os.path.relpath(p, base)] = open(p, errors="ignore").read()
                except OSError:
                    pass
        return out

    @property
    def impl(self) -> dict:
        if self._impl is None:
            self._impl = self._read_tree("impl")
        return self._impl

    @property
    def tests(self) -> dict:
        if self._tests is None:
            self._tests = self._read_tree("tests")
        return self._tests

    @property
    def context(self) -> dict:
        if self._context is None:
            self._context = self._read_tree("context")
        return self._context

    def impl_text(self, limit: int = 8000) -> str:
        parts = [f"# --- {p}\n{c}" for p, c in sorted(self.impl.items())
                 if p.endswith(".py")]
        return "\n\n".join(parts)[:limit]

    def tests_text(self, limit: int = 4000) -> str:
        return "\n\n".join(f"# --- {p}\n{c}"
                           for p, c in sorted(self.tests.items()))[:limit]

    def card(self) -> str:
        """Short description used for retrieval text and relevance prompts."""
        return (f"{self.name}: {self.summary}\n"
                f"capabilities: {'; '.join(self.capabilities)}\n"
                f"exports: {', '.join(self.exports[:30])}\n"
                f"depends on: {', '.join(self.external_deps) or '(stdlib)'}")

    def search_text(self) -> str:
        return " ".join([self.name, self.summary, " ".join(self.capabilities),
                         " ".join(self.exports), self.signatures[:2000]])


def load(root: str) -> Primitive:
    with open(os.path.join(root, "primitive.json")) as fh:
        meta = json.load(fh)
    return Primitive(pid=meta.get("pid") or os.path.basename(root), root=root,
                     meta=meta)


def write(root: str, meta: dict, impl: dict, tests: dict | None = None,
          context: dict | None = None) -> Primitive:
    os.makedirs(root, exist_ok=True)
    for sub, files in (("impl", impl), ("tests", tests or {}),
                       ("context", context or {})):
        for rel, text in files.items():
            p = os.path.join(root, sub, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(text)
    meta = dict(meta)
    meta.setdefault("tests", sorted((tests or {}).keys()))
    meta.setdefault("stats", {
        "files": len([f for f in impl if f.endswith(".py")]),
        "lines": sum(t.count("\n") for t in impl.values()),
        "n_tests": sum(t.count("def test_") for t in (tests or {}).values())})
    with open(os.path.join(root, "primitive.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return load(root)
