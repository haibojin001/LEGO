"""DECOMPOSE (Eq. 3): the ordered capability requirements Pi_t = (r_1..r_K).

For a LEGO-REPO task the target components are the package's modules. Each
requirement carries the interface the capability must expose (the module's
interface stub and exports), the package-internal modules it depends on, and the
external dependencies it may assume (the project's declared dependencies). The
order is topological over package-internal imports, so producers are adapted and
generated before their consumers and cross-primitive requirements mu_{i->j} can
flow along dependency edges within a round.

The capability text of each requirement (used as the retrieval request) is
written by the backbone with the activation prompt (``decompose: llm``), or
derived from identifiers (``decompose: heuristic``; no model call).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from lego.harness import core
from lego.harness.interface import (exports, external_imports, interface_text,
                                    internal_imports)

_ACT_PROMPT = open(os.path.join(os.path.dirname(__file__), "..", "prompts",
                                "activation.md")).read()
BATCH = 25


@dataclass
class Requirement:
    rid: str
    module: str
    interface: str
    exports: list
    internal_deps: list = field(default_factory=list)
    consumers: list = field(default_factory=list)
    allowed_deps: list = field(default_factory=list)
    capability: str = ""
    retrieval_request: str = ""
    sibling_exports: dict = field(default_factory=dict)
    level: int = 0

    def target_context(self) -> str:
        sib = "\n".join(f"  {m}: {', '.join(e[:15])}"
                        for m, e in list(self.sibling_exports.items())[:15])
        return (f"target module: {self.module}\n"
                f"required exports: {', '.join(self.exports[:40])}\n"
                f"may depend on: {', '.join(self.allowed_deps[:30]) or '(stdlib)'}"
                f"\nsibling interfaces it binds to:\n{sib or '  (none)'}\n"
                f"required signatures:\n{self.interface[:5000]}")


def _resolve(dep: str, modules: set) -> str | None:
    for cand in (dep + ".py", dep + "/__init__.py"):
        if cand in modules:
            return cand
    return None


def build_requirements(ws, interface_mode: str = "stub") -> list[Requirement]:
    mods = set(ws.modules)
    declared = sorted(set(core.declared_dep_names(ws.proj_dir)) |
                      set(ws.record.get("deps") or []))
    reqs = {}
    for m in ws.modules:
        src = ws.orig_src.get(m, "")
        deps = []
        for d in internal_imports(src, ws.pkg, m):
            r = _resolve(d, mods)
            if r and r != m and r not in deps:
                deps.append(r)
        ext = external_imports(src, ws.pkg)
        reqs[m] = Requirement(
            rid=f"r{len(reqs) + 1}", module=m,
            interface=interface_text(src, interface_mode),
            exports=exports(src), internal_deps=deps,
            allowed_deps=sorted(set(declared) | set(ext)))
    for m, r in reqs.items():
        for d in r.internal_deps:
            reqs[d].consumers.append(m)
        r.sibling_exports = {d: reqs[d].exports for d in r.internal_deps}
    # topological levels (cycles broken by module order)
    level, visiting = {}, set()

    def lv(m):
        if m in level:
            return level[m]
        if m in visiting:
            return 0
        visiting.add(m)
        v = 1 + max((lv(d) for d in reqs[m].internal_deps), default=-1)
        visiting.discard(m)
        level[m] = v
        return v
    for m in reqs:
        reqs[m].level = lv(m)
    return sorted(reqs.values(), key=lambda r: (r.level, r.module))


def describe(llm, ws, reqs: list[Requirement], statement: str = "",
             context: str = "") -> None:
    """Fill ``capability`` / ``retrieval_request`` with the backbone (batched)."""
    for i in range(0, len(reqs), BATCH):
        chunk = reqs[i:i + BATCH]
        listing = "\n\n".join(
            f"[{r.rid}] {r.module}\nexports: {', '.join(r.exports[:20])}\n"
            f"imports siblings: {', '.join(r.internal_deps[:10]) or '-'}\n"
            f"{r.interface[:1200]}" for r in chunk)
        prompt = (
            f"{_ACT_PROMPT}\n\n---\n\n# Construction Request\n"
            f"Reconstruct the Python package `{ws.pkg}` of `{ws.task.name}`.\n"
            f"{statement[:3000]}\n\n# Construction Context\n"
            f"{context[:4000] or '(initial empty repository)'}\n\n"
            f"# Target Interface (this batch)\n{listing}\n\n"
            "For THIS STEP return only the `requirements` list of the output "
            "format, one entry per module above, with `id` equal to the bracketed "
            "id, `target` the module path, `capability` one sentence describing "
            "the reusable functionality it needs (not its name), and "
            "`retrieval_request` a short query for CodeFace.")
        obj = llm.json(prompt, max_tokens=4000) or {}
        by = {str(x.get("id")): x for x in obj.get("requirements") or []
              if isinstance(x, dict)}
        for r in chunk:
            x = by.get(r.rid) or {}
            r.capability = str(x.get("capability") or "")[:400]
            r.retrieval_request = str(x.get("retrieval_request") or "")[:300]
    for r in reqs:
        if not r.capability:
            r.capability = heuristic_query(r, ws.pkg)
        if not r.retrieval_request:
            r.retrieval_request = r.capability


def heuristic_query(r: Requirement, pkg: str) -> str:
    return core.capability_terms(os.path.basename(r.module), r.interface, pkg)


def decompose(llm, ws, mode: str = "llm", interface_mode: str = "stub",
              statement: str = "") -> list[Requirement]:
    reqs = build_requirements(ws, interface_mode)
    if mode == "llm" and llm is not None:
        describe(llm, ws, reqs, statement)
    else:
        for r in reqs:
            r.capability = r.retrieval_request = heuristic_query(r, ws.pkg)
    return reqs
