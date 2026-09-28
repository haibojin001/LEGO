"""Run configurations: every row of the configuration-definitions table.

A ``RunConfig`` fixes the mechanisms (R retrieval, A primitive adaptation,
E generic editing, V execution feedback, D diagnosis), the library view, the
attempt budget, and the three model positions. ``CONFIGS`` maps the names used
in ``experiments/*/experiment.yaml`` to configs; an experiment may override any
field per arm (e.g. ``models.backbone`` for backbone sweeps, ``budget`` for the
attempt ladder).

``arm_id`` is a stable hash of the full config, so the same arm requested by two
experiments is run once and its records are shared.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

DEFAULT_MODEL = "gpt-5.6-terra"


@dataclass
class Models:
    backbone: str = DEFAULT_MODEL
    resident: str | None = None      # None = same as backbone
    diagnosis: str | None = None     # None = same as backbone

    def resolved(self) -> dict:
        return {"backbone": self.backbone,
                "resident": self.resident or self.backbone,
                "diagnosis": self.diagnosis or self.backbone}


@dataclass
class LibraryView:
    path: str = "codeface"            # library root (a re-mined library is another path)
    allow_unvalidated: bool = False   # explicit exploratory use of flat proxies
    kinds: list | None = None         # None = all; else subset of mined/harvested/web/target
    exclude: list = field(default_factory=list)
    exclude_repos: list = field(default_factory=list)
    near_dup_thresh: float = 0.6
    subset_frac: float | None = None
    subset_seed: int = 0


@dataclass
class RunConfig:
    name: str
    library: bool = False             # R
    source: str = "codeface"          # codeface | file_rag
    consume: str = "none"             # none | paste | adapt | edit | context | import
    adapt_request: str = "freeform"   # freeform | structured
    carried_tests: bool = True        # V_i used by adaptation / validation
    edit_fields: list = field(default_factory=list)   # for consume=edit: tests, interface, context, deps
    collaboration: bool = True        # mu_{i->j} messages between activated primitives
    assess: bool = True               # resident relevance assessment before activation
    diagnosis: bool = False           # D
    budget: int = 5                   # b; 1 = single pass (no feedback)
    top_m: int = 2
    adapt_retries: int = 1
    decompose: str = "llm"            # llm | heuristic
    interface: str = "stub"           # stub | source (source is NOT a valid LEGO-REPO setting)
    embedder: str = "tfidf"
    view: LibraryView = field(default_factory=LibraryView)
    models: Models = field(default_factory=Models)
    temperature: float | None = None
    replicate: int = 0                # replicate index for repeated runs
    stages: str = ""                  # display label, e.g. RADV

    def to_dict(self) -> dict:
        return asdict(self)

    def arm_id(self) -> str:
        d = self.to_dict()
        d.pop("stages", None)
        # identical model assignments must hash identically whether a position
        # was left to default to the backbone or named explicitly
        d["models"] = self.models.resolved()
        if not self.library:
            d.pop("view", None)
            d["models"]["resident"] = d["models"]["backbone"]
        if not self.diagnosis:
            d["models"]["diagnosis"] = d["models"]["backbone"]
        name = d.pop("name")
        h = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]
        return f"{name}-{h}"


def _c(name, stages, **kw) -> RunConfig:
    view = kw.pop("view", {})
    cfg = RunConfig(name=name, stages=stages, **kw)
    for k, v in view.items():
        setattr(cfg.view, k, v)
    return cfg


_LIB = dict(library=True)
CONFIGS: dict[str, RunConfig] = {c.name: c for c in [
    # ---- no library
    _c("single_shot", "----", budget=1),
    _c("feedback", "---V"),
    _c("feedback_diag", "--DV", diagnosis=True),
    # ---- library, no adaptation
    _c("retrieval_one", "R---", consume="paste", budget=1, **_LIB),
    _c("retrieval_feedback", "R--V", consume="paste", **_LIB),
    _c("retrieval_diag", "R-DV", consume="paste", diagnosis=True, **_LIB),
    # ---- adaptation
    _c("retrieval_adapt", "RA-V", consume="adapt", **_LIB),
    _c("lego", "RADV", consume="adapt", diagnosis=True, **_LIB),
    _c("structured_adapt", "RADV", consume="adapt", adapt_request="structured",
       diagnosis=True, **_LIB),
    _c("lego_no_collab", "RADV", consume="adapt", collaboration=False,
       diagnosis=True, **_LIB),
    # ---- editing / passive-reuse controls
    _c("rag_edit", "RE DV", consume="edit", diagnosis=True, **_LIB),
    _c("file_rag_edit", "FE DV", consume="edit", source="file_rag",
       assess=False, diagnosis=True, **_LIB),
    _c("context_only", "RC DV", consume="context", diagnosis=True, **_LIB),
    _c("import_call", "RI DV", consume="import", diagnosis=True, **_LIB),
    # ---- primitive-field decomposition (matched RE DV / RADV)
    _c("rag_edit_V", "RE DV", consume="edit", edit_fields=["tests"],
       diagnosis=True, **_LIB),
    _c("rag_edit_VIX", "RE DV", consume="edit",
       edit_fields=["tests", "interface", "context"], diagnosis=True, **_LIB),
    _c("rag_edit_VIXD", "RE DV", consume="edit",
       edit_fields=["tests", "interface", "context", "deps"], diagnosis=True, **_LIB),
    _c("lego_no_V", "RADV", consume="adapt", carried_tests=False,
       diagnosis=True, **_LIB),
    # ---- provenance / contamination controls
    _c("lego_no_same_repo", "RADV", consume="adapt", diagnosis=True,
       view={"exclude": ["same_repo"]}, **_LIB),
    _c("lego_no_same_org", "RADV", consume="adapt", diagnosis=True,
       view={"exclude": ["same_repo", "same_org"]}, **_LIB),
    _c("lego_no_near_dup", "RADV", consume="adapt", diagnosis=True,
       view={"exclude": ["same_repo", "same_org", "near_dup"]}, **_LIB),
    _c("lego_remined", "RADV", consume="adapt", diagnosis=True,
       view={"path": "codeface_disjoint",
             "exclude": ["same_repo", "same_org", "near_dup"]}, **_LIB),
    # ---- library-source and size ablations
    _c("lego_mined_only", "RADV", consume="adapt", diagnosis=True,
       view={"kinds": ["mined"]}, **_LIB),
    _c("lego_mined_web", "RADV", consume="adapt", diagnosis=True,
       view={"kinds": ["mined", "web"]}, **_LIB),
    _c("lego_mined_harvested", "RADV", consume="adapt", diagnosis=True,
       view={"kinds": ["mined", "harvested"]}, **_LIB),
    _c("lego_web_only", "RADV", consume="adapt", diagnosis=True,
       view={"kinds": ["web"]}, **_LIB),
    _c("lego_harvested_only", "RADV", consume="adapt", diagnosis=True,
       view={"kinds": ["harvested"]}, **_LIB),
]}
for _s in range(3):
    CONFIGS[f"lego_half_s{_s}"] = _c(f"lego_half_s{_s}", "RADV", consume="adapt",
                                     diagnosis=True, view={"subset_frac": 0.5,
                                                           "subset_seed": _s},
                                     **_LIB)
for _m in (1, 4, 8):
    CONFIGS[f"lego_m{_m}"] = _c(f"lego_m{_m}", "RADV", consume="adapt",
                                diagnosis=True, top_m=_m, **_LIB)


def get(name: str, overrides: dict | None = None) -> RunConfig:
    if name not in CONFIGS:
        raise SystemExit(f"unknown config {name!r}; known: {sorted(CONFIGS)}")
    cfg = copy.deepcopy(CONFIGS[name])
    for k, v in (overrides or {}).items():
        if k == "models":
            for mk, mv in v.items():
                setattr(cfg.models, mk, mv)
        elif k == "view":
            for vk, vv in v.items():
                setattr(cfg.view, vk, vv)
        elif k == "label":
            cfg.name = v
        else:
            if not hasattr(cfg, k):
                raise SystemExit(f"config {name}: unknown field {k!r}")
            setattr(cfg, k, v)
    if (cfg.library and cfg.source == "codeface"
            and cfg.view.path == "codeface" and os.environ.get("LEGO_LIBRARY")):
        cfg.view.path = os.environ["LEGO_LIBRARY"]
    if cfg.library and os.environ.get("LEGO_ALLOW_UNVALIDATED") == "1":
        cfg.view.allow_unvalidated = True
    return cfg
