"""One construction run of one configuration on one task (Sec. 3.4).

    Pi_t      = DECOMPOSE(q, C_t)                               decompose.py
    F_k       = top-m retrieval per requirement                 CodeFace view
    (s_i,u_i) = M_i^assess(P_i, r_k, C_t)                       resident.assess
    A_t       = activation over assessments                     activate()
    (P~_i,mu_i) = M_i^adapt(P_i, r_k(i), C_t U mu_->i)          resident.adapt
    S_t       = INTEGRATE({P~_i}, g_t)                          integrate()
    o_t       = EXECUTE(S_t)                                    Workspace.execute
    (v_t,phi_t) = DIAGNOSE(q, Pi_t, S_t, o_t)                   diagnose()

The configuration decides which of these stages exist and how a retrieved
candidate is consumed (``RunConfig.consume``):

  none     no library: every module is task-specific code g_t
  paste    retrieved implementation pasted into the generation prompt (R)
  adapt    resident adaptation with carried validation, then integration (RA)
  edit     scoped generic editing by the backbone in place of adaptation (RE/FE)
  context  primitive, contract and tests shown read-only; never delivered (RC)
  import   primitive vendored unchanged outside the target prefix and called (RI)

Scores are computed per round against the frozen grading key, so every
trajectory also yields the b=1..b attempt ladder as a prefix.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import re
import threading
import time

from lego.harness import core
from lego.harness.interface import exports as module_exports
from lego.library import resident
from lego.llm.client import LLM
from lego.pipeline.decompose import Requirement, decompose, describe
from lego.pipeline.plan_tests import synthesize as synthesize_plan_tests

_ACT_PROMPT = open(os.path.join(os.path.dirname(__file__), "..", "prompts",
                                "activation.md")).read()
_DIAG_PROMPT = open(os.path.join(os.path.dirname(__file__), "..", "prompts",
                                 "diagnosis.md")).read()
MAX_REVISE = int(os.environ.get("LEGO_MAX_REVISE", 0))
VENDOR_PKG = "_lego_vendor"


def _pool(n):
    return cf.ThreadPoolExecutor(max_workers=max(1, min(core.GEN_WORKERS, n)))


class Construction:
    def __init__(self, cfg, ws, view=None):
        self.cfg, self.ws, self.view = cfg, ws, view
        m = cfg.models.resolved()
        t = cfg.temperature
        self.backbone = LLM(m["backbone"], "backbone", t)
        self.res_model = m["resident"]
        self.diag = LLM(m["diagnosis"], "diagnosis", t)
        self.temperature = t
        self.statement = ws.task.statement()
        self.funnel = {"requirements": 0, "query": 0, "cand": 0, "retain": 0,
                       "activated": 0, "adapt_ok": 0, "adapt_rejected": 0,
                       "adapt_unsuitable": 0, "adapt_failed": 0, "messages": 0,
                       "edited": 0, "vendored": 0}
        self.active: dict[str, dict] = {}   # module -> activation record
        self.components: dict[str, str] = {}  # module -> transformed component
        self.inbox: dict[str, list] = {}    # module -> incoming mu
        self.context: list[dict] = []      # accumulated diagnosis phi_t
        self.log = []
        self.successful_adaptation_usage: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _count(self, key: str, n: int = 1):
        with self._lock:
            self.funnel[key] = self.funnel.get(key, 0) + n

    def _context_text(self) -> str:
        return "\n".join(
            f"round {step['round']}: {step['decision']}; "
            f"failures={step['cause']}; "
            f"implicated={', '.join(step['implicated'])}; "
            f"guidance={step['text']}"
            for step in self.context)

    # ------------------------------------------------------------ retrieval
    def retrieve_and_activate(self, reqs: list[Requirement],
                              context: str = ""):
        cfg = self.cfg
        cand = {}
        for r in reqs:
            q = f"{r.retrieval_request}\n{r.capability}\n{' '.join(r.exports[:20])}"
            cand[r.module] = self.view.search(q, top_m=cfg.top_m)
            self.funnel["query"] += 1
            self.funnel["cand"] += len(cand[r.module])
        assessed: dict[str, list] = {m: [] for m in cand}
        if cfg.assess:
            jobs = [(r, p) for r in reqs for p in cand[r.module]]
            with _pool(len(jobs)) as pool:
                futs = {pool.submit(resident.assess, self._resident(p), p, r): (r, p)
                        for r, p in jobs}
                for f in cf.as_completed(futs):
                    r, p = futs[f]
                    try:
                        a = f.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"    [assess] {p.pid}: {type(e).__name__}", flush=True)
                        continue
                    if a.suitable:
                        assessed[r.module].append((p, a))
        else:
            assessed = {m: [(p, None) for p in ps] for m, ps in cand.items()}
        for m in assessed:
            order = {p.pid: i for i, p in enumerate(cand[m])}
            assessed[m].sort(key=lambda pa: order.get(pa[0].pid, 99))
            self.funnel["retain"] += len(assessed[m])
        self._activate(reqs, assessed, context)

    def _activate(self, reqs, assessed, context: str = ""):
        """Activation over candidate assessments (activation prompt), with a
        deterministic fallback: first suitable candidate whose dependency
        closure is compatible with the target's allowed dependencies."""
        by_mod = {r.module: r for r in reqs}
        pending = [r for r in reqs if assessed.get(r.module)]
        chosen, direct = {}, set()
        for i in range(0, len(pending), 20):
            chunk = pending[i:i + 20]
            listing = "\n\n".join(
                f"[{r.rid}] target {r.module}: {r.capability}\n" + "\n".join(
                    f"  candidate {p.pid}: {p.card()[:400]}\n"
                    f"    usage proposal: {(a.usage if a else '')[:300]}"
                    for p, a in assessed[r.module]) for r in chunk)
            obj = self.backbone.json(
                f"{_ACT_PROMPT}\n\n---\n\n# Construction Request\nReconstruct "
                f"`{self.ws.pkg}`.\n\n# Construction Context\n"
                f"{context or '(initial empty repository)'}\n\n"
                f"# Candidate Assessments\n{listing}\n\n"
                "For THIS STEP return only `activated_primitives` and "
                "`task_specific_code`.", max_tokens=4000) or {}
            for x in obj.get("activated_primitives") or []:
                if isinstance(x, dict):
                    chosen[str(x.get("requirement"))] = x
            for x in obj.get("task_specific_code") or []:
                if isinstance(x, dict):
                    direct.add(str(x.get("requirement")))
        for r in pending:
            opts = assessed[r.module]
            pick, areq = None, ""
            x = chosen.get(r.rid)
            if x is not None:
                pick = next((pa for pa in opts
                             if pa[0].pid == str(x.get("primitive"))), None)
                areq = str(x.get("adaptation_requirement") or "")
            if pick is None and x is None and r.rid not in direct:
                compat = [pa for pa in opts
                          if set(pa[0].external_deps) <= set(r.allowed_deps)]
                pick = compat[0] if compat else None
            if pick is None:
                continue
            p, a = pick
            self.active[r.module] = {"prim": p, "usage": a.usage if a else "",
                                     "adaptation_requirement": areq,
                                     "req": by_mod[r.module]}
        self.funnel["activated"] = len(self.active)

    def _resident(self, prim) -> LLM:
        return LLM(self.res_model, "resident", self.temperature)

    # ------------------------------------------------------------ transform
    def transform(self, reqs: list[Requirement], files: dict,
                  modules: list[str] | None = None, diagnostics: dict | None = None):
        """Adapt or edit the activated primitives, level by level so that
        requirements emitted by producers reach consumers in the same round."""
        cfg = self.cfg
        if cfg.consume not in ("adapt", "edit"):
            return
        todo = [r for r in reqs if r.module in self.active
                and (modules is None or r.module in modules)]
        for lvl in sorted({r.level for r in todo}):
            batch = [r for r in todo if r.level == lvl]
            with _pool(len(batch)) as pool:
                futs = {pool.submit(self._transform_one, r, files,
                                    (diagnostics or {}).get(r.module, ""),
                                    modules is None): r
                        for r in batch}
                for f in cf.as_completed(futs):
                    r = futs[f]
                    try:
                        f.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"    [transform] {r.module}: {type(e).__name__}: {e}",
                              flush=True)

    def _transform_one(self, r: Requirement, files: dict, diag: str,
                       initial: bool = True):
        """``initial`` separates the round-0 funnel (adapt_*) from
        re-adaptations triggered by diagnosis (readapt_*)."""
        act = self.active[r.module]
        p = act["prim"]
        if self.cfg.consume == "edit":
            code = self._edit(p, r, diag)
            if code:
                self.components[r.module] = code
                self._count("edited" if initial else "reedited")
            return
        model = self._resident(p)
        res = resident.adapt(
            model, p, r, self.ws, files, usage=act["usage"],
            incoming=self.inbox.get(r.module, []) if self.cfg.collaboration else [],
            adaptation_req=act["adaptation_requirement"], diagnostics=diag,
            request=self.cfg.adapt_request, carried_tests=self.cfg.carried_tests,
            retries=self.cfg.adapt_retries)
        self._count(("adapt_" if initial else "readapt_")
                    + {"adapted": "ok"}.get(res.status, res.status))
        act["last_status"] = res.status
        print(f"    [adapt] {r.module} <- {p.pid}: {res.status} "
              f"(attempts {res.attempts})"
              + (f" {res.validation.get('log', '')[-200:]!r}"
                 if res.status == "rejected" else ""), flush=True)
        if res.status != "adapted":
            self.components.pop(r.module, None)
            return
        with self._lock:
            key = f"resident|{model.alias}"
            total = self.successful_adaptation_usage.setdefault(
                key, {"calls": 0, "in": 0, "out": 0})
            for field in total:
                total[field] += res.call_usage.get(field, 0)
        self.components[r.module] = res.code
        act["updated_interface"] = {
            "exports": res.exports, "contract": res.contract}
        act["updated_deps"] = res.deps
        if self.cfg.collaboration:
            for msg in res.messages:
                tgt = self._resolve_target(str(msg.get("target", "")), r)
                if tgt:
                    with self._lock:
                        self.inbox.setdefault(tgt, []).append(
                            {"source": r.module,
                             "requirement": str(msg.get("requirement", ""))[:800]})
                    self._count("messages")

    def _resolve_target(self, name: str, r: Requirement) -> str | None:
        name = name.strip().replace(".", "/")
        cands = r.consumers + r.internal_deps
        for m in cands:
            if m not in self.active:
                continue
            stem = m[:-3]
            if name in (m, stem) or stem.endswith("/" + name) or \
                    os.path.basename(stem) == os.path.basename(name):
                return m
        return None

    def _edit(self, p, r: Requirement, diag: str) -> str:
        f = set(self.cfg.edit_fields)
        extra = ""
        if "interface" in f:
            extra += (f"\nCOMPONENT INTERFACE CONTRACT:\nexports: "
                      f"{', '.join(p.exports)}\n{p.signatures[:2500]}\n"
                      f"{p.contract[:1500]}\n")
        if "context" in f:
            src = p.source
            extra += (f"\nCOMPONENT CONTEXT AND PROVENANCE:\n{p.summary[:1200]}\n"
                      f"source: {src.get('repo', '')} "
                      f"{', '.join(src.get('paths', [])[:5])}\n"
                      + "\n".join(
                          f"# {name}\n{text[:1500]}"
                          for name, text in list(p.context.items())[:2]))
        if "deps" in f:
            extra += f"\nCOMPONENT DEPENDENCY CLOSURE: {', '.join(p.external_deps)}\n"
        if "tests" in f and p.tests:
            extra += f"\nCOMPONENT TESTS:\n{p.tests_text(limit=4000)}\n"
        return self.backbone.code(
            "You are a scoped editing subagent. Edit the RETRIEVED CODE into the "
            f"module `{self.ws.pkg}/{r.module}` so that it satisfies the TARGET "
            "INTERFACE exactly (names, signatures, defaults, exceptions, return "
            "shapes all come from the target). Output only the complete module.\n\n"
            f"TARGET INTERFACE:\n{r.interface}\n\n"
            f"SIBLING INTERFACES:\n{r.target_context()[:3000]}\n\n"
            f"RETRIEVED CODE:\n{p.impl_text(limit=8000)}\n{extra}"
            + (f"\nREVISION GUIDANCE:\n{diag[:3000]}\n" if diag else ""),
            max_tokens=16000)

    # ------------------------------------------------------------ integrate
    def _module_prompt(self, r: Requirement) -> str:
        cfg = self.cfg
        head = (f"Implement the module `{self.ws.pkg}/{r.module}` of the "
                f"`{self.ws.task.name}` package. Reproduce every public name, "
                f"signature and documented behaviour of the TARGET INTERFACE; "
                f"implementations are yours to write.\n\n"
                f"TARGET INTERFACE:\n{r.interface}\n\n"
                f"SIBLING MODULES (import only names they define):\n"
                + "\n".join(f"  {m}: {', '.join(e[:20])}"
                            for m, e in r.sibling_exports.items())[:3000])
        inc = self.inbox.get(r.module) if cfg.collaboration else None
        if inc:
            head += "\n\nREQUIREMENTS FROM ACTIVATED COMPONENTS:\n" + "\n".join(
                f"- {m['source']}: {m['requirement']}" for m in inc)
        act = self.active.get(r.module)
        body = ""
        if act and cfg.consume == "paste":
            body = ("\n\nPRIMITIVE to adapt:\n" + act["prim"].impl_text(limit=8000))
        elif act and cfg.consume == "context":
            p = act["prim"]
            body = ("\n\nREFERENCE (read-only; nothing from it is copied into "
                    "the repository, write every line yourself):\n"
                    f"{p.impl_text(limit=6000)}\nCONTRACT: {', '.join(p.exports)}\n"
                    f"{p.contract[:1500]}\nTESTS:\n{p.tests_text(limit=2500)}")
        elif act and cfg.consume == "import":
            p = act["prim"]
            vp = f"{VENDOR_PKG}.p_{_safe(p.pid)}"
            body = (f"\n\nA reusable component is installed as the package "
                    f"`{vp}` (do not modify it). Import it and call it to "
                    f"implement this module; write only the glue.\n"
                    f"Its modules: {', '.join(sorted(p.impl))}\n"
                    f"Its exports: {', '.join(p.exports[:40])}\n"
                    f"{p.signatures[:3000]}")
        elif r.module in self.components:
            updated = (act or {}).get("updated_interface") or {}
            body = ("\n\nADAPTED COMPONENT (already rewritten for this target; "
                    "integrate it, and where it disagrees with the TARGET "
                    f"INTERFACE the target wins):\n"
                    f"{self.components[r.module][:9000]}\n"
                    f"UPDATED COMPONENT EXPORTS: "
                    f"{', '.join(updated.get('exports') or [])}\n"
                    f"UPDATED COMPONENT CONTRACT: "
                    f"{str(updated.get('contract') or '')[:1500]}\n"
                    f"UPDATED COMPONENT DEPENDENCIES: "
                    f"{', '.join((act or {}).get('updated_deps') or [])}")
        return head + body + "\n\nOutput ONLY the complete module code."

    def integrate(self, reqs: list[Requirement]) -> dict:
        # A locally validated adapted primitive is executable source, not
        # context for the backbone to regenerate. Place that source in the
        # delivered tree. The backbone writes g_t for requirements left
        # outside the activated library (including rejected adaptations).
        files = {r.module: self.components[r.module]
                 for r in reqs if r.module in self.components
                 and self.cfg.consume in ("adapt", "edit")}
        direct = [r for r in reqs if r.module not in files]
        with _pool(len(direct)) as pool:
            futs = {pool.submit(self.backbone.code, self._module_prompt(r),
                                16000 if len(r.interface) > 6000 else 8000): r
                    for r in direct}
            for f in cf.as_completed(futs):
                r = futs[f]
                try:
                    code = f.result()
                except Exception:  # noqa: BLE001
                    code = ""
                files[r.module] = code or "# generation failed\n"
        return files

    def vendor(self):
        if self.cfg.consume != "import" or not self.active:
            core.EXTRA_PYTHONPATH[:] = []
            return
        tree = {f"{VENDOR_PKG}/__init__.py": ""}
        for act in self.active.values():
            p = act["prim"]
            base = f"{VENDOR_PKG}/p_{_safe(p.pid)}"
            tree[f"{base}/__init__.py"] = ""
            for rel, code in p.impl.items():
                tree[f"{base}/{rel}"] = code
                d = os.path.dirname(rel)
                while d:
                    tree.setdefault(f"{base}/{d}/__init__.py", "")
                    d = os.path.dirname(d)
            self.funnel["vendored"] += 1
        core.EXTRA_PYTHONPATH[:] = [self.ws.write_vendor(tree)]

    # ------------------------------------------------------------ diagnose
    def diagnose(self, reqs, files, res, prev: str, modified: list[str]):
        suspects = _blame(files, res.feedback)
        ordered = list(dict.fromkeys(suspects + list(files)))
        state = "\n\n".join(
            f"## {m} ({len(files[m] or '')}B)\n"
            f"exports: {', '.join(sorted(module_exports(files[m] or ''))[:12])}\n"
            f"{(files[m] or '')[:1200]}" for m in ordered)[:12000]
        plan = "\n".join(f"  {r.module}: {r.capability[:120]}"
                         + (f" [primitive {self.active[r.module]['prim'].pid}]"
                            if r.module in self.active else "")
                         for r in reqs)[:4000]
        obj = self.diag.json(
            f"{_DIAG_PROMPT}\n\n---\n\n# Construction Request\n"
            f"{self.statement[:2500]}\nReconstruct the package "
            f"`{self.ws.pkg}`.\n\n# Current Plan\n{plan}\n\n"
            f"# Repository State\n{state}\n\n# Execution Evidence\n"
            f"build_ok={res.build_ok} import_ok={res.import_ok} "
            f"pytest_exit={res.pytest_exit} "
            f"native_passed={res.passed} "
            f"native_failed={res.failed} synthesized_passed={res.plan_passed} "
            f"synthesized_failed={res.plan_failed}\n"
            f"{res.feedback[-6000:]}\n\n# Modified Components\n"
            f"{', '.join(modified)}\n\n# Previous Diagnosis\n{prev or '(none)'}",
            max_tokens=2500)
        if not isinstance(obj, dict):
            return {"decision": "FAIL", "implicated": [], "guidance": {},
                    "text": "", "cause": "unparsed", "criteria": {}}
        mods = [m for m in (obj.get("implicated_components") or [])
                if isinstance(m, str) and m in files]
        guide = {}
        for g in obj.get("revision_guidance") or []:
            if isinstance(g, dict) and g.get("component") in files:
                guide[g["component"]] = str(g.get("guidance", ""))[:1200]
        types = [str(f.get("type", "")) for f in obj.get("observed_failures") or []
                 if isinstance(f, dict)]
        text = "; ".join(f"{m}: {g}" for m, g in guide.items())[:3000]
        criteria = obj.get("criteria")
        if not isinstance(criteria, dict):
            criteria = {}
        decision = str(obj.get("decision", "")).upper()
        if decision not in ("PASS", "FAIL"):
            decision = "FAIL"
        expected = ("task_completion", "build_success", "test_pass_rate",
                    "interface_consistency", "code_quality")
        if any(str(criteria.get(k, "")).lower() != "pass" for k in expected):
            decision = "FAIL"
        if (not res.build_ok or not res.import_ok or res.failed or res.plan_failed
                or res.verdict != "ok" or _interface_gaps(reqs, files)):
            decision = "FAIL"
        return {"decision": decision, "implicated": mods, "guidance": guide,
                "text": text, "cause": ",".join(types[:5]),
                "criteria": criteria}

    # ------------------------------------------------------------ revise
    def revise(self, reqs, files, blamed, feedback, guide, attempts):
        by = {r.module: r for r in reqs}
        targets = [m for m in dict.fromkeys(blamed)
                   if m in by and attempts.get(m, 1) < self.cfg.budget]
        if MAX_REVISE > 0:
            targets = targets[:MAX_REVISE]
        if not targets:
            return []
        if self.cfg.consume in ("adapt", "edit"):
            re_adapt = [m for m in targets if m in self.active]
            if re_adapt:
                # Repository-level native test output belongs to the backbone.
                # Residents receive only their refreshed localized requirement,
                # sibling interfaces and incoming primitive messages.
                self.transform(reqs, files, modules=re_adapt)
        sib = "\n".join(f"# {fn} defines: {sorted(module_exports(c or ''))[:20]}"
                        for fn, c in files.items())[:3000]

        def fix(m):
            r = by[m]
            if (self.cfg.consume in ("adapt", "edit")
                    and m in self.components):
                # A newly validated adaptation (or scoped generic edit)
                # replaces this component. Cross-component failures will be
                # diagnosed on the next execution round.
                return m, self.components[m]
            comp = ""
            if m in self.components and self.cfg.consume in ("adapt", "edit"):
                act = self.active.get(m) or {}
                comp = ("\n\nADAPTED COMPONENT (revised):\n"
                        + self.components[m][:6000]
                        + "\nUPDATED CONTRACT:\n"
                        + str((act.get("updated_interface") or {}).get(
                            "contract", ""))[:1200]
                        + "\nUPDATED DEPENDENCIES: "
                        + ", ".join(act.get("updated_deps") or []))
            g = guide.get(m, "")
            return m, self.backbone.code(
                f"Fix module `{self.ws.pkg}/{m}` so the package imports and its "
                f"tests pass.\n\nCURRENT CODE:\n{files[m][:9000]}\n\n"
                f"TARGET INTERFACE (source of truth for names/behaviour):\n"
                f"{r.interface[:6000]}\n\nFAILURE / TEST OUTPUT:\n{feedback[-5000:]}"
                + (f"\n\nREVISION GUIDANCE:\n{g}" if g else "")
                + f"\n\nSIBLING MODULES (import only names they define):\n{sib}"
                + comp + "\n\nOutput ONLY the complete corrected module.",
                max_tokens=12000)
        with _pool(len(targets)) as pool:
            for m, code in pool.map(fix, targets):
                attempts[m] = attempts.get(m, 1) + 1
                if code:
                    try:
                        compile(code, m, "exec")
                        files[m] = code
                    except SyntaxError:
                        pass
        return targets

    # ------------------------------------------------------------ main loop
    def run(self, score_fn) -> dict:
        t0 = time.time()
        cfg, ws = self.cfg, self.ws
        reqs = decompose(self.backbone, ws, cfg.decompose, cfg.interface,
                         self.statement)
        self.funnel["requirements"] = len(reqs)
        if cfg.library and self.view is not None:
            self.retrieve_and_activate(reqs)
            self.transform(reqs, {})
            self.vendor()
        files = self.integrate(reqs)
        traj, best = [], None
        stalls, timeouts = 0, 0
        attempts = {r.module: 1 for r in reqs}
        modified = list(files)
        max_rounds = 1 + (cfg.budget - 1) * len(reqs)
        for rnd in range(1, max_rounds + 1):
            plan_tests = synthesize_plan_tests(
                self.backbone, ws.pkg, self.statement, reqs)
            res = ws.execute(files, plan_tests=plan_tests)
            s = score_fn(res)
            missing = _interface_gaps(reqs, files)
            traj.append({"round": rnd, "build_ok": res.build_ok,
                         "import_ok": res.import_ok,
                         "passed": s["passed"], "failed": res.failed,
                         "pytest_exit": res.pytest_exit,
                         "verdict": res.verdict, "score": s["score"],
                         "missing_exports": len(missing),
                         "plan_passed": res.plan_passed,
                         "plan_failed": res.plan_failed,
                         "feedback": res.feedback[-4000:],
                         "t": round(time.time() - t0)})
            print(f"    [r{rnd}] import={'ok' if res.import_ok else 'FAIL'} "
                  f"pass={s['passed']} fail={res.failed} prov={res.verdict} "
                  f"score={s['score']:.3f}", flush=True)
            if best is None or s["passed"] > best["passed"] or (
                    s["passed"] == best["passed"] and best["verdict"] != "ok"
                    and res.verdict == "ok"):
                best = {"round": rnd, "passed": s["passed"], "score": s["score"],
                        "verdict": res.verdict, "files": dict(files),
                        "exercised": res.exercised}
                stalls = 0
            elif res.import_ok:
                stalls += 1
            diagnosis = None
            if cfg.diagnosis:
                diagnosis = self.diagnose(
                    reqs, files, res, self._context_text(), modified)
                traj[-1]["diagnosis"] = {
                    "decision": diagnosis["decision"],
                    "criteria": diagnosis["criteria"],
                    "implicated": diagnosis["implicated"],
                    "cause": diagnosis["cause"]}
                self.context.append({"round": rnd, **diagnosis})
            execution_pass = (res.build_ok and res.import_ok
                              and res.verdict == "ok"
                              and res.failed == 0 and res.plan_failed == 0
                              and res.passed > 0 and not missing)
            if execution_pass and (not cfg.diagnosis
                                   or diagnosis["decision"] == "PASS"):
                break
            if rnd == max_rounds:
                break
            if res.verdict == "timeout":
                timeouts += 1
                if timeouts >= 2:
                    break
            if stalls >= 2:
                break
            feedback = res.feedback
            if res.verdict == "timeout":
                feedback = ("The test suite did not finish within the time limit; "
                            "the generated code probably hangs (work at import "
                            "time, an unbounded loop, a blocking read).\n" + feedback)
            if missing:
                feedback += "\n\nINTERFACE CHECK: missing public names:\n" + \
                    "\n".join(f"  {m}: {', '.join(n[:15])}" for m, n in missing.items())
            blamed = _blame(files, feedback) + [m for m in missing
                                                if m not in _blame(files, feedback)]
            guide = diagnosis["guidance"] if diagnosis else {}
            if diagnosis and diagnosis["implicated"]:
                blamed = diagnosis["implicated"] + [
                    m for m in missing if m not in diagnosis["implicated"]]
            if not blamed:
                blamed = list(modified or files)
            affected = [r for r in reqs if r.module in blamed
                        and attempts.get(r.module, 1) < cfg.budget]
            if not affected:
                break
            if cfg.decompose == "llm":
                context = "\n\n".join(
                    f"## {r.module}\n{files.get(r.module, '')[:1200]}"
                    for r in affected)
                describe(self.backbone, ws, affected,
                         self.statement + "\n\nLatest execution feedback:\n"
                         + feedback[-2000:],
                         context=context + "\n\nDiagnosis history:\n"
                         + self._context_text())
            if cfg.library and self.view is not None:
                for r in affected:
                    self.active.pop(r.module, None)
                    self.components.pop(r.module, None)
                self.retrieve_and_activate(
                    affected, context=feedback[-2000:] + "\n"
                    + self._context_text())
                if cfg.consume == "import":
                    self.vendor()
            modified = self.revise(reqs, files, blamed, feedback, guide,
                                   attempts)
            if not modified:
                break
        core.EXTRA_PYTHONPATH[:] = []
        return {"trajectory": traj, "best": best, "funnel": self.funnel,
                "module_attempts": attempts,
                "successful_adaptation_usage": self.successful_adaptation_usage,
                "context": self.context,
                "activated": {m: a["prim"].pid for m, a in self.active.items()},
                "activation_status": {m: a.get("last_status", "")
                                      for m, a in self.active.items()},
                "seconds": round(time.time() - t0, 1)}


def _interface_gaps(reqs, files) -> dict:
    out = {}
    for r in reqs:
        have = module_exports(files.get(r.module, "") or "")
        miss = [e for e in r.exports if e not in have and not e.startswith("__")]
        if miss:
            out[r.module] = miss
    return out


_TB_FILE = re.compile(r'(?:^|[\s"\'(/])([A-Za-z_][\w\-]*\.py)(?=["\'):,\s]|:\d|$)',
                      re.M)


def _blame(files, feedback) -> list[str]:
    """Modules named in a file position of the failure text (traceback path or
    'cannot import name X from pkg.Y'); basenames shared by several modules are
    ambiguous and skipped."""
    by_base = {}
    for k in files:
        by_base.setdefault(os.path.basename(k), []).append(k)
    out = []

    def add(base):
        ks = by_base.get(base, [])
        if len(ks) == 1 and ks[0] not in out:
            out.append(ks[0])
    for c in _TB_FILE.findall(feedback or ""):
        add(os.path.basename(c))
    for m in re.findall(r"(?:from|named)\s+'?[\w.]*\.(\w+)'?", feedback or ""):
        add(m + ".py")
    return out


def _safe(pid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", pid)[:60]
