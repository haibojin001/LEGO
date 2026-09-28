"""The resident model M_i of a Code Primitive: ASSESS (Eq. 1) and ADAPT (Eq. 2).

Both modes use the Code Primitive system prompt (``lego/prompts/primitive.md``)
with the primitive's own state (C_i, I_i, D_i, V_i, X_i) and a bounded target
context: target module path, required exports and signatures, sibling
interfaces, incoming requirements mu_{->i}, and diagnostics from a rejected
attempt. The resident never sees the native tests, unrelated implementations,
the global plan, or repository-level diagnosis.

Carried validation is executed by the harness (``validate``): the adapted module
is placed at its target path inside a scratch tree whose other modules are the
current round's generated code (or import-safe interface stubs), and the retained
carried tests are run in the task environment. An adaptation is rejected when its
response cannot be parsed, a required field is missing, or a retained carried
test fails; after the retry budget the requirement returns to repository-level
handling (task-specific code g_t).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field

from lego.harness import core
from lego.harness.interface import exports as module_exports
from lego.llm.client import LLM

_PROMPT = open(os.path.join(os.path.dirname(__file__), "..", "prompts",
                            "primitive.md")).read()
VALIDATE_TIMEOUT = int(os.environ.get("LEGO_PRIM_TEST_TIMEOUT", 180))


@dataclass
class Assessment:
    pid: str
    suitable: bool
    usage: str = ""
    required: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    ok: bool = True                 # False = unparsable response


@dataclass
class Adapted:
    pid: str
    module: str
    status: str                     # adapted | rejected | failed | unsuitable
    code: str = ""
    exports: list = field(default_factory=list)
    contract: str = ""
    deps: list = field(default_factory=list)
    messages: list = field(default_factory=list)   # [{"target", "requirement"}]
    tests: dict = field(default_factory=dict)       # retained tests: name->code
    test_actions: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    attempts: int = 0
    call_usage: dict = field(default_factory=dict)  # token usage of this response


def _state(prim, carried_tests: bool = True, max_impl: int = 8000) -> str:
    parts = [f"## C_i implementation\n{prim.impl_text(limit=max_impl)}",
             f"## I_i interface\nexports: {', '.join(prim.exports)}\n"
             f"{prim.signatures[:3000]}\ncontract: {prim.contract[:2000]}",
             f"## D_i dependencies\n{', '.join(prim.external_deps) or '(stdlib only)'}"]
    if carried_tests and prim.tests:
        parts.append(f"## V_i carried tests\n{prim.tests_text(limit=5000)}")
    src = prim.source
    parts.append(f"## X_i context and provenance\n{prim.summary}\n"
                 f"source: {src.get('repo', '')} {', '.join(src.get('paths', [])[:5])}")
    for p, c in list(prim.context.items())[:2]:
        parts.append(f"# {p}\n{c[:1500]}")
    return "\n\n".join(parts)


def assess(llm: LLM, prim, req, ctx: str = "") -> Assessment:
    prompt = (f"MODE: ASSESS\n\n# Primitive state\n{_state(prim)}\n\n"
              f"# Requirement\ncapability: {req.capability}\n"
              f"target: {req.module}\n\n# Target Context\n{req.target_context()}"
              f"\n{ctx}")
    obj = llm.json(_PROMPT + "\n\n---\n\n" + prompt, max_tokens=1500)
    if not isinstance(obj, dict) or "assessment" not in obj:
        return Assessment(prim.pid, False, ok=False)
    return Assessment(prim.pid,
                      str(obj.get("assessment", "")).upper() == "SUITABLE",
                      usage=str(obj.get("usage_proposal", ""))[:1500],
                      required=list(obj.get("required_adaptations") or [])[:10],
                      notes=list(obj.get("compatibility_notes") or [])[:10])


def _structured_request(req) -> str:
    payload = {"target_module": req.module, "exports": req.exports,
               "signatures": req.interface[:6000],
               "sibling_bindings": {d: e for d, e in req.sibling_exports.items()}}
    return ("# Requirement (structured payload; no prose channel)\n"
            + json.dumps(payload, indent=1))


def _freeform_request(req, usage: str, incoming: list, adaptation_req: str) -> str:
    inc = "\n".join(f"- from {m.get('source')}: {m.get('requirement')}"
                    for m in incoming) or "(none)"
    return (f"# Requirement\n{adaptation_req or req.capability}\n"
            f"Your own usage proposal: {usage or '(none)'}\n\n"
            f"# Target Context\n{req.target_context()}\n\n"
            f"# Incoming Requirements\n{inc}")


def adapt(llm: LLM, prim, req, ws, current_files: dict, *, usage: str = "",
          incoming: list | None = None, adaptation_req: str = "",
          diagnostics: str = "", request: str = "freeform",
          carried_tests: bool = True, retries: int = 1) -> Adapted:
    """ADAPT with carried validation and the rejection rule."""
    incoming = incoming or []
    body = (_structured_request(req) if request == "structured" else
            _freeform_request(req, usage, incoming, adaptation_req))
    diag = diagnostics
    last = Adapted(prim.pid, req.module, "failed")
    for attempt in range(1, retries + 2):
        prompt = (f"MODE: ADAPT\n\n# Primitive state\n"
                  f"{_state(prim, carried_tests)}\n\n{body}")
        if diag:
            prompt += f"\n\n# Adaptation Diagnostics\n{diag[:4000]}"
        before = dict(getattr(llm, "usage", {}))
        obj = llm.json(_PROMPT + "\n\n---\n\n" + prompt, max_tokens=16000)
        res = _parse_adapt(obj, prim, req, carried_tests)
        res.attempts = attempt
        after = getattr(llm, "usage", {})
        res.call_usage = {field: after.get(field, 0) - before.get(field, 0)
                          for field in ("calls", "in", "out")}
        if res.status != "adapted":
            last = res
            diag = res.validation.get("log", "response could not be parsed or "
                                             "was missing a required field")
            continue
        ok, log = validate(ws, req, res, current_files)
        res.validation = {"status": "PASS" if ok else "FAIL", "log": log[-3000:]}
        if ok:
            return res
        res.status = "rejected"
        last, diag = res, log
    return last


def _parse_adapt(obj, prim, req, carried_tests) -> Adapted:
    out = Adapted(prim.pid, req.module, "failed")
    if not isinstance(obj, dict):
        return out
    if str(obj.get("mode", "")).upper() != "ADAPT":
        out.validation = {"log": "response mode must be ADAPT"}
        return out
    if str(obj.get("status", "")).upper() == "FAILURE":
        out.status = "unsuitable"
        return out
    required = ("implementation", "interface", "dependencies",
                "outgoing_requirements", "carried_tests")
    missing = [key for key in required if key not in obj]
    if missing or str(obj.get("status", "")).upper() != "SUCCESS":
        out.validation = {"log": f"missing or invalid ADAPT fields: {missing}"}
        return out
    impl = obj["implementation"]
    iface = obj["interface"]
    if (not isinstance(impl, list) or len(impl) != 1
            or not isinstance(impl[0], dict)
            or not isinstance(impl[0].get("content"), str)
            or not isinstance(iface, dict)
            or not isinstance(iface.get("exports"), list)
            or not isinstance(iface.get("contract"), str)
            or not isinstance(obj["dependencies"], list)
            or not isinstance(obj["outgoing_requirements"], list)
            or not isinstance(obj["carried_tests"], list)):
        out.validation = {"log": "ADAPT response has invalid field types"}
        return out
    code = impl[0]["content"]
    if not code.strip():
        out.validation = {"log": "adapted implementation is empty"}
        return out
    try:
        compile(code, req.module, "exec")
    except SyntaxError as e:
        out.validation = {"log": f"SyntaxError in adapted implementation: {e}"}
        return out
    out.code = code
    out.exports = [str(e) for e in iface["exports"]][:50]
    out.contract = iface["contract"][:3000]
    out.deps = [str(d) for d in obj["dependencies"]][:30]
    out.messages = [m for m in obj["outgoing_requirements"]
                    if isinstance(m, dict) and m.get("target")][:10]
    out.test_actions = [t for t in obj["carried_tests"]
                        if isinstance(t, dict)]
    if carried_tests and prim.tests:
        actions = {str(t.get("test")): t for t in out.test_actions}
        for name, text in prim.tests.items():
            a = actions.get(name) or actions.get(os.path.basename(name))
            if a is None:
                out.validation = {"log": f"test {name} lacks an explicit action"}
                return out
            act = str(a.get("action", "")).lower()
            if act not in ("kept", "rewritten", "dropped"):
                out.validation = {"log": f"test {name} has invalid action"}
                return out
            if act == "dropped":
                if not str(a.get("justification", "")).strip():
                    out.validation = {"log": f"test {name} dropped without "
                                             "justification"}
                    return out
                continue
            if act == "rewritten":
                if (not str(a.get("justification", "")).strip()
                        or not isinstance(a.get("content"), str)
                        or not a["content"].strip()):
                    out.validation = {"log": f"test {name} marked rewritten "
                                             "without justification or content"}
                    return out
                out.tests[name] = str(a["content"])
            else:
                out.tests[name] = text
        if not out.tests:
            out.validation = {"log": "all carried tests were dropped"}
            return out
    out.status = "adapted"
    return out


def _sibling(current: str | None, original: str) -> str:
    """A sibling module for the validation tree: its current generated code if
    that provides the sibling's whole public interface, else an import-safe
    stub. A sibling that is still broken must not fail this component's tests."""
    from lego.harness.interface import stub
    if current:
        need = {e for e in module_exports(original) if not e.startswith("__")}
        if need <= set(module_exports(current)):
            return current
    return stub(original, import_safe=True)


def validate(ws, req, res: Adapted, current_files: dict) -> tuple[bool, str]:
    """Run the retained carried tests against the adapted component."""
    scratch = tempfile.mkdtemp(prefix="lego_val_", dir=os.environ.get("TMPDIR"))
    try:
        src = scratch
        single = bool(ws.record.get("single_module"))
        files = {m: _sibling(current_files.get(m), ws.orig_src.get(m, ""))
                 for m in ws.modules}
        files[req.module] = res.code
        for m, code in files.items():
            p = os.path.join(src, m) if single else os.path.join(src, ws.pkg, m)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(code or "")
        env = core.clean_env(PYTHONPATH=src)
        mod = ws.pkg if single else (
            ws.pkg + "." + req.module[:-3].replace("/", ".")).replace(
            ".__init__", "")
        imp = core.sh([core.py(), "-c", f"import {mod}"], 60, env=env, cwd=src)
        if not imp or imp.returncode != 0:
            return False, "adapted module failed to import:\n" + (
                (imp.stderr if imp else "timeout") or "")[-2000:]
        missing = [e for e in req.exports if e not in module_exports(res.code)
                   and not e.startswith("__")]
        if missing:
            return False, f"adapted module lacks required exports {missing}"
        if not res.tests:
            return True, "no retained carried tests; import + exports ok"
        tdir = os.path.join(scratch, "_carried_tests")
        for name, code in res.tests.items():
            p = os.path.join(tdir, os.path.basename(name) if
                             os.path.basename(name).startswith("test")
                             else "test_" + os.path.basename(name))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(code)
        r = core.sh([core.py(), "-m", "pytest", tdir, "-q", "--tb=short",
                     "-p", "no:cacheprovider", "-x"], VALIDATE_TIMEOUT,
                    env=env, cwd=src)
        if r is None:
            return False, "carried tests timed out"
        out = (r.stdout + r.stderr)[-3000:]
        return r.returncode == 0, out
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
