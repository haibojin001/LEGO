"""Deterministic stand-in for a model endpoint, used only by tests/smoke.py.

Code requests are answered from LEGO_MOCK_ANSWERS (the toy package's sources);
the first request for ``stats.py`` returns a broken module so that the revision
path is exercised. JSON requests are answered by prompt type.
"""

import collections
import json
import os
import re
import threading

_seen = collections.Counter()
_lock = threading.Lock()


def _answer(module: str) -> str:
    p = os.path.join(os.environ["LEGO_MOCK_ANSWERS"], module)
    return open(p).read() if os.path.exists(p) else ""


class Provider:
    def __init__(self, spec):
        self.spec = spec

    def __call__(self, system, prompt, max_tokens, temperature):
        out = self._reply(system, prompt)
        return out, len(prompt) // 4, len(out) // 4

    def _reply(self, system, prompt):
        if "add(a, b)" in prompt and "returns a + b" in prompt:
            return "def add(a, b):\n    return a + b\n"
        if "JSON" in system:
            return json.dumps(self._json(prompt))
        m = re.search(r"`toymath/([\w/]+\.py)`", prompt)
        mod = m.group(1) if m else "__init__.py"
        with _lock:
            _seen[mod] += 1
            n = _seen[mod]
        if mod == "stats.py" and n == 1:
            return "from .ops import add\n\ndef total(xs):\n    return 0\n"
        return _answer(mod)

    def _json(self, prompt):
        if "Generate pytest unit and integration tests" in prompt:
            return {"tests": [{"content":
                    "def test_plan_import():\n"
                    "    import toymath\n"
                    "    assert hasattr(toymath, 'add')\n"}]}
        if "MODE: ASSESS" in prompt:
            return {"mode": "ASSESS", "assessment": "SUITABLE",
                    "usage_proposal": "use as the addition helper"}
        if "MODE: ADAPT" in prompt:
            tgt = (re.search(r"target module: ([\w/]+\.py)", prompt) or
                   re.search(r'"target_module": "([\w/]+\.py)"', prompt))
            mod = tgt.group(1) if tgt else "ops.py"
            test = ("from toymath.%s import *\n\ndef test_import():\n"
                    "    assert True\n" % mod[:-3].replace("/", "."))
            return {"mode": "ADAPT", "status": "SUCCESS",
                    "implementation": [{"path": mod, "content": _answer(mod)}],
                    "interface": {"exports": [], "contract": "adapted"},
                    "dependencies": [],
                    "outgoing_requirements": [
                        {"target": "stats", "requirement": "add(a, b) returns a+b"}],
                    "carried_tests": [{"test": "test_adder.py",
                                       "action": "rewritten",
                                       "justification": "new import path",
                                       "content": test}]}
        if "# Candidate Assessments" in prompt:
            ids = re.findall(r"^\[(r\d+)\] target \S+: .*\n  candidate (\S+):",
                             prompt, re.M)
            return {"activated_primitives": [
                {"requirement": r, "primitive": p, "reason": "match",
                 "adaptation_requirement": "provide the target interface"}
                for r, p in ids], "task_specific_code": []}
        if "# Target Interface (this batch)" in prompt:
            ids = re.findall(r"^\[(r\d+)\] ([\w/]+\.py)", prompt, re.M)
            return {"requirements": [
                {"id": r, "target": m, "capability": f"arithmetic helpers {m}",
                 "retrieval_request": "add numbers"} for r, m in ids]}
        if "# Current Plan" in prompt:
            if "native_failed=0" in prompt and "synthesized_failed=0" in prompt:
                return {"decision": "PASS",
                        "criteria": {k: "pass" for k in (
                            "task_completion", "build_success", "test_pass_rate",
                            "interface_consistency", "code_quality")},
                        "implicated_components": [], "revision_guidance": [],
                        "observed_failures": []}
            return {"decision": "FAIL",
                    "criteria": {"task_completion": "fail",
                                 "build_success": "pass",
                                 "test_pass_rate": "fail",
                                 "interface_consistency": "pass",
                                 "code_quality": "fail"},
                    "implicated_components": ["stats.py"],
                    "revision_guidance": [{"component": "stats.py",
                                           "guidance": "implement total/mean"}],
                    "observed_failures": [{"type": "implementation",
                                           "evidence": "test_total"}]}
        return {}
