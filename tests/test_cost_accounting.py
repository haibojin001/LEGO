"""Manuscript RQ5 counts successful adaptation responses only."""

from types import SimpleNamespace

import pytest

from lego.analysis import metrics
from lego.library import resident


def test_paper_cost_uses_successful_adaptations_and_fixed_retrieval(monkeypatch):
    monkeypatch.setattr(metrics, "price", lambda alias: (1.0, 2.0))
    rec = {
        "stages": "RADV",
        "usage": {
            "backbone|model": {"in": 1_000_000, "out": 0},
            "diagnosis|model": {"in": 0, "out": 1_000_000},
            "resident|model": {"in": 10_000_000, "out": 0},
        },
        "successful_adaptation_usage": {
            "resident|model": {"in": 2_000_000, "out": 0},
        },
    }
    assert metrics.record_cost(rec) == pytest.approx(5.011)
    task = SimpleNamespace(name="toy")
    costs = metrics.cost({"toy": rec}, [task])
    assert costs["usd_per_task"] == pytest.approx(5.011)
    assert costs["per_role"]["resident"] == pytest.approx(2.0)
    assert costs["per_role"]["retrieval"] == pytest.approx(0.011)


def test_cost_stays_blank_for_legacy_attribution_or_incomplete_split(monkeypatch):
    monkeypatch.setattr(metrics, "price", lambda alias: (1.0, 2.0))
    task = SimpleNamespace(name="toy")
    legacy = {"stages": "RADV", "usage": {
        "resident|model": {"in": 1_000_000, "out": 0}}}
    assert metrics.record_cost(legacy) is None
    assert metrics.cost({"toy": legacy}, [task])["usd_per_task"] is None
    good = {"stages": "---V", "usage": {
        "backbone|model": {"in": 1_000_000, "out": 0}}}
    assert metrics.record_cost(good) == pytest.approx(1.0)
    tasks = [task, SimpleNamespace(name="missing")]
    assert metrics.cost({"toy": good}, tasks)["usd_per_task"] is None


def test_retry_cost_attribution_keeps_only_passing_response(monkeypatch):
    class Model:
        def __init__(self):
            self.usage = {"calls": 0, "in": 0, "out": 0}

        def json(self, prompt, max_tokens):
            self.usage["calls"] += 1
            self.usage["in"] += 100
            self.usage["out"] += 20
            return {
                "mode": "ADAPT", "status": "SUCCESS",
                "implementation": [{"path": "ops.py", "content":
                                    "def add(a, b): return a + b\n"}],
                "interface": {"exports": ["add"], "contract": "add numbers"},
                "dependencies": [], "outgoing_requirements": [],
                "carried_tests": [],
            }

    checks = iter([(False, "test failed"), (True, "test passed")])
    monkeypatch.setattr(resident, "validate",
                        lambda ws, req, res, files: next(checks))
    prim = SimpleNamespace(
        pid="p", impl_text=lambda limit: "def plus(a, b): return a + b",
        exports=["plus"], signatures="def plus(a, b): ...",
        contract="addition", external_deps=[], tests={}, source={},
        summary="addition", context={})
    req = SimpleNamespace(
        module="ops.py", capability="addition", target_context=lambda: "")
    model = Model()
    result = resident.adapt(model, prim, req, None, {}, retries=1)
    assert result.status == "adapted"
    assert result.attempts == 2
    assert model.usage["in"] == 200
    assert result.call_usage == {"calls": 1, "in": 100, "out": 20}
