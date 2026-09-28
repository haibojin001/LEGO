"""Checks for the execution and trajectory rules used by the paper arms."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from lego import config
from lego.analysis.coverage import OfflineWorkspace
from lego.analysis.report import prefix_records
from lego.harness import core, grading
from lego.harness import task as task_module
from lego.harness.interface import stub
from lego.harness.prepare import ExecResult, Workspace
from lego.library import resident
from lego.library.codeface import CodeFace
from lego.library.primitive import write as write_primitive
from lego.llm import client
from lego.pipeline import construct
from lego.pipeline.plan_tests import synthesize as synthesize_plan_tests
from lego.pipeline.decompose import Requirement
from tests import mock_provider, smoke


def test_synthesized_tests_execute_and_fail(tmp_path, monkeypatch):
    zero = tmp_path / "zero"
    pkg = zero / "toyplan"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("def plus(a, b): return a + b\n")
    ws = Workspace(None, str(tmp_path), str(zero), str(zero), ".", "toyplan",
                   "tests", str(pkg), ["__init__.py"], {}, {})
    monkeypatch.setattr(core, "VENV", sys.prefix)
    name = "test_lego_plan.py"
    passed = ws._execute_plan_tests(
        str(zero), str(zero),
        {name: "from toyplan import plus\n"
               "def test_plus(): assert plus(1, 2) == 3\n"})
    failed = ws._execute_plan_tests(
        str(zero), str(zero),
        {name: "from toyplan import plus\n"
               "def test_plus(): assert plus(1, 2) == 4\n"})
    assert passed[:2] == (1, 0)
    assert failed[1] == 1


def test_plan_tests_retry_malformed_model_response():
    class Model:
        calls = 0

        def json(self, prompt, max_tokens):
            self.calls += 1
            if self.calls == 1:
                return {"tests": [{"content": "not valid Python !"}]}
            return {"tests": [{"content":
                    "def test_import():\n    import toyplan\n"}]}

    model = Model()
    req = Requirement("r1", "__init__.py", "def plus(a, b): ...",
                      ["plus"], capability="addition")
    tests = synthesize_plan_tests(model, "toyplan", "Make toyplan", [req])
    assert model.calls == 2
    assert list(tests) == ["test_lego_plan_0_0.py"]


def test_offline_coverage_uses_target_modules_only(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    (root / "ops.py").write_text("def plus(a, b): return a + b\n")
    (root / "test_ops.py").write_text("def test_plus(): assert True\n")
    (root / "conftest.py").write_text("import pytest\n")
    task = SimpleNamespace(dir=str(tmp_path), package="toyplan",
                           name="toyplan")
    ws = OfflineWorkspace(task)
    assert ws.modules == ["ops.py"]


def test_model_task_statement_excludes_grading_and_answer_sections(
        tmp_path, monkeypatch):
    monkeypatch.setattr(task_module, "BENCH_DIR", str(tmp_path))
    card = tmp_path / "tasks" / "toy" / "task.md"
    card.parent.mkdir(parents=True)
    card.write_text(
        "# Build toy\nReconstruct the package.\n"
        "## What to build\nCreate `toy/`.\n"
        "## API surface the tests require\n- `toy.add`\n"
        "## How you are graded\nscore = clamp((passed - 2) / 8)\n"
        "## Rules\n`original/` is the answer.\n"
        "## Provenance\nsecret grading details\n")
    statement = task_module.Task(
        name="toy", clone="", commit="", package="toy").statement()
    assert "Create `toy/`" in statement
    assert "toy.add" in statement
    assert "clamp" not in statement
    assert "original/" not in statement
    assert "Provenance" not in statement


def test_repository_build_failure_is_execution_evidence(tmp_path):
    ws = Workspace(None, str(tmp_path), str(tmp_path), str(tmp_path), ".",
                   "toyplan", "tests", str(tmp_path), ["ops.py"], {}, {})
    result = ws.execute({"ops.py": "def broken(\n"})
    assert result.build_ok is False
    assert result.import_ok is False
    assert result.verdict == "build-failed"
    assert "ops.py" in result.feedback


def test_adaptation_cannot_discard_all_carried_tests():
    prim = SimpleNamespace(
        pid="primitive", tests={"test_behavior.py":
                                "def test_behavior(): assert True\n"})
    req = SimpleNamespace(module="ops.py")
    answer = {
        "mode": "ADAPT", "status": "SUCCESS",
        "implementation": [{"path": "ops.py", "content":
                            "def add(a, b): return a + b\n"}],
        "interface": {"exports": ["add"], "contract": "add numbers"},
        "dependencies": [], "outgoing_requirements": [],
        "carried_tests": [{"test": "test_behavior.py", "action": "dropped",
                           "justification": "donor specific"}]}
    assert resident._parse_adapt(answer, prim, req, True).status == "failed"
    answer["carried_tests"] = []
    assert resident._parse_adapt(answer, prim, req, True).status == "failed"
    answer["carried_tests"] = [
        {"test": "test_behavior.py", "action": "kept"}]
    assert resident._parse_adapt(answer, prim, req, True).status == "adapted"


def test_validated_component_is_delivered_without_regeneration(monkeypatch):
    cfg = config.get("lego", {"models": {"backbone": "mock",
                                          "resident": "mock",
                                          "diagnosis": "mock"}})
    ws = SimpleNamespace(
        task=SimpleNamespace(name="toy", statement=lambda: "Build toy"),
        pkg="toy")
    runner = construct.Construction(cfg, ws)
    component = "def add(a, b):\n    return a + b\n"
    runner.components["ops.py"] = component
    calls = []

    def generate(prompt, max_tokens):
        calls.append(prompt)
        return "def total(xs):\n    return sum(xs)\n"

    monkeypatch.setattr(runner.backbone, "code", generate)
    reqs = [Requirement("r1", "ops.py", "def add(a, b): ...", ["add"]),
            Requirement("r2", "stats.py", "def total(xs): ...", ["total"])]
    files = runner.integrate(reqs)
    assert files["ops.py"] == component
    assert files["stats.py"].startswith("def total")
    assert len(calls) == 1 and "stats.py" in calls[0]


def test_carried_test_cannot_hide_missing_target_export(monkeypatch):
    ws = SimpleNamespace(
        record={"single_module": False}, modules=["ops.py"],
        orig_src={"ops.py": ""}, pkg="toy")
    req = SimpleNamespace(module="ops.py", exports=["add"])
    adapted = resident.Adapted(
        pid="p", module="ops.py", status="adapted",
        code="def other(): return 1\n",
        tests={"test_other.py": "def test_other(): assert True\n"})
    monkeypatch.setattr(core, "sh",
                        lambda *args, **kwargs: SimpleNamespace(
                            returncode=0, stdout="", stderr=""))
    ok, log = resident.validate(ws, req, adapted, {})
    assert not ok
    assert "required exports" in log


def test_context_field_and_message_targets_follow_activated_primitives(monkeypatch):
    cfg = config.get("rag_edit_VIX", {
        "models": {"backbone": "mock", "resident": "mock",
                   "diagnosis": "mock"}})
    ws = SimpleNamespace(
        task=SimpleNamespace(name="toy", statement=lambda: "Build toy"),
        pkg="toy")
    runner = construct.Construction(cfg, ws)
    prompts = []
    monkeypatch.setattr(
        runner.backbone, "code",
        lambda prompt, max_tokens: prompts.append(prompt) or "pass\n")
    prim = SimpleNamespace(
        impl_text=lambda limit: "def add(a, b): return a + b",
        exports=["add"], signatures="def add(a, b): ...",
        contract="addition", source={"repo": "donor/lib",
                                     "paths": ["lib/ops.py"]},
        summary="Reusable addition", context={"README.md": "usage notes"},
        external_deps=[], tests={"test_add.py": "def test_add(): pass"},
        tests_text=lambda limit: "def test_add(): pass")
    req = Requirement("r1", "ops.py", "def add(a, b): ...", ["add"],
                      consumers=["stats.py"])
    assert "context" in cfg.edit_fields
    runner._edit(prim, req, "")
    assert "COMPONENT CONTEXT AND PROVENANCE" in prompts[0]
    assert "usage notes" in prompts[0]
    runner.active["ops.py"] = {"prim": prim}
    assert runner._resolve_target("stats", req) is None
    runner.active["stats.py"] = {"prim": prim}
    assert runner._resolve_target("stats", req) == "stats.py"


def test_activation_fallback_respects_dependency_constraints(monkeypatch):
    cfg = config.get("lego", {
        "models": {"backbone": "mock", "resident": "mock",
                   "diagnosis": "mock"}})
    ws = SimpleNamespace(
        task=SimpleNamespace(name="toy", statement=lambda: "Build toy"),
        pkg="toy")
    runner = construct.Construction(cfg, ws)
    monkeypatch.setattr(runner.backbone, "json",
                        lambda prompt, max_tokens: {})
    prim = SimpleNamespace(
        pid="p", external_deps=["not_allowed"], card=lambda: "candidate")
    req = Requirement("r1", "ops.py", "def add(a, b): ...", ["add"],
                      allowed_deps=[], capability="addition")
    runner._activate([req], {"ops.py": [(prim, None)]})
    assert runner.active == {}


def test_primary_library_requires_recorded_passing_isolation(tmp_path):
    common = {
        "kind": "mined", "source": {"repo": "example/lib"},
        "interface": {"exports": ["plus"]},
        "dependencies": {"external": []}, "validated": True}
    for pid, validation in [
            ("missing", None),
            ("import_only", {"mode": "import", "passed": 0}),
            ("tested", {"mode": "tests", "passed": 1})]:
        meta = dict(common, pid=pid)
        if validation is not None:
            meta["validation"] = validation
        write_primitive(
            str(tmp_path / pid), meta,
            impl={"lib.py": "def plus(a, b): return a + b\n"},
            tests={"test_lib.py": "def test_plus(): assert True\n"})
    assert [p.pid for p in CodeFace(str(tmp_path)).prims] == ["tested"]


def test_diagnosis_revises_only_implicated_module(monkeypatch):
    src = os.path.join(os.path.dirname(__file__), "fixtures", "toyrepo",
                       "toymath")
    monkeypatch.setenv("LEGO_MOCK_ANSWERS", src)
    monkeypatch.setitem(client.MODELS, "mock",
                        {"provider": "tests.mock_provider:Provider"})
    mock_provider._seen.clear()
    reqs = []
    for i, module in enumerate(("ops.py", "stats.py")):
        source = open(os.path.join(src, module)).read()
        reqs.append(Requirement(
            rid=f"r{i + 1}", module=module, interface=stub(source),
            exports=["add", "mul"] if module == "ops.py" else ["total", "mean"],
            capability="arithmetic", retrieval_request="add numbers"))
    monkeypatch.setattr(construct, "decompose", lambda *a, **kw: reqs)

    class FakeWorkspace:
        pkg = "toymath"
        task = SimpleNamespace(name="toymath", statement=lambda: "build toymath")

        def execute(self, files, plan_tests=None):
            assert plan_tests
            broken = "return 0" in files["stats.py"]
            return ExecResult(
                import_ok=True, passed=1 if broken else 8,
                failed=1 if broken else 0,
                feedback="test_total failed" if broken else "8 passed",
                verdict="ok", plan_passed=1, plan_failed=0)

    cfg = config.get("feedback_diag", {
        "models": {"backbone": "mock", "diagnosis": "mock"}})
    out = construct.Construction(cfg, FakeWorkspace()).run(
        lambda r: {"score": r.passed / 8, "passed": r.passed})
    assert len(out["trajectory"]) == 2
    assert out["module_attempts"] == {"ops.py": 1, "stats.py": 2}
    assert [r["diagnosis"]["decision"] for r in out["trajectory"]] == [
        "FAIL", "PASS"]
    assert [step["round"] for step in out["context"]] == [1, 2]


def test_one_attempt_row_uses_first_round():
    record = {
        "name": "toy", "arm": "feedback-x", "harness_rev": core.HARNESS_REV,
        "status": "done", "score": 1.0, "span": 7,
        "usage": {"backbone|mock": {"in": 100, "out": 100}},
        "trajectory": [
            {"round": 1, "score": 0.2, "passed": 2, "verdict": "ok"},
            {"round": 2, "score": 1.0, "passed": 8, "verdict": "ok"}]}
    first = prefix_records({"toy": record}, 1)["toy"]
    assert first["best_round"] == 1
    assert first["score"] == 0.2
    assert first["status"] == "done"
    assert "usage" not in first


def test_pytest_node_ids_match_frozen_score_key(tmp_path):
    zero = tmp_path / "work" / "zero"
    test_file = zero / "tests" / "test_ops.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_add(): assert True\n")
    relative_to_cwd = os.path.relpath(test_file, os.getcwd())
    variants = [str(test_file), relative_to_cwd,
                "tests/test_ops.py"]
    for path in variants:
        actual = core.canonical_test_outcomes(
            {path + "::test_add": "passed"}, str(zero))
        assert actual == {"tests/test_ops.py::test_add": "passed"}
        task = SimpleNamespace(
            ceiling_ids=lambda: {"tests/test_ops.py::test_add"},
            floor_ids=lambda: set())
        result = ExecResult(True, 1, 0, "", "ok", tests=actual)
        assert grading.make_score_fn(task)(result)["score"] == 1.0


def test_full_feedback_loop_with_native_and_plan_tests(tmp_path, monkeypatch):
    smoke.build(str(tmp_path))
    repo = tmp_path / "toyrepo"
    pkg = repo / "toymath"
    modules = ["__init__.py", "ops.py", "stats.py"]
    task = SimpleNamespace(
        name="toymath", clone="file://" + str(repo),
        statement=lambda: "Build toymath from public interfaces.",
        ceiling_ids=lambda: set((tmp_path / "benchmark" / "tasks" /
                                 "toymath" / "expected" / "ceiling.txt")
                                .read_text().splitlines()),
        floor_ids=lambda: set((tmp_path / "benchmark" / "tasks" /
                               "toymath" / "expected" / "floor.txt")
                              .read_text().splitlines()))
    ws = Workspace(
        task, str(tmp_path / "work"), str(repo), str(repo), ".",
        "toymath", "tests", str(pkg), modules,
        {m: (pkg / m).read_text() for m in modules},
        {"deps": [], "single_module": False})
    os.makedirs(ws.tdir)
    monkeypatch.setenv("LEGO_MOCK_ANSWERS", str(pkg))
    monkeypatch.setitem(client.MODELS, "mock",
                        {"provider": "tests.mock_provider:Provider"})
    monkeypatch.setattr(core, "VENV", sys.prefix)
    monkeypatch.setattr(core, "EXTRA_PYTHONPATH", [])
    original_env = core.clean_env

    def clean_env(**overrides):
        overrides.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        return original_env(**overrides)

    monkeypatch.setattr(core, "clean_env", clean_env)
    mock_provider._seen.clear()
    cfg = config.get("feedback", {"models": {"backbone": "mock"}})
    out = construct.Construction(cfg, ws).run(grading.make_score_fn(task, ws))
    assert out["best"]["score"] == 1.0
    assert out["trajectory"][-1]["plan_passed"] == 1
    assert out["trajectory"][-1]["plan_failed"] == 0
