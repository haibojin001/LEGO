"""Recovery and admission checks for a newly built CodeFace."""

import json
import os
import shutil
import sys
from types import SimpleNamespace

from lego.harness import core
from lego.library.audit_snapshot import audit
from lego.library.primitive import load
from lego.mining import (disjoint_plan, harvest, merge_shards, segment,
                         source_plan, synthesize, web)
from lego.mining.web_plan import plan as plan_web_requests
from lego.mining.graph import build_graph


def test_source_plan_resolves_real_donors_and_reports_unknowns(
        tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"file": "one.py", "kind": "verified",
         "from_repo": "(verified:repo_toy)"},
        {"file": "two.py", "kind": "mined",
         "from_repo": "Other__lib+Third__pkg+4more"},
        {"file": "three.py", "from_repo": "Web/lib"},
        {"file": "four.py", "from_repo": "(verified:missing)"},
    ]))
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(json.dumps({
        "name": "toy", "clone": "https://github.com/example/toy.git"}) + "\n")
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"pins": {
        "https://github.com/example/toy.git": {"sha": "abc123"}}}))
    out = tmp_path / "corpus.txt"
    report = source_plan.plan(str(manifest), str(metadata), str(pins),
                              str(out))
    assert report["source_repositories"] == 4
    assert report["pinned_repositories"] == 1
    assert report["unresolved_reasons"] == {"unknown_verified_task": 1}
    assert out.read_text().splitlines() == [
        "https://github.com/Other/lib.git",
        "https://github.com/Third/pkg.git",
        "https://github.com/Web/lib.git",
        "https://github.com/example/toy.git abc123"]
    monkeypatch.setattr(source_plan, "_head", lambda _url: "f" * 40)
    pinned = source_plan.plan(str(manifest), str(metadata), str(pins),
                              str(out), pin_heads=True)
    assert pinned["pinned_repositories"] == pinned["source_repositories"] == 4


def test_web_requests_use_only_uncovered_capabilities_and_trace_demand():
    rows = [
        {"task": "a", "module": "reader.py", "covered": False,
         "retrieval_request": "parse records |", "exports": ["read_records"]},
        {"task": "b", "module": "reader.py", "covered": False,
         "retrieval_request": "parse records # note",
         "exports": ["parse_records"]},
        {"task": "c", "module": "writer.py", "covered": True,
         "retrieval_request": "write records", "exports": ["write_records"]},
    ]
    requests = plan_web_requests(rows)
    assert len(requests) == 1
    assert requests[0]["query"] == "parse records"
    assert requests[0]["hints"] == ["parse_records", "read_records"]
    assert requests[0]["modules"] == [
        {"task": "a", "module": "reader.py"},
        {"task": "b", "module": "reader.py"}]


def test_disjoint_corpus_excludes_targets_organizations_and_forks(tmp_path):
    sha = "a" * 40
    corpus = tmp_path / "sources.txt"
    urls = [
        "https://github.com/Org/target.git",
        "https://github.com/Org/other.git",
        "https://github.com/Else/target.git",
        "https://github.com/Fork/renamed.git",
        "https://github.com/Else/clean.git",
    ]
    corpus.write_text("".join(f"{url} {sha}\n" for url in urls))
    target = SimpleNamespace(
        clone=urls[0], extra={"forks": [urls[3]]})
    report = disjoint_plan.plan(
        str(corpus), [target], str(tmp_path / "disjoint.txt"))
    assert report["kept"] == 1
    assert report["excluded"] == {
        "target_name": 1, "target_or_recorded_fork": 2,
        "target_organization": 1}
    assert (tmp_path / "disjoint.txt").read_text() == f"{urls[4]} {sha}\n"


def test_web_rate_limit_exits_for_resumable_cluster_job(tmp_path, monkeypatch):
    requests = tmp_path / "requests.txt"
    requests.write_text("parse records | read_records\n")

    class Env:
        def __init__(self, python=None):
            pass

        def close(self):
            pass

    def limited(*args):
        raise web.RateLimited("test quota")

    monkeypatch.setattr(web.synthesize, "Env", Env)
    monkeypatch.setattr(web, "source_request", limited)
    assert web.main(["--requests", str(requests), "--out", str(tmp_path / "out"),
                     "--python", sys.executable]) == 75


def test_mined_and_harvested_components_carry_passing_tests(
        tmp_path, monkeypatch):
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "toyrepo")
    original_env = core.clean_env

    def clean_env(**overrides):
        overrides.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        return original_env(**overrides)

    monkeypatch.setattr(core, "clean_env", clean_env)
    env = synthesize.Env(python=sys.executable)
    graph = build_graph(fixture)
    components, _ = segment.segment(graph, 6, 1500)
    comp = next(c for c in components if c.seed == "ops.py")
    out = tmp_path / "mined"
    row = synthesize.synthesize(
        graph, comp, str(out), "toy_ops",
        {"repo": "example/toy", "url": "https://github.com/example/toy.git"},
        env, synthesize.Options(kind="mined"))
    assert row["status"] == "admitted", row
    primitive = load(str(out / "toy_ops"))
    assert primitive.meta["validation"]["mode"] == "tests"
    assert primitive.meta["validation"]["passed"] > 0
    assert primitive.tests
    shards = tmp_path / "shards" / "shard_0"
    shards.mkdir(parents=True)
    shutil.copytree(out / "toy_ops", shards / "toy_ops")
    (shards / "_mining_log.jsonl").write_text(
        '{"event":"repo","status":"done"}\n')
    merged = tmp_path / "merged"
    assert merge_shards.merge(str(tmp_path / "shards"),
                             str(merged))["total"] == 1
    assert load(str(merged / "toy_ops")).tests
    assert "missing_passing_isolation_validation" not in audit(
        str(merged), paper=True)["problems"]

    generated = {
        "__init__.py": (os.path.join(fixture, "toymath", "__init__.py")),
        "ops.py": (os.path.join(fixture, "toymath", "ops.py")),
        "stats.py": (os.path.join(fixture, "toymath", "stats.py")),
    }
    generated = {name: open(path).read() for name, path in generated.items()}
    generated["stats.py"] += "\n\ndef count(xs):\n    return len(list(xs))\n"
    rec = {"name": "toy", "score": 1.0, "status": "done",
           "trajectory": []}
    info = {"package": "toymath", "clone": "https://github.com/example/toy.git",
            "commit": "abc123",
            "test_source": os.path.join(fixture, "tests")}
    rows = harvest.harvest_task(
        "separate-pass", rec, generated, info, str(tmp_path / "harvested"),
        env, synthesize.Options(kind="harvested", mode="tests"))
    admitted = [r for r in rows if r["status"] == "admitted"]
    assert admitted, rows
    assert all(load(str(tmp_path / "harvested" / r["pid"])).tests
               for r in admitted)
