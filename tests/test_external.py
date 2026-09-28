"""Tests for the external-system adapters (baselines/) and transfer benchmarks
(external/), on the toy fixture of tests/smoke.py.

    LEGO_SCRATCH=/tmp/lego_t python -m pytest tests/test_external.py -x -q

Needs what smoke.py needs: ``git``, ``uv`` (or ``python -m venv``) and network
access for installing pytest into the task environment.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys

import pytest

from tests import smoke

ROOT = smoke.ROOT
TOY = os.path.join(smoke.HERE, "fixtures", "toyrepo")
# Implementation lines of the toy package: none may reach an agent.
IMPL = ["t = add(t, x)", "return total(xs) / len(xs)", "return a + b",
        "return a * b", 'raise ValueError("mean of empty sequence")',
        "xs = list(xs)"]


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("lego_ext"))
    paths = smoke.build(tmp)
    from lego.harness import task as task_mod
    from lego.llm import client
    old = {k: os.environ.get(k) for k in ("LEGO_SCRATCH", "LEGO_RUNS",
                                           "LEGO_MOCK_ANSWERS")}
    os.environ["LEGO_SCRATCH"] = os.path.join(tmp, "scratch")
    os.environ["LEGO_RUNS"] = os.path.join(tmp, "runs")
    os.environ["LEGO_MOCK_ANSWERS"] = paths["answers"]
    saved_bench = task_mod.BENCH_DIR
    task_mod.BENCH_DIR = paths["bench"]
    client.MODELS["mock"] = {"provider": "tests.mock_provider:Provider"}
    paths["tmp"] = tmp
    paths["task"] = task_mod.load_index()["toymath"]
    paths["ids"] = paths["task"].ceiling_ids() | paths["task"].floor_ids()
    yield paths
    task_mod.BENCH_DIR = saved_bench
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _build(toy, mode):
    from baselines import workspace as W
    from lego.run import configure_harness
    configure_harness(f"t_{mode}")
    out = os.path.join(toy["tmp"], f"ws_{mode}")
    return W.build(toy["task"], out, mode=mode, env="copy")


@pytest.fixture(scope="module")
def instruct_ws(toy):
    return _build(toy, "instruct")


@pytest.fixture(scope="module")
def issue_ws(toy):
    return _build(toy, "issue")


def _all_files(root):
    for r, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in fs:
            p = os.path.join(r, f)
            if not os.path.islink(p):
                yield p


def _assert_no_leak(meta, toy, host_too=True):
    tree = meta["tree"]
    for p in _all_files(meta["out"]):
        if p.startswith(os.path.join(meta["out"], "env") + os.sep):
            continue
        text = open(p, errors="ignore").read()
        in_tests = p.startswith(os.path.join(tree, "tests") + os.sep)
        for line in IMPL:
            assert line not in text, f"{line!r} leaked into {p}"
        if not in_tests:
            assert "::" not in text, f"node id syntax in {p}"
            for i in toy["ids"]:
                assert i not in text
    # the git history of the tree is the single stubbed/emptied commit
    log = subprocess.run(["git", "-C", tree, "log", "--oneline"],
                         capture_output=True, text=True, check=True).stdout
    assert len(log.strip().splitlines()) == 1
    objs = subprocess.run(["git", "-C", tree, "grep", "-I", "-F", "-e",
                           "t = add(t, x)", "HEAD"], capture_output=True,
                          text=True).stdout
    assert not objs
    wsj = json.load(open(os.path.join(meta["out"], "workspace.json")))
    blob = json.dumps(wsj)
    for k in ("ceiling_run", "floor_run", "baseline", "ceiling", "floor"):
        assert f'"{k}"' not in blob
    if host_too:            # the sanitized grading clone keeps no original
        for p in _all_files(meta["grading"]["tdir"]):
            if p.endswith("bounds.json"):
                continue
            text = open(p, errors="ignore").read()
            for line in IMPL:
                assert line not in text, f"{line!r} left in grading clone {p}"


# ---------------------------------------------------------------- workspace
def test_workspace_instruct(toy, instruct_ws):
    _ws, meta = instruct_ws
    tree = meta["tree"]
    assert meta["mode"] == "instruct" and meta["pkg"] == "toymath"
    assert sorted(meta["modules"]) == ["__init__.py", "ops.py", "stats.py"]
    ops = open(os.path.join(tree, "toymath", "ops.py")).read()
    assert "def add(a, b):" in ops and "..." in ops and "Return a + b." in ops
    for t in ("test_ops.py", "test_stats.py"):
        assert open(os.path.join(tree, "tests", t)).read() == \
            open(os.path.join(TOY, "tests", t)).read()
    instr = open(meta["instruction"]).read()
    assert meta["instruction"].endswith("INSTRUCTION.md")
    assert "`toymath`" in instr and "python -m pytest tests -q" in instr
    assert "def mean(xs):" in instr and "toymath/stats.py" in instr
    assert os.path.exists(os.path.join(meta["out"], "env", "bin", "python"))
    assert not os.path.exists(os.path.join(tree, "toymath", "__pycache__"))
    _assert_no_leak(meta, toy)


def test_workspace_issue(toy, issue_ws):
    _ws, meta = issue_ws
    tree = meta["tree"]
    for m in ("__init__.py", "ops.py", "stats.py"):
        assert not os.path.exists(os.path.join(tree, "toymath", m))
    assert os.path.exists(os.path.join(tree, "tests", "test_stats.py"))
    issue = open(meta["instruction"]).read()
    assert meta["instruction"].endswith("ISSUE.md")
    assert "missing its implementation" in issue
    assert "def total(xs):" in issue and "python -m pytest tests -q" in issue
    _assert_no_leak(meta, toy)


# ---------------------------------------------------------------- grading
def _deliver(meta, files):
    pkg = os.path.join(meta["tree"], meta["pkg_rel"])
    for m, code in files.items():
        with open(os.path.join(pkg, m), "w") as fh:
            fh.write(code)


def test_grade_original_tree_scores_one(toy, instruct_ws):
    from baselines import grade as G
    from lego.harness.grading import admissible
    ws, meta = instruct_ws
    _deliver(meta, {m: open(os.path.join(TOY, "toymath", m)).read()
                    for m in meta["modules"]})
    runs = os.path.join(toy["tmp"], "runs")
    rec = G.grade(meta, "claude_code", "mock", ws=ws, runs=runs,
                  agent={"exit_code": 0, "wall_seconds": 1.0,
                         "usage": {"tokens_in": 10, "tokens_out": 5,
                                   "cost_usd": 0.01}})
    assert rec["score"] == 1.0 and rec["verdict"] == "ok"
    assert rec["status"] == "done" and admissible(rec)
    assert rec["config"] == "ext_claude_code" and rec["stages"] == "ext"
    assert rec["span"] == 7 and rec["frozen"] is True
    assert rec["usage"] == {"agent|mock": {"calls": 0, "in": 10, "out": 5,
                                           "failed": 0}}
    assert rec["env_changed"] is False
    adir = os.path.join(runs, "arms", "ext_claude_code-mock")
    from lego.run import load_records
    assert load_records(adir)["toymath"]["score"] == 1.0
    with gzip.open(os.path.join(adir, "trees", "toymath.json.gz"), "rt") as fh:
        assert set(json.load(fh)) == set(meta["modules"])
    # the standalone path (workspace.json only, no in-memory Workspace)
    rec2 = G.grade(meta, "claude_code", "mock", runs=runs, write=False)
    assert rec2["score"] == 1.0 and admissible(rec2)


def test_grade_empty_tree_scores_zero(toy, instruct_ws):
    from baselines import grade as G
    ws, meta = instruct_ws
    _deliver(meta, {m: "" for m in meta["modules"]})
    # edits to the tests are not part of the result
    with open(os.path.join(meta["tree"], "tests", "test_ops.py"), "w") as fh:
        fh.write("def test_add():\n    assert True\n")
    rec = G.grade(meta, "openhands", "mock", ws=ws,
                  runs=os.path.join(toy["tmp"], "runs"), write=False)
    assert rec["score"] == 0.0
    assert rec["passed"] <= rec["floor"]


def test_agent_runner_template(toy, issue_ws, tmp_path):
    """A stand-in 'system' that delivers a patch: the runner applies it."""
    from baselines import agents as A
    from baselines import grade as G
    ws, meta = issue_ws
    script = tmp_path / "fake_agent.py"
    script.write_text(
        "import json, os, subprocess, sys\n"
        "repo, log = sys.argv[1], sys.argv[2]\n"
        "src = os.environ['TOY_SRC']\n"
        "os.makedirs(os.path.join(repo, 'toymath'), exist_ok=True)\n"
        "for m in ('__init__.py', 'ops.py', 'stats.py'):\n"
        "    open(os.path.join(repo, 'toymath', m), 'w').write("
        "open(os.path.join(src, m)).read())\n"
        "subprocess.run(['git', '-C', repo, 'add', '-A'], check=True)\n"
        "d = subprocess.run(['git', '-C', repo, 'diff', '--cached'],"
        " capture_output=True, text=True).stdout\n"
        "subprocess.run(['git', '-C', repo, 'reset', '-q', '--hard'], check=True)\n"
        "os.makedirs(log, exist_ok=True)\n"
        "json.dump({'model_patch': d}, open(os.path.join(log, 'preds.json'), 'w'))\n"
        "print(json.dumps({'total_cost_usd': 0.5, 'usage': "
        "{'input_tokens': 100, 'output_tokens': 20}, 'num_turns': 3}))\n")
    cfg = tmp_path / "agents.yaml"
    cfg.write_text(json.dumps({
        "defaults": {"timeout": 120, "runs_on": "sandbox", "cwd": "{repo}"},
        "systems": {"fake": {"version_pin": "0", "mode": "issue",
                             "cmd": [sys.executable, str(script), "{repo}",
                                     "{log_dir}/fake"],
                             "env": {"TOY_SRC": os.path.join(TOY, "toymath")},
                             "delivery": "patch",
                             "patch_glob": ["{log_dir}/fake/*.json"]}}}))
    info = A.run("fake", meta, "mock", config=str(cfg))
    assert info["exit_code"] == 0 and not info["timed_out"]
    assert info["patch"]["applied"], info["patch"]
    assert info["usage"]["cost_usd"] == 0.5
    assert info["usage"]["tokens_in"] == 100 and info["usage"]["calls"] == 3
    rec = G.grade(meta, "swe_agent", "mock", ws=ws, agent=info,
                  runs=os.path.join(toy["tmp"], "runs"), write=False)
    assert rec["score"] == 1.0 and rec["agent_cost_usd"] == 0.5


def test_agent_timeout(toy, instruct_ws, tmp_path):
    from baselines import agents as A
    _ws, meta = instruct_ws
    cfg = tmp_path / "agents.yaml"
    cfg.write_text(json.dumps({"systems": {"slow": {
        "mode": "instruct", "runs_on": "host",
        "cmd": [sys.executable, "-c", "import time; time.sleep(60)"]}}}))
    info = A.run("slow", meta, "mock", timeout=2, config=str(cfg))
    assert info["timed_out"] and info["wall_seconds"] < 40


# ---------------------------------------------------------------- library filter
TARGET_SRC = '''"""Target module of an external benchmark."""


def normalize_rows(rows, width, fill=None):
    """Pad or cut every row to ``width`` cells."""
    out = []
    for row in rows:
        row = list(row)[:width]
        row.extend([fill] * (width - len(row)))
        out.append(row)
    return out


def column_totals(rows):
    """Sum each column, skipping cells that are None."""
    totals = {}
    for row in rows:
        for i, cell in enumerate(row):
            if cell is None:
                continue
            totals[i] = totals.get(i, 0) + cell
    return [totals[i] for i in sorted(totals)]
'''
OTHER_SRC = '''def tokenize(text):
    """Split text on whitespace and punctuation."""
    import re
    return [t for t in re.split(r"[^A-Za-z0-9]+", text) if t]


def bigrams(tokens):
    return list(zip(tokens, tokens[1:]))
'''


def _prim(lib, pid, source, impl):
    from lego.library.primitive import write
    return write(os.path.join(lib, pid), {
        "pid": pid, "name": pid, "kind": "mined", "summary": pid,
        "capabilities": [pid], "source": source,
        "interface": {"exports": [], "signatures": "", "contract": ""},
        "dependencies": {"external": []}, "validated": True},
        impl={"m.py": impl})


@pytest.fixture()
def synthetic(tmp_path):
    lib = str(tmp_path / "lib")
    _prim(lib, "p_url", {"repo": "org/target",
                         "url": "https://github.com/Org/Target.git"}, OTHER_SRC)
    _prim(lib, "p_name", {"repo": "someone/target"}, OTHER_SRC)
    _prim(lib, "p_fork_file", {"url": "git@github.com:mirror/tgt-mirror.git"},
          OTHER_SRC)
    _prim(lib, "p_fork_field", {"url": "https://github.com/x/renamed",
                                "forks": ["https://github.com/org/target"]},
          OTHER_SRC)
    _prim(lib, "p_dup", {"url": "https://github.com/else/lib"},
          TARGET_SRC.replace("Target module", "Vendored module"))
    _prim(lib, "p_keep", {"url": "https://github.com/else/other"}, OTHER_SRC)
    orig = tmp_path / "originals" / "t1" / "target"
    orig.mkdir(parents=True)
    (orig / "tables.py").write_text(TARGET_SRC)
    (orig / "__init__.py").write_text("")
    from external.common import ExternalTask
    task = ExternalTask(name="t1", upstream_repo="https://github.com/org/target",
                        target_package="target",
                        original_sources=str(tmp_path / "originals" / "t1"))
    return {"lib": lib, "task": task, "tmp": tmp_path,
            "pairs": [("https://github.com/org/target",
                       "https://github.com/mirror/tgt-mirror")]}


def test_filter_canonical_forms():
    from external.filter_library import canon, name_of
    for u in ("https://github.com/Org/Repo.git", "git@github.com:org/repo",
              "org/repo", "http://www.github.com/ORG/repo/",
              "https://gitlab.example/org/repo.git"):
        assert canon(u) == "org/repo", u
    assert name_of("https://github.com/org/Repo") == "repo"


def test_filter_rules(synthetic):
    from external import filter_library as F
    from lego.library.codeface import CodeFace
    lib = CodeFace(synthetic["lib"], allow_unvalidated=True)
    kept, excluded, report = F.filter_library(lib, [synthetic["task"]],
                                              synthetic["pairs"], 0.6)
    rule = {x["pid"]: x["rule"] for x in excluded}
    assert rule == {"p_url": "url", "p_name": "name", "p_fork_file": "fork",
                    "p_fork_field": "fork", "p_dup": "near_dup"}
    assert [p.pid for p in kept] == ["p_keep"]
    assert report["removed"] == {"url": 1, "name": 1, "fork": 2, "near_dup": 1}
    # banded candidate generation agrees with all-pairs comparison
    _k2, ex2, _r2 = F.filter_library(lib, [synthetic["task"]],
                                     synthetic["pairs"], 0.6, exact=True)
    assert {x["pid"] for x in ex2} == set(rule)
    # the MinHash rule alone: unrelated code is far below the threshold
    t = [("x", F.minhash.signature(TARGET_SRC))]
    assert F.near_dups({"a": F.minhash.signature(OTHER_SRC)}, t) == {}
    assert "b" in F.near_dups({"b": F.minhash.signature(TARGET_SRC)}, t)
    out = str(synthetic["tmp"] / "filtered")
    F.write_view(lib, kept, excluded, report, out)
    assert [p.pid for p in CodeFace(
        out, allow_unvalidated=True).prims] == ["p_keep"]
    import yaml
    view = yaml.safe_load(open(os.path.join(out, "view.yaml")))["view"]
    assert set(view["exclude_repos"]) == {"org/target", "mirror/tgt-mirror"}
    assert len(open(os.path.join(out, "excluded.jsonl")).readlines()) == 5


def test_plan_to_requirements():
    from external.common import (ExternalTask, ExternalWorkspace, apply_plan,
                                 normalize_plan, requirements)
    obj = {"package": "toyplan", "requirements": [
        {"id": "r1", "target": "toyplan/core.py",
         "interface": '"""Core."""\n\ndef run(x):\n    """Run."""\n    return x\n',
         "dependencies": ["util.py", "numpy"],
         "capability": "run a pipeline step", "retrieval_request": "pipeline"},
        {"id": "r2", "target": "util", "interface": "def helper(): ...",
         "dependencies": []}]}
    p = normalize_plan(obj)
    assert p["package"] == "toyplan"
    assert set(p["modules"]) == {"core.py", "util.py", "__init__.py"}
    assert "return x" not in p["modules"]["core.py"]["interface"]
    ws = ExternalWorkspace(ExternalTask(name="toyplan", spec_text="spec"))
    assert ws.modules == []
    apply_plan(ws, p)
    assert ws.pkg == "toyplan" and "numpy" in ws.record["deps"]
    assert ws.orig_src["core.py"].startswith('"""Core."""\nimport toyplan.util')
    reqs = {r.module: r for r in requirements(ws)}
    assert reqs["core.py"].internal_deps == ["util.py"]
    assert reqs["util.py"].level < reqs["core.py"].level
    assert reqs["core.py"].capability == "run a pipeline step"
    assert reqs["core.py"].exports == ["run"]


# ---------------------------------------------------------------- transfer run
def _toy_benchmark(toy):
    """A one-task transfer benchmark in the generic loader's format: the
    starting tree is the toy repo with interface-stub modules."""
    from lego.harness.interface import stub
    tmp = toy["tmp"]
    rel = os.path.join(tmp, "toybench")
    work = os.path.join(rel, "work", "toymath")
    shutil.copytree(TOY, work)
    for m in ("__init__.py", "ops.py", "stats.py"):
        p = os.path.join(work, "toymath", m)
        open(p, "w").write(stub(open(p).read()))
    shutil.copytree(os.path.join(TOY, "toymath"),
                    os.path.join(rel, "originals", "toymath", "toymath"))
    os.makedirs(os.path.join(rel, "tasks"))
    json.dump({"id": "toymath", "description": "Rebuild the toymath package.",
               "workdir": "../work/toymath", "package": "toymath",
               "tests": "tests", "repo": "https://github.com/example/toymath"},
              open(os.path.join(rel, "tasks", "toymath.json"), "w"))
    spec = {"toybench": {
        "release_path": rel, "loader": "generic", "task_glob": "tasks/*.json",
        "fields": {"name": "id", "spec": "description", "workdir": "workdir",
                   "target_package": "package", "upstream_repo": "repo"},
        "original_sources": "{release_path}/originals/{name}",
        "feedback_cmd": "{python} -m pytest {tests} -q",
        "grader_cmd": "cd {workdir} && {python} -m pytest tests -q "
                      "-p no:cacheprovider",
        "metric": "tests_passed", "metric_parse": {"regex": r"(\d+) passed"}}}
    path = os.path.join(tmp, "benchmarks.yaml")
    import yaml
    yaml.safe_dump(spec, open(path, "w"))
    return path


def test_external_transfer_end_to_end(toy):
    bpath = _toy_benchmark(toy)
    tmp = toy["tmp"]
    runs = os.path.join(tmp, "runs_ext")
    env = dict(os.environ, LEGO_RUNS=runs, LEGO_MODELS=toy["models"],
               LEGO_MOCK_ANSWERS=toy["answers"],
               LEGO_SCRATCH=os.path.join(tmp, "scratch_ext"),
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))

    def py(*args):
        r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
        return r.stdout
    flt = os.path.join(tmp, "codeface_toybench")
    out = py("external.filter_library", "--benchmark", "toybench",
             "--benchmarks", bpath, "--library", toy["lib"], "--out", flt,
             "--workers", "1")
    assert "kept                    1 / 1" in out
    rep = json.load(open(os.path.join(flt, "filter_report.json")))
    assert rep["upstream_repos"] == ["example/toymath"]

    # a library config refuses an unfiltered library
    r = subprocess.run([sys.executable, "-m", "external.run", "--benchmark",
                        "toybench", "--benchmarks", bpath, "--config", "lego",
                        "--model", "mock", "--library", toy["lib"]],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "not a library filtered" in r.stderr

    from lego.run import load_records
    for cfg, extra in (("feedback", []), ("lego", ["--library", flt])):
        py("external.run", "--benchmark", "toybench", "--benchmarks", bpath,
           "--config", cfg, "--model", "mock", "--no-preflight", *extra)
        arms = [d for d in os.listdir(os.path.join(runs, "external",
                                                   "toybench"))
                if d.startswith(cfg + "-")]
        assert len(arms) == 1
        adir = os.path.join(runs, "external", "toybench", arms[0])
        assert os.path.exists(os.path.join(adir, "records.jsonl"))
        rec = load_records(adir)["toymath"]
        assert rec["status"] == "graded", rec.get("traceback") or rec
        assert rec["metric"] == {"name": "tests_passed", "value": 8.0}
        assert rec["interface_source"] == "skeleton"
        assert os.path.exists(os.path.join(adir, "grader", "toymath",
                                           "stdout.log"))
        if cfg == "lego":
            assert rec["funnel"]["cand"] >= 1 and rec["library"] == flt

    out = py("external.coverage", "--benchmark", "toybench", "--benchmarks",
             bpath, "--library", flt, "--source", "originals", "--out",
             os.path.join(tmp, "cov"))
    summ = json.load(open(os.path.join(tmp, "cov",
                                       "toybench_originals.summary.json")))
    assert summ["all"]["modules"] == 3
    assert 0.0 <= summ["all"]["coverage"] <= 1.0
