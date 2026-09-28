"""End-to-end smoke test: every construction mode on a toy task, mock models.

    python tests/smoke.py [--configs feedback lego ...] [--keep]

Builds, in a temporary directory, a toy git repository, a one-task benchmark
with a frozen grading key, a one-primitive CodeFace and a File RAG index; then
runs each configuration through ``lego.run`` with the mock provider and checks
that a scored, admissible record was written. Needs ``git``, ``uv`` (or a
working ``python -m venv``) and network access for ``pip install pytest``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DEFAULT = ["single_shot", "feedback", "feedback_diag", "retrieval_feedback",
           "lego", "structured_adapt", "rag_edit", "file_rag_edit",
           "context_only", "import_call"]


def build(tmp: str) -> dict:
    repo = os.path.join(tmp, "toyrepo")
    shutil.copytree(os.path.join(HERE, "fixtures", "toyrepo"), repo)
    g = ["git", "-C", repo]
    subprocess.run(["git", "init", "-q", repo], check=True)
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["-c", "user.email=t@t", "-c", "user.name=t", "commit",
                        "-qm", "init"], check=True)
    subprocess.run(g + ["config", "uploadpack.allowAnySHA1InWant", "true"],
                   check=True)
    sha = subprocess.run(g + ["rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    bench = os.path.join(tmp, "benchmark")
    tdir = os.path.join(bench, "tasks", "toymath")
    os.makedirs(os.path.join(tdir, "expected"))
    ceiling = ["tests/test_ops.py::test_add", "tests/test_ops.py::test_add_neg",
               "tests/test_ops.py::test_mul", "tests/test_stats.py::test_total",
               "tests/test_stats.py::test_total_empty",
               "tests/test_stats.py::test_mean",
               "tests/test_stats.py::test_mean_empty",
               "tests/test_stats.py::test_version_free"]
    open(os.path.join(tdir, "expected", "ceiling.txt"), "w").write(
        "\n".join(ceiling) + "\n")
    open(os.path.join(tdir, "expected", "floor.txt"), "w").write(
        "tests/test_stats.py::test_version_free\n")
    open(os.path.join(tdir, "task.md"), "w").write(
        "# Task: rebuild `toymath`\nModules: toymath, toymath.ops, toymath.stats\n")
    os.makedirs(os.path.join(bench, "splits"))
    open(os.path.join(bench, "splits", "smoke.txt"), "w").write("toymath\n")
    with open(os.path.join(bench, "tasks.jsonl"), "w") as fh:
        fh.write(json.dumps({"name": "toymath", "clone": "file://" + repo,
                             "commit": sha, "package": "toymath",
                             "src_prefix": ".", "tests": "tests",
                             "domain": "scientific-computing", "track": "toy",
                             "n_modules": 3, "ceiling": 8, "floor": 1,
                             "band": 1}) + "\n")
    from lego.library.primitive import write
    lib = os.path.join(tmp, "codeface")
    write(os.path.join(lib, "adder"), {
        "pid": "adder", "name": "adder", "kind": "mined",
        "summary": "add numbers and sum sequences",
        "capabilities": ["add numbers", "arithmetic helpers"],
        "source": {"repo": "someorg/adderlib",
                   "url": "https://github.com/someorg/adderlib.git"},
        "interface": {"exports": ["plus"], "signatures": "def plus(a, b): ...",
                      "contract": "plus(a, b) -> a + b"},
        "dependencies": {"external": []}, "validated": True,
        "validation": {"mode": "tests", "passed": 1}},
        impl={"adder.py": "def plus(a, b):\n    return a + b\n"},
        tests={"test_adder.py": "from adder import plus\n\n"
                                "def test_plus():\n    assert plus(1, 2) == 3\n"})
    frag = os.path.join(tmp, "file_rag")
    os.makedirs(frag)
    with open(os.path.join(frag, "index.jsonl"), "w") as fh:
        fh.write(json.dumps({"repo": "someorg/adderlib",
                             "url": "https://github.com/someorg/adderlib.git",
                             "path": "adder.py",
                             "text": "def plus(a, b):\n    \"\"\"Add numbers.\"\"\"\n    return a + b\n"}) + "\n")
    models = os.path.join(tmp, "models.yaml")
    open(models, "w").write("mock:\n  provider: tests.mock_provider:Provider\n")
    return {"bench": bench, "lib": lib, "frag": frag, "models": models,
            "answers": os.path.join(repo, "toymath")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="lego_smoke_")
    try:
        paths = build(tmp)
        env = dict(os.environ, LEGO_BENCH=paths["bench"],
                   LEGO_RUNS=os.path.join(tmp, "runs"),
                   LEGO_MODELS=paths["models"], LEGO_MOCK_ANSWERS=paths["answers"],
                   LEGO_SCRATCH=os.path.join(tmp, "scratch"),
                   PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        failed = []
        for cfg in a.configs:
            view = paths["frag"] if cfg == "file_rag_edit" else paths["lib"]
            cmd = [sys.executable, "-m", "lego.run", "--config", cfg,
                   "--split", "smoke", "--backbone", "mock",
                   "--set", f"view.path={json.dumps(view)}",
                   "view.exclude=[]"]
            print("$", " ".join(cmd), flush=True)
            r = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True,
                               text=True)
            out = r.stdout + r.stderr
            os.makedirs(os.path.join(tmp, "logs"), exist_ok=True)
            with open(os.path.join(tmp, "logs", cfg + ".log"), "w") as fh:
                fh.write(out)
            recs = []
            for root, _d, fs in os.walk(os.path.join(tmp, "runs", "arms")):
                for f in fs:
                    if f.startswith("records") and cfg + "-" in root:
                        recs += [json.loads(l) for l in open(os.path.join(root, f))]
            rec = recs[-1] if recs else {}
            # single-pass configs are expected to fail on this task (the mock's
            # first stats.py is broken); every other config must reach done
            ok = bool(rec) and rec.get("span") == 7 and (
                rec.get("status") == "done" or cfg in ("single_shot",
                                                       "retrieval_one"))
            if cfg in ("feedback", "lego") and float(rec.get("score") or 0) <= 0:
                ok = False
            if cfg != "feedback" and cfg not in ("single_shot", "feedback_diag") \
                    and not (rec.get("funnel") or {}).get("cand"):
                ok = False              # library configs must retrieve something
            print(f"  {cfg:20s} status={rec.get('status')} "
                  f"score={rec.get('score')} rounds={len(rec.get('trajectory', []))}"
                  f" funnel={ {k: v for k, v in (rec.get('funnel') or {}).items() if v} }",
                  flush=True)
            if not ok:
                failed.append(cfg)
                print(out[-4000:])
                if rec.get("traceback"):
                    print(rec["traceback"])
        print("FAILED: " + " ".join(failed) if failed else "ALL OK")
        return 1 if failed else 0
    finally:
        if a.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
