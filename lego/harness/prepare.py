"""Per-task preparation and execution.

``prepare(task)`` clones the pinned commit, builds the task's environment,
removes the installed target, and re-measures the identity ceiling and the
empty-package floor through the same path generated code will take. The frozen
grading key (``expected/*.txt``) is authoritative for scoring; the re-measured
values are recorded next to it so environment drift is visible per record.

``Workspace.execute(files, plan_tests)`` writes a candidate tree and runs the
native suite plus tests synthesized from the current plan. Nothing here calls a
model.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from lego.harness import core
from lego.harness.task import Task


class NotGradeable(Exception):
    """The task cannot be measured in this environment (status in args[0])."""


@dataclass
class ExecResult:
    import_ok: bool
    passed: int                     # summary-line count
    failed: int
    feedback: str                   # tail of the pytest output / import error
    verdict: str                    # provenance verdict: ok | leaked | ...
    exercised: int = 0
    tests: dict = field(default_factory=dict)   # node id -> outcome
    plan_passed: int = 0
    plan_failed: int = 0
    plan_feedback: str = ""
    pytest_exit: int | None = None
    build_ok: bool = True

    def passed_ids(self) -> set[str]:
        return {k for k, v in self.tests.items() if v == "passed"}


@dataclass
class Workspace:
    task: Task
    tdir: str
    repo_dir: str
    proj_dir: str
    src_prefix: str
    pkg: str
    test_path: str
    pkg_dir: str
    modules: list[str]
    orig_src: dict[str, str]
    record: dict

    def execute(self, files: dict[str, str],
                plan_tests: dict[str, str] | None = None) -> ExecResult:
        for name, source in files.items():
            if not name.endswith(".py"):
                continue
            try:
                compile(source, name, "exec")
            except (SyntaxError, ValueError) as exc:
                return ExecResult(
                    import_ok=False, passed=0, failed=1,
                    feedback=f"build failed in {name}: {exc}",
                    verdict="build-failed", build_ok=False)
        zero, zsrc, gen = core._write_zero(self.tdir, self.src_prefix, self.pkg,
                                           files, self.repo_dir, self.test_path)
        ok, npass, nfail, fb, prov = core._test_zero(
            zero, zsrc, self.pkg, self.test_path, self.repo_dir, gen)
        out = ExecResult(import_ok=ok, passed=npass, failed=nfail,
                         feedback=fb or "", verdict=prov.get("verdict", "?"),
                         exercised=prov.get("exercised", 0),
                         tests=prov.get("tests") or {},
                         pytest_exit=prov.get("pytest_exit"))
        if plan_tests and ok and out.verdict != "timeout":
            out.plan_passed, out.plan_failed, out.plan_feedback = (
                self._execute_plan_tests(zero, zsrc, plan_tests))
            if out.plan_feedback:
                out.feedback += "\n\nSYNTHESIZED PLAN TESTS:\n" + out.plan_feedback
        return out

    def _execute_plan_tests(self, zero: str, zsrc: str,
                            plan_tests: dict[str, str]) -> tuple[int, int, str]:
        tdir = os.path.join(self.tdir, "plan_tests")
        core.rm_rf(tdir)
        os.makedirs(tdir)
        for name, code in plan_tests.items():
            p = os.path.join(tdir, os.path.basename(name))
            with open(p, "w") as fh:
                fh.write(code)
        env = core.clean_env(
            PYTHONPATH=os.pathsep.join([zsrc, *core.EXTRA_PYTHONPATH]),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        run = core.sh([core.py(), "-m", "pytest", tdir, "-q", "--tb=line",
                       "-p", "no:cacheprovider", "-o", "addopts=",
                       "--rootdir", tdir], core.TEST_TIMEOUT, env=env,
                      cwd=zero)
        if run is None:
            return 0, 1, "pytest timeout"
        text = (run.stdout + run.stderr).strip()
        passed = re.search(r"(\d+) passed", text)
        failed = re.search(r"(\d+) (?:failed|error)", text)
        nfailed = int(failed.group(1)) if failed else 0
        if run.returncode != 0 and nfailed == 0:
            nfailed = 1
        return (int(passed.group(1)) if passed else 0, nfailed, text[-2000:])

    def write_vendor(self, vendor: dict[str, str]) -> str:
        """Write files OUTSIDE the target source prefix (Import+Call control).
        Returns the directory that must be on PYTHONPATH."""
        vdir = os.path.join(self.tdir, "vendor_root")
        core.rm_rf(vdir)
        for rel, code in vendor.items():
            p = os.path.join(vdir, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(code)
        return vdir

    def python(self) -> str:
        return core.py()


def prepare(task: Task, tdir: str, measure_bounds: bool = True) -> Workspace:
    """Clone + environment + bounds. Raises NotGradeable(status)."""
    core.PINS[task.clone] = {"sha": task.commit}
    repo_dir = os.path.join(tdir, "repo")
    rec = {"name": task.name, "clone": task.clone, "domain": task.domain,
           "track": task.track, "band": task.band, "stars": task.stars}
    # The fixed paper split includes some tasks that exceeded the generation
    # cap. They remain in the delivery denominator with a zero score, without
    # spending time cloning and installing their large environments.
    if task.n_modules > core.MAX_MODULES:
        rec["n_modules"] = task.n_modules
        raise NotGradeable("too-many-modules", rec)
    os.makedirs(tdir, exist_ok=True)
    if not os.path.isdir(repo_dir):
        sha = core.clone_repo(task.clone, repo_dir)
        if not sha:
            raise NotGradeable("clone-failed", rec)
        rec["commit"] = sha
    res = core.resolve_target(repo_dir, _TaskView(task))
    if not res["package"] or not res["tests"]:
        raise NotGradeable("no-package" if not res["package"] else "no-tests",
                           rec)
    rec.update(res)
    src_prefix, pkg, test_path = res["src_prefix"], res["package"], res["tests"]
    proj_dir = os.path.join(repo_dir, res["project_dir"])
    rec["deps"] = sorted(core.detect_deps(proj_dir))
    rec["python"], rec["python_why"] = core.fresh_venv(proj_dir)
    _ok, rec["env_setup"] = core.setup_env(proj_dir, repo_root=repo_dir,
                                           exclude_pkg=pkg)
    base, line, dirty, late = core.baseline(repo_dir, src_prefix, test_path, pkg)
    rec.update(baseline=base, baseline_line=line, baseline_dirty=dirty)
    if base <= 0:
        raise NotGradeable("baseline-unrunnable", rec)

    src_root = repo_dir if src_prefix == "." else os.path.join(repo_dir,
                                                              src_prefix)
    pkg_dir = os.path.join(src_root, pkg)
    only = None
    if not os.path.isdir(pkg_dir) and os.path.isfile(pkg_dir + ".py"):
        pkg_dir, only = src_root, [pkg + ".py"]
    mods = only or core.package_modules(pkg_dir)
    mods = [m for m in mods if m.endswith(".py") and not core.is_test_file(m)
            and os.path.basename(m) != "conftest.py"]
    rec["n_modules"] = len(mods)
    rec["single_module"] = only is not None
    if not mods:
        raise NotGradeable("no-modules", rec)
    if len(mods) > core.MAX_MODULES:
        raise NotGradeable("too-many-modules", rec)
    orig = {f: open(os.path.join(pkg_dir, f), errors="ignore").read()
            for f in mods}
    core.uninstall_target(proj_dir, pkg)
    ws = Workspace(task, tdir, repo_dir, proj_dir, src_prefix, pkg, test_path,
                   pkg_dir, mods, orig, rec)
    if measure_bounds:
        ident = ws.execute(orig)
        empty = ws.execute({f: "" for f in orig})
        rec["ceiling_run"] = len(ident.passed_ids()) or ident.passed
        rec["ceiling_run_verdict"] = ident.verdict
        rec["floor_run"] = len(empty.passed_ids()) or empty.passed
        frozen_c = task.ceiling_ids()
        if frozen_c and ident.tests:
            rec["ceiling_drift"] = len(frozen_c - ident.passed_ids())
    return ws


class _TaskView:
    """Attribute view consumed by core.resolve_target."""

    def __init__(self, t: Task):
        self.name, self.clone = t.name, t.clone
        self.package, self.src_prefix = t.package, t.src_prefix or "."
        self.tests = t.tests


def cleanup(tdir: str):
    for d in ("repo", "zero", "vendor_root", "prov"):
        core.rm_rf(os.path.join(tdir, d))
    core.rm_rf(core.VENV)
    tmp = os.environ.get("TMPDIR", "")
    if tmp and "lego_tmp_" in tmp and os.path.isdir(tmp):
        for leftover in os.listdir(tmp):
            core.rm_rf(os.path.join(tmp, leftover))


_SAFE = re.compile(r"[^a-z0-9_.-]")


def task_dir(clone_base: str, name: str) -> str:
    return os.path.join(clone_base, _SAFE.sub("_", name.lower()))


def now() -> float:
    return time.time()
