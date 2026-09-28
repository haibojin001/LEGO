"""Generate execution tests from the current construction plan.

These tests are feedback for the construction loop. The frozen native test
identifiers alone determine the benchmark score.
"""

from __future__ import annotations

import ast

BATCH = 10


def _valid_test(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name.startswith("test_") for n in ast.walk(tree))


def synthesize(llm, pkg: str, statement: str, reqs) -> dict[str, str]:
    """Return pytest files based only on the task and public plan.

    No original implementations, native tests, or frozen grading keys are
    included in the request. A malformed response fails the construction run
    instead of silently turning the paper's supplemental test stage off.
    """
    tests = {}
    for start in range(0, len(reqs), BATCH):
        chunk = reqs[start:start + BATCH]
        plan = "\n\n".join(
            f"[{r.rid}] {r.module}\ncapability: {r.capability}\n"
            f"required interface:\n{r.interface[:1600]}\n"
            f"sibling dependencies: {', '.join(r.internal_deps)}"
            for r in chunk)
        prompt = (
            "Generate pytest unit and integration tests from the CURRENT PLAN "
            "for a repository construction task. Test documented behavior and "
            "public interfaces; include interactions between modules when the "
            "plan identifies them. Do not assume access to original source "
            "implementations or native benchmark tests. Return a JSON object "
            'of the form {"tests": [{"content": "<complete Python test file>"}]}. '
            "Every file must contain at least one test_ function. Do not "
            "include markdown.\n\n"
            f"# Construction Request\n{statement[:2000]}\n\n"
            f"# Target Package\n{pkg}\n\n# Current Plan\n{plan}")
        for attempt in range(2):
            obj = llm.json(
                prompt + ("\n\nThe previous response was malformed; return "
                          "complete pytest test files in the required JSON."
                          if attempt else ""), max_tokens=5000)
            items = obj.get("tests") if isinstance(obj, dict) else None
            if (isinstance(items, list) and items and len(items) <= BATCH
                    and all(isinstance(item, dict)
                            and isinstance(item.get("content"), str)
                            and _valid_test(item["content"]) for item in items)):
                for j, item in enumerate(items):
                    tests[f"test_lego_plan_{start // BATCH}_{j}.py"] = (
                        item["content"])
                break
        else:
            raise ValueError(f"plan-test synthesis failed for batch "
                             f"{start // BATCH} after 2 attempts")
    return tests
