"""Scoring (Eq. 7) and admissibility (Eq. 8 / App. "Evaluation protocol").

    s(G;T) = clamp_[0,1]( (Pi(G) - f(T)) / (c(T) - f(T)) )

Pi(G) counts the frozen ceiling node ids that pass on the generated tree, so a
test the original sources could not pass never contributes. c and f are the
sizes of the frozen ceiling and floor id sets. A record is admissible only if
status == done, harness revision matches, provenance of the best round is ok,
and c - f >= MIN_SPAN; an inadmissible record scores 0 and still counts in the
denominator of S_delivery.
"""

from __future__ import annotations

from lego.harness import core


def make_score_fn(task, ws=None):
    C = task.ceiling_ids()
    F = task.floor_ids()
    if C:
        c, f = len(C), len(F & C) if F else 0
    else:                        # no frozen key: fall back to the in-run bounds
        c = (ws.record.get("ceiling_run") if ws else 0) or 0
        f = (ws.record.get("floor_run") if ws else 0) or 0
    span = c - f

    def score(res) -> dict:
        if C and res.tests:
            passed = len(res.passed_ids() & C)
        else:
            passed = min(res.passed, c) if c else res.passed
        raw = (passed - f) / span if span > 0 else 0.0
        return {"passed": passed, "score": round(min(max(raw, 0.0), 1.0), 4),
                "raw": round(raw, 4)}

    score.bounds = {"ceiling": c, "floor": f, "span": span,
                    "frozen": bool(C)}
    return score


def admissible(rec: dict) -> bool:
    return (rec.get("status") == "done"
            and rec.get("harness_rev") == core.HARNESS_REV
            and rec.get("verdict") == "ok"
            and (rec.get("span") or 0) >= core.MIN_SPAN)


def delivery_score(rec: dict | None) -> float:
    """s~_T of Eq. 8: the score if admissible, else 0 (missing record = 0)."""
    if not rec or not admissible(rec):
        return 0.0
    return float(rec.get("score") or 0.0)
