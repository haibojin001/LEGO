from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Set, Tuple


def _to_set(items: Any) -> Set[str]:
    """Convert items to a set of strings for comparison."""
    if isinstance(items, (list, tuple)):
        return {str(value) for value in items}
    return {str(items)}


def precision(predicted: List[Any], expected: List[Any]) -> float:
    """Compute precision: TP / (TP + FP)."""
    if not predicted:
        return 0.0

    predicted_set = _to_set(predicted)
    expected_set = _to_set(expected)
    if not predicted_set:
        return 0.0

    return len(predicted_set & expected_set) / len(predicted_set)


def recall(predicted: List[Any], expected: List[Any]) -> float:
    """Compute recall: TP / (TP + FN)."""
    if not expected:
        return 0.0

    predicted_set = _to_set(predicted)
    expected_set = _to_set(expected)
    if not expected_set:
        return 0.0

    return len(predicted_set & expected_set) / len(expected_set)


def f1(predicted: List[Any], expected: List[Any]) -> float:
    """Compute F1 score: harmonic mean of precision and recall."""
    precision_value = precision(predicted, expected)
    recall_value = recall(predicted, expected)
    denominator = precision_value + recall_value

    if denominator == 0:
        return 0.0

    return 2 * precision_value * recall_value / denominator


def accuracy(predicted: List[Any], expected: List[Any]) -> float:
    """Compute accuracy: fraction of correct predictions."""
    if not predicted or len(predicted) != len(expected):
        return 0.0

    matching = sum(
        str(prediction) == str(reference)
        for prediction, reference in zip(predicted, expected)
    )
    return matching / len(predicted)


def exact_match(predicted: str, expected: str) -> float:
    """Exact string match (1.0 or 0.0)."""
    return float(str(predicted).strip() == str(expected).strip())


def rouge(predicted: str, expected: str) -> Dict[str, float]:
    """Compute ROUGE scores. Requires rouge-score package."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score is required for ROUGE metrics. "
            "Install with: pip install promptify[eval]"
        )

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    score_values = scorer.score(str(expected), str(predicted))

    return {
        "rouge1": score_values["rouge1"].fmeasure,
        "rouge2": score_values["rouge2"].fmeasure,
        "rougeL": score_values["rougeL"].fmeasure,
    }


METRIC_REGISTRY: Dict[str, Any] = {
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "accuracy": accuracy,
    "exact_match": exact_match,
    "rouge": rouge,
}