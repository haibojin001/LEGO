from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from promptify.core.exceptions import EvaluationError
from promptify.eval.metrics import METRIC_REGISTRY

logger = logging.getLogger("promptify")


def _extract_comparable(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump()
    return result


def _compare_for_metric(
    metric_name: str,
    predicted: Any,
    expected: Any,
) -> float:
    metric_fn = METRIC_REGISTRY.get(metric_name)
    if metric_fn is None:
        raise EvaluationError(f"Unknown metric: {metric_name}")

    predicted_value = _extract_comparable(predicted)
    expected_value = _extract_comparable(expected)

    if metric_name in ("precision", "recall", "f1"):
        if isinstance(predicted_value, dict):
            flattened_prediction = []
            for value in predicted_value.values():
                if isinstance(value, list):
                    flattened_prediction.extend(str(item) for item in value)
                else:
                    flattened_prediction.append(str(value))
            predicted_value = flattened_prediction

        if isinstance(expected_value, dict):
            flattened_expected = []
            for value in expected_value.values():
                if isinstance(value, list):
                    flattened_expected.extend(str(item) for item in value)
                else:
                    flattened_expected.append(str(value))
            expected_value = flattened_expected

        if not isinstance(predicted_value, list):
            predicted_value = [predicted_value]
        if not isinstance(expected_value, list):
            expected_value = [expected_value]

        return metric_fn(predicted_value, expected_value)

    if metric_name == "accuracy":
        if not isinstance(predicted_value, list):
            predicted_value = [predicted_value]
        if not isinstance(expected_value, list):
            expected_value = [expected_value]

        return metric_fn(predicted_value, expected_value)

    if metric_name == "exact_match":
        return metric_fn(str(predicted_value), str(expected_value))

    if metric_name == "rouge":
        values = metric_fn(str(predicted_value), str(expected_value))
        return values.get("rougeL", 0.0)

    return metric_fn(predicted_value, expected_value)


def evaluate(
    task: Any,
    dataset: List[Dict[str, Any]],
    metrics: List[str],
    max_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, float]:
    if not dataset:
        raise EvaluationError("Dataset is empty")

    samples = dataset[:max_samples] if max_samples else dataset
    total = len(samples)
    scores: Dict[str, List[float]] = {metric: [] for metric in metrics}

    for index, sample in enumerate(samples):
        text_input = sample.get("input", "")
        expected = sample.get("expected")
        extra_kwargs = {
            key: value
            for key, value in sample.items()
            if key not in ("input", "expected")
        }

        try:
            predicted = task(text_input, **extra_kwargs)
        except Exception as exc:
            logger.warning("Evaluation failed on sample %d: %s", index, exc)
            for metric in metrics:
                scores[metric].append(0.0)
            continue

        for metric in metrics:
            try:
                scores[metric].append(
                    _compare_for_metric(metric, predicted, expected)
                )
            except Exception as exc:
                logger.warning(
                    "Metric %s failed on sample %d: %s",
                    metric,
                    index,
                    exc,
                )
                scores[metric].append(0.0)

        if progress_callback:
            progress_callback(index + 1, total)

    return {
        metric: round(sum(values) / len(values), 4) if values else 0.0
        for metric, values in scores.items()
    }