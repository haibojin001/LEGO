from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union


def load_dataset(source: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if isinstance(source, list):
        return _validate(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of objects")

        return _validate(data)

    if suffix == ".csv":
        return _load_csv(path)

    raise ValueError(f"Unsupported file format: {suffix}. Use .json or .csv")


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            item: Dict[str, Any] = {"input": row.get("input", "")}
            expected = row.get("expected", "")

            try:
                item["expected"] = json.loads(expected)
            except (json.JSONDecodeError, ValueError):
                item["expected"] = expected

            for key, value in row.items():
                if key not in ("input", "expected"):
                    item[key] = value

            items.append(item)

    return items


def _validate(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for index, item in enumerate(data):
        if "input" not in item:
            raise ValueError(f"Dataset item {index} missing 'input' key")
        if "expected" not in item:
            raise ValueError(f"Dataset item {index} missing 'expected' key")

    return data