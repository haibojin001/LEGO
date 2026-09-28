import importlib.metadata
import importlib.util
import os
import re
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Optional, Union

import numpy as np
import yaml
from codetiming import Timer
from packaging import version
from yaml import Dumper


def is_sci_notation(number: float) -> bool:
    matcher = re.compile(r"^[+-]?\d+(\.\d*)?[eE][+-]?\d+$")
    return bool(matcher.match(str(number)))


def float_representer(dumper: Dumper, number: Union[float, np.float32, np.float64]):
    if is_sci_notation(number):
        rendered = str(number)
        if "." not in rendered and "e" in rendered:
            rendered = rendered.replace("e", ".0e", 1)
    else:
        rendered = str(round(number, 3))

    return dumper.represent_scalar("tag:yaml.org,2002:float", rendered)


yaml.add_representer(float, float_representer)
yaml.add_representer(np.float32, float_representer)
yaml.add_representer(np.float64, float_representer)


@lru_cache
def is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def get_package_version(name: str) -> "version.Version":
    try:
        package_version = importlib.metadata.version(name)
    except Exception:
        package_version = "0.0.0"
    return version.parse(package_version)


@lru_cache
def is_transformers_version_greater_than(content: str):
    return get_package_version("transformers") >= version.parse(content)


def union_two_dict(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """Union two dict. Will throw an error if there is an item not the same object with the same key."""
    for key, value in dict2.items():
        if key in dict1:
            assert dict1[key] == value, f"{key} in dict1 and dict2 are not the same object"
        dict1[key] = value
    return dict1


def append_to_dict(data: dict[str, list[Any]], new_data: dict[str, Any]) -> None:
    """Append dict to a dict of list."""
    for key, value in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(value)


def unflatten_dict(data: dict[str, Any], sep: str = "/") -> dict[str, Any]:
    result = {}
    for key, value in data.items():
        current = result
        parts = key.split(sep)
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def flatten_dict(data: dict[str, Any], parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    result = {}
    for key, value in data.items():
        combined_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, combined_key, sep=sep))
        else:
            result[combined_key] = value
    return result


def convert_dict_to_str(data: dict[str, Any]) -> str:
    return yaml.dump(data, indent=2)


def get_abs_path(path: str, prompt: str = "File") -> Optional[str]:
    if path is not None:
        if os.path.exists(path):
            return os.path.abspath(path)
        print(f"{prompt} {path} not found.")


@contextmanager
def timer(name: str, timing_raw: dict[str, float]):
    with Timer(name=name, logger=None) as elapsed_timer:
        yield
    timing_raw[name] = elapsed_timer.last