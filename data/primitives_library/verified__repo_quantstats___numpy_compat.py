import warnings
from typing import Any, Optional, Union

import numpy as np
from packaging import version

NUMPY_VERSION = version.parse(np.__version__)

if NUMPY_VERSION >= version.parse("1.25.0"):
    product = np.prod
else:
    product = getattr(np, "product", np.prod)


def safe_numpy_operation(data, operation: str):
    if operation == "product":
        return product(data)
    if operation == "prod":
        return np.prod(data)
    return getattr(np, operation)(data)


def safe_array_function(func_name: str, *args, **kwargs) -> Any:
    if func_name == "product":
        return product(*args, **kwargs)
    if hasattr(np, func_name):
        return getattr(np, func_name)(*args, **kwargs)
    raise AttributeError(
        f"numpy has no attribute '{func_name}' in version {NUMPY_VERSION}"
    )


def handle_numpy_warnings():
    return warnings.catch_warnings()


def safe_percentile(data, percentile: Union[float, list], **kwargs):
    return np.percentile(data, percentile, **kwargs)


def safe_nanpercentile(data, percentile: Union[float, list], **kwargs):
    return np.nanpercentile(data, percentile, **kwargs)


def safe_quantile(data, quantile: Union[float, list], **kwargs):
    return np.quantile(data, quantile, **kwargs)


def safe_random_seed(seed: Optional[int]):
    if seed is not None:
        np.random.default_rng(seed)


def safe_datetime64_unit(dt, unit: str):
    return np.datetime64(dt, unit)