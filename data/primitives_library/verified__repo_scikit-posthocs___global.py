from typing import Union, List, Tuple

from numpy import array, ndarray, log, isfinite, sort
from scipy.stats import chi2


def _validate_pvalues(p_vals: Union[List, ndarray]) -> ndarray:
    values = array(p_vals, dtype=float)

    invalid = (
        values.ndim != 1
        or values.size == 0
        or not isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    )

    if invalid:
        raise ValueError(
            "p_vals must be a non-empty one-dimensional sequence of finite "
            "p-values between 0 and 1; remove invalid values before calling "
            "this function"
        )

    return values


def global_simes_test(p_vals: Union[List, ndarray]) -> float:
    """Global Simes test of the intersection null hypothesis."""
    values = sort(_validate_pvalues(p_vals))
    ranks = array(range(1, values.size + 1))
    return float(min(values.size * values / ranks))


def global_f_test(
    p_vals: Union[List, ndarray], stat: bool = False
) -> Union[float, Tuple[float, float]]:
    """Fisher's combination test for the global null hypothesis."""
    values = _validate_pvalues(p_vals)
    t_stat = -2 * sum(log(values))
    p_value = chi2.sf(t_stat, df=2 * len(values))
    return (p_value, t_stat) if stat else p_value