import warnings
from typing import List, Optional, Union

import numpy as np
from pandas import DataFrame, Index, Series


def compact_letter_display(
    p_values: Union[DataFrame, np.ndarray],
    alpha: float = 0.05,
    names: Optional[List] = None,
    maxiter: int = 100,
) -> Series:
    """Create a compact-letter representation of pairwise p-values."""
    matrix = np.asarray(p_values, dtype=float)
    n_groups = matrix.shape[0]

    if names is None:
        if isinstance(p_values, DataFrame):
            names = list(p_values.index)
        else:
            names = list(range(n_groups))

    compatible = (matrix >= alpha) | (matrix < 0)
    np.fill_diagonal(compatible, True)

    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if maxiter > 0 and compatible.all():
        return Series(["a"] * n_groups, index=Index(names), name="letters")

    if maxiter > 0 and np.count_nonzero(compatible) == n_groups:
        if n_groups > len(alphabet):
            raise ValueError(
                "Too many letter groups (%d); only %d letters available."
                % (n_groups, len(alphabet))
            )
        values = [
            "".join(alphabet[column] if row == column else " " for column in range(n_groups))
            for row in range(n_groups)
        ]
        return Series(values, index=Index(names), name="letters")

    neighborhoods = [
        frozenset(np.flatnonzero(compatible[group])) for group in range(n_groups)
    ]
    groups = set(neighborhoods)

    stable = False
    for _ in range(maxiter):
        updated = {
            letter_group.intersection(neighborhoods[member])
            for letter_group in groups
            for member in letter_group
        }
        updated.discard(frozenset())

        if updated == groups:
            stable = True
            break
        groups = updated

    if not stable:
        warnings.warn(
            "compact_letter_display did not converge after %d iterations." % maxiter,
            RuntimeWarning,
            stacklevel=2,
        )

    groups = [
        letter_group
        for letter_group in groups
        if all(letter_group <= neighborhoods[member] for member in letter_group)
    ]

    if not groups:
        groups = [frozenset((group,)) for group in range(n_groups)]

    groups.sort(key=len, reverse=True)
    groups.sort(key=min)

    if len(groups) > len(alphabet):
        raise ValueError(
            "Too many letter groups (%d); only %d letters available."
            % (len(groups), len(alphabet))
        )

    result = [
        "".join(
            alphabet[letter] if group in letter_group else " "
            for letter, letter_group in enumerate(groups)
        )
        for group in range(n_groups)
    ]

    return Series(result, index=Index(names), name="letters")