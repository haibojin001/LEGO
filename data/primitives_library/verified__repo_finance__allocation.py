from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance._validation import finite, frame, series, window_size


def _inputs(mean: pd.Series, covariance: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    ordered_mean = series(mean.sort_index(), missing=False)
    ordered_covariance = frame(covariance.sort_index().sort_index(axis=1))

    if (
        not ordered_mean.index.equals(ordered_covariance.index)
        or not ordered_mean.index.equals(ordered_covariance.columns)
    ):
        raise ValueError("mean returns and covariance must have matching asset labels")

    matrix = ordered_covariance.to_numpy()
    symmetric = np.allclose(matrix, matrix.T, atol=1e-12)
    positive_semidefinite = np.linalg.eigvalsh(matrix).min() >= -1e-10
    if not symmetric or not positive_semidefinite:
        raise ValueError("covariance must be symmetric positive semidefinite")

    return ordered_mean, ordered_covariance


def _weights(weights: pd.Series, labels: pd.Index) -> pd.Series:
    valid = (
        isinstance(weights, pd.Series)
        and weights.index.is_unique
        and set(weights.index) == set(labels)
    )
    if not valid:
        raise ValueError("weights must match asset labels")

    aligned = weights.reindex(labels).astype(float)
    acceptable = (
        np.isfinite(aligned).all()
        and not (aligned < 0).any()
        and np.isclose(aligned.sum(), 1, atol=1e-8)
    )
    if not acceptable:
        raise ValueError("long-only weights must be nonnegative and sum to one")

    return aligned


def portfolio_statistics(
    weights: pd.Series,
    mean: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_free: float = 0,
) -> pd.Series:
    """Mean/covariance and risk-free must use the same time unit (usually annual)."""
    mean, covariance = _inputs(mean, covariance)
    weights = _weights(weights, mean.index)
    finite(risk_free, "risk_free")

    expected_return = float(weights @ mean)
    variance = float(weights @ covariance @ weights)
    volatility = float(np.sqrt(max(variance, 0)))

    return pd.Series(
        {
            "return": expected_return,
            "volatility": volatility,
            "sharpe": (expected_return - risk_free) / volatility if volatility > 0 else np.nan,
        }
    )


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Constant-weight rebalanced returns, before costs; use backtest for actual execution."""
    asset_returns = frame(asset_returns)
    if (asset_returns < -1).any().any():
        raise ValueError("simple returns cannot be below -100%")

    valid_weights = _weights(weights, asset_returns.columns)
    return (asset_returns @ valid_weights).rename("portfolio_return")


@dataclass(frozen=True)
class Allocation:
    weights: pd.Series
    statistics: pd.Series


def optimize(
    mean: pd.Series,
    covariance: pd.DataFrame,
    *,
    objective: str = "minimum_variance",
    risk_free: float = 0,
    bounds: tuple[float, float] = (0, 1),
    target_return: float | None = None,
) -> Allocation:
    """Find long-only weights summing to one within the shared asset bounds.

    mean and both covariance axes must have the same asset labels. Use fractional
    returns and a consistent time basis for mean, covariance, risk_free and target_return.
    objective is minimum_variance or maximum_sharpe; target_return is an equality.
    Return Allocation with asset-indexed weights and return/volatility/sharpe statistics.
    Requires the portfolio extra; invalid inputs or failed constraints raise ValueError.
    """
    from scipy.optimize import minimize

    mean, covariance = _inputs(mean, covariance)
    lower, upper = bounds

    finite(lower, "lower", minimum=0)
    finite(upper, "upper", minimum=lower)
    finite(risk_free, "risk_free")

    count = len(mean)
    if upper > 1 or count * lower > 1 + 1e-12 or count * upper < 1 - 1e-12:
        raise ValueError("infeasible bounds")

    if objective not in ("minimum_variance", "maximum_sharpe"):
        raise ValueError("objective must be minimum_variance or maximum_sharpe")

    returns = mean.to_numpy()
    matrix = covariance.to_numpy()

    if objective == "maximum_sharpe" and returns.max() <= risk_free:
        raise ValueError("maximum Sharpe requires an asset with positive expected excess return")

    def loss(values):
        variance = float(values @ matrix @ values)
        if objective == "minimum_variance":
            return variance
        return -(values @ returns - risk_free) / np.sqrt(max(variance, 1e-18))

    def gradient(values):
        covariance_product = matrix @ values
        if objective == "minimum_variance":
            return 2 * covariance_product

        variance = max(float(values @ covariance_product), 1e-18)
        excess_return = values @ returns - risk_free
        return (
            -returns / np.sqrt(variance)
            + excess_return * covariance_product / variance**1.5
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda values: values.sum() - 1,
            "jac": lambda values: np.ones(count),
        }
    ]

    if target_return is not None:
        finite(target_return, "target_return")
        if target_return < returns.min() - 1e-10 or target_return > returns.max() + 1e-10:
            raise ValueError("target return is outside the feasible asset-return range")

        if np.ptp(returns) > 1e-12:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda values: values @ returns - target_return,
                    "jac": lambda values: returns,
                }
            )

    solution = minimize(
        loss,
        np.full(count, 1 / count),
        method="SLSQP",
        jac=gradient,
        bounds=[bounds] * count,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not solution.success:
        raise ValueError(f"portfolio optimization failed: {solution.message}")

    result_weights = pd.Series(solution.x, index=mean.index)
    violates_bounds = (
        (result_weights < lower - 1e-7).any()
        or (result_weights > upper + 1e-7).any()
        or abs(result_weights.sum() - 1) > 1e-7
    )
    if violates_bounds:
        raise ValueError("solver returned constraint-violating weights")

    if target_return is not None and abs(result_weights @ mean - target_return) > 1e-7:
        raise ValueError("solver did not meet target return")

    result_weights = result_weights.clip(lower=0)
    result_weights /= result_weights.sum()

    statistics = portfolio_statistics(
        result_weights,
        mean,
        covariance,
        risk_free=risk_free,
    )
    return Allocation(result_weights, statistics)


def efficient_frontier(
    mean: pd.Series,
    covariance: pd.DataFrame,
    points: int = 20,
) -> pd.DataFrame:
    """Efficient branch from the minimum-variance return to the highest asset return."""
    window_size(points, 2)

    minimum = optimize(mean, covariance)
    targets = np.linspace(minimum.statistics["return"], mean.max(), points)
    allocations = [
        optimize(mean, covariance, target_return=target)
        for target in targets
    ]

    rows = [
        allocation.statistics.to_dict()
        | {
            f"weight_{label}": value
            for label, value in allocation.weights.items()
        }
        for allocation in allocations
    ]
    return pd.DataFrame(rows)


def random_allocations(
    mean: pd.Series,
    covariance: pd.DataFrame,
    count: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    mean, covariance = _inputs(mean, covariance)
    window_size(count)

    generated = np.random.default_rng(seed).dirichlet(np.ones(len(mean)), count)
    result = pd.DataFrame(generated, columns=mean.index)

    result["return"] = generated @ mean.to_numpy()
    variances = np.einsum(
        "ij,jk,ik->i",
        generated,
        covariance.to_numpy(),
        generated,
    )
    result["volatility"] = np.sqrt(np.maximum(variances, 0))

    return result


def discrete_allocation(
    weights: pd.Series,
    prices: pd.Series,
    budget: float,
) -> tuple[pd.Series, float]:
    """Whole-share floor allocation; returns unspent cash without overspending."""
    prices = series(prices.sort_index(), positive=True, missing=False)
    weights = _weights(weights, prices.index)
    finite(budget, "budget", minimum=0)

    quantities = np.floor(weights * budget / prices).astype(int)
    cash = float(budget - quantities @ prices)
    return quantities, cash