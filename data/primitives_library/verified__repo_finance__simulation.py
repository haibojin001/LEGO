import numpy as np
import pandas as pd

from finance._validation import finite, series, window_size
from finance.portfolio.allocation import _inputs, _weights


def geometric_brownian_motion(
    initial: float,
    drift: float,
    volatility: float,
    *,
    years: float = 1,
    steps: int = 252,
    paths: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate geometric Brownian motion price paths."""
    finite(initial, "initial", minimum=np.finfo(float).tiny)
    finite(drift, "drift")
    finite(volatility, "volatility", minimum=0)
    finite(years, "years", minimum=np.finfo(float).tiny)
    window_size(steps)
    window_size(paths)

    interval = years / steps
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal((steps, paths))
    log_returns = (
        (drift - volatility**2 / 2) * interval
        + volatility * np.sqrt(interval) * noise
    )
    cumulative = np.cumsum(log_returns, axis=0)
    path_values = initial * np.exp(
        np.concatenate((np.zeros((1, paths)), cumulative), axis=0)
    )

    time_index = pd.Index(
        np.linspace(0, years, steps + 1),
        name="years",
    )
    return pd.DataFrame(path_values, index=time_index)


def simulate_portfolio(
    weights: pd.Series,
    mean: pd.Series,
    covariance: pd.DataFrame,
    *,
    initial: float = 10000,
    years: float = 1,
    steps: int = 252,
    paths: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate fixed-weight buy-and-hold portfolio values."""
    mean, covariance = _inputs(mean, covariance)
    weights = _weights(weights, mean.index)

    finite(initial, "initial", minimum=np.finfo(float).tiny)
    finite(years, "years", minimum=np.finfo(float).tiny)
    window_size(steps)
    window_size(paths)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))

    interval = years / steps
    generator = np.random.default_rng(seed)
    asset_count = len(mean)
    relative_values = np.ones((paths, asset_count))
    portfolio_values = np.empty((steps + 1, paths))
    portfolio_values[0] = initial

    log_drift = (mean.to_numpy() - np.diag(covariance) / 2) * interval
    allocation = weights.to_numpy()

    for step in range(1, steps + 1):
        independent = generator.standard_normal(relative_values.shape)
        correlated = independent @ covariance_root.T * np.sqrt(interval)
        relative_values *= np.exp(log_drift + correlated)
        portfolio_values[step] = initial * (relative_values @ allocation)

    time_index = pd.Index(
        np.linspace(0, years, steps + 1),
        name="years",
    )
    return pd.DataFrame(portfolio_values, index=time_index)


def lump_sum_vs_dca(
    prices: pd.Series,
    budget: float = 12000,
    installments: int = 12,
) -> pd.DataFrame:
    """Compare immediate investment against periodic equal cash investments."""
    prices = series(prices, positive=True, missing=False)
    finite(budget, "budget", minimum=0)
    window_size(installments)

    if installments > len(prices):
        raise ValueError("installments exceed available bars")

    purchase_bars = set(
        np.linspace(0, len(prices) - 1, installments, dtype=int)
    )

    cash_remaining = budget
    share_count = 0.0
    dca_values = []

    for position, price in enumerate(prices):
        if position in purchase_bars:
            contribution = min(budget / installments, cash_remaining)
            cash_remaining -= contribution
            share_count += contribution / price

        dca_values.append(cash_remaining + share_count * price)

    return pd.DataFrame(
        {
            "lump_sum": budget * prices / prices.iloc[0],
            "dca": dca_values,
        },
        index=prices.index,
    )