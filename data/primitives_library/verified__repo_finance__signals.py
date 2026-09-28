import numpy as np
import pandas as pd

from finance._validation import aligned, series, window_size
from finance.indicators import bollinger_bands, ema, ribbon, rsi, sma, zscore


def crossover(fast: pd.Series, slow: pd.Series, *, short: bool = False) -> pd.Series:
    """Target exposure at bar close; execution delay belongs exclusively to the backtester."""
    fast, slow = aligned(fast, slow)
    values = np.where(fast > slow, 1.0, -1.0 if short else 0.0)
    targets = pd.Series(values, index=fast.index)
    return targets.mask(fast.isna() | slow.isna(), 0).rename("target")


def moving_average(
    close: pd.Series,
    fast: int = 20,
    slow: int = 50,
    *,
    exponential: bool = False,
    short: bool = False,
) -> pd.Series:
    """Close-time targets: 1 when fast > slow, otherwise 0 (or -1 with short=True).

    Keep the close index; warm-up targets are zero. Pass targets unshifted to backtest.
    """
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")
    indicator = ema if exponential else sma
    return crossover(indicator(close, fast), indicator(close, slow), short=short)


def threshold_reversion(
    oscillator: pd.Series,
    lower: float,
    upper: float,
    exit_level: float,
    *,
    short: bool = False,
) -> pd.Series:
    """Enter outside thresholds, exit on return to the middle; missing bars hold state."""
    oscillator = series(oscillator)
    if not lower < exit_level < upper:
        raise ValueError("require lower < exit_level < upper")

    position = 0.0
    output = []

    for value in oscillator:
        if np.isfinite(value):
            if (position > 0 and value >= exit_level) or (
                position < 0 and value <= exit_level
            ):
                position = 0.0
            elif position == 0:
                if value <= lower:
                    position = 1.0
                elif short and value >= upper:
                    position = -1.0
        output.append(position)

    return pd.Series(output, index=oscillator.index, name="target")


def rsi_reversion(
    close: pd.Series,
    window: int = 14,
    lower: float = 30,
    upper: float = 70,
    *,
    short: bool = False,
) -> pd.Series:
    return threshold_reversion(rsi(close, window), lower, upper, 50, short=short)


def bollinger_reversion(
    close: pd.Series,
    window: int = 20,
    deviations: float = 2,
    *,
    short: bool = False,
) -> pd.Series:
    bands = bollinger_bands(close, window, deviations)
    width = (bands.upper - bands.lower) / 2
    oscillator = (close - bands.middle) / width.replace(0, np.nan)
    return threshold_reversion(oscillator, -1, 1, 0, short=short)


def pairs_trade(
    first: pd.Series,
    second: pd.Series,
    window: int = 60,
    entry: float = 2,
    exit: float = 0.5,
) -> pd.DataFrame:
    """Experimental equal-dollar log-ratio reversion; requires separate cointegration assessment."""
    first, second = aligned(first, second)
    series(first, positive=True, missing=False)
    series(second, positive=True, missing=False)

    if not 0 <= exit < entry:
        raise ValueError("require 0 <= exit < entry")

    scores = zscore(np.log(first / second), window)
    state = 0.0
    output = []

    for score in scores:
        if np.isfinite(score):
            if (state > 0 and score >= -exit) or (state < 0 and score <= exit):
                state = 0.0
            elif state == 0:
                if score <= -entry:
                    state = 1.0
                elif score >= entry:
                    state = -1.0
        output.append(state)

    values = np.asarray(output)
    return pd.DataFrame(
        {"first": values / 2, "second": -values / 2},
        index=first.index,
    )


def ribbon_trend(close: pd.Series) -> pd.Series:
    values = ribbon(close)
    fast = values.iloc[:, :6].min(axis=1, skipna=False)
    slow = values.iloc[:, 6:].max(axis=1, skipna=False)
    return crossover(fast, slow)


def breakout(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    high, low, close = aligned(high, low, close)
    window_size(window)

    upper = high.rolling(window).max().shift()
    lower = low.rolling(window).min().shift()

    target = 0.0
    output = []

    for price, top, bottom in zip(close, upper, lower, strict=True):
        if price > top:
            target = 1.0
        elif price < bottom:
            target = 0.0
        output.append(target)

    return pd.Series(output, index=close.index, name="target")


def trailing_stop(
    close: pd.Series,
    targets: pd.Series,
    fraction: float = 0.1,
) -> pd.Series:
    """Long-only close-based trailing stop, executed next open; rearm after source signal goes flat."""
    close, targets = aligned(close, targets)

    if not 0 < fraction < 1 or not targets.isin([0, 1]).all():
        raise ValueError("require fraction in (0,1) and binary long targets")

    active = False
    stopped = False
    peak = 0.0
    output = []

    for price, target in zip(close, targets, strict=True):
        if target == 0:
            active = False
            stopped = False
            peak = 0.0
        elif not stopped:
            active = True
            peak = max(peak, price)
            if price <= peak * (1 - fraction):
                active = False
                stopped = True
        output.append(float(active))

    return pd.Series(output, index=close.index, name="target")


def lag_reversal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    step: int = 1,
    confirmation: int = 5,
) -> pd.Series:
    """Contrarian experiment: long on lower close/low, short on higher close/high versus lagged bars."""
    high, low, close = aligned(high, low, close)
    window_size(step)
    window_size(confirmation)

    long = (close < close.shift(step)) & (low < low.shift(confirmation))
    short = (close > close.shift(step)) & (high > high.shift(confirmation))

    values = np.select([long, short], [1.0, -1.0], default=0.0)
    return pd.Series(values, index=close.index, name="target")