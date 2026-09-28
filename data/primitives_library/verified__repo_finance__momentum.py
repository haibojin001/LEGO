import numpy as np
import pandas as pd

from finance._validation import aligned, series, window_size
from finance.indicators.trend import ema, sma, smma


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = series(close).diff()
    gains = smma(delta.clip(lower=0), window)
    losses = smma((-delta.clip(upper=0)), window)
    combined = gains + losses
    result = 100 * gains / combined.where(combined != 0)
    return result.mask(combined == 0, 50)


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    smooth_k: int = 1,
    smooth_d: int = 3,
) -> pd.DataFrame:
    high, low, close = aligned(high, low, close)
    window_size(window)

    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    spread = highest - lowest

    raw_k = 100 * (close - lowest) / spread.where(spread != 0)
    raw_k = raw_k.mask(spread == 0, 50)
    k = sma(raw_k, smooth_k)
    d = sma(k, smooth_d)

    return pd.DataFrame({"k": k, "d": d})


def stochastic_rsi(
    close: pd.Series, window: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> pd.DataFrame:
    indicator = rsi(close, window)
    return stochastic(indicator, indicator, indicator, window, smooth_k, smooth_d)


def williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    return stochastic(high, low, close, window)["k"] - 100


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")

    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)

    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line,
        }
    )


def apo(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return macd(close, fast, slow)["macd"]


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    return series(close).diff(window_size(window))


def roc(close: pd.Series, window: int = 10) -> pd.Series:
    periods = window_size(window)
    return series(close, positive=True).pct_change(periods, fill_method=None) * 100


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    high, low, close = aligned(high, low, close)
    typical_price = (high + low + close) / 3
    mean_price = sma(typical_price, window)
    mean_deviation = typical_price.rolling(window).apply(
        lambda values: np.mean(np.abs(values - values.mean())),
        raw=True,
    )
    result = (typical_price - mean_price) / (
        0.015 * mean_deviation.where(mean_deviation != 0)
    )
    return result.mask(mean_deviation == 0, 0)


def dpo(close: pd.Series, window: int = 20) -> pd.Series:
    size = window_size(window)
    return series(close).shift(size // 2 + 1) - sma(close, size)


def tsi(close: pd.Series, slow: int = 25, fast: int = 13) -> pd.Series:
    delta = series(close).diff()
    numerator = ema(ema(delta, slow), fast)
    denominator = ema(ema(delta.abs(), slow), fast)
    result = 100 * numerator / denominator.where(denominator != 0)
    return result.mask(denominator == 0, 0)


def ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    windows: tuple[int, int, int] = (7, 14, 28),
) -> pd.Series:
    high, low, close = aligned(high, low, close)

    if len(windows) != 3 or not windows[0] < windows[1] < windows[2]:
        raise ValueError("provide three increasing windows")

    previous_close = close.shift()
    buying_low = pd.concat([low, previous_close], axis=1).min(axis=1, skipna=False)
    true_high = pd.concat([high, previous_close], axis=1).max(axis=1, skipna=False)

    buying_pressure = close - buying_low
    true_range = true_high - buying_low
    averages = []

    for period in windows:
        size = window_size(period)
        range_sum = true_range.rolling(size).sum()
        average = buying_pressure.rolling(period).sum() / range_sum.where(range_sum != 0)
        averages.append(average.mask(range_sum == 0, 0.5))

    return 100 * (4 * averages[0] + 2 * averages[1] + averages[2]) / 7


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.DataFrame:
    from finance.indicators.volatility import true_range

    high, low, close = aligned(high, low, close)

    upward_move = high.diff()
    downward_move = -low.diff()

    positive_move = upward_move.where(
        (upward_move > downward_move) & (upward_move > 0),
        0,
    ).mask(upward_move.isna())
    negative_move = downward_move.where(
        (downward_move > upward_move) & (downward_move > 0),
        0,
    ).mask(downward_move.isna())

    ranges = true_range(high, low, close).mask(upward_move.isna())
    average_range = smma(ranges, window)

    plus_di = 100 * smma(positive_move, window) / average_range.where(average_range != 0)
    plus_di = plus_di.mask(average_range == 0, 0)

    minus_di = 100 * smma(negative_move, window) / average_range.where(average_range != 0)
    minus_di = minus_di.mask(average_range == 0, 0)

    total_direction = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / total_direction.where(total_direction != 0)
    dx = dx.mask(total_direction == 0, 0)

    return pd.DataFrame(
        {
            "adx": smma(dx, window),
            "plus_di": plus_di,
            "minus_di": minus_di,
        }
    )


def aroon(high: pd.Series, low: pd.Series, window: int = 25) -> pd.DataFrame:
    high, low = aligned(high, low)
    window_size(window)

    up = high.rolling(window + 1).apply(
        lambda values: 100 * (window - np.argmax(values[::-1])) / window,
        raw=True,
    )
    down = low.rolling(window + 1).apply(
        lambda values: 100 * (window - np.argmin(values[::-1])) / window,
        raw=True,
    )

    return pd.DataFrame({"up": up, "down": down, "oscillator": up - down})


def price_momentum_oscillator(
    close: pd.Series, first: int = 35, second: int = 20, signal: int = 10
) -> pd.DataFrame:
    line = ema(10 * ema(roc(close, 1), first), second)
    return pd.DataFrame({"pmo": line, "signal": ema(line, signal)})


def dynamic_momentum_index(close: pd.Series) -> pd.Series:
    close = series(close)
    short_volatility = close.rolling(5).std()
    relative_volatility = short_volatility / short_volatility.rolling(10).mean()
    adaptive_windows = (14 / relative_volatility.where(relative_volatility > 0)).clip(5, 30)

    result = pd.Series(np.nan, index=close.index)
    indicators = {period: rsi(close, period) for period in range(5, 31)}

    for position, period in enumerate(adaptive_windows):
        if pd.notna(period):
            result.iloc[position] = indicators[int(period)].iloc[position]

    return result