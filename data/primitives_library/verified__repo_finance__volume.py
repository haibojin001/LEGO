import numpy as np
import pandas as pd

from finance._validation import aligned, series, window_size
from finance.indicators.trend import ema, sma


def _volume(price: pd.Series, volume: pd.Series) -> tuple[pd.Series, pd.Series]:
    checked_price, checked_volume = aligned(price, volume)
    if (checked_volume < 0).any():
        raise ValueError("volume must be nonnegative")
    return checked_price, checked_volume


def vwma(price: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    price, volume = _volume(price, volume)
    size = window_size(window)
    volume_sum = volume.rolling(size).sum()
    weighted_sum = (price * volume).rolling(window).sum()
    return weighted_sum / volume_sum.where(volume_sum != 0)


def vwap(
    price: pd.Series, volume: pd.Series, sessions: pd.Series | None = None
) -> pd.Series:
    """Cumulative VWAP; pass exchange-session labels to reset intraday calculations."""
    price, volume = _volume(price, volume)

    if price.isna().any() or volume.isna().any():
        raise ValueError("VWAP requires complete observations")

    weighted = price * volume

    if sessions is None:
        return weighted.cumsum() / volume.cumsum().replace(0, np.nan)

    if not sessions.index.equals(price.index) or sessions.isna().any():
        raise ValueError("session labels must match the price index")

    grouped_weighted = weighted.groupby(sessions).cumsum()
    grouped_volume = volume.groupby(sessions).cumsum().replace(0, np.nan)
    return grouped_weighted / grouped_volume


def twap(price: pd.Series, window: int = 20) -> pd.Series:
    """TWAP for equally spaced observations; irregular bars must first be resampled."""
    return sma(price, window)


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    high, low, close, volume = aligned(high, low, close, volume)
    typical_price, volume = _volume((high + low + close) / 3, volume)

    change = typical_price.diff()
    raw_flow = typical_price * volume
    size = window_size(window)

    positive_flow = raw_flow.where(change > 0, 0).mask(change.isna())
    negative_flow = raw_flow.where(change < 0, 0).mask(change.isna())

    positive_total = positive_flow.rolling(size).sum()
    negative_total = negative_flow.rolling(window).sum()
    total = positive_total + negative_total

    result = 100 * positive_total / total.where(total != 0)
    return result.mask(total == 0, 50)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    close, volume = _volume(close, volume)
    signs = np.sign(close.diff().fillna(0))
    return (signs * volume).cumsum()


def pvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    close, volume = _volume(close, volume)
    changes = series(close, positive=True, missing=False).pct_change(fill_method=None).fillna(0)
    selected = changes.where(volume.diff() > 0, 0)
    return 1000 * (1 + selected).cumprod()


def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
    close, volume = _volume(close, volume)
    returns = series(close, positive=True).pct_change(fill_method=None).fillna(0)
    return (returns * volume).cumsum()


def accumulation_distribution(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    high, low, close, volume = aligned(high, low, close, volume)
    _volume(close, volume)

    spread = high - low
    numerator = 2 * close - high - low
    factor = (numerator / spread.where(spread != 0)).mask(spread == 0, 0)
    return (factor * volume).cumsum()


def chaikin_money_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    distribution = accumulation_distribution(high, low, close, volume)
    flow = distribution.diff()
    flow.iloc[0] = distribution.iloc[0]

    numerator = flow.rolling(window_size(window)).sum()
    denominator = volume.rolling(window).sum().replace(0, np.nan)
    return numerator / denominator


def chaikin_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")

    distribution = accumulation_distribution(high, low, close, volume)
    return ema(distribution, fast) - ema(distribution, slow)


def force_index(close: pd.Series, volume: pd.Series, window: int = 13) -> pd.Series:
    close, volume = _volume(close, volume)
    return ema(close.diff() * volume, window)


def ease_of_movement(
    high: pd.Series, low: pd.Series, volume: pd.Series, window: int = 14
) -> pd.Series:
    high, low, volume = aligned(high, low, volume)
    _volume(high, volume)

    midpoint = (high + low) / 2
    range_size = high - low
    scaled_volume = (volume / 1e8).replace(0, np.nan)
    movement = midpoint.diff() * range_size / scaled_volume
    return sma(movement, window)


def balance_of_power(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    open, high, low, close = aligned(open, high, low, close)
    return (close - open) / (high - low).replace(0, np.nan)


def vpci(
    close: pd.Series, volume: pd.Series, short: int = 5, long: int = 20
) -> pd.Series:
    if short >= long:
        raise ValueError("short must be smaller than long")

    confirmation = vwma(close, volume, long) - sma(close, long)
    price_component = vwma(close, volume, short) / sma(close, short).replace(0, np.nan)
    volume_component = sma(volume, short) / sma(volume, long).replace(0, np.nan)

    return confirmation * price_component * volume_component