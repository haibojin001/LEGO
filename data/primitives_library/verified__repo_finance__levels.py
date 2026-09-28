import numpy as np
import pandas as pd

from finance._validation import aligned, finite, frame, series, window_size
from finance.indicators.trend import ema


def pivot_points(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    method: str = "classic",
    open: pd.Series | None = None,
) -> pd.DataFrame:
    high, low, close = aligned(high, low, close)

    previous_high = high.shift(1)
    previous_low = low.shift(1)
    previous_close = close.shift(1)

    pivot = (previous_high + previous_low + previous_close) / 3
    spread = previous_high - previous_low

    if method == "woodie":
        pivot = (previous_high + previous_low + 2 * previous_close) / 4

    if method == "demark":
        if open is None:
            raise ValueError("DeMark pivots require open prices")

        aligned(close, open)
        previous_open = open.shift(1)

        lower_case = previous_high + 2 * previous_low + previous_close
        upper_case = 2 * previous_high + previous_low + previous_close
        equal_case = previous_high + previous_low + 2 * previous_close

        value = lower_case.where(
            previous_close < previous_open,
            upper_case.where(previous_close > previous_open, equal_case),
        )

        return pd.DataFrame(
            {
                "pivot": value / 4,
                "s1": value / 2 - previous_high,
                "r1": value / 2 - previous_low,
            }
        )

    output = {"pivot": pivot}

    if method in ("classic", "woodie"):
        output["r1"] = 2 * pivot - previous_low
        output["s1"] = 2 * pivot - previous_high
        output["r2"] = pivot + spread
        output["s2"] = pivot - spread
        output["r3"] = previous_high + 2 * (pivot - previous_low)
        output["s3"] = previous_low - 2 * (previous_high - pivot)
    elif method == "fibonacci":
        for number, ratio in enumerate((0.382, 0.618, 1), start=1):
            output[f"r{number}"] = pivot + ratio * spread
            output[f"s{number}"] = pivot - ratio * spread
    elif method == "camarilla":
        for number, divisor in enumerate((12, 6, 4, 2), start=1):
            adjustment = spread * 1.1 / divisor
            output[f"r{number}"] = previous_close + adjustment
            output[f"s{number}"] = previous_close - adjustment
    else:
        raise ValueError("method must be classic, woodie, demark, fibonacci or camarilla")

    midpoint = (previous_high + previous_low) / 2
    counterpart = 2 * pivot - midpoint
    output["cpr_low"] = np.minimum(counterpart, midpoint)
    output["cpr_high"] = np.maximum(counterpart, midpoint)

    return pd.DataFrame(output)


def fibonacci_levels(low: float, high: float) -> pd.Series:
    finite(low, "low", minimum=0)
    finite(high, "high", minimum=low)

    ratios = np.array((0, 0.236, 0.382, 0.5, 0.618, 0.786, 1))
    values = high - ratios * (high - low)

    return pd.Series(values, index=ratios, name="level")


def confirmed_extrema(high: pd.Series, low: pd.Series, radius: int = 2) -> pd.DataFrame:
    high, low = aligned(high, low)
    window_size(radius)

    shifted_high = high.shift(radius)
    shifted_low = low.shift(radius)
    width = 2 * radius + 1

    resistance = shifted_high.where(shifted_high == high.rolling(width).max())
    support = shifted_low.where(shifted_low == low.rolling(width).min())

    return pd.DataFrame({"support": support, "resistance": resistance})


def green_line(monthly_highs: pd.Series, confirmations: int = 3) -> pd.Series:
    monthly_highs = series(monthly_highs, positive=True, missing=False)
    window_size(confirmations)

    highest = -np.inf
    line = np.nan
    lower_months = 0
    values = []

    for value in monthly_highs:
        if value >= highest:
            highest = value
            lower_months = 0
        else:
            lower_months += 1
            if lower_months == confirmations:
                line = highest
        values.append(line)

    return pd.Series(values, index=monthly_highs.index, name="green_line")


def breadth(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    prices = frame(prices, positive=True)
    window_size(window, 2)

    movements = prices.diff()
    advances = (movements > 0).sum(axis=1)
    declines = (movements < 0).sum(axis=1)

    result = pd.DataFrame(
        {
            "advances": advances,
            "declines": declines,
            "net_advances": advances - declines,
            "new_highs": prices.eq(prices.rolling(window).max()).sum(axis=1),
            "new_lows": prices.eq(prices.rolling(window).min()).sum(axis=1),
        }
    )

    result.loc[result.index[: window - 1], ["new_highs", "new_lows"]] = np.nan
    result.iloc[0, :3] = np.nan
    result["ad_line"] = result["net_advances"].cumsum()

    return result


def mcclellan(advances: pd.Series, declines: pd.Series) -> pd.Series:
    advances, declines = aligned(advances, declines)

    if (advances < 0).any() or (declines < 0).any():
        raise ValueError("counts must be nonnegative")

    total = advances + declines
    difference = advances - declines
    adjusted = 1000 * difference / total.where(total != 0)
    adjusted = adjusted.mask(total == 0, 0)

    return ema(adjusted, 19) - ema(adjusted, 39)


def arms_index(
    advances: pd.Series,
    declines: pd.Series,
    advancing_volume: pd.Series,
    declining_volume: pd.Series,
) -> pd.Series:
    advances, declines, advancing_volume, declining_volume = aligned(
        advances,
        declines,
        advancing_volume,
        declining_volume,
    )

    issue_ratio = advances / declines.replace(0, np.nan)
    volume_ratio = advancing_volume / declining_volume.replace(0, np.nan)

    return issue_ratio / volume_ratio.replace(0, np.nan)


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    conversion: int = 9,
    base: int = 26,
    span: int = 52,
) -> pd.DataFrame:
    high, low = aligned(high, low)

    def midpoint(length):
        checked_length = window_size(length)
        return (high.rolling(checked_length).max() + low.rolling(length).min()) / 2

    conversion_line = midpoint(conversion)
    base_line = midpoint(base)

    return pd.DataFrame(
        {
            "conversion": conversion_line,
            "base": base_line,
            "span_a": (conversion_line + base_line) / 2,
            "span_b": midpoint(span),
        }
    )


def gann_fan(
    bars: int,
    anchor_price: float,
    price_per_bar: float,
    ratios: tuple[float, ...] = (0.125, 0.25, 0.5, 1, 2, 4, 8),
) -> pd.DataFrame:
    window_size(bars)
    finite(anchor_price, "anchor_price", minimum=0)
    finite(price_per_bar, "price_per_bar")

    if not ratios or not np.isfinite(ratios).all() or any(ratio <= 0 for ratio in ratios):
        raise ValueError("positive finite slope ratios required")

    offsets = np.arange(bars)
    columns = {
        f"{ratio:g}": anchor_price + price_per_bar * ratio * offsets
        for ratio in ratios
    }

    return pd.DataFrame(
        columns,
        index=pd.Index(offsets, name="bars_from_anchor"),
    )


def speed_resistance(
    start_price: float,
    end_price: float,
    duration: int,
    bars: int,
) -> pd.DataFrame:
    window_size(duration)
    window_size(bars)
    finite(start_price, "start_price", minimum=0)
    finite(end_price, "end_price", minimum=0)

    slope = (end_price - start_price) / duration
    result = gann_fan(bars, start_price, slope, (1 / 3, 2 / 3, 1))
    result.iloc[: min(duration, bars)] = np.nan

    return result


def pivot_midpoints(levels: pd.DataFrame) -> pd.DataFrame:
    if "pivot" not in levels or levels.columns.has_duplicates:
        raise ValueError("provide pivot_points output")

    supports = sorted(
        (column for column in levels if column.startswith("s") and column[1:].isdigit()),
        key=lambda column: int(column[1:]),
        reverse=True,
    )
    resistances = sorted(
        (column for column in levels if column.startswith("r") and column[1:].isdigit()),
        key=lambda column: int(column[1:]),
    )

    ordered = [*supports, "pivot", *resistances]

    return pd.DataFrame(
        {
            f"{left}_{right}": (levels[left] + levels[right]) / 2
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        }
    )