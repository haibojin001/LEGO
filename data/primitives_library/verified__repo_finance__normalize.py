import re

import numpy as np
import pandas as pd


_TICKER_PATTERN = re.compile(r"[A-Z0-9^][A-Z0-9^=\-]{0,24}")


def normalize_ticker(ticker: str) -> str:
    symbol = ticker.strip().upper().replace(".", "-")
    if _TICKER_PATTERN.fullmatch(symbol) is None:
        raise ValueError(f"invalid ticker: {symbol!r}")
    return symbol


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("provider returned no price rows")

    frame = raw.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        removable_levels = []
        for level in range(frame.columns.nlevels):
            values = frame.columns.get_level_values(level)
            if len(values.unique()) == 1:
                removable_levels.append(level)

        if len(removable_levels) != 1:
            raise ValueError("expected exactly one ticker")

        frame.columns = frame.columns.droplevel(removable_levels[0])

    frame.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in frame.columns
    ]

    if not frame.columns.is_unique:
        raise ValueError("duplicate columns")

    if "date" in frame:
        frame = frame.set_index("date")

    if not isinstance(frame.index, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(frame.index):
            raise ValueError("price index must contain dates")
        frame.index = pd.to_datetime(frame.index, errors="raise")

    if frame.index.hasnans or not frame.index.is_unique:
        raise ValueError("missing or duplicate timestamps")

    frame = frame.sort_index()

    needed = ["open", "high", "low", "close", "volume"]
    absent = set(needed).difference(frame.columns)
    if absent:
        raise ValueError(f"missing OHLCV columns: {sorted(absent)}")

    frame = frame.loc[:, needed].apply(pd.to_numeric, errors="raise").astype(float)

    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("OHLCV contains missing or nonfinite values")

    price_columns = ["open", "high", "low", "close"]
    if (frame[price_columns] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError("prices must be positive and volume nonnegative")

    allowance = frame.high * 1e-8
    upper_violation = frame.high + allowance < frame[price_columns].max(axis=1)
    lower_violation = frame.low - allowance > frame[price_columns].min(axis=1)

    if (upper_violation | lower_violation).any():
        raise ValueError("inconsistent OHLC bounds; check adjustment basis")

    frame.index.name = "date"
    return frame