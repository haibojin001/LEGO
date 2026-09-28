import numpy as np
import pandas as pd

from finance.data import normalize_ohlcv


def candles(prices: pd.DataFrame, overlays: pd.DataFrame | None = None):
    """Return a matplotlib figure; no display or file writes on calculation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    data = normalize_ohlcv(prices)

    if overlays is not None and not overlays.index.equals(data.index):
        raise ValueError("overlays must align with prices")

    figure, axis = plt.subplots(figsize=(11, 5))

    for position, candle in enumerate(data.itertuples()):
        rising = candle.close >= candle.open
        shade = "#16806a" if rising else "#c44848"
        axis.vlines(position, candle.low, candle.high, color=shade, linewidth=1)

        lower = min(candle.open, candle.close)
        body = abs(candle.close - candle.open)
        if body == 0:
            body = candle.close * 0.00001

        axis.add_patch(
            Rectangle(
                (position - 0.3, lower),
                0.6,
                body,
                color=shade,
            )
        )

    if overlays is not None:
        positions = range(len(data))
        for name in overlays:
            axis.plot(positions, overlays[name], label=name, linewidth=1)
        axis.legend()

    tick_positions = np.linspace(
        0,
        len(data) - 1,
        min(6, len(data)),
        dtype=int,
    )
    axis.set_xticks(tick_positions, data.index[tick_positions].strftime("%Y-%m-%d"))
    axis.set_ylabel("Price")
    axis.autoscale_view()
    figure.tight_layout()

    return figure


def correlation_heatmap(correlation: pd.DataFrame):
    import matplotlib.pyplot as plt

    labels_match = correlation.index.equals(pd.Index(correlation.columns))
    values_finite = np.isfinite(correlation).all().all()

    if not labels_match or not values_finite:
        raise ValueError("finite square matrix with matching labels required")

    figure, axis = plt.subplots(figsize=(7, 6))
    plot = axis.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")

    axis.set_xticks(
        range(len(correlation)),
        correlation.columns,
        rotation=45,
        ha="right",
    )
    axis.set_yticks(range(len(correlation)), correlation.index)

    figure.colorbar(plot, ax=axis)
    figure.tight_layout()

    return figure


def equity_chart(result):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))

    result.equity.plot(ax=axis, label="Strategy")
    result.benchmark.plot(ax=axis, label="Buy and hold")

    axis.set_ylabel("Account value")
    axis.legend()
    figure.tight_layout()

    return figure