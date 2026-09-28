from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance._validation import finite, frame, window_size
from finance.analytics import performance


@dataclass(frozen=True)
class BacktestResult:
    """Bar-indexed equity/cash, share holdings, fractional returns and benchmark equity."""

    equity: pd.Series
    cash: pd.Series
    holdings: pd.DataFrame
    trades: pd.DataFrame
    returns: pd.Series
    benchmark: pd.Series
    metrics: pd.Series


def backtest(
    open_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    targets: pd.Series | pd.DataFrame,
    *,
    initial_cash: float = 10000,
    commission: float = 0.001,
    slippage: float = 0,
    borrow_rate: float = 0,
    periods: int = 252,
    rebalance: bool = False,
    liquidate: bool = False,
    high_prices: pd.Series | pd.DataFrame | None = None,
    low_prices: pd.Series | pd.DataFrame | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_fraction: float | None = None,
) -> BacktestResult:
    """Run a close-signal, next-open execution backtest."""

    def as_frame(data):
        return data.to_frame("asset") if isinstance(data, pd.Series) else data

    opens = frame(as_frame(open_prices), positive=True)
    closes = frame(as_frame(close_prices), positive=True)
    target_frame = frame(as_frame(targets))

    if not opens.index.equals(closes.index) or not opens.index.equals(target_frame.index):
        raise ValueError("open, close and target timestamps must match exactly")
    if not opens.columns.equals(closes.columns) or not opens.columns.equals(target_frame.columns):
        raise ValueError("open, close and target asset columns must match exactly")
    if len(opens) < 2 or (target_frame.abs().sum(axis=1) > 1 + 1e-12).any():
        raise ValueError("require at least two bars and gross targets <= 1")

    finite(initial_cash, "initial_cash", minimum=np.finfo(float).tiny)
    for name, value in (
        ("commission", commission),
        ("slippage", slippage),
        ("borrow_rate", borrow_rate),
    ):
        finite(value, name, minimum=0)
        if value >= 1:
            raise ValueError(f"{name} must be below 1")

    protective = any(
        value is not None for value in (stop_loss, take_profit, trailing_fraction)
    )
    for value in (stop_loss, take_profit, trailing_fraction):
        if value is not None and (not np.isfinite(value) or not 0 < value < 1):
            raise ValueError("protective order fractions must be in (0,1)")

    highs = lows = None
    if protective:
        if high_prices is None or low_prices is None:
            raise ValueError("protective orders require high_prices and low_prices")
        highs = frame(as_frame(high_prices), positive=True)
        lows = frame(as_frame(low_prices), positive=True)
        for values in (highs, lows):
            if not values.index.equals(opens.index) or not values.columns.equals(opens.columns):
                raise ValueError("protective OHLC must align exactly")
        invalid = (
            (highs < opens)
            | (highs < closes)
            | (lows > opens)
            | (lows > closes)
            | (lows > highs)
        )
        if invalid.any().any():
            raise ValueError("inconsistent OHLC bounds")

    window_size(periods)

    index = opens.index
    columns = opens.columns
    open_values = opens.to_numpy(dtype=float)
    close_values = closes.to_numpy(dtype=float)
    delayed_targets = target_frame.shift(1, fill_value=0).to_numpy(dtype=float)

    cash = float(initial_cash)
    shares = np.zeros(len(columns), dtype=float)
    previous_target = np.zeros(len(columns), dtype=float)
    stopped = np.zeros(len(columns), dtype=bool)
    entries = np.zeros(len(columns), dtype=float)
    extremes = np.zeros(len(columns), dtype=float)

    cash_rows = []
    equity_rows = []
    holding_rows = []
    fills = []
    fees_total = 0.0
    borrow_total = 0.0

    def execute(quantity, price, timestamp, signal_time, phase):
        nonlocal cash, shares, fees_total
        for asset_number, delta in enumerate(np.asarray(quantity, dtype=float)):
            if abs(delta) < 1e-10:
                continue
            execution_price = price[asset_number] * (
                1 + np.sign(delta) * slippage
            )
            fee = abs(delta) * execution_price * commission
            cash -= delta * execution_price + fee
            shares[asset_number] += delta
            fees_total += fee
            fills.append(
                {
                    "date": timestamp,
                    "signal_date": signal_time,
                    "asset": columns[asset_number],
                    "quantity": delta,
                    "price": execution_price,
                    "commission": fee,
                    "phase": phase,
                }
            )

    for bar_number, timestamp in enumerate(index):
        opening = open_values[bar_number]
        closing = close_values[bar_number]
        source_target = delayed_targets[bar_number]

        stopped[source_target != previous_target] = False
        effective_target = np.where(stopped, 0.0, source_target)
        shares_before_open = shares.copy()
        opening_equity = cash + shares @ opening

        if opening_equity <= 0:
            raise ValueError(
                f"account insolvent at {timestamp}; short losses exceeded equity"
            )

        if rebalance or not np.array_equal(source_target, previous_target):

            def residual(value):
                quantity = effective_target * value / opening - shares
                execution = opening * (1 + np.sign(quantity) * slippage)
                costs = np.sum(
                    quantity * (execution - opening)
                    + np.abs(quantity) * execution * commission
                )
                return value + costs - opening_equity

            if residual(0.0) > 0:
                raise ValueError(f"costs exceed equity at {timestamp}")

            lower = 0.0
            upper = opening_equity
            for _ in range(60):
                midpoint = (lower + upper) / 2
                if residual(midpoint) > 0:
                    upper = midpoint
                else:
                    lower = midpoint

            post_cost_equity = (lower + upper) / 2
            desired_shares = effective_target * post_cost_equity / opening
            signal_time = index[bar_number - 1] if bar_number else pd.NaT
            execute(
                desired_shares - shares,
                opening,
                timestamp,
                signal_time,
                "open",
            )

        previous_target = source_target.copy()

        changed_side = (shares != 0) & (
            (shares_before_open == 0)
            | (np.sign(shares) != np.sign(shares_before_open))
        )
        if changed_side.any():
            entries[changed_side] = opening[changed_side] * (
                1 + np.sign(shares[changed_side]) * slippage
            )
            extremes[changed_side] = entries[changed_side]

        borrow = np.maximum(-shares, 0) @ opening * borrow_rate / periods
        cash -= borrow
        borrow_total += borrow

        if protective:
            high = highs.iloc[bar_number].to_numpy(dtype=float)
            low = lows.iloc[bar_number].to_numpy(dtype=float)

            for asset_number in range(len(columns)):
                side = np.sign(shares[asset_number])
                if side == 0:
                    continue

                stop = None
                if stop_loss is not None:
                    stop = entries[asset_number] * (1 - side * stop_loss)

                if trailing_fraction is not None:
                    trailing_stop = extremes[asset_number] * (
                        1 - side * trailing_fraction
                    )
                    if stop is None:
                        stop = trailing_stop
                    elif side > 0:
                        stop = max(stop, trailing_stop)
                    else:
                        stop = min(stop, trailing_stop)

                profit = None
                if take_profit is not None:
                    profit = entries[asset_number] * (1 + side * take_profit)

                stop_hit = stop is not None and (
                    low[asset_number] <= stop
                    if side > 0
                    else high[asset_number] >= stop
                )
                profit_hit = profit is not None and (
                    high[asset_number] >= profit
                    if side > 0
                    else low[asset_number] <= profit
                )
                profit_gap = profit is not None and (
                    opening[asset_number] >= profit
                    if side > 0
                    else opening[asset_number] <= profit
                )

                if stop_hit or profit_hit:
                    if profit_gap:
                        fill_price = opening[asset_number]
                        phase = "take_profit"
                    elif stop_hit:
                        fill_price = (
                            min(opening[asset_number], stop)
                            if side > 0
                            else max(opening[asset_number], stop)
                        )
                        phase = "stop_loss"
                    else:
                        fill_price = (
                            max(opening[asset_number], profit)
                            if side > 0
                            else min(opening[asset_number], profit)
                        )
                        phase = "take_profit"

                    quantity = np.zeros(len(columns), dtype=float)
                    quantity[asset_number] = -shares[asset_number]
                    execution_basis = opening.copy()
                    execution_basis[asset_number] = fill_price
                    execute(
                        quantity,
                        execution_basis,
                        timestamp,
                        pd.NaT,
                        phase,
                    )
                    stopped[asset_number] = True

            active = shares != 0
            long_positions = active & (shares > 0)
            short_positions = active & (shares < 0)
            if long_positions.any():
                extremes[long_positions] = np.maximum(
                    extremes[long_positions], high[long_positions]
                )
            if short_positions.any():
                extremes[short_positions] = np.minimum(
                    extremes[short_positions], low[short_positions]
                )

        cash_rows.append(cash)
        holding_rows.append(shares.copy())
        equity_rows.append(cash + shares @ closing)

    if liquidate:
        timestamp = index[-1]
        close_price = close_values[-1]
        execute(-shares.copy(), close_price, timestamp, pd.NaT, "liquidate")
        cash_rows[-1] = cash
        holding_rows[-1] = shares.copy()
        equity_rows[-1] = cash

    equity = pd.Series(equity_rows, index=index, name="equity")
    cash_series = pd.Series(cash_rows, index=index, name="cash")
    holdings = pd.DataFrame(holding_rows, index=index, columns=columns)
    returns = equity.pct_change().fillna(0.0)
    returns.name = "returns"

    relative_prices = closes.divide(opens.iloc[0], axis=1)
    benchmark = initial_cash * relative_prices.mean(axis=1)
    benchmark.name = "benchmark"

    trades = pd.DataFrame(
        fills,
        columns=[
            "date",
            "signal_date",
            "asset",
            "quantity",
            "price",
            "commission",
            "phase",
        ],
    )

    total_return = equity.iloc[-1] / initial_cash - 1
    annual_return = (
        (equity.iloc[-1] / initial_cash) ** (periods / len(equity)) - 1
        if equity.iloc[-1] > 0
        else np.nan
    )
    volatility = returns.iloc[1:].std(ddof=1) * np.sqrt(periods)
    sharpe = (
        returns.iloc[1:].mean() / returns.iloc[1:].std(ddof=1) * np.sqrt(periods)
        if returns.iloc[1:].std(ddof=1) not in (0, np.nan)
        else np.nan
    )
    drawdown = equity / equity.cummax() - 1

    metrics = pd.Series(
        {
            "initial_cash": float(initial_cash),
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(drawdown.min()),
            "commission": float(fees_total),
            "borrow_cost": float(borrow_total),
            "trades": int(len(trades)),
        },
        dtype=float,
    )

    return BacktestResult(
        equity=equity,
        cash=cash_series,
        holdings=holdings,
        trades=trades,
        returns=returns,
        benchmark=benchmark,
        metrics=metrics,
    )