from collections import defaultdict, deque

import numpy as np
import pandas as pd


def completed_trades(fills: pd.DataFrame) -> pd.DataFrame:
    """Create FIFO-matched completed trade portions from an ordered fill table."""
    needed = ["date", "asset", "quantity", "price", "commission"]
    if not set(needed).issubset(fills.columns):
        raise ValueError(f"required fill columns: {needed}")

    numeric = fills[["quantity", "price", "commission"]]
    if (
        not np.isfinite(numeric).all().all()
        or (fills.price <= 0).any()
        or (fills.commission < 0).any()
    ):
        raise ValueError("finite quantities, positive prices and nonnegative fees required")

    if not pd.Index(fills.date).is_monotonic_increasing:
        raise ValueError("fills must be chronological, preserving within-bar execution order")

    inventory = defaultdict(deque)
    matched_rows = []

    for execution in fills.itertuples():
        unfilled = float(execution.quantity)

        if unfilled == 0:
            if execution.commission != 0:
                raise ValueError("zero-quantity fills cannot carry commissions")
            continue

        unit_commission = execution.commission / abs(unfilled)
        queue = inventory[execution.asset]

        while queue and np.sign(queue[0]["quantity"]) != np.sign(unfilled):
            opening = queue[0]
            amount = min(abs(opening["quantity"]), abs(unfilled))
            opening_sign = np.sign(opening["quantity"])

            gross_pnl = opening_sign * amount * (execution.price - opening["price"])
            commissions = amount * (opening["fee"] + unit_commission)
            net_pnl = gross_pnl - commissions

            matched_rows.append(
                {
                    "asset": execution.asset,
                    "entry_date": opening["date"],
                    "exit_date": execution.date,
                    "side": "long" if opening_sign > 0 else "short",
                    "quantity": amount,
                    "entry_price": opening["price"],
                    "exit_price": execution.price,
                    "gross_pnl": gross_pnl,
                    "commission": commissions,
                    "net_pnl": net_pnl,
                    "return": net_pnl / (amount * opening["price"]),
                    "duration": pd.Timestamp(execution.date)
                    - pd.Timestamp(opening["date"]),
                }
            )

            opening["quantity"] -= opening_sign * amount
            unfilled += opening_sign * amount

            if abs(opening["quantity"]) < 1e-10:
                queue.popleft()

            if abs(unfilled) < 1e-10:
                unfilled = 0
                break

        if unfilled:
            queue.append(
                {
                    "quantity": unfilled,
                    "price": execution.price,
                    "date": execution.date,
                    "fee": unit_commission,
                }
            )

    return pd.DataFrame(
        matched_rows,
        columns=[
            "asset",
            "entry_date",
            "exit_date",
            "side",
            "quantity",
            "entry_price",
            "exit_price",
            "gross_pnl",
            "commission",
            "net_pnl",
            "return",
            "duration",
        ],
    )


def trade_statistics(trades: pd.DataFrame) -> pd.Series:
    """Summarize realized FIFO trade portions."""
    if not {"net_pnl", "return", "duration"}.issubset(trades.columns):
        raise ValueError("provide completed_trades output")

    profitable = trades.loc[trades.net_pnl > 0, "net_pnl"]
    unprofitable = trades.loc[trades.net_pnl < 0, "net_pnl"]

    return pd.Series(
        {
            "matched_lots": len(trades),
            "win_rate": (trades.net_pnl > 0).mean(),
            "average_win": profitable.mean(),
            "average_loss": unprofitable.mean(),
            "profit_factor": (
                profitable.sum() / -unprofitable.sum()
                if len(unprofitable)
                else np.nan
            ),
            "net_pnl": trades.net_pnl.sum(),
            "mean_return": trades["return"].mean(),
            "mean_holding_days": (
                trades.duration.dt.total_seconds().mean() / 86400
                if len(trades)
                else np.nan
            ),
        }
    )