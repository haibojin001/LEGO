from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    data: pd.DataFrame
    original: pd.Series
    bust_threshold: Optional[float] = None
    goal_threshold: Optional[float] = None
    _maxdd_cache: Optional[pd.Series] = field(default=None, repr=False)

    @property
    def stats(self) -> Dict[str, float]:
        values = self.data.iloc[-1]
        return {
            "min": values.min(),
            "max": values.max(),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "percentile_5": values.quantile(0.05),
            "percentile_25": values.quantile(0.25),
            "percentile_75": values.quantile(0.75),
            "percentile_95": values.quantile(0.95),
        }

    @property
    def maxdd(self) -> Dict[str, float]:
        if self._maxdd_cache is None:
            drawdowns = []
            for column in self.data.columns:
                series = self.data[column]
                growth = series + 1
                peaks = growth.cummax()
                drawdowns.append(((growth - peaks) / peaks).min())

            object.__setattr__(
                self,
                "_maxdd_cache",
                pd.Series(drawdowns, index=self.data.columns),
            )

        values = self._maxdd_cache
        return {
            "min": values.min(),
            "max": values.max(),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "percentile_5": values.quantile(0.05),
            "percentile_95": values.quantile(0.95),
        }

    @property
    def bust_probability(self) -> Optional[float]:
        if self.bust_threshold is None:
            return None

        if self._maxdd_cache is None:
            _ = self.maxdd

        return (self._maxdd_cache <= self.bust_threshold).sum() / len(
            self._maxdd_cache
        )

    @property
    def goal_probability(self) -> Optional[float]:
        if self.goal_threshold is None:
            return None

        terminal = self.data.iloc[-1]
        return (terminal >= self.goal_threshold).sum() / len(terminal)

    def percentile(self, p: float) -> pd.Series:
        return self.data.quantile(p / 100, axis=1)

    def confidence_band(
        self, level: float = 0.95
    ) -> Tuple[pd.Series, pd.Series]:
        tail = (1 - level) / 2
        return (
            self.data.quantile(tail, axis=1),
            self.data.quantile(1 - tail, axis=1),
        )

    def plot(self, **kwargs) -> Any:
        from . import plots

        return plots.montecarlo(self, **kwargs)


def run_montecarlo(
    returns: pd.Series,
    sims: int = 1000,
    bust: Optional[float] = None,
    goal: Optional[float] = None,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    generator = np.random.default_rng(seed)
    values = returns.dropna().values
    periods = len(values)

    paths = np.empty((periods, sims))
    paths[:, 0] = values

    for simulation in range(1, sims):
        paths[:, simulation] = generator.permutation(values)

    cumulative_paths = np.cumprod(1 + paths, axis=0) - 1
    columns = [f"sim_{number}" for number in range(sims)]
    frame = pd.DataFrame(
        cumulative_paths,
        index=range(periods),
        columns=columns,
    )
    original = pd.Series(
        cumulative_paths[:, 0],
        index=range(periods),
        name="original",
    )

    return MonteCarloResult(
        data=frame,
        original=original,
        bust_threshold=bust,
        goal_threshold=goal,
    )