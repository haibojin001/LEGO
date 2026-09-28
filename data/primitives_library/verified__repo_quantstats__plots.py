import importlib as _importlib

try:
    from pandas.plotting import register_matplotlib_converters as _rmc

    _rmc()
except (ImportError, AttributeError):
    pass

try:
    _wrappers = _importlib.import_module("quantstats._plotting.wrappers")
    _exported = getattr(_wrappers, "__all__", None)
    if _exported is None:
        _exported = tuple(
            name for name in vars(_wrappers) if not name.startswith("_")
        )

    for _name in _exported:
        globals()[_name] = getattr(_wrappers, _name)

except ModuleNotFoundError as _error:
    if _error.name not in {
        "quantstats._plotting",
        "quantstats._plotting.wrappers",
    }:
        raise

    _wrappers = None

    def _figure(ax=None, figsize=(10, 6), title=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        if title:
            ax.set_title(title)
        return fig, ax


    def _as_frame(data):
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, pd.Series):
            return data.to_frame(name=data.name or "Returns")
        return pd.DataFrame(data)


    def _finish(fig, show=True, savefig=None):
        import matplotlib.pyplot as plt

        if savefig:
            if isinstance(savefig, dict):
                fig.savefig(**savefig)
            else:
                fig.savefig(savefig, bbox_inches="tight")
        if show:
            plt.show(block=False)
        return fig


    def _returns_series(returns):
        import pandas as pd

        frame = _as_frame(returns)
        if frame.empty:
            return frame
        return frame.apply(pd.to_numeric, errors="coerce").fillna(0)


    def _cumulative(returns, compounded=True):
        values = _returns_series(returns)
        if compounded:
            return (1 + values).cumprod() - 1
        return values.cumsum()


    def plot_timeseries(
        returns,
        benchmark=None,
        title="Cumulative Returns",
        figsize=(10, 6),
        grayscale=False,
        compounded=True,
        ax=None,
        show=True,
        savefig=None,
        **kwargs
    ):
        frame = _cumulative(returns, compounded=compounded)
        fig, axis = _figure(ax=ax, figsize=figsize, title=title)
        style = "k" if grayscale else None
        frame.plot(ax=axis, color=style)
        if benchmark is not None:
            bench = _cumulative(benchmark, compounded=compounded)
            bench.plot(ax=axis, linestyle="--", color="gray", label="Benchmark")
        axis.set_ylabel("Cumulative Return")
        axis.legend(loc="best")
        axis.grid(alpha=0.25)
        return _finish(fig, show, savefig)


    def returns(returns, benchmark=None, **kwargs):
        return plot_timeseries(
            returns, benchmark=benchmark, title=kwargs.pop("title", "Cumulative Returns"), **kwargs
        )


    def log_returns(returns, benchmark=None, **kwargs):
        import numpy as np

        frame = _returns_series(returns)
        logged = np.log1p(frame)
        return plot_timeseries(
            logged,
            benchmark=None if benchmark is None else np.log1p(_returns_series(benchmark)),
            compounded=False,
            title=kwargs.pop("title", "Log Returns"),
            **kwargs
        )


    def daily_returns(
        returns,
        title="Daily Returns",
        figsize=(10, 6),
        grayscale=False,
        ax=None,
        show=True,
        savefig=None,
        **kwargs
    ):
        frame = _returns_series(returns)
        fig, axis = _figure(ax=ax, figsize=figsize, title=title)
        frame.plot(ax=axis, color="k" if grayscale else None, alpha=0.8)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Return")
        axis.grid(alpha=0.25)
        return _finish(fig, show, savefig)


    def histogram(
        returns,
        benchmark=None,
        title="Return Distribution",
        figsize=(10, 6),
        grayscale=False,
        ax=None,
        show=True,
        savefig=None,
        bins=50,
        **kwargs
    ):
        frame = _returns_series(returns)
        fig, axis = _figure(ax=ax, figsize=figsize, title=title)
        for column in frame:
            axis.hist(
                frame[column].dropna(),
                bins=bins,
                alpha=0.5,
                label=str(column),
                color="gray" if grayscale else None,
            )
        if benchmark is not None:
            bench = _returns_series(benchmark)
            for column in bench:
                axis.hist(
                    bench[column].dropna(),
                    bins=bins,
                    alpha=0.35,
                    histtype="step",
                    linewidth=1.5,
                    label="Benchmark",
                    color="black",
                )
        axis.legend(loc="best")
        axis.grid(alpha=0.25)
        return _finish(fig, show, savefig)


    def drawdown(
        returns,
        title="Drawdown",
        figsize=(10, 6),
        grayscale=False,
        ax=None,
        show=True,
        savefig=None,
        **kwargs
    ):
        wealth = _cumulative(returns) + 1
        drawdowns = wealth.divide(wealth.cummax()).subtract(1)
        fig, axis = _figure(ax=ax, figsize=figsize, title=title)
        for column in drawdowns:
            axis.fill_between(
                drawdowns.index,
                drawdowns[column].astype(float),
                0,
                alpha=0.35,
                color="gray" if grayscale else None,
                label=str(column),
            )
        axis.set_ylabel("Drawdown")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        return _finish(fig, show, savefig)


    def drawdowns_periods(returns, **kwargs):
        kwargs.setdefault("title", "Drawdown Periods")
        return drawdown(returns, **kwargs)


    def rolling_volatility(
        returns,
        rolling_period=126,
        period=None,
        title="Rolling Volatility",
        figsize=(10, 6),
        ax=None,
        show=True,
        savefig=None,
        **kwargs
    ):
        period = period or rolling_period
        frame = _returns_series(returns).rolling(period).std() * (252 ** 0.5)
        fig, axis = _figure(ax=ax, figsize=figsize, title=title)
        frame.plot(ax=axis)
        axis.set_ylabel("Annualized Volatility")
        axis.grid(alpha=0.25)
        return _finish(fig, show, savefig)


    def rolling_sharpe(
        returns,
        rolling_period=126,
        period=None,
        rf=0,
        title="Rolling Sharpe",
        figsize=(10, 6),
        ax=None,
        show=True,
        savefig=None,
        **kwargs
    ):
        period = period or rolling_period
        frame = _returns_series(returns) - rf / 252
        values = frame.rolling(period).mean().divide(frame.rolling(period).std()) * (252 ** 0.5)
        fig, axis = _figure(ax=ax, figsize=figsize, title=title)
        values.plot(ax=axis)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
        return _finish(fig, show, savefig)


    def rolling_sortino(
        returns,
        rolling_period=126,
        period=None,
        rf=0,
        title="Rolling Sortino",
        figsize=(10, 6),
        ax=None,
        show=True,
        savefig=None,
        **kwargs
    ):
        period = period or rolling_period
        frame = _returns_series(returns) - rf / 252
        downside = frame.where(frame < 0, 0).rolling(period).std()
        values = frame.rolling(period).mean().divide(downside) * (252 ** 0.5)
        fig, axis = _figure(ax=ax, figsize=figsize, title=title)
        values.plot(ax=axis)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
        return _finish(fig, show, savefig)


    def rolling_beta(
        returns,
        benchmark,
        rolling_period=126,
        period=None,
        title="Rolling Beta",
        figsize=(10, 6),
        ax=None,
        show=True,
        savefig=None,
        **kwargs
    ):
        period = period or rolling_period
        asset = _returns_series(returns).iloc[:, 0]
        bench = _returns_series(benchmark).iloc[:, 0].reindex(asset.index)
        beta = asset.rolling(period).cov(bench).divide(bench.rolling(period).var())
        fig, axis = _figure(ax=ax, figsize=figsize, title=title)
        beta.plot(ax=axis, label="Beta")
        axis.axhline(1, color="black", linewidth=0.8, linestyle="--")
        axis.legend(loc="best")
        axis.grid(alpha=0.25)
        return _finish(fig, show, savefig)


    def yearly_returns(
        returns,
        title="Yearly Returns",
        figsize=(10, 6),
        ax=None,
        show=True,
        savefig=None,
        **kwargs
    ):
        frame = _returns_series(returns)
        annual = frame.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        annual.index = annual.index.year
        fig, axis = _figure(ax=ax, figsize=figsize, title=title)
        annual.plot(kind="bar", ax=axis)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Return")
        return _finish(fig, show, savefig)


    def eoy_returns(returns, **kwargs):
        kwargs.setdefault("title", "End of Year Returns")
        return yearly_returns(returns, **kwargs)


    def monthly_heatmap(
        returns,
        title="Monthly Returns",
        figsize=(10, 6),
        ax=None,
        show=True,
        savefig=None,
        **kwargs
    ):
        import numpy as np

        frame = _returns_series(returns)
        series = frame.iloc[:, 0]
        monthly = series.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        table = monthly.to_frame("return")
        table["year"] = table.index.year
        table["month"] = table.index.month
        matrix = table.pivot(index="year", columns="month", values="return")
        fig, axis = _figure(ax=ax, figsize=figsize, title=title)
        image = axis.imshow(matrix.fillna(0).values, aspect="auto", cmap="RdYlGn")
        axis.set_yticks(range(len(matrix.index)))
        axis.set_yticklabels(matrix.index)
        axis.set_xticks(range(12))
        axis.set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )
        fig.colorbar(image, ax=axis, format="%.0%%")
        return _finish(fig, show, savefig)


    def distribution(returns, benchmark=None, **kwargs):
        return histogram(returns, benchmark=benchmark, **kwargs)


    def plot_distribution(returns, benchmark=None, **kwargs):
        return histogram(returns, benchmark=benchmark, **kwargs)


    def plot_histogram(returns, benchmark=None, **kwargs):
        return histogram(returns, benchmark=benchmark, **kwargs)


    def plot_monthly_heatmap(returns, **kwargs):
        return monthly_heatmap(returns, **kwargs)


    def plot_monthly_returns(returns, **kwargs):
        return monthly_heatmap(returns, **kwargs)


    def plot_yearly_returns(returns, **kwargs):
        return yearly_returns(returns, **kwargs)


    def plot_rolling_stats(returns, **kwargs):
        return rolling_sharpe(returns, **kwargs)


    def plot_rolling_beta(returns, benchmark=None, **kwargs):
        return rolling_beta(returns, benchmark=benchmark, **kwargs)


    def plot_drawdown_periods(returns, **kwargs):
        return drawdowns_periods(returns, **kwargs)


    def plot_longest_drawdowns(returns, **kwargs):
        return drawdowns_periods(returns, **kwargs)


    def plot_returns_bars(returns, **kwargs):
        return yearly_returns(returns, **kwargs)


    def plot_table(data, title=None, figsize=(10, 6), ax=None, show=True, savefig=None, **kwargs):
        fig, axis = _figure(ax=ax, figsize=figsize, title=title)
        axis.axis("off")
        frame = _as_frame(data)
        axis.table(
            cellText=frame.values,
            rowLabels=frame.index,
            colLabels=frame.columns,
            loc="center",
        )
        return _finish(fig, show, savefig)


    def plot_worst_returns(returns, **kwargs):
        return histogram(returns, **kwargs)


    def plot_earnings(returns, **kwargs):
        return plot_timeseries(returns, **kwargs)


    def earnings(returns, **kwargs):
        return plot_earnings(returns, **kwargs)


    def snapshot(returns, benchmark=None, title="Portfolio Snapshot", figsize=(12, 8), show=True, savefig=None, **kwargs):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        plot_timeseries(returns, benchmark=benchmark, ax=axes[0, 0], show=False, title="Cumulative Returns")
        drawdown(returns, ax=axes[0, 1], show=False, title="Drawdown")
        histogram(returns, benchmark=benchmark, ax=axes[1, 0], show=False, title="Distribution")
        rolling_sharpe(returns, ax=axes[1, 1], show=False, title="Rolling Sharpe")
        fig.suptitle(title)
        fig.tight_layout()
        return _finish(fig, show, savefig)


    _exported = (
        "snapshot",
        "monthly_heatmap",
        "returns",
        "log_returns",
        "yearly_returns",
        "histogram",
        "daily_returns",
        "rolling_beta",
        "rolling_volatility",
        "rolling_sharpe",
        "rolling_sortino",
        "drawdown",
        "drawdowns_periods",
        "eoy_returns",
        "distribution",
        "earnings",
        "plot_returns_bars",
        "plot_histogram",
        "plot_timeseries",
        "plot_rolling_stats",
        "plot_rolling_beta",
        "plot_longest_drawdowns",
        "plot_drawdown_periods",
        "plot_distribution",
        "plot_table",
        "plot_worst_returns",
        "plot_earnings",
        "plot_monthly_heatmap",
        "plot_monthly_returns",
        "plot_yearly_returns",
    )

__all__ = tuple(_exported)