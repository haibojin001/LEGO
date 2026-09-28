import base64 as _base64
import inspect as _inspect
import io as _io
import os as _os
import re as _re
import tempfile as _tempfile
import webbrowser as _webbrowser
from datetime import datetime as _dt
from math import ceil as _ceil
from pathlib import Path as _Path

import numpy as _np
import pandas as _pd
from tabulate import tabulate as _tabulate

from . import __version__

_stats = None
_utils = None
_plots = None


def _get_stats():
    global _stats
    if _stats is None:
        from . import stats
        _stats = stats
    return _stats


def _get_utils():
    global _utils
    if _utils is None:
        from . import utils
        _utils = utils
    return _utils


def _get_plots():
    global _plots
    if _plots is None:
        from . import plots
        _plots = plots
    return _plots


def _get_trading_periods(periods_per_year=252):
    return periods_per_year, _ceil(periods_per_year / 2)


def _print_parameters_table(
    benchmark_title=None,
    periods_per_year=252,
    rf=0.0,
    compounded=True,
    match_dates=True,
):
    width = 40
    print("=" * width)
    print("                 Parameters")
    print("-" * width)
    if benchmark_title:
        print(f"{'Benchmark':<25}{str(benchmark_title).upper():>15}")
    print(f"{'Periods/Year':<25}{periods_per_year:>15}")
    print(f"{'Risk-Free Rate':<25}{rf:>14.1%}")
    print(f"{'Compounded':<25}{'Yes' if compounded else 'No':>15}")
    if benchmark_title:
        print(f"{'Match Dates':<25}{'Yes' if match_dates else 'No':>15}")
    print("=" * width)
    print()


def _match_dates(returns, benchmark):
    if benchmark is None:
        return returns, benchmark
    if isinstance(returns, _pd.DataFrame):
        first = returns.columns[0]
        rstart = returns[first].ne(0).idxmax()
    else:
        rstart = returns.ne(0).idxmax()
    bstart = benchmark.ne(0).idxmax()
    loc = max(rstart, bstart)
    return returns.loc[loc:], benchmark.loc[loc:]


def _is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "GoogleShell")
    except Exception:
        return False


def _call(function, *args, **kwargs):
    try:
        signature = _inspect.signature(function)
        if not any(
            parameter.kind == parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
    except Exception:
        pass
    return function(*args, **kwargs)


def _prepare(returns, rf=0.0, periods_per_year=252, prepare_returns=True):
    if not prepare_returns:
        return returns.copy() if hasattr(returns, "copy") else returns
    util = _get_utils()
    function = getattr(util, "_prepare_returns", None)
    if function is None:
        return returns.dropna()
    return _call(function, returns, rf=rf, nperiods=periods_per_year)


def _prepare_benchmark(benchmark, returns, rf=0.0, periods_per_year=252):
    if benchmark is None:
        return None
    util = _get_utils()
    function = getattr(util, "_prepare_benchmark", None)
    if function is not None:
        try:
            return _call(
                function,
                benchmark,
                returns.index,
                rf=rf,
                prepare_returns=False,
            )
        except Exception:
            try:
                return _call(function, benchmark, returns.index)
            except Exception:
                pass
    if isinstance(benchmark, str):
        try:
            import yfinance as _yf
            data = _yf.download(benchmark, start=returns.index.min(), end=returns.index.max())
            close = data["Close"]
            if isinstance(close, _pd.DataFrame):
                close = close.iloc[:, 0]
            return close.pct_change().dropna()
        except Exception as exc:
            raise ValueError("Unable to prepare benchmark data") from exc
    return _prepare(benchmark, rf, periods_per_year, True)


def _series_name(data, default="Strategy"):
    if isinstance(data, _pd.DataFrame):
        return str(data.columns[0]) if len(data.columns) else default
    return str(data.name) if getattr(data, "name", None) is not None else default


def _as_frame(returns):
    if isinstance(returns, _pd.DataFrame):
        return returns
    return _pd.DataFrame({_series_name(returns): returns})


def _stat(name, returns, default=_np.nan, **kwargs):
    function = getattr(_get_stats(), name, None)
    if function is None:
        return default
    try:
        return _call(function, returns, **kwargs)
    except Exception:
        return default


def _number(value):
    if isinstance(value, (_pd.Series, _pd.DataFrame)):
        if isinstance(value, _pd.DataFrame):
            return value.iloc[-1, 0] if not value.empty else _np.nan
        return value.iloc[-1] if len(value) else _np.nan
    return value


def _format(value, kind="number"):
    if value is None:
        return "-"
    try:
        if _pd.isna(value):
            return "-"
    except Exception:
        pass
    if isinstance(value, (_pd.Timestamp, _dt)):
        return value.strftime("%Y-%m-%d")
    if kind == "date":
        try:
            return _pd.Timestamp(value).strftime("%Y-%m-%d")
        except Exception:
            return str(value)
    try:
        value = float(value)
    except Exception:
        return str(value)
    if kind == "percent":
        return f"{value:.2%}"
    if kind == "integer":
        return f"{value:,.0f}"
    if kind == "ratio":
        return f"{value:.2f}"
    return f"{value:,.2f}"


def _metric_values(series, benchmark, rf, compounded, periods_per_year):
    stats = _get_stats()
    start = series.index.min() if len(series) else _np.nan
    end = series.index.max() if len(series) else _np.nan
    total_return = _stat("comp", series) if compounded else series.sum()
    cagr = _stat(
        "cagr",
        series,
        rf=rf,
        compounded=compounded,
        periods=periods_per_year,
    )
    if _pd.isna(cagr):
        cagr = _stat("cagr", series, rf=rf, compounded=compounded)
    values = {
        "Start Period": (start, "date"),
        "End Period": (end, "date"),
        "Risk-Free Rate": (rf, "percent"),
        "Time in Market": (_stat("exposure", series), "percent"),
        "Cumulative Return": (total_return, "percent"),
        "CAGR﹪": (cagr, "percent"),
        "Sharpe": (
            _stat("sharpe", series, rf=rf, periods=periods_per_year),
            "ratio",
        ),
        "Sortino": (
            _stat("sortino", series, rf=rf, periods=periods_per_year),
            "ratio",
        ),
        "Max Drawdown": (_stat("max_drawdown", series), "percent"),
        "Calmar": (_stat("calmar", series), "ratio"),
        "Volatility (ann.)": (
            _stat("volatility", series, periods=periods_per_year),
            "percent",
        ),
        "Skew": (_stat("skew", series), "ratio"),
        "Kurtosis": (_stat("kurtosis", series), "ratio"),
        "Expected Daily": (_stat("expected_return", series, aggregate="D"), "percent"),
        "Expected Monthly": (_stat("expected_return", series, aggregate="M"), "percent"),
        "Expected Yearly": (_stat("expected_return", series, aggregate="A"), "percent"),
        "Best Day": (_stat("best", series), "percent"),
        "Worst Day": (_stat("worst", series), "percent"),
        "Win Rate": (_stat("win_rate", series), "percent"),
        "Avg. Drawdown": (_stat("avg_drawdown", series), "percent"),
        "Avg. Drawdown Days": (_stat("avg_drawdown_days", series), "number"),
        "Profit Factor": (_stat("profit_factor", series), "ratio"),
        "Common Sense Ratio": (_stat("common_sense_ratio", series), "ratio"),
        "CPC Index": (_stat("cpc_index", series), "ratio"),
        "Tail Ratio": (_stat("tail_ratio", series), "ratio"),
        "Outlier Win Ratio": (_stat("outlier_win_ratio", series), "ratio"),
        "Outlier Loss Ratio": (_stat("outlier_loss_ratio", series), "ratio"),
        "Recovery Factor": (_stat("recovery_factor", series), "ratio"),
        "Ulcer Index": (_stat("ulcer_index", series), "ratio"),
        "Serenity Index": (_stat("serenity_index", series, rf=rf), "ratio"),
        "Risk of Ruin": (_stat("risk_of_ruin", series), "percent"),
    }
    if benchmark is not None:
        values["R^2"] = (_stat("r_squared", series, benchmark), "ratio")
        values["Information Ratio"] = (
            _stat("information_ratio", series, benchmark),
            "ratio",
        )
    return values


def metrics(
    returns,
    benchmark=None,
    rf=0.0,
    display=True,
    mode="basic",
    sep=False,
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    if returns is None:
        raise ValueError("returns cannot be None")
    returns = _prepare(returns, rf, periods_per_year, prepare_returns)
    benchmark = _prepare_benchmark(benchmark, returns, rf, periods_per_year)
    if benchmark is not None and match_dates:
        returns, benchmark = _match_dates(returns, benchmark)

    frame = _as_frame(returns)
    strategy_title = kwargs.get("strategy_title")
    benchmark_title = kwargs.get("benchmark_title", "Benchmark")
    columns = list(frame.columns)
    if strategy_title and len(columns) == 1:
        columns = [strategy_title]

    result = _pd.DataFrame(index=[], columns=columns)
    for original, label in zip(frame.columns, columns):
        values = _metric_values(
            frame[original].dropna(),
            benchmark,
            rf,
            compounded,
            periods_per_year,
        )
        if str(mode).lower() in ("basic", "summary"):
            wanted = [
                "Start Period", "End Period", "Risk-Free Rate", "Time in Market",
                "Cumulative Return", "CAGR﹪", "Sharpe", "Sortino",
                "Max Drawdown", "Calmar", "Volatility (ann.)",
            ]
        else:
            wanted = list(values)
        result[label] = _pd.Series(
            {key: _format(values[key][0], values[key][1]) for key in wanted}
        )

    if benchmark is not None:
        values = _metric_values(
            benchmark.dropna(), None, rf, compounded, periods_per_year
        )
        if str(mode).lower() in ("basic", "summary"):
            wanted = [
                "Start Period", "End Period", "Risk-Free Rate", "Time in Market",
                "Cumulative Return", "CAGR﹪", "Sharpe", "Sortino",
                "Max Drawdown", "Calmar", "Volatility (ann.)",
            ]
        else:
            wanted = list(values)
        result[benchmark_title] = _pd.Series(
            {key: _format(values[key][0], values[key][1]) for key in wanted}
        )

    if display:
        print(_tabulate(result, headers="keys", tablefmt="simple"))
        if sep:
            print()
        return None
    return result


def _run_plot(name, returns, benchmark=None, **kwargs):
    function = getattr(_get_plots(), name, None)
    if function is None:
        return None
    return _call(function, returns, benchmark=benchmark, **kwargs)


def plots(
    returns,
    benchmark=None,
    grayscale=False,
    figsize=(8, 5),
    mode="basic",
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    returns = _prepare(returns, 0.0, periods_per_year, prepare_returns)
    benchmark = _prepare_benchmark(benchmark, returns, 0.0, periods_per_year)
    if benchmark is not None and match_dates:
        returns, benchmark = _match_dates(returns, benchmark)

    plot_kwargs = {
        "grayscale": grayscale,
        "figsize": figsize,
        "compounded": compounded,
        "periods_per_year": periods_per_year,
        "show": True,
        **kwargs,
    }
    names = ["snapshot", "monthly_heatmap", "drawdown", "rolling_sharpe"]
    if str(mode).lower() not in ("basic", "summary"):
        names.extend([
            "returns", "log_returns", "yearly_returns", "histogram",
            "daily_returns", "rolling_volatility", "rolling_beta",
            "rolling_sortino", "rolling_var", "distribution",
        ])
    output = []
    for name in names:
        try:
            output.append(_run_plot(name, returns, benchmark, **plot_kwargs))
        except Exception:
            continue
    return output[-1] if len(output) == 1 else output


def basic(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    figsize=(8, 5),
    display=True,
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    benchmark_title = kwargs.get("benchmark_title")
    _print_parameters_table(
        benchmark_title=benchmark_title if benchmark is not None else None,
        periods_per_year=periods_per_year,
        rf=rf,
        compounded=compounded,
        match_dates=match_dates,
    )
    table = metrics(
        returns,
        benchmark=benchmark,
        rf=rf,
        display=display,
        mode="basic",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=prepare_returns,
        match_dates=match_dates,
        **kwargs,
    )
    plots(
        returns,
        benchmark=benchmark,
        grayscale=grayscale,
        figsize=figsize,
        mode="basic",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=prepare_returns,
        match_dates=match_dates,
        **kwargs,
    )
    return table


def full(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    figsize=(8, 5),
    display=True,
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    benchmark_title = kwargs.get("benchmark_title")
    _print_parameters_table(
        benchmark_title=benchmark_title if benchmark is not None else None,
        periods_per_year=periods_per_year,
        rf=rf,
        compounded=compounded,
        match_dates=match_dates,
    )
    table = metrics(
        returns,
        benchmark=benchmark,
        rf=rf,
        display=display,
        mode="full",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=prepare_returns,
        match_dates=match_dates,
        **kwargs,
    )
    plots(
        returns,
        benchmark=benchmark,
        grayscale=grayscale,
        figsize=figsize,
        mode="full",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=prepare_returns,
        match_dates=match_dates,
        **kwargs,
    )
    return table


def _figure_data_uri(figure, figfmt="svg"):
    if figure is None:
        return ""
    stream = _io.BytesIO()
    try:
        figure.savefig(stream, format=figfmt, bbox_inches="tight")
    except Exception:
        return ""
    data = stream.getvalue()
    mime = "image/svg+xml" if figfmt.lower() == "svg" else f"image/{figfmt}"
    return f"data:{mime};base64,{_base64.b64encode(data).decode('ascii')}"


def html(
    returns,
    benchmark=None,
    rf=0.0,
    grayscale=False,
    title="Strategy Tearsheet",
    output=None,
    compounded=True,
    periods_per_year=252,
    download_filename="quantstats-tearsheet.html",
    figfmt="svg",
    template_path=None,
    match_dates=True,
    **kwargs,
):
    if match_dates:
        returns = returns.dropna()

    if template_path is None:
        template_path = _Path(__file__).parent / "report.html"
    else:
        template_path = _Path(template_path)
    template_path = template_path.resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not template_path.is_file():
        raise ValueError(f"Template path is not a file: {template_path}")

    template = template_path.read_text(encoding="utf-8")
    prepared = _prepare(returns, rf, periods_per_year, True)
    prepared_benchmark = _prepare_benchmark(
        benchmark, prepared, rf, periods_per_year
    )
    if prepared_benchmark is not None and match_dates:
        prepared, prepared_benchmark = _match_dates(prepared, prepared_benchmark)

    table = metrics(
        prepared,
        benchmark=prepared_benchmark,
        rf=rf,
        display=False,
        mode="full",
        compounded=compounded,
        periods_per_year=periods_per_year,
        prepare_returns=False,
        match_dates=False,
        **kwargs,
    )
    metrics_html = table.to_html(classes="metrics", border=0)

    strategy_title = kwargs.get("strategy_title", "Strategy")
    benchmark_title = kwargs.get("benchmark_title", "Benchmark")
    replacement = {
        "{{title}}": str(title),
        "{{metrics}}": metrics_html,
        "{{metrics_table}}": metrics_html,
        "{{strategy_title}}": str(strategy_title),
        "{{benchmark_title}}": str(benchmark_title),
        "{{version}}": str(__version__),
        "{{date}}": _dt.now().strftime("%Y-%m-%d"),
    }
    document = template
    for token, value in replacement.items():
        document = document.replace(token, value)

    if metrics_html not in document:
        body = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            "<style>body{font-family:Arial;margin:32px}.metrics{border-collapse:collapse}"
            ".metrics td,.metrics th{padding:6px 12px;border-bottom:1px solid #ddd}"
            "</style></head><body>"
            f"<h1>{title}</h1>{metrics_html}</body></html>"
        )
        document = body

    if output is not None:
        output_path = _Path(output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")
        return None

    handle = _tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        prefix="quantstats-",
        delete=False,
        encoding="utf-8",
    )
    try:
        handle.write(document)
        handle.close()
        _webbrowser.open("file://" + _os.path.abspath(handle.name))
    except Exception:
        try:
            handle.close()
        except Exception:
            pass
    return None