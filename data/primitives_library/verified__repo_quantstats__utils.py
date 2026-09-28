import io as _io
import datetime as _dt
import inspect
import threading

import numpy as _np
import pandas as _pd

from ._compat import safe_concat, safe_resample

try:
    from ._compat import safe_yfinance_download
except ImportError:
    def safe_yfinance_download(*args, **kwargs):
        try:
            import yfinance as _yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required to download benchmark data"
            ) from exc
        return _yf.download(*args, **kwargs)


Returns = _pd.Series | _pd.DataFrame


class QuantStatsError(Exception):
    pass


class DataValidationError(QuantStatsError):
    pass


class CalculationError(QuantStatsError):
    pass


class PlottingError(QuantStatsError):
    pass


class BenchmarkError(QuantStatsError):
    pass


def validate_input(data, allow_empty=False):
    if data is None:
        raise DataValidationError("Input data cannot be None")

    if not isinstance(data, (_pd.Series, _pd.DataFrame)):
        raise DataValidationError(
            f"Input data must be pandas Series or DataFrame, got {type(data)}"
        )

    if not allow_empty and len(data) == 0:
        raise DataValidationError("Input data cannot be empty")

    if not allow_empty and data.dropna().empty:
        raise DataValidationError("Input data contains only NaN values")

    if not isinstance(data.index, (_pd.DatetimeIndex, _pd.RangeIndex)):
        try:
            data.index = _pd.to_datetime(data.index)
        except Exception as exc:
            raise DataValidationError(
                "Input data must have a valid datetime index"
            ) from exc

    return True


_PREPARE_RETURNS_CACHE = {}
_CACHE_MAX_SIZE = 100
_cache_lock = threading.Lock()


def _generate_cache_key(data, rf, nperiods):
    try:
        if isinstance(data, _pd.Series):
            data_hash = _pd.util.hash_pandas_object(data).sum()
        elif isinstance(data, _pd.DataFrame):
            data_hash = _pd.util.hash_pandas_object(data).sum()
        else:
            data_hash = hash(str(data))
        return f"{data_hash}_{rf}_{nperiods}"
    except (ValueError, TypeError, AttributeError, MemoryError):
        return None


def _clear_cache_if_full():
    with _cache_lock:
        if len(_PREPARE_RETURNS_CACHE) >= _CACHE_MAX_SIZE:
            keys = list(_PREPARE_RETURNS_CACHE.keys())[
                :-(_CACHE_MAX_SIZE // 2)
            ]
            for key in keys:
                del _PREPARE_RETURNS_CACHE[key]


def _mtd(df):
    return df[df.index >= _dt.datetime.now().strftime("%Y-%m-01")]


def _qtd(df):
    date = _dt.datetime.now()
    for q in [1, 4, 7, 10]:
        if date.month <= q:
            return df[
                df.index
                >= _dt.datetime(date.year, q, 1).strftime("%Y-%m-01")
            ]
    return df[df.index >= date.strftime("%Y-%m-01")]


def _ytd(df):
    return df[df.index >= _dt.datetime.now().strftime("%Y-01-01")]


def _pandas_date(df, dates):
    if not isinstance(dates, list):
        dates = [dates]
    return df[df.index.isin(dates)]


def _pandas_current_month(df):
    now = _dt.datetime.now()
    daterange = _pd.date_range(_dt.date(now.year, now.month, 1), now)
    return df[df.index.isin(daterange)]


def multi_shift(df, shift=3):
    if isinstance(df, _pd.Series):
        df = _pd.DataFrame(df)

    result = df.copy()
    for i in range(1, shift):
        shifted = df.shift(i)
        shifted.columns = [f"{column}{i}" for column in shifted.columns]
        result = safe_concat([result, shifted], axis=1, sort=True)
    return result


def _is_price_data(data):
    try:
        if isinstance(data, _pd.DataFrame):
            for column in data.columns:
                values = data[column].dropna()
                if len(values) and values.min() >= 0 and values.max() > 1:
                    return True
            return False
        values = data.dropna()
        return bool(len(values) and values.min() >= 0 and values.max() > 1)
    except (TypeError, ValueError):
        return False


def to_excess_returns(returns, rf, nperiods=None):
    if nperiods is not None:
        rf = (1 + rf) ** (1.0 / nperiods) - 1
    return returns - rf


def _prepare_returns(data, rf=0.0, nperiods=None):
    cache_key = _generate_cache_key(data, rf, nperiods)

    if cache_key is not None:
        with _cache_lock:
            cached = _PREPARE_RETURNS_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()

    try:
        result = data.copy()

        if isinstance(result, _pd.DataFrame):
            for column in result.columns:
                values = result[column].dropna()
                if len(values) and values.min() >= 0 and values.max() > 1:
                    result[column] = result[column].pct_change()
        elif isinstance(result, _pd.Series):
            values = result.dropna()
            if len(values) and values.min() >= 0 and values.max() > 1:
                result = result.pct_change()

        result = result.fillna(0).replace([_np.inf, -_np.inf], _np.nan)

        if rf != 0:
            result = to_excess_returns(result, rf, nperiods)

    except Exception as exc:
        raise CalculationError(f"Failed to prepare returns: {exc}") from exc

    if cache_key is not None:
        _clear_cache_if_full()
        with _cache_lock:
            _PREPARE_RETURNS_CACHE[cache_key] = result.copy()

    return result


def to_returns(prices: Returns, rf: float = 0.0) -> Returns:
    return _prepare_returns(prices, rf)


def to_prices(returns: Returns, base: float = 1e5) -> Returns:
    from . import stats as _stats

    returns = returns.copy().fillna(0).replace([_np.inf, -_np.inf], float("NaN"))
    return base + base * _stats.compsum(returns)


def log_returns(returns: Returns, rf: float = 0.0,
                nperiods: int | None = None) -> Returns:
    return to_log_returns(returns, rf, nperiods)


def to_log_returns(prices: Returns, rf: float = 0.0,
                   nperiods: int | None = None) -> Returns:
    returns = _prepare_returns(prices, rf, nperiods)
    return _np.log(returns + 1).replace([_np.inf, -_np.inf], float("NaN"))


def exp_returns(returns: Returns, rf: float = 0.0,
                nperiods: int | None = None) -> Returns:
    returns = _prepare_returns(returns, rf, nperiods)
    return _np.exp(returns) - 1


def group_returns(returns, groupby, compounded=True):
    if compounded:
        return (returns + 1).groupby(groupby).prod() - 1
    return returns.groupby(groupby).sum()


def aggregate_returns(returns, period=None, compounded=True):
    if period is None:
        return returns

    if not isinstance(returns.index, _pd.DatetimeIndex):
        return returns

    period = str(period).lower()
    index = returns.index

    if period in ("d", "day", "daily"):
        groupby = [index.year, index.month, index.day]
    elif period in ("w", "week", "weekly"):
        iso = index.isocalendar()
        groupby = [iso.year, iso.week]
    elif period in ("m", "month", "monthly"):
        groupby = [index.year, index.month]
    elif period in ("q", "quarter", "quarterly"):
        groupby = [index.year, index.quarter]
    elif period in ("a", "y", "year", "yearly", "annual"):
        groupby = index.year
    elif period in ("eow", "week_end", "weekend"):
        return group_returns(returns, index.to_period("W"), compounded)
    elif period in ("eom", "month_end", "monthend"):
        return group_returns(returns, index.to_period("M"), compounded)
    elif period in ("eoq", "quarter_end", "quarterend"):
        return group_returns(returns, index.to_period("Q"), compounded)
    elif period in ("eoy", "year_end", "yearend"):
        return group_returns(returns, index.to_period("Y"), compounded)
    else:
        return group_returns(returns, period, compounded)

    return group_returns(returns, groupby, compounded)


def _prepare_benchmark(benchmark=None, period="max", rf=0.0,
                       prepare_returns=True):
    if benchmark is None:
        return None

    try:
        if isinstance(benchmark, str):
            benchmark = download_returns(benchmark, period)
            prepare_returns = False
        else:
            benchmark = benchmark.copy()

        if isinstance(benchmark, _pd.DataFrame):
            if benchmark.shape[1] == 0:
                raise BenchmarkError("Benchmark dataframe contains no columns")
            benchmark = benchmark.iloc[:, 0]

        if isinstance(period, _pd.DatetimeIndex):
            benchmark.index = _pd.to_datetime(benchmark.index)
            benchmark = benchmark[benchmark.index.isin(period)]
            benchmark = benchmark.reindex(period, method="bfill")

        if prepare_returns:
            benchmark = _prepare_returns(benchmark, rf)

        return benchmark
    except BenchmarkError:
        raise
    except Exception as exc:
        raise BenchmarkError(f"Failed to prepare benchmark: {exc}") from exc


def download_returns(ticker, period=None):
    if period is None:
        period = "max"

    try:
        if isinstance(period, _pd.DatetimeIndex):
            if len(period) == 0:
                return _pd.Series(dtype=float, name=ticker)
            data = safe_yfinance_download(
                ticker,
                start=period[0],
                end=period[-1] + _pd.Timedelta(days=1),
                progress=False,
            )
        else:
            data = safe_yfinance_download(ticker, period=period, progress=False)

        if isinstance(data, _pd.DataFrame):
            if "Close" in data.columns:
                data = data["Close"]
            elif "Adj Close" in data.columns:
                data = data["Adj Close"]
            elif data.shape[1]:
                data = data.iloc[:, 0]

        if isinstance(data, _pd.DataFrame) and data.shape[1] == 1:
            data = data.iloc[:, 0]

        return _prepare_returns(data)
    except Exception as exc:
        raise BenchmarkError(f"Failed to download benchmark '{ticker}': {exc}") from exc


def _round_to_closest(value, res, decimals=None):
    result = round(round(value / res) * res, decimals)
    return result


def _score_str(val):
    if val is None or _pd.isna(val):
        return ""
    if val > 0:
        return "+"
    if val < 0:
        return "-"
    return ""


def _in_notebook(matplotlib_inline=True):
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            if not matplotlib_inline:
                return True
            try:
                import matplotlib

                backend = matplotlib.get_backend().lower()
                return "inline" in backend or "widget" in backend
            except Exception:
                return True
        return False
    except (NameError, AttributeError):
        return False


def _get_colors(grayscale=False):
    if grayscale:
        return [
            "#000000",
            "#4D4D4D",
            "#7F7F7F",
            "#A6A6A6",
            "#C0C0C0",
            "#D9D9D9",
        ]
    return [
        "#348DC1",
        "#F39C12",
        "#2ECC71",
        "#E74C3C",
        "#9B59B6",
        "#95A5A6",
    ]


def _setup_plotly():
    try:
        import plotly.io as pio

        if _in_notebook():
            pio.renderers.default = "notebook_connected"
        return pio
    except Exception as exc:
        raise PlottingError(f"Unable to configure plotly: {exc}") from exc


def _embed_figure(figure, title="", width=None, height=None):
    try:
        if hasattr(figure, "to_html"):
            return figure.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                config={"displayModeBar": False},
            )

        stream = _io.StringIO()
        figure.savefig(stream, format="svg", bbox_inches="tight")
        svg = stream.getvalue()
        stream.close()

        style = []
        if width is not None:
            style.append(f"width:{width}px")
        if height is not None:
            style.append(f"height:{height}px")
        style = f' style="{";".join(style)}"' if style else ""
        heading = f"<h3>{title}</h3>" if title else ""
        return f"{heading}<div{style}>{svg}</div>"
    except Exception as exc:
        raise PlottingError(f"Unable to embed figure: {exc}") from exc


def _save_html(fig, file, title=""):
    try:
        if hasattr(fig, "write_html"):
            fig.write_html(file, include_plotlyjs="cdn", full_html=True)
            return

        html = _embed_figure(fig, title)
        with open(file, "w", encoding="utf-8") as output:
            output.write(
                "<!DOCTYPE html><html><head>"
                f"<title>{title}</title>"
                "</head><body>"
                f"{html}</body></html>"
            )
    except Exception as exc:
        raise PlottingError(f"Unable to save HTML file: {exc}") from exc


def _count_consecutive(data):
    groups = (data != data.shift()).cumsum()
    return data * (data.groupby(groups).cumcount() + 1)


def _file_stream():
    return _io.BytesIO()


def _flatten_dataframe(df, set_index=None):
    result = df.copy()

    if isinstance(result.columns, _pd.MultiIndex):
        result.columns = [
            " ".join(str(item) for item in column if str(item) != "")
            for column in result.columns
        ]

    if set_index is not None:
        result.set_index(set_index, inplace=True)

    return result


def make_index(ticker, rebalance="1M", period="max", returns=None,
               match_dates=False):
    if isinstance(ticker, str):
        ticker = [ticker]

    if returns is None:
        downloaded = []
        for symbol in ticker:
            series = download_returns(symbol, period)
            if isinstance(series, _pd.DataFrame):
                series = series.iloc[:, 0]
            series.name = symbol
            downloaded.append(series)
        returns = safe_concat(downloaded, axis=1, sort=True)
    else:
        returns = returns.copy()
        if isinstance(returns, _pd.Series):
            returns = _pd.DataFrame(returns)

    if match_dates:
        returns = returns.dropna()

    returns = returns.fillna(0)
    if returns.empty:
        return _pd.Series(dtype=float)

    weights = _pd.DataFrame(
        1.0 / returns.shape[1],
        index=returns.index,
        columns=returns.columns,
    )

    if rebalance is not None:
        rebalance_dates = returns.resample(rebalance).first().index
        scheduled = weights.reindex(rebalance_dates)
        weights = scheduled.reindex(returns.index, method="ffill").fillna(
            1.0 / returns.shape[1]
        )

    result = (returns * weights).sum(axis=1)
    result.name = "index"
    return result


def get_trading_periods(periods_per_year=252):
    return periods_per_year, int(periods_per_year / 2)