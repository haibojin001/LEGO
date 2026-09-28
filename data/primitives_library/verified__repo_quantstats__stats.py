from warnings import warn
from typing import Literal
import numpy as _np
import pandas as _pd
from math import ceil as _ceil, sqrt as _sqrt
from scipy.stats import norm as _norm, linregress as _linregress

from . import utils as _utils
try:
    from ._compat import safe_concat
except Exception:
    def safe_concat(objs, **kwargs):
        return _pd.concat(objs, **kwargs)

try:
    from .utils import validate_input
except Exception:
    def validate_input(x, *args, **kwargs):
        return x

Returns = _pd.Series | _pd.DataFrame


def _prepare(x, rf=0.0, nperiods=None):
    try:
        return _utils._prepare_returns(x, rf, nperiods)
    except TypeError:
        try:
            return _utils._prepare_returns(x, rf)
        except Exception:
            return x
    except Exception:
        return x


def _aggregate(x, period=None, compounded=True):
    if period is None:
        return x
    try:
        return _utils.aggregate_returns(x, period, compounded)
    except Exception:
        rule = {
            "D": "D", "W": "W", "M": "ME", "Q": "QE", "Y": "YE",
            "A": "YE", "weekly": "W", "monthly": "ME",
            "quarterly": "QE", "yearly": "YE",
        }.get(str(period), period)
        return x.resample(rule).apply(comp if compounded else _np.sum)


def pct_rank(prices: _pd.Series, window: int = 60) -> _pd.Series:
    try:
        return _utils.multi_shift(prices, window).T.rank(pct=True).T.iloc[:, 0] * 100.0
    except Exception:
        return prices.rolling(window).rank(pct=True) * 100.0


def compsum(returns: Returns) -> Returns:
    return returns.add(1).cumprod(axis=0) - 1


def comp(returns: Returns) -> _pd.Series | float:
    return returns.add(1).prod(axis=0) - 1


def distribution(returns: Returns, compounded: bool = True,
                 prepare_returns: bool = True) -> dict:
    def clean(data):
        q1, q3 = data.quantile(.25), data.quantile(.75)
        iqr = q3 - q1
        keep = (data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)
        return {"values": data.loc[keep].tolist(), "outliers": data.loc[~keep].tolist()}

    if isinstance(returns, _pd.DataFrame):
        warn("Pandas DataFrame was passed (Series expected). Only first column will be used.")
        x = returns.copy()
        x.columns = [str(c).lower() for c in x.columns]
        returns = x["close"] if len(x.columns) > 1 and "close" in x.columns else x.iloc[:, 0]
    daily = returns.dropna()
    if prepare_returns:
        daily = _prepare(daily)
    func = comp if compounded else _np.sum
    return {
        "Daily": clean(daily),
        "Weekly": clean(daily.resample("W-MON").apply(func)),
        "Monthly": clean(daily.resample("ME").apply(func)),
        "Quarterly": clean(daily.resample("QE").apply(func)),
        "Yearly": clean(daily.resample("YE").apply(func)),
    }


def expected_return(returns: Returns, aggregate: str | None = None,
                    compounded: bool = True, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    returns = _aggregate(returns, aggregate, compounded)
    return _np.prod(1 + returns, axis=0) ** (1 / len(returns)) - 1


def geometric_mean(returns: Returns, aggregate: str | None = None,
                   compounded: bool = True):
    return expected_return(returns, aggregate, compounded)


def ghpr(returns: Returns, aggregate: str | None = None,
         compounded: bool = True):
    return expected_return(returns, aggregate, compounded)


def outliers(returns: Returns, quantile: float = .95):
    returns = _prepare(returns)
    return returns[returns > returns.quantile(quantile)]


def remove_outliers(returns: Returns, quantile: float = .95):
    returns = _prepare(returns)
    return returns[returns <= returns.quantile(quantile)]


def best(returns: Returns, aggregate: str | None = None,
         compounded: bool = True, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    returns = _aggregate(returns, aggregate, compounded)
    return returns.max()


def worst(returns: Returns, aggregate: str | None = None,
          compounded: bool = True, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    returns = _aggregate(returns, aggregate, compounded)
    return returns.min()


def consecutive_wins(returns: Returns, aggregate: str | None = None,
                     compounded: bool = True, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    returns = _aggregate(returns, aggregate, compounded)
    if isinstance(returns, _pd.DataFrame):
        return returns.apply(lambda x: consecutive_wins(x, prepare_returns=False))
    groups = (returns <= 0).cumsum()
    return returns[returns > 0].groupby(groups).count().max()


def consecutive_losses(returns: Returns, aggregate: str | None = None,
                       compounded: bool = True, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    returns = _aggregate(returns, aggregate, compounded)
    if isinstance(returns, _pd.DataFrame):
        return returns.apply(lambda x: consecutive_losses(x, prepare_returns=False))
    groups = (returns >= 0).cumsum()
    return returns[returns < 0].groupby(groups).count().max()


def exposure(returns: Returns, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return (returns != 0).sum() / len(returns)


def win_rate(returns: Returns, aggregate: str | None = None,
             compounded: bool = True, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    returns = _aggregate(returns, aggregate, compounded)
    nonzero = returns[returns != 0]
    return (nonzero > 0).sum() / len(nonzero)


def avg_return(returns: Returns, aggregate: str | None = None,
               compounded: bool = True, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return _aggregate(returns, aggregate, compounded).mean()


def avg_win(returns: Returns, aggregate: str | None = None,
            compounded: bool = True, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    returns = _aggregate(returns, aggregate, compounded)
    return returns[returns > 0].mean()


def avg_loss(returns: Returns, aggregate: str | None = None,
             compounded: bool = True, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    returns = _aggregate(returns, aggregate, compounded)
    return returns[returns < 0].mean()


def volatility(returns: Returns, periods: int = 252, annualize: bool = True,
               prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    v = returns.std(ddof=1)
    return v * _sqrt(periods) if annualize else v


def rolling_volatility(returns: Returns, rolling_period: int = 126,
                       periods_per_year: int = 252, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return returns.rolling(rolling_period).std() * _sqrt(periods_per_year)


def implied_volatility(returns: Returns, periods: int = 252,
                       annualize: bool = True):
    returns = _prepare(returns)
    v = _np.log(1 + returns).rolling(periods).std()
    return v * _sqrt(periods) if annualize else v


def autocorr_penalty(returns: Returns, prepare_returns: bool = False):
    if prepare_returns:
        returns = _prepare(returns)
    if isinstance(returns, _pd.DataFrame):
        return returns.apply(lambda x: autocorr_penalty(x, False))
    n = len(returns)
    if n < 2:
        return 1.0
    coef = returns.autocorr(lag=1)
    if _pd.isna(coef):
        return 1.0
    return _sqrt(1 + 2 * sum((1 - i / n) * coef ** i for i in range(1, n)))


def sharpe(returns: Returns, rf: float = 0., periods: int = 252,
           annualize: bool = True, smart: bool = False):
    returns = _prepare(returns, rf, periods)
    divisor = returns.std(ddof=1)
    if smart:
        divisor *= autocorr_penalty(returns)
    result = returns.mean() / divisor
    return result * _sqrt(periods) if annualize else result


def smart_sharpe(returns: Returns, rf: float = 0., periods: int = 252,
                 annualize: bool = True):
    return sharpe(returns, rf, periods, annualize, True)


def rolling_sharpe(returns: Returns, rf: float = 0., rolling_period: int = 126,
                   annualize: bool = True, periods_per_year: int = 252,
                   prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns, rf, periods_per_year)
    result = returns.rolling(rolling_period).mean() / returns.rolling(rolling_period).std()
    return result * _sqrt(periods_per_year) if annualize else result


def sortino(returns: Returns, rf: float = 0., periods: int = 252,
            annualize: bool = True, smart: bool = False):
    returns = _prepare(returns, rf, periods)
    downside = returns[returns < 0].std(ddof=1)
    result = returns.mean() / downside
    if smart:
        result = result / autocorr_penalty(returns)
    return result * _sqrt(periods) if annualize else result


def smart_sortino(returns: Returns, rf: float = 0., periods: int = 252,
                  annualize: bool = True):
    return sortino(returns, rf, periods, annualize, True)


def rolling_sortino(returns: Returns, rf: float = 0., rolling_period: int = 126,
                    annualize: bool = True, periods_per_year: int = 252,
                    prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns, rf, periods_per_year)
    downside = returns.where(returns < 0, 0).rolling(rolling_period).std()
    result = returns.rolling(rolling_period).mean() / downside
    return result * _sqrt(periods_per_year) if annualize else result


def adjusted_sortino(returns: Returns, rf: float = 0., periods: int = 252,
                     annualize: bool = True, smart: bool = False):
    return sortino(returns, rf, periods, annualize, smart) / _sqrt(2)


def probabilistic_sharpe_ratio(returns: Returns, rf: float = 0.,
                               periods: int = 252, annualize: bool = False,
                               smart: bool = False):
    returns = _prepare(returns, rf, periods)
    sr = sharpe(returns, 0, periods, annualize, smart)
    n = len(returns)
    skew = returns.skew()
    kurt = returns.kurtosis()
    denom = _sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr ** 2)
    return _norm.cdf(sr * _sqrt(n - 1) / denom)


def probabilistic_sortino_ratio(returns: Returns, rf: float = 0.,
                                periods: int = 252, annualize: bool = False,
                                smart: bool = False):
    returns = _prepare(returns, rf, periods)
    sr = sortino(returns, 0, periods, annualize, smart)
    n = len(returns)
    skew = returns.skew()
    kurt = returns.kurtosis()
    denom = _sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr ** 2)
    return _norm.cdf(sr * _sqrt(n - 1) / denom)


def omega(returns: Returns, rf: float = 0., required_return: float = 0.,
          periods: int = 252):
    returns = _prepare(returns, rf, periods)
    threshold = (1 + required_return) ** (1 / periods) - 1
    numer = (returns - threshold)[returns > threshold].sum()
    denom = -(returns - threshold)[returns < threshold].sum()
    return numer / denom


def gain_to_pain_ratio(returns: Returns, rf: float = 0.,
                       resolution: Literal["D", "W", "M", "Q", "Y"] = "D"):
    returns = _prepare(returns, rf)
    returns = _aggregate(returns, resolution, False)
    return returns.sum() / abs(returns[returns < 0].sum())


def cagr(returns: Returns, rf: float = 0., compounded: bool = True,
         periods: int = 252):
    returns = _prepare(returns, rf)
    if len(returns) == 0:
        return _np.nan
    total = comp(returns) if compounded else returns.sum()
    years = len(returns) / periods
    return (1 + total) ** (1 / years) - 1


def rar(returns: Returns, rf: float = 0., compounded: bool = True,
        periods: int = 252):
    return cagr(returns, rf, compounded, periods) / exposure(returns)


def risk_return_ratio(returns: Returns, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return returns.mean() / returns.std()


def to_drawdown_series(returns: Returns):
    returns = _prepare(returns)
    if isinstance(returns, _pd.DataFrame):
        return returns.apply(to_drawdown_series)
    wealth = (1 + returns).cumprod()
    highwater = wealth.cummax()
    dd = wealth / highwater - 1
    return dd.replace([_np.inf, -_np.inf, -0], 0)


def max_drawdown(prices: Returns):
    if isinstance(prices, _pd.DataFrame):
        return prices.apply(max_drawdown)
    prices = _pd.Series(prices).dropna()
    if len(prices) == 0:
        return _np.nan
    if prices.min() <= 0 or prices.max() <= 2:
        return to_drawdown_series(prices).min()
    return (prices / prices.cummax() - 1).min()


def calmar(returns: Returns, rf: float = 0., periods: int = 252):
    return cagr(returns, rf, periods=periods) / abs(max_drawdown(returns))


def sterling(returns: Returns, rf: float = 0., periods: int = 252,
             annualize: bool = True):
    returns = _prepare(returns, rf, periods)
    dd = to_drawdown_series(returns)
    avg_dd = abs(dd[dd < 0].mean())
    value = returns.mean() / avg_dd
    return value * _sqrt(periods) if annualize else value


def ulcer_index(returns: Returns):
    dd = to_drawdown_series(returns)
    return _sqrt((dd.pow(2)).mean())


def ulcer_performance_index(returns: Returns, rf: float = 0.,
                            periods: int = 252):
    returns = _prepare(returns, rf, periods)
    return (returns.mean() * periods) / ulcer_index(returns)


def serenity_index(returns: Returns, rf: float = 0.):
    returns = _prepare(returns, rf)
    dd = to_drawdown_series(returns)
    pitfall = _np.sqrt((dd[dd < 0] ** 2).mean())
    return cagr(returns) / (pitfall * (1 + abs(dd.min())))


def risk_of_ruin(returns: Returns, prepare_returns: bool = True):
    return (1 - win_rate(returns, prepare_returns=prepare_returns)) ** len(returns)


def r_squared(returns: Returns, benchmark: Returns, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
        benchmark = _prepare(benchmark)
    data = safe_concat([returns, benchmark], axis=1).dropna()
    if len(data) < 2:
        return _np.nan
    return _linregress(data.iloc[:, 1], data.iloc[:, 0]).rvalue ** 2


def information_ratio(returns: Returns, benchmark: Returns,
                      prepare_returns: bool = True):
    if prepare_returns:
        returns, benchmark = _prepare(returns), _prepare(benchmark)
    active = returns - benchmark
    return active.mean() / active.std()


def greeks(returns: Returns, benchmark: Returns, periods: int = 252,
           prepare_returns: bool = True):
    if prepare_returns:
        returns, benchmark = _prepare(returns), _prepare(benchmark)
    data = safe_concat([returns, benchmark], axis=1).dropna()
    data.columns = ["returns", "benchmark"]
    beta_ = data["returns"].cov(data["benchmark"]) / data["benchmark"].var()
    alpha_ = (data["returns"].mean() - beta_ * data["benchmark"].mean()) * periods
    return _pd.Series({"beta": beta_, "alpha": alpha_})


def rolling_greeks(returns: Returns, benchmark: Returns, periods: int = 252,
                   prepare_returns: bool = True):
    if prepare_returns:
        returns, benchmark = _prepare(returns), _prepare(benchmark)
    beta_ = returns.rolling(periods).cov(benchmark) / benchmark.rolling(periods).var()
    alpha_ = (returns.rolling(periods).mean() - beta_ * benchmark.rolling(periods).mean()) * periods
    return _pd.DataFrame({"beta": beta_, "alpha": alpha_})


def beta(returns: Returns, benchmark: Returns, prepare_returns: bool = True):
    return greeks(returns, benchmark, prepare_returns=prepare_returns)["beta"]


def alpha(returns: Returns, benchmark: Returns, rf: float = 0.,
          periods: int = 252, prepare_returns: bool = True):
    if rf:
        returns = _prepare(returns, rf, periods)
        benchmark = _prepare(benchmark, rf, periods)
        prepare_returns = False
    return greeks(returns, benchmark, periods, prepare_returns)["alpha"]


def rolling_beta(returns: Returns, benchmark: Returns, periods: int = 252,
                 prepare_returns: bool = True):
    return rolling_greeks(returns, benchmark, periods, prepare_returns)["beta"]


def rolling_alpha(returns: Returns, benchmark: Returns, periods: int = 252,
                  prepare_returns: bool = True):
    return rolling_greeks(returns, benchmark, periods, prepare_returns)["alpha"]


def correlation(returns: Returns, benchmark: Returns,
                prepare_returns: bool = True):
    if prepare_returns:
        returns, benchmark = _prepare(returns), _prepare(benchmark)
    return returns.corr(benchmark)


def treynor_ratio(returns: Returns, benchmark: Returns, rf: float = 0.,
                  periods: int = 252):
    returns = _prepare(returns, rf, periods)
    benchmark = _prepare(benchmark, rf, periods)
    return returns.mean() * periods / beta(returns, benchmark, False)


def compare(returns: Returns, benchmark: Returns, aggregate: str | None = None,
            compounded: bool = True, round_vals: bool = False,
            prepare_returns: bool = True):
    if prepare_returns:
        returns, benchmark = _prepare(returns), _prepare(benchmark)
    data = safe_concat([returns, benchmark], axis=1)
    data.columns = ["Returns", "Benchmark"]
    if aggregate:
        data = _aggregate(data, aggregate, compounded)
    if round_vals:
        data = data.round(2)
    return data


def monthly_returns(returns: Returns, eoy: bool = True,
                    compounded: bool = True, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    if isinstance(returns, _pd.DataFrame):
        return returns.apply(lambda x: monthly_returns(x, eoy, compounded, False))
    monthly = returns.resample("ME").apply(comp if compounded else _np.sum)
    table = monthly.to_frame("returns")
    table["year"] = table.index.year
    table["month"] = table.index.month
    result = table.pivot(index="year", columns="month", values="returns")
    result = result.reindex(columns=range(1, 13))
    result.columns = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                      "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    if eoy:
        yearly = returns.resample("YE").apply(comp if compounded else _np.sum)
        result["EOY"] = yearly
    return result


def kelly_criterion(returns: Returns, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    win = win_rate(returns, prepare_returns=False)
    aw = avg_win(returns, prepare_returns=False)
    al = abs(avg_loss(returns, prepare_returns=False))
    return win - ((1 - win) / (aw / al))


def tail_ratio(returns: Returns, cutoff: float = .95,
               prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return abs(returns.quantile(cutoff) / returns.quantile(1 - cutoff))


def common_sense_ratio(returns: Returns, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return profit_factor(returns, prepare_returns=False) * tail_ratio(returns, prepare_returns=False)


def cpc_index(returns: Returns, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return profit_factor(returns, False) * win_rate(returns, prepare_returns=False) * payoff_ratio(returns, prepare_returns=False)


def profit_factor(returns: Returns, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return returns[returns > 0].sum() / abs(returns[returns < 0].sum())


def payoff_ratio(returns: Returns, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return avg_win(returns, prepare_returns=False) / abs(avg_loss(returns, prepare_returns=False))


def profit_ratio(returns: Returns, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return payoff_ratio(returns, False) * win_rate(returns, prepare_returns=False)


def recovery_factor(returns: Returns, prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return comp(returns) / abs(max_drawdown(returns))


def value_at_risk(returns: Returns, sigma: float = 1, confidence: float = .95,
                  prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    return _norm.ppf(1 - confidence, returns.mean(), returns.std() * sigma)


def conditional_value_at_risk(returns: Returns, sigma: float = 1,
                              confidence: float = .95,
                              prepare_returns: bool = True):
    if prepare_returns:
        returns = _prepare(returns)
    var = value_at_risk(returns, sigma, confidence, False)
    return returns[returns <= var].mean()


def expected_shortfall(returns: Returns, sigma: float = 1,
                       confidence: float = .95,
                       prepare_returns: bool = True):
    return conditional_value_at_risk(returns, sigma, confidence, prepare_returns)