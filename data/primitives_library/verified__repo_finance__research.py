from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance._validation import frame, series, window_size


@dataclass(frozen=True)
class FactorResult:
    loadings: pd.DataFrame
    communalities: pd.Series
    diagnostics: pd.Series


def factor_analysis(returns: pd.DataFrame, factors: int = 2) -> FactorResult:
    from scipy.stats import chi2
    from sklearn.decomposition import FactorAnalysis
    from sklearn.preprocessing import StandardScaler

    data = frame(returns)
    window_size(factors)

    observations, assets = data.shape
    if (
        factors < 1
        or factors >= assets
        or observations <= assets + 2
        or (data.std() == 0).any()
    ):
        raise ValueError(
            "require fewer factors than nonconstant assets and more observations than assets"
        )

    correlation = data.corr().to_numpy(copy=True)
    determinant_sign, log_determinant = np.linalg.slogdet(correlation)
    if determinant_sign <= 0 or np.linalg.cond(correlation) > 1e10:
        raise ValueError("factor diagnostics require nonsingular correlation")

    precision = np.linalg.inv(correlation)
    partial = -precision / np.sqrt(
        np.outer(np.diag(precision), np.diag(precision))
    )
    np.fill_diagonal(partial, 0)
    np.fill_diagonal(correlation, 0)

    correlation_sum = (correlation**2).sum()
    kmo = (
        correlation_sum / (correlation_sum + (partial**2).sum())
        if correlation_sum
        else np.nan
    )
    bartlett_statistic = -(
        observations - 1 - (2 * assets + 5) / 6
    ) * log_determinant

    standardized = StandardScaler().fit_transform(data)
    fitted = FactorAnalysis(
        n_components=factors,
        random_state=0,
        max_iter=2000,
    ).fit(standardized)

    if fitted.n_iter_ >= 2000:
        raise ValueError("factor analysis did not converge")

    loadings = pd.DataFrame(
        fitted.components_.T,
        index=data.columns,
        columns=[f"factor_{number}" for number in range(1, factors + 1)],
    )
    communalities = (loadings**2).sum(axis=1).rename("communality")
    diagnostics = pd.Series(
        {
            "kmo": kmo,
            "bartlett_statistic": bartlett_statistic,
            "bartlett_p_value": chi2.sf(
                bartlett_statistic,
                assets * (assets - 1) / 2,
            ),
        }
    )
    return FactorResult(loadings, communalities, diagnostics)


def student_t_fit(values: pd.Series) -> pd.Series:
    from scipy.stats import t

    observations = series(values, missing=False)
    if len(observations) < 30 or observations.std() == 0:
        raise ValueError("at least 30 nonconstant observations required")

    degrees_of_freedom, location, scale = t.fit(observations.to_numpy())
    standard_deviation = (
        scale * np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2))
        if degrees_of_freedom > 2
        else np.inf
    )

    return pd.Series(
        {
            "df": degrees_of_freedom,
            "location": location,
            "scale": scale,
            "std": standard_deviation,
            "q01": t.ppf(
                0.01,
                degrees_of_freedom,
                loc=location,
                scale=scale,
            ),
            "q99": t.ppf(
                0.99,
                degrees_of_freedom,
                loc=location,
                scale=scale,
            ),
        }
    )


def volatility_regimes(
    close: pd.Series,
    *,
    train_fraction: float = 0.7,
    regimes: int = 2,
    window: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    from sklearn.mixture import GaussianMixture

    prices = series(close, positive=True, missing=False)
    window_size(window, 2)
    window_size(regimes, 2)

    if not 0.2 < train_fraction < 0.9:
        raise ValueError("invalid chronological split")

    volatility = prices.pct_change(fill_method=None).rolling(window).std().dropna()
    if (volatility <= 0).any():
        raise ValueError("positive rolling volatility required")

    boundary = int(len(volatility) * train_fraction)
    training = volatility.iloc[:boundary]
    held_out = volatility.iloc[boundary:]

    if len(training) < max(40, regimes * 10) or len(held_out) < 10:
        raise ValueError("insufficient training/test observations")

    mixture = GaussianMixture(
        n_components=regimes,
        random_state=seed,
        n_init=5,
    ).fit(np.log(training).to_numpy()[:, None])

    if not mixture.converged_:
        raise ValueError("volatility mixture did not converge")

    ascending_components = np.argsort(mixture.means_[:, 0])
    probabilities = mixture.predict_proba(
        np.log(held_out).to_numpy()[:, None]
    )[:, ascending_components]

    output = pd.DataFrame(
        probabilities,
        index=held_out.index,
        columns=[f"regime_{number}" for number in range(1, regimes + 1)],
    )
    output["volatility"] = held_out
    output["regime"] = probabilities.argmax(axis=1) + 1
    output.attrs.update(
        training_end=training.index[-1],
        ordering="ascending training log-volatility",
    )
    return output


def partial_correlations_cv(returns: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    from sklearn.covariance import GraphicalLassoCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    data = frame(returns)
    window_size(folds, 2)

    if len(data) < folds * 10 or (data.std() == 0).any():
        raise ValueError("insufficient nonconstant observations")

    splitter = TimeSeriesSplit(n_splits=folds)
    initial_training_rows, _ = next(splitter.split(data))
    scaler = StandardScaler().fit(data.iloc[initial_training_rows])

    model = GraphicalLassoCV(cv=splitter, max_iter=500).fit(
        scaler.transform(data)
    )
    if model.n_iter_ >= 500:
        raise ValueError("graphical lasso did not converge")

    precision = model.precision_
    partial = -precision / np.sqrt(
        np.outer(np.diag(precision), np.diag(precision))
    )
    np.fill_diagonal(partial, 1)

    output = pd.DataFrame(
        partial,
        index=data.columns,
        columns=data.columns,
    )
    output.attrs["alpha"] = model.alpha_
    return output


def arima_forecast(
    close: pd.Series,
    steps: int = 10,
    *,
    order: tuple[int, int, int] = (1, 1, 0),
    confidence: float = 0.95,
) -> pd.DataFrame:
    from statsmodels.tsa.arima.model import ARIMA

    prices = series(close, positive=True, missing=False)
    window_size(steps)

    if (
        len(prices) < 40
        or not 0 < confidence < 1
        or len(order) != 3
        or any(not isinstance(value, int) or value < 0 for value in order)
    ):
        raise ValueError("invalid history, confidence or ARIMA order")

    fitted = ARIMA(prices.to_numpy(), order=order).fit()
    if not fitted.mle_retvals.get("converged", True):
        raise ValueError("ARIMA failed to converge")

    prediction = fitted.get_forecast(steps)
    interval = prediction.conf_int(alpha=1 - confidence)
    output = pd.DataFrame(
        {
            "forecast": prediction.predicted_mean,
            "lower": interval[:, 0],
            "upper": interval[:, 1],
        },
        index=pd.RangeIndex(1, steps + 1, name="step"),
    )
    output.attrs.update(
        as_of=prices.index[-1],
        confidence=confidence,
        order=order,
    )
    return output


def arima_diagnostics(
    close: pd.Series,
    *,
    order: tuple[int, int, int] = (1, 1, 0),
    seasonal_period: int = 21,
) -> tuple[pd.Series, pd.DataFrame]:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller

    prices = series(close, positive=True, missing=False)
    window_size(seasonal_period, 2)

    if len(prices) < max(60, 2 * seasonal_period) or prices.std() == 0:
        raise ValueError("insufficient nonconstant history")

    fitted = ARIMA(prices.to_numpy(), order=order).fit()
    if not fitted.mle_retvals.get("converged", True):
        raise ValueError("ARIMA failed to converge")

    residuals = fitted.resid[max(order[1], 1):]
    ljung_box = acorr_ljungbox(residuals, lags=[10], return_df=True)
    decomposition = STL(
        prices,
        period=seasonal_period,
        robust=True,
    ).fit()

    diagnostics = pd.Series(
        {
            "adf_price_p_value": adfuller(prices)[1],
            "adf_difference_p_value": adfuller(prices.diff().dropna())[1],
            "residual_ljung_box_p_value": ljung_box.lb_pvalue.iloc[0],
            "aic": fitted.aic,
            "bic": fitted.bic,
        }
    )
    components = pd.DataFrame(
        {
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid,
        },
        index=prices.index,
    )
    return diagnostics, components


def select_arima_order(
    close: pd.Series,
    orders: tuple[tuple[int, int, int], ...] = ((0, 1, 0), (1, 1, 0), (0, 1, 1)),
    *,
    train_fraction: float = 0.8,
) -> pd.DataFrame:
    from statsmodels.tsa.arima.model import ARIMA

    prices = series(close, positive=True, missing=False)

    if not orders or not 0.2 < train_fraction < 0.9:
        raise ValueError("orders and chronological split required")

    split = int(len(prices) * train_fraction)
    training = prices.iloc[:split]
    held_out = prices.iloc[split:]

    if len(training) < 40 or len(held_out) < 10:
        raise ValueError("insufficient training/test observations")

    records = []
    for candidate in orders:
        try:
            if (
                len(candidate) != 3
                or any(
                    not isinstance(value, int) or value < 0
                    for value in candidate
                )
            ):
                raise ValueError("invalid ARIMA order")

            fitted = ARIMA(training.to_numpy(), order=candidate).fit()
            if not fitted.mle_retvals.get("converged", True):
                raise ValueError("ARIMA failed to converge")

            records.append(
                {
                    "order": candidate,
                    "aic": fitted.aic,
                    "bic": fitted.bic,
                    "error": None,
                }
            )
        except Exception as error:
            records.append(
                {
                    "order": candidate,
                    "aic": np.nan,
                    "bic": np.nan,
                    "error": str(error),
                }
            )

    output = (
        pd.DataFrame(records)
        .sort_values("bic", ascending=True, na_position="last")
        .reset_index(drop=True)
    )
    output.attrs.update(
        training_end=training.index[-1],
        holdout_start=held_out.index[0],
    )
    return output