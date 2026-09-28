from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance._validation import frame, window_size


@dataclass(frozen=True)
class PCAResult:
    loadings: pd.DataFrame
    scores: pd.DataFrame
    explained_variance: pd.Series


def pca(returns: pd.DataFrame, components: int = 2) -> PCAResult:
    """Descriptive cross-asset PCA of standardized returns; not an out-of-sample portfolio."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    data = frame(returns)
    window_size(components)

    if components > min(data.shape) or (data.std() == 0).any():
        raise ValueError("invalid component count or constant asset")

    standardized = StandardScaler().fit_transform(data)
    model = PCA(n_components=components).fit(standardized)
    labels = [f"pc{number}" for number in range(1, components + 1)]

    return PCAResult(
        loadings=pd.DataFrame(
            model.components_.T,
            index=data.columns,
            columns=labels,
        ),
        scores=pd.DataFrame(
            model.transform(standardized),
            index=data.index,
            columns=labels,
        ),
        explained_variance=pd.Series(
            model.explained_variance_ratio_,
            index=labels,
        ),
    )


def cluster_assets(
    returns: pd.DataFrame,
    clusters: int = 3,
    *,
    components: int | None = None,
    method: str = "kmeans",
    seed: int = 0,
) -> pd.Series:
    """Cluster assets (rows), not dates, using standardized return paths or PCA loadings."""
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    data = frame(returns)
    window_size(clusters)

    if clusters > data.shape[1] or (data.std() == 0).any():
        raise ValueError("too many clusters or constant asset")

    if components is None:
        matrix = pd.DataFrame(
            StandardScaler().fit_transform(data).T,
            index=data.columns,
        )
    else:
        matrix = pca(data, components).loadings

    if method == "kmeans":
        assignments = KMeans(
            n_clusters=clusters,
            n_init=20,
            random_state=seed,
        ).fit_predict(matrix)
    elif method == "mixture":
        assignments = GaussianMixture(
            n_components=clusters,
            random_state=seed,
            reg_covar=1e-5,
        ).fit_predict(matrix)
    else:
        raise ValueError("method must be kmeans or mixture")

    return pd.Series(assignments, index=data.columns, name="cluster")


def cluster_features(
    features: pd.DataFrame,
    clusters: int = 3,
    seed: int = 0,
) -> pd.Series:
    """Cluster one row per asset of supplied technical/fundamental features."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    data = frame(features)
    window_size(clusters)

    if clusters > len(data):
        raise ValueError("more clusters than assets")

    assignments = KMeans(
        n_clusters=clusters,
        n_init=20,
        random_state=seed,
    ).fit_predict(StandardScaler().fit_transform(data))

    return pd.Series(assignments, index=data.index, name="cluster")


def partial_correlations(returns: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """Descriptive regularized conditional dependence; fixed alpha, no predictive claim."""
    from sklearn.covariance import GraphicalLasso
    from sklearn.preprocessing import StandardScaler

    data = frame(returns)

    if not np.isfinite(alpha) or alpha <= 0 or (data.std() == 0).any():
        raise ValueError("positive alpha and nonconstant assets required")

    model = GraphicalLasso(alpha=alpha, max_iter=500).fit(
        StandardScaler().fit_transform(data)
    )

    if model.n_iter_ >= 500:
        raise ValueError("graphical lasso did not converge")

    inverse_covariance = model.precision_
    diagonal = np.sqrt(np.diag(inverse_covariance))
    correlations = -inverse_covariance / np.outer(diagonal, diagonal)
    np.fill_diagonal(correlations, 1)

    return pd.DataFrame(
        correlations,
        index=data.columns,
        columns=data.columns,
    )


def anomaly_scores(
    training: pd.DataFrame,
    observations: pd.DataFrame,
    seed: int = 0,
) -> pd.Series:
    """Isolation Forest novelty scores fitted only on earlier observations; lower is more unusual."""
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_data = frame(training)
    observation_data = frame(observations)

    if (
        not train_data.columns.equals(observation_data.columns)
        or train_data.index[-1] >= observation_data.index[0]
    ):
        raise ValueError(
            "require matching features and training strictly before observations"
        )

    estimator = make_pipeline(
        StandardScaler(),
        IsolationForest(random_state=seed, contamination="auto"),
    )
    estimator.fit(train_data)

    return pd.Series(
        estimator.decision_function(observation_data),
        index=observation_data.index,
        name="anomaly_score",
    )


def cointegration_pairs(prices: pd.DataFrame) -> pd.DataFrame:
    """Engle-Granger tests on supplied training prices; Holm-adjusted p-values across pairs."""
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tsa.stattools import coint

    data = frame(prices, positive=True)

    if len(data) < 40 or data.shape[1] < 2:
        raise ValueError("provide at least two assets and 40 training observations")

    records = []
    for position, left in enumerate(data):
        for right in data.columns[position + 1 :]:
            test_statistic, p_value, _ = coint(
                np.log(data[left]),
                np.log(data[right]),
            )
            records.append(
                {
                    "first": left,
                    "second": right,
                    "statistic": test_statistic,
                    "p_value": p_value,
                }
            )

    result = pd.DataFrame(records)
    result["adjusted_p_value"] = multipletests(
        result.p_value,
        method="holm",
    )[1]

    return result.sort_values("adjusted_p_value", ignore_index=True)