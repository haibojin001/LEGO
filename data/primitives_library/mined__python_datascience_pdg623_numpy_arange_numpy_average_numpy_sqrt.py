# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg623::numpy.arange+numpy.average+numpy.sqrt
# name: numpy_sklearn_primitive
# summary: Uses numpy.arange, numpy.average, numpy.sqrt, sklearn.linear_model.lars_path across 2 repos
# anchor_symbols: ['numpy.arange', 'numpy.average', 'numpy.sqrt', 'sklearn.linear_model.lars_path']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/interpretation/lime.py::LimeTextExplainer._feature_selection ---
def _feature_selection(
        self,
        data: pd.DataFrame,
        y: np.array,
        weights: np.array,
        n_features: int,
        mode: str = "none",
    ) -> List[int]:
        if mode == "none":
            return np.arange(data.shape[1])
        if mode == "lasso":
            weighted_data = (data - np.average(data, axis=0, weights=weights)) * np.sqrt(weights[:, np.newaxis])
            weighted_y = (y - np.average(y, weights=weights)) * np.sqrt(weights)

            features = np.arange(weighted_data.shape[1])
            _, _, coefs = lars_path(weighted_data, weighted_y, method="lasso", verbose=False)

            for i in range(len(coefs.T) - 1, 0, -1):
                features = coefs.T[i].nonzero()[0]
                if len(features) <= n_features:
                    break

            return features

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/interpretation/lime.py::LimeTextExplainer._feature_selection ---
def _feature_selection(
        self,
        data: pd.DataFrame,
        y: np.array,
        weights: np.array,
        n_features: int,
        mode: str = "none",
    ) -> List[int]:
        if mode == "none":
            return np.arange(data.shape[1])
        if mode == "lasso":
            weighted_data = (data - np.average(data, axis=0, weights=weights)) * np.sqrt(weights[:, np.newaxis])
            weighted_y = (y - np.average(y, weights=weights)) * np.sqrt(weights)

            features = np.arange(weighted_data.shape[1])
            _, _, coefs = lars_path(weighted_data, weighted_y, method="lasso", verbose=False)

            for i in range(len(coefs.T) - 1, 0, -1):
                features = coefs.T[i].nonzero()[0]
                if len(features) <= n_features:
                    break

            return features
