# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg282::numpy.clip+numpy.finfo+numpy.log
# name: numpy_primitive
# summary: Uses numpy.clip, numpy.finfo, numpy.log, numpy.nan_to_num across 2 repos
# anchor_symbols: ['numpy.clip', 'numpy.finfo', 'numpy.log', 'numpy.nan_to_num']
# observed in 2 repos: ['BiomedSciAI__causallib', 'yzhao062__pyod']...

# --- from BiomedSciAI__causallib::causallib/estimation/doubly_robust.py::PropensityFeatureStandardization._get_feature_function.logit_propensity_vector ---
def logit_propensity_vector(X, a, safe=True):
            p = propensity_vector(X, a)
            if safe:
                epsilon = np.finfo(float).eps
                p = np.clip(p, epsilon, 1 - epsilon)
            return np.log(p / (1 - p))

# --- from BiomedSciAI__causallib::causallib/estimation/tmle.py::_logit ---
def _logit(p, safe=True):
    # TODO: move logit as a method, and do a clipped version with bounds specified in constructor
    if safe:
        epsilon = np.finfo(float).eps
        p = np.clip(p, epsilon, 1 - epsilon)
    return np.log(p / (1 - p))

# --- from yzhao062__pyod::pyod/models/embedding.py::EmbeddingOD._preprocess_transform ---
def _preprocess_transform(self, X_emb):
        """Transform embeddings using fitted preprocessing."""
        X_emb = np.nan_to_num(X_emb)
        X_emb = np.clip(X_emb, np.finfo(np.float32).min,
                        np.finfo(np.float32).max)

        if self.standardize:
            X_emb = self.scaler_.transform(X_emb)

        if self.reduce_dim is not None:
            X_emb = self.pca_.transform(X_emb)

        return X_emb.astype(np.float32)

# --- from yzhao062__pyod::pyod/models/embedding.py::EmbeddingOD._preprocess_fit ---
def _preprocess_fit(self, X_emb):
        """Fit preprocessing and transform embeddings."""
        X_emb = np.nan_to_num(X_emb)
        X_emb = np.clip(X_emb, np.finfo(np.float32).min,
                        np.finfo(np.float32).max)

        if self.standardize:
            self.scaler_ = StandardScaler()
            X_emb = self.scaler_.fit_transform(X_emb)

        if self.reduce_dim is not None:
            # PCA can pick a randomized solver under svd_solver='auto' on
            # high-dimensional embeddings; pass the engine seed through so
            # `EmbeddingOD(random_state=...)` covers the preprocessing
            # step alongside the inner detector.
            self.pca_ = PCA(n_components=self.reduce_dim,
                            random_state=self.random_state)
            X_emb = self.pca_.fit_transform(X_emb)

        return X_emb.astype(np.float32)
