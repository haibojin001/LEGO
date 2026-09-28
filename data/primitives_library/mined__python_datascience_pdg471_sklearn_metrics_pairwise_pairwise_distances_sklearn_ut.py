# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg471::sklearn.metrics.pairwise.pairwise_distances+sklearn.utils.check_array
# name: sklearn_primitive
# summary: Uses sklearn.metrics.pairwise.pairwise_distances, sklearn.utils.check_array across 2 repos
# anchor_symbols: ['sklearn.metrics.pairwise.pairwise_distances', 'sklearn.utils.check_array']
# observed in 2 repos: ['ScottfreeLLC__AlphaPy', 'nidhaloff__igel']...

# --- from nidhaloff__igel::igel/extras/kmedians.py::KMedians.transform ---
def transform(self, X):
        """Transforms given data into cluster space of size {n_samples, n_clusters}"""
        X = check_array(X, accept_sparse=["csr", "csc"])
        check_is_fitted(self, "cluster_centers_")

        Y = self.cluster_centers_
        return pairwise_distances(X, Y=Y, metric=self.metric)

# --- from nidhaloff__igel::igel/extras/kmedoids.py::KMedoids.transform ---
def transform(self, X):
        """Transforms X to cluster-distance space.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Data to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape=(n_query, n_clusters)
            X transformed in the new space of distances to cluster centers.
        """
        X = check_array(X, accept_sparse=["csr", "csc"])
        check_is_fitted(self, "cluster_centers_")

        Y = self.cluster_centers_
        return pairwise_distances(X, Y=Y, metric=self.metric)

# --- from ScottfreeLLC__AlphaPy::alphapy/optimize.py::hyper_grid_search ---
def hyper_grid_search(model, estimator):
    r"""Return the best hyperparameters for a grid search.

    Parameters
    ----------
    model : alphapy.Model
        The model object with grid search parameters.
    estimator : alphapy.Estimator
        The estimator containing the hyperparameter grid.

    Returns
    -------
    model : alphapy.Model
        The model object with the grid search estimator.

    Notes
    -----
    To reduce the time required for grid search, use either
    randomized grid search with a fixed number of iterations
    or a full grid search with subsampling. AlphaPy uses
    the scikit-learn Pipeline with feature selection to
    reduce the feature space.

    References
    ----------
    For more information about grid search, refer to [GRID]_.

    .. [GRID] http://scikit-learn.org/stable/modules/grid_search.html#grid-search

    To learn about pipelines, refer to [PIPE]_.

    .. [PIPE] http://scikit-learn.org/stable/modules/pipeline.html#pipeline

    """

    # Extract estimator parameters.

    grid = estimator.grid
    if not grid:
        logger.info("No grid is defined for grid search")
        return model

    # Get estimator.

    algo = estimator.algorithm
    est = model.estimators[algo]

    # Extract model data.

    try:
        support = model.support[algo]
        X_train = model.X_train[:, support]
    except:
        X_train = model.X_train
    y_train = model.y_train

    # Extract model parameters.

    cv_folds = model.specs['cv_folds']
    feature_selection = model.specs['feature_selection']
    fs_percentage = model.specs['fs_percentage']
    fs_score_func = model.specs['fs_score_func']
    fs_uni_grid = model.specs['fs_uni_grid']
    gs_iters = model.specs['gs_iters']
    gs_random = model.specs['gs_random']
    gs_sample = model.specs['gs_sample']
    gs_sample_pct = model.specs['gs_sample_pct']
    n_jobs = model.specs['n_jobs']
    scorer = model.specs['scorer']
    verbosity = model.specs['verbosity']

    # Subsample if necessary to reduce grid search duration.

    if gs_sample:
        length = len(X_train)
        subset = int(length * gs_sample_pct)
        indices = np.random.choice(length, subset, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    # Convert the grid to pipeline format

    grid_new = {}
    for k, v in list(grid.items()):
        new_key = '__'.join(['est', k])
        grid_new[new_key] = grid[k]

    # Create the pipeline for grid search

    if feature_selection:
        # Augment the grid for feature selection.
        fs = SelectPercentile(score_func=fs_score_func,
                              percentile=fs_percentage)
        # Combine the feature selection and estimator grids.
        fs_grid = dict(fs__percentile=fs_uni_grid)
        grid_new.update(fs_grid)
        # Create a pipeline with the selected features and estimator.
        pipeline = Pipeline([("fs", fs), ("est", est)])
    else:
        pipeline = Pipeline([("est", est)])

    # Create the randomized grid search iterator.

    if gs_random:
        logger.info("Randomized Grid Search")
        gscv = RandomizedSearchCV(pipeline, param_distributions=grid_new,
                                  n_iter=gs_iters, scoring=scorer,
                                  n_jobs=n_jobs, cv=cv_folds, verbose=verbosity)
    else:
        logger.info("Full Grid Search")
        gscv = GridSearchCV(pipeline, param_grid=grid_new, scoring=scorer,
                            n_jobs=n_jobs, cv=cv_folds, verbose=verbosity)

    # Fit the randomized search and time it.

    start = time()
    gscv.fit(X_train, y_train)
    if gs_iters > 0:
        logger.info("Grid Search took %.2f seconds for %d candidate"
                    " parameter settings." % ((time() - start), gs_iters))
    else:
        logger.info("Grid Search took %.2f seconds for %d candidate parameter"
                    " settings." % (time() - start, len(gscv.cv_results_['params'])))

    # Log the grid search scoring statistics.

    grid_report(gscv.cv_results_)
    logger.info("Algorithm: %s, Best Score: %.4f, Best Parameters: %s",
                algo, gscv.best_score_, gscv.best_params_)

    # Assign the Grid Search estimator for this algorithm

    model.estimators[algo] = gscv

    # Return the model with Grid Search estimators
    return model
