# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg63::matplotlib.pyplot.plot+matplotlib.pyplot.subplots_adjust+matplotlib.pyplot.title
# name: matplotlib_primitive
# summary: Uses matplotlib.pyplot.plot, matplotlib.pyplot.subplots_adjust, matplotlib.pyplot.title, matplotlib.pyplot.xlabel across 12 repos
# anchor_symbols: ['matplotlib.pyplot.plot', 'matplotlib.pyplot.subplots_adjust', 'matplotlib.pyplot.title', 'matplotlib.pyplot.xlabel', 'matplotlib.pyplot.ylabel']
# observed in 12 repos: ['BiomedSciAI__causallib', 'CamDavidsonPilon__lifelines', 'MatthewReid854__reliability', 'ScottfreeLLC__AlphaPy', 'annoviko__pyclustering']...

# --- from MatthewReid854__reliability::reliability/Utils.py::distribution_confidence_intervals.gamma_CI.u ---
def u(t, mu, beta):  # u = R
                return agammaincc(beta, t / anp.exp(mu))

# --- from MatthewReid854__reliability::tests/test_Other_functions.py::test_distribution_explorer ---
def test_distribution_explorer():
    plt.ion()
    distribution_explorer()
    plt.close()
    plt.ioff()

# --- from vaexio__vaex::packages/vaex-core/vaex/legacy.py::SubspaceBounded.lim ---
def lim(self):
        from matplotlib import pyplot as plt
        xmin, xmax = self.bounds[0]
        ymin, ymax = self.bounds[1]
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

# --- from ploomber__sklearn-evaluation::src/sklearn_evaluation/plot/util.py::no_display_plots ---
def no_display_plots():
    """Turn off matplotlib interactive plotting

    Examples
    --------
    >>> from sklearn_evaluation.plot.util import no_display_plots
    >>> import matplotlib.pyplot as plt
    >>> with no_display_plots():
    ...     ax = plt.plot([1, 2, 3])

    """
    if plt.isinteractive():
        plt.ioff()

        try:
            yield
        finally:
            plt.close("all")
            plt.ion()
    else:
        yield

# --- from BiomedSciAI__causallib::causallib/positivity/datasets/positivity_data_simulator.py::make_1d_overlap_data ---
def make_1d_overlap_data(treatment_bounds=(0, 75),
                         control_bounds=(25, 100)):
    """Generate 1d overlap data with integer covariates

    Args:
        treatment_bounds (tuple, optional): Bounds for covariates in treatment
            group. Defaults to (0, 75).
        control_bounds (tuple, optional): Bounds for covariates in control
            group. Defaults to (25, 100).

    Returns:
        X (pd.DataFrame), a (pd.Series): covariate and treatment assignment
    """

    X_treatment = np.arange(*treatment_bounds)
    a_treatment = np.ones_like(X_treatment)

    X_control = np.arange(*control_bounds)
    a_control = np.zeros_like(X_control)

    X = pd.DataFrame(data=np.hstack((X_treatment, X_control)), columns=["X1"])
    a = pd.Series(data=np.hstack((a_treatment, a_control)), name="treatment")

    return X, a

# --- from annoviko__pyclustering::pyclustering/cluster/bang.py::bang_visualizer.show_dendrogram ---
def show_dendrogram(dendrogram):
        """!
        @brief Display dendrogram of BANG-blocks.

        @param[in] dendrogram (list): List representation of dendrogram of BANG-blocks.

        @see bang.get_dendrogram()

        """
        figure = plt.figure()
        axis = plt.subplot(1, 1, 1)

        current_position = 0
        for index_cluster in range(len(dendrogram)):
            densities = [ block.get_density() for block in dendrogram[index_cluster] ]
            xrange = range(current_position, current_position + len(densities))

            axis.bar(xrange, densities, 1.0, linewidth=0.0, color=color_list.get_color(index_cluster))

            current_position += len(densities)

        axis.set_ylabel("density")
        axis.set_xlabel("block")
        axis.xaxis.set_ticklabels([])

        plt.xlim([-0.5, current_position - 0.5])
        plt.show()
        plt.close(figure)

# --- from BiomedSciAI__causallib::causallib/metrics/propensity_metrics.py::expected_roc_auc_error ---
def expected_roc_auc_error(y_true, y_pred, **kwargs):
    """
    Compute the squared error between the expected ROC-AUC given the provided
    scores and the actual ROC-AUC they produce.

    Shimoni, Y., et al. (2019)
    An evaluation toolkit to guide model selection and cohort definition in causal inference.

    Args:
        y_true (pd.Series): True binary label assignment of size (num_subjects,)
        y_pred (pd.Series): Predicted probability of each sample being
                            the positive label of size (num_subjects,).

    Returns:
        score (float):
    """
    # Calculate expected roc auc:
    p = np.hstack((y_pred, y_pred))
    w = np.hstack((y_pred, 1 - y_pred))
    target = np.hstack((np.ones_like(y_pred), np.zeros_like(y_pred)))
    expected_auc = roc_auc_score(target, p, sample_weight=w)

    auc = roc_auc_score(y_true, y_pred)
    score = (expected_auc - auc) ** 2
    return score

# --- from CamDavidsonPilon__lifelines::lifelines/fitters/npmle.py::scipy_minimize_fit ---
def scipy_minimize_fit(turnbull_interval_lookup, turnbull_intervals, weights, tol, verbose):
    import autograd.numpy as anp
    from autograd import value_and_grad
    from scipy.optimize import minimize

    def cumulative_sum(p):
        return anp.concatenate((anp.zeros(1), p)).cumsum()

    def negative_log_likelihood(p, turnbull_interval_lookup, weights):
        P = cumulative_sum(p)
        ix = anp.array(list(turnbull_interval_lookup.values()))
        return -(weights * anp.log(P[ix[:, 1] + 1] - P[ix[:, 0]])).sum()

    def con(p):
        return p.sum() - 1

    # initialize to equal weight
    T = len(turnbull_intervals)
    p = 1 / T * np.ones(T)

    cons = {"type": "eq", "fun": con}
    results = minimize(
        value_and_grad(negative_log_likelihood),
        args=(turnbull_interval_lookup, weights),
        x0=p,
        bounds=[(0, 1)] * T,
        jac=True,
        constraints=cons,
        tol=tol,
        options={"disp": verbose},
    )
    return results.x

# --- from google__uncertainty-baselines::experimental/near_ood/vit/ood_utils.py::compute_ood_metrics ---
def compute_ood_metrics(targets,
                        predictions,
                        tpr_thres=0.95,
                        targets_threshold=None):
  """Computes Area Under the ROC and PR curves and FPRN.

  ROC - Receiver Operating Characteristic
  PR  - Precision and Recall
  FPRN - False positive rate at which true positive rate is N.

  Args:
    targets: np.ndarray of targets, either 0 or 1, or continuous values.
    predictions: np.ndarray of predictions, any value.
    tpr_thres: float, threshold for true positive rate.
    targets_threshold: float, if target values are continuous values, this
      threshold binarizes them.

  Returns:
    A dictionary with AUC-ROC, AUC-PR, and FPRN scores.
  """

  if targets_threshold is not None:
    targets = np.array(targets)
    targets = np.where(targets < targets_threshold,
                       np.zeros_like(targets, dtype=np.int32),
                       np.ones_like(targets, dtype=np.int32))

  fpr, tpr, _ = sklearn.metrics.roc_curve(targets, predictions)
  fprn = fpr[np.argmax(tpr >= tpr_thres)]

  return {
      'auroc': sklearn.metrics.roc_auc_score(targets, predictions),
      'auprc': sklearn.metrics.average_precision_score(targets, predictions),
      'fprn': fprn,
  }

# --- from google__uncertainty-baselines::baselines/jft/ood_utils.py::compute_ood_metrics ---
def compute_ood_metrics(targets,
                        predictions,
                        tpr_thres=0.95,
                        targets_threshold=None):
  """Computes Area Under the ROC and PR curves and FPRN.

  ROC - Receiver Operating Characteristic
  PR  - Precision and Recall
  FPRN - False positive rate at which true positive rate is N.

  Args:
    targets: np.ndarray of targets, either 0 or 1, or continuous values.
    predictions: np.ndarray of predictions, any value.
    tpr_thres: float, threshold for true positive rate.
    targets_threshold: float, if target values are continuous values, this
      threshold binarizes them.

  Returns:
    A dictionary with AUC-ROC, AUC-PR, and FPRN scores.
  """

  if targets_threshold is not None:
    targets = np.array(targets)
    targets = np.where(targets < targets_threshold,
                       np.zeros_like(targets, dtype=np.int32),
                       np.ones_like(targets, dtype=np.int32))

  fpr, tpr, _ = sklearn.metrics.roc_curve(targets, predictions)
  fprn = fpr[np.argmax(tpr >= tpr_thres)]

  return {
      'auroc': sklearn.metrics.roc_auc_score(targets, predictions),
      'auprc': sklearn.metrics.average_precision_score(targets, predictions),
      'fprn': fprn,
  }

# --- from modin-project__modin::stress_tests/kaggle/kaggle12.py::plot_learning_curve ---
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.legend(loc="best")
    return plt

# --- from microsoft__RD-Agent::rdagent/scenarios/kaggle/tpl_ex/aerial-cactus-identification/ensemble.py::ensemble_workflow ---
def ensemble_workflow(test_pred_l: list[np.ndarray], val_pred_l: list[np.ndarray], val_label: np.ndarray) -> np.ndarray:
    """
    Handle the following:
    1) Ensemble predictions using a simple average.
    2) Make final decision after ensemble (convert the predictions to final binary form).

    Parameters
    ----------
    test_pred_l : list[np.ndarray]
        List of predictions on the test data.
    val_pred_l : list[np.ndarray]
        List of predictions on the validation data.
    val_label : np.ndarray
        True labels of the validation data.

    Returns
    -------
    np.ndarray
        Binary predictions on the test data.
    """

    scores = []
    for id, val_pred in enumerate(val_pred_l):
        scores.append(roc_auc_score(val_label, val_pred))

    # Normalize the scores to get weights
    total_score = sum(scores)
    weights = [score / total_score for score in scores]

    # Weighted average of test predictions
    weighted_test_pred = np.zeros_like(test_pred_l[0])
    for weight, test_pred in zip(weights, test_pred_l):
        weighted_test_pred += weight * test_pred

    weighted_valid_pred = np.zeros_like(val_pred_l[0])
    for weight, val_pred in zip(weights, val_pred_l):
        weighted_valid_pred += weight * val_pred

    weighted_valid_pred_score = roc_auc_score(val_label, weighted_valid_pred)

    scores_df = pd.DataFrame(
        {
            "Model": list(range(len(val_pred_l))) + ["weighted_average_ensemble"],
            "AUROC": scores + [weighted_valid_pred_score],
        }
    )
    scores_df.to_csv("scores.csv", index=False)

    pred_binary_l = [0 if value < 0.50 else 1 for value in weighted_test_pred]
    return np.array(pred_binary_l)
