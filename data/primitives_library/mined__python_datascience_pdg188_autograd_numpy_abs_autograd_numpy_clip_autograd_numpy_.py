# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg188::autograd.numpy.abs+autograd.numpy.clip+autograd.numpy.dot
# name: autograd_primitive
# summary: Uses autograd.numpy.abs, autograd.numpy.clip, autograd.numpy.dot, autograd.numpy.exp across 8 repos
# anchor_symbols: ['autograd.numpy.abs', 'autograd.numpy.clip', 'autograd.numpy.dot', 'autograd.numpy.exp', 'autograd.numpy.isinf', 'autograd.numpy.log']
# observed in 8 repos: ['CamDavidsonPilon__lifelines', 'CamDavidsonPilon__lifetimes', 'MatthewReid854__reliability', 'cleanlab__cleanlab', 'lazyprogrammer__machine_learning_examples']...

# --- from MatthewReid854__reliability::reliability/Fitters.py::Fit_Weibull_DS.logR ---
def logR(t, a, b, ds):  # Log SF (Weibull DS)
        return anp.log(1 - ((1 - anp.exp(-((t / a) ** b))) * ds))

# --- from MatthewReid854__reliability::reliability/ALT_fitters.py::Fit_Exponential_Exponential.logf ---
def logf(t, T, a, b):  # Log PDF
        life = b * anp.exp(a / T)
        return anp.log(1 / life) - 1 / life * t

# --- from CamDavidsonPilon__lifelines::lifelines/fitters/log_logistic_fitter.py::LogLogisticFitter._log_1m_sf ---
def _log_1m_sf(self, params, times):
        alpha_, beta_ = params
        return -np.logaddexp(-beta_ * (np.log(times) - np.log(alpha_)), 0)

# --- from CamDavidsonPilon__lifelines::lifelines/fitters/log_logistic_fitter.py::LogLogisticFitter._cumulative_hazard ---
def _cumulative_hazard(self, params, times):
        alpha_, beta_ = params
        return np.logaddexp(beta_ * (np.log(np.clip(times, 1e-25, np.inf)) - np.log(alpha_)), 0)

# --- from wilsonrljr__sysidentpy::sysidentpy/general_estimators/tests/test_general_narx.py::test_fit_raise ---
def test_fit_raise():
    assert_raises(
        ValueError,
        NARX,
        base_estimator=LinearRegression(),
        basis_function=Polynomial(degree=1),
        model_type="NARARMAX",
    )

# --- from wilsonrljr__sysidentpy::sysidentpy/_lib/_vendor/array_api_compat/torch/_aliases.py::max ---
def max(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.clone(x)
    return torch.amax(x, axis, keepdims=keepdims)

# --- from mwaskom__seaborn::tests/test_algorithms.py::test_bootstrap_multiarg ---
def test_bootstrap_multiarg(random):
    """Test that bootstrap works with multiple input arrays."""
    x = np.vstack([[1, 10] for i in range(10)])
    y = np.vstack([[5, 5] for i in range(10)])

    def f(x, y):
        return np.vstack((x, y)).max(axis=0)

    out_actual = algo.bootstrap(x, y, n_boot=2, func=f)
    out_wanted = np.array([[5, 10], [5, 10]])
    assert_array_equal(out_actual, out_wanted)

# --- from CamDavidsonPilon__lifetimes::lifetimes/fitters/modified_beta_geo_fitter.py::ModifiedBetaGeoFitter._negative_log_likelihood ---
def _negative_log_likelihood(log_params, freq, rec, T, weights, penalizer_coef):
        warnings.simplefilter(action="ignore", category=FutureWarning)

        params = np.exp(log_params)
        r, alpha, a, b = params

        A_1 = gammaln(r + freq) - gammaln(r) + r * log(alpha)
        A_2 = gammaln(a + b) + gammaln(b + freq + 1) - gammaln(b) - gammaln(a + b + freq + 1)
        A_3 = -(r + freq) * log(alpha + T)
        A_4 = log(a) - log(b + freq) + (r + freq) * (log(alpha + T) - log(alpha + rec))

        penalizer_term = penalizer_coef * sum(params ** 2)
        return -(weights * (A_1 + A_2 + A_3 + logaddexp(A_4, 0))).sum() / weights.sum() + penalizer_term

# --- from lazyprogrammer__machine_learning_examples::rl3v2/visualize_hill_climbing.py::visualize_es ---
def visualize_es(history, bounds, f, resolution=100):
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    plt.figure(figsize=(8, 6))
    for i, (pop, mu) in enumerate(history):
        plt.clf()
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label="f(x, y)")
        plt.scatter(pop[:, 0], pop[:, 1], c='white', s=20, label='Population')
        plt.scatter(mu[0], mu[1], c='red', s=80, label='Mean', edgecolors='black')
        plt.title(f"Hill Climbing - Step {i+1}")
        plt.xlim(bounds[0], bounds[1])
        plt.ylim(bounds[0], bounds[1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        # plt.pause(0.1)
        plt.waitforbuttonpress()
    plt.show()

# --- from lazyprogrammer__machine_learning_examples::rl3v2/visualize_es.py::visualize_es ---
def visualize_es(history, bounds, f, resolution=100):
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    plt.figure(figsize=(8, 6))
    for i, (pop, mu) in enumerate(history):
        plt.clf()
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label="f(x, y)")
        plt.scatter(pop[:, 0], pop[:, 1], c='white', s=20, label='Population')
        plt.scatter(mu[0], mu[1], c='red', s=80, label='Mean', edgecolors='black')
        plt.title(f"Evolution Strategies - Step {i+1}")
        plt.xlim(bounds[0], bounds[1])
        plt.ylim(bounds[0], bounds[1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        # plt.pause(0.1)
        plt.waitforbuttonpress()
    plt.show()

# --- from yzhao062__pyod::pyod/models/sod.py::SOD._sod ---
def _sod(self, X):
        """This function is called internally to perform subspace outlier 
        detection algorithm.
        
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        ref_inds = self._snn(X)
        anomaly_scores = np.zeros(shape=(X.shape[0],))
        for i in range(X.shape[0]):
            obs = X[i]
            ref = X[ref_inds[i,],]
            means = np.mean(ref, axis=0)  # mean of each column
            # average squared distance of the reference to the mean
            var_total = np.sum(np.sum(np.square(ref - means))) / self.ref_set
            var_expect = self.alpha * var_total / X.shape[1]
            var_actual = np.var(ref, axis=0)  # variance of each attribute
            var_inds = [1 if (j < var_expect) else 0 for j in var_actual]
            rel_dim = np.sum(var_inds)
            if rel_dim != 0:
                anomaly_scores[i] = np.sqrt(
                    np.dot(var_inds, np.square(obs - means)) / rel_dim)

        return anomaly_scores

# --- from cleanlab__cleanlab::tests/test_multiannotator.py::test_single_label_active_learning ---
def test_single_label_active_learning():
    labels = np.array(small_data["complete_labels"])
    labels_unlabeled = small_data["true_labels_train_unlabeled"]
    pred_probs = small_data["pred_probs_complete"]
    pred_probs_unlabeled = small_data["pred_probs_unlabeled"]

    assert len(labels) == 15

    # test 5 rounds of active learning
    for i in range(5):
        active_learning_scores, active_learning_scores_unlabeled = get_active_learning_scores(
            labels, pred_probs, pred_probs_unlabeled
        )

        min_ind = np.argmin(active_learning_scores_unlabeled)

        labels = np.append(labels, labels_unlabeled[min_ind]).reshape(-1, 1)
        pred_probs = np.append(pred_probs, pred_probs_unlabeled[min_ind].reshape(1, -1), axis=0)
        labels_unlabeled = np.delete(labels_unlabeled, min_ind)
        pred_probs_unlabeled = np.delete(pred_probs_unlabeled, min_ind, axis=0)

    assert len(labels) == 20

    # make sure error is thrown if labels are not 2D
    labels_flat = np.array(small_data["complete_labels"]).reshape(1, -1)
    try:
        active_learning_scores, active_learning_scores_unlabeled = get_active_learning_scores(
            labels, pred_probs, pred_probs_unlabeled
        )
    except ValueError as e:
        assert "labels_multiannotator must be a 2D array or dataframe" in str(e)
