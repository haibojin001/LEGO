# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg24::numpy.clip+numpy.shape
# name: numpy_primitive
# summary: Uses numpy.clip, numpy.shape across 2 repos
# anchor_symbols: ['numpy.clip', 'numpy.shape']
# observed in 2 repos: ['cleanlab__cleanlab', 'eriklindernoren__ML-From-Scratch']...

# --- from eriklindernoren__ML-From-Scratch::mlfromscratch/deep_learning/optimizers.py::NesterovAcceleratedGradient.update ---
def update(self, w, grad_func):
        # Calculate the gradient of the loss a bit further down the slope from w
        approx_future_grad = np.clip(grad_func(w - self.momentum * self.w_updt), -1, 1)
        # Initialize on first update
        if not self.w_updt.any():
            self.w_updt = np.zeros(np.shape(w))

        self.w_updt = self.momentum * self.w_updt + self.learning_rate * approx_future_grad
        # Move against the gradient to minimize loss
        return w - self.w_updt

# --- from cleanlab__cleanlab::cleanlab/internal/latent_algebra.py::compute_py ---
def compute_py(
    ps, noise_matrix, inverse_noise_matrix, *, py_method="cnt", true_labels_class_counts=None
) -> np.ndarray:
    """Compute ``py := P(true_labels=k)`` from ``ps := P(labels=k)``, `noise_matrix`, and
    `inverse_noise_matrix`.

    This method is ** ROBUST ** when ``py_method = 'cnt'``
    It may work well even when the noise matrices are estimated
    poorly by using the diagonals of the matrices
    instead of all the probabilities in the entire matrix.

    Parameters
    ----------
    ps : np.ndarray
        Array of shape ``(K, )`` or ``(1, K)`` containing the fraction (prior probability) of each observed, noisy label, P(labels = k)

    noise_matrix : np.ndarray
        A conditional probability matrix ( of shape ``(K, K)``) of the form ``P(label=k_s|true_label=k_y)`` containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    inverse_noise_matrix : np.ndarray of shape (K, K), K = number of classes
        A conditional probability matrix ( of shape ``(K, K)``) of the form ``P(true_label=k_y|label=k_s)`` representing
        the estimated fraction observed examples in each class `k_s`, that are
        mislabeled examples from every other class `k_y`. If ``None``, the
        inverse_noise_matrix will be computed from `pred_probs` and `labels`.
        Assumes columns of `inverse_noise_matrix` sum to 1.

    py_method : str (Options: ["cnt", "eqn", "marginal", "marginal_ps"])
        How to compute the latent prior ``p(true_label=k)``. Default is "cnt" as it often
        works well even when the noise matrices are estimated poorly by using
        the matrix diagonals instead of all the probabilities.

    true_labels_class_counts : np.ndarray
        Array of shape ``(K, )`` or ``(1, K)`` containing the marginal counts of the confident joint
        (like ``cj.sum(axis = 0)``).

    Returns
    -------
    py : np.ndarray
        Array of shape ``(K, )`` or ``(1, K)``.
        The fraction (prior probability) of each TRUE class label, ``P(true_label = k)``."""

    if len(np.shape(ps)) > 2 or (len(np.shape(ps)) == 2 and np.shape(ps)[0] != 1):
        w = "Input parameter np.ndarray ps has shape " + str(np.shape(ps))
        w += ", but shape should be (K, ) or (1, K)"
        warnings.warn(w)

    if py_method == "marginal" and true_labels_class_counts is None:
        msg = (
            'py_method == "marginal" requires true_labels_class_counts, '
            "but true_labels_class_counts is None. "
        )
        msg += " Provide parameter true_labels_class_counts."
        raise ValueError(msg)

    if py_method == "cnt":
        # Computing py this way avoids dividing by zero noise rates.
        # More robust bc error est_p(true_label|labels) / est_p(labels|y) ~ p(true_label|labels) / p(labels|y)
        py = (
            inverse_noise_matrix.diagonal()
            / np.clip(noise_matrix.diagonal(), a_min=TINY_VALUE, a_max=None)
            * ps
        )
        # Equivalently: py = (true_labels_class_counts / labels_class_counts) * ps
    elif py_method == "eqn":
        py = np.linalg.inv(noise_matrix).dot(ps)
    elif py_method == "marginal":
        py = true_labels_class_counts / np.clip(
            float(sum(true_labels_class_counts)), a_min=TINY_VALUE, a_max=None
        )
    elif py_method == "marginal_ps":
        py = np.dot(inverse_noise_matrix, ps)
    else:
        err = "py_method {}".format(py_method)
        err += " should be in [cnt, eqn, marginal, marginal_ps]"
        raise ValueError(err)

    # Clip py (0,1), s.t. no class should have prob 0, hence 1e-6
    py = clip_values(py, low=CLIPPING_LOWER_BOUND, high=1.0, new_sum=1.0)
    return py
