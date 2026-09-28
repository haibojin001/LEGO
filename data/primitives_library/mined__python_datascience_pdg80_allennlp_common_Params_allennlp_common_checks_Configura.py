# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg80::allennlp.common.Params+allennlp.common.checks.ConfigurationError+allennlp.common.file_utils.is_url_or_existing_file
# name: allennlp_collections_primitive
# summary: Uses allennlp.common.Params, allennlp.common.checks.ConfigurationError, allennlp.common.file_utils.is_url_or_existing_file, allennlp.data.Vocabulary across 17 repos
# anchor_symbols: ['allennlp.common.Params', 'allennlp.common.checks.ConfigurationError', 'allennlp.common.file_utils.is_url_or_existing_file', 'allennlp.data.Vocabulary', 'collections.defaultdict', 'combo.models.classifier_comb.SimpleClassifierAggregator']
# observed in 17 repos: ['BiomedSciAI__causallib', 'Lightning-AI__torchmetrics', 'ScottfreeLLC__AlphaPy', 'allenai__allennlp', 'annoviko__pyclustering']...

# --- from yzhao062__pyod::pyod/models/ecod.py::skew ---
def skew(X, axis=0):
    return np.nan_to_num(skew_sp(X, axis=axis))

# --- from yzhao062__pyod::pyod/models/copod.py::skew ---
def skew(X, axis=0):
    return np.nan_to_num(skew_sp(X, axis=axis))

# --- from rnorm__book_sample::code/chapter5/lasso.py::sto ---
def sto(z, theta):
    return np.sign(z)*np.maximum(np.abs(z)-np.full(len(z),theta), 0.0)

# --- from pykale__pykale::tests/helpers/mock_graph.py::create_mock_graph ---
def create_mock_graph(num_nodes, num_edges, in_feats):
    x = torch.randn(num_nodes, in_feats)  # node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # random edges
    return Data(x=x, edge_index=edge_index)

# --- from BiomedSciAI__causallib::causallib/contrib/shared_sparsity_selection/shared_sparsity_selection.py::MCPSelector._mcp_q_grad ---
def _mcp_q_grad(self, t):
        if self.alpha == 0:
            return np.sign(t) * self.lmda_ * (-1)
        else:
            return (np.sign(t) * self.lmda_
                    * np.maximum(-1, - np.abs(t) / (self.lmda_ * self.alpha)))

# --- from yzhao062__combo::examples/temp_do_not_use.py::generate_similarity_mat ---
def generate_similarity_mat(labels):
        
#        labels = column_or_1d(labels)
        l_mat = np.repeat(labels, len(labels), axis=1)
        l_mat_t = l_mat.T
        sim_mat = np.equal(l_mat, l_mat_t).astype(int)
        return sim_mat

# --- from yzhao062__combo::combo/test/test_classifier_comb.py::TestWeightedAverage.test_prediction_scores ---
def test_prediction_scores(self):
        y_test_predicted = self.clf.predict(self.X_test)
        assert_equal(len(y_test_predicted), self.X_test.shape[0])

        # check performance
        assert (accuracy_score(self.y_test, y_test_predicted) >=
                self.accuracy_floor)

# --- from microsoft__nni::examples/nas/legacy/oneshot/proxylessnas/putils.py::LabelSmoothingLoss.forward ---
def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.size(self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# --- from microsoft__nni::examples/nas/legacy/oneshot/proxylessnas/retrain.py::cross_entropy_with_label_smoothing ---
def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

# --- from wilsonrljr__sysidentpy::sysidentpy/utils/tests/test_check_arrays.py::test_check_linear_dependence_rows_accepts_array_api_full_rank ---
def test_check_linear_dependence_rows_accepts_array_api_full_rank():
    """Array API inputs should use the backend SVD path for full-rank matrices."""
    xp = pytest.importorskip("array_api_strict")
    psi = xp.asarray(np.eye(3, dtype=float))

    with config_context(array_api_dispatch=True):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            check_linear_dependence_rows(psi)

    assert recorded == []

# --- from Lightning-AI__torchmetrics::src/torchmetrics/utilities/distributed.py::reduce ---
def reduce(x: Tensor, reduction: Optional[Literal["elementwise_mean", "sum", "none"]]) -> Tensor:
    """Reduces a given tensor by a given reduction method.

    Args:
        x: the tensor, which shall be reduced
        reduction:  a string specifying the reduction method ('elementwise_mean', 'none', 'sum')

    Return:
        reduced Tensor

    Raise:
        ValueError if an invalid reduction parameter was given

    """
    if reduction == "elementwise_mean":
        return torch.mean(x)
    if reduction == "none" or reduction is None:
        return x
    if reduction == "sum":
        return torch.sum(x)
    raise ValueError("Reduction parameter unknown.")

# --- from recommenders-team__recommenders::recommenders/models/vae/standard_vae.py::StandardVAE._vae_loss ---
def _vae_loss(
        self,
        x: torch.Tensor,
        x_bar: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """Calculate negative ELBO (NELBO)."""
        # Reconstruction error: sum over features, mean over batch
        # Matches TF: original_dim * binary_crossentropy averages over features then Keras averages over batch
        reconst_loss = torch.mean(
            torch.sum(F.binary_cross_entropy(x_bar, x, reduction="none"), dim=-1)
        )
        # Kullback–Leibler divergence
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1)
        )
        return reconst_loss + beta * kl_loss
