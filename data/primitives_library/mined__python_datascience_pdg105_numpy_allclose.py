# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg105::numpy.allclose
# name: numpy_primitive
# summary: Uses numpy.allclose across 6 repos
# anchor_symbols: ['numpy.allclose']
# observed in 6 repos: ['NannyML__nannyml', 'awslabs__gluonts', 'dmbee__seglearn', 'graspologic-org__graspologic', 'microsoft__nni']...

# --- from microsoft__nni::examples/nas/legacy/textnas/ops.py::LinearCombine.forward ---
def forward(self, seq):
        nw = F.softmax(self.w, dim=0)
        seq = torch.mul(seq, nw)
        seq = torch.sum(seq, dim=0)
        return seq

# --- from awslabs__gluonts::src/gluonts/shell/util.py::forecaster_type_by_name ---
def forecaster_type_by_name(name: str) -> Forecaster:
    forecaster = pydoc.locate(name)

    ForecasterNotFound.guard(
        forecaster is not None,
        f'Cannot locate estimator with classname "{name}".',
    )

    return cast(Forecaster, forecaster)

# --- from graspologic-org__graspologic::tests/test_utils.py::TestToLaplace.test_to_laplacian_DAD ---
def test_to_laplacian_DAD(self):
        expected_L_normed = [
            [0, 1 / sqrt(2), 0],
            [1 / sqrt(2), 0, 1 / sqrt(2)],
            [0, 1 / sqrt(2), 0],
        ]

        L_normed = gus.to_laplacian(self.A, form="DAD")

        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))

# --- from graspologic-org__graspologic::tests/test_utils.py::TestToLaplace.test_to_laplacian_RDAD ---
def test_to_laplacian_RDAD(self):
        expected_L_normed = [
            [0, 3 / sqrt(70), 0],
            [3 / sqrt(70), 0, 3 / sqrt(70)],
            [0, 3 / sqrt(70), 0],
        ]

        L_normed = gus.to_laplacian(self.A, form="R-DAD")

        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))

# --- from awslabs__gluonts::src/gluonts/mx/trainer/callback.py::TerminateOnNaN.on_train_epoch_end ---
def on_train_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: nn.HybridBlock,
        trainer: gluon.Trainer,
    ) -> bool:
        if math.isnan(epoch_loss):
            logging.warning(
                "TerminateOnNaN Callback initiated stop of training at epoch"
                f" {epoch_no}."
            )
            return False
        return True

# --- from piskvorky__gensim::gensim/test/test_similarities.py::TestSparseTermSimilarityMatrix.test_inner_product_vector_corpus_false_true ---
def test_inner_product_vector_corpus_false_true(self):
        """Test the inner product between a vector and a corpus with the (False, True) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(False, True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

# --- from piskvorky__gensim::gensim/test/test_similarities.py::TestSparseTermSimilarityMatrix.test_inner_product_vector_corpus_true_false ---
def test_inner_product_vector_corpus_true_false(self):
        """Test the inner product between a vector and a corpus with the (True, False) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(True, False))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

# --- from dmbee__seglearn::seglearn/preprocessing.py::TargetRunLengthEncoder._rle ---
def _rle(self, a):
        """
        rle implementation credit to Thomas Browne from his SOF post Sept 2015

        Parameters
        ----------
        a : array, shape[n,]
            input vector

        Returns
        -------
        z : array, shape[nt,]
            run lengths
        p : array, shape[nt,]
            start positions of each run
        ar : array, shape[nt,]
            values for each run
        """
        ia = np.asarray(a)
        n = len(ia)
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return z, p, ia[i]

# --- from NannyML__nannyml::nannyml/performance_estimation/confidence_based/metrics.py::estimate_ap ---
def estimate_ap(
    calibrated_y_pred_proba: Union[pd.Series, np.ndarray],
    uncalibrated_y_pred_proba: Union[pd.Series, np.ndarray]
) -> float:
    """Estimates the AP metric.

    Parameters
    ----------
    calibrated_y_pred_proba: Union[pd.Series, np.ndarray]
        Calibrated probability estimates of the sample for each class in the model.
    uncalibrated_y_pred_proba: Union[pd.Series, np.ndarray]
        Raw probability estimates of the sample for each class in the model.

    Returns
    -------
    metric: float
        Estimated AP score.
    """
    # TODO: Update Code to only accept np.ndarray (and add checkand remove code below)
    calibrated_y_pred_proba = np.asarray(calibrated_y_pred_proba)
    uncalibrated_y_pred_proba = np.asarray(uncalibrated_y_pred_proba)

    descending_order_index = np.argsort(uncalibrated_y_pred_proba)[::-1]
    calibrated_y_pred_proba = calibrated_y_pred_proba[descending_order_index]

    tps = np.cumsum(calibrated_y_pred_proba)
    fps = 1 + np.arange(calibrated_y_pred_proba.shape[0]) - tps
    tps = np.round(tps, 5)
    fps = np.round(fps, 5)
    ps = np.arange(1, tps.shape[0] + 1)

    precision = tps / ps
    # we add an element for (tps, fps) = (0,0) after the division to avoid error
    precision = np.r_[1, precision]
    tps = np.r_[0, tps]
    recall = tps / tps[-1]
    # reverse so (0,1) is last element
    # recall is descending from 1 to 0
    precision = precision[::-1]
    recall = recall[::-1]

    # actual AP calculation
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py#L236
    # non unique values will be eliminated because diff will be 0!
    metric = -np.sum(np.diff(recall) * precision[:-1])
    return metric
