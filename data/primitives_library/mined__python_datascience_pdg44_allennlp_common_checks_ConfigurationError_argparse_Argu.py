# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg44::allennlp.common.checks.ConfigurationError+argparse.ArgumentParser+attack_modules.attack_var.Attack
# name: allennlp_argparse_primitive
# summary: Uses allennlp.common.checks.ConfigurationError, argparse.ArgumentParser, attack_modules.attack_var.Attack, attack_modules.attack_var.SparseLayerAttack across 37 repos
# anchor_symbols: ['allennlp.common.checks.ConfigurationError', 'argparse.ArgumentParser', 'attack_modules.attack_var.Attack', 'attack_modules.attack_var.SparseLayerAttack', 'cleanlab.Datalab', 'cleanlab.benchmarking.noise_generation.generate_noise_matrix_from_trace']
# observed in 37 repos: ['AgnostiqHQ__covalent', 'BiomedSciAI__causallib', 'HazyResearch__meerkat', 'Lightning-AI__torchmetrics', 'OML-Team__open-metric-learning']...

# --- from microsoft__nni::nni/nas/hub/pytorch/modules/autoactivation.py::UnaryMin.forward ---
def forward(self, x):
        return torch.min(x, torch.zeros_like(x))

# --- from stellargraph__stellargraph::stellargraph/utils/hyperbolic.py::_tanh ---
def _tanh(x):
    return tf.tanh(tf.clip_by_value(x, -TANH_LIMIT, TANH_LIMIT))

# --- from awslabs__gluonts::src/gluonts/nursery/spliced_binned_pareto/genpareto.py::GenPareto.cdf ---
def cdf(self, x):
        x_shifted = torch.div(x, self.beta)
        u = 1 - torch.pow(1 + self.xi * x_shifted, -torch.reciprocal(self.xi))
        return u

# --- from awslabs__gluonts::src/gluonts/nursery/spliced_binned_pareto/genpareto.py::GenPareto.icdf ---
def icdf(self, value):
        x_shifted = torch.div(torch.pow(1 - value, -self.xi) - 1, self.xi)
        x = torch.mul(x_shifted, self.beta)
        return x

# --- from stellargraph__stellargraph::tests/layer/test_graph_classification.py::test_pooling.shifted_sum_pooling ---
def shifted_sum_pooling(tensor, mask):
            mask_floats = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
            return tf.math.reduce_sum(tf.multiply(mask_floats, shift + tensor), axis=1)

# --- from pydoit__doit::doit/cmd_base.py::ModuleTaskLoader.__init__ ---
def __init__(self, mod_dict):
        super().__init__()
        if inspect.ismodule(mod_dict):
            self.namespace = dict(inspect.getmembers(mod_dict))
        else:
            self.namespace = mod_dict

# --- from graspologic-org__graspologic::tests/test_orthogonal_procrustes.py::TestOrthogonalProcrustes.test_identity ---
def test_identity(self):
        Y = np.array([[1234, 19], [6798, 18], [9876, 17], [4321, 16]])

        aligner = OrthogonalProcrustes()
        aligner.fit(Y, Y)

        assert np.all(np.isclose(aligner.Q_, np.eye(2)))

# --- from insitro__redun::redun/tests/utils.py::get_docstring_owners_in_module.is_valid ---
def is_valid(obj):
        if getmodule(obj) == module:
            if ismethod(obj) or isfunction(obj):
                return not obj.__name__.startswith("_")
            if isclass(obj):
                return True
        return False

# --- from pykale__pykale::tests/pipeline/test_drugban_trainer.py::test_compute_entropy_weights ---
def test_compute_entropy_weights(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    logits = torch.rand(4, 1)
    weights = trainer._compute_entropy_weights(logits)
    assert weights.shape == torch.Size([4])  # Adjusted assertion

# --- from cleanlab__cleanlab::tests/internal/test_numerics.py::TestSoftmax.test_basic_softmax ---
def test_basic_softmax(self):
        input_arr = np.array([1.0, 2.0, 3.0])
        output = softmax(input_arr)
        expected_output = np.array([0.09003057, 0.24472847, 0.66524096])
        assert np.isclose(np.sum(output), 1.0)
        assert np.allclose(output, expected_output)

# --- from piskvorky__gensim::gensim/topic_coherence/text_analysis.py::WordOccurrenceAccumulator.analyze_text ---
def analyze_text(self, window, doc_num=None):
        self._slide_window(window, doc_num)
        mask = self._uniq_words[:-1]  # to exclude none token
        if mask.any():
            self._occurrences[mask] += 1
            self._counter.update(itertools.combinations(np.nonzero(mask)[0], 2))

# --- from ottogroup__palladium::palladium/tests/test_fit.py::TestWithParallelBackend.estimator ---
def estimator(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import LogisticRegression

        return GridSearchCV(
            LogisticRegression(solver='liblinear'),
            param_grid={'C': [0.001, 0.01]},
            cv=3,
            )
