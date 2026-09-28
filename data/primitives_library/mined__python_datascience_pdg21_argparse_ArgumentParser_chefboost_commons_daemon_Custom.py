# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg21::argparse.ArgumentParser+chefboost.commons.daemon.CustomPool+chefboost.commons.module.load_module
# name: argparse_chefboost_primitive
# summary: Uses argparse.ArgumentParser, chefboost.commons.daemon.CustomPool, chefboost.commons.module.load_module, collections.OrderedDict across 21 repos
# anchor_symbols: ['argparse.ArgumentParser', 'chefboost.commons.daemon.CustomPool', 'chefboost.commons.module.load_module', 'collections.OrderedDict', 'collections.UserDict', 'collections.defaultdict']
# observed in 21 repos: ['CamDavidsonPilon__lifelines', 'DeepWisdom__AutoDL', 'HazyResearch__meerkat', 'HoloClean__holoclean', 'Lightning-AI__torchmetrics']...

# --- from zama-ai__concrete-ml::tests/virtual_lib/test_virtual_lib.py::test_torch_matmul_fhe_simulation.g ---
def g(x, weights):
        return numpy.rint(numpy.sin(f(x, weights))).astype(numpy.int64)

# --- from OML-Team__open-metric-learning::oml/transforms/images/torchvision.py::get_normalisation_torch ---
def get_normalisation_torch(mean: TNormParam = MEAN, std: TNormParam = STD) -> Compose:
    return Compose([ToTensor(), Normalize(mean=mean, std=std)])

# --- from pykale__pykale::tests/pipeline/test_base_nn_trainer.py::data ---
def data():
    # Create dummy data for testing. The dimension is following the CIFAR10 dataset.
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 2, (8,))
    return TensorDataset(x, y)

# --- from pykale__pykale::tests/pipeline/test_fewshot_trainer.py::data ---
def data():
    # Create dummy data for testing. The dimension is following the CIFAR10 dataset.
    x = torch.randn(5, 20, 3, 84, 84)
    y = torch.randint(0, 10, (5,))
    return TensorDataset(x, y)

# --- from zama-ai__concrete-ml::conftest.py::function_to_seed_torch ---
def function_to_seed_torch(seed):
    """Seed torch, for determinism."""

    # Seed torch with something which is seed by pytest-randomly
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

# --- from OML-Team__open-metric-learning::oml/transforms/images/torchvision.py::get_normalisation_resize_torch ---
def get_normalisation_resize_torch(im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD) -> Compose:
    return Compose([t.Resize(size=(im_size, im_size), antialias=True), ToTensor(), Normalize(mean=mean, std=std)])

# --- from SforAiDl__KD_Lib::KD_Lib/KD/text/BERT2LSTM/utils.py::to_dataset ---
def to_dataset(x, y_real):
    torch_x = torch.tensor(x, dtype=torch.long)
    # torch_y = torch.tensor(y, dtype=torch.float)
    torch_real_y = torch.tensor(y_real, dtype=torch.long)
    return TensorDataset(torch_x, torch_real_y)

# --- from microsoft__nni::examples/trials/sklearn/classification/main.py::load_data ---
def load_data():
    '''Load dataset, use 20newsgroups dataset'''
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, random_state=99, test_size=0.25)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return X_train, X_test, y_train, y_test

# --- from microsoft__nni::examples/tutorials/scripts/trial_sklearn.py::load_data ---
def load_data():
    """Load dataset, use 20newsgroups dataset"""
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, random_state=99, test_size=0.25)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return X_train, X_test, y_train, y_test

# --- from reiinakano__xcessiv::xcessiv/tests/test_functions.py::TestVerifyEstimatorClass.test_non_serializable_parameters ---
def test_non_serializable_parameters(self):
        pipeline = Pipeline([('pca', PCA()), ('rf', RandomForestClassifier())])
        performance_dict, hyperparameters = functions.verify_estimator_class(
            pipeline,
            'predict_proba',
            dict(Accuracy=self.source),
            self.dataset_properties
        )
        assert functions.is_valid_json(hyperparameters)

# --- from HazyResearch__meerkat::meerkat/ops/search.py::_torch_search ---
def _torch_search(
    query: "torch.Tensor", by: "torch.Tensor", metric: str, k: int
) -> "torch.Tensor":
    with torch.no_grad():
        if len(query.shape) == 1:
            query = query.unsqueeze(0)

        if metric == "dot":
            scores = (by @ query.T).squeeze()
        else:
            raise ValueError("")

        scores, indices = torch.topk(scores, k=k)
    return scores.to("cpu").numpy(), indices.to("cpu")

# --- from awslabs__gluonts::src/gluonts/model/trivial/identity.py::IdentityPredictor.predict_item ---
def predict_item(self, item: DataEntry) -> Forecast:
        prediction = item["target"][-self.prediction_length :]
        samples = np.broadcast_to(
            array=np.expand_dims(prediction, 0),
            shape=(self.num_samples, self.prediction_length),
        )

        return SampleForecast(
            samples=samples,
            start_date=forecast_start(item),
            item_id=item.get(FieldName.ITEM_ID),
        )
