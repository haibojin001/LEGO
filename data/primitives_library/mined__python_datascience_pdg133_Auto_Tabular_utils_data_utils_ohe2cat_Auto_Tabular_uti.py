# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg133::Auto_Tabular.utils.data_utils.ohe2cat+Auto_Tabular.utils.log_utils.log+argparse.ArgumentParser
# name: Auto_Tabular_argparse_primitive
# summary: Uses Auto_Tabular.utils.data_utils.ohe2cat, Auto_Tabular.utils.log_utils.log, argparse.ArgumentParser, catboost.CatBoostClassifier across 18 repos
# anchor_symbols: ['Auto_Tabular.utils.data_utils.ohe2cat', 'Auto_Tabular.utils.log_utils.log', 'argparse.ArgumentParser', 'catboost.CatBoostClassifier', 'category_encoders.OneHotEncoder', 'category_encoders.OrdinalEncoder']
# observed in 18 repos: ['DeepWisdom__AutoDL', 'OML-Team__open-metric-learning', 'awslabs__gluonts', 'deepchecks__deepchecks', 'google__uncertainty-baselines']...

# --- from kyleskom__NBA-Machine-Learning-Sports-Betting::src/Train-Models/XGBoost_Model_ML.py::load_dataset ---
def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)

# --- from kyleskom__NBA-Machine-Learning-Sports-Betting::src/Train-Models/XGBoost_Model_UO.py::load_dataset ---
def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)

# --- from microsoft__RD-Agent::rdagent/scenarios/kaggle/experiment/templates/digit-recognizer/model/model_nn.py::predict ---
def predict(model, X):
    X_tensor = torch.tensor(X.values, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy().reshape(-1, 1)

# --- from ploomber__sklearn-evaluation::tests/test_models_comparer.py::test_compare_models ---
def test_compare_models(heart_dataset, tmp_directory):
    X_train, X_test, y_train, y_test = _get_split_data(heart_dataset)

    model_a = RandomForestClassifier()
    model_a.fit(X_train, y_train)

    model_b = DecisionTreeClassifier()
    model_b.fit(X_train, y_train)

    report = compare_models(model_a, model_b, X_test, y_test)
    report.save("example-compare-report.html")

# --- from ploomber__sklearn-evaluation::tests/test_models_comparer.py::test_functions_with_none_inputs ---
def test_functions_with_none_inputs():
    model_a = RandomForestClassifier()
    model_b = DecisionTreeClassifier()

    me = ModelsComparer(model_a, model_b)

    me.precision_and_recall(None, None)
    me.auc(None, None)
    me.computation(None)
    me.calibration(None, None)
    me.add_combined_cm(None, None)
    me.add_combined_pr(None, None)

    assert len(me.evaluation_state.keys()) == 0

# --- from deepchecks__deepchecks::tests/tabular/checks/model_evaluation/multi_model_performance_report_test.py::classification_models ---
def classification_models(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    model2 = RandomForestClassifier(random_state=0)
    model2.fit(train.data[train.features], train.data[train.label_name])
    model3 = DecisionTreeClassifier(random_state=0)
    model3.fit(train.data[train.features], train.data[train.label_name])
    return train, test, model, model2, model3

# --- from deepchecks__deepchecks::tests/tabular/checks/model_evaluation/multi_model_performance_report_test.py::regression_models ---
def regression_models(diabetes_split_dataset_and_model):
    train, test, model = diabetes_split_dataset_and_model
    model2 = RandomForestRegressor(random_state=0)
    model2.fit(train.data[train.features], train.data[train.label_name])
    model3 = DecisionTreeRegressor(random_state=0)
    model3.fit(train.data[train.features], train.data[train.label_name])
    return train, test, model, model2, model3

# --- from microsoft__RD-Agent::rdagent/scenarios/kaggle/experiment/templates/playground-series-s4e5/fea_share_preprocess.py::preprocess_fit ---
def preprocess_fit(X_train: pd.DataFrame):
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ["int64", "float64"]]

    numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
        ]
    )

    preprocessor.fit(X_train)

    return preprocessor, numerical_cols

# --- from iterative__mlem::tests/contrib/test_requirements.py::test_unix_requirement ---
def test_unix_requirement(capsys):
    np_payload = np.linspace(0, 2, 5).reshape((-1, 1))
    data_np = lgb.Dataset(
        np_payload,
        label=np_payload.reshape((-1,)).tolist(),
        free_raw_data=False,
    )
    booster = lgb.train({}, data_np, 1)
    model = MlemModel.from_obj(booster, sample_data=data_np)
    builder = RequirementsBuilder(req_type="unix")
    builder.build(model)
    captured = capsys.readouterr()
    assert str(captured.out).endswith(
        "\n".join(model.requirements.to_unix()) + "\n"
    )

# --- from microsoft__nni::docs/source/tutorials/darts.py::evaluate_model ---
def evaluate_model(model, cuda=False):
    device = torch.device('cuda' if cuda else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            _, predict = torch.max(logits, 1)
            correct += (predict == targets).sum().cpu().item()
            total += targets.size(0)
    print('Accuracy:', correct / total)
    return correct / total

# --- from microsoft__nni::examples/tutorials/darts.py::evaluate_model ---
def evaluate_model(model, cuda=False):
    device = torch.device('cuda' if cuda else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            _, predict = torch.max(logits, 1)
            correct += (predict == targets).sum().cpu().item()
            total += targets.size(0)
    print('Accuracy:', correct / total)
    return correct / total

# --- from awslabs__gluonts::src/gluonts/nursery/daf/tslib/engine/evaluator.py::Evaluator.evaluate ---
def evaluate(self) -> None:
        device = "cpu" if self.cuda_device < 0 else f"cuda:{self.cuda_device}"
        prog_bar = tqdm(
            desc=f"({device}) test",
            total=int(math.ceil(self.dataset.test_size / self.batch_size)),
            unit="batch",
        )
        _ = self.model.eval()
        for batch, data in enumerate(self.test_loader):
            with pt.no_grad():
                self._test(*data)
            prog_bar.update(1)
            if self.debug and batch > 0:
                break
        prog_bar.close()
