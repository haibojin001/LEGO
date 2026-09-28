# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg265::airflow.providers.amazon.aws.hooks.s3.S3Hook+autograd.elementwise_grad+autograd.hessian
# name: airflow_autograd_primitive
# summary: Uses airflow.providers.amazon.aws.hooks.s3.S3Hook, autograd.elementwise_grad, autograd.hessian, autograd.numpy.atleast_1d across 8 repos
# anchor_symbols: ['airflow.providers.amazon.aws.hooks.s3.S3Hook', 'autograd.elementwise_grad', 'autograd.hessian', 'autograd.numpy.atleast_1d', 'autograd.numpy.log', 'autograd.numpy.maximum']
# observed in 8 repos: ['CamDavidsonPilon__lifelines', 'CamDavidsonPilon__lifetimes', 'HazyResearch__meerkat', 'ahmetozlu__tensorflow_object_counting_api', 'alteryx__evalml']...

# --- from snorkel-team__snorkel::test/labeling/lf/test_nlp.py::TestNLPLabelingFunction.test_labeling_function_serialize ---
def test_labeling_function_serialize(self) -> None:
        lf = NLPLabelingFunction(name="my_lf", f=has_person_mention, pre=[combine_text])
        lf_load = dill.loads(dill.dumps(lf))
        self._run_lf(lf_load)

# --- from CamDavidsonPilon__lifelines::lifelines/tests/test_estimation.py::TestKaplanMeierFitter.test_sort_doesnt_affect_kmf ---
def test_sort_doesnt_affect_kmf(self, sample_lifetimes):
        T, _ = sample_lifetimes
        kmf = KaplanMeierFitter()
        assert_frame_equal(kmf.fit(T).survival_function_, kmf.fit(sorted(T)).survival_function_)

# --- from CamDavidsonPilon__lifelines::lifelines/tests/test_estimation.py::TestKaplanMeierFitter.test_kaplan_meier_no_censorship ---
def test_kaplan_meier_no_censorship(self, sample_lifetimes):
        T, _ = sample_lifetimes
        kmf = KaplanMeierFitter()
        kmf.fit(T)
        npt.assert_almost_equal(kmf.survival_function_.values, self.kaplan_meier(T))

# --- from CamDavidsonPilon__lifetimes::tests/test_estimation.py::TestParetoNBDFitter.test_overflow_error ---
def test_overflow_error(self):

        ptf = lt.ParetoNBDFitter()
        params = np.array([10.465, 7.98565181e-03, 3.0516, 2.820])
        freq = np.array([400.0, 500.0, 500.0])
        rec = np.array([5.0, 1.0, 4.0])
        age = np.array([6.0, 37.0, 37.0])
        assert all([r < 0 and not np.isinf(r) and not pd.isnull(r) for r in ptf._log_A_0(params, freq, rec, age)])

# --- from ahmetozlu__tensorflow_object_counting_api::utils/object_tracking_module/tracking_layer.py::Tracker.kalman_filter ---
def kalman_filter(self, z):       
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        #Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S)) # Kalman gain
        y = z - dot(self.H, x) # residual
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = x.astype(int)

# --- from HazyResearch__meerkat::tests/meerkat/columns/test_common.py::test_pickle ---
def test_pickle(column_testbed):
    import dill as pickle  # needed so that it works with lambda functions

    # important for dataloader
    col = column_testbed.col
    buf = pickle.dumps(col)
    new_col = pickle.loads(buf)

    assert isinstance(new_col, type(col))

    if isinstance(new_col, DeferredColumn):
        # the lambda function isn't exactly the same after reading
        new_col.data.fn = col.data.fn
    assert col.is_equal(new_col)

# --- from CamDavidsonPilon__lifetimes::tests/test_estimation.py::TestParetoNBDFitter.test_conditional_probability_alive_overflow_error ---
def test_conditional_probability_alive_overflow_error(self):
        ptf = lt.ParetoNBDFitter()
        ptf.params_ = pd.Series(*([10.465, 7.98565181e-03, 3.0516, 2.820], ["r", "alpha", "s", "beta"]))
        freq = np.array([40.0, 50.0, 50.0])
        rec = np.array([5.0, 1.0, 4.0])
        age = np.array([6.0, 37.0, 37.0])
        assert all(
            [
                r <= 1 and r >= 0 and not np.isinf(r) and not pd.isnull(r)
                for r in ptf.conditional_probability_alive(freq, rec, age)
            ]
        )

# --- from modin-project__modin::modin/tests/pandas/dataframe/test_default.py::test_setattr_axes ---
def test_setattr_axes():
    # Test that setting .index or .columns does not warn
    df = pd.DataFrame([[1, 2], [3, 4]])
    with warnings.catch_warnings():
        if get_current_execution() != "BaseOnPython":
            # In BaseOnPython, setting columns raises a warning because get_axis
            #  defaults to pandas.
            warnings.simplefilter("error")
        df.index = ["foo", "bar"]
        # Check that ensure_index was called
        pd.testing.assert_index_equal(df.index, pandas.Index(["foo", "bar"]))

        df.columns = [9, 10]
        pd.testing.assert_index_equal(df.columns, pandas.Index([9, 10]))

# --- from alteryx__evalml::evalml/tests/component_tests/test_imputer.py::test_imputer_empty_data ---
def test_imputer_empty_data(data_type, make_data_type):
    X = pd.DataFrame()
    y = pd.Series()
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    expected = pd.DataFrame(index=pd.Index([]), columns=pd.Index([]))
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    assert_frame_equal(
        transformed,
        expected,
        check_column_type=False,
        check_index_type=False,
    )

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(
        transformed,
        expected,
        check_column_type=False,
        check_index_type=False,
    )

# --- from deepchecks__deepchecks::examples/cicd/airflow.py::model_training_dag ---
def model_training_dag():

    @short_circuit_task
    def validate_data(**context):
        from deepchecks.tabular.suites import data_integrity
        from deepchecks.tabular import Dataset

        hook = S3Hook('aws_connection')
        file_name = hook.download_file(key=context['params']['data_key'], bucket_name=context['params']['bucket'],
                                       local_path='.')
        data_df = pd.read_csv(file_name)
        dataset = Dataset(data_df, label='label', cat_features=[])
        suite_result = data_integrity().run(dataset)
        suite_result.save_as_html('data_validation.html')
        hook.load_file(
            filename='data_validation.html',
            key='results/data_validation.html',
            bucket_name=context['params']['bucket'],
            replace=True
        )
        context['ti'].xcom_push(key='data', value=file_name)
        return suite_result.passed()

    @short_circuit_task
    def validate_train_test_split(**context):
        from deepchecks.tabular.suites import train_test_validation
        from deepchecks.tabular import Dataset

        data = pd.read_csv(context['ti'].xcom_pull(key='data'))
        train_df, test_df = data.iloc[:len(data) // 2], data.iloc[len(data) // 2:]
        train_df.to_csv(context['params']['train_path'])
        test_df.to_csv(context['params']['test_path'])

        train = Dataset(train_df, label='label', cat_features=[])
        test = Dataset(test_df, label='label', cat_features=[])
        suite_result = train_test_validation().run(train_dataset=train, test_dataset=test)
        suite_result.save_as_html('split_validation.html')
        hook = S3Hook('aws_connection')
        hook.load_file(
            filename='split_validation.html',
            key='results/split_validation.html',
            bucket_name=context['params']['bucket'],
            replace=True
        )
        return suite_result.passed()

    @task
    def train_model(**context):
        train_df = pd.read_csv(context['params']['train_path'])
        # Train model and upload to s3
        model = ...

        joblib.dump(model, context['params']['model_path'])
        hook = S3Hook('aws_connection')
        hook.load_file(
            filename=context['params']['model_path'],
            key='results/model.joblib',
            bucket_name=context['params']['bucket'],
            replace=True
        )

    @task
    def validate_model_performance(**context):
        from deepchecks.tabular.suites import model_evaluation
        from deepchecks.tabular import Dataset

        train_df = pd.read_csv(context['params']['train_path'])
        test_df = pd.read_csv(context['params']['test_path'])
        model = joblib.load(context['params']['model_path'])

        train = Dataset(train_df, label='label', cat_features=[])
        test = Dataset(test_df, label='label', cat_features=[])
        suite_result = model_evaluation().run(train_dataset=train, test_dataset=test, model=model)
        suite_result.save_as_html('model_validation.html')
        hook = S3Hook('aws_connection')
        hook.load_file(
            filename='model_validation.html',
            key='results/model_validation.html',
            bucket_name=context['params']['bucket'],
            replace=True
        )
        return suite_result.passed()

    validate_data() >> validate_train_test_split() >> train_model() >> validate_model_performance()
