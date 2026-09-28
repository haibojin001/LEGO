# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg625::lightautoml.automl.base.AutoML+lightautoml.ml_algo.linear_sklearn.LinearLBFGS+lightautoml.pipelines.features.linear_pipeline.LinearFeatures
# name: lightautoml_torch_primitive
# summary: Uses lightautoml.automl.base.AutoML, lightautoml.ml_algo.linear_sklearn.LinearLBFGS, lightautoml.pipelines.features.linear_pipeline.LinearFeatures, lightautoml.pipelines.ml.base.MLPipeline across 2 repos
# anchor_symbols: ['lightautoml.automl.base.AutoML', 'lightautoml.ml_algo.linear_sklearn.LinearLBFGS', 'lightautoml.pipelines.features.linear_pipeline.LinearFeatures', 'lightautoml.pipelines.ml.base.MLPipeline', 'lightautoml.reader.base.PandasToPandasReader', 'torch.set_num_threads']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/uplift/utils.py::create_linear_automl ---
def create_linear_automl(
    task: Task,
    n_folds: int = 5,
    timeout: Optional[None] = None,
    n_reader_jobs: int = 1,
    cpu_limit: int = 4,
    # verbose: int = 0,
    random_state: int = 42,
):
    """Linear automl.

    Args:
        task: Task.
        n_folds: number of folds.
        timeout: Stub, not used.
        n_reader_jobs: Number of reader jobs.
        cpu_limit: CPU limit.
        random_state: random_state.

    Returns:
        automl:

    """
    torch.set_num_threads(cpu_limit)

    reader = PandasToPandasReader(
        task,
        cv=n_folds,
        random_state=random_state,
        n_jobs=n_reader_jobs,
    )
    pipe = LinearFeatures()
    model = LinearLBFGS()
    pipeline = MLPipeline(
        [model],
        pre_selection=None,
        features_pipeline=pipe,
        post_selection=None,
    )
    automl = AutoML(reader, [[pipeline]], skip_conn=False)  # , verbose=0)

    return automl

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/uplift/utils.py::create_linear_automl ---
def create_linear_automl(
    task: Task,
    n_folds: int = 5,
    timeout: Optional[None] = None,
    n_reader_jobs: int = 1,
    cpu_limit: int = 4,
    # verbose: int = 0,
    random_state: int = 42,
):
    """Linear automl.

    Args:
        task: Task.
        n_folds: number of folds.
        timeout: Stub, not used.
        n_reader_jobs: Number of reader jobs.
        cpu_limit: CPU limit.
        random_state: random_state.

    Returns:
        automl:

    """
    torch.set_num_threads(cpu_limit)

    reader = PandasToPandasReader(
        task,
        cv=n_folds,
        random_state=random_state,
        n_jobs=n_reader_jobs,
    )
    pipe = LinearFeatures()
    model = LinearLBFGS()
    pipeline = MLPipeline(
        [model],
        pre_selection=None,
        features_pipeline=pipe,
        post_selection=None,
    )
    automl = AutoML(reader, [[pipeline]], skip_conn=False)  # , verbose=0)

    return automl
