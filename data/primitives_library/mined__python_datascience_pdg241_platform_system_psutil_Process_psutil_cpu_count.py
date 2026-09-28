# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg241::platform.system+psutil.Process+psutil.cpu_count
# name: platform_psutil_primitive
# summary: Uses platform.system, psutil.Process, psutil.cpu_count across 5 repos
# anchor_symbols: ['platform.system', 'psutil.Process', 'psutil.cpu_count']
# observed in 5 repos: ['alteryx__featuretools', 'capitalone__DataProfiler', 'cleanlab__cleanlab', 'cleanlab__cleanvision', 'nfstream__nfstream']...

# --- from nfstream__nfstream::nfstream/utils.py::available_cpus_count ---
def available_cpus_count():
    if platform.system() == "Linux":
        return len(psutil.Process().cpu_affinity())
    return psutil.cpu_count(logical=True)

# --- from cleanlab__cleanvision::src/cleanvision/utils/utils.py::get_max_n_jobs ---
def get_max_n_jobs() -> int:
    n_jobs = None
    if PSUTIL_EXISTS:
        n_jobs = psutil.cpu_count(logical=False)  # physical cores
    if not n_jobs:
        # either psutil does not exist
        # or psutil can return None when physical cores cannot be determined
        # switch to logical cores
        n_jobs = multiprocessing.cpu_count()
    return n_jobs

# --- from nfstream__nfstream::nfstream/utils.py::set_affinity ---
def set_affinity(idx):
    """CPU affinity setter"""
    if platform.system() == "Linux":
        c_cpus = psutil.Process().cpu_affinity()
        temp = list(chunks_of_list(c_cpus, 2))
        x = len(temp)
        try:
            psutil.Process().cpu_affinity(list(temp[idx % x]))
        except OSError as err:
            print("WARNING: failed to set CPU affinity ({err})".format(err))

# --- from alteryx__featuretools::featuretools/computational_backends/utils.py::n_jobs_to_workers ---
def n_jobs_to_workers(n_jobs):
    try:
        cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpus = psutil.cpu_count()

    # Taken from sklearn parallel_backends code
    # https://github.com/scikit-learn/scikit-learn/blob/27bbdb570bac062c71b3bb21b0876fd78adc9f7e/sklearn/externals/joblib/_parallel_backends.py#L120
    if n_jobs < 0:
        workers = max(cpus + 1 + n_jobs, 1)
    else:
        workers = min(n_jobs, cpus)

    assert workers > 0, "Need at least one worker"
    return workers

# --- from alteryx__featuretools::featuretools/tests/computational_backend/test_calculate_feature_matrix.py::test_n_jobs ---
def test_n_jobs():
    try:
        cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:  # pragma: no cover
        cpus = psutil.cpu_count()

    assert n_jobs_to_workers(1) == 1
    assert n_jobs_to_workers(-1) == cpus
    assert n_jobs_to_workers(cpus) == cpus
    assert n_jobs_to_workers((cpus + 1) * -1) == 1
    if cpus > 1:
        assert n_jobs_to_workers(-2) == cpus - 1

    error_text = "Need at least one worker"
    with pytest.raises(AssertionError, match=error_text):
        n_jobs_to_workers(0)

# --- from capitalone__DataProfiler::dataprofiler/profilers/profiler_utils.py::suggest_pool_size ---
def suggest_pool_size(data_size: int = None, cols: int = None) -> int | None:
    """
    Suggest the pool size based on resources.

    :param data_size: size of the dataset
    :type data_size: int
    :param cols: columns of the dataset
    :type cols: int
    :return suggested_pool_size: suggested pool size
    :rtype suggested_pool_size: int
    """
    # Return if there's no data_size
    if data_size is None:
        return None

    try:
        # Determine safest level of processes based on memory
        mb = 1000000
        svmem = psutil.virtual_memory()
        max_pool_mem = (data_size * 50) / (svmem.available / mb)
    except NotImplementedError:
        max_pool_mem = 4

    try:
        # Determine safest level of processes based on CPUs
        max_pool_cpu = psutil.cpu_count() - 1
    except NotImplementedError:
        max_pool_cpu = 1

    # Limit to cols if less than threads
    suggested_pool_size = min(max_pool_mem, max_pool_cpu)
    if cols is not None:
        suggested_pool_size = min(suggested_pool_size, cols)

    return int(suggested_pool_size)

# --- from cleanlab__cleanlab::cleanlab/experimental/label_issues_batched.py::LabelInspector.__init__ ---
def __init__(
        self,
        *,
        num_class: int,
        store_results: bool = True,
        verbose: bool = True,
        quality_score_kwargs: Optional[dict] = None,
        num_issue_kwargs: Optional[dict] = None,
        n_jobs: Optional[int] = 1,
    ):
        if quality_score_kwargs is None:
            quality_score_kwargs = {}
        if num_issue_kwargs is None:
            num_issue_kwargs = {}

        self.num_class = num_class
        self.store_results = store_results
        self.verbose = verbose
        self.quality_score_kwargs = quality_score_kwargs  # extra arguments for ``rank.get_label_quality_scores()`` to control label quality scoring
        self.num_issue_kwargs = num_issue_kwargs  # extra arguments for ``count.num_label_issues()`` to control estimation of the number of label issues (only supported argument for now is: `estimation_method`).
        self.off_diagonal_calibrated = False
        if num_issue_kwargs.get("estimation_method") == "off_diagonal_calibrated":
            # store extra attributes later needed for calibration:
            self.off_diagonal_calibrated = True
            self.prune_counts = np.zeros(self.num_class)
            self.class_counts = np.zeros(self.num_class)
            self.normalization = np.zeros(self.num_class)
        else:
            self.prune_count = 0  # number of label issues estimated based on data seen so far (only used when estimation_method is not calibrated)

        if self.store_results:
            self.label_quality_scores: List[float] = []

        self.confident_thresholds = np.zeros(
            (num_class,)
        )  # current estimate of thresholds based on data seen so far
        self.examples_per_class = np.zeros(
            (num_class,)
        )  # current counts of examples with each given label seen so far
        self.examples_processed_thresh = (
            0  # number of examples seen so far for estimating thresholds
        )
        self.examples_processed_quality = 0  # number of examples seen so far for estimating label quality and number of label issues
        # Determine number of cores for multiprocessing:
        self.n_jobs: Optional[int] = None
        os_name = platform.system()
        if os_name != "Linux":
            self.n_jobs = 1
            if n_jobs is not None and n_jobs != 1 and self.verbose:
                print(
                    "n_jobs is overridden to 1 because multiprocessing is only supported for Linux."
                )
        elif n_jobs is not None:
            self.n_jobs = n_jobs
        else:
            if PSUTIL_EXISTS:
                self.n_jobs = psutil.cpu_count(logical=False)  # physical cores
            if not self.n_jobs:
                # switch to logical cores
                self.n_jobs = mp.cpu_count()
                if self.verbose:
                    print(
                        f"Multiprocessing will default to using the number of logical cores ({self.n_jobs}). To default to number of physical cores: pip install psutil"
                    )
