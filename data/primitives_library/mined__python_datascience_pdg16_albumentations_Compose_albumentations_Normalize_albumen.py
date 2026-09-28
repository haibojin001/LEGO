# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg16::albumentations.Compose+albumentations.Normalize+albumentations.Resize
# name: albumentations_allennlp_primitive
# summary: Uses albumentations.Compose, albumentations.Normalize, albumentations.Resize, albumentations.pytorch.ToTensorV2 across 32 repos
# anchor_symbols: ['albumentations.Compose', 'albumentations.Normalize', 'albumentations.Resize', 'albumentations.pytorch.ToTensorV2', 'allennlp.data.Batch', 'allennlp.data.Instance']
# observed in 32 repos: ['AxeldeRomblay__MLBox', 'Data-Centric-AI-Community__fg-data-profiling', 'DeepWisdom__AutoDL', 'EpistasisLab__scikit-rebate', 'allenai__allennlp']...

# --- from vaexio__vaex::tests/conftest.py::repickle.wrapper ---
def wrapper(obj):
        f = BytesIO()
        picked = pickle.dump(obj, f)
        f.seek(0)
        return pickle.load(f)

# --- from vaexio__vaex::tests/conftest.py::rebuild_dataset_pickle ---
def rebuild_dataset_pickle(ds):
    # pick and unpickle
    f = BytesIO()
    picked = pickle.dump(ds, f)
    f.seek(0)
    return pickle.load(f)

# --- from donnemartin__data-science-ipython-notebooks::scipy/thinkplot.py::Clf ---
def Clf():
    """Clears the figure and any hints that have been set."""
    global LOC
    LOC = None
    _Brewer.ClearIter()
    pyplot.clf()
    fig = pyplot.gcf()
    fig.set_size_inches(8, 6)

# --- from DeepWisdom__AutoDL::AutoDL_scoring_program/score.py::LearningCurve.save_figure ---
def save_figure(self, output_dir):
    alc, ax = self.plot()
    fig_name = get_fig_name(self.task_name)
    path_to_fig = os.path.join(output_dir, fig_name)
    plt.savefig(path_to_fig)
    plt.close()

# --- from edtechre__pybroker::src/pybroker/common.py::default_parallel ---
def default_parallel() -> Parallel:
    """Returns a :class:`joblib.Parallel` instance with ``n_jobs`` equal to
    the number of CPUs on the host machine.
    """
    return Parallel(n_jobs=os.cpu_count(), prefer="processes", backend="loky")

# --- from allenai__allennlp::allennlp/data/fields/multilabel_field.py::MultiLabelField.as_tensor ---
def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        tensor = torch.zeros(self._num_labels, dtype=torch.long)  # vector of zeros
        if self._label_ids:
            tensor.scatter_(0, torch.LongTensor(self._label_ids), 1)

        return tensor

# --- from rpy2__rpy2::rpy2-rinterface/src/rpy2/rinterface/tests/test_embedded_r.py::test_pickle ---
def test_pickle():
    x = rinterface.IntSexpVector([1, 2, 3])
    with tempfile.NamedTemporaryFile() as f:
        pickle.dump(x, f)
        f.flush()
        f.seek(0)
        x_again = pickle.load(f)
    identical = rinterface.baseenv['identical']
    assert identical(x, x_again)[0]

# --- from cleanlab__cleanlab::tests/datalab/datalab/test_datalab.py::TestDatalab.test_pickle ---
def test_pickle(self, lab, tmp_path):
        """Test that the class can be pickled."""
        pickle_file = os.path.join(tmp_path, "lab.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(lab, f)
        with open(pickle_file, "rb") as f:
            lab2 = pickle.load(f)

        assert lab2.label_name == "star"

# --- from deepchecks__deepchecks::tests/nlp/conftest.py::movie_reviews_data ---
def movie_reviews_data():
    """Dataset of single sentence samples."""
    download_nltk_resources()
    sentences = [' '.join(x) for x in movie_reviews.sents()]
    random.seed(42)

    train_data = TextData(random.sample(sentences, k=10_000))
    test_data = TextData(random.sample(sentences, k=10_000))
    return train_data, test_data

# --- from dswah__pyGAM::gen_imgs.py::cake_data_in_one ---
def cake_data_in_one():
    """Generate cake dataset visualization plot."""
    X, y = cake()

    gam = LinearGAM(fit_intercept=True)
    gam.gridsearch(X, y)

    XX = gam.generate_X_grid(term=0)

    plt.figure()
    plt.plot(gam.partial_dependence(term=0, X=XX))

    plt.title("LinearGAM")
    plt.savefig("imgs/pygam_cake_data.png", dpi=300)

# --- from dswah__pyGAM::gen_imgs.py::trees_data_custom ---
def trees_data_custom():
    """Generate trees dataset custom GAM visualization plot."""
    X, y = trees()
    gam = GAM(distribution="gamma", link="log")
    gam.gridsearch(X, y)

    plt.figure()
    plt.scatter(y, gam.predict(X))
    plt.xlabel("true volume")
    plt.ylabel("predicted volume")
    plt.savefig("imgs/pygam_custom.png", dpi=300)

# --- from smazzanti__mrmr::mrmr/pandas.py::parallel_df ---
def parallel_df(func, df, series, n_jobs):
    n_jobs = min(cpu_count(), len(df.columns)) if n_jobs == -1 else min(cpu_count(), n_jobs)
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(df.iloc[:, col_chunk], series)
        for col_chunk in col_chunks
    )
    return pd.concat(lst)
