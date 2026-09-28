# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg59::MySQLdb.connect+asyncio.Event+asyncio.create_task
# name: MySQLdb_asyncio_primitive
# summary: Uses MySQLdb.connect, asyncio.Event, asyncio.create_task, asyncio.ensure_future across 45 repos
# anchor_symbols: ['MySQLdb.connect', 'asyncio.Event', 'asyncio.create_task', 'asyncio.ensure_future', 'asyncio.gather', 'asyncio.get_running_loop']
# observed in 45 repos: ['AgnostiqHQ__covalent', 'BiomedSciAI__causallib', 'CamDavidsonPilon__lifetimes', 'EpistasisLab__scikit-rebate', 'JacksonWuxs__DaPy']...

# --- from spotify__chartify::chartify/_core/axes.py::DatetimeXMixin._convert_timestamp_to_epoch_ms ---
def _convert_timestamp_to_epoch_ms(timestamp):
        return (pd.to_datetime(timestamp) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")

# --- from xorbitsai__xorbits::python/xorbits/_mars/deploy/oscar/session.py::SyncSession._new_cancel_event ---
def _new_cancel_event(self):
        async def new_event():
            return asyncio.Event()

        return asyncio.run_coroutine_threadsafe(new_event(), self._loop).result()

# --- from xorbitsai__xorbits::python/xorbits/_mars/deploy/oscar/session.py::AsyncSession.destroy ---
async def destroy(self):
        coro = self._isolated_session.destroy()
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._loop))
        self.reset_default()

# --- from spotify__chartify::chartify/_core/axes.py::DatetimeXMixin._convert_timestamp_list_to_epoch_ms ---
def _convert_timestamp_list_to_epoch_ms(ts_list):
        return list(
            map(
                lambda x: ((pd.to_datetime(x) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")),
                ts_list,
            )
        )

# --- from ploomber__sklearn-evaluation::examples/calibration_curve_diff_sample_size.py::make_dataset ---
def make_dataset(n_samples):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=0,
    )
    return train_test_split(X, y, test_size=0.33, random_state=0)

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/swift/utils/torch_utils.py::Serializer.from_tensor ---
def from_tensor(obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.cpu().numpy()
        res = obj.tobytes()
        buffer_size = np.frombuffer(res[:8], dtype=np.int64)[0]
        res = res[8:]
        return pickle.loads(res[:buffer_size])

# --- from alteryx__featuretools::featuretools/tests/synthesis/test_dfs_method.py::test_accepts_pd_timedelta_training_window ---
def test_accepts_pd_timedelta_training_window(datetime_es):
    feature_matrix, _ = dfs(
        entityset=datetime_es,
        target_dataframe_name="transactions",
        cutoff_time=pd.Timestamp("2012-3-31 04:00"),
        training_window=pd.Timedelta(61, "D"),
    )

    assert (feature_matrix.index == [2, 3, 4]).all()

# --- from rpy2__rpy2::rpy2-robjects/src/rpy2/ipython/rmagic.py::GraphicsDeviceRaster.iter_displayed_path ---
def iter_displayed_path(self, path):
        filename_seq = sorted(
            glob(os.path.join(path, f'Rplots*.{self.extension}'))
        )

        for imgfile in filename_seq:
            if os.stat(imgfile).st_size >= 1000:
                img = self.display_filename(imgfile)
                _sync_console()
                yield img

# --- from modin-project__modin::modin/tests/pandas/test_rolling.py::test_rolling_axis_1_depr ---
def test_rolling_axis_1_depr():
    index = pandas.date_range("31/12/2000", periods=12, freq="min")
    data = {"A": range(12), "B": range(12)}
    modin_df = pd.DataFrame(data, index=index)
    with pytest.warns(
        FutureWarning,
        match="Support for axis=1 in DataFrame.rolling is deprecated",
    ):
        modin_df.rolling(window=3, axis=1)

# --- from apachecn__python_data_analysis_and_mining_action::chapter12/code.py::programmer_2 ---
def programmer_2():
    engine = create_engine(
        "mysql+pymysql://root:password@host:port/database_name?charset=utf8")
    sql = pd.read_sql("sql_gzdata", engine, chunksize=10000)

    for i in sql:
        d = i[["realIP", "fullURL"]]
        d = d[d["fullURL"].str.contains("\.html")].copy()
        d.to_sql("cleaned_gzdata", engine, index=False, if_exists="append")

# --- from deepchecks__deepchecks::deepchecks/core/serialization/common.py::read_matplot_figures ---
def read_matplot_figures() -> t.List[io.BytesIO]:
    """Return all active matplot figures."""
    output = []
    figures = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figures:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        output.append(buffer)
        fig.clear()
        plt.close(fig)
    return output

# --- from awslabs__gluonts::src/gluonts/dataset/jsonl.py::_encode_json_array ---
def _encode_json_array(arg: np.ndarray):
    if np.issubdtype(arg.dtype, int):
        return arg.tolist()

    if np.issubdtype(arg.dtype, np.floating):
        b = np.array(arg, dtype=object)
        b[np.isnan(arg)] = "Nan"
        b[np.isposinf(arg)] = "Infinity"
        b[np.isneginf(arg)] = "-Infinity"

        return b.tolist()

    return _encode_json_list(arg.tolist())
