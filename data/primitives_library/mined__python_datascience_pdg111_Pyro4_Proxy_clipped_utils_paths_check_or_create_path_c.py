# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg111::Pyro4.Proxy+clipped.utils.paths.check_or_create_path+cv2.imencode
# name: Pyro4_clipped_primitive
# summary: Uses Pyro4.Proxy, clipped.utils.paths.check_or_create_path, cv2.imencode, cv2.imread across 6 repos
# anchor_symbols: ['Pyro4.Proxy', 'clipped.utils.paths.check_or_create_path', 'cv2.imencode', 'cv2.imread', 'cv2.resize', 'ffcv.DatasetWriter']
# observed in 6 repos: ['libffcv__ffcv', 'microsoft__nni', 'piskvorky__gensim', 'polyaxon__traceml', 'pykale__pykale']...

# --- from run-house__kubetorch::services/data_store/locks.py::RWLock.__init__ ---
def __init__(self):
        self._lock = Lock()
        self._cond = Condition(self._lock)
        self._readers = 0
        self._writer = False

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert_operators.py::TestOperators.test_basic_abs ---
def test_basic_abs(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = torch.abs(x)
                return out
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, ))

# --- from libffcv__ffcv::tests/test_cuda_nonblocking.py::run_experiment_cuda ---
def run_experiment_cuda(weight, loader, sync=False):
    total = 0.
    X = ch.empty(BATCH, SIZE, device=weight.device)
    for X_bool, _, __ in tqdm(loader):
        if sync: ch.cuda.synchronize()
        X.copy_(X_bool)
        total += X @ weight
        total += X @ weight
        total += X @ weight

    return total.sum(0)

# --- from run-house__kubetorch::python_client/kubetorch/serving/pdb_websocket.py::WebSocketIO.__init__ ---
def __init__(self, connection_timeout: int = 300):
        self.input_queue = queue.Queue()  # WebSocket -> PDB
        self.output_queue = queue.Queue()  # PDB -> WebSocket
        self._closed = False
        self._connected = threading.Event()
        self._line_buffer = ""  # Buffer for accumulating characters into lines
        self._connection_timeout = connection_timeout

# --- from libffcv__ffcv::tests/test_cuda_nonblocking.py::test_cuda ---
def test_cuda():
    weight = ch.randn(SIZE, SIZE).cuda()
    async_1 = run_cuda(weight, False)
    sync_1 = run_cuda(weight, True)
    sync_2 = run_cuda(weight, True)
    print(async_1)
    print(sync_1)
    print(sync_2)
    print(ch.abs(sync_1 - sync_2).max())
    print(ch.abs(sync_1 - async_1).max())
    assert ch.abs(sync_1 - sync_2).max().cpu().item() < float(WORKERS), 'Sync-sync mismatch'
    assert ch.abs(async_1 - sync_1).max().cpu().item() < float(WORKERS), 'Async-sync mismatch'

# --- from polyaxon__traceml::traceml/traceml/serialization/writer.py::EventFileWriter.__init__ ---
def __init__(self, run_path: str, max_queue_size: int = 20, flush_secs: int = 10):
        """Creates a `EventFileWriter`.

        Args:
          run_path: A string. Directory where events files will be written.
          max_queue_size: Integer. Size of the queue for pending events and summaries.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk.
        """
        super().__init__(run_path=run_path)

        check_or_create_path(get_event_path(run_path), is_dir=True)
        check_or_create_path(get_asset_path(run_path), is_dir=True)

        self._async_writer = EventAsyncManager(
            EventWriter(self._run_path, backend=EventWriter.EVENTS_BACKEND),
            max_queue_size,
            flush_secs,
        )

# --- from polyaxon__traceml::traceml/traceml/serialization/writer.py::BaseAsyncManager.__init__ ---
def __init__(
        self, event_writer: Union[EventWriter, LogWriter], max_queue_size: int = 20
    ):
        """Writes events json spec to files asynchronously. An instance of this class
        holds a queue to keep the incoming data temporarily. Data passed to the
        `write` function will be put to the queue and the function returns
        immediately. This class also maintains a thread to write data in the
        queue to disk.

        Args:
            event_writer: A EventWriter instance
            max_queue_size: Integer. Size of the queue for pending bytestrings.
            flush_secs: Number. How often, in seconds, to flush the
                pending bytestrings to disk.
        """
        self._event_writer = event_writer
        self._closed = False
        self._event_queue = queue.Queue(max_queue_size)
        self._lock = threading.Lock()
        self._worker = None

# --- from pykale__pykale::kale/loaddata/signal_access.py::load_ecg_from_folder ---
def load_ecg_from_folder(base_path, csv_file):
    """
    Loads and preprocesses a batch of ECG signals from a CSV file listing file paths.

    Args:
        base_path (str): Root directory containing ECG files.
        csv_file (str): CSV file listing files in column 'path'.

    Returns:
        Tensor: Batch of preprocessed ECG signals, shape (N, 1, total_samples).
    Example:
        ecg_tensor = load_ecg_from_csv("/data/ecg/", "ecg_files.csv")
    """
    cases = pd.read_csv(os.path.join(base_path, csv_file))
    full_paths = cases["path"].apply(lambda x: os.path.join(base_path, x))
    all_ecg = []

    for f in tqdm(full_paths, desc="Loading ECG data"):
        wave_array, meta = wfdb.rdsamp(f)
        wave_array = interpolate_signal(wave_array)
        num_channels = meta["n_sig"]
        num_samples_per_channel = wave_array.size // num_channels

        if wave_array.size % num_channels == 0:
            wave_array = wave_array.reshape(num_samples_per_channel, num_channels)
            wave_array = normalize_signal(wave_array)
            ecg_tensor = prepare_ecg_tensor(wave_array)
            all_ecg.append(ecg_tensor)
        else:
            logging.warning(f"Unexpected data size in {f}. Skipping file.")
    if all_ecg:
        return torch.cat(all_ecg, dim=0)
    else:
        return torch.empty(0)

# --- from piskvorky__gensim::gensim/models/lda_dispatcher.py::Dispatcher.initialize ---
def initialize(self, **model_params):
        """Fully initialize the dispatcher and all its workers.

        Parameters
        ----------
        **model_params
            Keyword parameters used to initialize individual workers, see :class:`~gensim.models.ldamodel.LdaModel`.

        Raises
        ------
        RuntimeError
            When no workers are found (the :mod:`gensim.models.lda_worker` script must be ran beforehand).

        """
        self.jobs = Queue(maxsize=self.maxsize)
        self.lock_update = threading.Lock()
        self._jobsdone = 0
        self._jobsreceived = 0

        self.workers = {}
        with utils.getNS(**self.ns_conf) as ns:
            self.callback = Pyro4.Proxy(ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
            for name, uri in ns.list(prefix=LDA_WORKER_PREFIX).items():
                try:
                    worker = Pyro4.Proxy(uri)
                    workerid = len(self.workers)
                    # make time consuming methods work asynchronously
                    logger.info("registering worker #%i at %s", workerid, uri)
                    worker.initialize(workerid, dispatcher=self.callback, **model_params)
                    self.workers[workerid] = worker
                except Pyro4.errors.PyroError:
                    logger.warning("unresponsive worker at %s,deleting it from the name server", uri)
                    ns.remove(name)

        if not self.workers:
            raise RuntimeError('no workers found; run some lda_worker scripts on your machines first!')

# --- from piskvorky__gensim::gensim/models/lsi_dispatcher.py::Dispatcher.initialize ---
def initialize(self, **model_params):
        """Fully initialize the dispatcher and all its workers.

        Parameters
        ----------
        **model_params
            Keyword parameters used to initialize individual workers
            (gets handed all the way down to :meth:`gensim.models.lsi_worker.Worker.initialize`).
            See :class:`~gensim.models.lsimodel.LsiModel`.

        Raises
        ------
        RuntimeError
            When no workers are found (the :mod:`gensim.model.lsi_worker` script must be ran beforehand).

        """
        self.jobs = Queue(maxsize=self.maxsize)
        self.lock_update = threading.Lock()
        self._jobsdone = 0
        self._jobsreceived = 0

        # locate all available workers and store their proxies, for subsequent RMI calls
        self.workers = {}
        with utils.getNS() as ns:
            self.callback = Pyro4.Proxy('PYRONAME:gensim.lsi_dispatcher')  # = self
            for name, uri in ns.list(prefix='gensim.lsi_worker').items():
                try:
                    worker = Pyro4.Proxy(uri)
                    workerid = len(self.workers)
                    # make time consuming methods work asynchronously
                    logger.info("registering worker #%i from %s", workerid, uri)
                    worker.initialize(workerid, dispatcher=self.callback, **model_params)
                    self.workers[workerid] = worker
                except Pyro4.errors.PyroError:
                    logger.exception("unresponsive worker at %s, deleting it from the name server", uri)
                    ns.remove(name)

        if not self.workers:
            raise RuntimeError('no workers found; run some lsi_worker scripts on your machines first!')
