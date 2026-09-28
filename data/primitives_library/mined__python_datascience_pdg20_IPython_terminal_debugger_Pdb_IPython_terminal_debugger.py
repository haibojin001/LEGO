# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg20::IPython.terminal.debugger.Pdb+IPython.terminal.debugger.TerminalPdb+Pyro4.naming.locateNS
# name: IPython_Pyro4_primitive
# summary: Uses IPython.terminal.debugger.Pdb, IPython.terminal.debugger.TerminalPdb, Pyro4.naming.locateNS, accelerate.utils.broadcast_object_list across 39 repos
# anchor_symbols: ['IPython.terminal.debugger.Pdb', 'IPython.terminal.debugger.TerminalPdb', 'Pyro4.naming.locateNS', 'accelerate.utils.broadcast_object_list', 'accelerate.utils.gather_object', 'accelerate.utils.is_peft_model']
# observed in 39 repos: ['AgnostiqHQ__covalent', 'DeepWisdom__AutoDL', 'HazyResearch__meerkat', 'InfuseAI__piperider', 'K-Dense-AI__agentic-data-scientist']...

# --- from ploomber__ploomber::tests/sources/test_python_interact.py::tmp_file ---
def tmp_file():
    fd, tmp = tempfile.mkstemp()
    os.close(fd)
    yield tmp
    Path(tmp).unlink()

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyagent/skyagent/tasks/verifiers/naive_dapo.py::timeout.__enter__ ---
def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

# --- from allenai__allennlp::allennlp/common/file_utils.py::_serialize ---
def _serialize(data):
    buffer = pickle.dumps(data, protocol=-1)
    return np.frombuffer(buffer, dtype=np.uint8)

# --- from xorbitsai__xorbits::python/xorbits/_mars/services/cluster/core.py::WatchNotifier.__init__ ---
def __init__(self):
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._version = 0

# --- from xorbitsai__xorbits::python/xorbits/pandas/_config/config.py::get_option ---
def get_option(pat: Any) -> Any:
    try:
        attr_list = pat.split(".")
        return reduce(getattr, attr_list, xorbits_options)
    except:
        return pd.get_option(pat)

# --- from cerndb__dist-keras::distkeras/networking.py::determine_host_address ---
def determine_host_address():
    """Determines the human-readable host address of the local machine."""
    host_address = socket.gethostbyname(socket.gethostname())

    return host_address

# --- from awslabs__gluonts::src/gluonts/shell/sagemaker/dyn.py::Installer.copy_install ---
def copy_install(self, path: Path):
        if path.is_file():
            shutil.copy(path, self.packages)
        elif path.is_dir():
            shutil.copytree(path, self.packages / path.name)

# --- from WecoAI__aideml::aide/backend/backend_openrouter.py::_setup_openrouter_client ---
def _setup_openrouter_client():
    global _client
    _client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=0,
    )

# --- from ruc-datalab__DeepAnalyze::playground/DS-1000/execution.py::swallow_io ---
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

# --- from vaexio__vaex::tests/execution_test.py::test_async_safe.do ---
async def do():
            promise = df.x.count(delay=True)
            import random
            r = random.random() * 0.01
            await asyncio.sleep(r)
            await df.execute_async()
            return await promise

# --- from WecoAI__aideml::aide/backend/backend_openai.py::_setup_openai_client ---
def _setup_openai_client():
    global _client
    # Use real OpenAI API with proper API key, explicitly override base_url
    api_key = os.getenv("OPENAI_API_KEY")
    _client = openai.OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL, max_retries=0)

# --- from AgnostiqHQ__covalent::tests/covalent_tests/executor/base_test.py::test_base_executor_cancel ---
async def test_base_executor_cancel(mocker):
    me = MockExecutor()
    me._init_runtime(loop=AsyncMock(), cancel_pool=MagicMock())
    me._loop.run_in_executor = AsyncMock()
    await me._cancel({}, 42)
    me._loop.run_in_executor.assert_awaited()
