# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg52::alfworld.agents.environment.get_environment+asyncio.ProactorEventLoop+asyncio.Queue
# name: alfworld_asyncio_primitive
# summary: Uses alfworld.agents.environment.get_environment, asyncio.ProactorEventLoop, asyncio.Queue, asyncio.create_task across 19 repos
# anchor_symbols: ['alfworld.agents.environment.get_environment', 'asyncio.ProactorEventLoop', 'asyncio.Queue', 'asyncio.create_task', 'asyncio.get_event_loop', 'asyncio.get_running_loop']
# observed in 19 repos: ['AgnostiqHQ__covalent', 'HDI-Project__ATM', 'HazyResearch__meerkat', 'insitro__redun', 'microsoft__RD-Agent']...

# --- from violit-dev__violit::src/violit/app.py::App._send_interval_ctrl._run ---
def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_push())
            loop.close()

# --- from okfn-brasil__querido-diario::data_collection/gazette/extensions.py::StatsPersist.spider_opened ---
def spider_opened(self, spider):
        engine = create_engine(self._database_url)
        DeclarativeBase.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)

# --- from AgnostiqHQ__covalent::tests/covalent_dispatcher_tests/_service/app_test.py::MockDataStore.__init__ ---
def __init__(self, db_URL):
        self.db_URL = db_URL
        self.engine = create_engine(self.db_URL)
        self.Session = sessionmaker(self.engine)

        MockBase.metadata.create_all(self.engine)

# --- from AgnostiqHQ__covalent::tests/covalent_dispatcher_tests/_service/assets_test.py::MockDataStore.__init__ ---
def __init__(self, db_URL):
        self.db_URL = db_URL
        self.engine = create_engine(self.db_URL)
        self.Session = sessionmaker(self.engine)

        MockBase.metadata.create_all(self.engine)

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyagent/skyagent/dispatcher/async_utils.py::call_async_from_sync.run ---
def run():
        loop_for_thread = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop_for_thread)
            return asyncio.run(arun())
        finally:
            loop_for_thread.close()

# --- from okfn-brasil__querido-diario::data_collection/gazette/monitors.py::ComparisonBetweenSpiderExecutionsMonitor._get_session ---
def _get_session(self):
        database_url = self.data.crawler.settings.get("QUERIDODIARIO_DATABASE_URL")
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)
        return Session()

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/examples/train/grpo/plugin/plugin.py::CodeRewardByJudge0.run_async_from_sync ---
def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

# --- from reiinakano__xcessiv::xcessiv/functions.py::DBContextManager.__enter__ ---
def __enter__(self):
        if not os.path.exists(self.path):
            raise exceptions.UserError('{} does not exist'.format(self.path))
        sqlite_url = 'sqlite:///{}'.format(self.path)
        engine = create_engine(sqlite_url)

        self.session = Session(bind=engine)

        return self.session

# --- from insitro__redun::redun/executors/local.py::LocalExecutor._async_worker ---
def _async_worker(self) -> None:
        """
        Worker thread for running an async loop.
        """
        with self._async_ready:
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            self._async_ready.notify()

        self._async_loop.run_forever()

# --- from vaexio__vaex::tests/jupyter/conftest.py::event_loop ---
def event_loop():
    """Don't close event loop at the end of every function decorated by
    @pytest.mark.asyncio
    """
    import asyncio
    print("CREATE" * 10)
    # asyncio.set_event_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # vaex.jupyter.utils.main_event_loop = loop
    yield loop
    loop.close()

# --- from sematic-ai__sematic::sematic/types/types/tests/test_list.py::test_to_binary_arbitrary ---
def test_to_binary_arbitrary():
    type_ = List[A]

    json_encodable = value_to_json_encodable([A(), A()], type_)

    assert len(json_encodable) == 2
    assert all(list(element.keys()) == ["pickle"] for element in json_encodable)
    for element in json_encodable:
        assert list(element.keys()) == ["pickle"]
        cloudpickle.loads(base64.b64decode(element["pickle"]))

# --- from ploomber__ploomber::src/ploomber/repo.py::_run_command ---
def _run_command(path, command):
    """Safely run command in certain path"""
    if not Path(path).is_dir():
        raise ValueError("{} is not a directory".format(path))

    out = subprocess.check_output(
        shlex.split(command), cwd=str(path), stderr=subprocess.PIPE
    )
    s = out.decode("utf-8")

    # remove trailing \n
    if s[-1:] == "\n":
        s = s[:-1]

    return s
