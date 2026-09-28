# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg117::asyncio.Event+asyncio.Future+asyncio.Queue
# name: asyncio_primitive
# summary: Uses asyncio.Event, asyncio.Future, asyncio.Queue, asyncio.all_tasks across 23 repos
# anchor_symbols: ['asyncio.Event', 'asyncio.Future', 'asyncio.Queue', 'asyncio.all_tasks', 'asyncio.as_completed', 'asyncio.create_task']
# observed in 23 repos: ['Data-Centric-AI-Community__fg-data-profiling', 'InfuseAI__piperider', 'ModelOriented__DALEX', 'airbnb__knowledge-repo', 'akfamily__aktools']...

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/swift/llm/infer/rollout.py::safe_set_start_method ---
def safe_set_start_method():
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn")

# --- from run-ai__genv::genv/utils/os_.py::Flock.__exit__ ---
def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)

# --- from ruc-datalab__DeepAnalyze::demo/chat_v2/backend_app/routers/workspace.py::upload_files ---
async def upload_files(
    files: list[UploadFile] = File(...),
    session_id: str = Query("default"),
):
    return await workspace_service.upload_files_to_workspace(session_id, files)

# --- from InfuseAI__piperider::piperider_cli/datasource/__init__.py::DataSource.verify_connection ---
def verify_connection(self):
        engine = self.get_engine_by_database()
        with engine.connect() as conn:
            stmt = select(text('1 as id'))
            conn.execute(stmt)
        return

# --- from akfamily__aktools::aktools/login/user_login.py::get_current_active_user ---
async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- from okfn-brasil__querido-diario::data_collection/tests/test_dates.py::TestGenerateDatesSequence.test_if_works_with_date_input ---
def test_if_works_with_date_input(self):
        assert generate_dates_sequence(
            start=date(2025, 3, 5), end=date(2025, 3, 7), recurrence=DAILY
        ) == [datetime(2025, 3, 5), datetime(2025, 3, 6), datetime(2025, 3, 7)]

# --- from microsoft__nni::test/ut/sdk/helper/websocket_server.py::ws_server ---
async def ws_server():
    async with websockets.serve(on_connect, 'localhost', 0) as server:
        port = server.sockets[0].getsockname()[1]
        print(port, flush=True)
        _debug(f'port: {port}')
        await asyncio.Future()

# --- from airbnb__knowledge-repo::knowledge_repo/app/models.py::Post.view_user_count ---
def view_user_count(self):
        return (select([func.count(distinct(PageView.user_id))])
                .where(PageView.object_id == self.id)
                .where(PageView.object_type == 'post')
                .label('view_user_count'))

# --- from run-ai__genv::genv/utils/os_.py::Flock.__enter__ ---
def __enter__(self):
        self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, self._mode)

        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except (IOError, OSError):
            os.close(self._fd)
            raise

# --- from datafold__data-diff::data_diff/utils.py::number_to_human ---
def number_to_human(n):
    millnames = ["", "k", "m", "b"]
    n = float(n)
    millidx = max(
        0,
        min(len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))),
    )

    return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])

# --- from encord-team__encord-active::src/encord_active/server/routers/queries/metric_query.py::literal_bucket_depends ---
def literal_bucket_depends(default: int) -> Literal[10, 100, 1000]:
    def _parse_buckets(
        buckets: Literal["10", "100", "1000"] = Query(default, alias="buckets")
    ) -> Literal[10, 100, 1000]:
        return int(buckets)  # type: ignore

    return Depends(_parse_buckets, use_cache=False)

# --- from run-house__kubetorch::python_client/kubetorch/cli.py::main ---
def main(
    ctx: typer.Context,
    version: bool = typer.Option(None, "--version", "-v", help="Show the version and exit."),
):
    if version:
        from kubetorch import __version__

        print(f"{__version__}")
    elif ctx.invoked_subcommand is None:
        subprocess.run("kubetorch --help", shell=True)
