# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg158::atexit.register+threading.Thread
# name: atexit_threading_primitive
# summary: Uses atexit.register, threading.Thread across 2 repos
# anchor_symbols: ['atexit.register', 'threading.Thread']
# observed in 2 repos: ['microsoft__nni', 'polyaxon__haupt']...

# --- from polyaxon__haupt::haupt/haupt/cli/runners/cron.py::start_cron ---
def start_cron(host: str):
    stop_event = threading.Event()
    thread = threading.Thread(
        target=agent_cron,
        args=(
            host,
            stop_event,
        ),
    )
    thread.daemon = True
    thread.start()

    # Ensure the thread stops gracefully on exit
    atexit.register(stop_event.set)

# --- from microsoft__nni::nni/contrib/training_service/trial_runner.py::TrialRunner._server_start ---
def _server_start(self) -> HTTPServer:
        _logger.info('Starting trial server at %s.', TrialServerHandler.ADDRESS)
        atexit.register(self._server_stop)
        server_address = ('', TrialServerHandler.PORT)
        httpd = HTTPServer(server_address, partial(TrialServerHandler, self._processing_trials, self._on_metric))

        def _start() -> None:
            httpd.serve_forever()

        threading.Thread(target=_start, daemon=True).start()
        return httpd
