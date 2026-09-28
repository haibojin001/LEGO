# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg148::os.waitpid+time.sleep
# name: os_time_primitive
# summary: Uses os.waitpid, time.sleep across 2 repos
# anchor_symbols: ['os.waitpid', 'time.sleep']
# observed in 2 repos: ['microsoft__nni', 'sematic-ai__sematic']...

# --- from sematic-ai__sematic::sematic/runners/tests/test_local_runner.py::test_subprocess_error.fork_func ---
def fork_func(param: int) -> int:  # type: ignore
        subprocess_pid = os.fork()

        if subprocess_pid == 0:
            # sys.exit(1) would mess up pytest
            os._exit(1)

        os.waitpid(subprocess_pid, 0)
        time.sleep(1)
        return param

# --- from sematic-ai__sematic::sematic/runners/tests/test_local_runner.py::test_subprocess_signal_handling.fork_func ---
def fork_func(param: int) -> int:  # type: ignore
        subprocess_pid = os.fork()

        if subprocess_pid == 0:
            time.sleep(1)
            # sys.exit(1) would mess up pytest
            os._exit(1)

        os.kill(subprocess_pid, 15)
        os.waitpid(subprocess_pid, 0)
        return param

# --- from microsoft__nni::nni/tools/nnictl/command_utils.py::_check_pid_running ---
def _check_pid_running(pid):
    # Check whether process still running.
    # FIXME: the correct impl should be using ``proc.poll()``
    # Using pid here is unsafe.
    # We should make Popen object directly accessible.
    if sys.platform == 'win32':
        # NOTE: Tests show that the behavior of psutil is unreliable, and varies from runs to runs.
        # Also, Windows didn't explicitly handle child / non-child process.
        # This might be a potential problem.
        try:
            psutil.Process(pid).wait(timeout=0)
            return False
        except psutil.TimeoutExpired:
            return True
        except psutil.NoSuchProcess:
            return False
    else:
        try:
            indicator, _ = os.waitpid(pid, os.WNOHANG)
            return indicator == 0
        except ChildProcessError:
            # One of the reasons we reach here is: pid may be not a child process.
            # In that case, we can use the famous kill 0 to poll the process.
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
