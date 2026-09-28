# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg209::os.getpgid+os.killpg
# name: os_primitive
# summary: Uses os.getpgid, os.killpg across 3 repos
# anchor_symbols: ['os.getpgid', 'os.killpg']
# observed in 3 repos: ['microsoft__RD-Agent', 'run-house__kubetorch', 'violit-dev__violit']...

# --- from run-house__kubetorch::python_client/kubetorch/globals.py::_kill ---
def _kill(proc: Optional[subprocess.Popen]) -> None:
    if not proc:
        return
    try:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
    except Exception:
        pass

# --- from violit-dev__violit::src/violit/app_launcher.py::AppLauncherMixin._run_native_reload._kill_process ---
def _kill_process(proc):
            try:
                if is_unix:
                    import signal
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, OSError):
                pass

# --- from violit-dev__violit::src/violit/app_launcher.py::AppLauncherMixin._run_native_reload._terminate_process ---
def _terminate_process(proc):
            try:
                if is_unix:
                    import signal
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except (ProcessLookupError, OSError):
                pass

# --- from microsoft__RD-Agent::rdagent/scenarios/rl/autorl_bench/core/utils.py::kill_process_tree ---
def kill_process_tree(proc: "subprocess.Popen") -> None:
    """Kill a process session first, then its process group as a fallback."""
    import signal as _signal

    if proc.poll() is not None:
        return

    try:
        sid = os.getsid(proc.pid)
    except OSError:
        sid = None

    for sig in (_signal.SIGTERM, _signal.SIGKILL):
        if sid:
            try:
                ps_output = subprocess.check_output(["ps", "-eo", "pid=,sid="], text=True)
                for line in ps_output.splitlines():
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    pid_s, sid_s = parts
                    if int(sid_s) == sid:
                        try:
                            os.kill(int(pid_s), sig)
                        except (ProcessLookupError, OSError):
                            pass
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

            try:
                os.killpg(sid, sig)
            except (ProcessLookupError, OSError):
                pass

        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, OSError):
            pass

        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue

    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass
    proc.wait()

# --- from run-house__kubetorch::python_client/kubetorch/cli_utils.py::port_forward_to_pod ---
def port_forward_to_pod(
    pod_name,
    namespace: str = None,
    local_port: int = 8080,
    remote_port: int = provisioning_constants.DEFAULT_NGINX_PORT,
    health_endpoint: str = None,
):
    for attempt in range(MAX_PORT_TRIES):
        candidate_port = local_port + attempt
        if not is_port_available(candidate_port):
            logger.debug(f"Local port {candidate_port} is already in use, trying again...")
            continue

        cmd = [
            "kubectl",
            "port-forward",
            f"pod/{pod_name}",
            f"{candidate_port}:{remote_port}",
            "--namespace",
            namespace,
        ]
        logger.debug(f"Running port-forward command: {' '.join(cmd)}")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

        try:
            wait_for_port_forward(
                process,
                candidate_port,
                health_endpoint=health_endpoint,
                validate_kubetorch_versions=False,
            )
            time.sleep(2)
            yield candidate_port
            return

        finally:
            if process:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait()
                except (ProcessLookupError, OSError):
                    # Process may have already terminated
                    pass

    raise RuntimeError(f"Could not bind available port after {MAX_PORT_TRIES} attempts")
