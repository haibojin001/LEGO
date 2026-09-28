# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg348::psutil.Process+psutil.net_connections
# name: psutil_primitive
# summary: Uses psutil.Process, psutil.net_connections across 2 repos
# anchor_symbols: ['psutil.Process', 'psutil.net_connections']
# observed in 2 repos: ['business-science__ai-data-science-team', 'nfstream__nfstream']...

# --- from business-science__ai-data-science-team::ai_data_science_team/tools/mlflow.py::mlflow_stop_ui ---
def mlflow_stop_ui(port: int = 5000) -> str:
    """
    Kill any process currently listening on the given MLflow UI port.
    Requires `pip install psutil`.

    Parameters
    ----------
    port : int, optional
        The port on which the UI is running.
    """
    print("    * Tool: mlflow_stop_ui")
    import psutil

    # Attempt to find processes listening on port; on macOS this may require elevated perms.
    try:
        conns = psutil.net_connections(kind="inet")
    except psutil.AccessDenied:
        return (
            "Unable to enumerate network connections (permission denied). "
            "Try running with elevated permissions or stop the MLflow UI manually."
        )
    except Exception as e:
        return f"Failed to inspect network connections: {e}"

    for conn in conns:
        if conn.laddr and conn.laddr.port == port and conn.pid is not None:
            try:
                p = psutil.Process(conn.pid)
                p_name = p.name()
                p.kill()
                return f"Killed process {conn.pid} ({p_name}) listening on port {port}."
            except psutil.NoSuchProcess:
                return "Process was already terminated before we could kill it."
            except psutil.AccessDenied:
                return (
                    f"Process {conn.pid} is listening on port {port} but cannot be killed "
                    "due to insufficient permissions."
                )
            except Exception as e:
                return f"Failed to kill process {conn.pid} on port {port}: {e}"

    return f"No process found listening on port {port}."

# --- from nfstream__nfstream::nfstream/system.py::system_socket_worflow ---
def system_socket_worflow(channel, idle_timeout, poll_period):
    """Host ground-truth generation workflow"""
    conn_cache = ConnCache(channel=channel, timeout=idle_timeout)
    try:
        while True:
            current_time = time.time() * 1000
            for conn in net_connections(kind="inet"):
                # IMPORTANT: The rationale behind the usage of an active polling approach (net_connections call):
                # System process visibility is intended to generate the most accurate ground truth for traffic
                # classification end-host-based research experiments, as reported in the literature [1].
                # Thus, it must be a cross-platform approach that works the same on Linux, macOS, and
                # Windows (Gaming traffic classification challenges). On Linux, things can be done more elegantly
                # using eBPF tracing exec calls or NetLink monitoring. However, this will requires specific
                # implementation for Linux versus Windows and proper handling of old kernel versions.
                # We prefer to keep it out of the nfstream codebase. As we use net_connections from psutil
                # (https://github.com/giampaolo/psutil) (https://github.com/giampaolo/psutil) Python package,
                # future work may include providing such enhancements to psutil project, and thus, a broader
                # community can benefit from it.
                # [1]: http://tomasz.bujlow.com/publications/2012_journal_TELFOR.pdf
                key = get_conn_key(conn)
                if key is not None:  # We succeeded to obtain a key.
                    if key not in conn_cache:  # Create and send
                        process_name = Process(conn.pid).name()
                        conn_cache[key] = current_time
                        channel.put(
                            NFSocket(NFEvent.SOCKET_CREATE, key, conn.pid, process_name)
                        )  # Send to streamer
                    else:  # update time
                        conn_cache[key] = current_time
            conn_cache.scan(current_time)

            time.sleep(poll_period)  # Sleep with configured poll period
            #                          0 will ensure the maximum active polling capacity and accuracy.
            #                          Greater values will results is less intensive CPU load and less accuracy.
            #                          A tradeoff must be decided (as always).
            #                          Completeness versus Polling frequency study (per protocol, per platform)
            #                          are provided in [1]
            #                          [2]: https://dl.acm.org/doi/abs/10.1145/1629607.1629610
    except KeyboardInterrupt:
        return
