# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg330::os.getpid+socket.gethostname
# name: os_socket_primitive
# summary: Uses os.getpid, socket.gethostname across 2 repos
# anchor_symbols: ['os.getpid', 'socket.gethostname']
# observed in 2 repos: ['mahmoud__boltons', 'run-house__kubetorch']...

# --- from mahmoud__boltons::boltons/iterutils.py::GUIDerator.reseed ---
def reseed(self):
        import socket
        self.pid = os.getpid()
        self.salt = '-'.join([str(self.pid),
                              socket.gethostname() or '<nohostname>',
                              str(time.time()),
                              os.urandom(6).hex()])
        return

# --- from run-house__kubetorch::python_client/tests/assets/test_distributed/distributed_test_functions.py::adaptive_ray_fn_with_bs4.worker_task ---
def worker_task(worker_id):
            """Task that tests package availability and returns info."""
            import time

            time.sleep(0.1)  # Simulate some work

            # Test if beautifulsoup4 package is available (definitely not in base rayproject/ray)
            bs4_available = False
            bs4_version = None
            try:
                import bs4

                bs4_available = True
                bs4_version = bs4.__version__
            except ImportError:
                pass

            return {
                "worker_id": worker_id,
                "hostname": socket.gethostname(),
                "worker_pid": os.getpid(),
                "bs4_available": bs4_available,
                "bs4_version": bs4_version,
                "calculation": worker_id * 10,  # Simple calculation
            }
