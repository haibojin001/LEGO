from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
from multiprocessing import Event, Process
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventType


_HAS_NUMPY = importlib.util.find_spec("numpy") is not None

STRATEGY_NUMPY = "numpy"
STRATEGY_HASHLIB = "hashlib"
STRATEGIES = [STRATEGY_NUMPY, STRATEGY_HASHLIB]

STRATEGY_LABELS = {
    STRATEGY_NUMPY: "matmul burn",
    STRATEGY_HASHLIB: "hashlib SHA-256",
}

STRATEGY_REQUIREMENTS = {
    STRATEGY_NUMPY: "requires numpy",
}


def get_default_strategy() -> str:
    """Return the preferred workload available on this installation."""
    if _HAS_NUMPY:
        return STRATEGY_NUMPY
    return STRATEGY_HASHLIB


def strategy_available(strategy: str) -> bool:
    """Return whether *strategy* is runnable in the current environment."""
    return strategy != STRATEGY_NUMPY or _HAS_NUMPY


def _worker_numpy(stop_event: EventType) -> None:
    """Run cache-sized NumPy matrix products until asked to stop."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import numpy as np

    dimension = 128
    generator = np.random.default_rng()
    first = generator.random((dimension, dimension))
    second = generator.random((dimension, dimension))
    result = np.empty((dimension, dimension))

    while not stop_event.is_set():
        np.matmul(first, second, out=result)


def _worker_hashlib(stop_event: EventType) -> None:
    """Continuously calculate SHA-256 digests until stopped."""
    data = b"\x00" * 65536

    while not stop_event.is_set():
        hashlib.sha256(data).digest()


class BuiltinStresser:
    """Control a group of multiprocessing CPU stress workers."""

    def __init__(self) -> None:
        self._stop_event: EventType | None = None
        self._workers: list[Process] = []

    def start(self, num_workers: int, strategy: str | None = None) -> None:
        """Create *num_workers* processes using the selected workload."""
        chosen = get_default_strategy() if strategy is None else strategy

        if not strategy_available(chosen):
            logging.warning(
                "Strategy %s unavailable, falling back to hashlib", chosen
            )
            chosen = STRATEGY_HASHLIB

        target = (
            _worker_numpy
            if chosen == STRATEGY_NUMPY
            else _worker_hashlib
        )

        self.stop()
        self._stop_event = Event()

        try:
            for _ in range(num_workers):
                process = Process(
                    target=target,
                    args=(self._stop_event,),
                    daemon=True,
                )
                process.start()
                self._workers.append(process)
        except OSError:
            logging.exception(
                "Failed to start all built-in stress workers; cleaning up %d "
                "already-started workers",
                len(self._workers),
            )
            self.stop()
            raise

        logging.info(
            "Built-in stresser started %d workers (strategy: %s)",
            num_workers,
            STRATEGY_LABELS.get(chosen, chosen),
        )

    def stop(self, timeout: int = 3) -> None:
        """Signal workers and progressively force shutdown when necessary."""
        if not self._workers:
            return

        if self._stop_event is not None:
            self._stop_event.set()

        for process in self._workers:
            process.join(timeout=timeout)

        for process in self._workers:
            if process.is_alive():
                logging.debug("Terminating straggler worker %s", process.pid)
                process.terminate()
                process.join(timeout=1)

        for process in self._workers:
            if process.is_alive():
                logging.debug("Killing straggler worker %s", process.pid)
                process.kill()

        for process in self._workers:
            process.join(timeout=1)

        self._workers.clear()
        logging.info("Built-in stresser stopped")

    def is_running(self) -> bool:
        """Report whether at least one managed worker remains alive."""
        return any(process.is_alive() for process in self._workers)