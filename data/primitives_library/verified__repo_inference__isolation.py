import asyncio
import threading
from typing import Any, Coroutine


class Isolation:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        threaded: bool = True,
        daemon: bool = True,
    ):
        self._loop = loop
        self._threaded = threaded
        self._daemon = daemon
        self._stopped = None
        self._thread = None
        self._thread_ident = None

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._stopped = asyncio.Event()
        self._loop.run_until_complete(self._stopped.wait())
        self._cancel_all_tasks(self._loop)

    @staticmethod
    def _cancel_all_tasks(loop):
        tasks = asyncio.all_tasks(loop)
        if not tasks:
            return

        for task in tasks:
            task.cancel()

        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        for task in tasks:
            if task.cancelled():
                continue
            if task.exception() is not None:
                loop.call_exception_handler(
                    {
                        "message": "unhandled exception during asyncio.run() shutdown",
                        "exception": task.exception(),
                        "task": task,
                    }
                )

    def start(self):
        if not self._threaded:
            return

        worker = threading.Thread(target=self._run)
        if self._daemon:
            worker.daemon = True
        self._thread = worker
        worker.start()
        self._thread_ident = worker.ident

    def call(self, coro: Coroutine) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    @property
    def thread_ident(self):
        return self._thread_ident

    @property
    def loop(self):
        return self._loop

    async def _stop(self):
        self._stopped.set()

    def stop(self):
        if not self._threaded:
            return

        asyncio.run_coroutine_threadsafe(self._stop(), self._loop).result()
        self._thread.join()