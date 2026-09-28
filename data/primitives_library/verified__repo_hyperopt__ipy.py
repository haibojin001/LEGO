"""Utilities for parallel optimization using IPython engines."""

import sys
from time import sleep, time

import numpy as np

from .base import (
    JOB_STATE_DONE,
    JOB_STATE_ERROR,
    JOB_STATE_NEW,
    JOB_STATE_RUNNING,
    Ctrl,
    Domain,
    Trials,
    spec_from_misc,
)
from .utils import coarse_utcnow


print(sys.stderr, "WARNING: IPythonTrials is not as complete, stable", file=sys.stderr)
print("         or well tested as Trials or MongoTrials.", file=sys.stderr)


class LostEngineError(RuntimeError):
    """Raised when an IPython engine vanishes while it owns a job."""


class IPythonTrials(Trials):
    def __init__(self, client, job_error_reaction="raise", save_ipy_metadata=True):
        self._client = client
        self._clientlbv = client.load_balanced_view()
        self.job_map = {}
        self.job_error_reaction = job_error_reaction
        self.save_ipy_metadata = save_ipy_metadata
        super().__init__()
        self._testing_fmin_was_called = False

    def _insert_trial_docs(self, docs):
        tids = [doc["tid"] for doc in docs]
        self._dynamic_trials.extend(docs)
        return tids

    def refresh(self):
        current_jobs = {}

        for engine_id in self._client.ids:
            current_jobs[engine_id] = self.job_map.pop(engine_id, (None, None))

        for engine_id, (promise, trial) in list(self.job_map.items()):
            if self.job_error_reaction == "raise":
                raise LostEngineError(promise)
            if self.job_error_reaction == "log":
                trial["error"] = "LostEngineError (%s)" % str(promise)
                trial["state"] = JOB_STATE_ERROR
            else:
                raise ValueError(self.job_error_reaction)

        for engine_id, (promise, trial) in list(current_jobs.items()):
            if promise is None:
                continue

            if promise.ready():
                try:
                    trial["result"] = promise.get()
                    trial["state"] = JOB_STATE_DONE
                    current_jobs[engine_id] = (None, None)
                except Exception as exc:
                    if self.job_error_reaction == "raise":
                        raise
                    if self.job_error_reaction == "log":
                        trial["error"] = str(exc)
                        trial["state"] = JOB_STATE_ERROR
                    else:
                        raise ValueError(self.job_error_reaction)

                if self.save_ipy_metadata:
                    trial["ipy_metadata"] = promise.metadata

                trial["refresh_time"] = coarse_utcnow()
                del current_jobs[engine_id]

        self.job_map = current_jobs
        super().refresh()

    def fmin(self, fn, space, **kw):
        algo = kw.get("algo")
        max_evals = kw.get("max_evals")
        rstate = kw.get("rstate", None)
        verbose = kw.get("verbose", 0)

        if rstate is None:
            rstate = np.random

        self._testing_fmin_was_called = True

        pass_expr_memo_ctrl = (None,)
        if pass_expr_memo_ctrl is None:
            try:
                pass_expr_memo_ctrl = fn.pass_expr_memo_ctrl
            except AttributeError:
                pass_expr_memo_ctrl = False

        domain = Domain(fn, space, None, pass_expr_memo_ctrl=False)
        last_print_time = 0

        while len(self._dynamic_trials) < max_evals:
            self.refresh()

            if verbose and last_print_time + 1 < time():
                print(
                    "fmin: %4i/%4i/%4i/%4i  %f"
                    % (
                        self.count_by_state_unsynced(JOB_STATE_NEW),
                        self.count_by_state_unsynced(JOB_STATE_RUNNING),
                        self.count_by_state_unsynced(JOB_STATE_DONE),
                        self.count_by_state_unsynced(JOB_STATE_ERROR),
                        min(
                            [float("inf")]
                            + [loss for loss in self.losses() if loss is not None]
                        ),
                    )
                )
                last_print_time = time()

            idle_engines = [
                engine_id
                for engine_id, (promise, trial) in list(self.job_map.items())
                if promise is None
            ]

            if idle_engines:
                new_ids = self.new_trial_ids(len(idle_engines))
                new_trials = algo(new_ids, domain, self, rstate.integers(2**31 - 1))

                if len(new_trials) == 0:
                    break

                assert len(idle_engines) >= len(new_trials)

                for engine_id, new_trial in zip(idle_engines, new_trials):
                    now = coarse_utcnow()
                    new_trial["book_time"] = now
                    new_trial["refresh_time"] = now

                    (tid,) = self.insert_trial_docs([new_trial])
                    promise = call_domain(
                        domain,
                        spec_from_misc(new_trial["misc"]),
                        Ctrl(self, current_trial=new_trial),
                        new_trial,
                        self._clientlbv,
                        engine_id,
                        tid,
                    )

                    trial = self._dynamic_trials[-1]
                    assert trial["tid"] == tid
                    self.job_map[engine_id] = (promise, trial)
                    trial["state"] = JOB_STATE_RUNNING

        if True:
            if verbose:
                print("fmin: Waiting on remaining jobs...")
            self.wait(verbose=verbose)

        return self.argmin

    def wait(self, verbose=False, verbose_print_interval=1.0):
        last_print_time = 0

        while True:
            self.refresh()

            if verbose and last_print_time + verbose_print_interval < time():
                print(
                    "fmin: %4i/%4i/%4i/%4i  %f"
                    % (
                        self.count_by_state_unsynced(JOB_STATE_NEW),
                        self.count_by_state_unsynced(JOB_STATE_RUNNING),
                        self.count_by_state_unsynced(JOB_STATE_DONE),
                        self.count_by_state_unsynced(JOB_STATE_ERROR),
                        min(
                            [float("inf")]
                            + [loss for loss in self.losses() if loss is not None]
                        ),
                    )
                )
                last_print_time = time()

            if self.count_by_state_unsynced(JOB_STATE_NEW):
                sleep(1e-1)
                continue

            if self.count_by_state_unsynced(JOB_STATE_RUNNING):
                sleep(1e-1)
                continue

            break

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_client"]
        del state["_trials"]
        del state["job_map"]
        return state

    def __setstate__(self, dct):
        self.__dict__ = dct
        self.job_map = {}
        Trials.refresh(self)


class IPYAsync:
    def __init__(self, asynchronous, domain, rv, eid, tid, ctrl):
        self.asynchronous = asynchronous
        self.domain = domain
        self.rv = rv
        self.metadata = asynchronous.metadata
        self.eid = eid
        self.tid = tid
        self.ctrl = ctrl

    def ready(self):
        return self.asynchronous.ready()

    def get(self):
        if self.asynchronous.successful():
            value = self.asynchronous.get()
            return self.domain.evaluate_async2(value, self.ctrl)
        return self.rv


def call_domain(domain, spec, ctrl, trial, view, eid, tid):
    fallback_result = {"loss": None, "status": "fail"}
    coarse_utcnow()
    function, pyll_result = domain.evaluate_async(spec, ctrl)
    asynchronous = view.apply_async(function, pyll_result)
    return IPYAsync(asynchronous, domain, fallback_result, eid, tid, ctrl)