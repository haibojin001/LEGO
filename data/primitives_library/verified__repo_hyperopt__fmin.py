import functools
import inspect
import logging
import os
import sys
import time
from timeit import default_timer as timer

import numpy as np

from hyperopt import exceptions, tpe
from hyperopt.base import validate_loss_threshold, validate_timeout

from . import base, progress, pyll
from .utils import coarse_utcnow

logger = logging.getLogger(__name__)

try:
    import cloudpickle as pickler
except Exception:
    logger.info(
        "Unable to import cloudpickle; install it with "
        '"pip install cloudpickle" for improved serialization support.'
    )
    import pickle as pickler


def generate_trial(tid, space):
    labels = space.keys()
    return {
        "state": base.JOB_STATE_NEW,
        "tid": tid,
        "spec": None,
        "result": {"status": "new"},
        "misc": {
            "tid": tid,
            "cmd": ("domain_attachment", "FMinIter_Domain"),
            "workdir": None,
            "idxs": {label: [tid] for label in labels},
            "vals": {label: [value] for label, value in space.items()},
        },
        "exp_key": None,
        "owner": None,
        "version": 0,
        "book_time": None,
        "refresh_time": None,
    }


def generate_trials_to_calculate(points):
    trials = base.Trials()
    trials.insert_trial_docs(
        [generate_trial(trial_id, point) for trial_id, point in enumerate(points)]
    )
    return trials


def fmin_pass_expr_memo_ctrl(fn):
    fn.fmin_pass_expr_memo_ctrl = True
    return fn


def partial(fn, **kwargs):
    result = functools.partial(fn, **kwargs)
    if hasattr(fn, "fmin_pass_expr_memo_ctrl"):
        result.fmin_pass_expr_memo_ctrl = fn.fmin_pass_expr_memo_ctrl
    return result


def __objective_fmin_wrapper(func):
    def objective(arguments):
        return func(**arguments)

    return objective


class FMinIter:
    catch_eval_exceptions = False
    pickle_protocol = -1

    def __init__(
        self,
        algo,
        domain,
        trials,
        rstate,
        asynchronous=None,
        max_queue_len=1,
        poll_interval_secs=1.0,
        max_evals=sys.maxsize,
        timeout=None,
        loss_threshold=None,
        verbose=False,
        show_progressbar=True,
        early_stop_fn=None,
        trials_save_file="",
    ):
        self.algo = algo
        self.domain = domain
        self.trials = trials
        self.rstate = rstate
        self.max_queue_len = max_queue_len
        self.poll_interval_secs = poll_interval_secs
        self.max_evals = max_evals
        self.timeout = timeout
        self.loss_threshold = loss_threshold
        self.verbose = verbose
        self.early_stop_fn = early_stop_fn
        self.early_stop_args = []
        self.trials_save_file = trials_save_file
        self.start_time = timer()

        if asynchronous is None:
            self.asynchronous = trials.asynchronous
        else:
            self.asynchronous = asynchronous

        if not show_progressbar or not verbose:
            self.progress_callback = progress.no_progress_callback
        elif show_progressbar is True:
            self.progress_callback = progress.default_callback
        else:
            self.progress_callback = show_progressbar

        if self.asynchronous and not hasattr(trials, "_spark"):
            if "FMinIter_Domain" in trials.attachments:
                logger.warning("Replacing an existing FMinIter domain attachment")
            serialized_domain = pickler.dumps(domain, protocol=self.pickle_protocol)
            pickler.loads(serialized_domain)
            trials.attachments["FMinIter_Domain"] = serialized_domain

    def serial_evaluate(self, N=-1):
        remaining = N
        for trial in self.trials._dynamic_trials:
            if trial["state"] != base.JOB_STATE_NEW:
                continue

            now = coarse_utcnow()
            trial["state"] = base.JOB_STATE_RUNNING
            trial["book_time"] = now
            trial["refresh_time"] = now

            try:
                spec = base.spec_from_misc(trial["misc"])
                control = base.Ctrl(self.trials, current_trial=trial)
                result = self.domain.evaluate(spec, control)
            except Exception as exc:
                logger.error("job exception: %s", str(exc))
                trial["state"] = base.JOB_STATE_ERROR
                trial["misc"]["error"] = (str(type(exc)), str(exc))
                trial["refresh_time"] = coarse_utcnow()

                if not self.catch_eval_exceptions:
                    self.trials.refresh()
                    raise
            else:
                trial["state"] = base.JOB_STATE_DONE
                trial["result"] = result
                trial["refresh_time"] = coarse_utcnow()

            remaining -= 1
            if remaining == 0:
                break

        self.trials.refresh()

    @property
    def is_cancelled(self):
        return bool(getattr(self.trials, "_fmin_cancelled", False))

    def block_until_done(self):
        if not self.asynchronous:
            self.serial_evaluate()
            return

        waiting_states = [base.JOB_STATE_NEW, base.JOB_STATE_RUNNING]
        announced = False

        while True:
            queued = self.trials.count_by_state_unsynced(waiting_states)
            if queued <= 0:
                break
            if self.verbose and not announced:
                logger.info("Waiting for %d jobs to finish ...", queued)
                announced = True
            time.sleep(self.poll_interval_secs)

        self.trials.refresh()

    def save_trials(self):
        if not self.trials_save_file:
            return
        with open(self.trials_save_file, "wb") as output:
            pickler.dump(self.trials, output, protocol=self.pickle_protocol)

    def run(self, N, block_until_done=True):
        trials = self.trials
        submitted = 0

        def new_count():
            return trials.count_by_state_unsynced(base.JOB_STATE_NEW)

        def done_count():
            return trials.count_by_state_unsynced(base.JOB_STATE_DONE)

        def unfinished_count():
            return trials.count_by_state_unsynced(
                [base.JOB_STATE_NEW, base.JOB_STATE_RUNNING]
            )

        completed_at_start = done_count()
        latest_completed = completed_at_start
        best_loss = float("inf")

        with self.progress_callback(
            initial=completed_at_start, total=self.max_evals
        ) as progress_context:
            all_complete = False

            while (
                (submitted < N or (block_until_done and not all_complete))
                and (
                    self.timeout is None
                    or timer() - self.start_time < self.timeout
                )
                and (
                    self.loss_threshold is None
                    or best_loss >= self.loss_threshold
                )
            ):
                queued = new_count()

                while (
                    queued < self.max_queue_len
                    and submitted < N
                    and not self.is_cancelled
                ):
                    amount = min(self.max_queue_len - queued, N - submitted)
                    identifiers = trials.new_trial_ids(amount)
                    trials.refresh()

                    seed = self.rstate.integers(2**31 - 1)
                    suggestions = self.algo(
                        identifiers,
                        self.domain,
                        trials,
                        seed,
                    )
                    trials.insert_trial_docs(suggestions)
                    trials.refresh()

                    submitted += len(suggestions)
                    queued = new_count()

                    if not suggestions:
                        break

                if self.asynchronous:
                    time.sleep(self.poll_interval_secs)
                else:
                    self.serial_evaluate()

                if self.trials_save_file:
                    self.save_trials()

                if self.early_stop_fn is not None:
                    stop, arguments = self.early_stop_fn(
                        self.trials, *self.early_stop_args
                    )
                    self.early_stop_args = arguments
                    if stop:
                        logger.info("Early stop triggered. Stopping iterations.")
                        break

                current_completed = done_count()
                if current_completed != latest_completed:
                    progress_context.update(current_completed - latest_completed)
                    latest_completed = current_completed

                observed_losses = [
                    loss for loss in trials.losses() if loss is not None
                ]
                if observed_losses:
                    best_loss = np.nanmin(observed_losses)

                all_complete = unfinished_count() == 0

        if self.trials_save_file:
            self.save_trials()

        return self

    def exhaust(self):
        self.run(self.max_evals - len(self.trials.trials))
        self.trials.refresh()
        return self


def space_eval(space, hp_assignment):
    expression = pyll.as_apply(space)
    memo = {}

    for node in pyll.toposort(expression):
        if node.name != "hyperopt_param":
            continue
        label = node.arg["label"].obj
        if label in hp_assignment:
            memo[node] = hp_assignment[label]

    return pyll.rec_eval(expression, memo=memo)


def fmin(
    fn,
    space,
    algo=None,
    max_evals=None,
    timeout=None,
    loss_threshold=None,
    trials=None,
    rstate=None,
    allow_trials_fmin=True,
    pass_expr_memo_ctrl=None,
    catch_eval_exceptions=False,
    verbose=True,
    return_argmin=True,
    points_to_evaluate=None,
    max_queue_len=1,
    show_progressbar=True,
    early_stop_fn=None,
    trials_save_file="",
):
    if algo is None:
        algo = tpe.suggest

    if max_evals is None:
        max_evals = sys.maxsize

    validate_timeout(timeout)
    validate_loss_threshold(loss_threshold)

    if rstate is None:
        seed = os.environ.get("HYPEROPT_FMIN_SEED")
        if seed is None:
            rstate = np.random.default_rng()
        else:
            rstate = np.random.default_rng(int(seed))

    if isinstance(space, str):
        if space != "annotated":
            raise exceptions.InvalidAnnotatedParameter(
                "The only supported string search space is 'annotated'."
            )

        signature = inspect.signature(fn)
        inferred_space = {}
        for parameter in signature.parameters.values():
            if parameter.annotation is inspect.Parameter.empty:
                raise exceptions.InvalidAnnotatedParameter(
                    "All objective function parameters must be annotated when "
                    "using the 'annotated' search space."
                )
            inferred_space[parameter.name] = parameter.annotation

        space = inferred_space
        fn = __objective_fmin_wrapper(fn)

    if trials is None:
        if points_to_evaluate is None:
            trials = base.Trials()
        else:
            trials = generate_trials_to_calculate(points_to_evaluate)

    if allow_trials_fmin and hasattr(trials, "fmin"):
        return trials.fmin(
            fn,
            space,
            algo,
            max_evals,
            timeout,
            loss_threshold,
            rstate,
            pass_expr_memo_ctrl,
            catch_eval_exceptions,
            verbose,
            return_argmin,
            points_to_evaluate,
            max_queue_len,
            show_progressbar,
            early_stop_fn,
            trials_save_file,
        )

    if pass_expr_memo_ctrl is None:
        pass_expr_memo_ctrl = getattr(fn, "fmin_pass_expr_memo_ctrl", False)

    domain = base.Domain(
        fn,
        space,
        pass_expr_memo_ctrl=pass_expr_memo_ctrl,
    )

    iterator = FMinIter(
        algo=algo,
        domain=domain,
        trials=trials,
        rstate=rstate,
        max_queue_len=max_queue_len,
        max_evals=max_evals,
        timeout=timeout,
        loss_threshold=loss_threshold,
        verbose=verbose,
        show_progressbar=show_progressbar,
        early_stop_fn=early_stop_fn,
        trials_save_file=trials_save_file,
    )
    iterator.catch_eval_exceptions = catch_eval_exceptions
    iterator.exhaust()

    if return_argmin:
        if not trials.trials:
            raise exceptions.AllTrialsFailed()
        return trials.argmin

    return trials