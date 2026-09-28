import copy
import threading
import time
import timeit
import traceback

from hyperopt import Trials, base, fmin
from hyperopt.base import STATUS_OK, validate_loss_threshold, validate_timeout
from hyperopt.utils import _get_logger, _get_random_id, coarse_utcnow

try:
    import pyspark
    from py4j.clientserver import ClientServer
    from pyspark.sql import SparkSession
    from pyspark.util import VersionUtils

    _have_spark = True
    _spark_major_minor_version = VersionUtils.majorMinorVersion(pyspark.__version__)
except ImportError:
    _have_spark = False
    _spark_major_minor_version = None

logger = _get_logger("hyperopt-spark")

FMIN_CANCELLED_REASON_EARLY_STOPPING = "early stopping condition"
FMIN_CANCELLED_REASON_TIMEOUT = "fmin run timeout"
FMIN_CANCELLED_REASON_USER = "fmin run cancelled by user"


def _spark_evaluate_trial(domain, trial):
    """Evaluate one Hyperopt trial in a Spark Python worker."""
    try:
        trial = copy.deepcopy(trial)
        worker_trials = Trials()
        spec = base.spec_from_misc(trial["misc"])
        ctrl = base.Ctrl(worker_trials, trial, domain)
        result = domain.evaluate(spec, ctrl)
        return {
            "ok": True,
            "result": result,
        }
    except BaseException as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


class SparkTrials(Trials):
    """
    A Trials implementation which evaluates Hyperopt trials as Spark tasks.

    Every trial is evaluated by one Spark task.  The scheduling side remains on
    the driver, allowing Hyperopt's existing algorithms to propose new points
    while previously proposed points are still running.
    """

    asynchronous = True
    MAX_CONCURRENT_JOBS_ALLOWED = 128

    def __str__(self):
        return f"SparkTrials(trials={self.trials})"

    def __init__(
        self,
        parallelism=None,
        timeout=None,
        loss_threshold=None,
        spark_session=None,
        resource_profile=None,
    ):
        self._spark_lock = threading.RLock()
        self._spark_active_jobs = {}
        self._spark_domain = None
        self._fmin_start_time = None
        self._timeout_timer = None

        super().__init__(exp_key=None, refresh=False)

        if not _have_spark:
            raise Exception(
                "SparkTrials cannot import pyspark classes.  Make sure that PySpark "
                "is available in your environment.  E.g., try running 'import pyspark'"
            )

        validate_timeout(timeout)
        validate_loss_threshold(loss_threshold)

        self._spark = (
            SparkSession.builder.getOrCreate()
            if spark_session is None
            else spark_session
        )
        self._spark_context = self._spark.sparkContext
        self._spark_pinned_threads_enabled = isinstance(
            self._spark_context._gateway, ClientServer
        )

        self._spark_supports_job_cancelling = (
            self._spark_pinned_threads_enabled
            or hasattr(self._spark_context.parallelize([1]), "collectWithJobGroup")
        )

        self.parallelism = self._decide_parallelism(
            requested_parallelism=parallelism,
            spark_default_parallelism=self._spark_context.defaultParallelism,
        )
        self.user_specified_parallelism = parallelism

        self._spark_supports_resource_profile = (
            hasattr(self._spark_context.parallelize([1]), "withResources")
            and not self._spark.conf.get("spark.master", "").startswith("local")
        )
        if self._spark_supports_resource_profile:
            self._resource_profile = resource_profile
        else:
            self._resource_profile = None
            if resource_profile is not None:
                logger.warning(
                    "SparkTrials was constructed with a ResourceProfile, but this Apache "
                    "Spark version does not support stage-level scheduling."
                )

        if not self._spark_supports_job_cancelling and timeout is not None:
            logger.warning(
                "SparkTrials was constructed with a timeout specified, but this Apache "
                "Spark version does not support job group-based cancellation. The "
                "timeout will be respected when starting new Spark jobs, but "
                "SparkTrials will not be able to cancel running Spark jobs which exceed"
                " the timeout."
            )

        self.timeout = timeout
        self.loss_threshold = loss_threshold
        self._fmin_cancelled = False
        self._fmin_cancelled_reason = None
        self.refresh()

    @staticmethod
    def _decide_parallelism(requested_parallelism, spark_default_parallelism):
        if requested_parallelism is None or requested_parallelism <= 0:
            parallelism = max(spark_default_parallelism, 1)
            logger.warning(
                "Because the requested parallelism was None or a non-positive value, "
                f"parallelism will be set to ({parallelism}), which is Spark's default parallelism ({spark_default_parallelism}), "
                "or 1, whichever is greater. "
                "We recommend setting parallelism explicitly to a positive value because "
                "the total of Spark task slots is subject to cluster sizing."
            )
        else:
            parallelism = requested_parallelism

        if parallelism > SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED:
            logger.warning(
                f"Parallelism ({parallelism}) is capped at SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED ({SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED})."
            )
            parallelism = SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED
        return parallelism

    @property
    def fmin_cancelled_reason(self):
        return self._fmin_cancelled_reason

    def count_successful_trials(self):
        return self.count_by_state_unsynced(base.JOB_STATE_DONE)

    def count_failed_trials(self):
        return self.count_by_state_unsynced(base.JOB_STATE_ERROR)

    def count_cancelled_trials(self):
        return self.count_by_state_unsynced(base.JOB_STATE_CANCEL)

    def count_total_trials(self):
        return self.count_by_state_unsynced(
            [
                base.JOB_STATE_DONE,
                base.JOB_STATE_ERROR,
                base.JOB_STATE_CANCEL,
            ]
        )

    def delete_all(self):
        super().delete_all()
        self._fmin_cancelled = False
        self._fmin_cancelled_reason = None

    def trial_attachments(self, trial):
        raise NotImplementedError("SparkTrials does not support trial attachments.")

    def refresh(self):
        result = super().refresh()
        if hasattr(self, "_spark_lock"):
            self._launch_pending_trials()
        return result

    def insert_trial_docs(self, docs):
        result = super().insert_trial_docs(docs)
        self._launch_pending_trials()
        return result

    def _timed_out(self):
        return (
            self.timeout is not None
            and self._fmin_start_time is not None
            and timeit.default_timer() - self._fmin_start_time >= self.timeout
        )

    def _timeout_reached(self):
        self._cancel_trials(FMIN_CANCELLED_REASON_TIMEOUT)

    def _launch_pending_trials(self):
        if self._spark_domain is None:
            return

        with self._spark_lock:
            if self._fmin_cancelled:
                return

            if self._timed_out():
                self._cancel_trials(FMIN_CANCELLED_REASON_TIMEOUT)
                return

            running = sum(
                1
                for trial in self._dynamic_trials
                if trial.get("state") == base.JOB_STATE_RUNNING
            )
            available = max(0, self.parallelism - running)
            if available == 0:
                return

            pending = [
                trial
                for trial in self._dynamic_trials
                if trial.get("state") == base.JOB_STATE_NEW
            ]

            for trial in pending[:available]:
                now = coarse_utcnow()
                trial["state"] = base.JOB_STATE_RUNNING
                trial["book_time"] = now
                trial["refresh_time"] = now

                tid = trial["tid"]
                job_group_id = "hyperopt-" + _get_random_id()
                thread = threading.Thread(
                    target=self._run_trial,
                    args=(trial, self._spark_domain, job_group_id),
                    name="hyperopt-spark-trial-%s" % tid,
                    daemon=True,
                )
                self._spark_active_jobs[tid] = {
                    "thread": thread,
                    "job_group_id": job_group_id,
                }
                thread.start()

    def _run_trial(self, trial, domain, job_group_id):
        tid = trial["tid"]
        payload = None
        failure = None

        try:
            rdd = self._spark_context.parallelize([0], 1)
            if self._resource_profile is not None:
                rdd = rdd.withResources(self._resource_profile)

            def run_partition(_):
                yield _spark_evaluate_trial(domain, trial)

            mapped = rdd.mapPartitions(run_partition)
            description = "Hyperopt trial %s" % tid

            if self._spark_pinned_threads_enabled:
                self._spark_context.setJobGroup(
                    job_group_id,
                    description,
                    interruptOnCancel=True,
                )
                try:
                    collected = mapped.collect()
                finally:
                    try:
                        self._spark_context.clearJobGroup()
                    except Exception:
                        pass
            elif hasattr(mapped, "collectWithJobGroup"):
                collected = mapped.collectWithJobGroup(
                    job_group_id,
                    description,
                    True,
                )
            else:
                collected = mapped.collect()

            if not collected:
                failure = (
                    "SparkTrialError",
                    "Spark trial returned no evaluation result",
                )
            else:
                payload = collected[0]
        except BaseException as exc:
            failure = (type(exc).__name__, str(exc))

        with self._spark_lock:
            self._spark_active_jobs.pop(tid, None)

            if trial.get("state") == base.JOB_STATE_CANCEL:
                self._launch_pending_trials()
                return

            now = coarse_utcnow()
            trial["refresh_time"] = now

            if failure is not None:
                trial["state"] = base.JOB_STATE_ERROR
                trial.setdefault("misc", {})["error"] = failure
            elif not isinstance(payload, dict) or not payload.get("ok"):
                trial["state"] = base.JOB_STATE_ERROR
                if isinstance(payload, dict):
                    error_type = payload.get("error_type", "SparkTrialError")
                    error = payload.get("error", "Unknown Spark worker error")
                else:
                    error_type = "SparkTrialError"
                    error = "Invalid result returned by Spark worker"
                trial.setdefault("misc", {})["error"] = (error_type, error)
            else:
                trial["state"] = base.JOB_STATE_DONE
                trial["result"] = payload["result"]

                result = payload["result"]
                if (
                    self.loss_threshold is not None
                    and isinstance(result, dict)
                    and result.get("status") == STATUS_OK
                ):
                    try:
                        reached = result.get("loss") <= self.loss_threshold
                    except TypeError:
                        reached = False
                    if reached:
                        self._cancel_trials(FMIN_CANCELLED_REASON_EARLY_STOPPING)

            self._launch_pending_trials()

    def _cancel_trials(self, reason):
        with self._spark_lock:
            if self._fmin_cancelled:
                return

            self._fmin_cancelled = True
            self._fmin_cancelled_reason = reason
            now = coarse_utcnow()

            groups = []
            for trial in self._dynamic_trials:
                if trial.get("state") in (
                    base.JOB_STATE_NEW,
                    base.JOB_STATE_RUNNING,
                ):
                    trial["state"] = base.JOB_STATE_CANCEL
                    trial["refresh_time"] = now
                    if trial.get("book_time") is None:
                        trial["book_time"] = now

            for active in self._spark_active_jobs.values():
                groups.append(active["job_group_id"])

            if self._spark_supports_job_cancelling:
                for group in groups:
                    try:
                        self._spark_context.cancelJobGroup(group)
                    except Exception:
                        logger.debug(
                            "Unable to cancel Spark job group %s", group, exc_info=True
                        )

    def fmin(
        self,
        fn,
        space,
        algo,
        max_evals,
        timeout=None,
        loss_threshold=None,
        **kwargs
    ):
        if timeout is not None:
            validate_timeout(timeout)
            self.timeout = timeout
        if loss_threshold is not None:
            validate_loss_threshold(loss_threshold)
            self.loss_threshold = loss_threshold

        self._fmin_cancelled = False
        self._fmin_cancelled_reason = None
        self._fmin_start_time = timeit.default_timer()

        domain_kwargs = {}
        if "workdir" in kwargs:
            domain_kwargs["workdir"] = kwargs["workdir"]
        if "pass_expr_memo_ctrl" in kwargs:
            domain_kwargs["pass_expr_memo_ctrl"] = kwargs[
                "pass_expr_memo_ctrl"
            ]
        domain_kwargs["loss_target"] = self.loss_threshold
        self._spark_domain = base.Domain(fn, space, **domain_kwargs)

        if self.timeout is not None:
            self._timeout_timer = threading.Timer(
                self.timeout,
                self._timeout_reached,
            )
            self._timeout_timer.daemon = True
            self._timeout_timer.start()

        call_kwargs = dict(kwargs)
        call_kwargs["trials"] = self
        call_kwargs["timeout"] = self.timeout
        call_kwargs["loss_threshold"] = self.loss_threshold
        call_kwargs["max_queue_len"] = self.parallelism

        try:
            return fmin(
                fn,
                space,
                algo=algo,
                max_evals=max_evals,
                **call_kwargs
            )
        except KeyboardInterrupt:
            self._cancel_trials(FMIN_CANCELLED_REASON_USER)
            raise
        finally:
            if self._timeout_timer is not None:
                self._timeout_timer.cancel()
                self._timeout_timer = None
            self._spark_domain = None
            self._fmin_start_time = None