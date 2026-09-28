"""
Python job scheduling for humans.

github.com/dbader/schedule

An in-process scheduler for periodic jobs that uses the builder pattern
for configuration. Schedule lets you run Python functions (or any other
callable) periodically at pre-determined intervals using a simple,
human-friendly syntax.
"""

from collections.abc import Hashable
import datetime
import functools
import logging
import random
import re
import time
from typing import Callable, List, Optional, Set, Union

logger = logging.getLogger("schedule")

__version__ = "1.2.2"


class ScheduleError(Exception):
    """Base schedule exception"""

    pass


class ScheduleValueError(ScheduleError):
    """Base schedule value error"""

    pass


class IntervalError(ScheduleValueError):
    """An improper interval was used"""

    pass


class CancelJob:
    """
    Can be returned from a job to unschedule itself.
    """

    pass


class Scheduler:
    """
    Objects instantiated by the :class:`Scheduler <Scheduler>` are
    factories to create jobs, keep record of scheduled jobs and
    handle their execution.
    """

    def __init__(self) -> None:
        self.jobs: List["Job"] = []

    def run_pending(self) -> None:
        """
        Run all jobs that are scheduled to run.

        Please note that it is intended behavior that run_pending()
        does not run missed jobs.
        """
        runnable_jobs = (job for job in self.jobs if job.should_run)
        for job in sorted(runnable_jobs):
            self._run_job(job)

    def run_all(self, delay_seconds: int = 0) -> None:
        """
        Run all jobs regardless if they are scheduled to run or not.

        :param delay_seconds: A delay added between every executed job
        """
        logger.debug(
            "Running *all* %i jobs with %is delay in between",
            len(self.jobs),
            delay_seconds,
        )
        for job in self.jobs[:]:
            self._run_job(job)
            time.sleep(delay_seconds)

    def get_jobs(self, tag: Optional[Hashable] = None) -> List["Job"]:
        """
        Gets scheduled jobs marked with the given tag, or all jobs
        if tag is omitted.
        """
        if tag is None:
            return self.jobs[:]
        return [job for job in self.jobs if tag in job.tags]

    def clear(self, tag: Optional[Hashable] = None) -> None:
        """
        Deletes scheduled jobs marked with the given tag, or all jobs
        if tag is omitted.
        """
        if tag is None:
            logger.debug("Deleting *all* jobs")
            del self.jobs[:]
        else:
            logger.debug('Deleting all jobs tagged "%s"', tag)
            self.jobs[:] = [job for job in self.jobs if tag not in job.tags]

    def cancel_job(self, job: "Job") -> None:
        """
        Delete a scheduled job.
        """
        try:
            logger.debug('Cancelling job "%s"', str(job))
            self.jobs.remove(job)
        except ValueError:
            logger.debug('Cancelling not-scheduled job "%s"', str(job))

    def every(self, interval: int = 1) -> "Job":
        """
        Schedule a new periodic job.
        """
        return Job(interval, self)

    def _run_job(self, job: "Job") -> None:
        ret = job.run()
        if isinstance(ret, CancelJob) or ret is CancelJob:
            self.cancel_job(job)

    def get_next_run(
        self, tag: Optional[Hashable] = None
    ) -> Optional[datetime.datetime]:
        """
        Datetime when the next job should run.
        """
        if not self.jobs:
            return None
        jobs_filtered = self.get_jobs(tag)
        if not jobs_filtered:
            return None
        return min(jobs_filtered).next_run

    next_run = property(get_next_run)

    @property
    def idle_seconds(self) -> Optional[float]:
        """
        Number of seconds until next_run, or None if no jobs are scheduled.
        """
        if not self.next_run:
            return None
        return (self.next_run - datetime.datetime.now()).total_seconds()


class Job:
    """
    A periodic job as used by :class:`Scheduler`.
    """

    def __init__(self, interval: int, scheduler: Optional[Scheduler] = None):
        self.interval: int = interval
        self.latest: Optional[int] = None
        self.job_func: Optional[functools.partial] = None
        self.unit: Optional[str] = None
        self.at_time: Optional[datetime.time] = None
        self.at_time_zone = None
        self.last_run: Optional[datetime.datetime] = None
        self.next_run: Optional[datetime.datetime] = None
        self.start_day: Optional[str] = None
        self.cancel_after: Optional[datetime.datetime] = None
        self.tags: Set[Hashable] = set()
        self.scheduler: Optional[Scheduler] = scheduler

    def __lt__(self, other) -> bool:
        """
        Periodic jobs are sortable based on next run time.
        """
        return self.next_run < other.next_run

    def __str__(self) -> str:
        if self.job_func is None:
            job_func_name = "None"
            args = "()"
            kwargs = "{}"
        elif hasattr(self.job_func, "__name__"):
            job_func_name = self.job_func.__name__  # type: ignore[union-attr]
            args = self.job_func.args
            kwargs = self.job_func.keywords
        else:
            job_func_name = repr(self.job_func)
            args = self.job_func.args
            kwargs = self.job_func.keywords

        return "Job(interval={}, unit={}, do={}, args={}, kwargs={})".format(
            self.interval,
            self.unit,
            job_func_name,
            args,
            kwargs,
        )

    def __repr__(self):
        def format_time(t):
            return t.strftime("%Y-%m-%d %H:%M:%S") if t else "[never]"

        def format_arg(value):
            return str(value) if isinstance(value, Job) else repr(value)

        timestats = "(last run: %s, next run: %s)" % (
            format_time(self.last_run),
            format_time(self.next_run),
        )

        if self.job_func is None:
            call = "[None]"
        else:
            if hasattr(self.job_func, "__name__"):
                job_func_name = self.job_func.__name__
            else:
                job_func_name = repr(self.job_func)

            args = [format_arg(arg) for arg in self.job_func.args]
            kwargs = [
                "{}={}".format(key, format_arg(value))
                for key, value in (self.job_func.keywords or {}).items()
            ]
            call = "{}({})".format(job_func_name, ", ".join(args + kwargs))

        if self.latest is not None:
            interval = "{} to {}".format(self.interval, self.latest)
        else:
            interval = str(self.interval)

        unit = self.unit
        if (
            self.latest is None
            and self.interval == 1
            and isinstance(unit, str)
            and unit.endswith("s")
        ):
            unit = unit[:-1]

        at = ""
        if self.at_time is not None:
            at = " at {}".format(self.at_time)

        return "Every {} {}{} do {} {}".format(interval, unit, at, call, timestats)

    @property
    def second(self):
        if self.interval != 1:
            raise IntervalError("Use seconds instead of second")
        return self.seconds

    @property
    def seconds(self):
        self.unit = "seconds"
        return self

    @property
    def minute(self):
        if self.interval != 1:
            raise IntervalError("Use minutes instead of minute")
        return self.minutes

    @property
    def minutes(self):
        self.unit = "minutes"
        return self

    @property
    def hour(self):
        if self.interval != 1:
            raise IntervalError("Use hours instead of hour")
        return self.hours

    @property
    def hours(self):
        self.unit = "hours"
        return self

    @property
    def day(self):
        if self.interval != 1:
            raise IntervalError("Use days instead of day")
        return self.days

    @property
    def days(self):
        self.unit = "days"
        return self

    @property
    def week(self):
        if self.interval != 1:
            raise IntervalError("Use weeks instead of week")
        return self.weeks

    @property
    def weeks(self):
        self.unit = "weeks"
        return self

    def _weekday(self, day: str):
        if self.interval != 1:
            raise IntervalError(
                "Scheduling .{}() jobs is only allowed for weekly jobs. "
                "Using .{}() on a job scheduled to run every 2 or more weeks "
                "is not supported.".format(day, day)
            )
        self.start_day = day
        return self.weeks

    @property
    def monday(self):
        return self._weekday("monday")

    @property
    def tuesday(self):
        return self._weekday("tuesday")

    @property
    def wednesday(self):
        return self._weekday("wednesday")

    @property
    def thursday(self):
        return self._weekday("thursday")

    @property
    def friday(self):
        return self._weekday("friday")

    @property
    def saturday(self):
        return self._weekday("saturday")

    @property
    def sunday(self):
        return self._weekday("sunday")

    def tag(self, *tags: Hashable):
        """
        Tags the job with one or more unique identifiers.
        """
        if not all(isinstance(tag, Hashable) for tag in tags):
            raise TypeError("Tags must be hashable")
        self.tags.update(tags)
        return self

    def at(self, time_str: str, tz: Optional[str] = None):
        """
        Specify a particular time that the job should be run at.
        """
        if self.unit not in ("days", "hours", "minutes") and self.start_day is None:
            raise ScheduleValueError(
                "Invalid unit (valid units are `days`, `hours`, and `minutes`)"
            )

        if tz is not None:
            try:
                import pytz
            except ImportError:
                raise ImportError("Timezone support requires the pytz module")

            if isinstance(tz, str):
                self.at_time_zone = pytz.timezone(tz)
            elif isinstance(tz, pytz.BaseTzInfo):
                self.at_time_zone = tz
            else:
                raise ScheduleValueError("Timezone must be string or pytz.timezone object")

        if not isinstance(time_str, str):
            raise TypeError("at() should be passed a string")

        if self.unit == "days" or self.start_day:
            if not re.match(r"^[0-2]\d:[0-5]\d(:[0-5]\d)?$", time_str):
                raise ScheduleValueError(
                    "Invalid time format for a daily job (valid format is HH:MM(:SS)?)"
                )
        elif self.unit == "hours":
            if not re.match(r"^([0-5]\d)?:[0-5]\d$", time_str):
                raise ScheduleValueError(
                    "Invalid time format for an hourly job (valid format is (MM)?:SS)"
                )
        elif self.unit == "minutes":
            if not re.match(r"^:[0-5]\d$", time_str):
                raise ScheduleValueError(
                    "Invalid time format for a minutely job (valid format is :SS)"
                )

        time_values = time_str.split(":")
        hour = minute = second = 0

        if len(time_values) == 3:
            hour, minute, second = map(int, time_values)
        elif self.unit == "minutes":
            hour = 0
            minute = 0
            second = int(time_values[1])
        elif self.unit == "hours":
            hour = 0
            if time_values[0] == "":
                minute = int(time_values[1])
                second = 0
            else:
                minute = int(time_values[0])
                second = int(time_values[1])
        else:
            hour = int(time_values[0])
            minute = int(time_values[1])
            second = 0

        if hour > 23:
            raise ScheduleValueError("Invalid number of hours")
        if minute > 59:
            raise ScheduleValueError("Invalid number of minutes")
        if second > 59:
            raise ScheduleValueError("Invalid number of seconds")

        self.at_time = datetime.time(hour, minute, second)
        return self

    def to(self, latest: int):
        """
        Schedule the job to run at an irregular interval.
        """
        self.latest = latest
        return self

    def until(
        self,
        until_time: Union[datetime.datetime, datetime.timedelta, datetime.time, str],
    ):
        """
        Schedule job to run until the specified moment.
        """
        if isinstance(until_time, datetime.datetime):
            self.cancel_after = until_time
        elif isinstance(until_time, datetime.timedelta):
            self.cancel_after = datetime.datetime.now() + until_time
        elif isinstance(until_time, datetime.time):
            self.cancel_after = datetime.datetime.combine(
                datetime.datetime.now().date(), until_time
            )
        elif isinstance(until_time, str):
            cancel_after = self._decode_datetimestr(
                until_time,
                [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%d",
                    "%H:%M:%S",
                    "%H:%M",
                ],
            )
            if cancel_after is None:
                raise ScheduleValueError("Invalid string format for until()")
            if "-" not in until_time:
                cancel_after = datetime.datetime.combine(
                    datetime.datetime.now().date(), cancel_after.time()
                )
            self.cancel_after = cancel_after
        else:
            raise TypeError(
                "until() takes a datetime.datetime, datetime.timedelta, "
                "datetime.time, or string"
            )

        if self.cancel_after < datetime.datetime.now():
            raise ScheduleValueError("Cannot schedule a job to run until a time in the past")

        return self

    @staticmethod
    def _decode_datetimestr(datetime_str: str, formats: List[str]):
        for f in formats:
            try:
                return datetime.datetime.strptime(datetime_str, f)
            except ValueError:
                pass
        return None

    def do(self, job_func: Callable, *args, **kwargs):
        """
        Specifies the job_func that should be called every time the job runs.
        """
        self.job_func = functools.partial(job_func, *args, **kwargs)
        functools.update_wrapper(self.job_func, job_func)
        self._schedule_next_run()
        if self.scheduler is not None:
            self.scheduler.jobs.append(self)
        return self

    @property
    def should_run(self) -> bool:
        """
        True if the job should be run now.
        """
        if self.next_run is None:
            raise ScheduleError("must run _schedule_next_run before")
        return datetime.datetime.now() >= self.next_run

    def run(self):
        """
        Run the job and immediately reschedule it.
        """
        if self._is_overdue(datetime.datetime.now()):
            logger.debug("Cancelling job %s", self)
            return CancelJob

        logger.debug("Running job %s", self)
        if self.job_func is None:
            raise ScheduleError("Job function is not set")

        ret = self.job_func()
        self.last_run = datetime.datetime.now()
        self._schedule_next_run()

        if self._is_overdue(self.next_run):
            logger.debug("Cancelling job %s", self)
            return CancelJob
        return ret

    def _schedule_next_run(self) -> None:
        """
        Compute the instant when this job should run next.
        """
        if self.unit not in ("seconds", "minutes", "hours", "days", "weeks"):
            raise ScheduleValueError(
                "Invalid unit (valid units are `seconds`, `minutes`, `hours`, "
                "`days`, and `weeks`)"
            )

        if self.latest is not None:
            if self.latest < self.interval:
                raise ScheduleError("`latest` is greater than `interval`")
            interval = random.randint(self.interval, self.latest)
        else:
            interval = self.interval

        now = datetime.datetime.now(self.at_time_zone)
        next_run = now

        if self.start_day is not None:
            if self.unit != "weeks":
                raise ScheduleValueError("`unit` should be 'weeks'")
            next_run = _move_to_next_weekday(next_run, self.start_day)

        if self.at_time is not None:
            next_run = self._move_to_at_time(next_run)

        period = datetime.timedelta(**{self.unit: interval})
        if interval != 1:
            next_run += period

        while next_run <= now:
            next_run += period

        next_run = self._correct_utc_offset(
            next_run, fixate_time=self.at_time is not None
        )

        if self.at_time_zone is not None:
            next_run = next_run.astimezone()
            next_run = next_run.replace(tzinfo=None)

        self.next_run = next_run

    def _move_to_at_time(self, moment: datetime.datetime) -> datetime.datetime:
        if self.at_time is None:
            return moment

        kwargs = {"second": self.at_time.second, "microsecond": 0}
        if self.unit == "days" or self.start_day is not None:
            kwargs["hour"] = self.at_time.hour
        if self.unit in ("days", "hours") or self.start_day is not None:
            kwargs["minute"] = self.at_time.minute

        moment = moment.replace(**kwargs)
        moment = self._correct_utc_offset(moment, fixate_time=True)
        return moment

    def _correct_utc_offset(
        self, moment: datetime.datetime, fixate_time: bool
    ) -> datetime.datetime:
        if self.at_time_zone is None:
            return moment

        offset_before = moment.utcoffset()
        moment = self.at_time_zone.normalize(moment)
        offset_after = moment.utcoffset()

        if offset_before == offset_after:
            return moment

        if not fixate_time:
            return moment

        offset_diff = offset_after - offset_before
        moment -= offset_diff

        re_normalized = self.at_time_zone.normalize(moment)
        if re_normalized.utcoffset() != offset_after:
            moment += offset_diff

        return moment

    def _is_overdue(self, when: Optional[datetime.datetime]) -> bool:
        return self.cancel_after is not None and when is not None and when > self.cancel_after


def _move_to_next_weekday(moment: datetime.datetime, weekday: str):
    weekdays = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    if weekday not in weekdays:
        raise ScheduleValueError("Invalid start day")
    weekday_index = weekdays.index(weekday)
    days_ahead = weekday_index - moment.weekday()
    if days_ahead < 0:
        days_ahead += 7
    return moment + datetime.timedelta(days=days_ahead)


default_scheduler = Scheduler()
jobs = default_scheduler.jobs


def every(interval: int = 1) -> Job:
    return default_scheduler.every(interval)


def run_pending() -> None:
    default_scheduler.run_pending()


def run_all(delay_seconds: int = 0) -> None:
    default_scheduler.run_all(delay_seconds=delay_seconds)


def get_jobs(tag: Optional[Hashable] = None) -> List[Job]:
    return default_scheduler.get_jobs(tag)


def clear(tag: Optional[Hashable] = None) -> None:
    default_scheduler.clear(tag)


def cancel_job(job: Job) -> None:
    default_scheduler.cancel_job(job)


def next_run(tag: Optional[Hashable] = None) -> Optional[datetime.datetime]:
    return default_scheduler.get_next_run(tag)


def idle_seconds() -> Optional[float]:
    return default_scheduler.idle_seconds


def repeat(job: Job, *args, **kwargs):
    def _schedule_decorator(decorated_function):
        job.do(decorated_function, *args, **kwargs)
        return decorated_function

    return _schedule_decorator


__all__ = [
    "CancelJob",
    "IntervalError",
    "Job",
    "ScheduleError",
    "ScheduleValueError",
    "Scheduler",
    "cancel_job",
    "clear",
    "default_scheduler",
    "every",
    "get_jobs",
    "idle_seconds",
    "jobs",
    "next_run",
    "repeat",
    "run_all",
    "run_pending",
]