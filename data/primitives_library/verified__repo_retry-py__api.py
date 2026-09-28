import logging
import random
import time
from functools import partial

from .compat import decorator


logging_logger = logging.getLogger(__name__)


def __retry_internal(f, exceptions=Exception, tries=-1, delay=0, max_delay=None,
                     backoff=1, jitter=0, logger=logging_logger):
    remaining = tries
    wait_time = delay

    while remaining:
        try:
            return f()
        except exceptions as error:
            remaining -= 1
            if not remaining:
                raise

            if logger is not None:
                logger.warning(
                    '%s, retrying in %s seconds...',
                    error,
                    wait_time,
                )

            time.sleep(wait_time)
            wait_time *= backoff

            if isinstance(jitter, tuple):
                wait_time += random.uniform(*jitter)
            else:
                wait_time += jitter

            if max_delay is not None:
                wait_time = min(wait_time, max_delay)


def retry(exceptions=Exception, tries=-1, delay=0, max_delay=None, backoff=1,
          jitter=0, logger=logging_logger):
    @decorator
    def retry_decorator(f, *fargs, **fkwargs):
        positional = fargs if fargs else list()
        named = fkwargs if fkwargs else dict()
        call = partial(f, *positional, **named)

        return __retry_internal(
            call,
            exceptions,
            tries,
            delay,
            max_delay,
            backoff,
            jitter,
            logger,
        )

    return retry_decorator


def retry_call(f, fargs=None, fkwargs=None, exceptions=Exception, tries=-1,
               delay=0, max_delay=None, backoff=1, jitter=0,
               logger=logging_logger):
    positional = fargs if fargs else list()
    named = fkwargs if fkwargs else dict()
    call = partial(f, *positional, **named)

    return __retry_internal(
        call,
        exceptions,
        tries,
        delay,
        max_delay,
        backoff,
        jitter,
        logger,
    )