# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg329::math.ceil+math.log+random.random
# name: math_random_primitive
# summary: Uses math.ceil, math.log, random.random across 3 repos
# anchor_symbols: ['math.ceil', 'math.log', 'random.random']
# observed in 3 repos: ['HunterMcGushion__hyperparameter_hunter', 'insitro__redun', 'mahmoud__boltons']...

# --- from insitro__redun::examples/allele_freq/workflow.py::sample_exponential ---
def sample_exponential(lambd: float) -> float:
    """
    Sample from exponential distribution with rate lambda.
    """
    return -log(random.random()) / lambd

# --- from HunterMcGushion__hyperparameter_hunter::hyperparameter_hunter/utils/boltons_utils.py::backoff_iter ---
def backoff_iter(start, stop, count=None, factor=2.0, jitter=False):
    """Generates a sequence of geometrically-increasing floats, suitable
    for usage with `exponential backoff`_. Starts with *start*,
    increasing by *factor* until *stop* is reached, optionally
    stopping iteration once *count* numbers are yielded. *factor*
    defaults to 2. In general retrying with properly-configured
    backoff creates a better-behaved component for a larger service
    ecosystem.
    .. _exponential backoff: https://en.wikipedia.org/wiki/Exponential_backoff
    >>> list(backoff_iter(1.0, 10.0, count=5))
    [1.0, 2.0, 4.0, 8.0, 10.0]
    >>> list(backoff_iter(1.0, 10.0, count=8))
    [1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0]
    >>> list(backoff_iter(0.25, 100.0, factor=10))
    [0.25, 2.5, 25.0, 100.0]

    A simplified usage example:
    .. code-block:: python
      for timeout in backoff_iter(0.25, 5.0):
          try:
              res = network_call()
              break
          except Exception as e:
              log(e)
              time.sleep(timeout)
    An enhancement for large-scale systems would be to add variation,
    or *jitter*, to timeout values. This is done to avoid a thundering
    herd on the receiving end of the network call.
    Finally, for *count*, the special value ``'repeat'`` can be passed to
    continue yielding indefinitely.
    Args:
        start (float): Positive number for baseline.
        stop (float): Positive number for maximum.
        count (int): Number of steps before stopping
            iteration. Defaults to the number of steps between *start* and
            *stop*. Pass the string, `'repeat'`, to continue iteration
            indefinitely.
        factor (float): Rate of exponential increase. Defaults to `2.0`,
            e.g., `[1, 2, 4, 8, 16]`.
        jitter (float): A factor between `-1.0` and `1.0`, used to
            uniformly randomize and thus spread out timeouts in a distributed
            system, avoiding rhythm effects. Positive values use the base
            backoff curve as a maximum, negative values use the curve as a
            minimum. Set to 1.0 or `True` for a jitter approximating
            Ethernet's time-tested backoff solution. Defaults to `False`.
    """
    start = float(start)
    stop = float(stop)
    factor = float(factor)
    if start < 0.0:
        raise ValueError("expected start >= 0, not %r" % start)
    if factor < 1.0:
        raise ValueError("expected factor >= 1.0, not %r" % factor)
    if stop == 0.0:
        raise ValueError("expected stop >= 0")
    if stop < start:
        raise ValueError("expected stop >= start, not %r" % stop)
    if count is None:
        denom = start if start else 1
        count = 1 + math.ceil(math.log(stop / denom, factor))
        count = count if start else count + 1
    if count != "repeat" and count < 0:
        raise ValueError('count must be positive or "repeat", not %r' % count)
    if jitter:
        jitter = float(jitter)
        if not (-1.0 <= jitter <= 1.0):
            raise ValueError("expected jitter -1 <= j <= 1, not: %r" % jitter)

    cur, i = start, 0
    while count == "repeat" or i < count:
        if not jitter:
            cur_ret = cur
        elif jitter:
            cur_ret = cur - (cur * jitter * random.random())
        yield cur_ret
        i += 1
        if cur == 0:
            cur = 1
        elif cur < stop:
            cur *= factor
        if cur > stop:
            cur = stop
    return

# --- from mahmoud__boltons::boltons/iterutils.py::backoff_iter ---
def backoff_iter(start, stop, count=None, factor=2.0, jitter=False):
    """Generates a sequence of geometrically-increasing floats, suitable
    for usage with `exponential backoff`_. Starts with *start*,
    increasing by *factor* until *stop* is reached, optionally
    stopping iteration once *count* numbers are yielded. *factor*
    defaults to 2. In general retrying with properly-configured
    backoff creates a better-behaved component for a larger service
    ecosystem.

    .. _exponential backoff: https://en.wikipedia.org/wiki/Exponential_backoff

    >>> list(backoff_iter(1.0, 10.0, count=5))
    [1.0, 2.0, 4.0, 8.0, 10.0]
    >>> list(backoff_iter(1.0, 10.0, count=8))
    [1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0]
    >>> list(backoff_iter(0.25, 100.0, factor=10))
    [0.25, 2.5, 25.0, 100.0]

    A simplified usage example:

    .. code-block:: python

      for timeout in backoff_iter(0.25, 5.0):
          try:
              res = network_call()
              break
          except Exception as e:
              log(e)
              time.sleep(timeout)

    An enhancement for large-scale systems would be to add variation,
    or *jitter*, to timeout values. This is done to avoid a thundering
    herd on the receiving end of the network call.

    Finally, for *count*, the special value ``'repeat'`` can be passed to
    continue yielding indefinitely.

    Args:

        start (float): Positive number for baseline.
        stop (float): Positive number for maximum.
        count (int): Number of steps before stopping
            iteration. Defaults to the number of steps between *start* and
            *stop*. Pass the string, `'repeat'`, to continue iteration
            indefinitely.
        factor (float): Rate of exponential increase. Defaults to `2.0`,
            e.g., `[1, 2, 4, 8, 16]`.
        jitter (float): A factor between `-1.0` and `1.0`, used to
            uniformly randomize and thus spread out timeouts in a distributed
            system, avoiding rhythm effects. Positive values use the base
            backoff curve as a maximum, negative values use the curve as a
            minimum. Set to 1.0 or `True` for a jitter approximating
            Ethernet's time-tested backoff solution. Defaults to `False`.

    """
    start = float(start)
    stop = float(stop)
    factor = float(factor)
    if start < 0.0:
        raise ValueError('expected start >= 0, not %r' % start)
    if factor < 1.0:
        raise ValueError('expected factor >= 1.0, not %r' % factor)
    if stop == 0.0:
        raise ValueError('expected stop >= 0')
    if stop < start:
        raise ValueError('expected stop >= start, not %r' % stop)
    if count is None:
        denom = start if start else 1
        count = 1 + math.ceil(math.log(stop/denom, factor))
        count = count if start else count + 1
    if count != 'repeat' and count < 0:
        raise ValueError('count must be positive or "repeat", not %r' % count)
    if jitter:
        jitter = float(jitter)
        if not (-1.0 <= jitter <= 1.0):
            raise ValueError('expected jitter -1 <= j <= 1, not: %r' % jitter)

    cur, i = start, 0
    while count == 'repeat' or i < count:
        if not jitter:
            cur_ret = cur
        elif jitter:
            cur_ret = cur - (cur * jitter * random.random())
        yield cur_ret
        i += 1
        if cur == 0:
            cur = 1
        elif cur < stop:
            cur *= factor
        if cur > stop:
            cur = stop
    return
