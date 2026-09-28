import datetime
import logging
import os
import shutil
import sys
import uuid
from contextlib import contextmanager

import numpy
import numpy as np

from . import pyll


def _get_random_id():
    return uuid.uuid4().hex[-12:]


def _get_logger(name):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers and not logging.getLogger().handlers:
        log.addHandler(logging.StreamHandler(sys.stderr))
    return log


logger = _get_logger(__name__)

try:
    import cloudpickle as pickler
except Exception:
    logger.info(
        "Failed to load cloudpickle, try installing cloudpickle via "
        '"pip install cloudpickle" for enhanced pickling support.'
    )
    import pickle as pickler


def import_tokens(tokens):
    result = None
    for index in range(len(tokens)):
        module_name = ".".join(tokens[: index + 1])
        try:
            logger.info("importing %s" % module_name)
            namespace = {}
            exec("import %s" % module_name, namespace)
            exec("rval = %s" % module_name, namespace)
            result = namespace["rval"]
        except ImportError as exc:
            logger.info("failed to import %s" % module_name)
            logger.info("reason: %s" % str(exc))
            break
    return result, tokens[index:]


def load_tokens(tokens):
    logger.info("load_tokens: %s" % str(tokens))
    value, remaining = import_tokens(tokens)
    for name in remaining:
        value = getattr(value, name)
    return value


def json_lookup(json):
    return load_tokens(json.split("."))


def json_call(json, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if isinstance(json, (str, bytes)):
        return json_lookup(json)(*args, **kwargs)
    if isinstance(json, dict):
        raise NotImplementedError("dict calling convention undefined", json)
    if isinstance(json, (tuple, list)):
        raise NotImplementedError("seq calling convention undefined", json)
    raise TypeError(json)


def get_obj(f, argfile=None, argstr=None, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if argfile is not None:
        argstr = open(argfile).read()
    if argstr is not None:
        supplied = pickler.loads(argstr)
    else:
        supplied = {}
    args = args + supplied.get("args", ())
    kwargs.update(supplied.get("kwargs", {}))
    return json_call(f, args=args, kwargs=kwargs)


def pmin_sampled(mean, var, n_samples=1000, rng=None):
    if rng is None:
        rng = numpy.random.default_rng(232342)

    draws = rng.standard_normal((n_samples, len(mean))) * numpy.sqrt(var) + mean
    winning = (draws.T == draws.min(axis=1)).T
    counts = winning.sum(axis=0)
    assert counts.shape == mean.shape
    return counts.astype("float64") / counts.sum()


def fast_isin(X, Y):
    if len(Y) > 0:
        ordered = Y.copy()
        ordered.sort()
        positions = ordered.searchsorted(X)
        ordered = np.append(ordered, np.array([0]))
        present = ordered[positions] == X
        if isinstance(present, bool):
            return np.zeros((len(X),), bool)
        return ordered[positions] == X
    return np.zeros((len(X),), bool)


def get_most_recent_inds(obj):
    records = numpy.rec.array(
        [(item["_id"], int(item["version"])) for item in obj],
        names=["_id", "version"],
    )
    ordering = records.argsort(order=["_id", "version"])
    records = records[ordering]
    ends = (records["_id"][1:] != records["_id"][:-1]).nonzero()[0]
    ends = numpy.append(ends, [len(records) - 1])
    return ordering[ends]


def use_obj_for_literal_in_memo(expr, obj, lit, memo):
    for node in pyll.dfs(expr):
        try:
            if node.obj == lit:
                memo[node] = obj
        except (AttributeError, ValueError):
            pass
    return memo


def coarse_utcnow():
    now = datetime.datetime.now(datetime.timezone.utc)
    milliseconds = (now.microsecond // 1000) * 1000
    return datetime.datetime(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second,
        milliseconds,
    )


@contextmanager
def working_dir(dir):
    previous = os.getcwd()
    os.chdir(dir)
    yield
    os.chdir(previous)


def path_split_all(path):
    components = []
    while True:
        path, leaf = os.path.split(path)
        if len(leaf) == 0:
            break
        components.append(leaf)
    return reversed(components)


def get_closest_dir(workdir):
    existing = ""
    for component in path_split_all(workdir):
        if os.path.isdir(os.path.join(existing, component)):
            existing = os.path.join(existing, component)
        else:
            break
    assert existing != workdir
    return existing, component


@contextmanager
def temp_dir(dir, erase_after=False, with_sentinel=True):
    made_here = False
    if not os.path.exists(dir):
        if os.pardir in dir:
            raise RuntimeError("workdir contains os.pardir ('..')")
        if erase_after and with_sentinel:
            closest, filename = get_closest_dir(dir)
            sentinel = os.path.join(closest, filename + ".inuse")
            open(sentinel, "w").close()
        os.makedirs(dir)
        made_here = True
    else:
        assert os.path.isdir(dir)

    yield

    if erase_after and made_here:
        shutil.rmtree(dir)
        if with_sentinel:
            os.mkdir(dir)
            os.removedirs(dir)
            os.remove(sentinel)