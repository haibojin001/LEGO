"""
Constructs for annotating base graphs.
"""

import sys

import numpy as np

from .base import as_apply, clone, dfs, rec_eval, scope


def ERR(msg):
    print(msg, file=sys.stderr)


implicit_stochastic_symbols = set()


def implicit_stochastic(fn):
    implicit_stochastic_symbols.add(fn.__name__)
    return fn


@scope.define
def rng_from_seed(seed):
    return np.random.default_rng(seed)


@implicit_stochastic
@scope.define
def uniform(low, high, rng=None, size=()):
    return rng.uniform(low, high, size=size)


@implicit_stochastic
@scope.define
def loguniform(low, high, rng=None, size=()):
    return np.exp(rng.uniform(low, high, size=size))


@implicit_stochastic
@scope.define
def quniform(low, high, q, rng=None, size=()):
    return np.round(rng.uniform(low, high, size=size) / q) * q


@implicit_stochastic
@scope.define
def qloguniform(low, high, q, rng=None, size=()):
    return np.round(np.exp(rng.uniform(low, high, size=size)) / q) * q


@implicit_stochastic
@scope.define
def normal(mu, sigma, rng=None, size=()):
    return rng.normal(mu, sigma, size=size)


@implicit_stochastic
@scope.define
def qnormal(mu, sigma, q, rng=None, size=()):
    return np.round(rng.normal(mu, sigma, size=size) / q) * q


@implicit_stochastic
@scope.define
def lognormal(mu, sigma, rng=None, size=()):
    return np.exp(rng.normal(mu, sigma, size=size))


@implicit_stochastic
@scope.define
def qlognormal(mu, sigma, q, rng=None, size=()):
    return np.round(np.exp(rng.normal(mu, sigma, size=size)) / q) * q


@implicit_stochastic
@scope.define
def randint(low, high=None, rng=None, size=()):
    """
    See np.random.randint documentation.
    rng = random number generator, typically equals np.random.Generator
    """
    return rng.integers(low, high, size)


@implicit_stochastic
@scope.define
def randint_via_categorical(p, rng=None, size=()):
    """
    Draw integer samples through categorical, chiefly for use with priors.
    """
    return scope.categorical(p, rng, size)


@implicit_stochastic
@scope.define
def categorical(p, rng=None, size=()):
    """Draw i with probability p[i]."""
    if len(p) == 1 and isinstance(p[0], np.ndarray):
        p = p[0]
    p = np.asarray(p)

    if size == ():
        size = (1,)
    elif isinstance(size, (int, np.number)):
        size = (size,)
    else:
        size = tuple(size)

    if size == (0,):
        return np.asarray([])
    assert len(size)

    if p.ndim == 0:
        raise NotImplementedError()
    if p.ndim == 1:
        count = int(np.prod(size))
        draws = rng.multinomial(n=1, pvals=p, size=count)
        assert draws.shape == size + (len(p),)
        result = np.dot(draws, np.arange(len(p)))
        result.shape = size
        return result
    if p.ndim == 2:
        number_of_draws, _ = p.shape
        (requested_draws,) = size
        assert requested_draws == number_of_draws
        result = [
            np.where(rng.multinomial(pvals=p[index], n=1))[0][0]
            for index in range(requested_draws)
        ]
        result = np.asarray(result)
        result.shape = size
        return result
    raise NotImplementedError()


def choice(args):
    return scope.one_of(*args)


scope.choice = choice


def one_of(*args):
    index = scope.randint(len(args))
    return scope.switch(index, *args)


scope.one_of = one_of


def recursive_set_rng_kwarg(expr, rng=None):
    """
    Make every implicit stochastic node in expr receive rng as its rng keyword.
    """
    if rng is None:
        rng = np.random.default_rng()
    rng_expr = as_apply(rng)

    for node in dfs(expr):
        if node.name in implicit_stochastic_symbols:
            for index, (name, value) in enumerate(list(node.named_args)):
                if name == "rng":
                    node.named_args[index] = ("rng", rng_expr)
                    break
            else:
                node.named_args.append(("rng", rng_expr))
    return expr


def sample(expr, rng=None, **kwargs):
    """
    Evaluate a pyll expression after attaching a random number generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    expression = clone(as_apply(expr))
    recursive_set_rng_kwarg(expression, as_apply(rng))
    return rec_eval(expression, **kwargs)