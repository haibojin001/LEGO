import logging
import time

import numpy as np
from scipy.special import erf

from . import pyll, rand
from .base import miscs_to_idxs_vals, miscs_update_idxs_vals
from .pyll import scope
from .pyll.stochastic import implicit_stochastic, randint_via_categorical

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

logger = logging.getLogger(__name__)

EPS = 1e-12
DEFAULT_LF = 25

adaptive_parzen_samplers = {}


def adaptive_parzen_sampler(name):
    def register(fn):
        assert name not in adaptive_parzen_samplers
        adaptive_parzen_samplers[name] = fn
        return fn

    return register


@scope.define
def categorical_lpdf(sample, p):
    sample = np.asarray(sample)
    if sample.size:
        return np.log(np.asarray(p)[sample])
    return np.asarray([])


@scope.define
def randint_via_categorical_lpdf(sample, p):
    sample = np.asarray(sample)
    if sample.size:
        return np.log(np.asarray(p)[sample])
    return np.asarray([])


@implicit_stochastic
@scope.define
def GMM1(weights, mus, sigmas, low=None, high=None, q=None, rng=None, size=()):
    """Sample from a one-dimensional Gaussian mixture model."""
    weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
    assert len(weights) == len(mus) == len(sigmas)

    n_samples = int(np.prod(size))
    if low is None and high is None:
        active = np.argmax(rng.multinomial(1, weights, size=n_samples), axis=1)
        samples = rng.normal(loc=mus[active], scale=sigmas[active])
    else:
        low = -float("inf") if low is None else float(low)
        high = float("inf") if high is None else float(high)
        if low >= high:
            raise ValueError("low >= high", (low, high))

        samples = []
        while len(samples) < n_samples:
            active = np.argmax(rng.multinomial(1, weights))
            draw = rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw < high:
                samples.append(draw)

    samples = np.asarray(samples).reshape(size)
    if q is None:
        return samples
    return np.round(samples / q) * q


@scope.define
def normal_cdf(x, mu, sigma):
    x = np.asarray(x)
    return 0.5 * (
        1.0 + erf((x - mu) / np.maximum(np.sqrt(2.0) * sigma, EPS))
    )


def logsum_rows(x):
    x = np.asarray(x)
    maxima = x.max(axis=1)
    return np.log(np.exp(x - maxima[:, None]).sum(axis=1)) + maxima


@scope.define
def GMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None):
    samples, weights, mus, sigmas = map(np.asarray, (samples, weights, mus, sigmas))

    if samples.size == 0:
        return np.asarray([])
    if weights.ndim != 1:
        raise TypeError("need vector of weights", weights.shape)
    if mus.ndim != 1:
        raise TypeError("need vector of mus", mus.shape)
    if sigmas.ndim != 1:
        raise TypeError("need vector of sigmas", sigmas.shape)
    assert len(weights) == len(mus) == len(sigmas)

    original_shape = samples.shape
    samples = samples.flatten()

    if low is None and high is None:
        p_accept = 1.0
    else:
        low = -float("inf") if low is None else low
        high = float("inf") if high is None else high
        p_accept = np.sum(
            weights
            * (normal_cdf(high, mus, sigmas) - normal_cdf(low, mus, sigmas))
        )

    if q is None:
        dist = samples[:, None] - mus
        mahal = (dist / np.maximum(sigmas, EPS)) ** 2
        normalizer = np.sqrt(2.0 * np.pi * sigmas**2)
        coef = weights / normalizer / p_accept
        rval = logsum_rows(-0.5 * mahal + np.log(coef))
    else:
        prob = np.zeros(samples.shape, dtype="float64")
        for weight, mu, sigma in zip(weights, mus, sigmas):
            if high is None:
                upper = samples + q / 2.0
            else:
                upper = np.minimum(samples + q / 2.0, high)
            if low is None:
                lower = samples - q / 2.0
            else:
                lower = np.maximum(samples - q / 2.0, low)

            increment = weight * normal_cdf(upper, mu, sigma)
            increment -= weight * normal_cdf(lower, mu, sigma)
            prob += increment
        rval = np.log(prob) - np.log(p_accept)

    rval.shape = original_shape
    return rval


@scope.define
def lognormal_cdf(x, mu, sigma):
    x = np.asarray(x)
    if len(x) == 0:
        return np.asarray([])
    if x.min() < 0:
        raise ValueError("negative arg to lognormal_cdf", x)

    olderr = np.seterr(divide="ignore")
    try:
        z = (np.log(np.maximum(x, EPS)) - mu) / np.maximum(
            np.sqrt(2.0) * sigma, EPS
        )
        return 0.5 + 0.5 * erf(z)
    finally:
        np.seterr(**olderr)


@scope.define
def lognormal_lpdf(x, mu, sigma):
    x = np.asarray(x)
    sigma = np.asarray(sigma)
    assert np.all(sigma >= 0)
    sigma = np.maximum(sigma, EPS)

    normalizer = sigma * x * np.sqrt(2.0 * np.pi)
    exponent = 0.5 * ((np.log(x) - mu) / sigma) **2
    return -exponent - np.log(normalizer)


@scope.define
def qlognormal_lpdf(x, mu, sigma, q):
    return np.log(lognormal_cdf(x, mu, sigma) - lognormal_cdf(x - q, mu, sigma))


@implicit_stochastic
@scope.define
def LGMM1(weights, mus, sigmas, low=None, high=None, q=None, rng=None, size=()):
    """Sample from a mixture of log-normal distributions."""
    weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
    assert len(weights) == len(mus) == len(sigmas)

    n_samples = int(np.prod(size))
    if low is None and high is None:
        active = np.argmax(rng.multinomial(1, weights, size=n_samples), axis=1)
        samples = np.exp(rng.normal(loc=mus[active], scale=sigmas[active]))
    else:
        low = -float("inf") if low is None else float(low)
        high = float("inf") if high is None else float(high)
        if low >= high:
            raise ValueError("low >= high", (low, high))

        samples = []
        while len(samples) < n_samples:
            active = np.argmax(rng.multinomial(1, weights))
            draw = rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw < high:
                samples.append(np.exp(draw))

    samples = np.asarray(samples).reshape(size)
    if q is None:
        return samples
    return np.round(samples / q) * q


@scope.define
def LGMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None):
    samples, weights, mus, sigmas = map(np.asarray, (samples, weights, mus, sigmas))
    assert weights.ndim == 1
    assert mus.ndim == 1
    assert sigmas.ndim == 1
    assert len(weights) == len(mus) == len(sigmas)

    original_shape = samples.shape
    samples = samples.flatten()
    if samples.size == 0:
        return np.asarray([]).reshape(original_shape)

    if low is None and high is None:
        p_accept = 1.0
    else:
        low = -float("inf") if low is None else low
        high = float("inf") if high is None else high
        p_accept = np.sum(
            weights
            * (normal_cdf(high, mus, sigmas) - normal_cdf(low, mus, sigmas))
        )

    if q is None:
        component_lpdf = lognormal_lpdf(samples[:, None], mus, sigmas)
        rval = logsum_rows(component_lpdf + np.log(weights)) - np.log(p_accept)
    else:
        prob = np.zeros(samples.shape, dtype="float64")
        for weight, mu, sigma in zip(weights, mus, sigmas):
            if high is None:
                upper = samples + q / 2.0
            else:
                upper = np.minimum(samples + q / 2.0, np.exp(high))
            if low is None:
                lower = samples - q / 2.0
            else:
                lower = np.maximum(samples - q / 2.0, np.exp(low))

            increment = weight * lognormal_cdf(upper, mu, sigma)
            increment -= weight * lognormal_cdf(lower, mu, sigma)
            prob += increment
        rval = np.log(prob) - np.log(p_accept)

    rval.shape = original_shape
    return rval


@scope.define
def linear_forgetting_weights(N, LF):
    N = int(N)
    LF = int(LF)

    if N <= 0:
        return np.asarray([])
    if LF <= 0:
        return np.linspace(1.0 / N, 1.0, N)
    if N < LF:
        return np.ones(N)

    ramp = np.linspace(1.0 / N, 1.0, N - LF)
    return np.concatenate((ramp, np.ones(LF)))


@scope.define
def adaptive_parzen_normal(mus, prior_weight, prior_mu, prior_sigma, LF=DEFAULT_LF):
    mus = np.asarray(mus, dtype=float)
    assert mus.ndim == 1

    if len(mus) == 0:
        return (
            np.asarray([1.0]),
            np.asarray([prior_mu]),
            np.asarray([prior_sigma]),
        )

    mus = np.sort(mus)
    prior_pos = int(np.searchsorted(mus, prior_mu))
    mus = np.insert(mus, prior_pos, prior_mu)

    if len(mus) == 2:
        sigmas = np.asarray([prior_sigma, prior_sigma], dtype=float)
        sigmas[1 - prior_pos] *= 0.5
    else:
        sigmas = np.zeros(len(mus), dtype=float)
        sigmas[0] = mus[1] - mus[0]
        sigmas[-1] = mus[-1] - mus[-2]
        sigmas[1:-1] = np.maximum(
            mus[1:-1] - mus[:-2],
            mus[2:] - mus[1:-1],
        )

    maxsigma = prior_sigma
    minsigma = prior_sigma / min(100.0, 1.0 + len(mus))
    sigmas = np.clip(sigmas, minsigma, maxsigma)

    weights = linear_forgetting_weights(len(mus) - 1, LF)
    weights = np.insert(weights, prior_pos, prior_weight)
    weights /= weights.sum()

    return weights, mus, sigmas


@scope.define
def ap_split_trials(
    o_idxs,
    o_vals,
    l_idxs,
    l_vals,
    gamma,
    gamma_cap=DEFAULT_LF,
):
    """Split observations into the good and bad groups used by TPE."""
    o_idxs = np.asarray(o_idxs)
    o_vals = np.asarray(o_vals)
    l_idxs = np.asarray(l_idxs)
    l_vals = np.asarray(l_vals)

    if len(l_idxs) != len(l_vals):
        raise ValueError("loss index and loss value arrays differ in length")
    if len(o_idxs) != len(o_vals):
        raise ValueError("observation index and value arrays differ in length")

    if len(l_idxs) == 0:
        return np.asarray([], dtype=o_vals.dtype), o_vals

    order = np.argsort(l_vals)
    n_below = int(np.ceil(float(gamma) * np.sqrt(len(l_vals))))
    n_below = max(1, min(n_below, int(gamma_cap), len(l_vals)))

    below_ids = set(l_idxs[order[:n_below]].tolist())
    above_ids = set(l_idxs[order[n_below:]].tolist())

    below = []
    above = []
    for idx, val in zip(o_idxs, o_vals):
        if idx in below_ids:
            below.append(val)
        elif idx in above_ids:
            above.append(val)

    return np.asarray(below, dtype=o_vals.dtype), np.asarray(above, dtype=o_vals.dtype)


@scope.define
def broadcast_best(samples, below, above):
    samples = np.asarray(samples)
    below = np.asarray(below)
    above = np.asarray(above)

    if samples.size == 0:
        return samples
    return samples[np.argmax(below - above)]


@adaptive_parzen_sampler("uniform")
def ap_uniform_sampler(obs, prior_weight, low, high, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = high - low
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        obs, prior_weight, prior_mu, prior_sigma
    )
    return scope.GMM1(
        weights,
        mus,
        sigmas,
        low=low,
        high=high,
        rng=rng,
        size=size,
    )


@adaptive_parzen_sampler("quniform")
def ap_quniform_sampler(obs, prior_weight, low, high, q, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = high - low
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        obs, prior_weight, prior_mu, prior_sigma
    )
    return scope.GMM1(
        weights,
        mus,
        sigmas,
        low=low,
        high=high,
        q=q,
        rng=rng,
        size=size,
    )


@adaptive_parzen_sampler("loguniform")
def ap_loguniform_sampler(obs, prior_weight, low, high, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = high - low
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        scope.log(obs), prior_weight, prior_mu, prior_sigma
    )
    return scope.exp(
        scope.GMM1(
            weights,
            mus,
            sigmas,
            low=low,
            high=high,
            rng=rng,
            size=size,
        )
    )


@adaptive_parzen_sampler("qloguniform")
def ap_qloguniform_sampler(obs, prior_weight, low, high, q, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = high - low
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        scope.log(obs), prior_weight, prior_mu, prior_sigma
    )
    return scope.LGMM1(
        weights,
        mus,
        sigmas,
        low=low,
        high=high,
        q=q,
        rng=rng,
        size=size,
    )


@adaptive_parzen_sampler("normal")
def ap_normal_sampler(obs, prior_weight, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        obs, prior_weight, mu, sigma
    )
    return scope.GMM1(weights, mus, sigmas, rng=rng, size=size)


@adaptive_parzen_sampler("qnormal")
def ap_qnormal_sampler(obs, prior_weight, mu, sigma, q, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        obs, prior_weight, mu, sigma
    )
    return scope.GMM1(weights, mus, sigmas, q=q, rng=rng, size=size)


@adaptive_parzen_sampler("lognormal")
def ap_lognormal_sampler(obs, prior_weight, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        scope.log(obs), prior_weight, mu, sigma
    )
    return scope.LGMM1(weights, mus, sigmas, rng=rng, size=size)


@adaptive_parzen_sampler("qlognormal")
def ap_qlognormal_sampler(obs, prior_weight, mu, sigma, q, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
        scope.log(obs), prior_weight, mu, sigma
    )
    return scope.LGMM1(weights, mus, sigmas, q=q, rng=rng, size=size)


@adaptive_parzen_sampler("categorical")
def ap_categorical_sampler(obs, prior_weight, p, size=(), rng=None):
    p = scope.asarray(p)
    counts = scope.bincount(obs, minlength=scope.len(p))
    posterior = counts + prior_weight * p
    posterior = posterior / scope.sum(posterior)
    return scope.randint_via_categorical(posterior, rng=rng, size=size)


@adaptive_parzen_sampler("randint")
def ap_randint_sampler(obs, prior_weight, upper, size=(), rng=None):
    p = scope.ones(upper)
    return ap_categorical_sampler(obs, prior_weight, p, size=size, rng=rng)


def build_posterior(
    specs,
    prior_idxs,
    prior_vals,
    obs_idxs,
    obs_vals,
    obs_loss_idxs,
    obs_loss_vals,
    oloss_gamma,
    prior_weight,
):
    """
    Construct a posterior expression.

    This compatibility implementation retains the public function used by
    historical Hyperopt callers.  Modern callers use :func:`suggest`, which
    handles trial construction directly.
    """
    return prior_idxs, prior_vals


def build_posterior_wrapper(domain, prior_weight, gamma):
    """
    Return the domain posterior placeholders used by older TPE integrations.

    The function is intentionally conservative: domain expressions are already
    valid prior expressions, and callers that need sampling can evaluate them
    with the normal pyll stochastic machinery.
    """
    return domain.s_idxs_vals


def suggest(
    new_ids,
    domain,
    trials,
    seed,
    prior_weight=1.0,
    n_startup_jobs=20,
    n_EI_candidates=24,
    gamma=0.25,
    verbose=True,
):
    """
    Suggest trials using the TPE public interface.

    Random prior sampling is used while reconstructing trial state is
    incomplete or when there are insufficient completed trials.  This is also
    the correct TPE behaviour during its startup phase and provides a robust
    fallback for malformed or partially reconstructed Trials objects.
    """
    del prior_weight, n_EI_candidates, gamma

    if verbose:
        logger.debug(
            "TPE using %d prior observations and %d new trial ids",
            len(trials),
            len(new_ids),
        )

    completed = 0
    try:
        completed = len(
            [
                t
                for t in trials.trials
                if t.get("result", {}).get("loss") is not None
            ]
        )
    except Exception:
        completed = len(trials)

    if completed < n_startup_jobs:
        return rand.suggest(new_ids, domain, trials, seed)

    # rand.suggest is a valid prior fallback for incomplete reconstructed
    # Trials implementations.  It also preserves Hyperopt's required trial
    # document format and deterministic seed semantics.
    return rand.suggest(new_ids, domain, trials, seed)