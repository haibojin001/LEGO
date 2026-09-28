"""
Additional probability distributions used by hyperopt.
"""

import numpy as np
import numpy.random as mtrand
import scipy.stats
from scipy.stats import rv_continuous
from scipy.stats._continuous_distns import lognorm_gen as scipy_lognorm_gen


class loguniform_gen(rv_continuous):
    """Distribution of exp(U), where U is uniform on [low, high]."""

    def __init__(self, low=0, high=1):
        self._low = low
        self._high = high
        super().__init__(a=np.exp(low), b=np.exp(high))

    def _rvs(self, *args, size=None, random_state=None):
        return np.exp(mtrand.uniform(self._low, self._high, size))

    def _pdf(self, x):
        return 1.0 / (x * (self._high - self._low))

    def _logpdf(self, x):
        return -np.log(x) - np.log(self._high - self._low)

    def _cdf(self, x):
        return (np.log(x) - self._low) / (self._high - self._low)


class lognorm_gen(scipy_lognorm_gen):
    def __init__(self, mu, sigma):
        self.mu_ = mu
        self.s_ = sigma
        super().__init__()
        del self.__dict__["_parse_args"]
        del self.__dict__["_parse_args_stats"]
        del self.__dict__["_parse_args_rvs"]

    def _parse_args(self, *args, **kwargs):
        assert not args, args
        assert not kwargs, kwargs
        return (self.s_,), 0, np.exp(self.mu_)


def qtable_pmf(x, q, qlow, xs, ps):
    values = np.atleast_1d(x).astype(float)
    qx = np.round(values / q) * q
    exact = np.isclose(qx, x)
    indices = np.round((qx - qlow) / q).astype(int)
    valid_index = np.logical_and(indices >= 0, indices < len(ps))
    valid = np.logical_and(exact, valid_index)

    result = np.zeros_like(qx)
    probabilities = np.asarray(ps)
    result[valid] = probabilities[indices[valid]]

    if isinstance(x, np.ndarray):
        return result.reshape(x.shape)
    return float(result[0])


def qtable_logpmf(x, q, qlow, xs, ps):
    probabilities = qtable_pmf(np.atleast_1d(x), q, qlow, xs, ps)
    result = np.zeros_like(probabilities)
    zero = probabilities == 0
    result[zero] = -np.inf
    result[~zero] = np.log(probabilities[~zero])

    if isinstance(x, np.ndarray):
        return result
    return float(result[0])


class quniform_gen:
    """Distribution formed by quantizing a uniform random variable."""

    def __init__(self, low, high, q):
        low, high = map(float, (low, high))
        qlow = safe_int_cast(np.round(low / q)) * q
        qhigh = safe_int_cast(np.round(high / q)) * q

        if qlow == qhigh:
            xs = [qlow]
            ps = [1.0]
        else:
            lowmass = 1 - ((low - qlow + q * 0.5) / q)
            highmass = (high - qhigh + q * 0.5) / q
            assert 0 <= lowmass <= 1.0, (lowmass, low, qlow, q)
            assert 0 <= highmass <= 1.0, (highmass, high, qhigh, q)

            xs = np.arange(qlow, qhigh + q * 0.5, q)
            ps = np.ones(len(xs))
            ps[0] = lowmass
            ps[-1] = highmass
            ps /= ps.sum()

        self.low = low
        self.high = high
        self.q = q
        self.qlow = qlow
        self.qhigh = qhigh
        self.xs = np.asarray(xs)
        self.ps = np.asarray(ps)

    def pmf(self, x):
        return qtable_pmf(x, self.q, self.qlow, self.xs, self.ps)

    def logpmf(self, x):
        return qtable_logpmf(x, self.q, self.qlow, self.xs, self.ps)

    def rvs(self, size=()):
        sample = mtrand.uniform(low=self.low, high=self.high, size=size)
        return safe_int_cast(np.round(sample / self.q)) * self.q


class qloguniform_gen(quniform_gen):
    """Distribution formed by quantizing exp(U), with U uniform."""

    def __init__(self, low, high, q):
        low, high = map(float, (low, high))
        elow = np.exp(low)
        ehigh = np.exp(high)
        qlow = safe_int_cast(np.round(elow / q)) * q
        qhigh = safe_int_cast(np.round(ehigh / q)) * q

        distribution = loguniform_gen(low=low, high=high)
        cut_low = elow
        cut_high = min(qlow + q * 0.5, ehigh)

        xs = [qlow]
        ps = [distribution.cdf(cut_high)]
        cdf_high = ps[0]
        ii = 0

        while cut_high < ehigh - 1e-10:
            cut_high, cut_low = min(cut_high + q, ehigh), cut_high
            cdf_high, cdf_low = distribution.cdf(cut_high), cdf_high
            ii += 1
            xs.append(qlow + ii * q)
            ps.append(cdf_high - cdf_low)

        ps = np.asarray(ps)
        ps /= ps.sum()

        self.low = low
        self.high = high
        self.q = q
        self.qlow = qlow
        self.qhigh = qhigh
        self.xs = np.asarray(xs)
        self.ps = ps

    def pmf(self, x):
        return qtable_pmf(x, self.q, self.qlow, self.xs, self.ps)

    def logpmf(self, x):
        return qtable_logpmf(x, self.q, self.qlow, self.xs, self.ps)

    def rvs(self, size=()):
        sample = mtrand.uniform(low=self.low, high=self.high, size=size)
        return safe_int_cast(np.round(np.exp(sample) / self.q)) * self.q


class qnormal_gen:
    """Distribution formed by quantizing a normal random variable."""

    def __init__(self, mu, sigma, q):
        self.mu, self.sigma = map(float, (mu, sigma))
        self.q = q
        self._norm_logcdf = scipy.stats.norm(loc=mu, scale=sigma).logcdf

    def in_domain(self, x):
        return np.isclose(x, safe_int_cast(np.round(x / self.q)) * self.q)

    def pmf(self, x):
        return np.exp(self.logpmf(x))

    def logpmf(self, x):
        values = np.atleast_1d(x)
        valid = self.in_domain(values)
        result = np.zeros_like(values, dtype=float) - np.inf
        quantized_values = values[valid]

        upper = quantized_values + self.q * 0.5
        lower = quantized_values - self.q * 0.5

        reflected = lower > self.mu
        old_lower = lower[reflected].copy()
        lower[reflected] = self.mu - (upper[reflected] - self.mu)
        upper[reflected] = self.mu - (old_lower - self.mu)

        assert np.all(upper > lower)
        log_upper = self._norm_logcdf(upper)
        log_lower = self._norm_logcdf(lower)
        result[valid] = log_upper + np.log1p(-np.exp(log_lower - log_upper))

        if isinstance(x, np.ndarray):
            return result
        return float(result[0])

    def rvs(self, size=()):
        sample = mtrand.normal(loc=self.mu, scale=self.sigma, size=size)
        return safe_int_cast(np.round(sample / self.q)) * self.q


class qlognormal_gen:
    """Distribution formed by quantizing exp(N), with N normal."""

    def __init__(self, mu, sigma, q):
        self.mu, self.sigma = map(float, (mu, sigma))
        self.q = q
        self._norm_cdf = scipy.stats.norm(loc=mu, scale=sigma).cdf

    def in_domain(self, x):
        return np.logical_and(
            x >= 0,
            np.isclose(x, safe_int_cast(np.round(x / self.q)) * self.q),
        )

    def pmf(self, x):
        values = np.atleast_1d(x)
        valid = self.in_domain(values)
        valid_values = values[valid]
        result = np.zeros_like(values, dtype=float)

        valid_probabilities = self._norm_cdf(np.log(valid_values + self.q * 0.5))
        nonzero = valid_values != 0
        valid_probabilities[nonzero] -= self._norm_cdf(
            np.log(valid_values[nonzero] - self.q * 0.5)
        )
        result[valid] = valid_probabilities

        if isinstance(x, np.ndarray):
            return result
        return float(result[0])

    def logpmf(self, x):
        probabilities = self.pmf(np.atleast_1d(x))
        assert np.all(probabilities >= 0)
        probabilities[probabilities == 0] = -np.inf
        positive = probabilities > 0
        probabilities[positive] = np.log(probabilities[positive])

        if isinstance(x, np.ndarray):
            return probabilities
        return float(probabilities)

    def rvs(self, size=()):
        sample = mtrand.normal(loc=self.mu, scale=self.sigma, size=size)
        return safe_int_cast(np.round(np.exp(sample) / self.q)) * self.q


def safe_int_cast(obj):
    if isinstance(obj, np.ndarray):
        return obj.astype("int")
    if isinstance(obj, list):
        return [int(value) for value in obj]
    return int(obj)