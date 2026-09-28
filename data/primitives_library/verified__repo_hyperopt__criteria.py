import numpy as np
import scipy.stats


def EI_empirical(samples, thresh):
    """Compute sample-based expected improvement above a threshold."""
    values = np.maximum(samples - thresh, 0)
    return values.mean()


def EI_gaussian_empirical(mean, var, thresh, rng, N):
    """Estimate Gaussian expected improvement through random sampling."""
    draws = rng.standard_normal(N) * np.sqrt(var) + mean
    return EI_empirical(draws, thresh)


def EI_gaussian(mean, var, thresh):
    """Compute the analytic expected improvement for a Gaussian variable."""
    stddev = np.sqrt(var)
    standardized = (mean - thresh) / stddev
    normal = scipy.stats.norm
    return stddev * (
        standardized * normal.cdf(standardized) + normal.pdf(standardized)
    )


def logEI_gaussian(mean, var, thresh):
    """Compute logarithmic Gaussian expected improvement stably."""
    assert np.asarray(var).min() >= 0

    stddev = np.sqrt(var)
    standardized = (mean - thresh) / stddev
    normal = scipy.stats.norm

    try:
        float(mean)
        scalar_input = True
    except TypeError:
        scalar_input = False

    if scalar_input:
        if standardized < 0:
            log_density = normal.logpdf(standardized)
            correction = np.exp(
                np.log(-standardized)
                + normal.logcdf(standardized)
                - log_density
            )
            result = np.log(stddev) + log_density + np.log1p(-correction)
            if not np.isfinite(result):
                return -np.inf
            return result

        return np.log(stddev) + np.log(
            standardized * normal.cdf(standardized) + normal.pdf(standardized)
        )

    standardized = np.asarray(standardized)
    result = np.zeros_like(standardized)

    previous_settings = np.seterr(all="ignore")
    try:
        negative = standardized < 0
        nonnegative = np.logical_not(negative)

        negative_scores = standardized[negative]
        negative_log_density = normal.logpdf(negative_scores)
        correction = np.exp(
            np.log(-negative_scores)
            + normal.logcdf(negative_scores)
            - negative_log_density
        )
        result[negative] = (
            np.log(stddev[negative])
            + negative_log_density
            + np.log1p(-correction)
        )

        nonnegative_scores = standardized[nonnegative]
        result[nonnegative] = np.log(stddev[nonnegative]) + np.log(
            nonnegative_scores * normal.cdf(nonnegative_scores)
            + normal.pdf(nonnegative_scores)
        )

        result[np.logical_not(np.isfinite(result))] = -np.inf
    finally:
        np.seterr(**previous_settings)

    return result


def UCB(mean, var, zscore):
    """Compute the upper confidence bound of a Gaussian prediction."""
    return mean + np.sqrt(var) * zscore