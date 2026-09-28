import bisect
from collections import Counter
from math import ceil, floor


class _StatsProperty:
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.internal_name = '_' + name
        self.__doc__ = (func.__doc__ or '').partition('>>>')[0]

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not obj.data:
            return obj.default
        try:
            return getattr(obj, self.internal_name)
        except AttributeError:
            value = self.func(obj)
            setattr(obj, self.internal_name, value)
            return value


def _stats_property(func):
    return _StatsProperty(func.__name__, func)


class Stats:
    """Container providing lazily computed descriptive statistics."""

    def __init__(self, data, default=0.0, use_copy=True, is_sorted=False):
        self._use_copy = use_copy
        self._is_sorted = is_sorted
        self.data = list(data) if use_copy else data
        self.default = default

        cls = self.__class__
        self._prop_attr_names = [
            name for name in dir(self)
            if isinstance(getattr(cls, name, None), _StatsProperty)
        ]
        self._pearson_precision = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def _get_sorted_data(self):
        if not self._use_copy:
            return sorted(self.data)
        if not self._is_sorted:
            self.data.sort()
            self._is_sorted = True
        return self.data

    def clear_cache(self):
        """Clear cached statistical calculations."""
        for name in self._prop_attr_names:
            internal_name = getattr(self.__class__, name).internal_name
            try:
                delattr(self, internal_name)
            except AttributeError:
                pass

    @_stats_property
    def count(self):
        """The number of data points."""
        return len(self.data)

    @_stats_property
    def mean(self):
        """The arithmetic mean of the data."""
        return sum(self.data) / self.count

    @_stats_property
    def max(self):
        """The largest data point."""
        return max(self.data)

    @_stats_property
    def min(self):
        """The smallest data point."""
        return min(self.data)

    @_stats_property
    def mode(self):
        """The most frequently occurring data point."""
        return Counter(self.data).most_common(1)[0][0]

    @_stats_property
    def median(self):
        """The midpoint of the sorted data."""
        sorted_data = self._get_sorted_data()
        midpoint = self.count // 2
        if self.count % 2:
            return sorted_data[midpoint]
        return (sorted_data[midpoint - 1] + sorted_data[midpoint]) / 2.0

    def get_quantile(self, quantile):
        """Get a linearly interpolated quantile from 0.0 through 1.0."""
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError('quantile must be between 0.0 and 1.0')
        if not self.data:
            return self.default

        sorted_data = self._get_sorted_data()
        index = quantile * (self.count - 1)
        lower = int(floor(index))
        upper = int(ceil(index))
        if lower == upper:
            return sorted_data[lower]

        lower_value = sorted_data[lower]
        upper_value = sorted_data[upper]
        return lower_value + ((upper_value - lower_value) * (index - lower))

    @_stats_property
    def iqr(self):
        """The interquartile range."""
        return self.get_quantile(0.75) - self.get_quantile(0.25)

    @_stats_property
    def trimean(self):
        """The trimean, a robust measure of central tendency."""
        return (self.get_quantile(0.25)
                + (2.0 * self.median)
                + self.get_quantile(0.75)) / 4.0

    @_stats_property
    def variance(self):
        """The population variance."""
        mean = self.mean
        return sum((value - mean) ** 2 for value in self.data) / self.count

    @_stats_property
    def std_dev(self):
        """The population standard deviation."""
        return self.variance ** 0.5

    @_stats_property
    def rel_std_dev(self):
        """The relative standard deviation."""
        mean = self.mean
        if mean == 0:
            return self.default
        return self.std_dev / abs(mean)

    @_stats_property
    def skewness(self):
        """The third standardized moment of the data."""
        std_dev = self.std_dev
        if std_dev == 0:
            return self.default
        mean = self.mean
        return (sum((value - mean) ** 3 for value in self.data)
                / self.count / (std_dev ** 3))

    @_stats_property
    def kurtosis(self):
        """The fourth standardized moment of the data."""
        std_dev = self.std_dev
        if std_dev == 0:
            return self.default
        mean = self.mean
        return (sum((value - mean) ** 4 for value in self.data)
                / self.count / (std_dev ** 4))

    @_stats_property
    def pearson_type(self):
        """The Pearson distribution type inferred from skewness and kurtosis."""
        skewness = self.skewness
        kurtosis = self.kurtosis
        precision = self._pearson_precision

        if round(skewness, precision) == 0:
            if round(kurtosis - 3, precision) == 0:
                return 0
            if kurtosis < 3:
                return 2
            return 7

        beta_one = skewness ** 2
        denominator = (
            4.0
            * ((4.0 * kurtosis) - (3.0 * beta_one))
            * ((2.0 * kurtosis) - (3.0 * beta_one) - 6.0)
        )
        numerator = beta_one * ((kurtosis + 3.0) ** 2)

        if round(denominator, precision) == 0:
            return 3

        kappa = numerator / denominator
        rounded_kappa = round(kappa, precision)

        if rounded_kappa < 0:
            return 1
        if rounded_kappa == 0:
            return 2
        if rounded_kappa < 1:
            return 4
        if rounded_kappa == 1:
            return 5
        return 6

    @_stats_property
    def mad(self):
        """The median absolute deviation from the median."""
        median = self.median
        deviations = [abs(value - median) for value in self.data]
        return self.__class__(deviations, default=self.default).median

    def get_zscore(self, value):
        """Return the z-score for *value* relative to this dataset."""
        std_dev = self.std_dev
        if std_dev == 0:
            return self.default
        return (value - self.mean) / std_dev

    def trim_relative(self, amount=0.15):
        """Return a Stats object with equal fractions trimmed from both ends."""
        amount = float(amount)
        if amount < 0.0 or amount >= 0.5:
            raise ValueError('amount must be in the range [0.0, 0.5)')

        sorted_data = self._get_sorted_data()
        trim_count = int(self.count * amount)
        end = self.count - trim_count
        return self.__class__(
            sorted_data[trim_count:end],
            default=self.default,
            use_copy=False,
            is_sorted=True,
        )

    def get_histogram(self, bins=None, **kw):
        """Return a list of ``(bin_start, count)`` pairs for the data."""
        if not self.data:
            return []

        if bins is None:
            bins = int(ceil(self.count ** 0.5))

        if isinstance(bins, int):
            if bins < 1:
                raise ValueError('bins must be a positive integer')
            minimum = self.min
            maximum = self.max
            if minimum == maximum:
                bounds = [minimum]
            else:
                width = (maximum - minimum) / float(bins)
                bounds = [minimum + (width * index) for index in range(bins)]
        else:
            bounds = sorted(set(bins))
            if not bounds:
                return []

        counts = [0] * len(bounds)
        for value in self.data:
            index = bisect.bisect_right(bounds, value) - 1
            if index < 0:
                index = 0
            elif index >= len(bounds):
                index = len(bounds) - 1
            counts[index] += 1

        return list(zip(bounds, counts))


def _make_stat_helper(name):
    def helper(data, default=0.0):
        return getattr(Stats(data, default=default), name)

    helper.__name__ = name
    helper.__doc__ = getattr(Stats, name).__doc__
    return helper


for _name in [
    'count',
    'mean',
    'max',
    'min',
    'mode',
    'median',
    'iqr',
    'trimean',
    'variance',
    'std_dev',
    'rel_std_dev',
    'skewness',
    'kurtosis',
    'pearson_type',
    'mad',
]:
    globals()[_name] = _make_stat_helper(_name)

del _name