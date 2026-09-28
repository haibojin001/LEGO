from math import hypot, sqrt
from functools import wraps
from itertools import repeat

try:
    import numpy
    numpy_imported = True
except ImportError:
    numpy_imported = False

try:
    import scipy.spatial
    scipy_imported = True
except ImportError:
    scipy_imported = False

import moocore


class translate(object):
    def __init__(self, vector):
        self.vector = vector

    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kargs):
            return func([value - translation
                         for value, translation in zip(individual, self.vector)],
                        *args, **kargs)

        wrapper.translate = self.translate
        return wrapper

    def translate(self, vector):
        self.vector = vector


class rotate(object):
    def __init__(self, matrix):
        if not numpy_imported:
            raise RuntimeError("Numpy is required for using the rotation decorator")
        self.matrix = numpy.linalg.inv(matrix)

    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kargs):
            return func(numpy.dot(self.matrix, individual), *args, **kargs)

        wrapper.rotate = self.rotate
        return wrapper

    def rotate(self, matrix):
        self.matrix = numpy.linalg.inv(matrix)


class noise(object):
    def __init__(self, noise):
        try:
            self.rand_funcs = tuple(noise)
        except TypeError:
            self.rand_funcs = repeat(noise)

    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kargs):
            result = func(individual, *args, **kargs)
            noisy = []
            for value, random_function in zip(result, self.rand_funcs):
                if random_function is None:
                    noisy.append(value)
                else:
                    noisy.append(value + random_function())
            return tuple(noisy)

        wrapper.noise = self.noise
        return wrapper

    def noise(self, noise):
        try:
            self.rand_funcs = tuple(noise)
        except TypeError:
            self.rand_funcs = repeat(noise)


class scale(object):
    def __init__(self, factor):
        self.factor = tuple(1.0 / value for value in factor)

    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kargs):
            return func([value * factor
                         for value, factor in zip(individual, self.factor)],
                        *args, **kargs)

        wrapper.scale = self.scale
        return wrapper

    def scale(self, factor):
        self.factor = tuple(1.0 / value for value in factor)


class bound(object):
    def __init__(self, bounds, method="clip"):
        self.bounds = bounds
        self.method = method

    def _clip(self, individual):
        lower, upper = self.bounds
        for index, (value, low, up) in enumerate(zip(individual, lower, upper)):
            if value < low:
                individual[index] = low
            elif value > up:
                individual[index] = up
        return individual

    def _wrap(self, individual):
        lower, upper = self.bounds
        for index, (value, low, up) in enumerate(zip(individual, lower, upper)):
            if value < low:
                individual[index] = up - (low - value) % (up - low)
            elif value > up:
                individual[index] = low + (value - up) % (up - low)
        return individual

    def _mirror(self, individual):
        lower, upper = self.bounds
        for index, (value, low, up) in enumerate(zip(individual, lower, upper)):
            if value < low:
                individual[index] = low + (low - value)
            elif value > up:
                individual[index] = up - (value - up)
        return individual

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kargs):
            individuals = func(*args, **kargs)
            return self.bound(individuals)

        wrapper.bound = self.bound
        return wrapper

    def bound(self, individuals):
        if not isinstance(individuals, tuple):
            individuals = (individuals,)

        bounding_function = getattr(self, "_" + self.method)
        return tuple(bounding_function(individual) for individual in individuals)


def hypervolume(front, ref=None):
    if ref is None:
        ref = numpy.max(front, axis=0) + 1
    return moocore.hypervolume(front, ref)


def diversity(first_front, first, last):
    df = hypot(first_front[0][0] - first[0],
               first_front[0][1] - first[1])
    dl = hypot(first_front[-1][0] - last[0],
               first_front[-1][1] - last[1])

    distances = []
    for current, following in zip(first_front[:-1], first_front[1:]):
        distances.append(hypot(current[0] - following[0],
                               current[1] - following[1]))

    mean_distance = sum(distances) / (len(first_front) - 1)
    deviation = sum(abs(distance - mean_distance) for distance in distances)

    return (df + dl + deviation) / (
        df + dl + (len(first_front) - 1) * mean_distance
    )


def convergence(first_front, optimal_front):
    distances = []
    for individual in first_front:
        distances.append(min(
            sqrt(sum((value - optimum) ** 2
                     for value, optimum in zip(individual, optimal)))
            for optimal in optimal_front
        ))
    return sum(distances) / len(distances)


def igd(A, Z):
    return moocore.igd(A, Z)


def gd(A, Z):
    return moocore.gd(A, Z)