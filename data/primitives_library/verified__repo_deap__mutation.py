import math
import random
from itertools import repeat

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


def mutGaussian(individual, mu, sigma, indpb):
    """Apply independent Gaussian perturbations to an individual."""
    size = len(individual)

    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError(
            "mu must be at least the size of individual: %d < %d"
            % (len(mu), size)
        )

    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError(
            "sigma must be at least the size of individual: %d < %d"
            % (len(sigma), size)
        )

    for index, mean, deviation in zip(range(size), mu, sigma):
        if random.random() < indpb:
            individual[index] += random.gauss(mean, deviation)

    return individual,


def mutPolynomialBounded(individual, eta, low, up, indpb):
    """Apply a bounded polynomial mutation."""
    size = len(individual)

    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError(
            "low must be at least the size of individual: %d < %d"
            % (len(low), size)
        )

    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError(
            "up must be at least the size of individual: %d < %d"
            % (len(up), size)
        )

    for index, lower, upper in zip(range(size), low, up):
        if random.random() <= indpb:
            value = individual[index]
            delta_lower = (value - lower) / (upper - lower)
            delta_upper = (upper - value) / (upper - lower)
            draw = random.random()
            exponent = 1.0 / (eta + 1.0)

            if draw < 0.5:
                distance = 1.0 - delta_lower
                factor = (
                    2.0 * draw
                    + (1.0 - 2.0 * draw) * distance ** (eta + 1)
                )
                delta = factor ** exponent - 1.0
            else:
                distance = 1.0 - delta_upper
                factor = (
                    2.0 * (1.0 - draw)
                    + 2.0 * (draw - 0.5) * distance ** (eta + 1)
                )
                delta = 1.0 - factor ** exponent

            value += delta * (upper - lower)
            individual[index] = min(max(value, lower), upper)

    return individual,


def mutShuffleIndexes(individual, indpb):
    """Randomly exchange individual entries independently."""
    size = len(individual)

    for index in range(size):
        if random.random() < indpb:
            other = random.randint(0, size - 2)
            if other >= index:
                other += 1
            individual[index], individual[other] = (
                individual[other],
                individual[index],
            )

    return individual,


def mutFlipBit(individual, indpb):
    """Independently negate entries of an individual."""
    for index in range(len(individual)):
        if random.random() < indpb:
            individual[index] = type(individual[index])(not individual[index])

    return individual,


def mutUniformInt(individual, low, up, indpb):
    """Replace entries with uniformly sampled inclusive-range integers."""
    size = len(individual)

    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError(
            "low must be at least the size of individual: %d < %d"
            % (len(low), size)
        )

    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError(
            "up must be at least the size of individual: %d < %d"
            % (len(up), size)
        )

    for index, lower, upper in zip(range(size), low, up):
        if random.random() < indpb:
            individual[index] = random.randint(lower, upper)

    return individual,


def mutInversion(individual):
    """Reverse the section lying between two randomly selected indices."""
    size = len(individual)

    if size == 0:
        return individual,

    first = random.randrange(size)
    second = random.randrange(size)
    begin = min(first, second)
    end = max(first, second)

    individual[begin:end] = individual[begin:end][::-1]

    return individual,


def mutESLogNormal(individual, c, indpb):
    """Mutate an evolution-strategy individual and its strategy values."""
    size = len(individual)
    tau = c / math.sqrt(2.0 * math.sqrt(size))
    tau0 = c / math.sqrt(2.0 * size)
    common = tau0 * random.gauss(0, 1)

    for index in range(size):
        if random.random() < indpb:
            individual.strategy[index] *= math.exp(
                common + tau * random.gauss(0, 1)
            )
            individual[index] += random.gauss(0, individual.strategy[index])

    return individual,