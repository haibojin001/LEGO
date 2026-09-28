import random
import warnings

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from itertools import repeat


def cxOnePoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    point = random.randint(1, size - 1)
    ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
    return ind1, ind2


def cxTwoPoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    first = random.randint(1, size)
    second = random.randint(1, size - 1)

    if second >= first:
        second += 1
    else:
        first, second = second, first

    ind1[first:second], ind2[first:second] = (
        ind2[first:second],
        ind1[first:second],
    )
    return ind1, ind2


def cxTwoPoints(ind1, ind2):
    warnings.warn(
        "tools.cxTwoPoints has been renamed. Use cxTwoPoint instead.",
        FutureWarning,
    )
    return cxTwoPoint(ind1, ind2)


def cxUniform(ind1, ind2, indpb):
    size = min(len(ind1), len(ind2))
    for index in range(size):
        if random.random() < indpb:
            ind1[index], ind2[index] = ind2[index], ind1[index]
    return ind1, ind2


def cxPartialyMatched(ind1, ind2):
    size = min(len(ind1), len(ind2))
    positions1 = [0] * size
    positions2 = [0] * size

    for index in range(size):
        positions1[ind1[index]] = index
        positions2[ind2[index]] = index

    first = random.randint(0, size)
    second = random.randint(0, size - 1)

    if second >= first:
        second += 1
    else:
        first, second = second, first

    for index in range(first, second):
        value1 = ind1[index]
        value2 = ind2[index]

        ind1[index], ind1[positions1[value2]] = value2, value1
        ind2[index], ind2[positions2[value1]] = value1, value2

        positions1[value1], positions1[value2] = (
            positions1[value2],
            positions1[value1],
        )
        positions2[value1], positions2[value2] = (
            positions2[value2],
            positions2[value1],
        )

    return ind1, ind2


def cxUniformPartialyMatched(ind1, ind2, indpb):
    size = min(len(ind1), len(ind2))
    positions1 = [0] * size
    positions2 = [0] * size

    for index in range(size):
        positions1[ind1[index]] = index
        positions2[ind2[index]] = index

    for index in range(size):
        if random.random() < indpb:
            value1 = ind1[index]
            value2 = ind2[index]

            ind1[index], ind1[positions1[value2]] = value2, value1
            ind2[index], ind2[positions2[value1]] = value1, value2

            positions1[value1], positions1[value2] = (
                positions1[value2],
                positions1[value1],
            )
            positions2[value1], positions2[value2] = (
                positions2[value2],
                positions2[value1],
            )

    return ind1, ind2


def cxOrdered(ind1, ind2):
    size = min(len(ind1), len(ind2))
    first, second = random.sample(range(size), 2)

    if first > second:
        first, second = second, first

    holes1 = [True] * size
    holes2 = [True] * size

    for index in range(size):
        if index < first or index > second:
            holes1[ind2[index]] = False
            holes2[ind1[index]] = False

    source1, source2 = ind1, ind2
    write1 = second + 1
    write2 = second + 1

    for index in range(size):
        value1 = source1[(index + second + 1) % size]
        value2 = source2[(index + second + 1) % size]

        if not holes1[value1]:
            ind1[write1 % size] = value1
            write1 += 1

        if not holes2[value2]:
            ind2[write2 % size] = value2
            write2 += 1

    for index in range(first, second + 1):
        ind1[index], ind2[index] = ind2[index], ind1[index]

    return ind1, ind2


def cxBlend(ind1, ind2, alpha):
    for index, (value1, value2) in enumerate(zip(ind1, ind2)):
        gamma = (1.0 + 2.0 * alpha) * random.random() - alpha
        ind1[index] = (1.0 - gamma) * value1 + gamma * value2
        ind2[index] = gamma * value1 + (1.0 - gamma) * value2
    return ind1, ind2


def cxESBlend(ind1, ind2, alpha):
    for index, (value1, value2, strategy1, strategy2) in enumerate(
        zip(ind1, ind2, ind1.strategy, ind2.strategy)
    ):
        gamma = (1.0 + 2.0 * alpha) * random.random() - alpha
        ind1[index] = (1.0 - gamma) * value1 + gamma * value2
        ind2[index] = gamma * value1 + (1.0 - gamma) * value2

        gamma = (1.0 + 2.0 * alpha) * random.random() - alpha
        ind1.strategy[index] = (1.0 - gamma) * strategy1 + gamma * strategy2
        ind2.strategy[index] = gamma * strategy1 + (1.0 - gamma) * strategy2

    return ind1, ind2


def cxESTwoPoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    first = random.randint(1, size)
    second = random.randint(1, size - 1)

    if second >= first:
        second += 1
    else:
        first, second = second, first

    ind1[first:second], ind2[first:second] = (
        ind2[first:second],
        ind1[first:second],
    )
    ind1.strategy[first:second], ind2.strategy[first:second] = (
        ind2.strategy[first:second],
        ind1.strategy[first:second],
    )

    return ind1, ind2


def cxESTwoPoints(ind1, ind2):
    warnings.warn(
        "tools.cxESTwoPoints has been renamed. Use cxESTwoPoint instead.",
        FutureWarning,
    )
    return cxESTwoPoint(ind1, ind2)


def cxSimulatedBinary(ind1, ind2, eta):
    for index, (value1, value2) in enumerate(zip(ind1, ind2)):
        rand = random.random()
        if rand <= 0.5:
            beta = (2.0 * rand) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta + 1.0))

        ind1[index] = 0.5 * (
            (1.0 + beta) * value1 + (1.0 - beta) * value2
        )
        ind2[index] = 0.5 * (
            (1.0 - beta) * value1 + (1.0 + beta) * value2
        )

    return ind1, ind2


def cxSimulatedBinaryBounded(ind1, ind2, eta, low, up):
    size = min(len(ind1), len(ind2))

    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError(
            "low must be at least the size of the shorter individual: %d < %d"
            % (len(low), size)
        )

    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError(
            "up must be at least the size of the shorter individual: %d < %d"
            % (len(up), size)
        )

    for index, (value1, value2, lower, upper) in enumerate(
        zip(ind1, ind2, low, up)
    ):
        if random.random() <= 0.5:
            if abs(value1 - value2) > 1e-14:
                if value1 < value2:
                    smaller, larger = value1, value2
                else:
                    smaller, larger = value2, value1

                rand = random.random()

                beta = 1.0 + 2.0 * (smaller - lower) / (larger - smaller)
                alpha = 2.0 - beta ** -(eta + 1.0)

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (
                        1.0 / (eta + 1.0)
                    )

                child1 = 0.5 * (
                    smaller + larger - beta_q * (larger - smaller)
                )

                beta = 1.0 + 2.0 * (upper - larger) / (larger - smaller)
                alpha = 2.0 - beta ** -(eta + 1.0)

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (
                        1.0 / (eta + 1.0)
                    )

                child2 = 0.5 * (
                    smaller + larger + beta_q * (larger - smaller)
                )

                child1 = min(max(child1, lower), upper)
                child2 = min(max(child2, lower), upper)

                if random.random() <= 0.5:
                    ind1[index], ind2[index] = child2, child1
                else:
                    ind1[index], ind2[index] = child1, child2

    return ind1, ind2


def cxMessyOnePoint(ind1, ind2):
    point1 = random.randint(0, len(ind1))
    point2 = random.randint(0, len(ind2))

    ind1[point1:], ind2[point2:] = ind2[point2:], ind1[point1:]

    return ind1, ind2