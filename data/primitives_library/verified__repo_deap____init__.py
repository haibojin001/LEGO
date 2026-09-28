import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce


def rand(individual):
    return random.random(),


def plane(individual):
    return individual[0],


def sphere(individual):
    return sum(gene * gene for gene in individual),


def cigar(individual):
    return individual[0] ** 2 + 1e6 * sum(gene * gene for gene in individual[1:]),


def rosenbrock(individual):
    return sum(
        100 * (x * x - y) ** 2 + (1.0 - x) ** 2
        for x, y in zip(individual[:-1], individual[1:])
    ),


def h1(individual):
    numerator = (
        sin(individual[0] - individual[1] / 8) ** 2
        + sin(individual[1] + individual[0] / 8) ** 2
    )
    denominator = (
        (individual[0] - 8.6998) ** 2
        + (individual[1] - 6.7665) ** 2
    ) ** 0.5 + 1
    return numerator / denominator,


def ackley(individual):
    size = len(individual)
    return (
        20
        - 20 * exp(-0.2 * sqrt(sum(x ** 2 for x in individual) / size))
        + e
        - exp(sum(cos(2 * pi * x) for x in individual) / size),
    )


def bohachevsky(individual):
    return sum(
        x ** 2
        + 2 * y ** 2
        - 0.3 * cos(3 * pi * x)
        - 0.4 * cos(4 * pi * y)
        + 0.7
        for x, y in zip(individual[:-1], individual[1:])
    ),


def griewank(individual):
    product = reduce(
        mul,
        (cos(x / sqrt(index + 1.0)) for index, x in enumerate(individual)),
        1,
    )
    return sum(x ** 2 for x in individual) / 4000.0 - product + 1,


def rastrigin(individual):
    return (
        10 * len(individual)
        + sum(x * x - 10 * cos(2 * pi * x) for x in individual),
    )


def rastrigin_scaled(individual):
    size = len(individual)
    return (
        10 * size
        + sum(
            (10 ** (index / (size - 1)) * x) ** 2
            - 10 * cos(2 * pi * 10 ** (index / (size - 1)) * x)
            for index, x in enumerate(individual)
        ),
    )


def rastrigin_skew(individual):
    size = len(individual)
    return (
        10 * size
        + sum(
            (10 * x if x > 0 else x) ** 2
            - 10 * cos(2 * pi * (10 * x if x > 0 else x))
            for x in individual
        ),
    )


def schaffer(individual):
    return sum(
        (x ** 2 + y ** 2) ** 0.25
        * (sin(50 * (x ** 2 + y ** 2) ** 0.1) ** 2 + 1.0)
        for x, y in zip(individual[:-1], individual[1:])
    ),


def schwefel(individual):
    return (
        418.9828872724339 * len(individual)
        - sum(x * sin(sqrt(abs(x))) for x in individual),
    )


def kursawe(individual):
    first = sum(
        -10 * exp(-0.2 * sqrt(x ** 2 + y ** 2))
        for x, y in zip(individual[:-1], individual[1:])
    )
    second = sum(abs(x) ** 0.8 + 5 * sin(x ** 3) for x in individual)
    return first, second


def fonseca(individual):
    size = len(individual)
    shift = 1.0 / sqrt(size)
    first = 1 - exp(-sum((x - shift) ** 2 for x in individual))
    second = 1 - exp(-sum((x + shift) ** 2 for x in individual))
    return first, second


def poloni(individual):
    x1, x2 = individual[0], individual[1]

    a1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2)
    a2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2)
    b1 = 0.5 * sin(x1) - 2 * cos(x1) + sin(x2) - 1.5 * cos(x2)
    b2 = 1.5 * sin(x1) - cos(x1) + 2 * sin(x2) - 0.5 * cos(x2)

    return (
        1 + (a1 - b1) ** 2 + (a2 - b2) ** 2,
        (x1 + 3) ** 2 + (x2 + 1) ** 2,
    )


def dent(individual, lambda_=0.85):
    x1, x2 = individual[0], individual[1]
    base = 0.5 * (
        sqrt(1 + (x1 + x2) ** 2) + sqrt(1 + (x1 - x2) ** 2)
    )
    indentation = lambda_ * exp(-(x1 - x2) ** 2)
    return base + x1 - x2 + indentation, base - x1 + x2 + indentation


def zdt1(individual):
    g = 1 + 9 * sum(individual[1:]) / (len(individual) - 1)
    f1 = individual[0]
    return f1, g * (1 - sqrt(f1 / g))


def zdt2(individual):
    g = 1 + 9 * sum(individual[1:]) / (len(individual) - 1)
    f1 = individual[0]
    return f1, g * (1 - (f1 / g) ** 2)


def zdt3(individual):
    g = 1 + 9 * sum(individual[1:]) / (len(individual) - 1)
    f1 = individual[0]
    return f1, g * (1 - sqrt(f1 / g) - f1 / g * sin(10 * pi * f1))


def zdt4(individual):
    g = 1 + 10 * (len(individual) - 1) + sum(
        x ** 2 - 10 * cos(4 * pi * x) for x in individual[1:]
    )
    f1 = individual[0]
    return f1, g * (1 - sqrt(f1 / g))


def zdt6(individual):
    f1 = 1 - exp(-4 * individual[0]) * sin(6 * pi * individual[0]) ** 6
    g = 1 + 9 * (sum(individual[1:]) / (len(individual) - 1)) ** 0.25
    return f1, g * (1 - (f1 / g) ** 2)


def dtlz1(individual, obj):
    g = 100 * (
        len(individual) - obj + 1
        + sum(
            (x - 0.5) ** 2 - cos(20 * pi * (x - 0.5))
            for x in individual[obj - 1:]
        )
    )

    values = []
    for index in range(obj):
        value = 0.5 * (1 + g)
        value *= reduce(mul, individual[:obj - index - 1], 1)
        if index != 0:
            value *= 1 - individual[obj - index - 1]
        values.append(value)
    return tuple(values)


def dtlz2(individual, obj):
    g = sum((x - 0.5) ** 2 for x in individual[obj - 1:])

    values = []
    for index in range(obj):
        value = 1.0 + g
        value *= reduce(
            mul,
            (cos(x * pi / 2.0) for x in individual[:obj - index - 1]),
            1.0,
        )
        if index != 0:
            value *= sin(individual[obj - index - 1] * pi / 2.0)
        values.append(value)
    return tuple(values)


def dtlz3(individual, obj):
    g = 100 * (
        len(individual) - obj + 1
        + sum(
            (x - 0.5) ** 2 - cos(20 * pi * (x - 0.5))
            for x in individual[obj - 1:]
        )
    )

    values = []
    for index in range(obj):
        value = 1.0 + g
        value *= reduce(
            mul,
            (cos(x * pi / 2.0) for x in individual[:obj - index - 1]),
            1.0,
        )
        if index != 0:
            value *= sin(individual[obj - index - 1] * pi / 2.0)
        values.append(value)
    return tuple(values)


def dtlz4(individual, obj, alpha=100):
    g = sum((x - 0.5) ** 2 for x in individual[obj - 1:])

    values = []
    for index in range(obj):
        value = 1.0 + g
        value *= reduce(
            mul,
            (
                cos(x ** alpha * pi / 2.0)
                for x in individual[:obj - index - 1]
            ),
            1.0,
        )
        if index != 0:
            value *= sin(individual[obj - index - 1] ** alpha * pi / 2.0)
        values.append(value)
    return tuple(values)


def dtlz5(individual, obj):
    g = sum((x - 0.5) ** 2 for x in individual[obj - 1:])
    theta = [0.0] * (obj - 1)
    theta[0] = individual[0] * pi / 2.0

    for index in range(1, obj - 1):
        theta[index] = pi / (4.0 * (1.0 + g)) * (
            1.0 + 2.0 * g * individual[index]
        )

    values = []
    for index in range(obj):
        value = 1.0 + g
        value *= reduce(
            mul,
            (cos(x) for x in theta[:obj - index - 1]),
            1.0,
        )
        if index != 0:
            value *= sin(theta[obj - index - 1])
        values.append(value)
    return tuple(values)


def dtlz6(individual, obj):
    g = sum(x ** 0.1 for x in individual[obj - 1:])
    theta = [0.0] * (obj - 1)
    theta[0] = individual[0] * pi / 2.0

    for index in range(1, obj - 1):
        theta[index] = pi / (4.0 * (1.0 + g)) * (
            1.0 + 2.0 * g * individual[index]
        )

    values = []
    for index in range(obj):
        value = 1.0 + g
        value *= reduce(
            mul,
            (cos(x) for x in theta[:obj - index - 1]),
            1.0,
        )
        if index != 0:
            value *= sin(theta[obj - index - 1])
        values.append(value)
    return tuple(values)


def dtlz7(individual, obj):
    g = 1 + 9.0 / (len(individual) - obj + 1) * sum(individual[obj - 1:])
    values = list(individual[:obj - 1])
    h = obj - sum(
        x / (1.0 + g) * (1.0 + sin(3.0 * pi * x))
        for x in values
    )
    values.append((1.0 + g) * h)
    return tuple(values)