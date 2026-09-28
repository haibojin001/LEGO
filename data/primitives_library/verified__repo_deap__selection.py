import random
import numpy as np

from functools import partial
from operator import attrgetter


def selRandom(individuals, k):
    """Choose *k* members uniformly at random, allowing repeats."""
    return [random.choice(individuals) for _ in range(k)]


def selBest(individuals, k, fit_attr="fitness"):
    """Return the highest-ranking *k* individuals."""
    return sorted(individuals, key=attrgetter(fit_attr), reverse=True)[:k]


def selWorst(individuals, k, fit_attr="fitness"):
    """Return the lowest-ranking *k* individuals."""
    return sorted(individuals, key=attrgetter(fit_attr))[:k]


def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """Perform repeated tournaments and return their winners."""
    selected = []
    key = attrgetter(fit_attr)

    for _ in range(k):
        contestants = selRandom(individuals, tournsize)
        selected.append(max(contestants, key=key))

    return selected


def selRoulette(individuals, k, fit_attr="fitness"):
    """Select individuals proportionally to their first fitness objective."""
    ordered = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    total = sum(getattr(individual, fit_attr).values[0] for individual in individuals)
    selected = []

    for _ in range(k):
        threshold = random.random() * total
        accumulated = 0

        for individual in ordered:
            accumulated += getattr(individual, fit_attr).values[0]
            if accumulated > threshold:
                selected.append(individual)
                break

    return selected


def selDoubleTournament(individuals, k, fitness_size, parsimony_size,
                        fitness_first, fit_attr="fitness"):
    """Select using nested fitness and size tournaments."""
    assert 1 <= parsimony_size <= 2, (
        "Parsimony tournament size has to be in the range [1, 2]."
    )

    def size_tournament(pool, amount, select):
        chosen = []

        for _ in range(amount):
            probability = parsimony_size / 2.0
            first, second = select(pool, k=2)

            if len(first) > len(second):
                first, second = second, first
            elif len(first) == len(second):
                probability = 0.5

            if random.random() < probability:
                chosen.append(first)
            else:
                chosen.append(second)

        return chosen

    def fitness_tournament(pool, amount, select):
        chosen = []
        key = attrgetter(fit_attr)

        for _ in range(amount):
            contestants = select(pool, k=fitness_size)
            chosen.append(max(contestants, key=key))

        return chosen

    if fitness_first:
        select_fitness = partial(fitness_tournament, select=selRandom)
        return size_tournament(individuals, k, select_fitness)

    select_size = partial(size_tournament, select=selRandom)
    return fitness_tournament(individuals, k, select_size)


def selStochasticUniversalSampling(individuals, k, fit_attr="fitness"):
    """Select individuals using evenly spaced roulette-wheel pointers."""
    ordered = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    total = sum(getattr(individual, fit_attr).values[0] for individual in individuals)

    step = total / float(k)
    first_point = random.uniform(0, step)
    points = [first_point + index * step for index in range(k)]

    selected = []
    for point in points:
        index = 0
        accumulated = getattr(ordered[index], fit_attr).values[0]

        while accumulated < point:
            index += 1
            accumulated += getattr(ordered[index], fit_attr).values[0]

        selected.append(ordered[index])

    return selected


def selLexicase(individuals, k):
    """Select individuals with lexicase selection."""
    selected = []

    for _ in range(k):
        candidates = individuals
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)

        while cases and len(candidates) > 1:
            case = cases[0]

            if individuals[0].fitness.weights[case] > 0:
                best = max(candidate.fitness.values[case]
                           for candidate in candidates)
                candidates = [
                    candidate for candidate in candidates
                    if candidate.fitness.values[case] == best
                ]
            else:
                best = min(candidate.fitness.values[case]
                           for candidate in candidates)
                candidates = [
                    candidate for candidate in candidates
                    if candidate.fitness.values[case] == best
                ]

            cases.pop(0)

        selected.append(random.choice(candidates))

    return selected


def selEpsilonLexicase(individuals, k, epsilon):
    """Select individuals with epsilon lexicase selection."""
    selected = []

    for _ in range(k):
        candidates = individuals
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)

        while cases and len(candidates) > 1:
            case = cases[0]

            if individuals[0].fitness.weights[case] > 0:
                best = max(candidate.fitness.values[case]
                           for candidate in candidates)
                minimum = best - epsilon
                candidates = [
                    candidate for candidate in candidates
                    if candidate.fitness.values[case] >= minimum
                ]
            else:
                best = min(candidate.fitness.values[case]
                           for candidate in candidates)
                maximum = best + epsilon
                candidates = [
                    candidate for candidate in candidates
                    if candidate.fitness.values[case] <= maximum
                ]

            cases.pop(0)

        selected.append(random.choice(candidates))

    return selected


def selAutomaticEpsilonLexicase(individuals, k):
    """Select individuals with automatic epsilon lexicase selection."""
    selected = []

    for _ in range(k):
        candidates = individuals
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)

        while cases and len(candidates) > 1:
            case = cases[0]
            values = [candidate.fitness.values[case] for candidate in candidates]
            median = np.median(values)
            errors = [abs(value - median) for value in values]
            epsilon = np.median(errors)

            if individuals[0].fitness.weights[case] > 0:
                best = max(values)
                minimum = best - epsilon
                candidates = [
                    candidate for candidate in candidates
                    if candidate.fitness.values[case] >= minimum
                ]
            else:
                best = min(values)
                maximum = best + epsilon
                candidates = [
                    candidate for candidate in candidates
                    if candidate.fitness.values[case] <= maximum
                ]

            cases.pop(0)

        selected.append(random.choice(candidates))

    return selected