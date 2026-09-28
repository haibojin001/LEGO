import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random

import numpy


def selNSGA2(individuals, k, nd="standard"):
    if nd == "standard":
        fronts = sortNondominated(individuals, k)
    elif nd == "log":
        fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception(
            'selNSGA2: The choice of non-dominated sorting method "{}" is invalid.'.format(nd)
        )

    for front in fronts:
        assignCrowdingDist(front)

    selected = list(chain.from_iterable(fronts[:-1]))
    remaining = k - len(selected)
    if remaining > 0:
        last = sorted(
            fronts[-1],
            key=attrgetter("fitness.crowding_dist"),
            reverse=True,
        )
        selected.extend(last[:remaining])
    return selected


def sortNondominated(individuals, k, first_front_only=False):
    if k == 0:
        return []

    grouped = defaultdict(list)
    for individual in individuals:
        grouped[individual.fitness].append(individual)

    fitnesses = list(grouped)
    dominates_count = defaultdict(int)
    dominated = defaultdict(list)
    first = []

    for index, fitness in enumerate(fitnesses):
        for other in fitnesses[index + 1:]:
            if fitness.dominates(other):
                dominates_count[other] += 1
                dominated[fitness].append(other)
            elif other.dominates(fitness):
                dominates_count[fitness] += 1
                dominated[other].append(fitness)
        if dominates_count[fitness] == 0:
            first.append(fitness)

    fronts = [[]]
    for fitness in first:
        fronts[0].extend(grouped[fitness])

    if first_front_only:
        return fronts

    sorted_count = len(fronts[0])
    limit = min(k, len(individuals))
    current = first

    while sorted_count < limit:
        following = []
        front = []
        for fitness in current:
            for inferior in dominated[fitness]:
                dominates_count[inferior] -= 1
                if dominates_count[inferior] == 0:
                    following.append(inferior)
                    front.extend(grouped[inferior])
                    sorted_count += len(grouped[inferior])
        fronts.append(front)
        current = following

    return fronts


def assignCrowdingDist(individuals):
    size = len(individuals)
    if size == 0:
        return

    values = [(individual.fitness.values, position) for position, individual in enumerate(individuals)]
    distances = [0.0] * size
    objectives = len(values[0][0])

    for objective in range(objectives):
        values.sort(key=lambda entry: entry[0][objective])
        distances[values[0][1]] = float("inf")
        distances[values[-1][1]] = float("inf")

        low = values[0][0][objective]
        high = values[-1][0][objective]
        if high == low:
            continue

        denominator = objectives * float(high - low)
        for left, center, right in zip(values, values[1:], values[2:]):
            distances[center[1]] += (right[0][objective] - left[0][objective]) / denominator

    for individual, distance in zip(individuals, distances):
        individual.fitness.crowding_dist = distance


def selTournamentDCD(individuals, k):
    if k > len(individuals):
        raise ValueError("selTournamentDCD: k must be less than or equal to individuals length")
    if k == len(individuals) and k % 4 != 0:
        raise ValueError("selTournamentDCD: k must be divisible by four if k == len(individuals)")

    def contest(first, second):
        if first.fitness.dominates(second.fitness):
            return first
        if second.fitness.dominates(first.fitness):
            return second
        if first.fitness.crowding_dist > second.fitness.crowding_dist:
            return first
        if second.fitness.crowding_dist > first.fitness.crowding_dist:
            return second
        return first if random.random() <= 0.5 else second

    permutation_a = random.sample(individuals, len(individuals))
    permutation_b = random.sample(individuals, len(individuals))
    result = []

    for offset in range(0, k, 4):
        result.append(contest(permutation_a[offset], permutation_a[offset + 1]))
        result.append(contest(permutation_a[offset + 2], permutation_a[offset + 3]))
        result.append(contest(permutation_b[offset], permutation_b[offset + 1]))
        result.append(contest(permutation_b[offset + 2], permutation_b[offset + 3]))

    return result


def identity(obj):
    return obj


def isDominated(wvalues1, wvalues2):
    strict = False
    for value1, value2 in zip(wvalues1, wvalues2):
        if value1 > value2:
            return False
        if value1 < value2:
            strict = True
    return strict


def median(seq, key=identity):
    ordered = sorted(seq, key=key)
    length = len(ordered)
    middle = length // 2
    if length % 2:
        return key(ordered[middle])
    return (key(ordered[middle - 1]) + key(ordered[middle])) / 2.0


def sortLogNondominated(individuals, k, first_front_only=False):
    if k == 0:
        return []

    grouped = defaultdict(list)
    for individual in individuals:
        grouped[individual.fitness.wvalues].append(individual)

    fitnesses = sorted(grouped, reverse=True)
    ranks = {fitness: 0 for fitness in fitnesses}
    sortNDHelperA(fitnesses, len(fitnesses[0]) - 1, ranks)

    fronts = [[] for _ in range(max(ranks.values()) + 1)]
    for fitness in fitnesses:
        fronts[ranks[fitness]].extend(grouped[fitness])

    return fronts[0] if first_front_only else fronts


def sortNDHelperA(fitnesses, obj, front):
    count = len(fitnesses)
    if count < 2:
        return

    for later in range(count):
        target = fitnesses[later]
        rank = front[target]
        for earlier in range(later):
            candidate = fitnesses[earlier]
            if isDominated(target, candidate):
                rank = max(rank, front[candidate] + 1)
        front[target] = rank


def sortNDHelperB(best, worst, obj, front):
    if not best or not worst:
        return

    for inferior in worst:
        rank = front[inferior]
        for superior in best:
            if isDominated(inferior, superior):
                rank = max(rank, front[superior] + 1)
        front[inferior] = rank


def splitA(fitnesses, obj):
    pivot = median(fitnesses, key=itemgetter(obj))
    upper = []
    lower = []
    equal = []

    for fitness in fitnesses:
        if fitness[obj] > pivot:
            upper.append(fitness)
        elif fitness[obj] < pivot:
            lower.append(fitness)
        else:
            equal.append(fitness)

    needed = max(0, len(fitnesses) // 2 - len(upper))
    upper.extend(equal[:needed])
    lower.extend(equal[needed:])
    return upper, lower


def splitB(best, worst, obj):
    pivot = median(best, key=itemgetter(obj))
    best_upper = []
    best_lower = []
    worst_upper = []
    worst_lower = []

    for fitness in best:
        if fitness[obj] >= pivot:
            best_upper.append(fitness)
        else:
            best_lower.append(fitness)

    for fitness in worst:
        if fitness[obj] >= pivot:
            worst_upper.append(fitness)
        else:
            worst_lower.append(fitness)

    return best_upper, best_lower, worst_upper, worst_lower


def sweepA(fitnesses, front):
    for index, fitness in enumerate(fitnesses):
        rank = front[fitness]
        for candidate in fitnesses[:index]:
            if isDominated(fitness, candidate):
                rank = max(rank, front[candidate] + 1)
        front[fitness] = rank


def sweepB(best, worst, front):
    for inferior in worst:
        rank = front[inferior]
        for superior in best:
            if isDominated(inferior, superior):
                rank = max(rank, front[superior] + 1)
        front[inferior] = rank


def selNSGA3(individuals, k, ref_points, nd="log", best_point=None,
             worst_point=None, extreme_points=None, return_memory=False):
    if nd == "standard":
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == "log":
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception(
            'selNSGA3: The choice of non-dominated sorting method "{}" is invalid.'.format(nd)
        )

    ordered = list(chain.from_iterable(pareto_fronts))
    fitnesses = numpy.asarray([individual.fitness.wvalues for individual in ordered], dtype=float)
    fitnesses *= -1.0

    current_best = numpy.min(fitnesses, axis=0)
    current_worst = numpy.max(fitnesses, axis=0)

    if best_point is not None:
        best_point = numpy.min(numpy.concatenate((best_point, current_best[None, :])), axis=0)
    else:
        best_point = current_best

    if worst_point is not None:
        worst_point = numpy.max(numpy.concatenate((worst_point, current_worst[None, :])), axis=0)
    else:
        worst_point = current_worst

    extreme_points = find_extreme_points(fitnesses, best_point, extreme_points)
    intercepts = find_intercepts(extreme_points, best_point, current_worst)
    niches, distances = associate_to_niche(fitnesses, ref_points, best_point, intercepts)

    selected = list(chain.from_iterable(pareto_fronts[:-1]))
    selected_count = len(selected)
    niche_counts = numpy.zeros(len(ref_points), dtype=numpy.int64)

    if selected_count:
        niche_counts += numpy.bincount(niches[:selected_count], minlength=len(ref_points))

    remaining = k - selected_count
    if remaining > 0:
        selected.extend(
            niching(
                pareto_fronts[-1],
                remaining,
                niches[selected_count:],
                distances[selected_count:],
                niche_counts,
            )
        )

    if return_memory:
        return selected, NSGA3Memory(best_point, worst_point, extreme_points)
    return selected


def find_extreme_points(fitnesses, best_point, extreme_points=None):
    translated = fitnesses - best_point

    if extreme_points is not None:
        translated = numpy.concatenate((extreme_points - best_point, translated), axis=0)
        fitnesses = numpy.concatenate((extreme_points, fitnesses), axis=0)

    dimensions = best_point.shape[0]
    achievement = numpy.eye(dimensions)
    achievement[achievement == 0] = 1.0e6
    values = numpy.max(translated[None, :, :] * achievement[:, None, :], axis=2)
    return fitnesses[numpy.argmin(values, axis=1)]


def find_intercepts(extreme_points, best_point, current_worst):
    system = extreme_points - best_point
    rhs = numpy.ones(system.shape[1])

    try:
        coefficients = numpy.linalg.solve(system, rhs)
    except numpy.linalg.LinAlgError:
        return current_worst

    if numpy.count_nonzero(coefficients) != len(coefficients):
        return current_worst

    intercepts = 1.0 / coefficients
    if (
        not numpy.all(numpy.isfinite(intercepts))
        or numpy.any(intercepts <= 1.0e-6)
        or numpy.any(intercepts + best_point > current_worst)
    ):
        return current_worst

    return intercepts + best_point


def associate_to_niche(fitnesses, reference_points, best_point, intercepts):
    normalized = fitnesses - best_point
    denominator = intercepts - best_point
    normalized = normalized / denominator

    norms = numpy.linalg.norm(reference_points, axis=1)
    projection = numpy.dot(normalized, reference_points.T) / (norms * norms)
    projected = projection[:, :, None] * reference_points[None, :, :]
    distances = numpy.linalg.norm(normalized[:, None, :] - projected, axis=2)

    niches = numpy.argmin(distances, axis=1)
    distance = distances[numpy.arange(len(fitnesses)), niches]
    return niches, distance


def niching(individuals, k, niches, distances, niche_counts):
    selected = []
    available = numpy.ones(len(niche_counts), dtype=bool)
    niches = numpy.asarray(niches).copy()
    distances = numpy.asarray(distances)

    while len(selected) < k:
        active = numpy.flatnonzero(available)
        minimum = numpy.min(niche_counts[active])
        candidate_niches = active[niche_counts[active] == minimum]
        random.shuffle(candidate_niches)

        for niche in candidate_niches:
            if len(selected) >= k:
                break

            candidates = numpy.flatnonzero(niches == niche)
            if len(candidates) == 0:
                available[niche] = False
                continue

            if niche_counts[niche] == 0:
                chosen_index = candidates[numpy.argmin(distances[candidates])]
            else:
                chosen_index = random.choice(candidates)

            selected.append(individuals[chosen_index])
            niches[chosen_index] = -1
            niche_counts[niche] += 1

    return selected


NSGA3Memory = namedtuple("NSGA3Memory", ["best_point", "worst_point", "extreme_points"])


class selNSGA3WithMemory:
    def __init__(self, ref_points, nd="log"):
        self.ref_points = ref_points
        self.nd = nd
        self.best_point = numpy.full((1, ref_points.shape[1]), numpy.inf)
        self.worst_point = numpy.full((1, ref_points.shape[1]), -numpy.inf)
        self.extreme_points = None

    def __call__(self, individuals, k):
        chosen, memory = selNSGA3(
            individuals,
            k,
            self.ref_points,
            self.nd,
            self.best_point,
            self.worst_point,
            self.extreme_points,
            True,
        )
        self.best_point = memory.best_point.reshape((1, -1))
        self.worst_point = memory.worst_point.reshape((1, -1))
        self.extreme_points = memory.extreme_points
        return chosen


def uniform_reference_points(nobj, p=4, scaling=None):
    points = []

    def generate(point, left, total, depth):
        if depth == nobj - 1:
            point[depth] = left / total
            points.append(point.copy())
            return
        for value in range(left + 1):
            point[depth] = value / total
            generate(point, left - value, total, depth + 1)

    generate(numpy.zeros(nobj), p, p, 0)
    points = numpy.asarray(points)

    if scaling is not None:
        points *= scaling
        points += (1.0 - scaling) / nobj

    return points


def selSPEA2(individuals, k):
    size = len(individuals)
    strength = [0] * size
    raw_fitness = [0] * size
    dominated_by = [[] for _ in range(size)]

    for first in range(size):
        for second in range(first + 1, size):
            if individuals[first].fitness.dominates(individuals[second].fitness):
                strength[first] += 1
                dominated_by[second].append(first)
            elif individuals[second].fitness.dominates(individuals[first].fitness):
                strength[second] += 1
                dominated_by[first].append(second)

    for index in range(size):
        raw_fitness[index] = sum(strength[dominator] for dominator in dominated_by[index])

    selected = [index for index, value in enumerate(raw_fitness) if value < 1]

    if len(selected) < k:
        distances = [[0.0] * size for _ in range(size)]
        objective_values = [individual.fitness.values for individual in individuals]

        for first in range(size):
            for second in range(first + 1, size):
                distance = sum(
                    (left - right) ** 2
                    for left, right in zip(objective_values[first], objective_values[second])
                )
                distances[first][second] = distance
                distances[second][first] = distance

        order = int(math.sqrt(size))
        density = []
        for index in range(size):
            nearest = sorted(distances[index])[order]
            density.append(1.0 / (nearest + 2.0))

        scores = [raw_fitness[index] + density[index] for index in range(size)]
        remaining = [index for index in range(size) if index not in selected]
        remaining.sort(key=lambda index: scores[index])
        selected.extend(remaining[:k - len(selected)])

    elif len(selected) > k:
        count = len(selected)
        distances = [[0.0] * count for _ in range(count)]
        objective_values = [individuals[index].fitness.values for index in selected]

        for first in range(count):
            for second in range(first + 1, count):
                distance = sum(
                    (left - right) ** 2
                    for left, right in zip(objective_values[first], objective_values[second])
                )
                distances[first][second] = distance
                distances[second][first] = distance

        sorted_distances = []
        for index in range(count):
            ordered = sorted(
                (distances[index][other], other)
                for other in range(count)
                if other != index
            )
            sorted_distances.append(ordered)

        remove_count = count - k
        removed = set()

        for _ in range(remove_count):
            candidates = [index for index in range(count) if index not in removed]
            victim = min(
                candidates,
                key=lambda index: tuple(
                    distance for distance, other in sorted_distances[index]
                    if other not in removed
                ),
            )
            removed.add(victim)

        selected = [index for position, index in enumerate(selected) if position not in removed]

    return [individuals[index] for index in selected]