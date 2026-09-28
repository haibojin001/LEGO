from functools import wraps
from itertools import repeat

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


class DeltaPenalty(object):
    """Decorator applying a fixed and optional distance-based penalty."""

    def __init__(self, feasibility, delta, distance=None):
        self.fbty_fct = feasibility
        self.delta = delta if isinstance(delta, Sequence) else repeat(delta)
        self.dist_fct = distance

    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            if self.fbty_fct(individual):
                return func(individual, *args, **kwargs)

            weights = tuple(
                1 if weight >= 0 else -1
                for weight in individual.fitness.weights
            )

            distances = tuple(0 for _ in individual.fitness.weights)
            if self.dist_fct is not None:
                distances = self.dist_fct(individual)
                if not isinstance(distances, Sequence):
                    distances = repeat(distances)

            return tuple(
                delta - weight * distance
                for delta, weight, distance in zip(
                    self.delta, weights, distances
                )
            )

        return wrapper


DeltaPenality = DeltaPenalty


class ClosestValidPenalty(object):
    """Decorator evaluating a closest feasible individual with a penalty."""

    def __init__(self, feasibility, feasible, alpha, distance=None):
        self.fbty_fct = feasibility
        self.fbl_fct = feasible
        self.alpha = alpha
        self.dist_fct = distance

    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            if self.fbty_fct(individual):
                return func(individual, *args, **kwargs)

            feasible_individual = self.fbl_fct(individual)
            feasible_fitness = func(feasible_individual, *args, **kwargs)

            weights = tuple(
                1.0 if weight >= 0 else -1.0
                for weight in individual.fitness.weights
            )

            if len(weights) != len(feasible_fitness):
                raise IndexError(
                    "Fitness weights and computed fitness are of different size."
                )

            distances = tuple(0 for _ in individual.fitness.weights)
            if self.dist_fct is not None:
                distances = self.dist_fct(feasible_individual, individual)
                if not isinstance(distances, Sequence):
                    distances = repeat(distances)

            return tuple(
                fitness - weight * self.alpha * distance
                for fitness, weight, distance in zip(
                    feasible_fitness, weights, distances
                )
            )

        return wrapper


ClosestValidPenality = ClosestValidPenalty


__all__ = [
    "DeltaPenalty",
    "ClosestValidPenalty",
    "DeltaPenality",
    "ClosestValidPenality",
]