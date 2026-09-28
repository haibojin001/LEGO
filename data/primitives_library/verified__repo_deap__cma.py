import copy
from math import sqrt, log, exp
from itertools import cycle
import warnings

import numpy

from . import tools


class Strategy(object):
    def __init__(self, centroid, sigma, **kargs):
        self.params = kargs
        self.centroid = numpy.array(centroid)
        self.dim = len(self.centroid)
        self.sigma = sigma
        self.pc = numpy.zeros(self.dim)
        self.ps = numpy.zeros(self.dim)
        self.chiN = sqrt(self.dim) * (
            1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2)
        )

        self.C = self.params.get("cmatrix", numpy.identity(self.dim))
        self.diagD, self.B = numpy.linalg.eigh(self.C)
        indx = numpy.argsort(self.diagD)
        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD
        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]

        self.lambda_ = self.params.get(
            "lambda_", int(4 + 3 * log(self.dim))
        )
        self.update_count = 0
        self.computeParams(self.params)

    def generate(self, ind_init):
        arz = numpy.random.standard_normal((self.lambda_, self.dim))
        arz = self.centroid + self.sigma * numpy.dot(arz, self.BD.T)
        return [ind_init(a) for a in arz]

    def update(self, population):
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        old_centroid = self.centroid
        self.centroid = numpy.dot(self.weights, population[:self.mu])
        c_diff = self.centroid - old_centroid

        self.ps = (
            (1.0 - self.cs) * self.ps
            + sqrt(self.cs * (2.0 - self.cs) * self.mueff) / self.sigma
            * numpy.dot(
                self.B,
                (1.0 / self.diagD) * numpy.dot(self.B.T, c_diff),
            )
        )

        hsig = float(
            numpy.linalg.norm(self.ps)
            / sqrt(
                1.0
                - (1.0 - self.cs) ** (2.0 * (self.update_count + 1.0))
            )
            / self.chiN
            < 1.4 + 2.0 / (self.dim + 1.0)
        )
        self.update_count += 1

        self.pc = (
            (1.0 - self.cc) * self.pc
            + hsig
            * sqrt(self.cc * (2.0 - self.cc) * self.mueff)
            / self.sigma
            * c_diff
        )

        artmp = population[:self.mu] - old_centroid
        self.C = (
            (
                1.0
                - self.ccov1
                - self.ccovmu
                + (1.0 - hsig)
                * self.ccov1
                * self.cc
                * (2.0 - self.cc)
            )
            * self.C
            + self.ccov1 * numpy.outer(self.pc, self.pc)
            + self.ccovmu
            * numpy.dot(self.weights * artmp.T, artmp)
            / self.sigma ** 2
        )

        self.sigma *= numpy.exp(
            (numpy.linalg.norm(self.ps) / self.chiN - 1.0)
            * self.cs
            / self.damps
        )

        self.diagD, self.B = numpy.linalg.eigh(self.C)
        indx = numpy.argsort(self.diagD)
        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]
        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

    def computeParams(self, params):
        self.mu = params.get("mu", int(self.lambda_ / 2))
        rweights = params.get("weights", "superlinear")

        if rweights == "superlinear":
            self.weights = (
                log(self.mu + 0.5)
                - numpy.log(numpy.arange(1, self.mu + 1))
            )
        elif rweights == "linear":
            self.weights = self.mu + 0.5 - numpy.arange(1, self.mu + 1)
        elif rweights == "equal":
            self.weights = numpy.ones(self.mu)
        else:
            raise RuntimeError("Unknown weights : %s" % rweights)

        self.weights /= sum(self.weights)
        self.mueff = 1.0 / sum(self.weights ** 2)

        self.cs = params.get(
            "cs", (self.mueff + 2.0) / (self.dim + self.mueff + 3.0)
        )
        self.damps = params.get(
            "damps",
            1.0
            + 2.0
            * max(0.0, sqrt((self.mueff - 1.0) / (self.dim + 1.0)) - 1.0)
            + self.cs,
        )
        self.cc = params.get("ccum", 4.0 / (self.dim + 4.0))
        self.ccov1 = params.get(
            "ccov1", 2.0 / ((self.dim + 1.3) ** 2 + self.mueff)
        )
        self.ccovmu = params.get(
            "ccovmu",
            2.0
            * (self.mueff - 2.0 + 1.0 / self.mueff)
            / ((self.dim + 2.0) ** 2 + self.mueff),
        )


class StrategyOnePlusLambda(object):
    def __init__(self, parent, sigma, **kargs):
        self.params = kargs
        self.parent = parent
        self.centroid = numpy.array(parent)
        self.dim = len(self.centroid)
        self.sigma = sigma
        self.pc = numpy.zeros(self.dim)
        self.psucc = 0.0
        self.C = self.params.get("cmatrix", numpy.identity(self.dim))
        self.A = numpy.linalg.cholesky(self.C)
        self.lambda_ = self.params.get("lambda_", 1)
        self.computeParams(self.params)

    def generate(self, ind_init):
        arz = numpy.random.standard_normal((self.lambda_, self.dim))
        arz = self.centroid + self.sigma * numpy.dot(arz, self.A.T)
        return [ind_init(a) for a in arz]

    def update(self, population):
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        lambda_succ = sum(
            self.parent.fitness <= individual.fitness
            for individual in population
        )
        p_succ = float(lambda_succ) / self.lambda_
        self.psucc = (1.0 - self.cp) * self.psucc + self.cp * p_succ

        if self.parent.fitness <= population[0].fitness:
            old_centroid = self.centroid
            self.parent = copy.deepcopy(population[0])
            self.centroid = numpy.array(self.parent)

            if self.psucc < self.pthresh:
                self.pc = (
                    (1.0 - self.cc) * self.pc
                    + sqrt(self.cc * (2.0 - self.cc))
                    * (self.centroid - old_centroid)
                    / self.sigma
                )
                self.C = (
                    (1.0 - self.ccov) * self.C
                    + self.ccov * numpy.outer(self.pc, self.pc)
                )
            else:
                self.pc = (1.0 - self.cc) * self.pc
                self.C = (
                    (1.0 - self.ccov) * self.C
                    + self.ccov
                    * (
                        numpy.outer(self.pc, self.pc)
                        + self.cc * (2.0 - self.cc) * self.C
                    )
                )

            self.A = numpy.linalg.cholesky(self.C)

        self.sigma *= exp(
            (self.psucc - self.ptarg)
            / (self.damps * (1.0 - self.ptarg))
        )

    def computeParams(self, params):
        self.ptarg = params.get("ptarg", 1.0 / 5.0)
        self.cp = params.get(
            "cp",
            self.ptarg * self.lambda_
            / (2.0 + self.ptarg * self.lambda_),
        )
        self.cc = params.get("cc", 2.0 / (self.dim + 2.0))
        self.ccov = params.get(
            "ccov", 2.0 / (self.dim ** 2 + 6.0)
        )
        self.damps = params.get(
            "damps", 1.0 + self.dim / (2.0 * self.lambda_)
        )
        self.pthresh = params.get("pthresh", 0.44)


class StrategyMultiObjective(object):
    def __init__(self, parents, sigma, **kargs):
        self.params = kargs
        self.parents = list(parents)
        self.dim = len(self.parents[0])
        self.lambda_ = self.params.get("lambda_", len(self.parents))
        self.mu = self.params.get("mu", len(self.parents))

        if self.mu != len(self.parents):
            warnings.warn(
                "The number of parents differs from mu; using the number "
                "of parents as mu.",
                RuntimeWarning,
            )
            self.mu = len(self.parents)

        self.sigmas = [sigma for _ in self.parents]
        self.C = [
            numpy.array(
                self.params.get("cmatrix", numpy.identity(self.dim)),
                copy=True,
            )
            for _ in self.parents
        ]
        self.A = [numpy.linalg.cholesky(matrix) for matrix in self.C]
        self.pc = [numpy.zeros(self.dim) for _ in self.parents]
        self.psucc = [0.0 for _ in self.parents]
        self.ancestors_fitness = [
            copy.deepcopy(individual.fitness) for individual in self.parents
        ]
        self._sources = {}
        self.computeParams(self.params)

    def generate(self, ind_init):
        arz = numpy.random.standard_normal((self.lambda_, self.dim))
        offspring = []

        for index, parent_index in enumerate(cycle(range(len(self.parents)))):
            if index >= self.lambda_:
                break

            individual = ind_init(
                numpy.array(self.parents[parent_index])
                + self.sigmas[parent_index]
                * numpy.dot(arz[index], self.A[parent_index].T)
            )
            self._sources[id(individual)] = parent_index
            offspring.append(individual)

        return offspring

    def _select(self, individuals):
        fronts = tools.sortLogNondominated(individuals, self.mu)
        selected = []

        for front in fronts:
            if len(selected) + len(front) <= self.mu:
                selected.extend(front)
                continue

            tools.emo.assignCrowdingDist(front)
            front = sorted(
                front,
                key=lambda individual: individual.fitness.crowding_dist,
                reverse=True,
            )
            selected.extend(front[:self.mu - len(selected)])
            break

        return selected

    def update(self, population):
        old_parents = self.parents
        old_fitness = self.ancestors_fitness
        source_for_parent = {
            id(parent): index for index, parent in enumerate(old_parents)
        }

        selected = self._select(list(population) + list(old_parents))
        new_parents = []
        new_sigmas = []
        new_covariances = []
        new_factors = []
        new_paths = []
        new_success = []
        new_ancestors = []

        for individual in selected:
            source = self._sources.get(
                id(individual), source_for_parent.get(id(individual), 0)
            )
            source = min(source, len(old_parents) - 1)

            sigma = self.sigmas[source]
            covariance = numpy.array(self.C[source], copy=True)
            path = numpy.array(self.pc[source], copy=True)
            psucc = self.psucc[source]
            ancestor = old_fitness[source]

            successful = ancestor <= individual.fitness
            psucc = (
                (1.0 - self.cp) * psucc
                + self.cp * float(successful)
            )

            if successful:
                step = (
                    numpy.array(individual) - numpy.array(old_parents[source])
                )
                if psucc < self.pthresh:
                    path = (
                        (1.0 - self.cc) * path
                        + sqrt(self.cc * (2.0 - self.cc))
                        * step
                        / sigma
                    )
                    covariance = (
                        (1.0 - self.ccov) * covariance
                        + self.ccov * numpy.outer(path, path)
                    )
                else:
                    path = (1.0 - self.cc) * path
                    covariance = (
                        (1.0 - self.ccov) * covariance
                        + self.ccov
                        * (
                            numpy.outer(path, path)
                            + self.cc * (2.0 - self.cc) * covariance
                        )
                    )

            sigma *= exp(
                (psucc - self.ptarg)
                / (self.damps * (1.0 - self.ptarg))
            )

            new_parents.append(copy.deepcopy(individual))
            new_sigmas.append(sigma)
            new_covariances.append(covariance)
            new_factors.append(numpy.linalg.cholesky(covariance))
            new_paths.append(path)
            new_success.append(psucc)
            new_ancestors.append(copy.deepcopy(individual.fitness))

        self.parents = new_parents
        self.sigmas = new_sigmas
        self.C = new_covariances
        self.A = new_factors
        self.pc = new_paths
        self.psucc = new_success
        self.ancestors_fitness = new_ancestors
        self._sources.clear()

    def computeParams(self, params):
        self.ptarg = params.get("ptarg", 1.0 / 5.0)
        self.cp = params.get(
            "cp",
            self.ptarg * self.lambda_
            / (2.0 + self.ptarg * self.lambda_),
        )
        self.cc = params.get("cc", 2.0 / (self.dim + 2.0))
        self.ccov = params.get(
            "ccov", 2.0 / (self.dim ** 2 + 6.0)
        )
        self.damps = params.get(
            "damps", 1.0 + self.dim / (2.0 * self.lambda_)
        )
        self.pthresh = params.get("pthresh", 0.44)