import math
import itertools
import random

from collections.abc import Sequence


def cone(individual, position, height, width):
    value = 0.0
    for x, p in zip(individual, position):
        value += (x - p) ** 2
    return height - width * math.sqrt(value)


def sphere(individual, position, height, width):
    value = 0.0
    for x, p in zip(individual, position):
        value += (x - p) ** 2
    return height * value


def function1(individual, position, height, width):
    value = 0.0
    for x, p in zip(individual, position):
        value += (x - p) ** 2
    return height / (1 + width * value)


class MovingPeaks:
    def __init__(self, dim, random=random, **kargs):
        sc = SCENARIO_1.copy()
        sc.update(kargs)

        pfunc = sc.get("pfunc")
        npeaks = sc.get("npeaks")
        self.dim = dim

        self.minpeaks, self.maxpeaks = None, None
        if hasattr(npeaks, "__getitem__"):
            self.minpeaks, npeaks, self.maxpeaks = npeaks
            self.number_severity = sc.get("number_severity")

        try:
            if len(pfunc) == npeaks:
                self.peaks_function = pfunc
            else:
                self.peaks_function = self.random.sample(pfunc, npeaks)
            self.pfunc_pool = tuple(pfunc)
        except TypeError:
            self.peaks_function = list(itertools.repeat(pfunc, npeaks))
            self.pfunc_pool = (pfunc,)

        self.random = random
        self.basis_function = sc.get("bfunc")

        self.min_coord = sc.get("min_coord")
        self.max_coord = sc.get("max_coord")

        self.min_height = sc.get("min_height")
        self.max_height = sc.get("max_height")
        uniform_height = sc.get("uniform_height")

        self.min_width = sc.get("min_width")
        self.max_width = sc.get("max_width")
        uniform_width = sc.get("uniform_width")

        self.lambda_ = sc.get("lambda_")
        self.move_severity = sc.get("move_severity")
        self.height_severity = sc.get("height_severity")
        self.width_severity = sc.get("width_severity")
        self.period = sc.get("period")

        self.peaks_position = [
            [self.random.uniform(self.min_coord, self.max_coord) for _ in range(dim)]
            for _ in range(npeaks)
        ]

        if uniform_height != 0:
            self.peaks_height = [uniform_height for _ in range(npeaks)]
        else:
            self.peaks_height = [
                self.random.uniform(self.min_height, self.max_height)
                for _ in range(npeaks)
            ]

        if uniform_width != 0:
            self.peaks_width = [uniform_width for _ in range(npeaks)]
        else:
            self.peaks_width = [
                self.random.uniform(self.min_width, self.max_width)
                for _ in range(npeaks)
            ]

        self.last_change_vector = [[0.0] * dim for _ in range(npeaks)]
        self.nevals = 0

    def __call__(self, individual):
        values = []
        for func, position, height, width in zip(
            self.peaks_function,
            self.peaks_position,
            self.peaks_height,
            self.peaks_width,
        ):
            values.append(func(individual, position, height, width))

        if self.basis_function is not None:
            values.append(self.basis_function(individual))

        self.nevals += 1
        if self.nevals % self.period == 0:
            self.changePeaks()

        return max(values),

    def globalMaximum(self):
        potential_max = [
            func(position, position, height, width)
            for func, position, height, width in zip(
                self.peaks_function,
                self.peaks_position,
                self.peaks_height,
                self.peaks_width,
            )
        ]
        maximum = max(enumerate(potential_max), key=lambda item: item[1])
        return maximum[1], self.peaks_position[maximum[0]]

    def maximums(self):
        maximums = []
        for func, position, height, width in zip(
            self.peaks_function,
            self.peaks_position,
            self.peaks_height,
            self.peaks_width,
        ):
            maximums.append((func(position, position, height, width), position))
        return sorted(maximums, reverse=True)

    def changePeaks(self):
        if self.minpeaks is not None:
            npeaks = len(self.peaks_function)

            if self.random.random() < 0.5:
                nrem = int(self.random.uniform(0, self.number_severity * npeaks))
                nrem = min(nrem, npeaks - self.minpeaks)

                for _ in range(nrem):
                    index = self.random.randrange(len(self.peaks_function))
                    self.peaks_function.pop(index)
                    self.peaks_position.pop(index)
                    self.peaks_height.pop(index)
                    self.peaks_width.pop(index)
                    self.last_change_vector.pop(index)
            else:
                nadd = int(self.random.uniform(0, self.number_severity * npeaks))
                nadd = min(nadd, self.maxpeaks - npeaks)

                for _ in range(nadd):
                    self.peaks_function.append(self.random.choice(self.pfunc_pool))
                    self.peaks_position.append([
                        self.random.uniform(self.min_coord, self.max_coord)
                        for _ in range(self.dim)
                    ])
                    self.peaks_height.append(
                        self.random.uniform(self.min_height, self.max_height)
                    )
                    self.peaks_width.append(
                        self.random.uniform(self.min_width, self.max_width)
                    )
                    self.last_change_vector.append([0.0] * self.dim)

        for index in range(len(self.peaks_function)):
            change = self.random.gauss(0, 1) * self.height_severity
            new_height = self.peaks_height[index] + change

            if new_height < self.min_height:
                new_height = 2.0 * self.min_height - self.peaks_height[index] - change
            elif new_height > self.max_height:
                new_height = 2.0 * self.max_height - self.peaks_height[index] - change

            self.peaks_height[index] = new_height

            change = self.random.gauss(0, 1) * self.width_severity
            new_width = self.peaks_width[index] + change

            if new_width < self.min_width:
                new_width = 2.0 * self.min_width - self.peaks_width[index] - change
            elif new_width > self.max_width:
                new_width = 2.0 * self.max_width - self.peaks_width[index] - change

            self.peaks_width[index] = new_width

            shift = [self.random.random() - 0.5 for _ in range(self.dim)]
            shift_length = math.sqrt(sum(component ** 2 for component in shift))
            shift = [component / shift_length for component in shift]

            shift = [
                self.lambda_ * previous + (1.0 - self.lambda_) * current
                for previous, current in zip(self.last_change_vector[index], shift)
            ]

            shift_length = math.sqrt(sum(component ** 2 for component in shift))
            shift = [
                component * self.move_severity / shift_length
                for component in shift
            ]

            for coordinate in range(self.dim):
                previous = self.peaks_position[index][coordinate]
                new_position = previous + shift[coordinate]

                if new_position < self.min_coord:
                    new_position = 2.0 * self.min_coord - previous - shift[coordinate]
                    shift[coordinate] = -shift[coordinate]
                elif new_position > self.max_coord:
                    new_position = 2.0 * self.max_coord - previous - shift[coordinate]
                    shift[coordinate] = -shift[coordinate]

                self.peaks_position[index][coordinate] = new_position

            self.last_change_vector[index] = shift


SCENARIO_1 = {
    "pfunc": function1,
    "npeaks": 5,
    "bfunc": None,
    "min_coord": 0.0,
    "max_coord": 100.0,
    "min_height": 30.0,
    "max_height": 70.0,
    "uniform_height": 50.0,
    "min_width": 0.0001,
    "max_width": 0.2,
    "uniform_width": 0.1,
    "lambda_": 0.0,
    "move_severity": 1.0,
    "height_severity": 7.0,
    "width_severity": 0.01,
    "period": 5000,
}

SCENARIO_2 = {
    "pfunc": cone,
    "npeaks": 10,
    "bfunc": None,
    "min_coord": 0.0,
    "max_coord": 100.0,
    "min_height": 30.0,
    "max_height": 70.0,
    "uniform_height": 50.0,
    "min_width": 1.0,
    "max_width": 12.0,
    "uniform_width": 0.0,
    "lambda_": 0.5,
    "move_severity": 1.5,
    "height_severity": 7.0,
    "width_severity": 1.0,
    "period": 5000,
}

SCENARIO_3 = {
    "pfunc": cone,
    "npeaks": 50,
    "bfunc": lambda x: 10,
    "min_coord": 0.0,
    "max_coord": 100.0,
    "min_height": 30.0,
    "max_height": 70.0,
    "uniform_height": 0,
    "min_width": 1.0,
    "max_width": 12.0,
    "uniform_width": 0,
    "lambda_": 0.5,
    "move_severity": 1.0,
    "height_severity": 1.0,
    "width_severity": 0.5,
    "period": 1000,
}