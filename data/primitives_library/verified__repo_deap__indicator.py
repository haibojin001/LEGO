import numpy
import moocore


__all__ = ["hypervolume"]


def hypervolume(front, **kargs):
    values = numpy.array([individual.fitness.wvalues for individual in front]) * -1
    reference = kargs.get("ref", None)

    if reference is None:
        reference = numpy.max(values, axis=0) + 1

    contributions = [
        moocore.hypervolume(
            numpy.concatenate((values[:index], values[index + 1:])),
            ref=reference,
        )
        for index in range(len(front))
    ]

    return numpy.argmax(contributions)