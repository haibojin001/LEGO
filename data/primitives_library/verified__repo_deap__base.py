import sys

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from copy import deepcopy
from functools import partial
from operator import mul, truediv


class Toolbox(object):
    def __init__(self):
        self.register("clone", deepcopy)
        self.register("map", map)

    def register(self, alias, function, *args, **kargs):
        registered = partial(function, *args, **kargs)
        registered.__name__ = alias
        registered.__doc__ = function.__doc__

        if hasattr(function, "__dict__") and not isinstance(function, type):
            registered.__dict__.update(function.__dict__.copy())

        setattr(self, alias, registered)

    def unregister(self, alias):
        delattr(self, alias)

    def decorate(self, alias, *decorators):
        registered = getattr(self, alias)
        function = registered.func
        args = registered.args
        kargs = registered.keywords

        for decorator in decorators:
            function = decorator(function)

        self.register(alias, function, *args, **kargs)


class Fitness(object):
    weights = None
    wvalues = ()

    def __init__(self, values=()):
        if self.weights is None:
            raise TypeError(
                "Can't instantiate abstract %r with abstract attribute weights."
                % self.__class__
            )

        if not isinstance(self.weights, Sequence):
            raise TypeError(
                "Attribute weights of %r must be a sequence." % self.__class__
            )

        if len(values) > 0:
            self.values = values

    def getValues(self):
        return tuple(map(truediv, self.wvalues, self.weights))

    def setValues(self, values):
        assert len(values) == len(self.weights), (
            "Assigned values have not the same length than fitness weights"
        )
        try:
            self.wvalues = tuple(map(mul, values, self.weights))
        except TypeError:
            _, _, traceback = sys.exc_info()
            raise TypeError(
                "Both weights and assigned values must be a sequence of numbers "
                "when assigning to values of %r. Currently assigning value(s) %r "
                "of %r to a fitness with weights %s."
                % (self.__class__, values, type(values), self.weights)
            ).with_traceback(traceback)

    def delValues(self):
        self.wvalues = ()

    values = property(
        getValues,
        setValues,
        delValues,
        "Fitness values. Use directly ``individual.fitness.values = values`` "
        "in order to set the fitness and ``del individual.fitness.values`` "
        "in order to invalidate the fitness.",
    )

    @property
    def valid(self):
        return len(self.wvalues) != 0

    def dominates(self, other, obj=slice(None)):
        strictly_better = False

        for own, theirs in zip(self.wvalues[obj], other.wvalues[obj]):
            if own > theirs:
                strictly_better = True
            elif own < theirs:
                return False

        return strictly_better

    def __hash__(self):
        return hash(self.wvalues)

    def __lt__(self, other):
        return self.wvalues < other.wvalues

    def __le__(self, other):
        return self.wvalues <= other.wvalues

    def __eq__(self, other):
        return self.wvalues == other.wvalues

    def __ne__(self, other):
        return self.wvalues != other.wvalues

    def __gt__(self, other):
        return self.wvalues > other.wvalues

    def __ge__(self, other):
        return self.wvalues >= other.wvalues

    def __deepcopy__(self, memo):
        copied = self.__class__()
        copied.wvalues = self.wvalues
        return copied

    def __repr__(self):
        values = self.values if self.valid else ()
        return "%s.%s(%r)" % (self.__module__, self.__class__.__name__, values)

    def __str__(self):
        return str(self.values if self.valid else ())