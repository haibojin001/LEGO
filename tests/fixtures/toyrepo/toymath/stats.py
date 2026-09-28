"""Summary statistics built on ops."""
from .ops import add


def total(xs):
    """Sum of an iterable of numbers; 0 for an empty iterable."""
    t = 0
    for x in xs:
        t = add(t, x)
    return t


def mean(xs):
    """Arithmetic mean; raises ValueError on an empty sequence."""
    xs = list(xs)
    if not xs:
        raise ValueError("mean of empty sequence")
    return total(xs) / len(xs)
