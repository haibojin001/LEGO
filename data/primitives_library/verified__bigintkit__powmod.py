"""Modular exponentiation for bigintkit."""

from operator import index as _index

__all__ = ["pow_mod"]


def _as_int(value, name):
    try:
        return _index(value)
    except TypeError:
        raise TypeError(
            "pow() 3rd argument not allowed unless all arguments are integers"
        ) from None


def _mod_inverse(value, modulus):
    positive_modulus = modulus if modulus > 0 else -modulus

    r, new_r = positive_modulus, value % positive_modulus
    t, new_t = 0, 1

    while new_r:
        quotient = r // new_r
        r, new_r = new_r, r - quotient * new_r
        t, new_t = new_t, t - quotient * new_t

    if r != 1:
        raise ValueError("base is not invertible for the given modulus")

    return t % modulus


def pow_mod(base: int, exp: int, mod: int) -> int:
    """Return ``base`` raised to ``exp`` modulo ``mod``.

    This implements square-and-multiply modular exponentiation and follows the
    behavior of Python's three-argument ``pow(base, exp, mod)`` for integers.
    """
    base = _as_int(base, "base")
    exp = _as_int(exp, "exp")
    mod = _as_int(mod, "mod")

    if mod == 0:
        raise ValueError("pow() 3rd argument cannot be 0")

    if mod == 1 or mod == -1:
        return 0

    if exp < 0:
        base = _mod_inverse(base, mod)
        exp = -exp

    result = 1 % mod
    factor = base % mod

    while exp:
        if exp & 1:
            result = (result * factor) % mod
        exp >>= 1
        if exp:
            factor = (factor * factor) % mod

    return result