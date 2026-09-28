"""Euclidean algorithms for greatest common divisors and related operations."""

__all__ = ["gcd", "lcm", "egcd", "modinv"]


def _require_int(name: str, value: int) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return value


def gcd(a: int, b: int) -> int:
    """Return the non-negative greatest common divisor of two integers."""
    a = _require_int("a", a)
    b = _require_int("b", b)

    a = abs(a)
    b = abs(b)

    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Return the non-negative least common multiple of two integers.

    ``lcm(0, anything)`` is defined as ``0``.
    """
    a = _require_int("a", a)
    b = _require_int("b", b)

    if a == 0 or b == 0:
        return 0
    return abs(a // gcd(a, b) * b)


def egcd(a: int, b: int) -> tuple:
    """Return ``(g, x, y)`` such that ``g == gcd(a, b)`` and ``a*x + b*y == g``."""
    a = _require_int("a", a)
    b = _require_int("b", b)

    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1

    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y

    if old_r < 0:
        return -old_r, -old_x, -old_y
    return old_r, old_x, old_y


def modinv(a: int, m: int) -> int:
    """Return the modular inverse of ``a`` modulo ``m``.

    Raises ValueError if ``a`` and ``m`` are not coprime.
    """
    a = _require_int("a", a)
    m = _require_int("m", m)

    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m if m != 0 else x