def lcm(a: int, b: int) -> int:
    """Return the least common multiple of two integers.

    ``lcm(0, anything)`` is defined as ``0``. The result is always
    non-negative.

    Raises ``InvalidInputError`` if either argument is not an int.
    """
    first = require_int("a", a)
    second = require_int("b", b)
    if first == 0 or second == 0:
        return 0
    return abs(first // gcd(first, second) * second)