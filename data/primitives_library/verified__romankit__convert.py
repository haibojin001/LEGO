_VALUES = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)

_SYMBOL_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


def to_roman(n: int) -> str:
    """Convert an integer in the range 1..3999 to a canonical Roman numeral."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1 or n > 3999:
        raise ValueError("Roman numerals can only represent integers from 1 to 3999")

    result = []
    remaining = n

    for value, symbol in _VALUES:
        while remaining >= value:
            result.append(symbol)
            remaining -= value

    return "".join(result)


def from_roman(s: str) -> int:
    """Convert a canonical Roman numeral to an integer.

    Raises ValueError for empty, malformed, out-of-range, or non-canonical input.
    """
    if not isinstance(s, str) or not s:
        raise ValueError("Roman numeral must be a non-empty string")

    total = 0
    index = 0
    length = len(s)

    while index < length:
        current = s[index]
        if current not in _SYMBOL_VALUES:
            raise ValueError("Malformed Roman numeral")

        current_value = _SYMBOL_VALUES[current]

        if index + 1 < length:
            next_symbol = s[index + 1]
            if next_symbol not in _SYMBOL_VALUES:
                raise ValueError("Malformed Roman numeral")

            next_value = _SYMBOL_VALUES[next_symbol]
            if current_value < next_value:
                total += next_value - current_value
                index += 2
                continue

        total += current_value
        index += 1

    if total < 1 or total > 3999 or to_roman(total) != s:
        raise ValueError("Malformed or non-canonical Roman numeral")

    return total