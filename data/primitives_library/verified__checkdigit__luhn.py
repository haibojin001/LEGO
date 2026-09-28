__all__ = ["luhn_valid", "luhn_check_digit"]


def _is_ascii_digits(value: str) -> bool:
    return all("0" <= char <= "9" for char in value)


def _luhn_value(digit: int, doubled: bool) -> int:
    if doubled:
        digit *= 2
        if digit > 9:
            digit -= 9
    return digit


def luhn_valid(number: str) -> bool:
    """Return True if *number* is valid under the Luhn mod-10 algorithm.

    The input must contain ASCII digits only. Empty strings and strings
    containing non-digits are considered invalid.
    """
    if not isinstance(number, str) or not number or not _is_ascii_digits(number):
        return False

    total = 0
    doubled = False

    for char in reversed(number):
        total += _luhn_value(ord(char) - ord("0"), doubled)
        doubled = not doubled

    return total % 10 == 0


def luhn_check_digit(number_without_check: str) -> int:
    """Return the Luhn check digit for *number_without_check*.

    The input must contain ASCII digits only. The returned integer is the
    single digit that makes ``number_without_check + str(digit)`` valid.
    """
    if not isinstance(number_without_check, str):
        raise TypeError("number_without_check must be a str")
    if not _is_ascii_digits(number_without_check):
        raise ValueError("number_without_check must contain digits only")

    total = 0
    doubled = True

    for char in reversed(number_without_check):
        total += _luhn_value(ord(char) - ord("0"), doubled)
        doubled = not doubled

    return (10 - (total % 10)) % 10