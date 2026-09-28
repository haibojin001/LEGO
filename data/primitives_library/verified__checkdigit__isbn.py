"""ISBN-10 and ISBN-13 check digit validation."""

__all__ = ["isbn10_valid", "isbn13_valid"]


def _clean_isbn(s: str) -> str:
    """Remove hyphens and whitespace from an ISBN-like string."""
    return "".join(ch for ch in s if ch != "-" and not ch.isspace())


def _is_ascii_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def isbn10_valid(s: str) -> bool:
    """Return True if *s* is a valid ISBN-10."""
    if not isinstance(s, str):
        return False

    cleaned = _clean_isbn(s).upper()

    if len(cleaned) != 10:
        return False

    if not all(_is_ascii_digit(ch) for ch in cleaned[:9]):
        return False

    check = cleaned[9]
    if not (_is_ascii_digit(check) or check == "X"):
        return False

    total = 0
    for index, ch in enumerate(cleaned[:9]):
        total += (ord(ch) - ord("0")) * (10 - index)

    total += 10 if check == "X" else ord(check) - ord("0")

    return total % 11 == 0


def isbn13_valid(s: str) -> bool:
    """Return True if *s* is a valid ISBN-13."""
    if not isinstance(s, str):
        return False

    cleaned = _clean_isbn(s)

    if len(cleaned) != 13:
        return False

    if not all(_is_ascii_digit(ch) for ch in cleaned):
        return False

    total = 0
    for index, ch in enumerate(cleaned[:12]):
        digit = ord(ch) - ord("0")
        total += digit if index % 2 == 0 else digit * 3

    check_digit = ord(cleaned[12]) - ord("0")
    expected = (10 - (total % 10)) % 10

    return check_digit == expected