import bisect


def _encode_range(start: int, end: int) -> int:
    return (start << 32) | end


def _decode_range(r: int) -> tuple[int, int]:
    mask = (1 << 32) - 1
    return r >> 32, r & mask


def intranges_from_list(list_: list[int]) -> tuple[int, ...]:
    """Convert integers into encoded half-open consecutive ranges."""
    values = sorted(list_)
    encoded = []
    range_start = 0

    for index, value in enumerate(values):
        has_successor = index + 1 < len(values)
        if has_successor and value == values[index + 1] - 1:
            continue

        encoded.append(_encode_range(values[range_start], value + 1))
        range_start = index + 1

    return tuple(encoded)


def intranges_contain(int_: int, ranges: tuple[int, ...]) -> bool:
    """Return whether an integer belongs to one of the encoded ranges."""
    insertion_point = bisect.bisect_left(ranges, _encode_range(int_, 0))

    if insertion_point:
        start, stop = _decode_range(ranges[insertion_point - 1])
        if start <= int_ < stop:
            return True

    if insertion_point != len(ranges):
        start, _ = _decode_range(ranges[insertion_point])
        if start == int_:
            return True

    return False