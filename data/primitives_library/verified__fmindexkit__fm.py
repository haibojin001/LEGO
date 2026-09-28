__all__ = ["FMIndex"]


_MAX_UNICODE_CODE_POINT = 0x10FFFF


def _choose_sentinel(text: str) -> str:
    if "\0" not in text:
        return "\0"

    used = set(text)
    for code_point in range(1, _MAX_UNICODE_CODE_POINT + 1):
        char = chr(code_point)
        if char not in used:
            return char

    raise ValueError("cannot build FMIndex: input text contains every Unicode character")


def _build_suffix_array(text: str) -> list[int]:
    n = len(text)
    if n == 0:
        return []
    if n == 1:
        return [0]

    suffix_array = list(range(n))
    ranks = [ord(char) for char in text]
    temp = [0] * n
    step = 1

    while step < n:
        suffix_array.sort(
            key=lambda index: (
                ranks[index],
                ranks[index + step] if index + step < n else -1,
            )
        )

        temp[suffix_array[0]] = 0
        classes = 0
        previous = suffix_array[0]

        for current in suffix_array[1:]:
            previous_key = (
                ranks[previous],
                ranks[previous + step] if previous + step < n else -1,
            )
            current_key = (
                ranks[current],
                ranks[current + step] if current + step < n else -1,
            )

            if current_key != previous_key:
                classes += 1

            temp[current] = classes
            previous = current

        ranks, temp = temp, ranks

        if classes == n - 1:
            break

        step *= 2

    return suffix_array


def _lower_bound(values: list[int], target: int) -> int:
    left = 0
    right = len(values)

    while left < right:
        middle = (left + right) // 2
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle

    return left


class FMIndex:
    def __init__(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be a str")

        self.text = text
        self._sentinel = _choose_sentinel(text)
        self._indexed_text = text + self._sentinel
        self._length = len(self._indexed_text)

        suffix_array = _build_suffix_array(self._indexed_text)

        bwt_chars = []
        for suffix_start in suffix_array:
            if suffix_start == 0:
                bwt_chars.append(self._indexed_text[-1])
            else:
                bwt_chars.append(self._indexed_text[suffix_start - 1])

        self.bwt = "".join(bwt_chars)

        counts = {}
        for char in self._indexed_text:
            counts[char] = counts.get(char, 0) + 1

        total = 0
        self.C = {}
        for char in sorted(counts):
            self.C[char] = total
            total += counts[char]

        rank_positions = {}
        for index, char in enumerate(self.bwt):
            if char not in rank_positions:
                rank_positions[char] = []
            rank_positions[char].append(index)

        self._rank_positions = rank_positions

    def _rank(self, char: str, offset: int) -> int:
        positions = self._rank_positions.get(char)
        if positions is None:
            return 0
        return _lower_bound(positions, offset)

    def count(self, pattern: str) -> int:
        if not isinstance(pattern, str):
            raise TypeError("pattern must be a str")

        if self._sentinel in pattern:
            return 0

        left = 0
        right = self._length

        for char in reversed(pattern):
            base = self.C.get(char)
            if base is None:
                return 0

            left = base + self._rank(char, left)
            right = base + self._rank(char, right)

            if left >= right:
                return 0

        return right - left

    def contains(self, pattern: str) -> bool:
        return self.count(pattern) > 0