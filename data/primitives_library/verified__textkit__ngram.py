"""N-grams and token frequency utilities."""

from textkit.tokenize import tokens


def _require_positive_n(n: int) -> None:
    """Validate an n-gram size."""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")


def _bincount(indices: list[int], minlength: int | None = None) -> list[int]:
    """Count non-negative integer indices.

    This is a small standard-library adaptation of bincount-style frequency
    counting: determine the required output length, allocate a zero-filled
    counter array, then increment the slot for each observed integer index.
    """
    if minlength is None:
        minlength = len(set(indices))

    counts = [0] * minlength
    for index in indices:
        if index < 0:
            raise ValueError("indices must be non-negative")
        if index >= minlength:
            counts.extend([0] * (index + 1 - len(counts)))
            minlength = len(counts)
        counts[index] += 1
    return counts


def char_ngrams(s: str, n: int) -> list[str]:
    """Return contiguous character n-grams from *s*.

    If ``len(s) < n``, return an empty list.
    """
    _require_positive_n(n)
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def word_ngrams(text: str, n: int) -> list[tuple]:
    """Return contiguous token n-grams from *text*.

    Tokenization is delegated to :func:`textkit.tokenize.tokens`.
    """
    _require_positive_n(n)
    words = tokens(text)
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def freq(text: str) -> dict:
    """Return token frequency counts for *text*.

    Tokenization is delegated to :func:`textkit.tokenize.tokens`.
    """
    words = tokens(text)
    token_to_index: dict[str, int] = {}
    ordered_tokens: list[str] = []
    indices: list[int] = []

    for token in words:
        index = token_to_index.get(token)
        if index is None:
            index = len(ordered_tokens)
            token_to_index[token] = index
            ordered_tokens.append(token)
        indices.append(index)

    counts = _bincount(indices, minlength=len(ordered_tokens))
    return {token: counts[index] for index, token in enumerate(ordered_tokens)}