_SENTINEL = "\x00"

__all__ = ["bwt", "ibwt", "mtf_encode", "mtf_decode"]


def bwt(s: str) -> str:
    """Return the Burrows-Wheeler transform of *s*.

    A NUL sentinel ('\\x00') is appended when it is not already present.
    The returned string is the last column of the lexicographically sorted
    matrix of all cyclic rotations.
    """
    if _SENTINEL not in s:
        s += _SENTINEL

    n = len(s)
    rotations = [s[i:] + s[:i] for i in range(n)]
    rotations.sort()
    return "".join(rotation[-1] for rotation in rotations)


def ibwt(r: str) -> str:
    """Invert a Burrows-Wheeler transform and remove the NUL sentinel."""
    if not r:
        return ""

    n = len(r)
    row = r.index(_SENTINEL)

    counts = {}
    for ch in r:
        counts[ch] = counts.get(ch, 0) + 1

    starts = {}
    total = 0
    for ch in sorted(counts):
        starts[ch] = total
        total += counts[ch]

    seen = {}
    lf = [0] * n
    for i, ch in enumerate(r):
        occ = seen.get(ch, 0)
        lf[i] = starts[ch] + occ
        seen[ch] = occ + 1

    out = []
    for _ in range(n):
        ch = r[row]
        out.append(ch)
        row = lf[row]

    text = "".join(reversed(out))
    if text.endswith(_SENTINEL):
        return text[:-1]
    return text.replace(_SENTINEL, "", 1)


def mtf_encode(data: bytes) -> list:
    """Move-to-front encode bytes using the initial alphabet 0..255."""
    alphabet = list(range(256))
    codes = []

    for value in data:
        index = alphabet.index(value)
        codes.append(index)
        if index:
            alphabet.pop(index)
            alphabet.insert(0, value)

    return codes


def mtf_decode(codes: list) -> bytes:
    """Decode move-to-front codes using the initial alphabet 0..255."""
    alphabet = list(range(256))
    out = bytearray()

    for code in codes:
        if not isinstance(code, int):
            raise TypeError("MTF code must be an integer")
        if code < 0 or code >= len(alphabet):
            raise ValueError("MTF code out of range")

        value = alphabet[code]
        out.append(value)
        if code:
            alphabet.pop(code)
            alphabet.insert(0, value)

    return bytes(out)