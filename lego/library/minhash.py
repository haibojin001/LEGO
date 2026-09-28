"""MinHash over token shingles, for near-duplicate donor filtering."""

from __future__ import annotations

import hashlib
import re

_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")
NUM_PERM = 128
_MASK = (1 << 61) - 1


def shingles(text: str, k: int = 5) -> set[str]:
    toks = _TOK.findall(text or "")
    return {" ".join(toks[i:i + k]) for i in range(max(len(toks) - k + 1, 1))}


def _h(s: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{s}".encode(),
                                          digest_size=8).digest(), "big") & _MASK


def signature(text: str, num_perm: int = NUM_PERM) -> list[int]:
    sh = shingles(text)
    return [min(_h(s, i) for s in sh) for i in range(num_perm)]


def similarity(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)
