"""Text embedders shared by CodeFace retrieval and the File RAG control.

The File RAG control must use *the same* embedder and depth as CodeFace
retrieval, so both go through ``make_embedder(spec)``:

  tfidf               pure-Python TF-IDF over identifier-split tokens (default;
                      deterministic, no dependencies)
  st:<model name>     sentence-transformers model (``pip install
                      sentence-transformers``), e.g. ``st:BAAI/bge-small-en-v1.5``
"""

from __future__ import annotations

import math
import re
from collections import Counter

_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")
_STOP = {"self", "cls", "the", "and", "for", "from", "with", "import", "def",
         "class", "return", "none", "true", "false", "str", "int", "list",
         "dict", "args", "kwargs", "if", "else", "in", "is", "not", "of", "to",
         "a", "an", "py", "init"}


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _SPLIT.split(text or "")
            if len(t) > 1 and t.lower() not in _STOP]


class TfIdf:
    name = "tfidf"

    def fit(self, docs: list[str]):
        self.df = Counter()
        self.vecs = []
        toks = [Counter(tokens(d)) for d in docs]
        for c in toks:
            self.df.update(c.keys())
        self.n = max(len(docs), 1)
        self.vecs = [self._vec(c) for c in toks]
        return self

    def _vec(self, c: Counter) -> dict:
        v = {t: (1 + math.log(f)) * (1 + math.log((1 + self.n) / (1 + self.df.get(t, 0))))
             for t, f in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def query(self, text: str, k: int) -> list[tuple[int, float]]:
        q = self._vec(Counter(tokens(text)))
        scores = []
        for i, v in enumerate(self.vecs):
            s = sum(w * v.get(t, 0.0) for t, w in q.items())
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


class SentenceTransformer:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer as ST
        self.name = "st:" + model
        self.m = ST(model)

    def fit(self, docs: list[str]):
        self.E = self.m.encode(docs, normalize_embeddings=True,
                               batch_size=64, show_progress_bar=False)
        return self

    def query(self, text: str, k: int) -> list[tuple[int, float]]:
        import numpy as np
        q = self.m.encode([text], normalize_embeddings=True)[0]
        s = self.E @ q
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]


def make_embedder(spec: str = "tfidf"):
    if spec == "tfidf":
        return TfIdf()
    if spec.startswith("st:"):
        return SentenceTransformer(spec[3:])
    raise ValueError(f"unknown embedder {spec!r}")
