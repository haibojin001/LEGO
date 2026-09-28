"""Text processing utilities."""

from __future__ import annotations

import re
from typing import List


def chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]

    result: List[str] = []
    position = 0
    text_length = len(text)

    while position < text_length:
        limit = position + max_chars

        if limit < text_length:
            boundary = text.rfind(".", position, limit)
            if boundary > position + (max_chars // 2):
                limit = boundary + 1

        result.append(text[position:limit].strip())
        position = limit - overlap

    return result


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()