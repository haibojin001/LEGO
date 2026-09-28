"""Tokenization and normalization utilities for textkit."""


def tokens(text: str) -> list[str]:
    """Return lowercase tokens split on runs of non-alphanumeric characters."""
    result: list[str] = []
    current: list[str] = []

    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            result.append("".join(current))
            current = []

    if current:
        result.append("".join(current))

    return result


def sentences(text: str) -> list[str]:
    """Return non-empty trimmed sentences split on '.', '!', and '?'."""
    result: list[str] = []
    current: list[str] = []

    for char in text:
        if char in ".!?":
            sentence = "".join(current).strip()
            if sentence:
                result.append(sentence)
            current = []
        else:
            current.append(char)

    sentence = "".join(current).strip()
    if sentence:
        result.append(sentence)

    return result