"""Caesar and Vigenere cipher helpers."""

__all__ = [
    "caesar_encrypt",
    "caesar_decrypt",
    "vigenere_encrypt",
    "vigenere_decrypt",
]


_UPPER_A = ord("A")
_UPPER_Z = ord("Z")
_LOWER_A = ord("a")
_LOWER_Z = ord("z")
_ALPHABET_SIZE = 26


def _is_upper_ascii_letter(ch: str) -> bool:
    code = ord(ch)
    return _UPPER_A <= code <= _UPPER_Z


def _is_lower_ascii_letter(ch: str) -> bool:
    code = ord(ch)
    return _LOWER_A <= code <= _LOWER_Z


def _is_ascii_letter(ch: str) -> bool:
    return _is_upper_ascii_letter(ch) or _is_lower_ascii_letter(ch)


def _shift_char(ch: str, shift: int) -> str:
    code = ord(ch)

    if _UPPER_A <= code <= _UPPER_Z:
        return chr(_UPPER_A + ((code - _UPPER_A + shift) % _ALPHABET_SIZE))

    if _LOWER_A <= code <= _LOWER_Z:
        return chr(_LOWER_A + ((code - _LOWER_A + shift) % _ALPHABET_SIZE))

    return ch


def _key_shift(ch: str) -> int:
    code = ord(ch)

    if _UPPER_A <= code <= _UPPER_Z:
        return code - _UPPER_A

    if _LOWER_A <= code <= _LOWER_Z:
        return code - _LOWER_A

    raise ValueError("key must contain only ASCII letters A-Z and a-z")


def _validate_key(key: str) -> None:
    if not key:
        raise ValueError("key must not be empty")

    for ch in key:
        if not _is_ascii_letter(ch):
            raise ValueError("key must contain only ASCII letters A-Z and a-z")


def caesar_encrypt(text: str, shift: int) -> str:
    """Encrypt text with a Caesar cipher.

    ASCII letters A-Z and a-z are shifted by ``shift`` positions. All other
    characters are returned unchanged.
    """
    shift %= _ALPHABET_SIZE
    return "".join(_shift_char(ch, shift) for ch in text)


def caesar_decrypt(text: str, shift: int) -> str:
    """Decrypt text encrypted with a Caesar cipher."""
    return caesar_encrypt(text, -shift)


def vigenere_encrypt(text: str, key: str) -> str:
    """Encrypt text with a Vigenere cipher.

    The key must contain only ASCII letters A-Z and a-z. Non-letter characters
    in ``text`` are passed through unchanged and do not advance the key.
    """
    _validate_key(key)

    result = []
    key_index = 0
    key_length = len(key)

    for ch in text:
        if _is_ascii_letter(ch):
            shift = _key_shift(key[key_index % key_length])
            result.append(_shift_char(ch, shift))
            key_index += 1
        else:
            result.append(ch)

    return "".join(result)


def vigenere_decrypt(text: str, key: str) -> str:
    """Decrypt text encrypted with a Vigenere cipher."""
    _validate_key(key)

    result = []
    key_index = 0
    key_length = len(key)

    for ch in text:
        if _is_ascii_letter(ch):
            shift = -_key_shift(key[key_index % key_length])
            result.append(_shift_char(ch, shift))
            key_index += 1
        else:
            result.append(ch)

    return "".join(result)