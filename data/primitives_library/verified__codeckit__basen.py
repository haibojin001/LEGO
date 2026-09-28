"""Base16, Base32, and Base64 codecs implemented without external helpers."""

__all__ = [
    "b64encode",
    "b64decode",
    "b16encode",
    "b16decode",
    "b32encode",
    "b32decode",
]

_B64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_B32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
_B16_ALPHABET = "0123456789ABCDEF"

_B64_DECODE = [-1] * 128
for _idx, _ch in enumerate(_B64_ALPHABET):
    _B64_DECODE[ord(_ch)] = _idx

_B32_DECODE = [-1] * 128
for _idx, _ch in enumerate(_B32_ALPHABET):
    _B32_DECODE[ord(_ch)] = _idx

_B16_DECODE = [-1] * 128
for _idx, _ch in enumerate(_B16_ALPHABET):
    _B16_DECODE[ord(_ch)] = _idx

del _idx, _ch


def _as_bytes(data: bytes) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    try:
        return memoryview(data).tobytes()
    except TypeError:
        raise TypeError("a bytes-like object is required") from None


def _to_ascii_text(s: str) -> str:
    if isinstance(s, str):
        for ch in s:
            if ord(ch) > 127:
                raise ValueError("string argument should contain only ASCII characters")
        return s

    if isinstance(s, (bytes, bytearray)):
        raw = bytes(s)
    else:
        try:
            raw = memoryview(s).tobytes()
        except TypeError:
            raise TypeError("argument should be a str or bytes-like object") from None

    try:
        return raw.decode("ascii")
    except UnicodeDecodeError:
        raise ValueError("string argument should contain only ASCII characters") from None


def b64encode(data: bytes) -> str:
    """Encode bytes using RFC 4648 standard Base64 with '=' padding."""
    data = _as_bytes(data)
    out = []

    full_len = (len(data) // 3) * 3

    for i in range(0, full_len, 3):
        n = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
        out.append(_B64_ALPHABET[(n >> 18) & 0x3F])
        out.append(_B64_ALPHABET[(n >> 12) & 0x3F])
        out.append(_B64_ALPHABET[(n >> 6) & 0x3F])
        out.append(_B64_ALPHABET[n & 0x3F])

    rem = len(data) - full_len
    if rem == 1:
        b0 = data[full_len]
        out.append(_B64_ALPHABET[b0 >> 2])
        out.append(_B64_ALPHABET[(b0 & 0x03) << 4])
        out.append("=")
        out.append("=")
    elif rem == 2:
        b0 = data[full_len]
        b1 = data[full_len + 1]
        out.append(_B64_ALPHABET[b0 >> 2])
        out.append(_B64_ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)])
        out.append(_B64_ALPHABET[(b1 & 0x0F) << 2])
        out.append("=")

    return "".join(out)


def b64decode(s: str) -> bytes:
    """Decode standard Base64 text.

    Like Python's default base64.b64decode, non-alphabet ASCII characters are
    ignored. Padding is required for incomplete final quanta.
    """
    text = _to_ascii_text(s)

    chars = []
    for ch in text:
        o = ord(ch)
        if ch == "=" or _B64_DECODE[o] != -1:
            chars.append(ch)

    while chars and chars[0] == "=":
        del chars[0]

    if not chars:
        return b""

    first_pad = -1
    for i, ch in enumerate(chars):
        if ch == "=":
            first_pad = i
            break

    out = bytearray()

    if first_pad == -1:
        if len(chars) % 4 != 0:
            raise ValueError("Incorrect padding")
        body = chars
        final = []
        pad_count = 0
    else:
        body = chars[:first_pad]
        suffix = chars[first_pad:]
        for ch in suffix:
            if ch != "=":
                raise ValueError("Discontinuous padding not allowed")

        rem = len(body) % 4
        pad_count = len(suffix)

        if rem == 0:
            final = []
        elif rem == 2:
            if pad_count < 2:
                raise ValueError("Incorrect padding")
            final = body[-2:]
            body = body[:-2]
        elif rem == 3:
            if pad_count < 1:
                raise ValueError("Incorrect padding")
            final = body[-3:]
            body = body[:-3]
        else:
            raise ValueError("Incorrect padding")

    for i in range(0, len(body), 4):
        try:
            v0 = _B64_DECODE[ord(body[i])]
            v1 = _B64_DECODE[ord(body[i + 1])]
            v2 = _B64_DECODE[ord(body[i + 2])]
            v3 = _B64_DECODE[ord(body[i + 3])]
        except IndexError:
            raise ValueError("Incorrect padding") from None

        if v0 < 0 or v1 < 0 or v2 < 0 or v3 < 0:
            raise ValueError("Non-base64 digit found")

        n = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3
        out.append((n >> 16) & 0xFF)
        out.append((n >> 8) & 0xFF)
        out.append(n & 0xFF)

    if final:
        v0 = _B64_DECODE[ord(final[0])]
        v1 = _B64_DECODE[ord(final[1])]
        if v0 < 0 or v1 < 0:
            raise ValueError("Non-base64 digit found")

        out.append(((v0 << 2) | (v1 >> 4)) & 0xFF)

        if len(final) == 3:
            v2 = _B64_DECODE[ord(final[2])]
            if v2 < 0:
                raise ValueError("Non-base64 digit found")
            out.append((((v1 & 0x0F) << 4) | (v2 >> 2)) & 0xFF)

    return bytes(out)


def b16encode(data: bytes) -> str:
    """Encode bytes as uppercase hexadecimal Base16."""
    data = _as_bytes(data)
    out = []

    for b in data:
        out.append(_B16_ALPHABET[b >> 4])
        out.append(_B16_ALPHABET[b & 0x0F])

    return "".join(out)


def b16decode(s: str) -> bytes:
    """Decode uppercase RFC 4648 Base16 text."""
    text = _to_ascii_text(s)

    if len(text) % 2:
        raise ValueError("Odd-length string")

    out = bytearray()

    for i in range(0, len(text), 2):
        hi = _B16_DECODE[ord(text[i])]
        lo = _B16_DECODE[ord(text[i + 1])]
        if hi < 0 or lo < 0:
            raise ValueError("Non-base16 digit found")
        out.append((hi << 4) | lo)

    return bytes(out)


def b32encode(data: bytes) -> str:
    """Encode bytes using RFC 4648 Base32 with '=' padding."""
    data = _as_bytes(data)
    out = []

    for i in range(0, len(data), 5):
        chunk = data[i:i + 5]
        value = 0
        for b in chunk:
            value = (value << 8) | b

        bits = len(chunk) * 8
        chars_needed = (bits + 4) // 5

        for _ in range(chars_needed):
            if bits >= 5:
                bits -= 5
                idx = (value >> bits) & 0x1F
            else:
                idx = (value << (5 - bits)) & 0x1F
                bits = 0
            out.append(_B32_ALPHABET[idx])

        while len(out) % 8:
            out.append("=")

    return "".join(out)


def b32decode(s: str) -> bytes:
    """Decode RFC 4648 Base32 text with standard padding."""
    text = _to_ascii_text(s)

    if len(text) % 8 != 0:
        raise ValueError("Incorrect padding")

    if not text:
        return b""

    pad = len(text) - len(text.rstrip("="))
    if pad not in (0, 1, 3, 4, 6):
        raise ValueError("Incorrect padding")

    body = text[:-pad] if pad else text
    if "=" in body:
        raise ValueError("Incorrect padding")

    out = bytearray()
    buffer = 0
    bits = 0

    for ch in body:
        val = _B32_DECODE[ord(ch)]
        if val < 0:
            raise ValueError("Non-base32 digit found")

        buffer = (buffer << 5) | val
        bits += 5

        while bits >= 8:
            bits -= 8
            out.append((buffer >> bits) & 0xFF)

    return bytes(out)