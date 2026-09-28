import base64
import binascii
import re
from typing import Optional, Union

try:
    from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurve
    from cryptography.hazmat.primitives.asymmetric.utils import (
        decode_dss_signature,
        encode_dss_signature,
    )
except ModuleNotFoundError:
    pass


def force_bytes(value: Union[bytes, str]) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError("Expected a string value")


def base64url_decode(input: Union[bytes, str]) -> bytes:
    data = force_bytes(input)
    remainder = len(data) % 4
    if remainder:
        data += b"=" * (4 - remainder)
    return base64.urlsafe_b64decode(data)


def base64url_encode(input: bytes) -> bytes:
    return base64.urlsafe_b64encode(input).replace(b"=", b"")


def to_base64url_uint(val: int, *, bit_length: Optional[int] = None) -> bytes:
    if val < 0:
        raise ValueError("Must be a positive integer")

    data = bytes_from_int(val, bit_length=bit_length)
    if not data:
        data = b"\x00"

    return base64url_encode(data)


def from_base64url_uint(val: Union[bytes, str]) -> int:
    return int.from_bytes(base64url_decode(force_bytes(val)), byteorder="big")


def number_to_bytes(num: int, num_bytes: int) -> bytes:
    text = "%0*x" % (num_bytes * 2, num)
    return binascii.a2b_hex(text.encode("ascii"))


def bytes_to_number(string: bytes) -> int:
    return int(binascii.b2a_hex(string), 16)


def bytes_from_int(val: int, *, bit_length: Optional[int] = None) -> bytes:
    if bit_length is None:
        bit_length = val.bit_length()
    size = (bit_length + 7) // 8
    return val.to_bytes(size, "big", signed=False)


def der_to_raw_signature(der_sig: bytes, curve: "EllipticCurve") -> bytes:
    size = (curve.key_size + 7) // 8
    r_value, s_value = decode_dss_signature(der_sig)
    return number_to_bytes(r_value, size) + number_to_bytes(s_value, size)


def raw_to_der_signature(raw_sig: bytes, curve: "EllipticCurve") -> bytes:
    size = (curve.key_size + 7) // 8

    if len(raw_sig) != size * 2:
        raise ValueError("Invalid signature")

    r_value = bytes_to_number(raw_sig[:size])
    s_value = bytes_to_number(raw_sig[size:])
    return bytes(encode_dss_signature(r_value, s_value))


_PEMS = {
    b"CERTIFICATE",
    b"TRUSTED CERTIFICATE",
    b"PRIVATE KEY",
    b"PUBLIC KEY",
    b"ENCRYPTED PRIVATE KEY",
    b"OPENSSH PRIVATE KEY",
    b"DSA PRIVATE KEY",
    b"RSA PRIVATE KEY",
    b"RSA PUBLIC KEY",
    b"EC PRIVATE KEY",
    b"DH PARAMETERS",
    b"NEW CERTIFICATE REQUEST",
    b"CERTIFICATE REQUEST",
    b"SSH2 PUBLIC KEY",
    b"SSH2 ENCRYPTED PRIVATE KEY",
    b"X509 CRL",
}

_PEM_RE = re.compile(
    b"----[- ]BEGIN ("
    + b"|".join(_PEMS)
    + b""")[- ]----\r?
.+?\r?
----[- ]END \\1[- ]----\r?\n?""",
    re.DOTALL,
)


def is_pem_format(key: bytes) -> bool:
    return bool(_PEM_RE.search(key))


_SSH_KEY_FORMATS = (
    b"ssh-ed25519",
    b"ssh-rsa",
    b"ssh-dss",
    b"ecdsa-sha2-nistp256",
    b"ecdsa-sha2-nistp384",
    b"ecdsa-sha2-nistp521",
)


def is_ssh_key(key: bytes) -> bool:
    return key.startswith(_SSH_KEY_FORMATS)