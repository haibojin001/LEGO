from __future__ import annotations

import collections.abc as cabc
import hashlib
import hmac
import typing as t

from .encoding import _base64_alphabet
from .encoding import base64_decode
from .encoding import base64_encode
from .encoding import want_bytes
from .exc import BadSignature


class SigningAlgorithm:
    """Base interface for signature implementations."""

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        """Produce a signature for a key and value."""
        raise NotImplementedError()

    def verify_signature(self, key: bytes, value: bytes, sig: bytes) -> bool:
        """Return whether a signature matches a key and value."""
        return hmac.compare_digest(sig, self.get_signature(key, value))


class NoneAlgorithm(SigningAlgorithm):
    """A signing algorithm that always emits an empty signature."""

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        return b""


def _lazy_sha1(string: bytes = b"") -> t.Any:
    """Look up SHA-1 when called rather than during import."""
    return hashlib.sha1(string)


class HMACAlgorithm(SigningAlgorithm):
    """A signing algorithm backed by HMAC."""

    default_digest_method: t.Any = staticmethod(_lazy_sha1)

    def __init__(self, digest_method: t.Any = None):
        if digest_method is None:
            digest_method = self.default_digest_method

        self.digest_method: t.Any = digest_method

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        return hmac.new(key, msg=value, digestmod=self.digest_method).digest()


def _make_keys_list(
    secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
) -> list[bytes]:
    if isinstance(secret_key, (str, bytes)):
        return [want_bytes(secret_key)]

    return [want_bytes(item) for item in secret_key]  # pyright: ignore


class Signer:
    """Create and verify signatures for byte strings."""

    default_digest_method: t.Any = staticmethod(_lazy_sha1)
    default_key_derivation: str = "django-concat"

    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous.Signer",
        sep: str | bytes = b".",
        key_derivation: str | None = None,
        digest_method: t.Any | None = None,
        algorithm: SigningAlgorithm | None = None,
    ):
        self.secret_keys: list[bytes] = _make_keys_list(secret_key)
        self.sep: bytes = want_bytes(sep)

        if self.sep in _base64_alphabet:
            raise ValueError(
                "The given separator cannot be used because it may be"
                " contained in the signature itself. ASCII letters,"
                " digits, and '-_=' must not be used."
            )

        if salt is None:
            salt = b"itsdangerous.Signer"

        self.salt: bytes = want_bytes(salt)

        if key_derivation is None:
            key_derivation = self.default_key_derivation

        self.key_derivation: str = key_derivation

        if digest_method is None:
            digest_method = self.default_digest_method

        self.digest_method: t.Any = digest_method

        if algorithm is None:
            algorithm = HMACAlgorithm(self.digest_method)

        self.algorithm: SigningAlgorithm = algorithm

    @property
    def secret_key(self) -> bytes:
        """Return the newest configured secret key."""
        return self.secret_keys[-1]

    def derive_key(self, secret_key: str | bytes | None = None) -> bytes:
        """Derive the signing key from a secret key and the configured salt."""
        if secret_key is None:
            secret_key = self.secret_keys[-1]
        else:
            secret_key = want_bytes(secret_key)

        if self.key_derivation == "concat":
            return t.cast(bytes, self.digest_method(self.salt + secret_key).digest())

        if self.key_derivation == "django-concat":
            return t.cast(
                bytes, self.digest_method(self.salt + b"signer" + secret_key).digest()
            )

        if self.key_derivation == "hmac":
            mac = hmac.new(secret_key, digestmod=self.digest_method)
            mac.update(self.salt)
            return mac.digest()

        if self.key_derivation == "none":
            return secret_key

        raise TypeError("Unknown key derivation method")

    def get_signature(self, value: str | bytes) -> bytes:
        """Return the encoded signature for a value."""
        value = want_bytes(value)
        key = self.derive_key()
        return base64_encode(self.algorithm.get_signature(key, value))

    def sign(self, value: str | bytes) -> bytes:
        """Append a signature to a value."""
        value = want_bytes(value)
        return value + self.sep + self.get_signature(value)

    def verify_signature(self, value: str | bytes, sig: str | bytes) -> bool:
        """Check a value and its encoded signature."""
        try:
            sig = base64_decode(sig)
        except Exception:
            return False

        value = want_bytes(value)

        for secret_key in reversed(self.secret_keys):
            key = self.derive_key(secret_key)

            if self.algorithm.verify_signature(key, value, sig):
                return True

        return False

    def unsign(self, signed_value: str | bytes) -> bytes:
        """Verify a signed value and return its original portion."""
        signed_value = want_bytes(signed_value)

        if self.sep not in signed_value:
            raise BadSignature(f"No {self.sep!r} found in value")

        value, sig = signed_value.rsplit(self.sep, 1)

        if self.verify_signature(value, sig):
            return value

        raise BadSignature(f"Signature {sig!r} does not match", payload=value)