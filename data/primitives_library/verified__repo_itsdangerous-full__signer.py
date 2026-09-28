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
    """Interface for objects capable of creating and checking signatures."""

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        """Create a signature from a key and value."""
        raise NotImplementedError()

    def verify_signature(self, key: bytes, value: bytes, sig: bytes) -> bool:
        """Check whether a signature is valid for a key and value."""
        expected = self.get_signature(key, value)
        return hmac.compare_digest(sig, expected)


class NoneAlgorithm(SigningAlgorithm):
    """Algorithm implementation that creates no signature."""

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        return b""


def _lazy_sha1(string: bytes = b"") -> t.Any:
    """Resolve SHA-1 only when it is actually needed."""
    return hashlib.sha1(string)


class HMACAlgorithm(SigningAlgorithm):
    """Signature algorithm based on HMAC."""

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

    return [want_bytes(key) for key in secret_key]  # pyright: ignore


class Signer:
    """Sign byte strings and later verify their integrity."""

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
            algorithm = HMACAlgorithm(digest_method)

        self.algorithm: SigningAlgorithm = algorithm

    @property
    def secret_key(self) -> bytes:
        """Return the most recent configured secret key."""
        return self.secret_keys[-1]

    def derive_key(self, secret_key: str | bytes | None = None) -> bytes:
        """Derive the signing key from a secret key and the configured salt."""
        if secret_key is None:
            key = self.secret_keys[-1]
        else:
            key = want_bytes(secret_key)

        if self.key_derivation == "concat":
            return t.cast(bytes, self.digest_method(self.salt + key).digest())

        if self.key_derivation == "django-concat":
            material = self.salt + b"signer" + key
            return t.cast(bytes, self.digest_method(material).digest())

        if self.key_derivation == "hmac":
            mac = hmac.new(key, digestmod=self.digest_method)
            mac.update(self.salt)
            return mac.digest()

        if self.key_derivation == "none":
            return key

        raise TypeError("Unknown key derivation method")

    def get_signature(self, value: str | bytes) -> bytes:
        """Return the encoded signature for a value."""
        data = want_bytes(value)
        signing_key = self.derive_key()
        signature = self.algorithm.get_signature(signing_key, data)
        return base64_encode(signature)

    def sign(self, value: str | bytes) -> bytes:
        """Append a signature to a value."""
        data = want_bytes(value)
        return data + self.sep + self.get_signature(data)

    def verify_signature(self, value: str | bytes, sig: str | bytes) -> bool:
        """Return whether a signature is valid for a value."""
        try:
            decoded_sig = base64_decode(sig)
        except Exception:
            return False

        data = want_bytes(value)

        for secret_key in reversed(self.secret_keys):
            derived_key = self.derive_key(secret_key)

            if self.algorithm.verify_signature(derived_key, data, decoded_sig):
                return True

        return False

    def unsign(self, signed_value: str | bytes) -> bytes:
        """Verify and remove a signature from a signed value."""
        signed_value = want_bytes(signed_value)

        if self.sep not in signed_value:
            raise BadSignature(f"No {self.sep!r} found in value")

        value, sig = signed_value.rsplit(self.sep, 1)

        if self.verify_signature(value, sig):
            return value

        raise BadSignature(f"Signature {sig!r} does not match", payload=value)

    def validate(self, signed_value: str | bytes) -> bool:
        """Return whether a signed value has a valid signature."""
        try:
            self.unsign(signed_value)
        except BadSignature:
            return False

        return True