import base64
import hashlib
import hmac
from typing import Any, Optional


class OTP(object):
    """Base class for OTP handlers."""

    def __init__(
        self,
        s: str,
        digits: int = 6,
        digest: Any = hashlib.sha1,
        name: Optional[str] = None,
        issuer: Optional[str] = None,
    ) -> None:
        if digits > 10:
            raise ValueError("digits must be no greater than 10")
        if digest in (hashlib.md5, hashlib.shake_128):
            raise ValueError(
                "selected digest function must generate digest size greater than or equals to 18 bytes"
            )

        self.digits = digits
        self.digest = digest
        self.secret = s
        self.name = name or "Secret"
        self.issuer = issuer

    def generate_otp(self, input: int) -> str:
        """
        :param input: the HMAC counter value to use as the OTP input.
            Usually either the counter, or the computed integer based on the Unix timestamp
        """
        if input < 0:
            raise ValueError("input must be positive integer")

        authenticator = hmac.new(
            self.byte_secret(),
            self.int_to_bytestring(input),
            self.digest,
        )

        if authenticator.digest_size < 18:
            raise ValueError(
                "digest size is lower than 18 bytes, which will trigger error on otp generation"
            )

        result = bytearray(authenticator.digest())
        index = result[-1] & 0x0F
        value = (
            ((result[index] & 0x7F) << 24)
            | ((result[index + 1] & 0xFF) << 16)
            | ((result[index + 2] & 0xFF) << 8)
            | (result[index + 3] & 0xFF)
        )

        return str(10_000_000_000 + value % (10**self.digits))[-self.digits :]

    def byte_secret(self) -> bytes:
        encoded_secret = self.secret
        remainder = len(encoded_secret) % 8
        if remainder:
            encoded_secret += "=" * (8 - remainder)
        return base64.b32decode(encoded_secret, casefold=True)

    @staticmethod
    def int_to_bytestring(i: int, padding: int = 8) -> bytes:
        """
        Turns an integer to the OATH specified
        bytestring, which is fed to the HMAC
        along with the secret
        """
        data = bytearray()
        while i != 0:
            data.append(i & 0xFF)
            i >>= 8
        return bytes(reversed(data.ljust(padding, b"\0")))