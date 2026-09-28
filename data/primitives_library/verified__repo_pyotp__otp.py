import base64
import hashlib
import hmac
from typing import Any, Optional


class OTP(object):
    """Base class for one-time-password generators."""

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

        message = self.int_to_bytestring(input)
        signer = hmac.new(self.byte_secret(), message, self.digest)

        if signer.digest_size < 18:
            raise ValueError(
                "digest size is lower than 18 bytes, which will trigger error on otp generation"
            )

        raw = bytearray(signer.digest())
        position = raw[-1] & 15
        value = (
            ((raw[position] & 127) << 24)
            | ((raw[position + 1] & 255) << 16)
            | ((raw[position + 2] & 255) << 8)
            | (raw[position + 3] & 255)
        )
        return str(10_000_000_000 + value % (10 ** self.digits))[-self.digits:]

    def byte_secret(self) -> bytes:
        encoded = self.secret
        extra = len(encoded) % 8
        if extra:
            encoded += "=" * (8 - extra)
        return base64.b32decode(encoded, casefold=True)

    @staticmethod
    def int_to_bytestring(i: int, padding: int = 8) -> bytes:
        """
        Turns an integer to the OATH specified
        bytestring, which is fed to the HMAC
        along with the secret
        """
        output = bytearray()
        while i != 0:
            output.append(i & 255)
            i >>= 8
        return bytes(reversed(output.ljust(padding, b"\0")))