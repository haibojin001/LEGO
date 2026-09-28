import hashlib
from typing import Optional

from ..totp import TOTP

STEAM_CHARS = "23456789BCDFGHJKMNPQRTVWXY"
STEAM_DEFAULT_DIGITS = 5


class Steam(TOTP):
    """Time-based one-time passwords using Steam's character alphabet."""

    def __init__(
        self,
        s: str,
        name: Optional[str] = None,
        issuer: Optional[str] = None,
        interval: int = 30,
        digits: int = 5,
    ) -> None:
        self.interval = interval
        super().__init__(
            s=s,
            digits=10,
            digest=hashlib.sha1,
            name=name,
            issuer=issuer,
        )

    def generate_otp(self, input: int) -> str:
        value = int(super().generate_otp(input))
        alphabet_size = len(STEAM_CHARS)
        result = []

        for _ in range(STEAM_DEFAULT_DIGITS):
            value, index = divmod(value, alphabet_size)
            result.append(STEAM_CHARS[index])

        return "".join(result)