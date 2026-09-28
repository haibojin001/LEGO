import hashlib
from typing import Any, Optional

from . import utils
from .otp import OTP


class HOTP(OTP):
    """
    Handler for HMAC-based OTP counters.
    """

    def __init__(
        self,
        s: str,
        digits: int = 6,
        digest: Any = None,
        name: Optional[str] = None,
        issuer: Optional[str] = None,
        initial_count: int = 0,
    ) -> None:
        if digest is None:
            digest = hashlib.sha1
        elif digest in (hashlib.md5, hashlib.shake_128):
            raise ValueError(
                "selected digest function must generate digest size greater than or equals to 18 bytes"
            )

        self.initial_count = initial_count
        super().__init__(
            s=s,
            digits=digits,
            digest=digest,
            name=name,
            issuer=issuer,
        )

    def at(self, count: int) -> str:
        return self.generate_otp(self.initial_count + count)

    def verify(self, otp: str, counter: int) -> bool:
        return utils.strings_equal(str(otp), str(self.at(counter)))

    def provisioning_uri(
        self,
        name: Optional[str] = None,
        initial_count: Optional[int] = None,
        issuer_name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        return utils.build_uri(
            self.secret,
            name=name if name else self.name,
            initial_count=(
                self.initial_count if initial_count is None else initial_count
            ),
            issuer=issuer_name if issuer_name else self.issuer,
            algorithm=self.digest().name,
            digits=self.digits,
            **kwargs,
        )