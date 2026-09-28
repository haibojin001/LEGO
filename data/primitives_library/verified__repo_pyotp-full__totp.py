import calendar
import datetime
import hashlib
import time
from typing import Any, Literal, Optional, Union

from . import utils
from .otp import OTP


class TOTP(OTP):
    """Handler for time-based OTP counters."""

    def __init__(
        self,
        s: str,
        digits: int = 6,
        digest: Any = None,
        name: Optional[str] = None,
        issuer: Optional[str] = None,
        interval: int = 30,
    ) -> None:
        if digest is None:
            digest = hashlib.sha1
        elif digest in [hashlib.md5, hashlib.shake_128]:
            raise ValueError(
                "selected digest function must generate digest size greater than or equals to 18 bytes"
            )

        self.interval = interval
        super().__init__(s=s, digits=digits, digest=digest, name=name, issuer=issuer)

    def at(
        self,
        for_time: Union[int, datetime.datetime],
        counter_offset: int = 0,
    ) -> str:
        if not isinstance(for_time, datetime.datetime):
            for_time = datetime.datetime.fromtimestamp(int(for_time))
        return self.generate_otp(self.timecode(for_time) + counter_offset)

    def now(self) -> str:
        return self.generate_otp(self.timecode(datetime.datetime.now()))

    def verify(
        self,
        otp: str,
        for_time: Optional[datetime.datetime] = None,
        valid_window: int = 0,
    ) -> bool:
        if for_time is None:
            for_time = datetime.datetime.now()

        if valid_window:
            for offset in range(-valid_window, valid_window + 1):
                if utils.strings_equal(str(otp), str(self.at(for_time, offset))):
                    return True
            return False

        return utils.strings_equal(str(otp), str(self.at(for_time)))

    def verify_and_get_timecode(
        self,
        otp: str,
        for_time: Optional[Union[int, datetime.datetime]] = None,
        valid_window: int = 0,
    ) -> int | Literal[False]:
        if for_time is None:
            for_time = datetime.datetime.now()
        elif isinstance(for_time, (float, int)):
            for_time = datetime.datetime.fromtimestamp(int(for_time))

        if valid_window < 0:
            raise ValueError("valid_window cannot be negative")

        for offset in range(-valid_window, valid_window + 1):
            if utils.strings_equal(str(otp), str(self.at(for_time, offset))):
                return self.timecode(for_time) + offset

        return False

    def provisioning_uri(
        self,
        name: Optional[str] = None,
        issuer_name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        return utils.build_uri(
            self.secret,
            name if name else self.name,
            issuer=issuer_name if issuer_name else self.issuer,
            algorithm=self.digest().name,
            digits=self.digits,
            period=self.interval,
            **kwargs,
        )

    def timecode(self, for_time: datetime.datetime) -> int:
        if for_time.tzinfo:
            return int(calendar.timegm(for_time.utctimetuple()) / self.interval)
        return int(time.mktime(for_time.timetuple()) / self.interval)