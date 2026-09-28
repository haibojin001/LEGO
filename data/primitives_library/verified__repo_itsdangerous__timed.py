from __future__ import annotations

import collections.abc as cabc
import time
import typing as t
from datetime import datetime
from datetime import timezone

from .encoding import base64_decode
from .encoding import base64_encode
from .encoding import bytes_to_int
from .encoding import int_to_bytes
from .encoding import want_bytes
from .exc import BadSignature
from .exc import BadTimeSignature
from .exc import SignatureExpired
from .serializer import _TSerialized
from .serializer import Serializer
from .signer import Signer


class TimestampSigner(Signer):
    """A signer that stores a timestamp alongside signed data."""

    def get_timestamp(self) -> int:
        return int(time.time())

    def timestamp_to_datetime(self, ts: int) -> datetime:
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    def sign(self, value: str | bytes) -> bytes:
        value = want_bytes(value)
        sep = want_bytes(self.sep)
        timestamp = base64_encode(int_to_bytes(self.get_timestamp()))
        value = value + sep + timestamp
        return value + sep + self.get_signature(value)

    @t.overload
    def unsign(
        self,
        signed_value: str | bytes,
        max_age: int | None = None,
        return_timestamp: t.Literal[False] = False,
    ) -> bytes: ...

    @t.overload
    def unsign(
        self,
        signed_value: str | bytes,
        max_age: int | None = None,
        return_timestamp: t.Literal[True] = True,
    ) -> tuple[bytes, datetime]: ...

    def unsign(
        self,
        signed_value: str | bytes,
        max_age: int | None = None,
        return_timestamp: bool = False,
    ) -> bytes | tuple[bytes, datetime]:
        try:
            result = super().unsign(signed_value)
            signature_error = None
        except BadSignature as error:
            signature_error = error
            result = error.payload or b""

        sep = want_bytes(self.sep)

        if sep not in result:
            if signature_error is not None:
                raise signature_error

            raise BadTimeSignature("timestamp missing", payload=result)

        value, timestamp_bytes = result.rsplit(sep, 1)
        timestamp: int | None = None
        date_signed: datetime | None = None

        try:
            timestamp = bytes_to_int(base64_decode(timestamp_bytes))
        except Exception:
            pass

        if signature_error is not None:
            if timestamp is not None:
                try:
                    date_signed = self.timestamp_to_datetime(timestamp)
                except (ValueError, OSError, OverflowError) as error:
                    raise BadTimeSignature(
                        "Malformed timestamp", payload=value
                    ) from error

            raise BadTimeSignature(
                str(signature_error), payload=value, date_signed=date_signed
            )

        if timestamp is None:
            raise BadTimeSignature("Malformed timestamp", payload=value)

        if max_age is not None:
            age = self.get_timestamp() - timestamp

            if age > max_age:
                raise SignatureExpired(
                    f"Signature age {age} > {max_age} seconds",
                    payload=value,
                    date_signed=self.timestamp_to_datetime(timestamp),
                )

            if age < 0:
                raise SignatureExpired(
                    f"Signature age {age} < 0 seconds",
                    payload=value,
                    date_signed=self.timestamp_to_datetime(timestamp),
                )

        if return_timestamp:
            return value, self.timestamp_to_datetime(timestamp)

        return value

    def validate(self, signed_value: str | bytes, max_age: int | None = None) -> bool:
        try:
            self.unsign(signed_value, max_age=max_age)
        except BadSignature:
            return False

        return True


class TimedSerializer(Serializer[_TSerialized]):
    """A serializer using timestamp-aware signing."""

    default_signer: type[TimestampSigner] = TimestampSigner

    def iter_unsigners(
        self, salt: str | bytes | None = None
    ) -> cabc.Iterator[TimestampSigner]:
        return t.cast(cabc.Iterator[TimestampSigner], super().iter_unsigners(salt))

    def loads(
        self,
        s: str | bytes,
        max_age: int | None = None,
        return_timestamp: bool = False,
        salt: str | bytes | None = None,
    ) -> t.Any:
        s = want_bytes(s)
        last_exception = None

        for signer in self.iter_unsigners(salt):
            try:
                value, timestamp = signer.unsign(
                    s, max_age=max_age, return_timestamp=True
                )
                payload = self.load_payload(value)

                if return_timestamp:
                    return payload, timestamp

                return payload
            except SignatureExpired:
                raise
            except BadSignature as error:
                last_exception = error

        raise t.cast(BadSignature, last_exception)

    def loads_unsafe(
        self,
        s: str | bytes,
        max_age: int | None = None,
        salt: str | bytes | None = None,
    ) -> tuple[bool, t.Any]:
        return self._loads_unsafe_impl(s, salt, load_kwargs={"max_age": max_age})