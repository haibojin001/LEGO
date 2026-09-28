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
    """A signer that embeds a timestamp in each signed value."""

    def get_timestamp(self) -> int:
        """Return the current Unix timestamp."""
        return int(time.time())

    def timestamp_to_datetime(self, ts: int) -> datetime:
        """Convert a Unix timestamp to a timezone-aware UTC datetime."""
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    def sign(self, value: str | bytes) -> bytes:
        """Add a timestamp to a value and sign the resulting bytes."""
        value_bytes = want_bytes(value)
        separator = want_bytes(self.sep)
        timestamp = base64_encode(int_to_bytes(self.get_timestamp()))
        unsigned = value_bytes + separator + timestamp
        return unsigned + separator + self.get_signature(unsigned)

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
        """Verify a timestamped value and optionally check its age."""
        try:
            unsigned = super().unsign(signed_value)
            signature_error = None
        except BadSignature as error:
            signature_error = error
            unsigned = error.payload or b""

        separator = want_bytes(self.sep)

        if separator not in unsigned:
            if signature_error is not None:
                raise signature_error

            raise BadTimeSignature("timestamp missing", payload=unsigned)

        value, timestamp_data = unsigned.rsplit(separator, 1)
        timestamp: int | None = None
        signed_at: datetime | None = None

        try:
            timestamp = bytes_to_int(base64_decode(timestamp_data))
        except Exception:
            pass

        if signature_error is not None:
            if timestamp is not None:
                try:
                    signed_at = self.timestamp_to_datetime(timestamp)
                except (ValueError, OSError, OverflowError) as error:
                    raise BadTimeSignature(
                        "Malformed timestamp", payload=value
                    ) from error

            raise BadTimeSignature(
                str(signature_error), payload=value, date_signed=signed_at
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

    def validate(
        self, signed_value: str | bytes, max_age: int | None = None
    ) -> bool:
        """Return whether a signed value has a valid signature and timestamp."""
        try:
            self.unsign(signed_value, max_age=max_age)
        except BadSignature:
            return False

        return True


class TimedSerializer(Serializer[_TSerialized]):
    """A serializer that uses :class:`TimestampSigner`."""

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
        """Load signed serialized data, optionally enforcing an age limit."""
        value = want_bytes(s)
        last_error = None

        for signer in self.iter_unsigners(salt):
            try:
                payload_data, timestamp = signer.unsign(
                    value, max_age=max_age, return_timestamp=True
                )
                payload = self.load_payload(payload_data)

                if return_timestamp:
                    return payload, timestamp

                return payload
            except SignatureExpired:
                raise
            except BadSignature as error:
                last_error = error

        raise t.cast(BadSignature, last_error)

    def loads_unsafe(
        self,
        s: str | bytes,
        max_age: int | None = None,
        salt: str | bytes | None = None,
    ) -> tuple[bool, t.Any]:
        return self._loads_unsafe_impl(s, salt, load_kwargs={"max_age": max_age})