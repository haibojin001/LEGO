from __future__ import annotations

import collections.abc as cabc
import json
import typing as t

from .encoding import want_bytes
from .exc import BadPayload
from .exc import BadSignature
from .signer import _make_keys_list
from .signer import Signer

if t.TYPE_CHECKING:
    import typing_extensions as te

    _TSerialized = te.TypeVar("_TSerialized", bound=str | bytes, default=str | bytes)
else:
    _TSerialized = t.TypeVar("_TSerialized", bound=str | bytes)


class _PDataSerializer(t.Protocol[_TSerialized]):
    def loads(self, payload: _TSerialized, /) -> t.Any: ...
    def dumps(self, obj: t.Any, /) -> _TSerialized: ...


def is_text_serializer(
    serializer: _PDataSerializer[t.Any],
) -> te.TypeGuard[_PDataSerializer[str]]:
    """Checks whether a serializer generates text or binary."""
    return isinstance(serializer.dumps({}), str)


class Serializer(t.Generic[_TSerialized]):
    """Serialize Python objects and protect their serialized form with a signature."""

    default_serializer: _PDataSerializer[t.Any] = json
    default_signer: type[Signer] = Signer
    default_fallback_signers: list[
        dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
    ] = []

    @t.overload
    def __init__(
        self: Serializer[str],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        serializer: None | _PDataSerializer[str] = None,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None: ...

    @t.overload
    def __init__(
        self: Serializer[bytes],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None,
        serializer: _PDataSerializer[bytes],
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None: ...

    @t.overload
    def __init__(
        self: Serializer[bytes],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        *,
        serializer: _PDataSerializer[bytes],
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None: ...

    @t.overload
    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None,
        serializer: t.Any,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None: ...

    @t.overload
    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        *,
        serializer: t.Any,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None: ...

    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        serializer: t.Any | None = None,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ) -> None:
        self.secret_keys: list[bytes] = _make_keys_list(secret_key)

        if salt is not None:
            salt = want_bytes(salt)

        self.salt = salt

        if serializer is None:
            serializer = self.default_serializer

        self.serializer: _PDataSerializer[_TSerialized] = serializer
        self.is_text_serializer: bool = is_text_serializer(serializer)
        self.serializer_kwargs: dict[str, t.Any] = serializer_kwargs or {}

        if signer is None:
            signer = self.default_signer

        self.signer: type[Signer] = signer
        self.signer_kwargs: dict[str, t.Any] = signer_kwargs or {}

        if fallback_signers is None:
            fallback_signers = self.default_fallback_signers

        self.fallback_signers = fallback_signers

    def make_signer(self, salt: str | bytes | None = None) -> Signer:
        """Create a signer configured with this serializer's settings."""
        if salt is None:
            salt = self.salt

        return self.signer(self.secret_keys, salt=salt, **self.signer_kwargs)

    def iter_unsigners(
        self, salt: str | bytes | None = None
    ) -> cabc.Iterator[Signer]:
        """Iterate over all signers that can verify this serializer's data."""
        if salt is None:
            salt = self.salt

        yield self.make_signer(salt)

        for fallback in self.fallback_signers:
            if isinstance(fallback, dict):
                signer = self.signer
                kwargs = fallback
            elif isinstance(fallback, tuple):
                signer, kwargs = fallback
            else:
                signer = fallback
                kwargs = {}

            yield signer(self.secret_keys, salt=salt, **kwargs)

    def dumps(
        self, obj: t.Any, salt: str | bytes | None = None
    ) -> _TSerialized:
        """Serialize an object and append a signature."""
        payload = self.dump_payload(obj)
        signed = self.make_signer(salt).sign(payload)

        if self.is_text_serializer:
            return signed.decode("utf-8")  # type: ignore[return-value]

        return signed  # type: ignore[return-value]

    def dump_payload(self, obj: t.Any) -> bytes:
        """Serialize an object to the bytes passed to the signer."""
        return want_bytes(self.serializer.dumps(obj, **self.serializer_kwargs))

    def load_payload(
        self,
        payload: bytes,
        serializer: _PDataSerializer[t.Any] | None = None,
    ) -> t.Any:
        """Deserialize a verified payload."""
        if serializer is None:
            serializer = self.serializer

        try:
            if is_text_serializer(serializer):
                return serializer.loads(payload.decode("utf-8"))

            return serializer.loads(payload)
        except Exception as e:
            raise BadPayload(
                "Could not load the payload because an exception occurred"
                " on unserializing the data.",
                original_error=e,
            ) from e

    def loads(
        self,
        s: str | bytes,
        salt: str | bytes | None = None,
        serializer: _PDataSerializer[t.Any] | None = None,
    ) -> t.Any:
        """Verify a signed value and deserialize its payload."""
        value = want_bytes(s)
        last_exception: BadSignature | None = None

        for signer in self.iter_unsigners(salt):
            try:
                payload = signer.unsign(value)
            except BadSignature as e:
                last_exception = e
                continue

            return self.load_payload(payload, serializer=serializer)

        raise t.cast(BadSignature, last_exception)

    def loads_unsafe(
        self,
        s: str | bytes,
        salt: str | bytes | None = None,
        serializer: _PDataSerializer[t.Any] | None = None,
    ) -> tuple[bool, t.Any]:
        """Deserialize a value even if its signature is invalid."""
        try:
            return True, self.loads(s, salt=salt, serializer=serializer)
        except BadSignature as e:
            if e.payload is None:
                return False, None

            return False, self.load_payload(e.payload, serializer=serializer)