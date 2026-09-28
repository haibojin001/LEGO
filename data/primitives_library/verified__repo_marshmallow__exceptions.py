from __future__ import annotations

import typing

SCHEMA = "_schema"


class MarshmallowError(Exception):
    """Base exception type for marshmallow errors."""


class ValidationError(MarshmallowError):
    """Error raised when field or schema validation does not succeed."""

    def __init__(
        self,
        message: str | list | dict,
        field_name: str = SCHEMA,
        data: typing.Mapping[str, typing.Any]
        | typing.Iterable[typing.Mapping[str, typing.Any]]
        | None = None,
        valid_data: list[typing.Any] | dict[str, typing.Any] | None = None,
        **kwargs,
    ):
        if isinstance(message, (str, bytes)):
            self.messages = [message]
        else:
            self.messages = message
        self.field_name = field_name
        self.data = data
        self.valid_data = valid_data
        self.kwargs = kwargs
        Exception.__init__(self, message)

    def normalized_messages(self):
        if self.field_name == SCHEMA and isinstance(self.messages, dict):
            return self.messages
        return {self.field_name: self.messages}

    @property
    def messages_dict(self) -> dict[str, typing.Any]:
        if not isinstance(self.messages, dict):
            typename = type(self.messages).__name__
            raise TypeError(
                "cannot access 'messages_dict' when 'messages' is of type " + typename
            )
        return self.messages


class RegistryError(NameError):
    """Raised for invalid operations involving the schema registry."""


class StringNotCollectionError(MarshmallowError, TypeError):
    """Raised when text is used where a collection of strings is required."""


class _FieldInstanceResolutionError(MarshmallowError, TypeError):
    """Raised when a field argument cannot be turned into a field instance."""