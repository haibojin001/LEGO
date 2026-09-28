from __future__ import annotations

import ipaddress
import re
import typing
from abc import ABC, abstractmethod
from operator import attrgetter

from marshmallow.exceptions import ValidationError

if typing.TYPE_CHECKING:
    from marshmallow import types

_T = typing.TypeVar("_T")
_UNICODE_LETTERS = "\u00a1-\uffff"

__all__ = (
    "And",
    "ContainsNone",
    "ContainsOnly",
    "Email",
    "Equal",
    "IP",
    "IPv4",
    "IPv6",
    "Length",
    "NoneOf",
    "OneOf",
    "Predicate",
    "Range",
    "Regexp",
    "URL",
    "Validator",
)


class Validator(ABC):
    error: str | None = None

    def __repr__(self) -> str:
        arguments = self._repr_args()
        if arguments:
            arguments += ", "
        return f"<{self.__class__.__name__}({arguments}error={self.error!r})>"

    def _repr_args(self) -> str:
        return ""

    @abstractmethod
    def __call__(self, value: typing.Any) -> typing.Any:
        raise NotImplementedError


class And(Validator):
    def __init__(self, *validators: types.Validator):
        self.validators = tuple(validators)

    def _repr_args(self) -> str:
        return f"validators={self.validators!r}"

    def __call__(self, value: typing.Any) -> typing.Any:
        messages: list[str | dict] = []
        metadata: dict[str, typing.Any] = {}

        for validator in self.validators:
            try:
                validator(value)
            except ValidationError as error:
                metadata.update(error.kwargs)
                if isinstance(error.messages, dict):
                    messages.append(error.messages)
                else:
                    messages.extend(error.messages)

        if messages:
            raise ValidationError(messages, **metadata)

        return value


class URL(Validator):
    class RegexMemoizer:
        def __init__(self) -> None:
            self._memoized: dict[tuple[bool, bool, bool], re.Pattern[str]] = {}

        def _regex_generator(
            self, *, relative: bool, absolute: bool, require_tld: bool
        ) -> re.Pattern[str]:
            hosts = [
                (
                    r"(?:[A-Z0-9"
                    + _UNICODE_LETTERS
                    + r"](?:[A-Z0-9"
                    + _UNICODE_LETTERS
                    + r"-]{0,61}[A-Z0-9"
                    + _UNICODE_LETTERS
                    + r"])?\.)+"
                    r"(?:[A-Z"
                    + _UNICODE_LETTERS
                    + r"]{2,6}\.?|[A-Z0-9"
                    + _UNICODE_LETTERS
                    + r"-]{2,}\.?)"
                ),
                r"localhost",
                r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
                r"\[[A-F0-9]*:[A-F0-9:]+\]",
            ]

            if not require_tld:
                hosts.append(
                    r"(?:[A-Z0-9"
                    + _UNICODE_LETTERS
                    + r"](?:[A-Z0-9"
                    + _UNICODE_LETTERS
                    + r"-]{0,61}[A-Z0-9"
                    + _UNICODE_LETTERS
                    + r"])?\.?)"
                )

            absolute_url = "".join(
                (
                    r"(?:[a-z0-9\.\-\+]*)://",
                    r"(?:(?:[a-z0-9\-._~!$&'()*+,;=:]|%[0-9a-f]{2})*@)?",
                    r"(?:",
                    "|".join(hosts),
                    r")",
                    r"(?::\d+)?",
                )
            )
            relative_url = r"(?:/?|[/?#]\S+)\Z"

            if relative:
                if absolute:
                    expression = r"^(" + absolute_url + r")?" + relative_url
                else:
                    expression = r"^" + relative_url
            else:
                expression = r"^" + absolute_url + relative_url

            return re.compile(expression, re.IGNORECASE)

        def __call__(
            self, *, relative: bool, absolute: bool, require_tld: bool
        ) -> re.Pattern[str]:
            key = (relative, absolute, require_tld)
            try:
                return self._memoized[key]
            except KeyError:
                expression = self._regex_generator(
                    relative=relative,
                    absolute=absolute,
                    require_tld=require_tld,
                )
                self._memoized[key] = expression
                return expression

    _regex = RegexMemoizer()

    default_message = "Not a valid URL."
    default_schemes = {"http", "https", "ftp", "ftps"}

    def __init__(
        self,
        *,
        relative: bool = False,
        absolute: bool = True,
        schemes: types.StrSequenceOrSet | None = None,
        require_tld: bool = True,
        error: str | None = None,
    ):
        if not relative and not absolute:
            raise ValueError(
                "URL validation cannot set both relative and absolute to False."
            )

        self.relative = relative
        self.absolute = absolute
        self.error = error or self.default_message
        self.schemes = (
            {scheme.lower() for scheme in schemes}
            if schemes is not None
            else self.default_schemes
        )
        self.require_tld = require_tld

    def _repr_args(self) -> str:
        return f"relative={self.relative!r}, absolute={self.absolute!r}"

    def _format_error(self, value: str) -> str:
        return self.error.format(input=value)

    def __call__(self, value: str) -> str:
        message = self._format_error(value)

        if not value:
            raise ValidationError(message)

        scheme = None
        if "://" in value:
            scheme = value.split("://", 1)[0].lower()
            if scheme not in self.schemes:
                raise ValidationError(message)

        pattern = self._regex(
            relative=self.relative,
            absolute=self.absolute,
            require_tld=self.require_tld,
        )

        if scheme == "file" and value.lower().startswith("file:///"):
            match = pattern.search("file://localhost/" + value[8:])
        else:
            match = pattern.search(value)

        if not match:
            raise ValidationError(message)

        return value


class Email(Validator):
    USER_REGEX = re.compile(
        r"(^[-!#$%&'*+/=?^`{}|~\w]+(\.[-!#$%&'*+/=?^`{}|~\w]+)*\Z"
        r'|^"([\001-\010\013\014\016-\037!#-\[\]-\177]'
        r'|\\[\001-\011\013\014\016-\177])*"\Z)',
        re.IGNORECASE | re.UNICODE,
    )

    DOMAIN_REGEX = re.compile(
        r"(?:[A-Z0-9"
        + _UNICODE_LETTERS
        + r"](?:[A-Z0-9"
        + _UNICODE_LETTERS
        + r"-]{0,61}[A-Z0-9"
        + _UNICODE_LETTERS
        + r"])?\.)+"
        r"(?:[A-Z"
        + _UNICODE_LETTERS
        + r"]{2,6}|[A-Z0-9"
        + _UNICODE_LETTERS
        + r"-]{2,})\Z"
        r"|^\[(25[0-5]|2[0-4]\d|[0-1]?\d?\d)"
        r"(\.(25[0-5]|2[0-4]\d|[0-1]?\d?\d)){3}\]\Z"
        r"|^\[IPv6:[0-9A-F:.]+\]\Z",
        re.IGNORECASE | re.UNICODE,
    )

    DOMAIN_WHITELIST = ("localhost",)
    default_message = "Not a valid email address."

    def __init__(self, *, error: str | None = None):
        self.error = error or self.default_message

    def _format_error(self, value: str) -> str:
        return self.error.format(input=value)

    def __call__(self, value: str) -> str:
        message = self._format_error(value)

        if not value or value.count("@") != 1:
            raise ValidationError(message)

        user, domain = value.rsplit("@", 1)
        if not self.USER_REGEX.match(user):
            raise ValidationError(message)

        if domain.lower() in self.DOMAIN_WHITELIST:
            return value

        if not self.DOMAIN_REGEX.match(domain):
            raise ValidationError(message)

        if domain.lower().startswith("[ipv6:"):
            try:
                ipaddress.IPv6Address(domain[6:-1])
            except ValueError:
                raise ValidationError(message) from None

        return value


class Regexp(Validator):
    default_message = "String does not match expected pattern."

    def __init__(
        self,
        regex: str | bytes | re.Pattern,
        flags: int = 0,
        *,
        error: str | None = None,
    ):
        self.regex = re.compile(regex, flags) if isinstance(regex, (str, bytes)) else regex
        self.error = error or self.default_message

    def _repr_args(self) -> str:
        return f"regex={self.regex!r}"

    def _format_error(self, value: str | bytes) -> str:
        return self.error.format(input=value)

    def __call__(self, value: str | bytes) -> str | bytes:
        if self.regex.match(value) is None:
            raise ValidationError(self._format_error(value))
        return value


class Equal(Validator):
    default_message = "Must be equal to {other}."

    def __init__(self, comparable: typing.Any, *, error: str | None = None):
        self.comparable = comparable
        self.error = error or self.default_message

    def _repr_args(self) -> str:
        return f"comparable={self.comparable!r}"

    def _format_error(self, value: typing.Any) -> str:
        return self.error.format(input=value, other=self.comparable)

    def __call__(self, value: typing.Any) -> typing.Any:
        if value != self.comparable:
            raise ValidationError(self._format_error(value))
        return value


class OneOf(Validator):
    default_message = "Must be one of: {choices}."

    def __init__(
        self,
        choices: typing.Iterable[_T],
        labels: typing.Iterable[str] | None = None,
        *,
        error: str | None = None,
    ):
        self.choices = choices
        self.labels = labels
        self.error = error or self.default_message

    def _repr_args(self) -> str:
        return f"choices={self.choices!r}, labels={self.labels!r}"

    @property
    def options(self) -> typing.Iterator[tuple[typing.Any, typing.Any]]:
        if self.labels is not None:
            return zip(self.choices, self.labels)
        return zip(self.choices, self.choices)

    def _format_error(self, value: typing.Any) -> str:
        choices = self.labels if self.labels is not None else self.choices
        return self.error.format(
            input=value,
            choices=", ".join(str(choice) for choice in choices),
        )

    def __call__(self, value: _T) -> _T:
        if value not in self.choices:
            raise ValidationError(self._format_error(value))
        return value


class NoneOf(OneOf):
    default_message = "Invalid input."

    def __call__(self, value: _T) -> _T:
        if value in self.choices:
            raise ValidationError(self._format_error(value))
        return value


class ContainsOnly(OneOf):
    default_message = "Must be one of: {choices}."

    def __call__(self, value: typing.Iterable[_T]) -> typing.Iterable[_T]:
        if not value:
            return value

        if any(item not in self.choices for item in value):
            raise ValidationError(self._format_error(value))
        return value


class ContainsNone(NoneOf):
    default_message = "Invalid input."

    def __call__(self, value: typing.Iterable[_T]) -> typing.Iterable[_T]:
        if not value:
            return value

        if any(item in self.choices for item in value):
            raise ValidationError(self._format_error(value))
        return value


class Range(Validator):
    min_inclusive_message = "Must be greater than or equal to {min}."
    min_exclusive_message = "Must be greater than {min}."
    max_inclusive_message = "Must be less than or equal to {max}."
    max_exclusive_message = "Must be less than {max}."

    def __init__(
        self,
        min: typing.Any = None,
        max: typing.Any = None,
        *,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
        error: str | None = None,
    ):
        self.min = min
        self.max = max
        self.min_inclusive = min_inclusive
        self.max_inclusive = max_inclusive
        self.error = error

    def _repr_args(self) -> str:
        return (
            f"min={self.min!r}, max={self.max!r}, "
            f"min_inclusive={self.min_inclusive!r}, "
            f"max_inclusive={self.max_inclusive!r}"
        )

    def _format_error(self, value: typing.Any, message: str) -> str:
        return (self.error or message).format(
            input=value,
            min=self.min,
            max=self.max,
        )

    def __call__(self, value: typing.Any) -> typing.Any:
        if self.min is not None:
            if self.min_inclusive:
                if value < self.min:
                    raise ValidationError(
                        self._format_error(value, self.min_inclusive_message)
                    )
            elif value <= self.min:
                raise ValidationError(self._format_error(value, self.min_exclusive_message))

        if self.max is not None:
            if self.max_inclusive:
                if value > self.max:
                    raise ValidationError(
                        self._format_error(value, self.max_inclusive_message)
                    )
            elif value >= self.max:
                raise ValidationError(self._format_error(value, self.max_exclusive_message))

        return value


class Length(Validator):
    min_message = "Shorter than minimum length {min}."
    max_message = "Longer than maximum length {max}."
    equal_message = "Length must be {equal}."

    def __init__(
        self,
        min: int | None = None,
        max: int | None = None,
        *,
        equal: int | None = None,
        error: str | None = None,
    ):
        if equal is not None and (min is not None or max is not None):
            raise ValueError(
                "The `equal` parameter was provided, maximum or minimum parameter must not be provided."
            )

        self.min = min
        self.max = max
        self.equal = equal
        self.error = error

    def _repr_args(self) -> str:
        return f"min={self.min!r}, max={self.max!r}, equal={self.equal!r}"

    def _format_error(self, value: typing.Sized, message: str) -> str:
        return (self.error or message).format(
            input=value,
            min=self.min,
            max=self.max,
            equal=self.equal,
        )

    def __call__(self, value: typing.Sized) -> typing.Sized:
        length = len(value)

        if self.equal is not None:
            if length != self.equal:
                raise ValidationError(self._format_error(value, self.equal_message))
            return value

        if self.min is not None and length < self.min:
            raise ValidationError(self._format_error(value, self.min_message))

        if self.max is not None and length > self.max:
            raise ValidationError(self._format_error(value, self.max_message))

        return value


class Predicate(Validator):
    default_message = "Invalid input."

    def __init__(
        self,
        method: str,
        *,
        error: str | None = None,
        **kwargs: typing.Any,
    ):
        self.method = method
        self.error = error or self.default_message
        self.kwargs = kwargs

    def _repr_args(self) -> str:
        return f"method={self.method!r}, kwargs={self.kwargs!r}"

    def _format_error(self, value: typing.Any) -> str:
        return self.error.format(input=value)

    def __call__(self, value: typing.Any) -> typing.Any:
        method = attrgetter(self.method)(value)
        if not method(**self.kwargs):
            raise ValidationError(self._format_error(value))
        return value


class IP(Validator):
    default_message = "Not a valid IP address."

    def __init__(self, *, exploded: bool = False, error: str | None = None):
        self.exploded = exploded
        self.error = error or self.default_message

    def _repr_args(self) -> str:
        return f"exploded={self.exploded!r}"

    def _format_error(self, value: typing.Any) -> str:
        return self.error.format(input=value)

    def _validate_ip(self, value: typing.Any) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
        return ipaddress.ip_address(value)

    def __call__(self, value: typing.Any) -> typing.Any:
        try:
            address = self._validate_ip(value)
        except ValueError:
            raise ValidationError(self._format_error(value)) from None

        if self.exploded and value != address.exploded:
            raise ValidationError(self._format_error(value))

        return value


class IPv4(IP):
    default_message = "Not a valid IPv4 address."

    def _validate_ip(self, value: typing.Any) -> ipaddress.IPv4Address:
        address = ipaddress.ip_address(value)
        if not isinstance(address, ipaddress.IPv4Address):
            raise ValueError
        return address


class IPv6(IP):
    default_message = "Not a valid IPv6 address."

    def _validate_ip(self, value: typing.Any) -> ipaddress.IPv6Address:
        address = ipaddress.ip_address(value)
        if not isinstance(address, ipaddress.IPv6Address):
            raise ValueError
        return address