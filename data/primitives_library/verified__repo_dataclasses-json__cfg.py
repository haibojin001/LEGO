import functools
from enum import Enum
from typing import Callable, Dict, Optional, TypeVar, Union

from marshmallow.fields import Field as MarshmallowField  # type: ignore

from dataclasses_json.stringcase import camelcase, pascalcase, snakecase, spinalcase  # type: ignore
from dataclasses_json.undefined import Undefined, UndefinedParameterError

T = TypeVar("T")


class Exclude:
    ALWAYS: Callable[[object], bool] = lambda _: True
    NEVER: Callable[[object], bool] = lambda _: False


class _GlobalConfig:
    def __init__(self):
        self.encoders: Dict[Union[type, Optional[type]], Callable] = {}
        self.decoders: Dict[Union[type, Optional[type]], Callable] = {}
        self.mm_fields: Dict[Union[type, Optional[type]], MarshmallowField] = {}


global_config = _GlobalConfig()


class LetterCase(Enum):
    CAMEL = camelcase
    KEBAB = spinalcase
    SNAKE = snakecase
    PASCAL = pascalcase


def config(
    metadata: Optional[dict] = None,
    *,
    encoder: Optional[Callable] = None,
    decoder: Optional[Callable] = None,
    mm_field: Optional[MarshmallowField] = None,
    letter_case: Union[Callable[[str], str], LetterCase, None] = None,
    undefined: Optional[Union[str, Undefined]] = None,
    field_name: Optional[str] = None,
    exclude: Optional[Callable[[T], bool]] = None,
) -> Dict[str, dict]:
    if metadata is None:
        metadata = {}

    settings = metadata.setdefault("dataclasses_json", {})

    for key, value in (
        ("encoder", encoder),
        ("decoder", decoder),
        ("mm_field", mm_field),
    ):
        if value is not None:
            settings[key] = value

    if field_name is not None:
        if letter_case is None:
            def replacement(_, _name=field_name):  # type: ignore
                return _name
        else:
            @functools.wraps(letter_case)  # type: ignore
            def replacement(_, _name=field_name, _case=letter_case):
                return _case(_name)

        letter_case = replacement

    if letter_case is not None:
        settings["letter_case"] = letter_case

    if undefined is not None:
        if isinstance(undefined, str):
            name = undefined.upper()
            if not hasattr(Undefined, name):
                allowed = [member.name for member in Undefined]
                raise UndefinedParameterError(
                    f"Invalid undefined parameter action, must be one of {allowed}"
                )
            undefined = Undefined[name]

        settings["undefined"] = undefined

    if exclude is not None:
        settings["exclude"] = exclude

    return metadata