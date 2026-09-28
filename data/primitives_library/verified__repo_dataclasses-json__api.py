import abc
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, overload

from dataclasses_json.cfg import LetterCase, config
from dataclasses_json.core import Json, _ExtendedEncoder, _asdict, _decode_dataclass
from dataclasses_json.mm import JsonData, build_schema
from dataclasses_json.undefined import Undefined
from dataclasses_json.utils import _handle_undefined_parameters_safe

A = TypeVar("A", bound="DataClassJsonMixin")
T = TypeVar("T")
Fields = List[Tuple[str, Any]]


def _undefined_parameter_action_safe(cls: Type[Any]) -> Any:
    try:
        return cls.dataclass_json_config["undefined"]
    except (AttributeError, KeyError, TypeError):
        return None


class DataClassJsonMixin(abc.ABC):
    """
    DataClassJsonMixin is an ABC that functions as a Mixin.

    As with other ABCs, it should not be instantiated directly.
    """

    dataclass_json_config: Optional[dict] = None

    def to_json(
        self,
        *,
        skipkeys: bool = False,
        ensure_ascii: bool = True,
        check_circular: bool = True,
        allow_nan: bool = True,
        indent: Optional[Union[int, str]] = None,
        separators: Optional[Tuple[str, str]] = None,
        default: Optional[Callable] = None,
        sort_keys: bool = False,
        **kw
    ) -> str:
        return json.dumps(
            self.to_dict(encode_json=False),
            cls=_ExtendedEncoder,
            skipkeys=skipkeys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            indent=indent,
            separators=separators,
            default=default,
            sort_keys=sort_keys,
            **kw
        )

    @classmethod
    def from_json(
        cls: Type[A],
        s: JsonData,
        *,
        parse_float=None,
        parse_int=None,
        parse_constant=None,
        infer_missing=False,
        **kw
    ) -> A:
        kvs = json.loads(
            s,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            **kw
        )
        return cls.from_dict(kvs, infer_missing=infer_missing)

    @classmethod
    def from_dict(
        cls: Type[A],
        kvs: Json,
        *,
        infer_missing=False
    ) -> A:
        return _decode_dataclass(cls, kvs, infer_missing)

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        return _asdict(self, encode_json=encode_json)

    @classmethod
    def schema(
        cls: Type[A],
        *,
        infer_missing: bool = False,
        only=None,
        exclude=(),
        many: bool = False,
        context=None,
        load_only=(),
        dump_only=(),
        partial: bool = False,
        unknown=None
    ) -> Any:
        schema_type = build_schema(cls, DataClassJsonMixin, infer_missing, partial)

        if unknown is None:
            action = _undefined_parameter_action_safe(cls)
            if action is not None:
                unknown = (
                    action.name.lower()
                    if hasattr(action, "name")
                    else str(action).lower()
                )

        return schema_type(
            only=only,
            exclude=exclude,
            many=many,
            context=context,
            load_only=load_only,
            dump_only=dump_only,
            partial=partial,
            unknown=unknown,
        )


@overload
def dataclass_json(
    _cls: None = ...,
    *,
    letter_case: Optional[LetterCase] = ...,
    undefined: Optional[Union[str, Undefined]] = ...
) -> Callable[[Type[T]], Type[T]]:
    ...


@overload
def dataclass_json(
    _cls: Type[T],
    *,
    letter_case: Optional[LetterCase] = ...,
    undefined: Optional[Union[str, Undefined]] = ...
) -> Type[T]:
    ...


def dataclass_json(
    _cls: Optional[Type[T]] = None,
    *,
    letter_case: Optional[LetterCase] = None,
    undefined: Optional[Union[str, Undefined]] = None
) -> Union[Callable[[Type[T]], Type[T]], Type[T]]:
    def wrap(cls: Type[T]) -> Type[T]:
        return _process_class(cls, letter_case, undefined)

    if _cls is None:
        return wrap
    return wrap(_cls)


def _process_class(
    cls: Type[T],
    letter_case: Optional[LetterCase],
    undefined: Optional[Union[str, Undefined]]
) -> Type[T]:
    if letter_case is not None or undefined is not None:
        cls.dataclass_json_config = config(
            letter_case=letter_case,
            undefined=undefined,
        )["dataclasses_json"]

    cls.to_json = DataClassJsonMixin.to_json
    cls.from_json = classmethod(DataClassJsonMixin.from_json.__func__)
    cls.to_dict = DataClassJsonMixin.to_dict
    cls.from_dict = classmethod(DataClassJsonMixin.from_dict.__func__)
    cls.schema = classmethod(DataClassJsonMixin.schema.__func__)

    cls.__init__ = _handle_undefined_parameters_safe(cls, kvs=(), usage="init")
    DataClassJsonMixin.register(cls)
    return cls