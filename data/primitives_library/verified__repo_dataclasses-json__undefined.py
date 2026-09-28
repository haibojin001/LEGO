import abc
import dataclasses
import functools
import inspect
import sys
from dataclasses import Field, fields
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union, get_type_hints

from marshmallow.exceptions import ValidationError  # type: ignore

from dataclasses_json.utils import CatchAllVar


KnownParameters = Dict[str, Any]
UnknownParameters = Dict[str, Any]


class UndefinedParameterError(ValidationError):
    pass


class _UndefinedParameterAction(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def handle_from_dict(cls, kvs: Dict[Any, Any]) -> Dict[str, Any]:
        pass

    @staticmethod
    def handle_to_dict(obj, kvs: Dict[Any, Any]) -> Dict[Any, Any]:
        return kvs

    @staticmethod
    def handle_dump(obj) -> Dict[Any, Any]:
        return {}

    @staticmethod
    def create_init(obj) -> Callable:
        return obj.__init__

    @staticmethod
    def _separate_defined_undefined_kvs(
        cls, kvs: Dict
    ) -> Tuple[KnownParameters, UnknownParameters]:
        field_names = [field.name for field in fields(cls)]
        known = {key: value for key, value in kvs.items() if key in field_names}
        unknown = {key: value for key, value in kvs.items() if key not in field_names}
        return known, unknown


class _RaiseUndefinedParameters(_UndefinedParameterAction):
    @staticmethod
    def handle_from_dict(cls, kvs: Dict) -> Dict[str, Any]:
        known, unknown = _UndefinedParameterAction._separate_defined_undefined_kvs(
            cls, kvs
        )
        if unknown:
            raise UndefinedParameterError(
                f"Received undefined initialization arguments {unknown}"
            )
        return known


CatchAll = Optional[CatchAllVar]


class _IgnoreUndefinedParameters(_UndefinedParameterAction):
    @staticmethod
    def handle_from_dict(cls, kvs: Dict) -> Dict[str, Any]:
        known, _ = _UndefinedParameterAction._separate_defined_undefined_kvs(
            cls, kvs
        )
        return known

    @staticmethod
    def create_init(obj) -> Callable:
        original_init = obj.__init__
        init_signature = inspect.signature(original_init)

        @functools.wraps(original_init)
        def _ignore_init(self, *args, **kwargs):
            known_kwargs, _ = (
                _CatchAllUndefinedParameters._separate_defined_undefined_kvs(
                    obj, kwargs
                )
            )
            num_params_takeable = len(init_signature.parameters) - 1
            num_args_takeable = num_params_takeable - len(known_kwargs)
            args = args[:num_args_takeable]

            bound_parameters = init_signature.bind_partial(
                self, *args, **known_kwargs
            )
            bound_parameters.apply_defaults()

            arguments = bound_parameters.arguments
            arguments.pop("self", None)
            final_parameters = _IgnoreUndefinedParameters.handle_from_dict(
                obj, arguments
            )
            original_init(self, **final_parameters)

        return _ignore_init


class _CatchAllUndefinedParameters(_UndefinedParameterAction):
    class _SentinelNoDefault:
        pass

    @staticmethod
    def handle_from_dict(cls, kvs: Dict) -> Dict[str, Any]:
        known, unknown = _UndefinedParameterAction._separate_defined_undefined_kvs(
            cls, kvs
        )
        catch_all_field = _CatchAllUndefinedParameters._get_catch_all_field(cls)

        if catch_all_field.name in known:
            already_parsed = isinstance(known[catch_all_field.name], dict)
            default_value = _CatchAllUndefinedParameters._get_default(
                catch_all_field
            )
            received_default = default_value == known[catch_all_field.name]

            if received_default and len(unknown) == 0:
                value_to_write: Any = default_value
            elif received_default and len(unknown) > 0:
                value_to_write = unknown
            elif already_parsed:
                value_to_write = known[catch_all_field.name]
                if len(unknown) > 0:
                    value_to_write.update(unknown)
            else:
                raise UndefinedParameterError(
                    "Received input field with same name as catch-all field: "
                    f"'{catch_all_field.name}': '{known[catch_all_field.name]}'"
                )
        else:
            value_to_write = unknown

        known[catch_all_field.name] = value_to_write
        return known

    @staticmethod
    def _get_default(catch_all_field: Field) -> Any:
        has_default = not isinstance(
            catch_all_field.default, dataclasses._MISSING_TYPE
        )
        has_default_factory = not isinstance(
            catch_all_field.default_factory, dataclasses._MISSING_TYPE
        )

        default_value: Union[
            Type[_CatchAllUndefinedParameters._SentinelNoDefault], Any
        ] = _CatchAllUndefinedParameters._SentinelNoDefault

        if has_default:
            default_value = catch_all_field.default
        elif has_default_factory:
            default_value = catch_all_field.default_factory()  # type: ignore

        return default_value

    @staticmethod
    def _get_catch_all_field(cls) -> Field:
        if sys.version_info.minor == 6:
            type_hints = cls.__annotations__
        else:
            type_hints = get_type_hints(cls)

        catch_all_fields = [
            field
            for field in fields(cls)
            if type_hints.get(field.name) == CatchAll
            or type_hints.get(field.name) == CatchAllVar
            or field.type == CatchAll
            or field.type == CatchAllVar
        ]

        if len(catch_all_fields) == 0:
            raise UndefinedParameterError(
                f"No field of type CatchAll found in class {cls.__name__}"
            )
        if len(catch_all_fields) > 1:
            raise UndefinedParameterError(
                f"Multiple fields of type CatchAll found in class {cls.__name__}"
            )

        return catch_all_fields[0]

    @staticmethod
    def handle_to_dict(obj, kvs: Dict[Any, Any]) -> Dict[Any, Any]:
        catch_all_field = _CatchAllUndefinedParameters._get_catch_all_field(
            obj.__class__
        )
        undefined_parameters = kvs.pop(catch_all_field.name)
        if isinstance(undefined_parameters, dict):
            kvs.update(undefined_parameters)
        return kvs

    @staticmethod
    def handle_dump(obj) -> Dict[Any, Any]:
        catch_all_field = _CatchAllUndefinedParameters._get_catch_all_field(
            obj.__class__
        )
        return getattr(obj, catch_all_field.name)

    @staticmethod
    def create_init(obj) -> Callable:
        original_init = obj.__init__
        init_signature = inspect.signature(original_init)

        @functools.wraps(original_init)
        def _catch_all_init(self, *args, **kwargs):
            known_kwargs, _ = (
                _CatchAllUndefinedParameters._separate_defined_undefined_kvs(
                    obj, kwargs
                )
            )

            num_params_takeable = len(init_signature.parameters) - 1
            if (
                _CatchAllUndefinedParameters._get_catch_all_field(obj).name
                not in known_kwargs
            ):
                num_params_takeable -= 1

            num_args_takeable = num_params_takeable - len(known_kwargs)
            args, unknown_args = (
                args[:num_args_takeable],
                args[num_args_takeable:],
            )

            bound_parameters = init_signature.bind_partial(
                self, *args, **known_kwargs
            )
            unknown_args = {
                f"_UNKNOWN{i}": value for i, value in enumerate(unknown_args)
            }

            arguments = bound_parameters.arguments
            arguments.update(unknown_args)
            arguments.pop("self", None)

            final_parameters = _CatchAllUndefinedParameters.handle_from_dict(
                obj, arguments
            )
            original_init(self, **final_parameters)

        return _catch_all_init


class Undefined(Enum):
    INCLUDE = _CatchAllUndefinedParameters
    RAISE = _RaiseUndefinedParameters
    EXCLUDE = _IgnoreUndefinedParameters