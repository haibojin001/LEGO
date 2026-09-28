from __future__ import annotations

import copy
import datetime as dt
import decimal
import functools
import inspect
import ipaddress
import json
import operator
import typing
import uuid
from abc import ABCMeta
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import zip_longest

from marshmallow import class_registry, types
from marshmallow import fields as ma_fields
from marshmallow.constants import EXCLUDE, INCLUDE, RAISE, missing
from marshmallow.decorators import (
    POST_DUMP,
    POST_LOAD,
    PRE_DUMP,
    PRE_LOAD,
    VALIDATES,
    VALIDATES_SCHEMA,
)
from marshmallow.error_store import ErrorStore
from marshmallow.exceptions import SCHEMA, StringNotCollectionError, ValidationError
from marshmallow.orderedset import OrderedSet
from marshmallow.utils import (
    get_value,
    is_collection,
    is_sequence_but_not_string,
    set_value,
)

if typing.TYPE_CHECKING:
    from marshmallow.fields import Field


if not hasattr(OrderedSet, "update"):

    def _orderedset_update(self, iterable):
        for item in iterable:
            self.add(item)

    OrderedSet.update = _orderedset_update


def _get_fields(attrs) -> list[tuple[str, Field]]:
    result = []
    for name, value in attrs.items():
        if isinstance(value, type) and issubclass(value, ma_fields.Field):
            raise TypeError(
                f'Field for "{name}" must be declared as a Field instance, '
                f'not a class. Did you mean "fields.{value.__name__}()"?'
            )
        if isinstance(value, ma_fields.Field):
            result.append((name, value))
    return result


def _get_fields_by_mro(klass: SchemaMeta):
    return functools.reduce(
        operator.iadd,
        (
            _get_fields(getattr(base, "_declared_fields", base.__dict__))
            for base in inspect.getmro(klass)[:0:-1]
        ),
        [],
    )


class SchemaMeta(ABCMeta):
    Meta: type
    opts: typing.Any
    OPTIONS_CLASS: type
    _declared_fields: dict[str, Field]

    def __new__(mcs, name, bases, attrs):
        declared = _get_fields(attrs)
        for field_name, _ in declared:
            del attrs[field_name]

        klass = super().__new__(mcs, name, bases, attrs)
        inherited = _get_fields_by_mro(klass)
        klass.opts = klass.OPTIONS_CLASS(klass.Meta)
        declared.extend(klass.opts.include.items())
        klass._declared_fields = mcs.get_declared_fields(
            klass=klass,
            cls_fields=declared,
            inherited_fields=inherited,
            dict_cls=dict,
        )
        return klass

    @classmethod
    def get_declared_fields(
        mcs,
        klass: SchemaMeta,
        cls_fields: list[tuple[str, Field]],
        inherited_fields: list[tuple[str, Field]],
        dict_cls: type[dict] = dict,
    ) -> dict[str, Field]:
        return dict_cls(inherited_fields + cls_fields)

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name and cls.opts.register:
            class_registry.register(name, cls)
        cls._hooks = cls.resolve_hooks()

    def resolve_hooks(cls) -> dict[str, list[tuple[str, bool, dict]]]:
        hooks = defaultdict(list)
        mro = inspect.getmro(cls)

        for attr_name in dir(cls):
            for parent in mro:
                try:
                    attr = parent.__dict__[attr_name]
                except KeyError:
                    continue
                else:
                    break
            else:
                continue

            config = getattr(attr, "__marshmallow_hook__", None)
            if config:
                for tag, entries in config.items():
                    hooks[tag].extend(
                        (attr_name, many, kwargs) for many, kwargs in entries
                    )
        return hooks


class SchemaOpts:
    def __init__(self, meta: type):
        self.fields = getattr(meta, "fields", ())
        if not isinstance(self.fields, (list, tuple)):
            raise ValueError("`fields` option must be a list or tuple.")
        self.exclude = getattr(meta, "exclude", ())
        if not isinstance(self.exclude, (list, tuple)):
            raise ValueError("`exclude` must be a list or tuple.")
        self.dateformat = getattr(meta, "dateformat", None)
        self.datetimeformat = getattr(meta, "datetimeformat", None)
        self.timeformat = getattr(meta, "timeformat", None)
        self.render_module = getattr(meta, "render_module", json)
        self.index_errors = getattr(meta, "index_errors", True)
        self.include = getattr(meta, "include", {})
        self.load_only = getattr(meta, "load_only", ())
        self.dump_only = getattr(meta, "dump_only", ())
        self.unknown = getattr(meta, "unknown", RAISE)
        self.register = getattr(meta, "register", True)
        self.many = getattr(meta, "many", False)


class Schema(metaclass=SchemaMeta):
    TYPE_MAPPING = {
        str: ma_fields.String,
        bytes: ma_fields.String,
        dt.datetime: ma_fields.DateTime,
        dt.date: ma_fields.Date,
        dt.time: ma_fields.Time,
        dt.timedelta: ma_fields.TimeDelta,
        bool: ma_fields.Boolean,
        int: ma_fields.Integer,
        float: ma_fields.Float,
        decimal.Decimal: ma_fields.Decimal,
        uuid.UUID: ma_fields.UUID,
        tuple: ma_fields.Raw,
        list: ma_fields.List,
        set: ma_fields.List,
        frozenset: ma_fields.List,
        dict: ma_fields.Dict,
        ipaddress.IPv4Address: ma_fields.IPv4,
        ipaddress.IPv6Address: ma_fields.IPv6,
        ipaddress.IPv4Interface: ma_fields.IPv4Interface,
        ipaddress.IPv6Interface: ma_fields.IPv6Interface,
    }

    OPTIONS_CLASS = SchemaOpts
    set_class = OrderedSet
    dict_class = dict
    error_messages = {
        "type": "Invalid input type.",
        "unknown": "Unknown field.",
    }

    class Meta:
        pass

    def __init__(
        self,
        *,
        only: types.StrSequenceOrSet | None = None,
        exclude: types.StrSequenceOrSet = (),
        many: bool | None = None,
        context: dict | None = None,
        load_only: types.StrSequenceOrSet = (),
        dump_only: types.StrSequenceOrSet = (),
        partial: bool | types.StrSequenceOrSet | None = None,
        unknown: str | None = None,
    ):
        if only is not None and not is_collection(only):
            raise StringNotCollectionError('"only" should be a collection of strings.')
        if not is_collection(exclude):
            raise StringNotCollectionError(
                '"exclude" should be a collection of strings.'
            )
        if not is_collection(load_only):
            raise StringNotCollectionError(
                '"load_only" should be a collection of strings.'
            )
        if not is_collection(dump_only):
            raise StringNotCollectionError(
                '"dump_only" should be a collection of strings.'
            )

        self.declared_fields = copy.deepcopy(self._declared_fields)
        self.many = self.opts.many if many is None else many
        self.only = self.set_class(only) if only is not None else None
        self.exclude = self.set_class(self.opts.exclude) | self.set_class(exclude)
        self.load_only = self.set_class(self.opts.load_only) | self.set_class(load_only)
        self.dump_only = self.set_class(self.opts.dump_only) | self.set_class(dump_only)
        self.partial = partial
        self.unknown = self.opts.unknown if unknown is None else unknown
        self.context = context or {}
        self._normalize_nested_options()
        self.fields = {}
        self.load_fields = {}
        self.dump_fields = {}
        self._init_fields()
        self.error_messages = copy.deepcopy(self.error_messages)

    def __repr__(self):
        return f"<{self.__class__.__name__}(many={self.many})>"

    @classmethod
    def from_dict(cls, fields: dict[str, Field], *, name: str = "GeneratedSchema"):
        meta = type("Meta", (), {"register": False})
        return type(name, (cls,), dict(fields, Meta=meta))

    def _normalize_nested_options(self):
        if self.only is not None:
            self.only = self._normalize_nested_option(self.only)
        self.exclude = self._normalize_nested_option(self.exclude)
        self.load_only = self._normalize_nested_option(self.load_only)
        self.dump_only = self._normalize_nested_option(self.dump_only)

    @staticmethod
    def _normalize_nested_option(option):
        nested = defaultdict(list)
        for field_name in option:
            if "." in field_name:
                parent, child = field_name.split(".", 1)
                nested[parent].append(child)
            else:
                nested[field_name]
        result = OrderedSet()
        for field_name, children in nested.items():
            result.add(field_name)
            result.update(f"{field_name}.{child}" for child in children)
        return result

    def _init_fields(self):
        if self.opts.fields:
            available = self.set_class(self.opts.fields)
            unknown = available - self.set_class(self.declared_fields)
            if unknown:
                raise ValueError(
                    f"Invalid fields for {self}: {', '.join(sorted(unknown))}."
                )
        else:
            available = self.set_class(self.declared_fields)

        if self.only is not None:
            field_names = self.set_class(self.only)
            field_names = self._get_fields_for_nested(field_names)
        else:
            field_names = available

        field_names -= self.exclude
        field_names = self._get_fields_for_nested(field_names)

        unknown = field_names - self.set_class(self.declared_fields)
        if unknown:
            raise ValueError(f"Invalid fields for {self}: {', '.join(sorted(unknown))}.")

        for field_name in field_names:
            field_obj = self.declared_fields[field_name]
            self._bind_field(field_name, field_obj)
            self.fields[field_name] = field_obj

        self.load_fields = {
            name: field
            for name, field in self.fields.items()
            if not field.dump_only
        }
        self.dump_fields = {
            name: field
            for name, field in self.fields.items()
            if not field.load_only
        }

        external = defaultdict(list)
        for name, field in self.fields.items():
            external[field.data_key if field.data_key is not None else name].append(name)
        duplicates = [key for key, names in external.items() if len(names) > 1]
        if duplicates:
            raise ValueError(
                "The following field names are duplicated: "
                + ", ".join(sorted(duplicates))
            )

    @staticmethod
    def _get_fields_for_nested(options):
        return OrderedSet(option.split(".", 1)[0] for option in options)

    def _bind_field(self, field_name: str, field_obj):
        field_obj._bind_to_schema(field_name, self)
        if field_obj.load_only or field_name in self.load_only:
            field_obj.load_only = True
        if field_obj.dump_only or field_name in self.dump_only:
            field_obj.dump_only = True

        if isinstance(field_obj, ma_fields.Nested):
            if self.only is not None:
                only = self._nested_option(field_name, self.only)
                if only:
                    field_obj.only = only
            exclude = self._nested_option(field_name, self.exclude)
            if exclude:
                field_obj.exclude = exclude
            load_only = self._nested_option(field_name, self.load_only)
            if load_only:
                field_obj.load_only = load_only
            dump_only = self._nested_option(field_name, self.dump_only)
            if dump_only:
                field_obj.dump_only = dump_only

    @staticmethod
    def _nested_option(field_name, option):
        prefix = field_name + "."
        return [value[len(prefix) :] for value in option if value.startswith(prefix)]

    def _invoke_processors(self, tag, data, *, many, original_data=None, **kwargs):
        for attr_name, hook_many, processor_kwargs in self._hooks[tag]:
            if hook_many != many:
                continue
            processor = getattr(self, attr_name)
            if tag in (PRE_LOAD, POST_LOAD, VALIDATES_SCHEMA):
                processor_kwargs = dict(processor_kwargs)
                if "many" not in processor_kwargs:
                    processor_kwargs["many"] = many
                if tag == VALIDATES_SCHEMA:
                    processor_kwargs.setdefault("partial", kwargs.get("partial"))
                    processor_kwargs.setdefault("unknown", kwargs.get("unknown"))
            if tag == POST_DUMP:
                processor_kwargs = dict(processor_kwargs)
                processor_kwargs.setdefault("many", many)
                processor_kwargs.setdefault("original_data", original_data)
            data = processor(data, **processor_kwargs)
        return data

    def dump(self, obj, *, many: bool | None = None):
        many = self.many if many is None else many
        if many and obj is not None and not is_sequence_but_not_string(obj):
            raise ValidationError(
                {"_schema": ["Invalid type."]},
                data=obj,
                valid_data=[],
            )
        if not many:
            return self._dump(obj, many=many)
        return [self._dump(each, many=many) for each in obj]

    def _dump(self, obj, *, many):
        processed_obj = self._invoke_processors(PRE_DUMP, obj, many=many)
        result = self.dict_class()
        for attr_name, field_obj in self.dump_fields.items():
            value = field_obj.serialize(attr_name, processed_obj, accessor=self.get_attribute)
            if value is not missing:
                key = field_obj.data_key if field_obj.data_key is not None else attr_name
                result[key] = value
        return self._invoke_processors(
            POST_DUMP, result, many=many, original_data=obj
        )

    def dumps(self, obj, *args, many: bool | None = None, **kwargs):
        return self.opts.render_module.dumps(self.dump(obj, many=many), *args, **kwargs)

    def load(
        self,
        data,
        *,
        many: bool | None = None,
        partial: bool | types.StrSequenceOrSet | None = None,
        unknown: str | None = None,
    ):
        return self._do_load(
            data,
            many=self.many if many is None else many,
            partial=self.partial if partial is None else partial,
            unknown=self.unknown if unknown is None else unknown,
            postprocess=True,
        )

    def loads(
        self,
        s,
        *,
        many: bool | None = None,
        partial: bool | types.StrSequenceOrSet | None = None,
        unknown: str | None = None,
        **kwargs,
    ):
        data = self.opts.render_module.loads(s, **kwargs)
        return self.load(data, many=many, partial=partial, unknown=unknown)

    def validate(
        self,
        data,
        *,
        many: bool | None = None,
        partial: bool | types.StrSequenceOrSet | None = None,
    ):
        try:
            self._do_load(
                data,
                many=self.many if many is None else many,
                partial=self.partial if partial is None else partial,
                unknown=self.unknown,
                postprocess=False,
            )
        except ValidationError as error:
            return error.messages
        return {}

    def _do_load(self, data, *, many, partial, unknown, postprocess):
        error_store = ErrorStore()

        if many:
            if not is_sequence_but_not_string(data):
                error_store.store_error([self.error_messages["type"]], SCHEMA)
                raise ValidationError(error_store.errors, data=data)
            result = []
            for index, item in enumerate(data):
                try:
                    result.append(
                        self._do_load(
                            item,
                            many=False,
                            partial=partial,
                            unknown=unknown,
                            postprocess=postprocess,
                        )
                    )
                except ValidationError as error:
                    if self.opts.index_errors:
                        error_store.store_error(error.messages, index=index)
                    else:
                        error_store.store_error(error.messages)
            if error_store.errors:
                raise ValidationError(error_store.errors, data=data, valid_data=result)
            if postprocess:
                result = self._invoke_processors(POST_LOAD, result, many=True)
            return result

        if not isinstance(data, Mapping):
            error_store.store_error([self.error_messages["type"]], SCHEMA)
            raise ValidationError(error_store.errors, data=data)

        data = self._invoke_processors(PRE_LOAD, data, many=False)
        result = self._deserialize(
            data,
            error_store=error_store,
            many=False,
            partial=partial,
            unknown=unknown,
        )

        if not error_store.errors:
            self._invoke_field_validators(error_store, data, many=False)
            self._invoke_schema_validators(
                error_store,
                data,
                result,
                many=False,
                partial=partial,
                unknown=unknown,
            )

        if error_store.errors:
            raise ValidationError(error_store.errors, data=data, valid_data=result)

        if postprocess:
            result = self._invoke_processors(POST_LOAD, result, many=False)
        return result

    def _deserialize(self, data, *, error_store, many, partial, unknown):
        result = self.dict_class()
        partial_is_collection = is_collection(partial)

        for attr_name, field_obj in self.load_fields.items():
            field_name = field_obj.data_key if field_obj.data_key is not None else attr_name
            raw_value = data.get(field_name, missing)
            field_partial = partial is True or (
                partial_is_collection and attr_name in partial
            )
            if partial_is_collection:
                nested = self._nested_option(attr_name, partial)
                if nested:
                    field_partial = nested

            try:
                value = field_obj.deserialize(
                    raw_value,
                    attr_name,
                    data,
                    partial=field_partial,
                )
            except ValidationError as error:
                error_store.store_error(error.messages, field_name)
                if error.valid_data is not None:
                    set_value(result, field_obj.attribute or attr_name, error.valid_data)
            else:
                if value is not missing:
                    set_value(result, field_obj.attribute or attr_name, value)

        if unknown != EXCLUDE:
            known = {
                field.data_key if field.data_key is not None else name
                for name, field in self.load_fields.items()
            }
            for key, value in data.items():
                if key in known:
                    continue
                if unknown == INCLUDE:
                    result[key] = value
                elif unknown == RAISE:
                    error_store.store_error(
                        [self.error_messages["unknown"]], key
                    )
        return result

    def _invoke_field_validators(self, error_store, data, *, many):
        for attr_name, _, validator_kwargs in self._hooks[VALIDATES]:
            field_obj = self.fields.get(attr_name)
            if field_obj is None:
                continue
            field_name = field_obj.data_key if field_obj.data_key is not None else attr_name
            value = data.get(field_name, missing)
            if value is missing:
                continue
            try:
                validator = getattr(self, attr_name)
                validator(value, **validator_kwargs)
            except ValidationError as error:
                error_store.store_error(error.messages, field_name)

    def _invoke_schema_validators(
        self, error_store, data, original_data, *, many, partial, unknown
    ):
        for attr_name, hook_many, validator_kwargs in self._hooks[VALIDATES_SCHEMA]:
            if hook_many != many:
                continue
            validator = getattr(self, attr_name)
            kwargs = dict(validator_kwargs)
            kwargs.setdefault("many", many)
            kwargs.setdefault("partial", partial)
            kwargs.setdefault("unknown", unknown)
            if kwargs.pop("pass_original", False):
                args = (data, original_data)
            else:
                args = (data,)
            try:
                validator(*args, **kwargs)
            except ValidationError as error:
                field_name = kwargs.pop("field_name", SCHEMA)
                error_store.store_error(error.messages, field_name)

    def _validate(self, data, *, many, partial, unknown):
        try:
            self._do_load(
                data,
                many=many,
                partial=partial,
                unknown=unknown,
                postprocess=False,
            )
        except ValidationError as error:
            return error.messages
        return {}

    def get_attribute(self, obj, attr, default):
        return get_value(obj, attr, default)