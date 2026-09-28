from __future__ import annotations

import os

import fastjsonschema
import jsonschema
from fastjsonschema import JsonSchemaException as _JsonSchemaException
from jsonschema import Draft4Validator as _JsonSchemaValidator
from jsonschema.exceptions import ErrorTree, ValidationError

__all__ = [
    "VALIDATORS",
    "FastJsonSchemaValidator",
    "JsonSchemaValidator",
    "ValidationError",
    "get_current_validator",
]


class JsonSchemaValidator:
    """Validator backed by jsonschema."""

    name = "jsonschema"

    def __init__(self, schema):
        self._schema = schema
        self._default_validator = _JsonSchemaValidator(schema)
        self._validator = self._default_validator

    def validate(self, data):
        self._default_validator.validate(data)

    def iter_errors(self, data, schema=None):
        if schema is None:
            return self._default_validator.iter_errors(data)
        if hasattr(self._default_validator, "evolve"):
            return self._default_validator.evolve(schema=schema).iter_errors(data)
        return self._default_validator.iter_errors(data, schema)

    def error_tree(self, errors):
        return ErrorTree(errors=errors)


class FastJsonSchemaValidator(JsonSchemaValidator):
    """Validator backed by fastjsonschema."""

    name = "fastjsonschema"

    def __init__(self, schema):
        super().__init__(schema)
        self._validator = fastjsonschema.compile(schema)

    def validate(self, data):
        try:
            self._validator(data)
        except _JsonSchemaException as exc:
            raise ValidationError(str(exc), schema_path=exc.path) from exc

    def iter_errors(self, data, schema=None):
        if schema is not None:
            return super().iter_errors(data, schema)

        found = []
        try:
            self._validator(data)
        except _JsonSchemaException as exc:
            found = [ValidationError(str(exc), schema_path=exc.path)]
        return found

    def error_tree(self, errors):
        raise NotImplementedError(
            "JSON schema error introspection not enabled for fastjsonschema"
        )


_VALIDATOR_MAP = [
    ("fastjsonschema", fastjsonschema, FastJsonSchemaValidator),
    ("jsonschema", jsonschema, JsonSchemaValidator),
]

VALIDATORS = [entry[0] for entry in _VALIDATOR_MAP]


def _validator_for_name(validator_name):
    if validator_name not in VALIDATORS:
        raise ValueError(
            f"Invalid validator '{validator_name}' value!\n"
            f"Valid values are: {VALIDATORS}"
        )

    for name, module, validator_class in _VALIDATOR_MAP:
        if module and name == validator_name:
            return validator_class

    raise ValueError(f"Missing validator for {validator_name!r}")


def get_current_validator():
    """Return the validator selected by the environment."""
    return _validator_for_name(os.environ.get("NBFORMAT_VALIDATOR", "fastjsonschema"))