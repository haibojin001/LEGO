from __future__ import annotations

import json
import pprint
import warnings
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Any

from ._imports import import_item
from .corpus.words import generate_corpus_id
from .json_compat import ValidationError, _validator_for_name, get_current_validator
from .reader import get_version
from .warnings import DuplicateCellId, MissingIDFieldWarning

validators: dict[tuple[str, int | None, int | None, bool], Any] = {}

__all__ = [
    "NotebookValidationError",
    "ValidationError",
    "better_validation_error",
    "get_validator",
    "isvalid",
    "iter_validate",
    "normalize",
    "validate",
]


def _relax_additional_properties(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = True if key == "additionalProperties" else _relax_additional_properties(value)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            obj[index] = _relax_additional_properties(value)
    return obj


def _allow_undefined(schema):
    schema["definitions"]["cell"]["oneOf"].append({"$ref": "#/definitions/unrecognized_cell"})
    schema["definitions"]["output"]["oneOf"].append({"$ref": "#/definitions/unrecognized_output"})
    return schema


def _get_schema_json(v, version=None, version_minor=None):
    if (version, version_minor) in v.nbformat_schema:
        schema_name = v.nbformat_schema[(version, version_minor)]
    elif version_minor > v.nbformat_minor:
        schema_name = v.nbformat_schema[(None, None)]
    else:
        raise AttributeError("Cannot find appropriate nbformat schema file.")

    schema_file = Path(v.__file__).parent / schema_name
    with schema_file.open(encoding="utf8") as handle:
        return json.load(handle)


def get_validator(version=None, version_minor=None, relax_add_props=False, name=None):
    if version is None:
        from . import current_nbformat

        version = current_nbformat

    version_module = import_item(f"nbformat.v{version}")
    current_minor = getattr(version_module, "nbformat_minor", 0)

    if version_minor is None:
        version_minor = current_minor

    validator_class = _validator_for_name(name) if name else get_current_validator()
    cache_key = (validator_class.name, version, version_minor, relax_add_props)

    if cache_key not in validators:
        try:
            schema = _get_schema_json(
                version_module,
                version=version,
                version_minor=version_minor,
            )
        except AttributeError:
            return None

        if current_minor < version_minor:
            schema = _relax_additional_properties(schema)
            schema = _allow_undefined(schema)

        if relax_add_props:
            schema = _relax_additional_properties(schema)

        validators[cache_key] = validator_class(schema)

    return validators[cache_key]


def isvalid(nbjson, ref=None, version=None, version_minor=None):
    original = deepcopy(nbjson)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=MissingIDFieldWarning)
            _validate(
                nbjson,
                ref=ref,
                version=version,
                version_minor=version_minor,
                repair_duplicate_cell_ids=False,
            )
    except ValidationError:
        return False
    else:
        return True
    finally:
        if nbjson != original:
            raise AssertionError


def _format_as_index(indices):
    if not indices:
        return ""
    return "[%s]" % "][".join(repr(index) for index in indices)


_ITEM_LIMIT = 16
_STR_LIMIT = 64


def _truncate_obj(obj):
    if isinstance(obj, dict):
        result = {key: _truncate_obj(value) for key, value in list(obj.items())[:_ITEM_LIMIT]}

        if isinstance(result.get("cells"), list):
            result["cells"] = [f"...{len(obj['cells'])} cells..."]

        if isinstance(result.get("outputs"), list):
            result["outputs"] = [f"...{len(obj['outputs'])} outputs..."]

        if len(obj) > _ITEM_LIMIT:
            result["..."] = f"{len(obj) - _ITEM_LIMIT} keys truncated"

        return result

    if isinstance(obj, list):
        result = [_truncate_obj(value) for value in obj[:_ITEM_LIMIT]]
        if len(obj) > _ITEM_LIMIT:
            result.append(f"...{len(obj) - _ITEM_LIMIT} items truncated...")
        return result

    if isinstance(obj, str):
        result = obj[:_STR_LIMIT]
        if len(obj) > _STR_LIMIT:
            result += "..."
        return result

    return obj


class NotebookValidationError(ValidationError):  # type: ignore[misc]
    def __init__(self, original, ref=None):
        self.original = original
        self.ref = getattr(original, "ref", ref)
        self.message = original.message

    def __getattr__(self, key):
        return getattr(self.original, key)

    def __unicode__(self):
        error = self.original
        instance = _truncate_obj(error.instance)

        return "\n".join(
            [
                error.message,
                "",
                "Failed validating {!r} in {}{}:".format(
                    error.validator,
                    self.ref or "notebook",
                    _format_as_index(list(error.relative_schema_path)[:-1]),
                ),
                "",
                "On instance%s:" % _format_as_index(error.relative_path),
                pprint.pformat(instance, width=78),
            ]
        )

    __str__ = __unicode__


def better_validation_error(error, version, version_minor):
    if not len(error.schema_path):
        return error

    key = error.schema_path[-1]
    ref = None

    if key.endswith("Of") and isinstance(error.instance, dict):
        if "cell_type" in error.instance:
            ref = f"{error.instance['cell_type']}_cell"
        elif "output_type" in error.instance:
            ref = error.instance["output_type"]

    if ref:
        try:
            validate(
                error.instance,
                ref,
                version=version,
                version_minor=version_minor,
            )
        except ValidationError as sub_error:
            error.relative_path.extend(sub_error.relative_path)
            sub_error.relative_path = error.relative_path
            replacement = better_validation_error(sub_error, version, version_minor)
            if replacement.ref is None:
                replacement.ref = ref
            return replacement
        except Exception:
            pass

    return NotebookValidationError(error, ref)


def _normalize_cell_ids(nbdict, repair_duplicate_cell_ids=True):
    changes = {}
    cells = nbdict.get("cells")

    if not isinstance(cells, list):
        return nbdict, changes

    used_ids = set()

    for cell in cells:
        if not isinstance(cell, dict):
            continue

        cell_id = cell.get("id")

        if cell_id is None:
            new_id = generate_corpus_id()
            while new_id in used_ids:
                new_id = generate_corpus_id()

            cell["id"] = new_id
            used_ids.add(new_id)
            changes["cell_ids"] = True

            warnings.warn(
                "Cell is missing an id field, this will become a hard error in future "
                "nbformat versions. You may want to use `normalize()` on your notebooks "
                "before validations.",
                MissingIDFieldWarning,
                stacklevel=3,
            )
            continue

        if cell_id in used_ids:
            if not repair_duplicate_cell_ids:
                raise ValidationError(f"Non-unique cell id '{cell_id}' detected.")

            new_id = generate_corpus_id()
            while new_id in used_ids:
                new_id = generate_corpus_id()

            cell["id"] = new_id
            used_ids.add(new_id)
            changes["cell_ids"] = True

            warnings.warn(
                f"Non-unique cell id '{cell_id}' detected. Corrected to '{new_id}'.",
                DuplicateCellId,
                stacklevel=3,
            )
            continue

        used_ids.add(cell_id)

    return nbdict, changes


def _remove_invalid_metadata(nbdict, version, version_minor, relax_add_props):
    validator = get_validator(
        version=version,
        version_minor=version_minor,
        relax_add_props=relax_add_props,
    )

    if validator is None:
        return nbdict, {}

    changes = {}
    errors = list(validator.iter_errors(nbdict))

    for error in errors:
        if error.validator != "additionalProperties":
            continue

        path = list(error.absolute_path)
        if "metadata" not in path:
            continue

        instance = error.instance
        schema = error.schema

        if not isinstance(instance, dict) or not isinstance(schema, dict):
            continue

        known = set(schema.get("properties", {}))
        patterns = schema.get("patternProperties", {})

        for key in list(instance):
            allowed = key in known
            if not allowed and patterns:
                import re

                allowed = any(re.search(pattern, key) for pattern in patterns)

            if not allowed:
                del instance[key]
                changes["metadata"] = True

    return nbdict, changes


def normalize(
    nbdict: Any,
    version=None,
    version_minor=None,
    relax_add_props=False,
    strip_invalid_metadata=False,
    repair_duplicate_cell_ids=True,
):
    nbdict = deepcopy(nbdict)

    if version is None:
        version, detected_minor = get_version(nbdict)
        if version_minor is None:
            version_minor = detected_minor

    if version_minor is None:
        version_minor = 0

    changes = {}

    if version >= 4 and version_minor >= 5:
        nbdict, cell_changes = _normalize_cell_ids(
            nbdict,
            repair_duplicate_cell_ids=repair_duplicate_cell_ids,
        )
        changes.update(cell_changes)

    if strip_invalid_metadata:
        nbdict, metadata_changes = _remove_invalid_metadata(
            nbdict,
            version,
            version_minor,
            relax_add_props,
        )
        changes.update(metadata_changes)

    return changes, nbdict


def _validator_with_ref(validator, ref):
    if ref is None:
        return validator

    schema = validator.schema["definitions"][ref]

    if hasattr(validator, "evolve"):
        return validator.evolve(schema=schema)

    validator_class = type(validator)
    return validator_class(schema)


def _validate(
    nbjson,
    ref=None,
    version=None,
    version_minor=None,
    relax_add_props=False,
    strip_invalid_metadata=False,
    repair_duplicate_cell_ids=True,
):
    if version is None:
        version, detected_minor = get_version(nbjson)
        if version_minor is None:
            version_minor = detected_minor

    if version_minor is None:
        version_minor = 0

    if ref is None:
        _, nbjson = normalize(
            nbjson,
            version=version,
            version_minor=version_minor,
            relax_add_props=relax_add_props,
            strip_invalid_metadata=strip_invalid_metadata,
            repair_duplicate_cell_ids=repair_duplicate_cell_ids,
        )

    validator = get_validator(
        version=version,
        version_minor=version_minor,
        relax_add_props=relax_add_props,
    )

    if validator is None:
        raise ValidationError(f"No schema known for notebook format {version}.{version_minor}")

    validator = _validator_with_ref(validator, ref)

    for error in validator.iter_errors(nbjson):
        raise better_validation_error(error, version, version_minor)


def iter_validate(
    nbjson,
    ref=None,
    version=None,
    version_minor=None,
    relax_add_props=False,
    strip_invalid_metadata=False,
    repair_duplicate_cell_ids=True,
):
    if version is None:
        version, detected_minor = get_version(nbjson)
        if version_minor is None:
            version_minor = detected_minor

    if version_minor is None:
        version_minor = 0

    if ref is None:
        _, nbjson = normalize(
            nbjson,
            version=version,
            version_minor=version_minor,
            relax_add_props=relax_add_props,
            strip_invalid_metadata=strip_invalid_metadata,
            repair_duplicate_cell_ids=repair_duplicate_cell_ids,
        )

    validator = get_validator(
        version=version,
        version_minor=version_minor,
        relax_add_props=relax_add_props,
    )

    if validator is None:
        raise ValidationError(f"No schema known for notebook format {version}.{version_minor}")

    validator = _validator_with_ref(validator, ref)

    for error in validator.iter_errors(nbjson):
        yield better_validation_error(error, version, version_minor)


def validate(
    nbjson,
    ref=None,
    version=None,
    version_minor=None,
    relax_add_props=False,
    strip_invalid_metadata=False,
    repair_duplicate_cell_ids=True,
):
    for error in iter_validate(
        nbjson,
        ref=ref,
        version=version,
        version_minor=version_minor,
        relax_add_props=relax_add_props,
        strip_invalid_metadata=strip_invalid_metadata,
        repair_duplicate_cell_ids=repair_duplicate_cell_ids,
    ):
        raise error