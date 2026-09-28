from collections.abc import Mapping
from typing import Any

from .exceptions import HttpBadRequestError
from .scalars import ScalarType

SPEC_URL = "https://github.com/jaydenseric/graphql-multipart-request-spec"

FilesDict = Mapping[str, Any]


def combine_multipart_data(
    operations: dict | list, files_map: dict, files: FilesDict
) -> dict | list:
    if not isinstance(operations, dict | list):
        raise HttpBadRequestError(
            f"Invalid type for the 'operations' multipart field ({SPEC_URL})."
        )

    if not isinstance(files_map, dict):
        raise HttpBadRequestError(
            f"Invalid type for the 'map' multipart field ({SPEC_URL})."
        )

    paths_to_files = inverse_files_map(files_map, files)

    if isinstance(operations, list):
        for index, operation in enumerate(operations):
            add_files_to_variables(
                operation.get("variables"),
                f"{index}.variables",
                paths_to_files,
            )
    else:
        add_files_to_variables(
            operations.get("variables"),
            "variables",
            paths_to_files,
        )

    return operations


def inverse_files_map(files_map: dict, files: FilesDict) -> dict:
    result = {}

    for field_name, paths in files_map.items():
        if not isinstance(paths, list):
            raise HttpBadRequestError(
                "Invalid type for the 'map' multipart field entry "
                f"key '{field_name}' array ({SPEC_URL})."
            )

        for index, path in enumerate(paths):
            if not isinstance(path, str):
                raise HttpBadRequestError(
                    "Invalid type for the 'map' multipart field entry key "
                    f"'{field_name}' array index '{index}' value ({SPEC_URL})."
                )

            try:
                file_value = files[field_name]
            except KeyError as error:
                raise HttpBadRequestError(
                    f"File data was missing for entry key '{field_name}' ({SPEC_URL})."
                ) from error

            result[path] = file_value

    return result


def add_files_to_variables(variables: dict | list | None, path: str, files_map: dict):
    if isinstance(variables, dict):
        for name, value in variables.items():
            current_path = f"{path}.{name}"

            if isinstance(value, dict | list):
                add_files_to_variables(value, current_path, files_map)
            elif value is None:
                variables[name] = files_map.get(current_path)

    if isinstance(variables, list):
        for index, value in enumerate(variables):
            current_path = f"{path}.{index}"

            if isinstance(value, dict | list):
                add_files_to_variables(value, current_path, files_map)
            elif value is None:
                variables[index] = files_map.get(current_path)


upload_scalar = ScalarType("Upload")


@upload_scalar.serializer
def serialize_upload(*_):
    raise ValueError("'Upload' scalar serialization is not supported.")


@upload_scalar.literal_parser
def parse_upload_literal(*_):
    raise ValueError("'Upload' scalar literal is not supported.")


@upload_scalar.value_parser
def parse_upload_value(value):
    return value