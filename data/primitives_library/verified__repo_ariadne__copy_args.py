import os
from typing import Any

from starlette.datastructures import UploadFile

try:
    from python_multipart.multipart import File  # type: ignore[import-untyped]
except ImportError:

    class File:
        """Mock upload file used when python-multipart is not installed."""

        file_name: bytes | None = None
        size: int = 0


def copy_args_for_tracing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: copy_args_for_tracing(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [copy_args_for_tracing(item) for item in value]

    if isinstance(value, UploadFile | File):
        return repr_upload_file(value)

    return value


def repr_upload_file(upload_file: UploadFile | File) -> str:
    filename: str | bytes | None

    if isinstance(upload_file, File):
        filename = upload_file.file_name
        mime_type: str | None = "not/available"
        size = upload_file.size
    else:
        filename = upload_file.filename
        mime_type = upload_file.content_type
        file_object = upload_file.file
        file_object.seek(0, os.SEEK_END)
        size = file_object.tell()
        file_object.seek(0)

    if isinstance(filename, bytes):
        filename = filename.decode()

    return (
        f"{type(upload_file)}(mime_type={mime_type}, size={size}, "
        f"filename={filename})"
    )