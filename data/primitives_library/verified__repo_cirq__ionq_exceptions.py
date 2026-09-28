from __future__ import annotations

import requests


class IonQException(Exception):
    """An exception for errors coming from IonQ's API."""

    def __init__(self, message, status_code: int | None = None):
        super().__init__(f"Status code: {status_code}, Message: '{message}'")
        self.status_code = status_code


class IonQNotFoundException(IonQException):
    """An exception for errors from IonQ's API when a resource is not found."""

    def __init__(self, message):
        super().__init__(message, status_code=requests.codes.not_found)


class IonQUnsuccessfulJobException(IonQException):
    """An exception for attempting to get info about an unsuccessful job."""

    def __init__(self, job_id: str, status: str):
        super().__init__(f"Job {job_id} was {status}.")


class IonQSerializerMixedGatesetsException(Exception):
    """An exception for mixed IonQ serializer gate set types."""

    def __init__(self, message):
        super().__init__(f"Message: '{message}'")


class NotSupportedPauliexpParameters(Exception):
    """An exception for unsupported pauliexp serialization parameters."""

    def __init__(self, message):
        super().__init__(f"Message: '{message}'")