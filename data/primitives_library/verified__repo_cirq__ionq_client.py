from __future__ import annotations

import datetime
import json.decoder as jd
import platform
import sys
import time
import urllib.parse
import warnings
from collections.abc import Callable
from typing import Any, cast

import requests

import cirq_ionq
from cirq import __version__ as cirq_version
from cirq_ionq import ionq_exceptions

RETRIABLE_FOR_GETS = {requests.codes.conflict}
RETRIABLE_STATUS_CODES = {
    requests.codes.too_many_requests,
    requests.codes.internal_server_error,
    requests.codes.bad_gateway,
    requests.codes.service_unavailable,
    *range(520, 530),
}


def _is_retriable(code, method):
    return code in RETRIABLE_STATUS_CODES or (method == "GET" and code in RETRIABLE_FOR_GETS)


class _IonQClient:
    """Internal client used to communicate with IonQ's HTTP API."""

    SUPPORTED_TARGETS = {"qpu", "simulator"}
    SUPPORTED_VERSIONS = {"v0.4"}

    def __init__(
        self,
        remote_host: str,
        api_key: str,
        default_target: str | None = None,
        api_version: str = "v0.4",
        max_retry_seconds: int = 3600,
        verbose: bool = False,
    ):
        url = urllib.parse.urlparse(remote_host)
        assert url.scheme and url.netloc, (
            f"Specified remote_host {remote_host} is not a valid url, for example "
            "http://example.com"
        )
        assert api_version in self.SUPPORTED_VERSIONS, (
            f"Only api v0.4 is accepted but was {api_version}"
        )
        assert default_target is None or default_target in self.SUPPORTED_TARGETS, (
            f"Target can only be one of {self.SUPPORTED_TARGETS} but was {default_target}."
        )
        assert max_retry_seconds >= 0, "Negative retry not possible without time machine."

        self.url = f"{url.scheme}://{url.netloc}/{api_version}"
        self.headers = self.api_headers(api_key)
        self.default_target = default_target
        self.max_retry_seconds = max_retry_seconds
        self.verbose = verbose
        self.batch_mode = False

    @staticmethod
    def api_headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"apiKey {api_key}",
            "User-Agent": (
                f"cirq/{cirq_version} cirq-ionq/{cirq_ionq.__version__} "
                f"python/{platform.python_version()}"
            ),
        }

    def _target(self, target: str | None) -> str:
        actual_target = target if target is not None else self.default_target
        if actual_target is None:
            raise ionq_exceptions.IonQException(
                "No target was specified. Specify a target or set a default target."
            )
        if actual_target not in self.SUPPORTED_TARGETS:
            raise ionq_exceptions.IonQException(
                f"Target can only be one of {self.SUPPORTED_TARGETS} but was {actual_target}."
            )
        return actual_target

    def create_job(
        self,
        serialized_program: cirq_ionq.SerializedProgram,
        repetitions: int | None = None,
        target: str | None = None,
        name: str | None = None,
        extra_query_params: dict | None = None,
        batch_mode: bool = False,
    ) -> dict:
        actual_target = self._target(target)
        payload: dict[str, Any] = {
            "backend": actual_target,
            "type": "ionq.multi-circuit.v1" if batch_mode else "ionq.circuit.v1",
            "lang": "json",
            "input": serialized_program.input,
        }

        if name:
            payload["name"] = name

        payload["metadata"] = serialized_program.metadata

        if serialized_program.settings:
            payload["settings"] = serialized_program.settings

        payload["shots"] = str(repetitions)
        payload["metadata"]["shots"] = str(repetitions)

        if serialized_program.error_mitigation:
            if "settings" not in payload:
                payload["settings"] = {}
            payload["settings"]["error_mitigation"] = serialized_program.error_mitigation

        if serialized_program.compilation:
            if "settings" not in payload:
                payload["settings"] = {}
            payload["settings"]["compilation"] = serialized_program.compilation

        if serialized_program.noise:
            payload["noise"] = serialized_program.noise

        if serialized_program.dry_run:
            payload["dry_run"] = serialized_program.dry_run
            if payload["backend"] == "simulator":
                warnings.warn(
                    "Please note that the `dry_run` option has no effect on the simulator target."
                )

        if extra_query_params:
            payload.update(extra_query_params)

        def request():
            return requests.post(f"{self.url}/jobs", json=payload, headers=self.headers)

        response = self._make_request(request, payload).json()
        self.batch_mode = batch_mode
        return response

    def get_job(self, job_id: str) -> dict:
        def request():
            return requests.get(f"{self.url}/jobs/{job_id}", headers=self.headers)

        return self._make_request(request, {}).json()

    def get_results(
        self, job_id: str, sharpen: bool | None = None, extra_query_params: dict | None = None
    ):
        params: dict[str, Any] = {}

        if sharpen is not None:
            params["sharpen"] = sharpen

        if extra_query_params:
            params.update(extra_query_params)

        def request():
            return requests.get(
                f"{self.url}/jobs/{job_id}/results", params=params, headers=self.headers
            )

        return self._make_request(request, params).json()

    def cancel_job(self, job_id: str) -> dict:
        payload = {"status": "canceled"}

        def request():
            return requests.put(
                f"{self.url}/jobs/{job_id}/status", json=payload, headers=self.headers
            )

        return self._make_request(request, payload).json()

    def delete_job(self, job_id: str) -> dict:
        def request():
            return requests.delete(f"{self.url}/jobs/{job_id}", headers=self.headers)

        return self._make_request(request, {}).json()

    def list_jobs(
        self, status: str | None = None, limit: int = 100, batch_mode: bool = False
    ) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}

        if status is not None:
            params["status"] = status

        if batch_mode:
            params["type"] = "ionq.multi-circuit.v1"

        def request():
            return requests.get(f"{self.url}/jobs", params=params, headers=self.headers)

        return self._make_request(request, params).json()

    def get_current_calibration(self) -> dict:
        def request():
            return requests.get(f"{self.url}/calibrations/current", headers=self.headers)

        return self._make_request(request, {}).json()

    def list_calibrations(
        self,
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        limit: int = 100,
    ) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}

        if start is not None:
            params["start"] = start.isoformat()

        if end is not None:
            params["end"] = end.isoformat()

        def request():
            return requests.get(f"{self.url}/calibrations", params=params, headers=self.headers)

        return self._make_request(request, params).json()

    def _make_request(
        self, request: Callable[[], requests.Response], json: dict
    ) -> requests.Response:
        delay_seconds = 1
        start_time = datetime.datetime.now()
        response: requests.Response | None = None

        while True:
            response = request()

            if response.status_code < 400:
                return response

            request_method = getattr(getattr(response, "request", None), "method", None)
            if not _is_retriable(response.status_code, request_method):
                break

            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            if elapsed >= self.max_retry_seconds:
                break

            if self.verbose:
                print(
                    f"Request failed with status code {response.status_code}. "
                    f"Retrying in {delay_seconds} seconds.",
                    file=sys.stderr,
                )

            time.sleep(delay_seconds)
            delay_seconds *= 2

        assert response is not None

        if response.status_code == requests.codes.not_found:
            raise ionq_exceptions.IonQNotFoundException(
                f"Resource not found. Status code: {response.status_code}"
            )

        try:
            response_json = cast(dict[str, Any], response.json())
            error = response_json.get("error", response_json)
        except (jd.JSONDecodeError, ValueError):
            error = response.text

        raise ionq_exceptions.IonQException(
            f"Request to IonQ API failed with status code {response.status_code}: {error}"
        )