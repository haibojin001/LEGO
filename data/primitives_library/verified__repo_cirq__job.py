from __future__ import annotations

import json
import time
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import cirq
from cirq._doc import document
from cirq_ionq import ionq_exceptions, results

if TYPE_CHECKING:
    import cirq_ionq


def _little_endian_to_big(value: int, bit_count: int) -> int:
    return cirq.big_endian_bits_to_int(
        cirq.big_endian_int_to_bits(value, bit_count=bit_count)[::-1]
    )


class Job:
    """A job created on the IonQ API."""

    TERMINAL_STATES = ('completed', 'canceled', 'failed', 'deleted')
    document(
        TERMINAL_STATES,
        'States of the IonQ API job from which the job cannot transition. '
        'Note that deleted can only exist in a return call from a delete '
        '(subsequent calls will return not found).',
    )

    NON_TERMINAL_STATES = ('ready', 'submitted', 'running')
    document(
        NON_TERMINAL_STATES,
        'States of the IonQ API job which can transition to other states.',
    )

    ALL_STATES = TERMINAL_STATES + NON_TERMINAL_STATES
    document(ALL_STATES, 'All states that an IonQ API job can exist in.')

    UNSUCCESSFUL_STATES = ('canceled', 'failed', 'deleted')
    document(
        UNSUCCESSFUL_STATES,
        'States of the IonQ API job when it was not successful and so does not have any '
        'data associated with it beyond an id and a status.',
    )

    def __init__(self, client: cirq_ionq.ionq_client._IonQClient, job_dict: dict):
        """Constructs a job object from an IonQ API job response."""
        self._client = client
        self._job = job_dict

    def _refresh_job(self):
        """Refreshes this job unless its currently known state is terminal."""
        if self._job['status'] not in self.TERMINAL_STATES:
            self._job = self._client.get_job(self.job_id())

    def _check_if_unsuccessful(self):
        if self.status() in self.UNSUCCESSFUL_STATES:
            raise ionq_exceptions.IonQUnsuccessfulJobException(self.job_id(), self.status())

    def job_id(self) -> str:
        """Returns the API identifier of this job."""
        return self._job['id']

    def status(self) -> str:
        """Returns the current status of the job."""
        self._refresh_job()
        return self._job['status']

    def target(self) -> str:
        """Returns the backend target for this job."""
        self._check_if_unsuccessful()
        return self._job['backend']

    def name(self) -> str:
        """Returns the user-provided name of this job."""
        self._check_if_unsuccessful()
        return self._job['name']

    def num_qubits(self, circuit_index=None) -> int:
        """Returns the number of qubits used by this job or batch circuit."""
        self._check_if_unsuccessful()

        if 'metadata' in self._job and circuit_index is not None:
            metadata = self._job['metadata']
            if 'qubit_numbers' in metadata:
                qubit_numbers = json.loads(metadata['qubit_numbers'])
                for index, qubit_number in enumerate(qubit_numbers):
                    if index == circuit_index:
                        return qubit_number

        return int(self._job['stats']['qubits'])

    def repetitions(self) -> int:
        """Returns the requested number of repetitions."""
        self._check_if_unsuccessful()
        return int(self._job['metadata']['shots'])

    def measurement_dict(self, circuit_index=0) -> dict[str, Sequence[int]]:
        """Returns a mapping from measurement keys to measured qubit indices."""
        measurement_dict: dict[str, Sequence[int]] = {}

        if 'metadata' not in self._job:
            return measurement_dict

        metadata = self._job['metadata']
        measurement_metadata = None

        if 'measurements' in metadata:
            measurements = json.loads(metadata['measurements'])
            for index, measurement in enumerate(measurements):
                if index == circuit_index:
                    measurement_metadata = measurement
                    break
        else:
            measurement_metadata = metadata

        if measurement_metadata is None:
            return measurement_dict

        serialized_measurements = ''.join(
            value
            for key, value in measurement_metadata.items()
            if key.startswith('measurement')
        )
        if not serialized_measurements:
            return measurement_dict

        for measurement in serialized_measurements.split(chr(30)):
            key, qubits = measurement.split(chr(31))
            measurement_dict[key] = [int(qubit) for qubit in qubits.split(',')]

        return measurement_dict

    def results(
        self,
        timeout_seconds: int = 7200,
        polling_seconds: int = 1,
        sharpen: bool | None = None,
        extra_query_params: dict | None = None,
    ) -> (
        results.QPUResult
        | results.SimulatorResult
        | list[results.QPUResult]
        | list[results.SimulatorResult]
    ):
        """Waits for the job and returns its results."""
        time_waited = 0

        while time_waited < timeout_seconds:
            status = self.status()

            if status == 'completed':
                break

            if status in self.UNSUCCESSFUL_STATES:
                raise ionq_exceptions.IonQUnsuccessfulJobException(self.job_id(), status)

            if status not in self.NON_TERMINAL_STATES:
                raise RuntimeError(f'Unexpected job status: {status}')

            time.sleep(polling_seconds)
            time_waited += polling_seconds
        else:
            raise TimeoutError(f'Job timed out after {timeout_seconds} seconds.')

        raw_results = self._client.get_results(
            self.job_id(), sharpen=sharpen, extra_query_params=extra_query_params
        )
        is_batch = isinstance(raw_results, list)
        raw_result_list = raw_results if is_batch else [raw_results]

        target = self.target()
        converted_results = []

        for circuit_index, raw_result in enumerate(raw_result_list):
            num_qubits = self.num_qubits(circuit_index)
            measurement_dict = self.measurement_dict(circuit_index)
            histogram = raw_result['histogram']

            if target == 'qpu':
                repetitions = self.repetitions()
                counts = {
                    _little_endian_to_big(int(value), num_qubits): int(
                        round(probability * repetitions)
                    )
                    for value, probability in histogram.items()
                }

                if sum(counts.values()) != repetitions:
                    warnings.warn(
                        'The total number of counts returned by IonQ does not match the '
                        'number of requested repetitions. This can occur due to rounding.',
                        stacklevel=2,
                    )

                converted_results.append(
                    results.QPUResult(
                        counts=counts,
                        num_qubits=num_qubits,
                        measurement_dict=measurement_dict,
                    )
                )
            else:
                probabilities = {
                    _little_endian_to_big(int(value), num_qubits): probability
                    for value, probability in histogram.items()
                }
                converted_results.append(
                    results.SimulatorResult(
                        probabilities=probabilities,
                        num_qubits=num_qubits,
                        repetitions=self.repetitions(),
                        measurement_dict=measurement_dict,
                    )
                )

        if is_batch:
            return converted_results
        return converted_results[0]