from __future__ import annotations

import datetime
import os
from collections.abc import Sequence

import cirq
from cirq_ionq import calibration, ionq_client, job, results, sampler, serializer


class Service:
    """Interface for submitting and managing jobs through IonQ's API."""

    def __init__(
        self,
        remote_host: str | None = None,
        api_key: str | None = None,
        default_target: str | None = None,
        api_version='v0.4',
        max_retry_seconds: int = 3600,
        job_settings: dict | None = None,
        verbose=False,
    ):
        self.remote_host = (
            remote_host
            or os.getenv('CIRQ_IONQ_REMOTE_HOST')
            or os.getenv('IONQ_REMOTE_HOST')
            or f'https://api.ionq.co/{api_version}'
        )
        self.job_settings = job_settings or {}
        self.api_key = api_key or os.getenv('CIRQ_IONQ_API_KEY') or os.getenv('IONQ_API_KEY')

        if not self.api_key:
            raise EnvironmentError(
                'Parameter api_key was not specified and the environment variable '
                'IONQ_API_KEY was also not set.'
            )

        self._client = ionq_client._IonQClient(
            remote_host=self.remote_host,
            api_key=self.api_key,
            default_target=default_target,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def run(
        self,
        circuit: cirq.Circuit,
        repetitions: int,
        name: str | None = None,
        target: str | None = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        compilation: dict | None = None,
        error_mitigation: dict | None = None,
        noise: dict | None = None,
        metadata: dict | None = None,
        dry_run: bool = False,
        sharpen: bool | None = None,
        extra_query_params: dict | None = None,
    ) -> cirq.Result:
        resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
        job_results = self.create_job(
            circuit=resolved_circuit,
            repetitions=repetitions,
            name=name,
            target=target,
            compilation=compilation,
            error_mitigation=error_mitigation,
            noise=noise,
            metadata=metadata,
            dry_run=dry_run,
            extra_query_params=extra_query_params,
        ).results(sharpen=sharpen)

        result = job_results[0] if isinstance(job_results, list) else job_results
        params = cirq.ParamResolver(param_resolver)

        if isinstance(result, results.QPUResult):
            return result.to_cirq_result(params=params)
        if isinstance(result, results.SimulatorResult):
            return result.to_cirq_result(params=params, seed=seed)
        raise NotImplementedError(f"Unrecognized job result type '{type(result)}'.")

    def run_batch(
        self,
        circuits: list[cirq.AbstractCircuit],
        repetitions: int,
        name: str | None = None,
        target: str | None = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        compilation: dict | None = None,
        error_mitigation: dict | None = None,
        noise: dict | None = None,
        metadata: dict | None = None,
        dry_run: bool = False,
        sharpen: bool | None = None,
        extra_query_params: dict | None = None,
    ) -> list[cirq.Result]:
        resolved_circuits = [
            cirq.resolve_parameters(circuit, param_resolver) for circuit in circuits
        ]
        job_results = self.create_job(
            circuit=resolved_circuits,
            repetitions=repetitions,
            name=name,
            target=target,
            compilation=compilation,
            error_mitigation=error_mitigation,
            noise=noise,
            metadata=metadata,
            dry_run=dry_run,
            extra_query_params=extra_query_params,
        ).results(sharpen=sharpen)

        if not isinstance(job_results, list):
            job_results = [job_results]

        params = cirq.ParamResolver(param_resolver)
        converted_results = []
        for result in job_results:
            if isinstance(result, results.QPUResult):
                converted_results.append(result.to_cirq_result(params=params))
            elif isinstance(result, results.SimulatorResult):
                converted_results.append(result.to_cirq_result(params=params, seed=seed))
            else:
                raise NotImplementedError(f"Unrecognized job result type '{type(result)}'.")
        return converted_results

    def sampler(
        self,
        target: str | None = None,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> sampler.Sampler:
        return sampler.Sampler(service=self, target=target, seed=seed)

    def create_job(
        self,
        circuit: cirq.AbstractCircuit | Sequence[cirq.AbstractCircuit],
        repetitions: int,
        name: str | None = None,
        target: str | None = None,
        compilation: dict | None = None,
        error_mitigation: dict | None = None,
        noise: dict | None = None,
        metadata: dict | None = None,
        dry_run: bool = False,
        extra_query_params: dict | None = None,
    ) -> job.Job:
        program_serializer = serializer.Serializer()
        if isinstance(circuit, Sequence):
            serialized_program = program_serializer.serialize_many_circuits(circuit)
        else:
            serialized_program = program_serializer.serialize(circuit)

        job_dict = self._client.create_job(
            serialized_program=serialized_program,
            repetitions=repetitions,
            name=name,
            target=target,
            compilation=compilation,
            error_mitigation=error_mitigation,
            noise=noise,
            metadata=metadata,
            dry_run=dry_run,
            job_settings=self.job_settings,
            extra_query_params=extra_query_params,
        )
        return job.Job(client=self._client, job_dict=job_dict)

    def get_job(self, job_id: str) -> job.Job:
        return job.Job(client=self._client, job_dict=self._client.get_job(job_id))

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 100,
        batch_mode: bool | None = None,
    ) -> list[job.Job]:
        job_dicts = self._client.list_jobs(
            status=status,
            limit=limit,
            batch_mode=batch_mode,
        )
        return [job.Job(client=self._client, job_dict=job_dict) for job_dict in job_dicts]

    def get_current_calibration(
        self, target: str | None = None
    ) -> calibration.Calibration:
        calibration_dict = self._client.get_current_calibration(target=target)
        return calibration.Calibration(calibration_dict)

    def list_calibrations(
        self,
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        limit: int = 100,
        target: str | None = None,
    ) -> list[calibration.Calibration]:
        calibration_dicts = self._client.list_calibrations(
            start=start,
            end=end,
            limit=limit,
            target=target,
        )
        return [calibration.Calibration(calibration_dict) for calibration_dict in calibration_dicts]