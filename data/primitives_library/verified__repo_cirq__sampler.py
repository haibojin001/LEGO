from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import cirq
from cirq_ionq import results

if TYPE_CHECKING:
    import cirq_ionq


class Sampler(cirq.Sampler):
    """A Cirq sampler backed by the IonQ service."""

    def __init__(
        self,
        service: cirq_ionq.Service,
        target: str | None,
        timeout_seconds: int | None = None,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ):
        self._service = service
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._seed = seed

    def run_sweep(
        self, program: cirq.AbstractCircuit, params: cirq.Sweepable, repetitions: int = 1
    ) -> Sequence[cirq.Result]:
        resolvers = list(cirq.to_resolvers(params))
        jobs = [
            self._service.create_job(
                circuit=cirq.resolve_parameters(program, resolver),
                repetitions=repetitions,
                target=self._target,
            )
            for resolver in resolvers
        ]

        if self._timeout_seconds is None:
            job_results = [job.results() for job in jobs]
        else:
            job_results = [
                job.results(timeout_seconds=self._timeout_seconds) for job in jobs
            ]

        ionq_results: list[results.QPUResult | results.SimulatorResult] = []
        for job_result in job_results:
            if isinstance(job_result, list):
                ionq_results.extend(job_result)
            else:
                ionq_results.append(job_result)

        output: list[cirq.Result] = []
        for ionq_result, resolver in zip(ionq_results, resolvers):
            if isinstance(ionq_result, results.QPUResult):
                output.append(ionq_result.to_cirq_result(params=resolver))
            elif isinstance(ionq_result, results.SimulatorResult):
                output.append(
                    ionq_result.to_cirq_result(params=resolver, seed=self._seed)
                )
        return output