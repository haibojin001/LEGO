from __future__ import annotations

import collections
from collections import Counter
from collections.abc import Sequence

import numpy as np

import cirq


def _pretty_str_dict(values: dict[int, object], num_qubits: int) -> str:
    return '\n'.join(f'{value:0{num_qubits}b}: {count}' for value, count in values.items())


class QPUResult:
    """The results of running on an IonQ QPU."""

    def __init__(
        self, counts: dict[int, int], num_qubits: int, measurement_dict: dict[str, Sequence[int]]
    ):
        self._counts = collections.OrderedDict(sorted(counts.items()))
        self._num_qubits = num_qubits
        self._measurement_dict = measurement_dict
        self._repetitions = sum(self._counts.values())

    def num_qubits(self) -> int:
        """Returns the number of qubits the circuit was run on."""
        return self._num_qubits

    def repetitions(self) -> int:
        """Returns the number of times the circuit was run."""
        return self._repetitions

    def ordered_results(self, key: str | None = None) -> list[int]:
        """Returns consistently ordered measurement results as big-endian integers."""
        if key is not None and key not in self._measurement_dict:
            raise ValueError(
                f'Measurement key {key} is not a key for a measurement gate in the'
                'circuit that produced these results.'
            )

        targets = self._measurement_dict[key] if key is not None else range(self.num_qubits())
        results: list[int] = []
        for state, occurrences in self._counts.items():
            bits = [(state >> (self.num_qubits() - target - 1)) & 1 for target in targets]
            value = sum(bit * (1 << index) for index, bit in enumerate(reversed(bits)))
            results.extend([value] * occurrences)
        return results

    def counts(self, key: str | None = None) -> Counter[int]:
        """Returns processed counts for all results or a selected measurement key."""
        if key is None:
            return collections.Counter(self._counts)

        if key not in self._measurement_dict:
            raise ValueError(
                f'Measurement key {key} is not a key for a measurement gate in the'
                'circuit that produced these results.'
            )

        return collections.Counter(self.ordered_results(key))

    def measurement_dict(self) -> dict[str, Sequence[int]]:
        """Returns the mapping of measurement keys to qubit target indices."""
        return self._measurement_dict

    def to_cirq_result(self, params: cirq.ParamResolver | None = None) -> cirq.Result:
        """Converts these results to a Cirq result."""
        if len(self.measurement_dict()) == 0:
            raise ValueError(
                'Can convert to cirq results only if the circuit had measurement gates '
                'with measurement keys.'
            )

        measurements = {}
        for key, targets in self.measurement_dict().items():
            results = self.ordered_results(key)
            measurements[key] = np.array(
                [cirq.big_endian_int_to_bits(value, bit_count=len(targets)) for value in results]
            )

        return cirq.ResultDict(params=params or cirq.ParamResolver({}), measurements=measurements)

    def __eq__(self, other):
        if not isinstance(other, QPUResult):
            return NotImplemented
        return (
            self._counts == other._counts
            and self._num_qubits == other._num_qubits
            and self._measurement_dict == other._measurement_dict
            and self._repetitions == other._repetitions
        )

    def __str__(self) -> str:
        return _pretty_str_dict(self._counts, self._num_qubits)


class SimulatorResult:
    """The results of running on an IonQ simulator."""

    def __init__(
        self,
        probabilities: dict[int, float],
        num_qubits: int,
        measurement_dict: dict[str, Sequence[int]],
        repetitions: int,
    ):
        self._probabilities = probabilities
        self._num_qubits = num_qubits
        self._measurement_dict = measurement_dict
        self._repetitions = repetitions

    def num_qubits(self) -> int:
        """Returns the number of qubits the circuit was run on."""
        return self._num_qubits

    def repetitions(self) -> int:
        """Returns the number of times the circuit was run."""
        return self._repetitions

    def probabilities(self, key: str | None = None) -> dict[int, float]:
        """Returns probabilities for all qubits or for a measurement key."""
        if key is None:
            return self._probabilities

        if key not in self._measurement_dict:
            raise ValueError(
                f'Measurement key {key} is not a key for a measurement gate in the'
                'circuit that produced these results.'
            )

        targets = self._measurement_dict[key]
        result: dict[int, float] = {}
        for state, probability in self._probabilities.items():
            bits = [(state >> (self.num_qubits() - target - 1)) & 1 for target in targets]
            value = sum(bit * (1 << index) for index, bit in enumerate(reversed(bits)))
            result[value] = result.get(value, 0.0) + probability
        return result

    def measurement_dict(self) -> dict[str, Sequence[int]]:
        """Returns the mapping of measurement keys to qubit target indices."""
        return self._measurement_dict

    def to_cirq_result(self, params: cirq.ParamResolver | None = None) -> cirq.Result:
        """Samples the simulator probabilities and converts them to a Cirq result."""
        if len(self.measurement_dict()) == 0:
            raise ValueError(
                'Can convert to cirq results only if the circuit had measurement gates '
                'with measurement keys.'
            )

        sampled_states = np.random.choice(
            list(self._probabilities.keys()),
            size=self.repetitions(),
            p=list(self._probabilities.values()),
        )

        measurements = {}
        for key, targets in self.measurement_dict().items():
            values = []
            for state in sampled_states:
                bits = [(state >> (self.num_qubits() - target - 1)) & 1 for target in targets]
                values.append(sum(bit * (1 << index) for index, bit in enumerate(reversed(bits))))
            measurements[key] = np.array(
                [cirq.big_endian_int_to_bits(value, bit_count=len(targets)) for value in values]
            )

        return cirq.ResultDict(params=params or cirq.ParamResolver({}), measurements=measurements)

    def __eq__(self, other):
        if not isinstance(other, SimulatorResult):
            return NotImplemented
        return (
            self._probabilities == other._probabilities
            and self._num_qubits == other._num_qubits
            and self._measurement_dict == other._measurement_dict
            and self._repetitions == other._repetitions
        )

    def __str__(self) -> str:
        return _pretty_str_dict(self._probabilities, self._num_qubits)