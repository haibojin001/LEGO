from __future__ import annotations

import datetime

import cirq


class Calibration:
    """An object representing the current calibration state of a QPU."""

    def __init__(self, calibration_dict: dict):
        self._calibration_dict = calibration_dict

    def num_qubits(self) -> int:
        """The number of qubits for the QPU."""
        return int(self._calibration_dict['qubits'])

    def target(self) -> str:
        """The name of the QPU."""
        return self._calibration_dict['target']

    def calibration_time(self, tz: datetime.tzinfo | None = None) -> datetime.datetime:
        """Return a python datetime object for the calibration time.

        Args:
            tz: The timezone for the string. If None, the method uses the platform's local timezone.

        Returns:
            A `datetime` object with the time.
        """
        date_prefix, fractional_suffix = self._calibration_dict['date'].split('.')
        timestamp = f'{date_prefix}.{fractional_suffix[:3]}'
        parsed_time = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')
        return parsed_time.replace(tzinfo=datetime.timezone.utc).astimezone(tz=tz)

    def fidelities(self) -> dict:
        """Returns the metrics (fidelities)."""
        return self._calibration_dict['fidelity']

    def timings(self) -> dict:
        """Returns the gate, measurement, and resetting timings."""
        return self._calibration_dict['timing']

    def connectivity(self) -> set[tuple[cirq.LineQubit, cirq.LineQubit]]:
        """Returns which qubits and can interact with which.

        Returns:
            A set of the possible qubits that can interact as tuples. This contains both
            ordered pairs. If `(cirq.LineQubit(x), cirq.LineQubit(y))` is in the set, then
            `(cirq.LineQubit(y), cirq.LineQubit(x))` is in the set.
        """
        connections = self._calibration_dict['connectivity']

        def to_qubit(value):
            return cirq.LineQubit(int(value))

        direct_connections = {(to_qubit(a), to_qubit(b)) for a, b in connections}
        reverse_connections = {(to_qubit(b), to_qubit(a)) for a, b in connections}
        return direct_connections.union(reverse_connections)