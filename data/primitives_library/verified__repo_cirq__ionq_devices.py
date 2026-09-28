from __future__ import annotations

from collections.abc import Sequence

import cirq
from cirq_ionq import ionq_gateset


class IonQAPIDevice(cirq.Device):
    """A device representing the gates available through the IonQ API.

    This device accepts operations supported by the IonQ target gateset and
    exposes all ordered pairs of distinct device qubits as connected.
    """

    def __init__(self, qubits: Sequence[cirq.LineQubit] | int, atol=1e-8):
        """Creates an IonQ API device.

        Args:
            qubits: Device qubits, or an integer giving the number of line
                qubits to create beginning at index zero.
            atol: Absolute tolerance used by gate decomposition calculations.
        """
        if isinstance(qubits, int):
            device_qubits = cirq.LineQubit.range(qubits)
        else:
            device_qubits = qubits

        self.qubits = frozenset(device_qubits)
        self.atol = atol
        self.gateset = ionq_gateset.IonQTargetGateset()
        self._metadata = cirq.DeviceMetadata(
            self.qubits,
            [
                (first_qubit, second_qubit)
                for first_qubit in self.qubits
                for second_qubit in self.qubits
                if first_qubit != second_qubit
            ],
        )

    @property
    def metadata(self) -> cirq.DeviceMetadata:
        return self._metadata

    def validate_operation(self, operation: cirq.Operation):
        if operation.gate is None:
            raise ValueError(
                f'IonQAPIDevice does not support operations with no gates {operation}.'
            )
        if not self.is_api_gate(operation):
            raise ValueError(f'IonQAPIDevice has unsupported gate {operation.gate}.')
        if self.metadata.qubit_set.isdisjoint(operation.qubits):
            raise ValueError(f'Operation with qubits not on the device. Qubits: {operation.qubits}')

    def is_api_gate(self, operation: cirq.Operation) -> bool:
        return operation in self.gateset