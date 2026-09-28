from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import cirq


class IonQTargetGateset(cirq.TwoQubitCompilationTargetGateset):
    """Compilation target for the native operation set accepted by IonQ hardware."""

    def __init__(self, *, atol: float = 1e-8):
        super().__init__(
            cirq.H,
            cirq.CNOT,
            cirq.SWAP,
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.ZPowGate,
            cirq.XXPowGate,
            cirq.YYPowGate,
            cirq.ZZPowGate,
            cirq.MeasurementGate,
            cirq.GlobalPhaseGate,
            unroll_circuit_op=False,
        )
        self.atol = atol

    def _decompose_single_qubit_operation(
        self, op: cirq.Operation, _
    ) -> Iterator[cirq.OP_TREE]:
        qubit = op.qubits[0]
        unitary = cirq.unitary(op)
        for gate in cirq.single_qubit_matrix_to_gates(unitary, self.atol):
            yield gate(qubit)

    def _decompose_two_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        if not cirq.has_unitary(op):
            return NotImplemented

        first, second = op.qubits
        cz_decomposition = cirq.two_qubit_matrix_to_cz_operations(
            first,
            second,
            cirq.unitary(op),
            allow_partial_czs=False,
        )

        def convert_cz(operation: cirq.Operation, _):
            if operation.gate == cirq.CZ:
                control, target = operation.qubits
                return [
                    cirq.H(target),
                    cirq.CNOT(control, target),
                    cirq.H(target),
                ]
            return operation

        circuit = cirq.map_operations_and_unroll(
            cirq.Circuit(cz_decomposition), convert_cz
        )
        circuit = cirq.merge_k_qubit_unitaries(
            circuit,
            k=1,
            rewriter=lambda operation: self._decompose_single_qubit_operation(
                operation, -1
            ),
        )
        return circuit.all_operations()

    def _decompose_multi_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        if isinstance(op.gate, cirq.CCZPowGate):
            return decompose_all_to_all_connect_ccz_gate(op.gate, op.qubits)
        return NotImplemented

    @property
    def preprocess_transformers(self) -> list[cirq.TRANSFORMER]:
        return [
            cirq.create_transformer_with_kwargs(
                cirq.expand_composite,
                no_decomp=lambda operation: cirq.num_qubits(operation) <= 3,
            )
        ]

    @property
    def postprocess_transformers(self) -> list[cirq.TRANSFORMER]:
        return [cirq.drop_negligible_operations, cirq.drop_empty_moments]

    def __repr__(self) -> str:
        return f"cirq_ionq.IonQTargetGateset(atol={self.atol})"

    def _value_equality_values_(self) -> Any:
        return self.atol

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["atol"])

    @classmethod
    def _from_json_dict_(cls, atol, **kwargs):
        return cls(atol=atol)


def decompose_all_to_all_connect_ccz_gate(
    ccz_gate: cirq.CCZPowGate, qubits: tuple[cirq.Qid, ...]
) -> cirq.OP_TREE:
    """Returns an all-to-all-connectivity decomposition of a CCZ power gate."""
    if len(qubits) != 3:
        raise ValueError(f"Expect 3 qubits for CCZ gate, got {len(qubits)} qubits.")

    first, second, third = qubits
    phase_gate = cirq.T**ccz_gate._exponent

    phase = 1j ** (2 * ccz_gate.global_shift * ccz_gate._exponent)
    if cirq.is_parameterized(phase) and phase.is_complex:
        phase = complex(phase)

    global_phase_ops = (
        [cirq.global_phase_operation(phase)]
        if cirq.is_parameterized(phase) or abs(phase - 1.0) > 0
        else []
    )

    return [
        *global_phase_ops,
        cirq.CNOT(second, third),
        phase_gate(third) ** -1,
        cirq.CNOT(first, third),
        phase_gate(third),
        cirq.CNOT(second, third),
        phase_gate(third) ** -1,
        cirq.CNOT(first, third),
        phase_gate(second),
        phase_gate(third),
        cirq.CNOT(first, second),
        phase_gate(first),
        phase_gate(second) ** -1,
        cirq.CNOT(first, second),
    ]