from __future__ import annotations

from collections.abc import Iterator
from types import NotImplementedType
from typing import Any

import numpy as np

import cirq
from cirq import linalg, ops
from cirq_ionq.ionq_native_gates import GPI2Gate, GPIGate, MSGate, ZZGate


class IonqNativeGatesetBase(cirq.TwoQubitCompilationTargetGateset):
    """Common compiler support for IonQ's native gate families."""

    def __init__(self, *gates, atol: float = 1e-8):
        super().__init__(*gates, unroll_circuit_op=False)
        self.atol = atol

    def _decompose_single_qubit_operation(
        self, op: cirq.Operation, _
    ) -> Iterator[cirq.OP_TREE]:
        qubit = op.qubits[0]
        unitary = cirq.unitary(op)

        yield cirq.global_phase_operation(-1j)
        for gate in self.single_qubit_matrix_to_native_gates(unitary):
            yield gate.on(qubit)

    def _decompose_two_qubit_operation(
        self, op: cirq.Operation, _
    ) -> NotImplementedType | cirq.OP_TREE:
        if not cirq.has_unitary(op):
            return NotImplemented

        first, second = op.qubits
        cz_decomposition = cirq.two_qubit_matrix_to_cz_operations(
            first,
            second,
            cirq.unitary(op),
            allow_partial_czs=False,
            atol=self.atol,
        )

        def convert_cz(candidate: cirq.Operation, _) -> cirq.OP_TREE:
            if candidate.gate != cirq.CZ:
                return candidate
            control, target = candidate.qubits
            return [
                self._hadamard(target)
                + self._cnot(control, target)
                + self._hadamard(target)
            ]

        circuit = cirq.map_operations_and_unroll(
            cirq.Circuit(cz_decomposition),
            convert_cz,
        )
        rewritten = cirq.merge_k_qubit_unitaries(
            circuit,
            k=1,
            rewriter=lambda candidate: self._decompose_single_qubit_operation(
                candidate, None
            ),
        )
        return rewritten.all_operations()

    def _decompose_multi_qubit_operation(
        self, op: cirq.Operation, _
    ) -> NotImplementedType | cirq.OP_TREE:
        if isinstance(op.gate, cirq.CCZPowGate):
            return self.decompose_all_to_all_connect_ccz_gate(op.gate, op.qubits)
        return NotImplemented

    @property
    def preprocess_transformers(self) -> list[cirq.TRANSFORMER]:
        return [
            cirq.create_transformer_with_kwargs(
                cirq.expand_composite,
                no_decomp=lambda op: cirq.num_qubits(op) <= 3,
            )
        ]

    @property
    def postprocess_transformers(self) -> list[cirq.TRANSFORMER]:
        return [cirq.drop_negligible_operations, cirq.drop_empty_moments]

    def single_qubit_matrix_to_native_gates(self, mat: np.ndarray) -> list[cirq.Gate]:
        z_before, y_rotation, z_after = linalg.deconstruct_single_qubit_matrix_into_angles(
            mat
        )
        tau = 2.0 * np.pi
        return [
            GPI2Gate(phi=(np.pi - z_before) / tau),
            GPIGate(phi=(y_rotation / 2 + z_after / 2 - z_before / 2) / tau),
            GPI2Gate(phi=(np.pi + z_after) / tau),
        ]

    def _value_equality_values_(self) -> Any:
        return self.atol

    def _value_equality_values_cls_(self) -> Any:
        return type(self)

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["atol"])

    @classmethod
    def _from_json_dict_(cls, atol, **kwargs):
        return cls(atol=atol)

    def _hadamard(self, qubit):
        return [
            GPI2Gate(phi=0.25).on(qubit),
            GPIGate(phi=0).on(qubit),
        ]

    def _cnot(self, *qubits):
        raise NotImplementedError()

    def decompose_all_to_all_connect_ccz_gate(
        self, ccz_gate: cirq.CCZPowGate, qubits: tuple[cirq.Qid, ...]
    ) -> cirq.OP_TREE:
        if len(qubits) != 3:
            raise ValueError(f"Expect 3 qubits for CCZ gate, got {len(qubits)} qubits.")

        a, b, c = qubits
        phase_gate = cirq.T**ccz_gate._exponent

        phase = 1j ** (2 * ccz_gate.global_shift * ccz_gate._exponent)
        phase = (
            complex(phase)
            if cirq.is_parameterized(phase) and phase.is_complex
            else phase
        )
        phase_ops = (
            [cirq.global_phase_operation(phase)]
            if cirq.is_parameterized(phase) or abs(phase - 1.0) > 0
            else []
        )

        return [
            *phase_ops,
            self._cnot(b, c),
            phase_gate(c) ** -1,
            self._cnot(a, c),
            phase_gate(c),
            self._cnot(b, c),
            phase_gate(c) ** -1,
            self._cnot(a, c),
            phase_gate(b),
            phase_gate(c),
            self._cnot(a, b),
            phase_gate(a),
            phase_gate(b) ** -1,
            self._cnot(a, b),
        ]


class AriaNativeGateset(IonqNativeGatesetBase):
    """IonQ Aria native target gateset."""

    def __init__(self, *, atol: float = 1e-8):
        super().__init__(GPIGate, GPI2Gate, MSGate, ops.MeasurementGate, atol=atol)

    def __repr__(self) -> str:
        return f"cirq_ionq.AriaNativeGateset(atol={self.atol})"

    def _cnot(self, *qubits):
        control, target = qubits
        return [
            GPI2Gate(phi=1 / 4).on(control),
            MSGate(phi0=0, phi1=0).on(control, target),
            GPI2Gate(phi=1 / 2).on(target),
            GPI2Gate(phi=1 / 2).on(control),
            GPI2Gate(phi=-1 / 4).on(control),
        ]


class ForteNativeGateset(IonqNativeGatesetBase):
    """IonQ Forte native target gateset."""

    def __init__(self, *, atol: float = 1e-8):
        super().__init__(GPIGate, GPI2Gate, ZZGate, ops.MeasurementGate, atol=atol)

    def __repr__(self) -> str:
        return f"cirq_ionq.ForteNativeGateset(atol={self.atol})"

    def _cnot(self, *qubits):
        control, target = qubits
        return [
            GPI2Gate(phi=0).on(target),
            GPIGate(phi=-0.125).on(target),
            GPI2Gate(phi=0.5).on(target),
            ZZGate(theta=0.25).on(control, target),
            GPI2Gate(phi=0.75).on(control),
            GPIGate(phi=0.125).on(control),
            GPI2Gate(phi=0.5).on(control),
            GPI2Gate(phi=1.25).on(target),
            GPIGate(phi=0.5).on(target),
            GPI2Gate(phi=0.5).on(target),
        ]