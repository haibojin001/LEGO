from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Collection, Iterator, Sequence
from typing import Any, TYPE_CHECKING

import numpy as np

import cirq
from cirq.devices import line_qubit
from cirq_ionq.ionq_exceptions import (
    IonQSerializerMixedGatesetsException,
    NotSupportedPauliexpParameters,
)
from cirq_ionq.ionq_native_gates import GPI2Gate, GPIGate, MSGate, ZZGate

if TYPE_CHECKING:
    import sympy

    from cirq.ops.pauli_string_phasor import PauliStringPhasorGate


_NATIVE_GATES = cirq.Gateset(
    GPIGate, GPI2Gate, MSGate, ZZGate, cirq.MeasurementGate, unroll_circuit_op=False
)


@dataclasses.dataclass
class SerializedProgram:
    input: dict
    settings: dict
    metadata: dict
    compilation: dict
    error_mitigation: dict
    noise: dict
    dry_run: bool


class Serializer:
    def __init__(self, atol: float = 1e-8):
        self.atol = atol
        self._dispatch: dict[type[cirq.Gate], Callable] = {
            cirq.XPowGate: self._serialize_x_pow_gate,
            cirq.YPowGate: self._serialize_y_pow_gate,
            cirq.ZPowGate: self._serialize_z_pow_gate,
            cirq.XXPowGate: self._serialize_xx_pow_gate,
            cirq.YYPowGate: self._serialize_yy_pow_gate,
            cirq.ZZPowGate: self._serialize_zz_pow_gate,
            cirq.CNotPowGate: self._serialize_cnot_pow_gate,
            cirq.HPowGate: self._serialize_h_pow_gate,
            cirq.SwapPowGate: self._serialize_swap_gate,
            cirq.MeasurementGate: self._serialize_measurement_gate,
            cirq.ops.pauli_string_phasor.PauliStringPhasorGate: (
                self._serialize_pauli_string_phasor_gate
            ),
            GPIGate: self._serialize_gpi_gate,
            GPI2Gate: self._serialize_gpi2_gate,
            MSGate: self._serialize_ms_gate,
            ZZGate: self._serialize_zz_gate,
        }

    def serialize_single_circuit(
        self,
        circuit: cirq.AbstractCircuit,
        job_settings: dict | None = None,
        compilation: dict | None = None,
        error_mitigation: dict | None = None,
        noise: dict | None = None,
        metadata: dict | None = None,
        dry_run: bool = False,
    ) -> SerializedProgram:
        self._validate_circuit(circuit)
        self._validate_qubits(circuit.all_qubits())

        operations = self._serialize_circuit(circuit)
        gateset = "native" if _NATIVE_GATES.validate(circuit) else "qis"
        program_input = {
            "gateset": gateset,
            "qubits": self._num_qubits(circuit),
            "circuit": [operation for operation in operations if operation["gate"] != "meas"],
        }

        measurement_data = self._serialize_measurements(
            operation for operation in operations if operation["gate"] == "meas"
        )
        if metadata is None:
            metadata = measurement_data
        else:
            metadata.update(measurement_data)

        return SerializedProgram(
            input=program_input,
            settings=job_settings or {},
            metadata=metadata or {},
            compilation=compilation or {},
            error_mitigation=error_mitigation or {},
            noise=noise or {},
            dry_run=dry_run,
        )

    def serialize_many_circuits(
        self,
        circuits: list[cirq.AbstractCircuit],
        job_settings: dict | None = None,
        compilation: dict | None = None,
        error_mitigation: dict | None = None,
        noise: dict | None = None,
        metadata: dict | None = None,
        dry_run: bool = False,
    ) -> SerializedProgram:
        for circuit in circuits:
            self._validate_circuit(circuit)
            self._validate_qubits(circuit.all_qubits())

        qubit_count = max(self._num_qubits(circuit) for circuit in circuits)

        gateset: str | None = None
        for circuit in circuits:
            circuit_gateset = "native" if _NATIVE_GATES.validate(circuit) else "qis"
            if gateset is None:
                gateset = circuit_gateset
            elif circuit_gateset != gateset:
                raise IonQSerializerMixedGatesetsException(
                    "For batch circuit submission, all circuits in a batch must contain "
                    "the same type of gates: either 'qis' or 'native' gates."
                )

        program_input: dict[str, Any] = {
            "gateset": gateset,
            "qubits": qubit_count,
            "circuits": [],
        }
        all_measurements: list[dict] = []
        circuit_qubit_counts: list[int] = []

        for circuit in circuits:
            operations = self._serialize_circuit(circuit)
            program_input["circuits"].append(
                {"circuit": [operation for operation in operations if operation["gate"] != "meas"]}
            )
            all_measurements.append(
                self._serialize_measurements(
                    operation for operation in operations if operation["gate"] == "meas"
                )
            )
            circuit_qubit_counts.append(self._num_qubits(circuit))

        batch_metadata = {
            "measurements": json.dumps(all_measurements),
            "qubit_numbers": json.dumps(circuit_qubit_counts),
        }
        if metadata is None:
            metadata = batch_metadata
        else:
            metadata.update(batch_metadata)

        return SerializedProgram(
            input=program_input,
            settings=job_settings or {},
            metadata=metadata or {},
            compilation=compilation or {},
            error_mitigation=error_mitigation or {},
            noise=noise or {},
            dry_run=dry_run,
        )

    def _validate_circuit(self, circuit: cirq.AbstractCircuit):
        if len(circuit) == 0:
            raise ValueError("Cannot serialize empty circuit.")
        if not circuit.are_all_measurements_terminal():
            raise ValueError("All measurements in circuit must be at end of circuit.")

    def _validate_qubits(self, qubits: Collection[cirq.Qid]):
        for qubit in qubits:
            if not isinstance(qubit, line_qubit.LineQubit):
                raise ValueError(
                    "IonQ serializers require qubits to be instances of cirq.LineQubit."
                )
            if qubit.x < 0:
                raise ValueError("IonQ serializers do not support negative qubit indices.")

    def _num_qubits(self, circuit: cirq.AbstractCircuit) -> int:
        qubits = circuit.all_qubits()
        if not qubits:
            return 0
        return max(qubit.x for qubit in qubits) + 1

    def _serialize_circuit(self, circuit: cirq.AbstractCircuit) -> list[dict]:
        return [self._serialize_operation(operation) for operation in circuit.all_operations()]

    def _serialize_operation(self, operation: cirq.Operation) -> dict:
        if not isinstance(operation, cirq.GateOperation):
            raise ValueError(f"Operation {operation!r} is not supported by IonQ.")

        gate = operation.gate
        serializer = self._dispatch.get(type(gate))
        if serializer is None:
            raise ValueError(f"Gate {gate!r} is not supported by IonQ.")

        return serializer(gate, operation.qubits)

    def _serialize_measurements(self, measurement_ops: Iterator[dict]) -> dict:
        measurements: dict[str, list[int]] = {}
        for operation in measurement_ops:
            key = operation["key"]
            targets = operation["targets"]
            if key in measurements:
                measurements[key].extend(targets)
            else:
                measurements[key] = list(targets)

        if not measurements:
            return {}

        encoded = json.dumps(measurements)
        chunk_size = 40
        return {
            f"measurement{i}": encoded[i * chunk_size : (i + 1) * chunk_size]
            for i in range(math.ceil(len(encoded) / chunk_size))
        }

    def _serialize_x_pow_gate(
        self, gate: cirq.XPowGate, qubits: Sequence[cirq.Qid]
    ) -> dict:
        self._require_zero_global_shift(gate)
        exponent = gate.exponent
        target = self._target(qubits)
        if self._near_mod(exponent, 2, 1):
            return {"gate": "x", "target": target}
        if self._near_mod(exponent, 2, 0.5):
            return {"gate": "v", "target": target}
        if self._near_mod(exponent, 2, 1.5):
            return {"gate": "vi", "target": target}
        return {"gate": "rx", "target": target, "rotation": self._rotation(exponent)}

    def _serialize_y_pow_gate(
        self, gate: cirq.YPowGate, qubits: Sequence[cirq.Qid]
    ) -> dict:
        self._require_zero_global_shift(gate)
        exponent = gate.exponent
        target = self._target(qubits)
        if self._near_mod(exponent, 2, 1):
            return {"gate": "y", "target": target}
        return {"gate": "ry", "target": target, "rotation": self._rotation(exponent)}

    def _serialize_z_pow_gate(
        self, gate: cirq.ZPowGate, qubits: Sequence[cirq.Qid]
    ) -> dict:
        self._require_zero_global_shift(gate)
        exponent = gate.exponent
        target = self._target(qubits)
        if self._near_mod(exponent, 2, 1):
            return {"gate": "z", "target": target}
        if self._near_mod(exponent, 2, 0.5):
            return {"gate": "s", "target": target}
        if self._near_mod(exponent, 2, 1.5):
            return {"gate": "si", "target": target}
        if self._near_mod(exponent, 2, 0.25):
            return {"gate": "t", "target": target}
        if self._near_mod(exponent, 2, 1.75):
            return {"gate": "ti", "target": target}
        return {"gate": "rz", "target": target, "rotation": self._rotation(exponent)}

    def _serialize_xx_pow_gate(
        self, gate: cirq.XXPowGate, qubits: Sequence[cirq.Qid]
    ) -> dict:
        self._require_zero_global_shift(gate)
        return {
            "gate": "xx",
            "targets": self._targets(qubits, 2),
            "rotation": self._rotation(gate.exponent),
        }

    def _serialize_yy_pow_gate(
        self, gate: cirq.YYPowGate, qubits: Sequence[cirq.Qid]
    ) -> dict:
        self._require_zero_global_shift(gate)
        return {
            "gate": "yy",
            "targets": self._targets(qubits, 2),
            "rotation": self._rotation(gate.exponent),
        }

    def _serialize_zz_pow_gate(
        self, gate: cirq.ZZPowGate, qubits: Sequence[cirq.Qid]
    ) -> dict:
        self._require_zero_global_shift(gate)
        return {
            "gate": "zz",
            "targets": self._targets(qubits, 2),
            "rotation": self._rotation(gate.exponent),
        }

    def _serialize_cnot_pow_gate(
        self, gate: cirq.CNotPowGate, qubits: Sequence[cirq.Qid]
    ) -> dict:
        self._require_zero_global_shift(gate)
        if not self._near_mod(gate.exponent, 2, 1):
            raise ValueError("IonQ does not support fractional CNOT gates.")
        targets = self._targets(qubits, 2)
        return {"gate": "cnot", "control": targets[0], "target": targets[1]}

    def _serialize_h_pow_gate(
        self, gate: cirq.HPowGate, qubits: Sequence[cirq.Qid]
    ) -> dict:
        self._require_zero_global_shift(gate)
        if not self._near_mod(gate.exponent, 2, 1):
            raise ValueError("IonQ does not support fractional Hadamard gates.")
        return {"gate": "h", "target": self._target(qubits)}

    def _serialize_swap_gate(
        self, gate: cirq.SwapPowGate, qubits: Sequence[cirq.Qid]
    ) -> dict:
        self._require_zero_global_shift(gate)
        if not self._near_mod(gate.exponent, 2, 1):
            raise ValueError("IonQ does not support fractional SWAP gates.")
        return {"gate": "swap", "targets": self._targets(qubits, 2)}

    def _serialize_measurement_gate(
        self, gate: cirq.MeasurementGate, qubits: Sequence[cirq.Qid]
    ) -> dict:
        if gate.invert_mask and any(gate.invert_mask):
            raise ValueError("IonQ does not support inverted measurements.")
        return {
            "gate": "meas",
            "targets": [self._qubit_index(qubit) for qubit in qubits],
            "key": gate.key,
        }

    def _serialize_pauli_string_phasor_gate(
        self, gate: PauliStringPhasorGate, qubits: Sequence[cirq.Qid]
    ) -> dict:
        exponent_neg = getattr(gate, "exponent_neg", None)
        exponent_pos = getattr(gate, "exponent_pos", None)

        if exponent_neg is None or exponent_pos is None:
            raise NotSupportedPauliexpParameters(
                "IonQ cannot serialize this PauliStringPhasorGate."
            )

        try:
            exponent = exponent_neg - exponent_pos
        except TypeError as exc:
            raise NotSupportedPauliexpParameters(
                "IonQ cannot serialize this PauliStringPhasorGate."
            ) from exc

        pauli_string = gate.pauli_string
        coefficient = getattr(pauli_string, "coefficient", 1)
        if coefficient != 1:
            raise NotSupportedPauliexpParameters(
                "IonQ only supports PauliStringPhasorGate instances with coefficient 1."
            )

        paulis = [str(pauli) for pauli in pauli_string]
        if len(paulis) != len(qubits):
            raise NotSupportedPauliexpParameters(
                "IonQ cannot serialize PauliStringPhasorGate with incompatible qubits."
            )

        return {
            "gate": "pauliexp",
            "targets": [self._qubit_index(qubit) for qubit in qubits],
            "paulis": paulis,
            "rotation": self._rotation(exponent),
        }

    def _serialize_gpi_gate(self, gate: GPIGate, qubits: Sequence[cirq.Qid]) -> dict:
        return {
            "gate": "gpi",
            "target": self._target(qubits),
            "phase": self._number(gate.phi),
        }

    def _serialize_gpi2_gate(self, gate: GPI2Gate, qubits: Sequence[cirq.Qid]) -> dict:
        return {
            "gate": "gpi2",
            "target": self._target(qubits),
            "phase": self._number(gate.phi),
        }

    def _serialize_ms_gate(self, gate: MSGate, qubits: Sequence[cirq.Qid]) -> dict:
        return {
            "gate": "ms",
            "targets": self._targets(qubits, 2),
            "phases": [self._number(gate.phi0), self._number(gate.phi1)],
            "angle": self._number(gate.theta),
        }

    def _serialize_zz_gate(self, gate: ZZGate, qubits: Sequence[cirq.Qid]) -> dict:
        theta = getattr(gate, "theta", getattr(gate, "phase", None))
        return {
            "gate": "zz",
            "targets": self._targets(qubits, 2),
            "phase": self._number(theta),
        }

    def _target(self, qubits: Sequence[cirq.Qid]) -> int:
        if len(qubits) != 1:
            raise ValueError(f"Expected one target qubit but got {len(qubits)}.")
        return self._qubit_index(qubits[0])

    def _targets(self, qubits: Sequence[cirq.Qid], count: int) -> list[int]:
        if len(qubits) != count:
            raise ValueError(f"Expected {count} target qubits but got {len(qubits)}.")
        return [self._qubit_index(qubit) for qubit in qubits]

    def _qubit_index(self, qubit: cirq.Qid) -> int:
        if not isinstance(qubit, line_qubit.LineQubit):
            raise ValueError(
                "IonQ serializers require qubits to be instances of cirq.LineQubit."
            )
        if qubit.x < 0:
            raise ValueError("IonQ serializers do not support negative qubit indices.")
        return qubit.x

    def _require_zero_global_shift(self, gate: cirq.EigenGate) -> None:
        shift = gate.global_shift
        if cirq.is_parameterized(shift):
            raise ValueError("IonQ does not support parameterized global shifts.")
        if not np.isclose(float(shift), 0.0, atol=self.atol):
            raise ValueError("IonQ does not support gates with nonzero global shift.")

    def _near_mod(self, value: Any, modulus: float, expected: float) -> bool:
        if cirq.is_parameterized(value):
            return False
        try:
            remainder = float(value) % modulus
        except (TypeError, ValueError):
            return False
        return bool(np.isclose(remainder, expected, atol=self.atol))

    def _rotation(self, exponent: Any) -> float | str:
        return self._number(exponent / 2)

    def _number(self, value: Any) -> float | str:
        if cirq.is_parameterized(value):
            return str(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)