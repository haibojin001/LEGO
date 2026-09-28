from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

import cirq
from cirq import protocols
from cirq._doc import document


@cirq.value.value_equality
class GPIGate(cirq.Gate):
    """A one-qubit IonQ GPI pi-pulse gate."""

    def __init__(self, *, phi):
        self.phi = phi

    def _unitary_(self) -> np.ndarray:
        upper_right = cmath.exp(-2j * math.pi * self.phi)
        lower_left = cmath.exp(2j * math.pi * self.phi)
        return np.array([[0, upper_right], [lower_left, 0]])

    def __str__(self) -> str:
        return 'GPI'

    def _num_qubits_(self) -> int:
        return 1

    @property
    def phase(self) -> float:
        return self.phi

    def __repr__(self) -> str:
        return f'cirq_ionq.GPIGate(phi={self.phi!r})'

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['phi'])

    def _value_equality_values_(self) -> Any:
        return self.phi

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> str | protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=(f'GPI({self.phase!r})',))

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return self
        return NotImplemented


GPI = GPIGate(phi=0)
document(GPI, 'An instance of the single-qubit GPI gate with zero phase.')


@cirq.value.value_equality
class GPI2Gate(cirq.Gate):
    """A one-qubit IonQ GPI2 pi/2-pulse gate."""

    def __init__(self, *, phi):
        self.phi = phi

    def _unitary_(self) -> np.ndarray:
        upper_right = -1j * cmath.exp(-2j * math.pi * self.phase)
        lower_left = -1j * cmath.exp(2j * math.pi * self.phase)
        return np.array([[1, upper_right], [lower_left, 1]]) / math.sqrt(2)

    @property
    def phase(self) -> float:
        return self.phi

    def __str__(self) -> str:
        return 'GPI2'

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> str | protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=(f'GPI2({self.phase!r})',))

    def _num_qubits_(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f'cirq_ionq.GPI2Gate(phi={self.phi!r})'

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['phi'])

    def _value_equality_values_(self) -> Any:
        return self.phi

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return GPI2Gate(phi=self.phi + 0.5)
        return NotImplemented


GPI2 = GPI2Gate(phi=0)
document(GPI2, 'An instance of the single-qubit GPI2 gate with zero phase.')


@cirq.value.value_equality
class MSGate(cirq.Gate):
    """A two-qubit IonQ Mølmer-Sørensen gate."""

    def __init__(self, *, phi0, phi1, theta=0.25):
        self.phi0 = phi0
        self.phi1 = phi1
        self.theta = theta

    def _unitary_(self) -> np.ndarray:
        diagonal = np.cos(math.pi * self.theta)
        off_diagonal = np.sin(math.pi * self.theta)
        phase_sum = self.phi0 + self.phi1
        phase_difference = self.phi0 - self.phi1

        return np.array(
            [
                [
                    diagonal,
                    0,
                    0,
                    off_diagonal * -1j * cmath.exp(-2j * math.pi * phase_sum),
                ],
                [
                    0,
                    diagonal,
                    off_diagonal * -1j * cmath.exp(-2j * math.pi * phase_difference),
                    0,
                ],
                [
                    0,
                    off_diagonal * -1j * cmath.exp(2j * math.pi * phase_difference),
                    diagonal,
                    0,
                ],
                [
                    off_diagonal * -1j * cmath.exp(2j * math.pi * phase_sum),
                    0,
                    0,
                    diagonal,
                ],
            ]
        )

    @property
    def phases(self) -> Sequence[float]:
        return [self.phi0, self.phi1]

    def __str__(self) -> str:
        return 'MS'

    def _num_qubits_(self) -> int:
        return 2

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> str | protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=(f'MS({self.phi0!r})', f'MS({self.phi1!r})')
        )

    def __repr__(self) -> str:
        return f'cirq_ionq.MSGate(phi0={self.phi0!r}, phi1={self.phi1!r})'

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['phi0', 'phi1', 'theta'])

    def _value_equality_values_(self) -> Any:
        return (self.phi0, self.phi1)

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return MSGate(phi0=self.phi0 + 0.5, phi1=self.phi1, theta=self.theta)
        return NotImplemented


MS = MSGate(phi0=0, phi1=0)
document(MS, 'An instance of the two-qubit Mølmer-Sørensen gate with zero phases.')


@cirq.value.value_equality
class ZZGate(cirq.Gate):
    """A two-qubit IonQ ZZ gate."""

    def __init__(self, *, theta):
        self.theta = theta

    def _unitary_(self) -> np.ndarray:
        return np.array(
            [
                [cmath.exp(-1j * self.theta * math.pi), 0, 0, 0],
                [0, cmath.exp(1j * self.theta * math.pi), 0, 0],
                [0, 0, cmath.exp(1j * self.theta * math.pi), 0],
                [0, 0, 0, cmath.exp(-1j * self.theta * math.pi)],
            ]
        )

    @property
    def phase(self) -> float:
        return self.theta

    def __str__(self) -> str:
        return 'ZZ'

    def _num_qubits_(self) -> int:
        return 2

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> str | protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=(f'ZZ({self.theta!r})', 'ZZ'))

    def __repr__(self) -> str:
        return f'cirq_ionq.ZZGate(theta={self.theta!r})'

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['theta'])

    def _value_equality_values_(self) -> Any:
        return self.theta

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return ZZGate(theta=-self.theta)
        return NotImplemented


ZZ = ZZGate(theta=0.25)
document(ZZ, 'An instance of the two-qubit ZZ gate with theta equal to 0.25.')