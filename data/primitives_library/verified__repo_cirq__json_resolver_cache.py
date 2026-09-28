from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import cirq_ionq

if TYPE_CHECKING:
    from cirq.protocols.json_serialization import ObjectFactory


@functools.lru_cache()
def _class_resolver_dictionary() -> dict[str, ObjectFactory]:
    return {
        "GPIGate": cirq_ionq.GPIGate,
        "GPI2Gate": cirq_ionq.GPI2Gate,
        "MSGate": cirq_ionq.MSGate,
        "ZZGate": cirq_ionq.ZZGate,
        "IonQTargetGateset": cirq_ionq.IonQTargetGateset,
        "AriaNativeGateset": cirq_ionq.AriaNativeGateset,
        "ForteNativeGateset": cirq_ionq.ForteNativeGateset,
    }