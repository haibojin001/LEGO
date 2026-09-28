from __future__ import annotations

import pathlib

import cirq_ionq
from cirq.testing.json import ModuleJsonTestSpec
from cirq_ionq.json_resolver_cache import _class_resolver_dictionary

TestSpec = ModuleJsonTestSpec(
    name="cirq_ionq",
    packages=[cirq_ionq],
    test_data_path=pathlib.Path(__file__).parent,
    not_yet_serializable=[
        "SerializedProgram",
        "Calibration",
        "QPUResult",
        "IonQException",
        "IonQUnsuccessfulJobException",
        "IonQNotFoundException",
        "IonQAPIDevice",
        "IonQSerializerMixedGatesetsException",
        "Job",
        "SimulatorResult",
    ],
    should_not_be_serialized=["Sampler", "Service", "Serializer"],
    resolver_cache=_class_resolver_dictionary(),
    deprecated={},
)