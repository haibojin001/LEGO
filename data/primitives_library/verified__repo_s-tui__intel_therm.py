from __future__ import annotations

from typing import NamedTuple

from s_tui.sources.msr import msr_available, read_msr

IA32_THERM_STATUS = 0x19C

THERMAL_STATUS = 1 << 0
PROCHOT_STATUS = 1 << 2
CRITICAL_STATUS = 1 << 4
POWER_LIMIT_STATUS = 1 << 10
CURRENT_LIMIT_STATUS = 1 << 12
CROSS_DOMAIN_STATUS = 1 << 14

_REASON_BITS = (
    (THERMAL_STATUS, "T"),
    (PROCHOT_STATUS, "H"),
    (CRITICAL_STATUS, "C"),
    (POWER_LIMIT_STATUS, "W"),
    (CURRENT_LIMIT_STATUS, "A"),
    (CROSS_DOMAIN_STATUS, "X"),
)


class ThrottleStatus(NamedTuple):
    thermal: bool
    prochot: bool
    critical: bool
    power_limit: bool
    current_limit: bool
    cross_domain: bool

    @property
    def any_active(self) -> bool:
        return any(self)

    @property
    def label(self) -> str:
        return "/".join(
            label
            for (_, label), active in zip(_REASON_BITS, self)
            if active
        )


def read_therm_status(cpu: int) -> ThrottleStatus:
    value = read_msr(cpu, IA32_THERM_STATUS)
    return ThrottleStatus(
        thermal=bool(value & THERMAL_STATUS),
        prochot=bool(value & PROCHOT_STATUS),
        critical=bool(value & CRITICAL_STATUS),
        power_limit=bool(value & POWER_LIMIT_STATUS),
        current_limit=bool(value & CURRENT_LIMIT_STATUS),
        cross_domain=bool(value & CROSS_DOMAIN_STATUS),
    )


def available() -> bool:
    if not msr_available():
        return False
    try:
        read_msr(0, IA32_THERM_STATUS)
    except (OSError, ValueError):
        return False
    return True