from __future__ import annotations

import re
from typing import NamedTuple

from s_tui.helper_functions import cat
from s_tui.sources.msr import msr_available, read_msr

PSTATE_CUR_LIMIT = 0xC0010061
CUR_PSTATE_LIMIT_MASK = 0x7
PSTATE_CAP_LABEL = "Pc"
MIN_CPU_FAMILY = 0x17


def _family_supported() -> bool:
    cpuinfo = cat("/proc/cpuinfo", fallback="", binary=False)
    if "AuthenticAMD" not in cpuinfo:
        return False
    match = re.search(r"cpu family\s*:\s*(\d+)", cpuinfo)
    return match is not None and int(match.group(1)) >= MIN_CPU_FAMILY


class ThrottleStatus(NamedTuple):
    pstate_capped: bool

    @property
    def label(self) -> str:
        return PSTATE_CAP_LABEL if self.pstate_capped else ""


def read_therm_status(cpu: int) -> ThrottleStatus:
    """Read PStateCurLim for a CPU and decode the P-state cap."""
    value = read_msr(cpu, PSTATE_CUR_LIMIT)
    return ThrottleStatus(pstate_capped=bool(value & CUR_PSTATE_LIMIT_MASK))


def available() -> bool:
    """Check whether AMD MSR P-state cap detection is usable."""
    if not _family_supported():
        return False
    if not msr_available():
        return False
    try:
        read_msr(0, PSTATE_CUR_LIMIT)
    except (OSError, ValueError):
        return False
    return True