from __future__ import annotations

from sys import byteorder


def read_msr(cpu: int, register: int) -> int:
    """Read a 64-bit MSR value from /dev/cpu/{cpu}/msr."""
    filename = f"/dev/cpu/{cpu}/msr"
    with open(filename, "rb", buffering=0) as handle:
        handle.seek(register)
        raw_value = handle.read(8)
        if len(raw_value) != 8:
            raise OSError(
                f"Short read from MSR device for CPU {cpu}, register {register:#x}: "
                f"expected 8 bytes, got {len(raw_value)}"
            )
        return int.from_bytes(raw_value, byteorder)


def msr_available() -> bool:
    """Check if MSR device files are readable (requires root + msr module)."""
    try:
        with open("/dev/cpu/0/msr", "rb"):
            return True
    except (FileNotFoundError, PermissionError, OSError):
        return False