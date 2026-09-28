from __future__ import annotations

import glob
import logging
import os
import re
from collections import namedtuple
from multiprocessing import cpu_count

from s_tui.helper_functions import cat
from s_tui.sources.msr import msr_available, read_msr


INTER_RAPL_DIR = "/sys/class/powercap/intel-rapl/"
AMD_ENERGY_DIR_GLOB = "/sys/devices/platform/amd_energy.0/hwmon/hwmon*/"
MICRO_JOULE_IN_JOULE = 1000000.0

UNIT_MSR = 0xC0010299
CORE_MSR = 0xC001029A
PACKAGE_MSR = 0xC001029B
ENERGY_UNIT_MASK = 0x1F00

RaplStats = namedtuple("rapl", ["label", "current", "max"])


class RaplReader:
    def __init__(self) -> None:
        self.basenames = sorted(
            set(glob.glob("/sys/class/powercap/intel-rapl:*/"))
        )

    def read_power(self) -> list[RaplStats]:
        readings = []

        for base in self.basenames:
            try:
                label = cat(
                    os.path.join(base, "name"),
                    fallback=None,
                    binary=False,
                )
            except (OSError, ValueError) as error:
                logging.warning(
                    "ignoring %r for file %r",
                    (error, base),
                    RuntimeWarning,
                )
                continue

            if not label:
                continue

            try:
                energy = cat(os.path.join(base, "energy_uj"))
                readings.append(RaplStats(label, float(energy), 0.0))
            except (OSError, ValueError) as error:
                logging.warning(
                    "ignoring %r for file %r",
                    (error, base),
                    RuntimeWarning,
                )

        return readings

    @staticmethod
    def available() -> bool:
        return os.path.exists("/sys/class/powercap/intel-rapl")


class AMDEnergyReader:
    def __init__(self) -> None:
        labels = sorted(glob.glob(AMD_ENERGY_DIR_GLOB + "energy*_label"))
        inputs = sorted(glob.glob(AMD_ENERGY_DIR_GLOB + "energy*_input"))

        self.inputs = list(
            zip(
                (cat(filename, binary=False) for filename in labels),
                inputs,
            )
        )

        socket_number = sum(
            1 for label, _filename in self.inputs if "socket" in label
        )
        self.inputs.sort(
            key=lambda item: self.get_input_position(item[0], socket_number)
        )

    @staticmethod
    def match_label(label: str) -> re.Match[str] | None:
        return re.search(r"E(core|socket)([0-9]+)", label)

    @staticmethod
    def get_input_position(label: str, socket_number: int) -> int:
        match = AMDEnergyReader.match_label(label)
        assert match is not None, f"Unexpected label format: {label}"

        position = int(match.group(2))
        if "socket" in label:
            return position
        return position + socket_number

    def read_power(self) -> list[RaplStats]:
        readings = []
        for label, filename in self.inputs:
            readings.append(RaplStats(label, float(cat(filename)), 0.0))
        return readings

    @staticmethod
    def available() -> bool:
        return os.path.exists("/sys/devices/platform/amd_energy.0")


class AMDRaplMsrReader:
    def __init__(self) -> None:
        self.core_cpus: dict[int, int] = {}
        self.package_cpus: dict[int, int] = {}

        for cpu in range(cpu_count()):
            core_id = int(
                cat(
                    f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id",
                    binary=False,
                )
            )
            if core_id not in self.core_cpus:
                self.core_cpus[core_id] = cpu

            package_id = int(
                cat(
                    f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id",
                    binary=False,
                )
            )
            if package_id not in self.package_cpus:
                self.package_cpus[package_id] = cpu

    def read_power(self) -> list[RaplStats]:
        readings = []

        first_cpu = next(iter(self.package_cpus.values()))
        unit_value = read_msr(first_cpu, UNIT_MSR)
        energy_factor = 0.5 ** ((unit_value & ENERGY_UNIT_MASK) >> 8)

        for package_id, cpu in self.package_cpus.items():
            value = read_msr(cpu, PACKAGE_MSR)
            readings.append(
                RaplStats(
                    "Package " + str(package_id + 1),
                    value * energy_factor * MICRO_JOULE_IN_JOULE,
                    0.0,
                )
            )

        for core_id, cpu in self.core_cpus.items():
            value = read_msr(cpu, CORE_MSR)
            readings.append(
                RaplStats(
                    "Core " + str(core_id + 1),
                    value * energy_factor * MICRO_JOULE_IN_JOULE,
                    0.0,
                )
            )

        return readings

    @staticmethod
    def available() -> bool:
        try:
            cpuinfo = cat("/proc/cpuinfo", binary=False)

            vendor_match = re.search(
                r"vendor_id[\s]+: ([A-Za-z]+)",
                cpuinfo,
            )
            if not vendor_match or vendor_match is None:
                return False

            if vendor_match.group(1) != "AuthenticAMD":
                return False

            family_match = re.search(
                r"cpu family[\s]+: ([0-9]+)",
                cpuinfo,
            )
            if not family_match:
                return False

            if int(family_match[1]) != 0x17:
                return False
        except (FileNotFoundError, PermissionError):
            return False

        return msr_available()


def get_power_reader() -> RaplReader | AMDEnergyReader | AMDRaplMsrReader | None:
    for reader_type in (RaplReader, AMDEnergyReader, AMDRaplMsrReader):
        if reader_type.available():
            return reader_type()
    return None