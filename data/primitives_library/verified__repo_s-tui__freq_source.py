from __future__ import annotations

import logging
import os
from types import ModuleType

import psutil

from s_tui.helper_functions import cat
from s_tui.sources import amd_pstate_limit, intel_therm
from s_tui.sources.source import Source

SYSFS_THERMAL_THROTTLE = "/sys/devices/system/cpu/cpu{}/thermal_throttle"


def _read_throttle_count(core_id: int, counter: str) -> int | None:
    """Read a thermal_throttle counter from sysfs. Returns None if unavailable."""
    path = os.path.join(SYSFS_THERMAL_THROTTLE.format(core_id), counter)
    raw = cat(path, fallback=None, binary=False)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


class FreqSource(Source):
    """Source class implementing CPU frequency information polling."""

    def __init__(self):
        self.is_available = True
        if not hasattr(psutil, "cpu_freq"):
            self.is_available = False
            logging.debug("cpu_freq is not available from psutil")
            return

        Source.__init__(self)

        self.name = "Frequency"
        self.measurement_unit = "MHz"
        self.pallet = (
            "freq light",
            "freq dark",
            "freq light smooth",
            "freq dark smooth",
        )
        self.alert_pallet = (
            "freq throttle light",
            "freq throttle dark",
            "freq throttle light smooth",
            "freq throttle dark smooth",
        )

        try:
            per_cpu_freq = psutil.cpu_freq(True)
        except (OSError, NotImplementedError, TypeError):
            per_cpu_freq = None

        try:
            overall_freq = psutil.cpu_freq(False)
        except (OSError, NotImplementedError, TypeError):
            overall_freq = None

        total_cores = self._get_total_core_count()
        if per_cpu_freq and len(per_cpu_freq) > total_cores:
            total_cores = len(per_cpu_freq)

        self.top_freq = overall_freq.max if overall_freq else 0.0
        self.max_freq = self.top_freq

        self.available_sensors = ["Avg"]
        for core_id in range(total_cores):
            self.available_sensors.append("Core " + str(core_id))

        self.last_measurement = [0.0] * len(self.available_sensors)
        self.sensor_available = [True] * len(self.available_sensors)
        self.last_thresholds: list[float | None] = [None] * len(
            self.available_sensors
        )

        self._mark_offline_cores(total_cores, self._get_online_cpu_ids())

        if self.top_freq == 0.0 and max(self.last_measurement) >= 0:
            self.max_freq = max(self.last_measurement)

        self._num_cores = total_cores
        self._throttle_labels: list[str] = [""] * total_cores
        self._cached_suffixes: list[str] = [""] * len(self.available_sensors)
        self._cached_alerts: list[str | None] = [None] * len(
            self.available_sensors
        )

        self._msr_therm: ModuleType | None = None
        if intel_therm.available():
            self._msr_therm = intel_therm
        elif amd_pstate_limit.available():
            self._msr_therm = amd_pstate_limit

        self._prev_core_throttle: list[int | None] = [None] * total_cores
        self._prev_pkg_throttle: int | None = None
        self._throttle_available = self._msr_therm is not None or self._init_sysfs(
            total_cores
        )

    @staticmethod
    def _parse_cpu_list(value: str | None) -> set[int]:
        if not value:
            return set()

        cpus: set[int] = set()
        try:
            for item in value.strip().split(","):
                if not item:
                    continue
                if "-" in item:
                    start, end = item.split("-", 1)
                    cpus.update(range(int(start), int(end) + 1))
                else:
                    cpus.add(int(item))
        except ValueError:
            return set()
        return cpus

    def _get_total_core_count(self) -> int:
        present = self._parse_cpu_list(
            cat("/sys/devices/system/cpu/present", fallback=None, binary=False)
        )
        if present:
            return max(present) + 1

        count = os.cpu_count()
        if count is None:
            count = psutil.cpu_count(logical=True)
        return count or 0

    def _get_online_cpu_ids(self) -> set[int]:
        online = self._parse_cpu_list(
            cat("/sys/devices/system/cpu/online", fallback=None, binary=False)
        )
        if online:
            return online

        count = os.cpu_count()
        if count is None:
            count = psutil.cpu_count(logical=True)
        return set(range(count or 0))

    def _mark_offline_cores(self, total_cores: int, online_cpu_ids: set[int]) -> None:
        for core_id in range(total_cores):
            if core_id not in online_cpu_ids:
                self.sensor_available[core_id + 1] = False

    def _init_sysfs(self, total_cores: int) -> bool:
        """Initialize sysfs throttle counter baselines."""
        any_available = False
        for core_id in range(total_cores):
            count = _read_throttle_count(core_id, "core_throttle_count")
            if count is not None:
                self._prev_core_throttle[core_id] = count
                any_available = True

        pkg_count = _read_throttle_count(0, "package_throttle_count")
        if pkg_count is not None:
            self._prev_pkg_throttle = pkg_count
            any_available = True

        return any_available

    def _update_throttle_state(self) -> None:
        """Update per-core throttle labels, thresholds, and cached outputs."""
        if not self._throttle_available:
            return

        if self._msr_therm is not None:
            self._update_throttle_msr(self._msr_therm)
        else:
            self._update_throttle_sysfs()

        any_throttled = any(self._throttle_labels)
        self.last_thresholds[0] = 0.0 if any_throttled else None
        for core_id in range(self._num_cores):
            self.last_thresholds[core_id + 1] = (
                0.0 if self._throttle_labels[core_id] else None
            )

        suffixes = [""] * len(self.available_sensors)
        alerts: list[str | None] = [None] * len(self.available_sensors)
        for core_id in range(self._num_cores):
            index = core_id + 1
            if index < len(suffixes) and self.sensor_available[index]:
                label = self._throttle_labels[core_id]
                if label:
                    suffixes[index] = label
                    alerts[index] = "throttle txt"

        if any_throttled:
            suffixes[0] = next(label for label in self._throttle_labels if label)
            alerts[0] = "throttle txt"

        self._cached_suffixes = suffixes
        self._cached_alerts = alerts

    def _update_throttle_msr(self, therm: ModuleType) -> None:
        """Read the vendor throttle MSR per core."""
        for core_id in range(self._num_cores):
            try:
                status = therm.read_therm_status(core_id)
                self._throttle_labels[core_id] = status.label
            except OSError:
                self._throttle_labels[core_id] = ""

    def _update_throttle_sysfs(self) -> None:
        """Detect throttling via sysfs counter deltas."""
        for core_id in range(self._num_cores):
            count = _read_throttle_count(core_id, "core_throttle_count")
            previous = self._prev_core_throttle[core_id]
            throttled = count is not None and previous is not None and count > previous
            if count is not None:
                self._prev_core_throttle[core_id] = count
            self._throttle_labels[core_id] = "Tc" if throttled else ""

        pkg_count = _read_throttle_count(0, "package_throttle_count")
        previous_pkg = self._prev_pkg_throttle
        package_throttled = (
            pkg_count is not None
            and previous_pkg is not None
            and pkg_count > previous_pkg
        )
        if pkg_count is not None:
            self._prev_pkg_throttle = pkg_count

        if package_throttled:
            for core_id in range(self._num_cores):
                if not self._throttle_labels[core_id]:
                    self._throttle_labels[core_id] = "Tp"

    def update(self) -> None:
        try:
            per_cpu_freq = psutil.cpu_freq(True)
        except (OSError, AttributeError, NotImplementedError, TypeError) as error:
            logging.debug("cpu_freq() raised %s: %s", type(error).__name__, error)
            for index in range(1, len(self.sensor_available)):
                self.sensor_available[index] = False
            return

        if not per_cpu_freq:
            return

        num_cores = len(self.available_sensors) - 1
        online_freqs = []

        for core_id in range(num_cores):
            if core_id < len(per_cpu_freq) and per_cpu_freq[core_id].current > 0:
                current = per_cpu_freq[core_id].current
                self.last_measurement[core_id + 1] = current
                online_freqs.append(current)
                self.sensor_available[core_id + 1] = True
            else:
                self.sensor_available[core_id + 1] = False

        if online_freqs:
            self.last_measurement[0] = sum(online_freqs) / len(online_freqs)
            self.sensor_available[0] = True
            if self.max_freq == 0.0:
                self.max_freq = max(online_freqs)
        else:
            self.sensor_available[0] = False

        self._update_throttle_state()