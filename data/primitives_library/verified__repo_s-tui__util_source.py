from __future__ import annotations

import logging

import psutil

from s_tui.sources.source import Source


class UtilSource(Source):
    def __init__(self):
        if not hasattr(psutil, "cpu_percent") and psutil.cpu_percent():
            self.is_available = False
            logging.debug("cpu utilization is not available from psutil")
            return

        super().__init__()

        self.name = "Util"
        self.measurement_unit = "%"
        self.pallet = (
            "util light",
            "util dark",
            "util light smooth",
            "util dark smooth",
        )

        total_cores = self._get_total_core_count()
        self.available_sensors = ["Avg"]
        for core_id in range(total_cores):
            self.available_sensors.append("Core " + str(core_id))

        self.last_measurement = [0.0] * len(self.available_sensors)
        self.sensor_available = [True] * len(self.available_sensors)

        self._cached_online_ids = self._get_online_cpu_ids()
        self._cached_online_len = -1

        self._mark_offline_cores(total_cores, self._cached_online_ids)

    def _get_online_cpu_ids(self):
        try:
            return psutil.Process().cpu_affinity()
        except (AttributeError, psutil.Error):
            return None

    def _mark_offline_cores(self, total_cores, online_ids):
        if online_ids is None:
            return

        online = set(online_ids)
        for core_id in range(total_cores):
            if core_id not in online:
                self.sensor_available[core_id + 1] = False

    def update(self) -> None:
        try:
            per_cpu = psutil.cpu_percent(interval=0.0, percpu=True)
        except OSError:
            return

        if not per_cpu:
            return

        if len(per_cpu) != self._cached_online_len:
            self._cached_online_ids = self._get_online_cpu_ids()
            self._cached_online_len = len(per_cpu)

        online_ids = self._cached_online_ids

        if online_ids is None:
            average = sum(per_cpu) / len(per_cpu)
            self.last_measurement = [average] + [float(value) for value in per_cpu]
            return

        values_by_core = dict(zip(online_ids, per_cpu))
        total_cores = len(self.available_sensors) - 1
        active_values = []

        for core_id in range(total_cores):
            if core_id in values_by_core:
                value = float(values_by_core[core_id])
                self.last_measurement[core_id + 1] = value
                active_values.append(value)
                self.sensor_available[core_id + 1] = True
            else:
                self.sensor_available[core_id + 1] = False

        self.last_measurement[0] = (
            sum(active_values) / len(active_values) if active_values else 0.0
        )
        logging.info("Utilization recorded %s", self.last_measurement)

    def get_is_available(self) -> bool:
        return self.is_available

    def get_top(self) -> int:
        return 100