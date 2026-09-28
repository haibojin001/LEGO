from __future__ import annotations

import logging
import warnings
from collections import OrderedDict

import psutil

from s_tui.sources.source import Source


class TempSource(Source):
    """Temperature measurement provider backed by psutil sensors."""

    THRESHOLD_TEMP = 80

    def __init__(self, temp_thresh: int | str | None = None) -> None:
        warnings.filterwarnings("ignore", ".*FileNotFound.*")

        try:
            initial_data = psutil.sensors_temperatures()
            if not initial_data:
                self.is_available = False
                logging.debug("sensors_temperatures() returned empty/None")
                return
            self.is_available = True
        except (AttributeError, OSError, TypeError):
            self.is_available = False
            logging.debug("cpu temperature is not available from psutil")
            return

        Source.__init__(self)

        self.name = "Temp"
        self.measurement_unit = "C"
        self.max_last_temp = 0
        self.temp_thresh_is_set = False
        self.pallet = (
            "temp light",
            "temp dark",
            "temp light smooth",
            "temp dark smooth",
        )
        self.alert_pallet = (
            "high temp light",
            "high temp dark",
            "high temp light smooth",
            "high temp dark smooth",
        )
        self.max_temp = 10

        try:
            ordered_sensors = OrderedDict(
                sorted(psutil.sensors_temperatures().items())
            )
        except (OSError, TypeError):
            logging.debug("Unable to create sensors dict")
            self.is_available = False
            return

        self._sensor_lookup = {}
        duplicate_labels = []

        for group_name, entries in ordered_sensors.items():
            generated_name = "".join(group_name.title().split(" "))

            for position, entry in enumerate(entries):
                if entry.current <= 1.0 or entry.current >= 127.0:
                    continue

                if entry.label:
                    label_name = "".join(entry.label.title().split(" "))
                    duplicate_count = duplicate_labels.count(label_name)
                    duplicate_labels.append(label_name)
                    display_name = label_name + "," + str(duplicate_count)
                else:
                    display_name = generated_name + "," + str(position)

                logging.debug("Temp sensor name %s", display_name)
                self.available_sensors.append(display_name)
                self._sensor_lookup[(group_name, position)] = (
                    len(self.available_sensors) - 1
                )

        self.sensor_available = [True] * len(self.available_sensors)
        self.last_measurement = [0.0] * len(self.available_sensors)

        self.temp_thresh = self.THRESHOLD_TEMP
        if temp_thresh is not None:
            self.temp_thresh_is_set = True
            if int(temp_thresh) > 0:
                self.temp_thresh = int(temp_thresh)
                logging.debug("Updated custom threshold to %s", self.temp_thresh)

        self.last_thresholds = [self.temp_thresh] * len(self.available_sensors)

    def update(self) -> None:
        try:
            current_sensors = psutil.sensors_temperatures()
        except (OSError, TypeError) as error:
            logging.debug(
                "sensors_temperatures() raised %s, keeping stale data", error
            )
            return

        if current_sensors is None:
            return

        try:
            ordered_sensors = OrderedDict(sorted(current_sensors.items()))
        except OSError:
            return

        seen = set()

        for group_name, entries in ordered_sensors.items():
            for position, entry in enumerate(entries):
                sensor_number = self._sensor_lookup.get((group_name, position))
                if sensor_number is None:
                    continue

                if entry.current <= 1.0 or entry.current >= 127.0:
                    self.sensor_available[sensor_number] = False
                    continue

                self.last_measurement[sensor_number] = entry.current

                if (
                    entry.high is not None
                    and entry.high
                    and entry.high < 127.0
                    and not self.temp_thresh_is_set
                ):
                    self.last_thresholds[sensor_number] = entry.high
                else:
                    self.last_thresholds[sensor_number] = self.temp_thresh

                self.sensor_available[sensor_number] = True
                seen.add(sensor_number)

        for sensor_number in range(len(self.available_sensors)):
            if sensor_number not in seen:
                self.sensor_available[sensor_number] = False

        readings = [
            self.last_measurement[sensor_number]
            for sensor_number in range(len(self.available_sensors))
            if self.sensor_available[sensor_number]
        ]

        if readings:
            self.max_last_temp = max(readings)
            Source.update(self)

    def get_edge_triggered(self) -> bool:
        return self.max_last_temp > self.temp_thresh

    def get_sensor_alerts(self) -> list[str | None]:
        """Return alert styles for each currently known sensor."""
        globally_triggered = self.max_last_temp > self.temp_thresh
        result: list[str | None] = [None] * len(self.available_sensors)

        for sensor_number in range(len(self.available_sensors)):
            if not self.sensor_available[sensor_number]:
                continue

            threshold = self.last_thresholds[sensor_number]
            if (threshold is None and globally_triggered) or (
                threshold is not None
                and self.last_measurement[sensor_number] > threshold
            ):
                result[sensor_number] = "high temp txt"

        return result

    def get_max_triggered(self) -> bool:
        """Return whether the recorded maximum exceeds the threshold."""
        return self.max_temp > self.temp_thresh

    def reset(self) -> None:
        self.max_temp = 10

    def get_maximum(self) -> float:
        raise NotImplementedError("Get maximum is not implemented")

    def get_top(self) -> int:
        if hasattr(self, "_cached_top_temp"):
            return self._cached_top_temp

        ceiling = 10

        try:
            collections = psutil.sensors_temperatures().values()
        except (OSError, TypeError):
            return 10

        for collection in collections:
            for entry in collection:
                if (
                    entry.high is not None
                    and entry.high > ceiling
                    and entry.critical is not None
                ):
                    ceiling = entry.critical

        self._cached_top_temp = int(min(ceiling, 99))
        return self._cached_top_temp