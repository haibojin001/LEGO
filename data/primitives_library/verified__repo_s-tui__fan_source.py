from __future__ import annotations

import logging

import psutil

from s_tui.sources.source import Source


class FanSource(Source):
    """Provides measurements reported by system fan sensors."""

    def __init__(self):
        try:
            if psutil.sensors_fans():
                self.is_available = True
        except AttributeError:
            self.is_available = False
            logging.debug("Fan sensor support is unavailable in psutil")
            return
        except (OSError, TypeError):
            self.is_available = False
            logging.debug("Unable to query fan sensors")
            return

        Source.__init__(self)

        self.name = "Fan"
        self.measurement_unit = "RPM"
        self.pallet = (
            "fan light",
            "fan dark",
            "fan light smooth",
            "fan dark smooth",
        )

        try:
            fan_groups = psutil.sensors_fans()
        except (OSError, TypeError):
            logging.debug("Unable to build fan sensor list")
            self.is_available = False
            return

        if not fan_groups:
            self.is_available = False
            return

        self._sensor_lookup = {}

        for group_name, sensors in fan_groups.items():
            for position, sensor in enumerate(sensors):
                label = sensor.label
                display_name = label if label else group_name + "," + str(position)

                logging.debug("Fan sensor name %s", display_name)

                self.available_sensors.append(display_name)
                self._sensor_lookup[(group_name, position)] = (
                    len(self.available_sensors) - 1
                )

        self.sensor_available = [True] * len(self.available_sensors)
        self.last_measurement = [0] * len(self.available_sensors)

    def update(self) -> None:
        try:
            fan_groups = psutil.sensors_fans()
        except (TypeError, OSError):
            logging.debug("Fan sensor query failed; retaining previous values")
            return

        if fan_groups is None:
            logging.debug("Fan sensor query returned no data; retaining previous values")
            return

        seen = set()

        for group_name, sensors in fan_groups.items():
            for position, sensor in enumerate(sensors):
                sensor_index = self._sensor_lookup.get((group_name, position))
                if sensor_index is None:
                    continue

                if sensor.current > 10000:
                    self.sensor_available[sensor_index] = False
                    continue

                self.last_measurement[sensor_index] = int(sensor.current)
                self.sensor_available[sensor_index] = True
                seen.add(sensor_index)

        for sensor_index in range(len(self.available_sensors)):
            if sensor_index not in seen:
                self.sensor_available[sensor_index] = False

    def get_edge_triggered(self) -> bool:
        return False

    def get_top(self) -> int:
        return 1