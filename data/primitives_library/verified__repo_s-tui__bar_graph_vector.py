import contextlib
import logging
import math

from s_tui.sturwid.complex_bar_graph import LabeledBarGraphVector, ScalableBarGraph

logger = logging.getLogger(__name__)


class BarGraphVector(LabeledBarGraphVector):
    MAX_SAMPLES = 1000
    SCALE_DENSITY = 5

    @staticmethod
    def append_latest_value(values, new_val):
        values.append(new_val)
        return values[1:]

    def __init__(
        self,
        source,
        regular_colors,
        graph_count,
        visible_graph_list,
        alert_colors=None,
        bar_width=1,
    ):
        self.source = source
        self.graph_count = graph_count
        self.graph_name = source.get_source_name()
        self.measurement_unit = source.get_measurement_unit()
        self.num_samples = self.MAX_SAMPLES

        self.graph_data = [[0] * self.num_samples for _ in range(graph_count)]
        self.graph_max = max(source.get_top(), 1)
        self._needs_rebuild = True

        self.color_a = regular_colors[0]
        self.color_b = regular_colors[1]
        self.smooth_a = regular_colors[2]
        self.smooth_b = regular_colors[3]
        self.regular_colors = regular_colors
        self.alert_colors = alert_colors if alert_colors else regular_colors
        self.satt = None

        graphs = [
            ScalableBarGraph(["bg background", self.color_a, self.color_b])
            for _ in range(graph_count)
        ]
        title = self.graph_name + " [" + self.measurement_unit + "]"

        super().__init__(
            title,
            source.get_sensor_list(),
            [],
            graphs,
            visible_graph_list,
        )

        for graph in self.bar_graph_vector:
            graph.set_bar_width(bar_width)

        self.color_counter_vector = [0] * graph_count

    def _set_colors(self, graph, colors):
        self.color_a = colors[0]
        self.color_b = colors[1]
        self.smooth_a = colors[2]
        self.smooth_b = colors[3]

        if self.satt:
            self.satt = {
                (1, 0): self.smooth_a,
                (2, 0): self.smooth_b,
            }

        graph.set_segment_attributes(
            ["bg background", self.color_a, self.color_b],
            satt=self.satt,
        )

    def get_graph_name(self):
        return self.graph_name

    def get_measurement_unit(self):
        return self.measurement_unit

    def get_is_available(self):
        return self.source.get_is_available()

    def get_label_scale(self, min_val, max_val, size):
        label_count = (
            1 if size < self.SCALE_DENSITY else int(size / self.SCALE_DENSITY)
        )

        try:
            if max_val >= 100:
                return [
                    int(
                        min_val
                        + index * (max_val - min_val) / label_count
                    )
                    for index in range(label_count + 1)
                ]

            return [
                round(
                    min_val
                    + index * (max_val - min_val) / label_count,
                    1,
                )
                for index in range(label_count + 1)
            ]
        except ZeroDivisionError:
            logging.debug("Side label creation divided by 0")
            return ""

    def set_smooth_colors(self, smooth):
        self.smooth_mode = smooth
        if smooth:
            self.satt = {
                (1, 0): self.smooth_a,
                (2, 0): self.smooth_b,
            }
        else:
            self.satt = None

        for graph in self.bar_graph_vector:
            graph.set_segment_attributes(
                ["bg background", self.color_a, self.color_b],
                satt=self.satt,
            )

    def update(self):
        if not self.get_is_available():
            return

        triggered = False
        with contextlib.suppress(NotImplementedError):
            triggered = self.source.get_edge_triggered()

        current_reading = self.source.get_reading_list()
        current_thresholds = self.source.get_threshold_list()
        logging.info("Reading %s", current_reading)

        y_label_size_max = 0
        local_top_value = []
        graph_display_data = []

        for graph_idx, graph in enumerate(self.bar_graph_vector):
            has_threshold = (
                graph_idx < len(current_thresholds)
                and current_thresholds[graph_idx] is not None
            )

            if (not has_threshold and triggered) or (
                has_threshold
                and current_reading[graph_idx] > current_thresholds[graph_idx]
            ):
                self._set_colors(graph, self.alert_colors)
            else:
                self._set_colors(graph, self.regular_colors)

            try:
                self.visible_graph_list[graph_idx]
            except IndexError:
                self.visible_graph_list.append(True)

            if self.visible_graph_list[graph_idx]:
                self.graph_data[graph_idx] = self.append_latest_value(
                    self.graph_data[graph_idx],
                    current_reading[graph_idx],
                )

                num_displayed_bars = graph.get_size()[1]
                start_idx = self.MAX_SAMPLES - num_displayed_bars
                visible_graph_data = self.graph_data[graph_idx][start_idx - 1 :]

                local_top_value.append(max(visible_graph_data))
                graph_display_data.append(
                    (
                        graph_idx,
                        graph,
                        start_idx,
                        num_displayed_bars,
                    )
                )
            else:
                graph_display_data.append(None)

        update_max = False
        local_max = math.ceil(max(local_top_value))
        if local_max > self.graph_max:
            self.graph_max = local_max
            update_max = True

        for display_entry in graph_display_data:
            if display_entry is None:
                continue

            graph_idx, graph, start_idx, num_displayed_bars = display_entry
            bars = []
            swap = self.color_counter_vector[graph_idx] % 2 == 1

            for index in range(start_idx, self.MAX_SAMPLES):
                value = round(self.graph_data[graph_idx][index], 1)
                if (index & 1) ^ swap:
                    bars.append([0, value])
                else:
                    bars.append([value, 0])

            self.color_counter_vector[graph_idx] += 1
            graph.set_data(bars, float(self.graph_max))
            y_label_size_max = max(y_label_size_max, graph.get_size()[0])

        self.set_y_label(
            self.get_label_scale(
                0,
                self.graph_max,
                float(y_label_size_max),
            )
        )

        need_rebuild = update_max or self._needs_rebuild
        source_available = self.source.sensor_available

        if source_available:
            new_available = source_available[: len(self.sensor_available)]
            if new_available != self.sensor_available[: len(new_available)]:
                self.sensor_available[: len(new_available)] = new_available
                need_rebuild = True

        if need_rebuild:
            self.set_visible_graphs(self.visible_graph_list)
            self._needs_rebuild = False