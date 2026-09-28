from collections import OrderedDict

import urwid


class SummaryTextList:
    MAX_LABEL_L = 12

    def __init__(self, source, visible_sensors_list):
        self.source = source
        self.visible_summaries = OrderedDict()

        summary_keys = list(self.source.get_summary().keys())
        self.visible_summaries[summary_keys[0]] = any(visible_sensors_list)

        for summary_key, is_visible in zip(
            summary_keys[1:], visible_sensors_list
        ):
            self.visible_summaries[summary_key] = is_visible

        self.summary_text_items = OrderedDict()

    @staticmethod
    def _format_display_val(val, alerts, suffixes, sensor_idx):
        rendered = str(val)

        if 0 <= sensor_idx < len(suffixes):
            suffix = suffixes[sensor_idx]
            if suffix:
                rendered += " " + suffix

        if 0 <= sensor_idx < len(alerts):
            alert = alerts[sensor_idx]
            if alert:
                return alert, rendered

        return rendered

    def get_text_item_list(self):
        result = []
        alerts = self.source.get_sensor_alerts()
        suffixes = self.source.get_sensor_suffixes()

        for index, (key, value) in enumerate(self.source.get_summary().items()):
            label = urwid.Text(str(key[: self.MAX_LABEL_L]))
            display_value = self._format_display_val(
                value, alerts, suffixes, index - 1
            )
            value_widget = urwid.Text(display_value, align="right")

            self.summary_text_items[key] = value_widget

            row = urwid.Columns(
                [
                    ("weight", 1.5, label),
                    value_widget,
                ]
            )

            if self.visible_summaries.setdefault(key, True):
                result.append(row)

        return result

    def update_visibility(self, visible_sensors):
        keys = list(self.visible_summaries.keys())
        self.visible_summaries[keys[0]] = any(visible_sensors)

        for key, is_visible in zip(keys[1:], visible_sensors):
            self.visible_summaries[key] = is_visible

    def update(self):
        alerts = self.source.get_sensor_alerts()
        suffixes = self.source.get_sensor_suffixes()

        for index, (key, value) in enumerate(self.source.get_summary().items()):
            if key in self.summary_text_items:
                self.summary_text_items[key].set_text(
                    self._format_display_val(value, alerts, suffixes, index - 1)
                )

    def get_is_available(self):
        return self.source.get_is_available()