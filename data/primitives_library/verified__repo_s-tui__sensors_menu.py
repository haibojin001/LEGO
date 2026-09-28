import urwid

from s_tui.sturwid.ui_elements import ViListBox


class SensorsMenu:
    MAX_TITLE_LEN = 120

    def on_mode_button(self, button, state):
        return None

    def __init__(self, return_fn, source_list, default_source_conf):
        self.return_fn = return_fn
        self.sensor_status_dict = {}
        self.sensor_button_dict = {}
        self.active_sensors = {}

        apply = urwid.Button("Apply", on_press=self.on_apply)
        apply._label.align = "center"
        cancel = urwid.Button("Cancel", on_press=self.on_cancel)
        cancel._label.align = "center"
        button_row = urwid.Columns([apply, cancel])

        sensor_piles = []
        control_piles = []

        for source in source_list:
            name = source.get_source_name()
            sensors = source.get_sensor_list()
            configuration = default_source_conf.get(name) or {}

            states = [configuration.get(sensor.lower(), True) for sensor in sensors]
            self.sensor_status_dict[name] = states
            self.active_sensors[name] = list(states)

            heading = urwid.Text(("bold text", name), "center")
            boxes = [
                urwid.CheckBox(sensor_name, enabled)
                for sensor_name, enabled in zip(sensors, states)
            ]
            self.sensor_button_dict[name] = boxes

            controls = []
            if boxes:
                controls.extend(
                    [
                        urwid.Button(
                            "Check all",
                            on_press=self.on_checkall_col,
                            user_data=name,
                        ),
                        urwid.Button(
                            "Uncheck all",
                            on_press=self.on_uncheckall_col,
                            user_data=name,
                        ),
                    ]
                )

            sensor_piles.append(
                urwid.Pile(urwid.SimpleFocusListWalker([heading] + boxes))
            )
            control_piles.append(
                urwid.Pile(urwid.SimpleFocusListWalker(controls))
            )

        sensor_columns = urwid.Columns(sensor_piles)
        selector_columns = urwid.Columns(control_piles, box_columns=[0, 1])
        body = urwid.SimpleFocusListWalker(
            [sensor_columns, selector_columns, button_row]
        )
        self.main_window = urwid.LineBox(ViListBox(body))

        height = 6
        for source_states in self.active_sensors.values():
            height = max(height, len(source_states) + 6)
        self.size = (height, self.MAX_TITLE_LEN)

    def get_size(self):
        return self.size

    def set_checkbox_value(self):
        for source_name, boxes in self.sensor_button_dict.items():
            for checkbox, value in zip(boxes, self.active_sensors[source_name]):
                checkbox.set_state(value)

    def on_cancel(self, w):
        self.set_checkbox_value()
        self.return_fn(update=False)

    def on_apply(self, w):
        changed = False

        for source_name, boxes in self.sensor_button_dict.items():
            values = [checkbox.get_state() for checkbox in boxes]
            if values != self.active_sensors[source_name]:
                changed = True
            self.active_sensors[source_name] = values

        self.set_checkbox_value()
        self.return_fn(update=changed)

    def setall_cb_col(self, w, col, state):
        for source_name, boxes in self.sensor_button_dict.items():
            if source_name == col:
                for checkbox in boxes:
                    checkbox.set_state(state)

    def on_uncheckall_col(self, w, col):
        self.setall_cb_col(self, col, False)

    def on_checkall_col(self, w, col):
        self.setall_cb_col(self, col, True)