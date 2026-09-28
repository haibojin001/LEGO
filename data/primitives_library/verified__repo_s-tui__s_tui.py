import argparse
import atexit
import configparser
import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict

import psutil
import urwid

from s_tui.about_menu import AboutMenu
from s_tui.builtin_stress_menu import BuiltinStressMenu
from s_tui.builtin_stresser import BuiltinStresser
from s_tui.help_menu import HELP_MESSAGE, HelpMenu
from s_tui.helper_functions import (
    __version__,
    cat,
    get_processor_name,
    get_user_config_dir,
    get_user_config_file,
    kill_child_processes,
    make_user_config_dir,
    output_to_csv,
    output_to_json,
    output_to_terminal,
    seconds_to_text,
    str_to_bool,
    user_config_dir_exists,
    user_config_file_exists,
    which,
)
from s_tui.power_profile_menu import (
    SYSFS_AVAIL_EPP,
    SYSFS_AVAIL_GOVERNORS,
    SYSFS_EPP,
    SYSFS_GOVERNOR,
    PowerProfileMenu,
    read_available,
)
from s_tui.sensors_menu import SensorsMenu
from s_tui.sources.fan_source import FanSource
from s_tui.sources.freq_source import FreqSource
from s_tui.sources.rapl_power_source import RaplPowerSource
from s_tui.sources.script_hook_loader import ScriptHookLoader
from s_tui.sources.temp_source import TempSource
from s_tui.sources.util_source import UtilSource
from s_tui.stress_menu import StressMenu
from s_tui.sturwid.bar_graph_vector import BarGraphVector
from s_tui.sturwid.summary_text_list import SummaryTextList
from s_tui.sturwid.ui_elements import DEFAULT_PALETTE, ViListBox, button, radio_button

UPDATE_INTERVAL = 1
HOOK_INTERVAL = 30 * 1000
DEGREE_SIGN = "\N{DEGREE SIGN}"
ZERO_TIME = seconds_to_text(0)
DEFAULT_LOG_FILE = "_s-tui.log"
DEFAULT_CSV_FILE = "s-tui_log_" + time.strftime("%Y-%m-%d_%H_%M_%S") + ".csv"
VERSION_MESSAGE = (
    "s-tui "
    + __version__
    + " - (C) 2017-2025 Alex Manuskin, Gil Tsuker\n    Released under GNU GPLv2"
)
ERROR_MESSAGE = (
    "\n        Oops! s-tui has encountered a fatal error\n"
    "        Please report this bug here: https://github.com/amanusk/s-tui"
)

graph_controller = None


class MainLoop(urwid.MainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: object) -> None:
        if graph_controller is not None:
            graph_controller.stress_controller.kill_stress_process()
        raise urwid.ExitMainLoop()

    def unhandled_input(self, data):
        logging.debug("Caught %s", data)
        if graph_controller is None:
            return
        if data == "q":
            graph_controller.stress_controller.kill_stress_process()
            raise urwid.ExitMainLoop()
        if data == "f1":
            graph_controller.view.on_help_menu_open(graph_controller.view.main_window_w)
        elif data == "esc":
            graph_controller.view.on_menu_close()


class StressController:
    def __init__(self, stress_installed):
        self.stress_modes = ["Monitor", "s-tui stress"]
        if stress_installed:
            self.stress_modes.append("Stress (ext)")
        self.current_mode = self.stress_modes[0]
        self.stress_process = None
        self._builtin_stresser = None

    def get_modes(self):
        return self.stress_modes

    def get_current_mode(self):
        return self.current_mode

    def set_mode(self, mode):
        self.current_mode = mode

    def get_stress_process(self):
        return self.stress_process

    def set_stress_process(self, proc):
        self.stress_process = proc

    @property
    def builtin_stresser(self):
        if self._builtin_stresser is None:
            self._builtin_stresser = BuiltinStresser()
        return self._builtin_stresser

    def kill_stress_process(self):
        try:
            if self.stress_process is not None:
                kill_child_processes(self.stress_process)
        except psutil.NoSuchProcess:
            logging.debug("Stress process no longer exists")
        self.stress_process = None
        if self._builtin_stresser is not None:
            self._builtin_stresser.stop()

    def start_stress(self, stress_cmd):
        with open(os.devnull, "w") as dev_null:
            try:
                process = subprocess.Popen(
                    stress_cmd,
                    stdout=dev_null,
                    stderr=dev_null,
                    start_new_session=True,
                )
                self.set_stress_process(psutil.Process(process.pid))
            except OSError:
                logging.debug("Unable to start stress")

    def start_builtin_stress(self, num_workers, strategy=None):
        try:
            self.builtin_stresser.start(num_workers, strategy=strategy)
        except OSError as err:
            logging.error("Unable to start built-in stresser: %s", err)
            self.current_mode = "Monitor"


def _widget_text(widget, text):
    if hasattr(widget, "set_text"):
        widget.set_text(text)


def _source_name(source):
    for attr in ("name", "label", "source_name", "title"):
        value = getattr(source, attr, None)
        if value:
            return str(value)
    return source.__class__.__name__.replace("Source", "")


def _source_data(source):
    for method in ("get_data", "read", "update"):
        fn = getattr(source, method, None)
        if callable(fn):
            value = fn()
            if value is not None:
                return value
    return getattr(source, "data", None)


def _safe_construct(cls, *args):
    try:
        return cls(*args)
    except TypeError:
        return cls()


class GraphView(urwid.WidgetPlaceholder):
    def __init__(self, controller):
        self.left_margin = 0
        self.top_margin = 0
        self.controller = controller
        self.main_window_w = []
        self.source_update_errors = {}
        self.clock_view = urwid.Text(ZERO_TIME, align="center")
        self.governor_view = urwid.Text("", align="center")
        self.epp_view = urwid.Text("", align="center")
        self._update_cpu_policy()
        self.refresh_rate_ctrl = urwid.Edit(
            "Refresh[s]:", str(self.controller.refresh_rate)
        )
        self.hline = urwid.AttrMap(urwid.SolidFill(" "), "line")
        self.vline = urwid.WidgetPlaceholder(
            urwid.AttrMap(urwid.SolidFill("|"), "line")
        )
        self.mode_buttons = []
        self.summary_widget_index = None
        self.graph_place_holder = urwid.WidgetPlaceholder(urwid.Pile([]))
        self.stress_menu = _safe_construct(
            StressMenu, self.on_menu_close, self.controller.stress_exe
        )
        self.builtin_stress_menu = _safe_construct(
            BuiltinStressMenu,
            self.on_menu_close,
            self.controller.stress_strategy,
            self.controller.stress_workers,
        )
        self.help_menu = _safe_construct(HelpMenu, self.on_menu_close)
        self.about_menu = _safe_construct(AboutMenu, self.on_menu_close)
        self.graphs_menu = _safe_construct(
            SensorsMenu,
            self.on_graphs_menu_close,
            self.controller.sources,
            self.controller.graphs_default_conf,
        )
        self.power_profile_menu = _safe_construct(
            PowerProfileMenu, self.on_menu_close
        )
        self.summary_view = self._make_summary()
        self._build_main_window()
        super().__init__(self.main_window)

    def _make_summary(self):
        try:
            return SummaryTextList()
        except TypeError:
            try:
                return SummaryTextList([])
            except TypeError:
                return urwid.Text("")

    def _update_cpu_policy(self):
        governor = ""
        epp = ""
        with contextlib.suppress(Exception):
            available = read_available(SYSFS_GOVERNOR, SYSFS_AVAIL_GOVERNORS)
            current = cat(SYSFS_GOVERNOR).strip()
            governor = "Governor: " + current
            if available:
                governor += " (" + ", ".join(available) + ")"
        with contextlib.suppress(Exception):
            available = read_available(SYSFS_EPP, SYSFS_AVAIL_EPP)
            current = cat(SYSFS_EPP).strip()
            epp = "EPP: " + current
            if available:
                epp += " (" + ", ".join(available) + ")"
        _widget_text(self.governor_view, governor)
        _widget_text(self.epp_view, epp)

    def _build_main_window(self):
        title = urwid.Text(
            ("title", "s-tui - " + get_processor_name()), align="center"
        )
        controls = [
            title,
            self.clock_view,
            self.governor_view,
            self.epp_view,
            self.refresh_rate_ctrl,
        ]
        self.mode_buttons = []
        group = []
        for mode in self.controller.stress_controller.get_modes():
            selected = mode == self.controller.stress_controller.get_current_mode()
            try:
                item = radio_button(group, mode, selected, self.on_mode_change, mode)
            except TypeError:
                item = urwid.RadioButton(
                    group, mode, state=selected, on_state_change=self.on_mode_change
                )
            self.mode_buttons.append(item)
            controls.append(item)
        for label, callback in (
            ("Sensors", self.on_graphs_menu_open),
            ("Power", self.on_power_profile_menu_open),
            ("Help", self.on_help_menu_open),
            ("About", self.on_about_menu_open),
            ("Quit", self.on_quit),
        ):
            try:
                controls.append(button(label, callback))
            except TypeError:
                controls.append(urwid.Button(label, callback))
        sidebar = urwid.Pile(controls)
        columns = urwid.Columns(
            [
                ("weight", 3, self.graph_place_holder),
                ("weight", 1, urwid.LineBox(sidebar)),
            ],
            dividechars=1,
        )
        self.main_window_w = [columns]
        self.main_window = urwid.Filler(columns, valign="top")

    def on_quit(self, *args):
        self.controller.stress_controller.kill_stress_process()
        raise urwid.ExitMainLoop()

    def on_mode_change(self, button_widget, state=True, mode=None):
        if not state:
            return
        if mode is None:
            mode = getattr(button_widget, "label", None)
            if mode is None and hasattr(button_widget, "get_label"):
                mode = button_widget.get_label()
        if mode is not None:
            self.controller.set_stress_mode(mode)

    def on_refresh_rate_change(self, *args):
        try:
            rate = float(self.refresh_rate_ctrl.edit_text)
            if rate > 0:
                self.controller.set_refresh_rate(rate)
        except (TypeError, ValueError):
            logging.debug("Invalid refresh rate")

    def on_menu_close(self, *args):
        self.original_widget = self.main_window

    def on_graphs_menu_close(self, *args):
        self.on_menu_close()
        self.controller.refresh_graphs()

    def on_help_menu_open(self, *args):
        self._open_menu(self.help_menu)

    def on_about_menu_open(self, *args):
        self._open_menu(self.about_menu)

    def on_graphs_menu_open(self, *args):
        self._open_menu(self.graphs_menu)

    def on_power_profile_menu_open(self, *args):
        self._open_menu(self.power_profile_menu)

    def _open_menu(self, menu):
        if isinstance(menu, urwid.Widget):
            self.original_widget = menu
            return
        for attr in ("main_window", "widget", "view"):
            widget = getattr(menu, attr, None)
            if isinstance(widget, urwid.Widget):
                self.original_widget = widget
                return

    def update_clock(self, seconds):
        _widget_text(self.clock_view, seconds_to_text(seconds))

    def update_graphs(self, graphs=None):
        graphs = self.controller.graphs if graphs is None else graphs
        widgets = []
        for name, graph in graphs.items():
            if isinstance(graph, urwid.Widget):
                widgets.append(graph)
                continue
            widgets.append(urwid.LineBox(urwid.Text(str(graph)), title=str(name)))
        self.graph_place_holder.original_widget = urwid.Pile(widgets or [urwid.Text("")])

    def update_summary(self, values):
        if hasattr(self.summary_view, "update"):
            with contextlib.suppress(Exception):
                self.summary_view.update(values)
                return
        if hasattr(self.summary_view, "set_text"):
            self.summary_view.set_text(str(values))


class GraphController:
    def __init__(
        self,
        refresh_rate=UPDATE_INTERVAL,
        stress_exe=None,
        stress_workers=None,
        stress_strategy=None,
        graphs_default_conf=None,
        sources=None,
        **kwargs
    ):
        if not isinstance(refresh_rate, (int, float, str)):
            args = refresh_rate
            refresh_rate = getattr(args, "refresh", getattr(args, "refresh_rate", UPDATE_INTERVAL))
            stress_exe = getattr(args, "stress_exe", getattr(args, "stress", stress_exe))
            stress_workers = getattr(args, "stress_workers", stress_workers)
            stress_strategy = getattr(args, "stress_strategy", stress_strategy)
        try:
            self.refresh_rate = float(refresh_rate)
        except (TypeError, ValueError):
            self.refresh_rate = UPDATE_INTERVAL
        self.stress_exe = stress_exe or which("stress")
        self.stress_workers = stress_workers
        self.stress_strategy = stress_strategy
        self.graphs_default_conf = graphs_default_conf or {}
        self.stress_controller = StressController(bool(self.stress_exe))
        self.sources = sources if sources is not None else self._create_sources()
        if not isinstance(self.sources, OrderedDict):
            self.sources = OrderedDict(
                (_source_name(source), source) for source in self.sources
            )
        self.graphs = OrderedDict()
        self.values = OrderedDict()
        self.start_time = time.time()
        self.last_hook_time = 0
        self.view = None
        self.loop = None
        self.script_hook_loader = kwargs.get("script_hook_loader")
        if self.script_hook_loader is None:
            with contextlib.suppress(Exception):
                self.script_hook_loader = ScriptHookLoader()

    def _create_sources(self):
        result = []
        for source_type in (
            UtilSource,
            FreqSource,
            TempSource,
            FanSource,
            RaplPowerSource,
        ):
            with contextlib.suppress(Exception):
                result.append(source_type())
        return result

    def set_refresh_rate(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("refresh rate must be positive")
        self.refresh_rate = value

    def set_stress_mode(self, mode):
        old_mode = self.stress_controller.get_current_mode()
        if old_mode == mode:
            return
        self.stress_controller.kill_stress_process()
        self.stress_controller.set_mode(mode)
        if mode == "Stress (ext)" and self.stress_exe:
            command = self.stress_exe
            if isinstance(command, str):
                command = [command]
            self.stress_controller.start_stress(command)
        elif mode == "s-tui stress":
            workers = self.stress_workers or psutil.cpu_count() or 1
            self.stress_controller.start_builtin_stress(
                workers, strategy=self.stress_strategy
            )

    def refresh_graphs(self):
        if self.view is not None:
            self.view.update_graphs()

    def update(self):
        now = time.time()
        self.values = OrderedDict()
        for name, source in self.sources.items():
            try:
                data = _source_data(source)
                self.values[name] = data
                if name in getattr(self.view, "source_update_errors", {}):
                    del self.view.source_update_errors[name]
            except Exception as err:
                previous = None
                if self.view is not None:
                    previous = self.view.source_update_errors.get(name)
                    self.view.source_update_errors[name] = str(err)
                if previous != str(err):
                    logging.warning("Unable to update %s: %s", name, err)
        if self.view is not None:
            self.view.update_clock(now - self.start_time)
            self.view.update_summary(self.values)
        return self.values

    def _alarm(self, loop=None, user_data=None):
        self.update()
        if self.loop is not None:
            self.loop.set_alarm_in(self.refresh_rate, self._alarm)

    def run(self):
        global graph_controller
        graph_controller = self
        self.view = GraphView(self)
        self.refresh_graphs()
        self.loop = MainLoop(
            self.view,
            palette=DEFAULT_PALETTE,
            unhandled_input=None,
        )
        self.loop.set_alarm_in(self.refresh_rate, self._alarm)
        try:
            self.loop.run()
        finally:
            self.stress_controller.kill_stress_process()


def get_config():
    config = configparser.ConfigParser()
    if user_config_file_exists():
        with contextlib.suppress(OSError, configparser.Error):
            config.read(get_user_config_file())
    return config


def write_config(config):
    if not user_config_dir_exists():
        make_user_config_dir()
    with open(get_user_config_file(), "w") as config_file:
        config.write(config_file)


def get_parser():
    parser = argparse.ArgumentParser(
        description="CPU stress and monitoring utility",
        epilog=HELP_MESSAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=VERSION_MESSAGE)
    parser.add_argument(
        "-r", "--refresh", "--refresh-rate", dest="refresh", type=float,
        default=UPDATE_INTERVAL, help="refresh interval in seconds"
    )
    parser.add_argument(
        "-s", "--stress", action="store_true", help="start external stress"
    )
    parser.add_argument(
        "--stress-exe", default=which("stress"), help="external stress executable"
    )
    parser.add_argument(
        "--stress-workers", type=int, default=None,
        help="number of workers for built-in stress"
    )
    parser.add_argument(
        "--stress-strategy", default=None, help="built-in stress strategy"
    )
    parser.add_argument("--csv", nargs="?", const=DEFAULT_CSV_FILE)
    parser.add_argument("--json", nargs="?", const="s-tui_log.json")
    parser.add_argument("--terminal", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", action="store_true")
    return parser


def parse_args(args=None):
    return get_parser().parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    if parsed.debug:
        logging.basicConfig(
            filename=DEFAULT_LOG_FILE,
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
        )
    controller = GraphController(parsed)
    atexit.register(controller.stress_controller.kill_stress_process)
    if parsed.stress:
        controller.set_stress_mode("Stress (ext)")
    if parsed.terminal or parsed.csv or parsed.json:
        values = controller.update()
        if parsed.terminal:
            output_to_terminal(values)
        if parsed.csv:
            output_to_csv(values, parsed.csv)
        if parsed.json:
            output_to_json(values, parsed.json)
        return 0
    controller.run()
    return 0


if __name__ == "__main__":
    main()