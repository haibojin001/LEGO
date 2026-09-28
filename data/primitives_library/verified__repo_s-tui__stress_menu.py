from __future__ import annotations

import logging
import re
from collections.abc import Callable

import psutil
import urwid


class StressMenu:
    MAX_TITLE_LEN = 50

    def __init__(self, return_fn: Callable[[], None], stress_exe: str | None) -> None:
        self.return_fn = return_fn
        self.stress_exe = stress_exe

        self.time_out = "none"
        self.sqrt_workers = "1"
        try:
            self.sqrt_workers = str(psutil.cpu_count())
            logging.info("num cpus %s", self.sqrt_workers)
        except OSError as err:
            logging.debug(err)

        self.sync_workers = "0"
        self.memory_workers = "0"
        self.malloc_byte = "256M"
        self.byte_touch_cnt = "4096"
        self.malloc_delay = "none"
        self.no_malloc = False
        self.write_workers = "0"
        self.write_bytes = "1G"

        self.time_out_ctrl = urwid.Edit("Time out [sec]: ", self.time_out)
        self.sqrt_workers_ctrl = urwid.Edit("Sqrt() worker count: ", self.sqrt_workers)
        self.sync_workers_ctrl = urwid.Edit("Sync() worker count: ", self.sync_workers)
        self.memory_workers_ctrl = urwid.Edit(
            "Malloc() / Free() worker count: ", self.memory_workers
        )
        self.malloc_byte_ctrl = urwid.Edit("   Bytes per malloc*: ", self.malloc_byte)
        self.byte_touch_cnt_ctrl = urwid.Edit(
            "   Touch a byte after * bytes: ", self.byte_touch_cnt
        )
        self.malloc_delay_ctrl = urwid.Edit(
            "   Sleep time between Free() [sec]: ", self.malloc_delay
        )
        self.no_malloc_ctrl = urwid.CheckBox(
            '"dirty" the memory \ninstead of free / alloc', self.no_malloc
        )
        self.write_workers_ctrl = urwid.Edit(
            "Write() / Unlink() worker count: ", self.write_workers
        )
        self.write_bytes_ctrl = urwid.Edit("   Byte per Write(): ", self.write_bytes)

        default_button = urwid.Button("Default", on_press=self.on_default)
        default_button._label.align = "center"

        save_button = urwid.Button("Save", on_press=self.on_save)
        save_button._label.align = "center"

        cancel_button = urwid.Button("Cancel", on_press=self.on_cancel)
        cancel_button._label.align = "center"

        buttons = urwid.Columns([default_button, save_button, cancel_button])
        heading = urwid.Text(("bold text", "  Stress Options  \n"), "center")

        self.titles = [
            heading,
            self.time_out_ctrl,
            urwid.Divider("-"),
            self.sqrt_workers_ctrl,
            urwid.Divider("-"),
            self.sync_workers_ctrl,
            urwid.Divider("-"),
            self.memory_workers_ctrl,
            urwid.Divider(),
            self.malloc_byte_ctrl,
            urwid.Divider(),
            self.byte_touch_cnt_ctrl,
            urwid.Divider(),
            self.malloc_delay_ctrl,
            urwid.Divider(),
            self.no_malloc_ctrl,
            urwid.Divider("-"),
            self.write_workers_ctrl,
            urwid.Divider(),
            self.write_bytes_ctrl,
            urwid.Divider("-"),
            buttons,
        ]

        self.main_window = urwid.LineBox(urwid.ListBox(self.titles))

    def set_edit_texts(self) -> None:
        self.time_out_ctrl.set_edit_text(self.time_out)
        self.sqrt_workers_ctrl.set_edit_text(self.sqrt_workers)
        self.sync_workers_ctrl.set_edit_text(self.sync_workers)
        self.memory_workers_ctrl.set_edit_text(self.memory_workers)
        self.malloc_byte_ctrl.set_edit_text(self.malloc_byte)
        self.byte_touch_cnt_ctrl.set_edit_text(self.byte_touch_cnt)
        self.malloc_delay_ctrl.set_edit_text(self.malloc_delay)
        self.no_malloc_ctrl.set_state(bool(self.no_malloc))
        self.write_workers_ctrl.set_edit_text(self.write_workers)
        self.write_bytes_ctrl.set_edit_text(self.write_bytes)

    def on_default(self, _):
        self.time_out = "none"
        self.sqrt_workers = "1"
        self.sync_workers = "0"
        self.memory_workers = "0"
        self.malloc_byte = "256M"
        self.byte_touch_cnt = "4096"
        self.malloc_delay = "none"
        self.no_malloc = False
        self.write_workers = "0"
        self.write_bytes = "1G"

        self.set_edit_texts()
        self.return_fn()

    def get_size(self) -> tuple[int, int]:
        return len(self.titles) + 5, self.MAX_TITLE_LEN

    def on_save(self, _):
        self.time_out = self.get_pos_num(self.time_out_ctrl.get_edit_text(), "none")
        self.sqrt_workers = self.get_pos_num(
            self.sqrt_workers_ctrl.get_edit_text(), "4"
        )
        self.sync_workers = self.get_pos_num(
            self.sync_workers_ctrl.get_edit_text(), "0"
        )
        self.memory_workers = self.get_pos_num(
            self.memory_workers_ctrl.get_edit_text(), "0"
        )
        self.malloc_byte = self.get_valid_byte(
            self.malloc_byte_ctrl.get_edit_text(), "256M"
        )
        self.byte_touch_cnt = self.get_valid_byte(
            self.byte_touch_cnt_ctrl.get_edit_text(), "4096"
        )
        self.malloc_delay = self.get_pos_num(
            self.malloc_delay_ctrl.get_edit_text(), "none"
        )
        self.no_malloc = self.no_malloc_ctrl.get_state()
        self.write_workers = self.get_pos_num(
            self.write_workers_ctrl.get_edit_text(), "0"
        )
        self.write_bytes = self.get_valid_byte(
            self.write_bytes_ctrl.get_edit_text(), "1G"
        )

        self.set_edit_texts()
        self.return_fn()

    def on_cancel(self, _):
        self.set_edit_texts()
        self.return_fn()

    def get_stress_cmd(self) -> list[str]:
        assert self.stress_exe is not None

        command = [self.stress_exe]

        if int(self.sqrt_workers) > 0:
            command.extend(["-c", self.sqrt_workers])

        if int(self.sync_workers) > 0:
            command.extend(["-i", self.sync_workers])

        if int(self.memory_workers) > 0:
            command.extend(
                [
                    "--vm",
                    self.memory_workers,
                    "--vm-bytes",
                    self.malloc_byte,
                    "--vm-stride",
                    self.byte_touch_cnt,
                ]
            )

        if self.no_malloc:
            command.append("--vm-keep")

        if int(self.write_workers) > 0:
            command.extend(
                [
                    "--hdd",
                    self.write_workers,
                    "--hdd-bytes",
                    self.write_bytes,
                ]
            )

        if self.time_out != "none":
            command.extend(["-t", self.time_out])

        return command

    @staticmethod
    def get_pos_num(num: str, default: str) -> str:
        if re.match(r"\A([0-9]+)\Z", num, re.I) or (
            num == "none" and default == "none"
        ):
            return num
        return default

    @staticmethod
    def get_valid_byte(num: str, default: str) -> str:
        match = re.match(r"\A([0-9]+)(M|G|m|g|)(B|b|\b)\Z", num, re.I)
        if match:
            return num
        return default