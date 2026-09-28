from __future__ import annotations

import logging
import re
from collections.abc import Callable

import psutil
import urwid

from s_tui.builtin_stresser import (
    STRATEGIES,
    STRATEGY_LABELS,
    STRATEGY_REQUIREMENTS,
    get_default_strategy,
    strategy_available,
)


class BuiltinStressMenu:
    MAX_TITLE_LEN = 50

    def __init__(
        self,
        return_fn: Callable[[], None],
        initial_strategy: str | None = None,
        initial_workers: str | None = None,
    ) -> None:
        self.return_fn = return_fn

        self.num_workers = "1"
        try:
            self.num_workers = str(psutil.cpu_count() or 1)
            logging.info("builtin stress default workers %s", self.num_workers)
        except OSError as err:
            logging.debug(err)

        if (
            initial_workers is not None
            and re.match(r"\A[0-9]+\Z", initial_workers)
            and int(initial_workers) > 0
        ):
            self.num_workers = initial_workers

        self.strategy = get_default_strategy()
        if initial_strategy in STRATEGIES and strategy_available(initial_strategy):
            self.strategy = initial_strategy

        self.num_workers_ctrl = urwid.Edit("CPU worker count: ", self.num_workers)

        self._strategy_group: list[urwid.RadioButton] = []
        self._strategy_buttons: dict[str, urwid.RadioButton] = {}
        strategy_widgets: list[urwid.Widget] = []

        for key in STRATEGIES:
            label = STRATEGY_LABELS[key]
            if strategy_available(key):
                button = urwid.RadioButton(
                    self._strategy_group,
                    label,
                    state=(key == self.strategy),
                )
                self._strategy_buttons[key] = button
                strategy_widgets.append(button)
            else:
                requirement = STRATEGY_REQUIREMENTS.get(key, "unavailable")
                strategy_widgets.append(
                    urwid.Text(
                        [
                            f"  {label} ",
                            ("high temp txt", f"({requirement})"),
                        ]
                    )
                )

        default_button = urwid.Button("Default", on_press=self.on_default)
        default_button._label.align = "center"

        apply_button = urwid.Button("Apply", on_press=self.on_save)
        apply_button._label.align = "center"

        cancel_button = urwid.Button("Cancel", on_press=self.on_cancel)
        cancel_button._label.align = "center"

        buttons = urwid.Columns([default_button, apply_button, cancel_button])
        heading = urwid.Text(("bold text", "  s-tui stress options  \n"), "center")

        self.titles = [
            heading,
            self.num_workers_ctrl,
            urwid.Divider("-"),
            urwid.Text(("bold text", "Strategy:")),
            *strategy_widgets,
            urwid.Divider("-"),
            buttons,
        ]

        self.main_window = urwid.LineBox(urwid.ListBox(self.titles))

    def _selected_strategy(self) -> str:
        for key, button in self._strategy_buttons.items():
            if button.state:
                return key
        return self.strategy

    def get_size(self) -> tuple[int, int]:
        return len(self.titles) + 5, self.MAX_TITLE_LEN

    def get_num_workers(self) -> int:
        try:
            return max(1, int(self.num_workers))
        except ValueError:
            return 1

    def get_strategy(self) -> str:
        return self.strategy

    def _restore_ui(self) -> None:
        self.num_workers_ctrl.set_edit_text(self.num_workers)
        self._strategy_buttons[self.strategy].set_state(True)

    def on_default(self, _) -> None:
        self.num_workers = str(psutil.cpu_count() or 1)
        self.strategy = get_default_strategy()
        self._restore_ui()
        self.return_fn()

    def on_save(self, _) -> None:
        raw = self.num_workers_ctrl.get_edit_text()
        if re.match(r"\A[0-9]+\Z", raw) and int(raw) > 0:
            self.num_workers = raw
        else:
            self.num_workers = str(psutil.cpu_count() or 1)

        self.strategy = self._selected_strategy()
        self._restore_ui()
        self.return_fn()

    def on_cancel(self, _) -> None:
        self._restore_ui()
        self.return_fn()