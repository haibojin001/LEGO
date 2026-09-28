from __future__ import annotations

import glob
import logging
import subprocess
from collections.abc import Callable

import urwid

from s_tui.helper_functions import cat
from s_tui.sturwid.ui_elements import ViListBox

SYSFS_AVAIL_GOVERNORS = (
    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors"
)
SYSFS_GOVERNOR = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
SYSFS_AVAIL_EPP = (
    "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_available_preferences"
)
SYSFS_EPP = "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"

_SYSFS_ALL_GOVERNORS = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
_SYSFS_ALL_EPP = "/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference"

_EPP_TO_PROFILE = {
    "performance": "performance",
    "balance_performance": "balanced",
    "power": "power-saver",
}


def read_available(path: str) -> list[str]:
    try:
        return cat(path, binary=False).split()
    except OSError:
        return []


def _read_current(path: str) -> str:
    try:
        return cat(path, binary=False).strip()
    except OSError:
        return ""


def _write_all_cores(pattern: str, value: str) -> None:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise OSError(f"No sysfs paths found for {pattern}")

    errors: list[OSError] = []
    for path in paths:
        try:
            with open(path, "w") as handle:
                handle.write(value)
        except OSError as error:
            errors.append(error)

    if not errors:
        return

    if _read_current(paths[0]).strip() == value:
        logging.debug(
            "%d/%d cores reported errors writing '%s', but value applied successfully",
            len(errors),
            len(paths),
            value,
        )
        return

    reasons = {error.strerror or str(error) for error in errors}
    if len(reasons) == 1:
        raise OSError(f"{reasons.pop()} (all {len(errors)} cores)")

    raise OSError("; ".join(f"{error.filename}: {error}" for error in errors))


def _set_epp_via_powerprofilesctl(exe: str, epp_value: str) -> None:
    profile = _EPP_TO_PROFILE.get(epp_value)
    if profile is None:
        raise OSError(
            f"No powerprofilesctl mapping for '{epp_value}', "
            "direct sysfs write required"
        )

    try:
        result = subprocess.run(
            [exe, "set", profile],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired as error:
        raise OSError(f"powerprofilesctl timed out setting '{profile}'") from error

    if result.returncode != 0:
        if "busy" in result.stderr.strip().lower():
            governor = _read_current(SYSFS_GOVERNOR) or "unknown"
            raise OSError(f"Cannot change EPP (governor: {governor})")
        raise OSError(f"powerprofilesctl set {profile} failed")


class PowerProfileMenu:
    MAX_TITLE_LEN = 50

    def __init__(
        self,
        return_fn: Callable[[], None],
        powerprofilesctl_exe: str | None,
        can_write_governor: bool,
        can_write_epp: bool,
        available_governors: list[str] | None = None,
        available_epp: list[str] | None = None,
        update_size_fn: Callable[[int, int], None] | None = None,
    ) -> None:
        self.return_fn = return_fn
        self.update_size_fn = update_size_fn
        self.powerprofilesctl_exe = powerprofilesctl_exe
        self.can_write_governor = can_write_governor
        self.can_write_epp = can_write_epp

        self.available_governors = (
            available_governors
            if available_governors is not None
            else read_available(SYSFS_AVAIL_GOVERNORS)
        )
        self.available_epp = (
            available_epp
            if available_epp is not None
            else read_available(SYSFS_AVAIL_EPP)
        )

        self.governor_controllable = (
            can_write_governor and len(self.available_governors) > 1
        )
        self.epp_controllable = (
            can_write_epp or powerprofilesctl_exe is not None
        ) and len(self.available_epp) > 0

        self.status_text = urwid.Text("")
        self.titles: list[urwid.Widget] = []
        self._build_ui()

        self.walker = urwid.SimpleFocusListWalker(self.titles)
        self.main_window = urwid.LineBox(ViListBox(self.walker))

    def _build_ui(self) -> None:
        title = urwid.Text(("bold text", "  Power Profile  \n"), "center")
        self.titles = [title]

        self.governor_group: list[urwid.RadioButton] = []
        self.governor_buttons: list[urwid.AttrMap] = []
        if len(self.available_governors) > 1:
            self.titles.append(urwid.Text(("bold text", "Governor"), align="center"))
            current_governor = _read_current(SYSFS_GOVERNOR)

            if self.governor_controllable:
                for governor in self.available_governors:
                    button = urwid.RadioButton(
                        self.governor_group,
                        governor,
                        state=governor == current_governor,
                    )
                    mapped_button = urwid.AttrMap(
                        button, "button normal", "button select"
                    )
                    self.governor_buttons.append(mapped_button)
                    self.titles.append(mapped_button)
            else:
                for governor in self.available_governors:
                    selected = " *" if governor == current_governor else ""
                    self.titles.append(urwid.Text(f"  {governor}{selected}"))
                self.titles.append(
                    urwid.Text(("high temp txt", "  (read-only, needs root)"))
                )

            self.titles.append(urwid.Divider())

        self.epp_group: list[urwid.RadioButton] = []
        self.epp_buttons: list[urwid.AttrMap] = []
        if self.available_epp:
            self.titles.append(
                urwid.Text(("bold text", "Energy Pref"), align="center")
            )
            current_epp = _read_current(SYSFS_EPP)

            if self.epp_controllable:
                for epp in self.available_epp:
                    button = urwid.RadioButton(
                        self.epp_group,
                        epp,
                        state=epp == current_epp,
                    )
                    mapped_button = urwid.AttrMap(
                        button, "button normal", "button select"
                    )
                    self.epp_buttons.append(mapped_button)
                    self.titles.append(mapped_button)
            else:
                for epp in self.available_epp:
                    selected = " *" if epp == current_epp else ""
                    self.titles.append(urwid.Text(f"  {epp}{selected}"))
                self.titles.append(urwid.Text(("high temp txt", "  (read-only)")))

            self.titles.append(urwid.Divider())

        self.titles.append(self.status_text)

        apply_button = urwid.Button("Apply", on_press=self.on_apply)
        apply_button._label.align = "center"
        close_button = urwid.Button("Close", on_press=self.on_close)
        close_button._label.align = "center"
        self.titles.append(urwid.Columns([apply_button, close_button]))

    def get(self) -> urwid.Widget:
        if self.update_size_fn is not None:
            self.update_size_fn(self.MAX_TITLE_LEN, len(self.titles) + 2)
        return self.main_window

    @staticmethod
    def _selected_value(buttons: list[urwid.AttrMap]) -> str | None:
        for mapped_button in buttons:
            button = mapped_button.original_widget
            if button.get_state():
                return button.get_label()
        return None

    def on_apply(self, _button: urwid.Button) -> None:
        try:
            if self.governor_controllable:
                governor = self._selected_value(self.governor_buttons)
                if governor is not None:
                    _write_all_cores(_SYSFS_ALL_GOVERNORS, governor)

            if self.epp_controllable:
                epp = self._selected_value(self.epp_buttons)
                if epp is not None:
                    if self.can_write_epp:
                        _write_all_cores(_SYSFS_ALL_EPP, epp)
                    elif self.powerprofilesctl_exe is not None:
                        _set_epp_via_powerprofilesctl(self.powerprofilesctl_exe, epp)

        except OSError as error:
            self.status_text.set_text(("high temp txt", f"  Error: {error}"))
            return

        self.status_text.set_text(("bold text", "  Applied successfully"))

    def on_close(self, _button: urwid.Button) -> None:
        self.return_fn()