"""Help menu widget for s-tui."""

import urwid

from s_tui.sturwid.ui_elements import ViListBox


HELP_MESSAGE = (
    "\n"
    "TUI interface:\n"
    "\n"
    "The side bar houses the controls for the displayed graphs.\n"
    "At the bottom, all sensors reading are presented in text form.\n"
    "\n"
    "* Use the arrow keys or 'hjkl' to navigate the side bar\n"
    "* Toggle between stressed and regular operation using the radio buttons in "
    "'Modes'.\n"
    "* If you wish to alternate stress defaults, you can do it in <Stress "
    "options>\n"
    "* Select graphs to display in the <Graphs> menu \n"
    "* Select summaries to display in the <Summaries> menu \n"
    "* Change time between updates using the 'Refresh' field\n"
    "* Use the <Reset> button to reset graphs and statistics\n"
    "* If your system supports it, you can use the UTF-8 button to get a smoother "
    "graph\n"
    "* Save your current configuration with the <Save Settings> button\n"
    "* Press 'q' or the <Quit> button to quit\n"
    "\n"
    "* Run `s-tui --help` to get this message and additional cli options\n"
    "\n"
    "Throttle indicators (shown on frequency labels):\n"
    "  Intel, with root + msr module:\n"
    "    T = Thermal    H = PROCHOT (external)\n"
    "    C = Critical   W = Power limit (watts)\n"
    "    A = Current limit (amps)  X = Cross-domain\n"
    "  AMD, with root + msr module:\n"
    "    Pc = P-state cap (SMU barred the top P-state)\n"
    "  Without root (sysfs fallback):\n"
    "    Tc = Core thermal   Tp = Package thermal\n"
    "  Labels combine with / (e.g. T/W = thermal + power limit)\n"
)

MESSAGE_LEN = 40


class HelpMenu:
    """Widget presenting usage instructions."""

    MAX_TITLE_LEN = 90

    def __init__(self, return_fn):
        self.return_fn = return_fn
        self.help_message = HELP_MESSAGE
        self.time_out_ctrl = urwid.Text(self.help_message)

        cancel_button = urwid.Button("Exit", on_press=self.on_cancel)
        cancel_button._label.align = "center"

        if_buttons = urwid.Columns([cancel_button])
        title = urwid.Text(("bold text", "  Help Menu  \n"), "center")

        self.titles = [title, self.time_out_ctrl, if_buttons]
        self.main_window = urwid.LineBox(ViListBox(self.titles))

    def get_size(self):
        """Return the preferred dimensions of this menu."""
        return MESSAGE_LEN + 3, self.MAX_TITLE_LEN

    def on_cancel(self, w):
        """Restore the widget shown before the help menu."""
        self.return_fn()