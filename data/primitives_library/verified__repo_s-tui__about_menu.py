import urwid

from s_tui.helper_functions import __version__
from s_tui.sturwid.ui_elements import ViListBox


ABOUT_MESSAGE = """
s-tui is a monitoring tool for your CPU's temperature, frequency, utilization and power.

Code for s-tui is available on github
https://github.com/amanusk/s-tui

Help, issues and pull requests are appreciated.

Created by:
    - Alex Manuskin
    - Gil Tsuker
    - Maor Veitsman
    And others

April 2017

"""

ABOUT_MESSAGE += "s-tui " + __version__ + " Released under GNU GPLv2 "

MESSAGE_LEN = 20


class AboutMenu:
    """Displays the About message menu"""

    MAX_TITLE_LEN = 50

    def __init__(self, return_fn):
        self.return_fn = return_fn
        self.about_message = ABOUT_MESSAGE
        self.time_out_ctrl = urwid.Text(self.about_message)

        cancel_button = urwid.Button("Exit", on_press=self.on_cancel)
        cancel_button._label.align = "center"

        if_buttons = urwid.Columns([cancel_button])
        title = urwid.Text(("bold text", "  About Menu  \n"), "center")

        self.titles = [title, self.time_out_ctrl, if_buttons]
        self.main_window = urwid.LineBox(ViListBox(self.titles))

    def get_size(self):
        return MESSAGE_LEN + 3, self.MAX_TITLE_LEN

    def on_cancel(self, w):
        self.return_fn()