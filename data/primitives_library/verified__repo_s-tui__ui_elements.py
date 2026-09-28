import urwid

DEFAULT_PALETTE = [
    ("body", "default", "default", "standout"),
    ("header", "default", "dark red"),
    ("screen edge", "light blue", "brown"),
    ("main shadow", "dark gray", "black"),
    ("line", "default", "light gray", "standout"),
    ("menu button", "light gray", "black"),
    ("bg background", "default", "default"),
    ("overheat dark", "white", "light red", "standout"),
    ("bold text", "default,bold", "default", "bold"),
    ("under text", "default,underline", "default", "underline"),
    ("util light", "default", "light green"),
    ("util light smooth", "light green", "default"),
    ("util dark", "default", "dark green"),
    ("util dark smooth", "dark green", "default"),
    ("high temp dark", "default", "dark red"),
    ("high temp dark smooth", "dark red", "default"),
    ("high temp light", "default", "light red"),
    ("high temp light smooth", "light red", "default"),
    ("power dark", "default", "light gray", "standout"),
    ("power dark smooth", "light gray", "default"),
    ("power light", "default", "white", "standout"),
    ("power light smooth", "white", "default"),
    ("temp dark", "default", "dark cyan", "standout"),
    ("temp dark smooth", "dark cyan", "default"),
    ("temp light", "default", "light cyan", "standout"),
    ("temp light smooth", "light cyan", "default"),
    ("freq dark", "default", "dark magenta", "standout"),
    ("freq dark smooth", "dark magenta", "default"),
    ("freq light", "default", "light magenta", "standout"),
    ("freq light smooth", "light magenta", "default"),
    ("fan dark", "default", "dark blue", "standout"),
    ("fan dark smooth", "dark blue", "default"),
    ("fan light", "default", "light blue", "standout"),
    ("fan light smooth", "light blue", "default"),
    ("button normal", "dark green", "default", "standout"),
    ("button select", "white", "dark green"),
    ("line", "default", "default", "standout"),
    ("pg normal", "white", "default", "standout"),
    ("pg complete", "white", "dark magenta"),
    ("high temp txt", "light red", "default"),
    ("throttle txt", "yellow", "default"),
    ("freq throttle dark", "default", "brown", "standout"),
    ("freq throttle dark smooth", "brown", "default"),
    ("freq throttle light", "default", "yellow", "standout"),
    ("freq throttle light smooth", "yellow", "default"),
    ("pg smooth", "dark magenta", "default"),
]


class ViListBox(urwid.ListBox):
    def keypress(self, size, key):
        translations = {
            "j": "down",
            "k": "up",
            "h": "left",
            "l": "right",
            "G": "page down",
            "g": "page up",
            "x": "enter",
            "q": "q",
        }
        return super().keypress(size, translations.get(key, key))


def radio_button(group, label, fn):
    widget = urwid.RadioButton(group, label, False, on_state_change=fn)
    return urwid.AttrMap(widget, "button normal", "button select")


def button(t, fn, data=None):
    widget = urwid.Button(t, fn, data)
    return urwid.AttrMap(widget, "button normal", "button select")