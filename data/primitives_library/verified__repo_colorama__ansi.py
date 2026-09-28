CSI = "\033["
OSC = "\033]"
BEL = "\a"


def code_to_chars(code):
    return "{}{}m".format(CSI, code)


def set_title(title):
    return "{}2;{}{}".format(OSC, title, BEL)


def clear_screen(mode=2):
    return "{}{}J".format(CSI, mode)


def clear_line(mode=2):
    return "{}{}K".format(CSI, mode)


class AnsiCodes:
    def __init__(self):
        for attribute in dir(self):
            if attribute.startswith("_"):
                continue
            value = getattr(self, attribute)
            setattr(self, attribute, code_to_chars(value))


class AnsiCursor:
    def UP(self, n=1):
        return "{}{}A".format(CSI, n)

    def DOWN(self, n=1):
        return "{}{}B".format(CSI, n)

    def FORWARD(self, n=1):
        return "{}{}C".format(CSI, n)

    def BACK(self, n=1):
        return "{}{}D".format(CSI, n)

    def POS(self, x=1, y=1):
        return "{}{};{}H".format(CSI, y, x)


class AnsiFore(AnsiCodes):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    RESET = 39

    LIGHTBLACK_EX = 90
    LIGHTRED_EX = 91
    LIGHTGREEN_EX = 92
    LIGHTYELLOW_EX = 93
    LIGHTBLUE_EX = 94
    LIGHTMAGENTA_EX = 95
    LIGHTCYAN_EX = 96
    LIGHTWHITE_EX = 97


class AnsiBack(AnsiCodes):
    BLACK = 40
    RED = 41
    GREEN = 42
    YELLOW = 43
    BLUE = 44
    MAGENTA = 45
    CYAN = 46
    WHITE = 47
    RESET = 49

    LIGHTBLACK_EX = 100
    LIGHTRED_EX = 101
    LIGHTGREEN_EX = 102
    LIGHTYELLOW_EX = 103
    LIGHTBLUE_EX = 104
    LIGHTMAGENTA_EX = 105
    LIGHTCYAN_EX = 106
    LIGHTWHITE_EX = 107


class AnsiStyle(AnsiCodes):
    BRIGHT = 1
    DIM = 2
    NORMAL = 22
    RESET_ALL = 0


Fore = AnsiFore()
Back = AnsiBack()
Style = AnsiStyle()
Cursor = AnsiCursor()