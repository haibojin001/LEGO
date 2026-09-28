import os
import re
import sys

from .ansi import AnsiBack, AnsiFore, AnsiStyle, BEL, Style
from .win32 import windll, winapi_test
from .winterm import WinColor, WinStyle, WinTerm, enable_vt_processing


winterm = WinTerm() if windll is not None else None


class StreamWrapper:
    def __init__(self, wrapped, converter):
        self.__wrapped = wrapped
        self.__convertor = converter

    def __getattr__(self, name):
        return getattr(self.__wrapped, name)

    def __enter__(self, *args, **kwargs):
        return self.__wrapped.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        return self.__wrapped.__exit__(*args, **kwargs)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def write(self, text):
        self.__convertor.write(text)

    def isatty(self):
        stream = self.__wrapped
        if "PYCHARM_HOSTED" in os.environ:
            if stream is not None and (
                stream is sys.__stdout__ or stream is sys.__stderr__
            ):
                return True
        try:
            checker = stream.isatty
        except AttributeError:
            return False
        return checker()

    @property
    def closed(self):
        try:
            return self.__wrapped.closed
        except (AttributeError, ValueError):
            return True


class AnsiToWin32:
    ANSI_CSI_RE = re.compile("\001?\033\\[((?:\\d|;)*)([a-zA-Z])\002?")
    ANSI_OSC_RE = re.compile("\001?\033\\]([^\a]*)(\a)\002?")

    def __init__(self, wrapped, convert=None, strip=None, autoreset=False):
        self.wrapped = wrapped
        self.autoreset = autoreset
        self.stream = StreamWrapper(wrapped, self)

        on_windows = os.name == "nt"
        can_use_winapi = on_windows and winapi_test()

        try:
            fd = wrapped.fileno()
        except Exception:
            fd = -1

        native_ansi = not on_windows or enable_vt_processing(fd)
        tty = not self.stream.closed and self.stream.isatty()
        requires_conversion = can_use_winapi and not native_ansi

        if strip is None:
            strip = requires_conversion or not tty
        self.strip = strip

        if convert is None:
            convert = requires_conversion and tty
        self.convert = convert

        self.win32_calls = self.get_win32_calls()
        self.on_stderr = wrapped is sys.stderr

    def should_wrap(self):
        return self.convert or self.strip or self.autoreset

    def get_win32_calls(self):
        if not (self.convert and winterm):
            return {}

        return {
            AnsiStyle.RESET_ALL: (winterm.reset_all,),
            AnsiStyle.BRIGHT: (winterm.style, WinStyle.BRIGHT),
            AnsiStyle.DIM: (winterm.style, WinStyle.NORMAL),
            AnsiStyle.NORMAL: (winterm.style, WinStyle.NORMAL),
            AnsiFore.BLACK: (winterm.fore, WinColor.BLACK),
            AnsiFore.RED: (winterm.fore, WinColor.RED),
            AnsiFore.GREEN: (winterm.fore, WinColor.GREEN),
            AnsiFore.YELLOW: (winterm.fore, WinColor.YELLOW),
            AnsiFore.BLUE: (winterm.fore, WinColor.BLUE),
            AnsiFore.MAGENTA: (winterm.fore, WinColor.MAGENTA),
            AnsiFore.CYAN: (winterm.fore, WinColor.CYAN),
            AnsiFore.WHITE: (winterm.fore, WinColor.GREY),
            AnsiFore.RESET: (winterm.fore,),
            AnsiFore.LIGHTBLACK_EX: (winterm.fore, WinColor.BLACK, True),
            AnsiFore.LIGHTRED_EX: (winterm.fore, WinColor.RED, True),
            AnsiFore.LIGHTGREEN_EX: (winterm.fore, WinColor.GREEN, True),
            AnsiFore.LIGHTYELLOW_EX: (winterm.fore, WinColor.YELLOW, True),
            AnsiFore.LIGHTBLUE_EX: (winterm.fore, WinColor.BLUE, True),
            AnsiFore.LIGHTMAGENTA_EX: (winterm.fore, WinColor.MAGENTA, True),
            AnsiFore.LIGHTCYAN_EX: (winterm.fore, WinColor.CYAN, True),
            AnsiFore.LIGHTWHITE_EX: (winterm.fore, WinColor.GREY, True),
            AnsiBack.BLACK: (winterm.back, WinColor.BLACK),
            AnsiBack.RED: (winterm.back, WinColor.RED),
            AnsiBack.GREEN: (winterm.back, WinColor.GREEN),
            AnsiBack.YELLOW: (winterm.back, WinColor.YELLOW),
            AnsiBack.BLUE: (winterm.back, WinColor.BLUE),
            AnsiBack.MAGENTA: (winterm.back, WinColor.MAGENTA),
            AnsiBack.CYAN: (winterm.back, WinColor.CYAN),
            AnsiBack.WHITE: (winterm.back, WinColor.GREY),
            AnsiBack.RESET: (winterm.back,),
            AnsiBack.LIGHTBLACK_EX: (winterm.back, WinColor.BLACK, True),
            AnsiBack.LIGHTRED_EX: (winterm.back, WinColor.RED, True),
            AnsiBack.LIGHTGREEN_EX: (winterm.back, WinColor.GREEN, True),
            AnsiBack.LIGHTYELLOW_EX: (winterm.back, WinColor.YELLOW, True),
            AnsiBack.LIGHTBLUE_EX: (winterm.back, WinColor.BLUE, True),
            AnsiBack.LIGHTMAGENTA_EX: (winterm.back, WinColor.MAGENTA, True),
            AnsiBack.LIGHTCYAN_EX: (winterm.back, WinColor.CYAN, True),
            AnsiBack.LIGHTWHITE_EX: (winterm.back, WinColor.GREY, True),
        }

    def write(self, text):
        if self.strip or self.convert:
            self.write_and_convert(text)
        else:
            self.wrapped.write(text)
            self.wrapped.flush()

        if self.autoreset:
            self.reset_all()

    def reset_all(self):
        if self.convert:
            self.call_win32("m", (0,))
        elif not self.strip and not self.stream.closed:
            self.wrapped.write(Style.RESET_ALL)

    def write_and_convert(self, text):
        position = 0
        text = self.convert_osc(text)

        for match in self.ANSI_CSI_RE.finditer(text):
            start, end = match.span()
            self.write_plain_text(text, position, start)
            self.convert_ansi(*match.groups())
            position = end

        self.write_plain_text(text, position, len(text))

    def write_plain_text(self, text, start, end):
        if start < end:
            self.wrapped.write(text[start:end])
            self.wrapped.flush()

    def convert_ansi(self, paramstring, command):
        if self.convert:
            self.call_win32(command, self.extract_params(command, paramstring))

    def extract_params(self, command, paramstring):
        if command in "Hf":
            params = tuple(
                int(value) if value else 1
                for value in paramstring.split(";")
            )
            while len(params) < 2:
                params += (1,)
            return params

        params = tuple(int(value) for value in paramstring.split(";") if value)
        return params if params else (0,)

    def call_win32(self, command, params):
        if command == "m":
            for param in params:
                call = self.win32_calls.get(param)
                if call is not None:
                    function, *arguments = call
                    function(*arguments, on_stderr=self.on_stderr)
        elif command == "J":
            winterm.erase_screen(params[0], on_stderr=self.on_stderr)
        elif command == "K":
            winterm.erase_line(params[0], on_stderr=self.on_stderr)
        elif command in "Hf":
            winterm.set_cursor_position(params, on_stderr=self.on_stderr)
        elif command in "ABCD":
            amount = params[0]
            x, y = winterm.get_cursor_position()

            if command == "A":
                y -= amount
            elif command == "B":
                y += amount
            elif command == "C":
                x += amount
            else:
                x -= amount

            winterm.set_cursor_position((x, y), on_stderr=self.on_stderr)

    def convert_osc(self, text):
        for match in self.ANSI_OSC_RE.finditer(text):
            start, end = match.span()
            text = text[:start] + text[end:]

            params, terminator = match.groups()
            if terminator == BEL and params.count(";") == 1:
                operation, title = params.split(";")
                if operation in "02":
                    winterm.set_title(title)

        return text