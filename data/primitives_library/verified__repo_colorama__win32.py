STDOUT = -11
STDERR = -12

ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

try:
    import ctypes
    from ctypes import LibraryLoader
    windll = LibraryLoader(ctypes.WinDLL)
    from ctypes import wintypes
except (AttributeError, ImportError):
    windll = None
    SetConsoleTextAttribute = lambda *_: None
    winapi_test = lambda *_: None
else:
    from ctypes import POINTER, Structure, byref, c_char

    COORD = wintypes._COORD

    class CONSOLE_SCREEN_BUFFER_INFO(Structure):
        _fields_ = [
            ("dwSize", COORD),
            ("dwCursorPosition", COORD),
            ("wAttributes", wintypes.WORD),
            ("srWindow", wintypes.SMALL_RECT),
            ("dwMaximumWindowSize", COORD),
        ]

        def __str__(self):
            return "(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d)" % (
                self.dwSize.Y,
                self.dwSize.X,
                self.dwCursorPosition.Y,
                self.dwCursorPosition.X,
                self.wAttributes,
                self.srWindow.Top,
                self.srWindow.Left,
                self.srWindow.Bottom,
                self.srWindow.Right,
                self.dwMaximumWindowSize.Y,
                self.dwMaximumWindowSize.X,
            )

    _GetStdHandle = windll.kernel32.GetStdHandle
    _GetStdHandle.argtypes = [wintypes.DWORD]
    _GetStdHandle.restype = wintypes.HANDLE

    _GetConsoleScreenBufferInfo = windll.kernel32.GetConsoleScreenBufferInfo
    _GetConsoleScreenBufferInfo.argtypes = [
        wintypes.HANDLE,
        POINTER(CONSOLE_SCREEN_BUFFER_INFO),
    ]
    _GetConsoleScreenBufferInfo.restype = wintypes.BOOL

    _SetConsoleTextAttribute = windll.kernel32.SetConsoleTextAttribute
    _SetConsoleTextAttribute.argtypes = [
        wintypes.HANDLE,
        wintypes.WORD,
    ]
    _SetConsoleTextAttribute.restype = wintypes.BOOL

    _SetConsoleCursorPosition = windll.kernel32.SetConsoleCursorPosition
    _SetConsoleCursorPosition.argtypes = [
        wintypes.HANDLE,
        COORD,
    ]
    _SetConsoleCursorPosition.restype = wintypes.BOOL

    _FillConsoleOutputCharacterA = windll.kernel32.FillConsoleOutputCharacterA
    _FillConsoleOutputCharacterA.argtypes = [
        wintypes.HANDLE,
        c_char,
        wintypes.DWORD,
        COORD,
        POINTER(wintypes.DWORD),
    ]
    _FillConsoleOutputCharacterA.restype = wintypes.BOOL

    _FillConsoleOutputAttribute = windll.kernel32.FillConsoleOutputAttribute
    _FillConsoleOutputAttribute.argtypes = [
        wintypes.HANDLE,
        wintypes.WORD,
        wintypes.DWORD,
        COORD,
        POINTER(wintypes.DWORD),
    ]
    _FillConsoleOutputAttribute.restype = wintypes.BOOL

    _SetConsoleTitleW = windll.kernel32.SetConsoleTitleW
    _SetConsoleTitleW.argtypes = [wintypes.LPCWSTR]
    _SetConsoleTitleW.restype = wintypes.BOOL

    _GetConsoleMode = windll.kernel32.GetConsoleMode
    _GetConsoleMode.argtypes = [
        wintypes.HANDLE,
        POINTER(wintypes.DWORD),
    ]
    _GetConsoleMode.restype = wintypes.BOOL

    _SetConsoleMode = windll.kernel32.SetConsoleMode
    _SetConsoleMode.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
    ]
    _SetConsoleMode.restype = wintypes.BOOL

    def _winapi_test(handle):
        info = CONSOLE_SCREEN_BUFFER_INFO()
        return bool(_GetConsoleScreenBufferInfo(handle, byref(info)))

    def winapi_test():
        handles = (_GetStdHandle(STDOUT), _GetStdHandle(STDERR))
        return any(_winapi_test(handle) for handle in handles)

    def GetConsoleScreenBufferInfo(stream_id=STDOUT):
        handle = _GetStdHandle(stream_id)
        info = CONSOLE_SCREEN_BUFFER_INFO()
        _GetConsoleScreenBufferInfo(handle, byref(info))
        return info

    def SetConsoleTextAttribute(stream_id, attrs):
        return _SetConsoleTextAttribute(_GetStdHandle(stream_id), attrs)

    def SetConsoleCursorPosition(stream_id, position, adjust=True):
        requested = COORD(*position)
        if requested.Y <= 0 or requested.X <= 0:
            return

        target = COORD(requested.Y - 1, requested.X - 1)
        if adjust:
            window = GetConsoleScreenBufferInfo(STDOUT).srWindow
            target.Y += window.Top
            target.X += window.Left

        return _SetConsoleCursorPosition(_GetStdHandle(stream_id), target)

    def FillConsoleOutputCharacter(stream_id, char, length, start):
        written = wintypes.DWORD(0)
        _FillConsoleOutputCharacterA(
            _GetStdHandle(stream_id),
            c_char(char.encode()),
            wintypes.DWORD(length),
            start,
            byref(written),
        )
        return written.value

    def FillConsoleOutputAttribute(stream_id, attr, length, start):
        written = wintypes.DWORD(0)
        return _FillConsoleOutputAttribute(
            _GetStdHandle(stream_id),
            wintypes.WORD(attr),
            wintypes.DWORD(length),
            start,
            byref(written),
        )

    def SetConsoleTitle(title):
        return _SetConsoleTitleW(title)

    def GetConsoleMode(handle):
        mode = wintypes.DWORD()
        if not _GetConsoleMode(handle, byref(mode)):
            raise ctypes.WinError()
        return mode.value

    def SetConsoleMode(handle, mode):
        if not _SetConsoleMode(handle, mode):
            raise ctypes.WinError()