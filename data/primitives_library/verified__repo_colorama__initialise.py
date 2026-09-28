import atexit
import contextlib
import sys

from .ansitowin32 import AnsiToWin32


def _wipe_internal_state_for_tests():
    global orig_stdout, orig_stderr
    global wrapped_stdout, wrapped_stderr
    global atexit_done, fixed_windows_console

    orig_stdout = None
    orig_stderr = None
    wrapped_stdout = None
    wrapped_stderr = None
    atexit_done = False
    fixed_windows_console = False

    atexit.unregister(reset_all)


def reset_all():
    if AnsiToWin32 is not None:
        AnsiToWin32(orig_stdout).reset_all()


def init(autoreset=False, convert=None, strip=None, wrap=True):
    if not wrap and any((autoreset, convert, strip)):
        raise ValueError("wrap=False conflicts with any other arg=True")

    global orig_stdout, orig_stderr
    global wrapped_stdout, wrapped_stderr
    global atexit_done

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    if orig_stdout is None:
        wrapped_stdout = None
    else:
        wrapped_stdout = wrap_stream(
            orig_stdout, convert, strip, autoreset, wrap
        )
        sys.stdout = wrapped_stdout

    if orig_stderr is None:
        wrapped_stderr = None
    else:
        wrapped_stderr = wrap_stream(
            orig_stderr, convert, strip, autoreset, wrap
        )
        sys.stderr = wrapped_stderr

    if not atexit_done:
        atexit.register(reset_all)
        atexit_done = True


def deinit():
    if orig_stdout is not None:
        sys.stdout = orig_stdout
    if orig_stderr is not None:
        sys.stderr = orig_stderr


def just_fix_windows_console():
    global fixed_windows_console

    if sys.platform != "win32":
        return
    if fixed_windows_console:
        return
    if wrapped_stdout is not None or wrapped_stderr is not None:
        return

    stdout_wrapper = AnsiToWin32(
        sys.stdout, convert=None, strip=None, autoreset=False
    )
    if stdout_wrapper.convert:
        sys.stdout = stdout_wrapper

    stderr_wrapper = AnsiToWin32(
        sys.stderr, convert=None, strip=None, autoreset=False
    )
    if stderr_wrapper.convert:
        sys.stderr = stderr_wrapper

    fixed_windows_console = True


@contextlib.contextmanager
def colorama_text(*args, **kwargs):
    init(*args, **kwargs)
    try:
        yield
    finally:
        deinit()


def reinit():
    if wrapped_stdout is not None:
        sys.stdout = wrapped_stdout
    if wrapped_stderr is not None:
        sys.stderr = wrapped_stderr


def wrap_stream(stream, convert, strip, autoreset, wrap):
    if wrap:
        converter = AnsiToWin32(
            stream, convert=convert, strip=strip, autoreset=autoreset
        )
        if converter.should_wrap():
            stream = converter.stream
    return stream


_wipe_internal_state_for_tests()