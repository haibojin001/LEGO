import logging
from typing import Any, Optional

_logger = logging.getLogger("websocket")

try:
    from logging import NullHandler
except ImportError:

    class NullHandler(logging.Handler):  # type: ignore[no-redef]
        def emit(self, record: Any) -> None:
            return None


_logger.addHandler(NullHandler())

_traceEnabled = False
_trace_handler: Optional[logging.Handler] = None
_default_level = _logger.getEffectiveLevel()

__all__ = [
    "enableTrace",
    "dump",
    "error",
    "warning",
    "debug",
    "trace",
    "isEnabledForError",
    "isEnabledForDebug",
    "isEnabledForTrace",
]


def enableTrace(
    traceable: bool,
    handler: Optional[logging.Handler] = None,
    level: str = "DEBUG",
) -> None:
    global _traceEnabled, _trace_handler

    if not traceable:
        _traceEnabled = False
        if _trace_handler is not None and _trace_handler in _logger.handlers:
            _logger.removeHandler(_trace_handler)
        _trace_handler = None
        _logger.setLevel(_default_level)
        return

    selected_handler = handler if handler is not None else logging.StreamHandler()

    if _trace_handler is not None and _trace_handler in _logger.handlers:
        _logger.removeHandler(_trace_handler)

    _trace_handler = selected_handler

    if selected_handler not in _logger.handlers:
        _logger.addHandler(selected_handler)

    _logger.setLevel(getattr(logging, level))
    _traceEnabled = True


def dump(title: str, message: str) -> None:
    if _traceEnabled:
        _logger.debug(f"--- {title} ---")
        _logger.debug(message)
        _logger.debug("-----------------------")


def error(msg: str) -> None:
    _logger.error(msg)


def warning(msg: str) -> None:
    _logger.warning(msg)


def debug(msg: str) -> None:
    _logger.debug(msg)


def info(msg: str) -> None:
    _logger.info(msg)


def trace(msg: str) -> None:
    if _traceEnabled:
        _logger.debug(msg)


def isEnabledForError() -> bool:
    return _logger.isEnabledFor(logging.ERROR)


def isEnabledForDebug() -> bool:
    return _logger.isEnabledFor(logging.DEBUG)


def isEnabledForTrace() -> bool:
    return _traceEnabled