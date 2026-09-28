from typing import Any, Optional, Union

__all__ = ["NoLock", "validate_utf8", "extract_err_message", "extract_error_code"]


class NoLock:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self, exc_type: Any, exc_value: Any, traceback: Any
    ) -> None:
        return None


try:
    from wsaccel.utf8validator import Utf8Validator
except ImportError:
    _UTF8_ACCEPT = 0
    _UTF8_REJECT = 12

    def _byte_class(value: int) -> int:
        if 0 <= value <= 0x7F:
            return 0
        if 0x80 <= value <= 0x8F:
            return 1
        if 0x90 <= value <= 0x9F:
            return 9
        if 0xA0 <= value <= 0xBF:
            return 7
        if value in (0xC0, 0xC1) or 0xF5 <= value <= 0xFF:
            return 8
        if 0xC2 <= value <= 0xDF:
            return 2
        if value == 0xE0:
            return 10
        if 0xE1 <= value <= 0xEC or 0xEE <= value <= 0xEF:
            return 3
        if value == 0xED:
            return 4
        if value == 0xF0:
            return 11
        if 0xF1 <= value <= 0xF3:
            return 6
        if value == 0xF4:
            return 5
        raise IndexError("list index out of range")

    def _decode(state: int, codep: int, ch: int) -> tuple:
        tp = _byte_class(ch)

        if state == _UTF8_ACCEPT:
            codep = (0xFF >> tp) & ch
            transitions = {
                0: _UTF8_ACCEPT,
                2: 24,
                3: 36,
                4: 60,
                5: 96,
                6: 84,
                10: 48,
                11: 72,
            }
            return transitions.get(tp, _UTF8_REJECT), codep

        codep = (ch & 0x3F) | (codep << 6)

        if state == 24:
            next_state = _UTF8_ACCEPT if tp in (1, 7, 9) else _UTF8_REJECT
        elif state == 36:
            next_state = 24 if tp in (1, 7, 9) else _UTF8_REJECT
        elif state == 48:
            next_state = 24 if tp == 7 else _UTF8_REJECT
        elif state == 60:
            next_state = 24 if tp in (1, 9) else _UTF8_REJECT
        elif state == 72:
            next_state = 36 if tp in (7, 9) else _UTF8_REJECT
        elif state == 84:
            next_state = 36 if tp in (1, 7, 9) else _UTF8_REJECT
        elif state == 96:
            next_state = 36 if tp == 1 else _UTF8_REJECT
        else:
            next_state = _UTF8_REJECT

        return next_state, codep

    def _validate_utf8(utfbytes: Union[str, bytes]) -> bool:
        state = _UTF8_ACCEPT
        codep = 0

        for value in utfbytes:
            state, codep = _decode(state, codep, int(value))
            if state == _UTF8_REJECT:
                return False

        return True
else:
    def _validate_utf8(utfbytes: Union[str, bytes]) -> bool:
        return Utf8Validator().validate(utfbytes)[0]


def validate_utf8(utfbytes: Union[str, bytes]) -> bool:
    if isinstance(utfbytes, str):
        utfbytes = utfbytes.encode("utf-8", "surrogatepass")
    return _validate_utf8(utfbytes)


def extract_err_message(exception: Exception) -> Optional[str]:
    if exception.args:
        return str(exception.args[0])
    return None


def extract_error_code(exception: Exception) -> Optional[int]:
    if exception.args and len(exception.args) > 1:
        return exception.args[0] if isinstance(exception.args[0], int) else None
    return None