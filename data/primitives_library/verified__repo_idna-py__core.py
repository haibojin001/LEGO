from __future__ import annotations

import bisect
import re
import unicodedata
import warnings
from typing import Literal

from . import idnadata, uts46data
from .intranges import intranges_contain

_virama_combining_class = 9
_alabel_prefix = b"xn--"
_max_input_length = 1024
_max_domain_length = 253

_STATUS_VALID, _STATUS_MAPPED, _STATUS_DEVIATION, _STATUS_IGNORED = b"VMDI"

_unicode_dots_re = re.compile("[\u002e\u3002\uff0e\uff61]")
_std3_disallowed_re = re.compile("[\x00-\x2c\x2f\x3a-\x40A-Z\x5b-\x60\x7b-\x7f]")

_bidi_rtl_first = frozenset({"R", "AL"})
_bidi_rtl_categories = frozenset({"R", "AL", "AN"})
_bidi_rtl_allowed = frozenset(
    {"R", "AL", "AN", "EN", "ES", "CS", "ET", "ON", "BN", "NSM"}
)
_bidi_rtl_valid_ending = frozenset({"R", "AL", "EN", "AN"})
_bidi_rtl_numeric = frozenset({"AN", "EN"})
_bidi_ltr_allowed = frozenset({"L", "EN", "ES", "CS", "ET", "ON", "BN", "NSM"})
_bidi_ltr_valid_ending = frozenset({"L", "EN"})
_bidi_joiner_l_or_d = frozenset({"L", "D"})
_bidi_joiner_r_or_d = frozenset({"R", "D"})


_ErrorCode = Literal[
    "input_too_long",
    "label_too_long",
    "domain_too_long",
    "empty_label",
    "empty_domain",
    "not_nfc",
    "hyphen_3_4",
    "hyphen_start_end",
    "leading_combiner",
    "disallowed_codepoint",
    "contextj",
    "contexto",
    "unknown_codepoint",
    "bidi_rule_1",
    "bidi_rule_2",
    "bidi_rule_3",
    "bidi_rule_4",
    "bidi_rule_5",
    "bidi_rule_6",
    "bidi_unknown_direction",
    "invalid_alabel",
    "non_canonical_alabel",
    "invalid_ascii",
    "invalid_utf8",
    "uts46_disallowed",
    "uts46_std3",
    "unsupported_errors",
]


class IDNAError(UnicodeError):
    code: str | None
    text: str | None
    codepoint: int | None
    position: int | None

    def __init__(
        self,
        *args: object,
        code: _ErrorCode | None = None,
        text: str | None = None,
        codepoint: int | None = None,
        position: int | None = None,
    ) -> None:
        super().__init__(*args)
        self.code = code
        self.text = text
        self.codepoint = codepoint
        self.position = position


class IDNABidiError(IDNAError):
    pass


class InvalidCodepoint(IDNAError):
    pass


class InvalidCodepointContext(IDNAError):
    pass


def _joining_type(cp: int) -> str | None:
    for joining_type, ranges in idnadata.joining_types.items():
        if intranges_contain(cp, ranges):
            return joining_type
    return None


def _combining_class(cp: int) -> int:
    character = chr(cp)
    value = unicodedata.combining(character)
    if value == 0 and not unicodedata.name(character):
        raise ValueError("Unknown character in unicodedata")
    return value


def _is_script(cp: str, script: str) -> bool:
    return intranges_contain(ord(cp), idnadata.scripts[script])


def _punycode(s: str) -> bytes:
    return s.encode("punycode")


def _unot(s: int) -> str:
    return f"U+{s:04X}"


def valid_label_length(label: bytes | str) -> bool:
    return len(label) <= 63


def valid_string_length(domain: bytes | str, trailing_dot: bool) -> bool:
    return len(domain) <= _max_domain_length + trailing_dot


def check_bidi(label: str, check_ltr: bool = False) -> bool:
    if len(label) > _max_input_length:
        raise IDNAError("Label too long", code="input_too_long", text=label)

    bidi_label = False
    for position, character in enumerate(label, 1):
        direction = unicodedata.bidirectional(character)
        if not direction:
            raise IDNABidiError(
                f"Unknown directionality in label {label!r} at position {position}",
                code="bidi_unknown_direction",
                text=label,
                codepoint=ord(character),
                position=position,
            )
        if direction in _bidi_rtl_categories:
            bidi_label = True

    if not bidi_label and not check_ltr:
        return True

    if not label:
        return True

    first_direction = unicodedata.bidirectional(label[0])
    if first_direction in _bidi_rtl_first:
        rtl = True
    elif first_direction == "L":
        rtl = False
    else:
        raise IDNABidiError(
            f"First codepoint in label {label!r} must be directionality L, R or AL",
            code="bidi_rule_1",
            text=label,
            codepoint=ord(label[0]),
            position=1,
        )

    valid_ending = False
    ending_position = 1
    numeric_type: str | None = None

    for position, character in enumerate(label, 1):
        direction = unicodedata.bidirectional(character)

        if rtl:
            if direction not in _bidi_rtl_allowed:
                raise IDNABidiError(
                    f"Invalid direction for codepoint at position {position} in a right-to-left label",
                    code="bidi_rule_2",
                    text=label,
                    codepoint=ord(character),
                    position=position,
                )

            if direction in _bidi_rtl_valid_ending:
                valid_ending = True
                ending_position = position
            elif direction != "NSM":
                valid_ending = False
                ending_position = position

            if direction in _bidi_rtl_numeric:
                if numeric_type is None:
                    numeric_type = direction
                elif numeric_type != direction:
                    raise IDNABidiError(
                        "Can not mix numeral types in a right-to-left label",
                        code="bidi_rule_4",
                        text=label,
                        codepoint=ord(character),
                        position=position,
                    )
        else:
            if direction not in _bidi_ltr_allowed:
                raise IDNABidiError(
                    f"Invalid direction for codepoint at position {position} in a left-to-right label",
                    code="bidi_rule_5",
                    text=label,
                    codepoint=ord(character),
                    position=position,
                )

            if direction in _bidi_ltr_valid_ending:
                valid_ending = True
                ending_position = position
            elif direction != "NSM":
                valid_ending = False
                ending_position = position

    if not valid_ending:
        code = "bidi_rule_3" if rtl else "bidi_rule_6"
        direction_name = "right-to-left" if rtl else "left-to-right"
        raise IDNABidiError(
            f"Label ends with illegal directionality for a {direction_name} label",
            code=code,
            text=label,
            codepoint=ord(label[ending_position - 1]),
            position=ending_position,
        )

    return True


def check_initial_combiner(label: str) -> bool:
    if not label:
        raise IDNAError("Empty Label", code="empty_label", text=label)

    if unicodedata.category(label[0]).startswith("M"):
        raise IDNAError(
            "Label begins with an illegal combining character",
            code="leading_combiner",
            text=label,
            codepoint=ord(label[0]),
            position=1,
        )
    return True


def check_hyphen_ok(label: str) -> bool:
    if label[2:4] == "--":
        raise IDNAError(
            "Label has disallowed hyphens in 3rd and 4th position",
            code="hyphen_3_4",
            text=label,
            codepoint=ord(label[2]),
            position=3,
        )

    if label.startswith("-") or label.endswith("-"):
        position = 1 if label.startswith("-") else len(label)
        raise IDNAError(
            "Label must not start or end with a hyphen",
            code="hyphen_start_end",
            text=label,
            codepoint=ord(label[position - 1]),
            position=position,
        )
    return True


def check_nfc(label: str) -> bool:
    if unicodedata.normalize("NFC", label) != label:
        raise IDNAError(
            "Label must be in Normalization Form C",
            code="not_nfc",
            text=label,
        )
    return True


def valid_contextj(label: str, pos: int) -> bool:
    cp = ord(label[pos])

    if cp == 0x200C:
        if pos > 0:
            try:
                if _combining_class(ord(label[pos - 1])) == _virama_combining_class:
                    return True
            except ValueError:
                return False

        index = pos - 1
        while index >= 0:
            joining_type = _joining_type(ord(label[index]))
            if joining_type != "T":
                break
            index -= 1

        if index < 0 or joining_type not in _bidi_joiner_l_or_d:
            return False

        index = pos + 1
        while index < len(label):
            joining_type = _joining_type(ord(label[index]))
            if joining_type != "T":
                break
            index += 1

        return index < len(label) and joining_type in _bidi_joiner_r_or_d

    if cp == 0x200D:
        if pos == 0:
            return False
        try:
            return _combining_class(ord(label[pos - 1])) == _virama_combining_class
        except ValueError:
            return False

    return False


def valid_contexto(label: str, pos: int) -> bool:
    cp = ord(label[pos])

    if cp == 0x00B7:
        return pos > 0 and pos + 1 < len(label) and label[pos - 1 : pos + 2] == "l·l"

    if cp == 0x0375:
        return pos + 1 < len(label) and _is_script(label[pos + 1], "Greek")

    if cp in (0x05F3, 0x05F4):
        return pos > 0 and _is_script(label[pos - 1], "Hebrew")

    if cp == 0x30FB:
        return any(
            _is_script(character, "Hiragana")
            or _is_script(character, "Katakana")
            or _is_script(character, "Han")
            for character in label
        )

    if 0x0660 <= cp <= 0x0669:
        return not any(0x06F0 <= ord(character) <= 0x06F9 for character in label)

    if 0x06F0 <= cp <= 0x06F9:
        return not any(0x0660 <= ord(character) <= 0x0669 for character in label)

    return False


def check_label(label: str) -> None:
    if not isinstance(label, str):
        raise TypeError("Label must be a string")

    if len(label) > _max_input_length:
        raise IDNAError("Label too long", code="input_too_long", text=label)

    if not label:
        raise IDNAError("Empty Label", code="empty_label", text=label)

    check_hyphen_ok(label)
    check_initial_combiner(label)
    check_nfc(label)

    for position, character in enumerate(label, 1):
        codepoint = ord(character)

        if intranges_contain(codepoint, idnadata.codepoint_classes["PVALID"]):
            continue

        if intranges_contain(codepoint, idnadata.codepoint_classes["CONTEXTJ"]):
            if valid_contextj(label, position - 1):
                continue
            raise InvalidCodepointContext(
                f"Joiner {_unot(codepoint)} not allowed at position {position} in {label!r}",
                code="contextj",
                text=label,
                codepoint=codepoint,
                position=position,
            )

        if intranges_contain(codepoint, idnadata.codepoint_classes["CONTEXTO"]):
            if valid_contexto(label, position - 1):
                continue
            raise InvalidCodepointContext(
                f"Codepoint {_unot(codepoint)} not allowed at position {position} in {label!r}",
                code="contexto",
                text=label,
                codepoint=codepoint,
                position=position,
            )

        try:
            unicodedata.name(character)
        except ValueError:
            code = "unknown_codepoint"
        else:
            code = "disallowed_codepoint"

        raise InvalidCodepoint(
            f"Codepoint {_unot(codepoint)} at position {position} of {label!r} not allowed",
            code=code,
            text=label,
            codepoint=codepoint,
            position=position,
        )

    check_bidi(label)


def uts46_remap(
    domain: str,
    std3_rules: bool = True,
    transitional: bool = False,
) -> str:
    if len(domain) > _max_input_length:
        raise IDNAError("Domain too long", code="input_too_long", text=domain)

    remapped: list[str] = []
    data = uts46data.uts46data

    for position, character in enumerate(domain, 1):
        codepoint = ord(character)
        row_index = bisect.bisect_left(data, (codepoint,)) - 1
        row = data[row_index]
        status = row[1]

        if status == "V":
            remapped.append(character)
        elif status == "M":
            remapped.append(row[2])
        elif status == "D":
            if transitional:
                remapped.append(row[2])
            else:
                remapped.append(character)
        elif status == "I":
            continue
        elif status == "3":
            if std3_rules:
                raise InvalidCodepoint(
                    f"Codepoint {_unot(codepoint)} not allowed at position {position} in {domain!r}",
                    code="uts46_std3",
                    text=domain,
                    codepoint=codepoint,
                    position=position,
                )
            remapped.append(character)
        else:
            raise InvalidCodepoint(
                f"Codepoint {_unot(codepoint)} not allowed at position {position} in {domain!r}",
                code="uts46_disallowed",
                text=domain,
                codepoint=codepoint,
                position=position,
            )

    return unicodedata.normalize("NFC", "".join(remapped))


def alabel(label: str) -> bytes:
    try:
        label_bytes = label.encode("ascii")
    except UnicodeEncodeError:
        label_bytes = b""
    else:
        if label_bytes.lower().startswith(_alabel_prefix):
            raise IDNAError(
                "Label has disallowed hyphens in 3rd and 4th position",
                code="hyphen_3_4",
                text=label,
                codepoint=ord(label[2]) if len(label) > 2 else None,
                position=3 if len(label) > 2 else None,
            )
        check_label(label)
        if not valid_label_length(label_bytes):
            raise IDNAError(
                "Label too long",
                code="label_too_long",
                text=label,
            )
        return label_bytes

    check_label(label)
    encoded = _alabel_prefix + _punycode(label)

    if not valid_label_length(encoded):
        raise IDNAError("Label too long", code="label_too_long", text=label)

    return encoded


def ulabel(label: str | bytes) -> str:
    if isinstance(label, bytes):
        try:
            label_bytes = bytes(label)
            label_text = label_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise IDNAError(
                "Invalid ASCII in A-label",
                code="invalid_ascii",
            ) from exc
    elif isinstance(label, str):
        label_text = label
        try:
            label_bytes = label.encode("ascii")
        except UnicodeEncodeError:
            check_label(label)
            return label
    else:
        raise TypeError("Label must be a string or bytes")

    if not label_bytes.lower().startswith(_alabel_prefix):
        check_label(label_text)
        return label_text

    try:
        decoded = label_bytes[len(_alabel_prefix) :].decode("punycode")
    except UnicodeError as exc:
        raise IDNAError(
            "Invalid A-label",
            code="invalid_alabel",
            text=label_text,
        ) from exc

    if not decoded:
        raise IDNAError(
            "Invalid A-label",
            code="invalid_alabel",
            text=label_text,
        )

    decoded = unicodedata.normalize("NFC", decoded)

    if _punycode(decoded) != label_bytes[len(_alabel_prefix) :]:
        raise IDNAError(
            "Invalid A-label",
            code="non_canonical_alabel",
            text=label_text,
        )

    check_label(decoded)
    return decoded


def encode(
    s: str | bytes,
    strict: bool = False,
    uts46: bool = False,
    std3_rules: bool = False,
    transitional: bool = False,
) -> bytes:
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IDNAError("Invalid UTF-8", code="invalid_utf8") from exc
    elif not isinstance(s, str):
        raise TypeError("domain name must be a string or bytes")

    if uts46:
        s = uts46_remap(s, std3_rules, transitional)
    else:
        s = unicodedata.normalize("NFC", s)

    labels = s.split(".") if strict else _unicode_dots_re.split(s)

    trailing_dot = False
    if labels and labels[-1] == "":
        trailing_dot = True
        labels.pop()

    if not labels:
        raise IDNAError("Empty domain", code="empty_domain", text=s)

    encoded_labels: list[bytes] = []
    for label in labels:
        if not label:
            raise IDNAError("Empty Label", code="empty_label", text=s)
        encoded_labels.append(alabel(label))

    encoded = b".".join(encoded_labels)
    if trailing_dot:
        encoded += b"."

    if not valid_string_length(encoded, trailing_dot):
        raise IDNAError("Domain too long", code="domain_too_long", text=s)

    return encoded


def decode(
    s: str | bytes,
    strict: bool = False,
    uts46: bool = False,
    std3_rules: bool = False,
) -> str:
    if isinstance(s, bytes):
        try:
            s = s.decode("ascii")
        except UnicodeDecodeError as exc:
            raise IDNAError("Invalid ASCII in A-label", code="invalid_ascii") from exc
    elif not isinstance(s, str):
        raise TypeError("domain name must be a string or bytes")

    if uts46:
        s = uts46_remap(s, std3_rules, False)

    labels = s.split(".") if strict else _unicode_dots_re.split(s)

    trailing_dot = False
    if labels and labels[-1] == "":
        trailing_dot = True
        labels.pop()

    if not labels:
        raise IDNAError("Empty domain", code="empty_domain", text=s)

    decoded_labels: list[str] = []
    for label in labels:
        if not label:
            raise IDNAError("Empty Label", code="empty_label", text=s)
        decoded_labels.append(ulabel(label))

    decoded = ".".join(decoded_labels)
    if trailing_dot:
        decoded += "."

    return decoded