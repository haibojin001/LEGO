"""CSV parsing utilities."""

__all__ = ["parse"]


def _validate_delimiter(delimiter: str) -> None:
    if not isinstance(delimiter, str):
        raise TypeError('"delimiter" must be string')
    if len(delimiter) != 1:
        raise TypeError('"delimiter" must be a 1-character string')


def parse(text: str, delimiter: str = ",") -> list:
    """Parse CSV text into rows.

    The parser follows the standard Excel-style CSV rules: double-quoted
    fields may contain delimiters and newlines, and doubled quotes inside a
    quoted field represent one literal quote character.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    _validate_delimiter(delimiter)

    rows = []
    row = []
    field = []

    START_FIELD = 0
    IN_FIELD = 1
    IN_QUOTED_FIELD = 2
    QUOTE_IN_QUOTED_FIELD = 3

    state = START_FIELD
    record_started = False
    i = 0
    n = len(text)

    while i < n:
        char = text[i]

        if state == START_FIELD:
            if char == "\r" or char == "\n":
                if record_started:
                    row.append("")
                    rows.append(row)
                else:
                    rows.append([])

                row = []
                field = []
                record_started = False

                if char == "\r" and i + 1 < n and text[i + 1] == "\n":
                    i += 1

            elif char == '"':
                record_started = True
                state = IN_QUOTED_FIELD

            elif char == delimiter:
                row.append("")
                record_started = True

            else:
                field.append(char)
                record_started = True
                state = IN_FIELD

        elif state == IN_FIELD:
            if char == "\r" or char == "\n":
                row.append("".join(field))
                rows.append(row)

                row = []
                field = []
                record_started = False
                state = START_FIELD

                if char == "\r" and i + 1 < n and text[i + 1] == "\n":
                    i += 1

            elif char == delimiter:
                row.append("".join(field))
                field = []
                record_started = True
                state = START_FIELD

            else:
                field.append(char)

        elif state == IN_QUOTED_FIELD:
            if char == '"':
                state = QUOTE_IN_QUOTED_FIELD
            else:
                field.append(char)

        else:  # QUOTE_IN_QUOTED_FIELD
            if char == '"':
                field.append('"')
                state = IN_QUOTED_FIELD

            elif char == delimiter:
                row.append("".join(field))
                field = []
                record_started = True
                state = START_FIELD

            elif char == "\r" or char == "\n":
                row.append("".join(field))
                rows.append(row)

                row = []
                field = []
                record_started = False
                state = START_FIELD

                if char == "\r" and i + 1 < n and text[i + 1] == "\n":
                    i += 1

            else:
                field.append(char)
                state = IN_FIELD

        i += 1

    if state == IN_FIELD:
        row.append("".join(field))
        rows.append(row)
    elif state == IN_QUOTED_FIELD or state == QUOTE_IN_QUOTED_FIELD:
        row.append("".join(field))
        rows.append(row)
    elif state == START_FIELD and record_started:
        row.append("")
        rows.append(row)

    return rows