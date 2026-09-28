"""Edit script construction and application for strings.

Operations produced by :func:`edit_script` are character-level tuples:

* ``("keep", ch)`` keeps one source character.
* ``("sub", old_ch, new_ch)`` substitutes one source character.
* ``("ins", ch)`` inserts one target character.
* ``("del", ch)`` deletes one source character.

All edit costs are unit cost except ``keep``, which costs zero.
"""

__all__ = ["edit_script", "apply_script"]


def edit_script(a: str, b: str) -> list:
    """Return a minimal-cost edit script transforming string *a* into *b*.

    The script uses Levenshtein-style unit costs:

    * keep: 0
    * substitute: 1
    * insert: 1
    * delete: 1

    Ties are resolved deterministically during backtrace in this order:
    keep, substitute, delete, insert.
    """
    n = len(a)
    m = len(b)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        ca = a[i - 1]
        row = dp[i]
        prev = dp[i - 1]

        for j in range(1, m + 1):
            sub_cost = 0 if ca == b[j - 1] else 1

            diagonal = prev[j - 1] + sub_cost
            deletion = prev[j] + 1
            insertion = row[j - 1] + 1

            best = diagonal
            if deletion < best:
                best = deletion
            if insertion < best:
                best = insertion

            row[j] = best

    ops = []
    i = n
    j = m

    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and a[i - 1] == b[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            ops.append(("keep", a[i - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", a[i - 1]))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", b[j - 1]))
            j -= 1
        else:
            raise RuntimeError("failed to backtrace edit script")

    ops.reverse()
    return ops


def apply_script(a: str, ops: list) -> str:
    """Apply an edit script to string *a* and return the resulting string.

    This function accepts the operation shape produced by :func:`edit_script`.
    For convenience, it also accepts two-field substitutions of the form
    ``("sub", new_ch)``, which replace the next source character without
    validating the old character.
    """
    out = []
    pos = 0
    length = len(a)

    for index, op in enumerate(ops):
        try:
            op_len = len(op)
            name = op[0]
        except Exception as exc:
            raise ValueError("operation at index %d is not a valid sequence" % index) from exc

        if name == "keep":
            if op_len != 2:
                raise ValueError("'keep' operation at index %d must have 2 fields" % index)

            text = op[1]
            text_len = len(text)

            if a[pos:pos + text_len] != text:
                raise ValueError("'keep' operation at index %d does not match input" % index)

            out.append(text)
            pos += text_len

        elif name == "del":
            if op_len != 2:
                raise ValueError("'del' operation at index %d must have 2 fields" % index)

            text = op[1]
            text_len = len(text)

            if a[pos:pos + text_len] != text:
                raise ValueError("'del' operation at index %d does not match input" % index)

            pos += text_len

        elif name == "ins":
            if op_len != 2:
                raise ValueError("'ins' operation at index %d must have 2 fields" % index)

            out.append(op[1])

        elif name == "sub":
            if op_len == 3:
                old = op[1]
                new = op[2]
                old_len = len(old)

                if a[pos:pos + old_len] != old:
                    raise ValueError("'sub' operation at index %d does not match input" % index)

                out.append(new)
                pos += old_len

            elif op_len == 2:
                new = op[1]

                if pos >= length:
                    raise ValueError("'sub' operation at index %d has no input to consume" % index)

                out.append(new)
                pos += 1

            else:
                raise ValueError("'sub' operation at index %d must have 2 or 3 fields" % index)

        else:
            raise ValueError("unknown operation at index %d: %r" % (index, name))

    if pos != length:
        raise ValueError("edit script did not consume the entire input")

    return "".join(out)