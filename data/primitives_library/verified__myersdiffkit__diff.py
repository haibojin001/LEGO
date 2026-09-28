"""Myers diff and unified-diff formatting utilities."""

__all__ = ["diff", "apply_diff", "edit_distance", "unified_diff"]


def _snake(a_lines, b_lines, x, y):
    n = len(a_lines)
    m = len(b_lines)
    while x < n and y < m and a_lines[x] == b_lines[y]:
        x += 1
        y += 1
    return x, y


def _choose_step(previous_v, k, n, m):
    insert_candidate = None
    delete_candidate = None

    insert_prev_k = k + 1
    if insert_prev_k in previous_v:
        prev_x = previous_v[insert_prev_k]
        prev_y = prev_x - insert_prev_k
        if 0 <= prev_x <= n and 0 <= prev_y < m:
            insert_candidate = ("insert", insert_prev_k, prev_x, prev_y, prev_x)

    delete_prev_k = k - 1
    if delete_prev_k in previous_v:
        prev_x = previous_v[delete_prev_k]
        prev_y = prev_x - delete_prev_k
        if 0 <= prev_x < n and 0 <= prev_y <= m:
            delete_candidate = ("delete", delete_prev_k, prev_x, prev_y, prev_x + 1)

    if insert_candidate is None:
        return delete_candidate
    if delete_candidate is None:
        return insert_candidate

    if delete_candidate[4] >= insert_candidate[4]:
        return delete_candidate
    return insert_candidate


def _myers_distance(a_lines, b_lines):
    n = len(a_lines)
    m = len(b_lines)
    max_d = n + m

    x, y = _snake(a_lines, b_lines, 0, 0)
    if x == n and y == m:
        return 0

    v = {0: x}

    for d in range(1, max_d + 1):
        new_v = {}

        for k in range(-d, d + 1, 2):
            choice = _choose_step(v, k, n, m)
            if choice is None:
                continue

            op, _prev_k, prev_x, prev_y, _x_after_edit = choice
            if op == "delete":
                x = prev_x + 1
                y = prev_y
            else:
                x = prev_x
                y = prev_y + 1

            x, y = _snake(a_lines, b_lines, x, y)
            new_v[k] = x

            if x == n and y == m:
                return d

        v = new_v

    return max_d


def _myers_trace(a_lines, b_lines):
    n = len(a_lines)
    m = len(b_lines)
    max_d = n + m

    x, y = _snake(a_lines, b_lines, 0, 0)
    v = {0: x}
    trace = [v.copy()]

    if x == n and y == m:
        return 0, trace

    for d in range(1, max_d + 1):
        new_v = {}

        for k in range(-d, d + 1, 2):
            choice = _choose_step(v, k, n, m)
            if choice is None:
                continue

            op, _prev_k, prev_x, prev_y, _x_after_edit = choice
            if op == "delete":
                x = prev_x + 1
                y = prev_y
            else:
                x = prev_x
                y = prev_y + 1

            x, y = _snake(a_lines, b_lines, x, y)
            new_v[k] = x

            if x == n and y == m:
                trace.append(new_v.copy())
                return d, trace

        trace.append(new_v.copy())
        v = new_v

    return max_d, trace


def _backtrack(a_lines, b_lines, distance, trace):
    n = len(a_lines)
    m = len(b_lines)

    x = n
    y = m
    reversed_ops = []

    for d in range(distance, 0, -1):
        k = x - y
        previous_v = trace[d - 1]
        choice = _choose_step(previous_v, k, n, m)

        if choice is None:
            raise RuntimeError("failed to reconstruct Myers diff path")

        op, _prev_k, prev_x, prev_y, _x_after_edit = choice

        if op == "delete":
            edit_x = prev_x + 1
            edit_y = prev_y
        else:
            edit_x = prev_x
            edit_y = prev_y + 1

        while x > edit_x and y > edit_y:
            x -= 1
            y -= 1
            reversed_ops.append(("equal", a_lines[x]))

        if x != edit_x or y != edit_y:
            raise RuntimeError("invalid Myers diff trace")

        if op == "delete":
            reversed_ops.append(("delete", a_lines[prev_x]))
        else:
            reversed_ops.append(("insert", b_lines[prev_y]))

        x = prev_x
        y = prev_y

    while x > 0 and y > 0:
        x -= 1
        y -= 1
        reversed_ops.append(("equal", a_lines[x]))

    reversed_ops.reverse()
    return reversed_ops


def diff(a_lines: list, b_lines: list) -> list:
    """Return a shortest Myers edit script from a_lines to b_lines.

    Operations are returned as ``("equal", line)``, ``("delete", line)``, and
    ``("insert", line)`` tuples in application order.
    """
    distance, trace = _myers_trace(a_lines, b_lines)
    return _backtrack(a_lines, b_lines, distance, trace)


def apply_diff(a_lines: list, ops: list) -> list:
    """Apply an edit script produced by diff() to a_lines and return b_lines."""
    result = []
    index = 0
    length = len(a_lines)

    for entry in ops:
        try:
            op, line = entry
        except (TypeError, ValueError):
            raise ValueError("each diff operation must be a two-item tuple")

        if op == "equal":
            if index >= length or a_lines[index] != line:
                raise ValueError("equal operation does not match input")
            result.append(line)
            index += 1
        elif op == "delete":
            if index >= length or a_lines[index] != line:
                raise ValueError("delete operation does not match input")
            index += 1
        elif op == "insert":
            result.append(line)
        else:
            raise ValueError("unknown diff operation: {!r}".format(op))

    if index != length:
        raise ValueError("diff operations do not consume all input lines")

    return result


def edit_distance(a_lines: list, b_lines: list) -> int:
    """Return the Myers insert/delete edit distance between two line lists."""
    return _myers_distance(a_lines, b_lines)


def _line_records(ops):
    records = []
    old_index = 0
    new_index = 0

    for op, line in ops:
        if op == "equal":
            records.append((" ", old_index, new_index, line))
            old_index += 1
            new_index += 1
        elif op == "delete":
            records.append(("-", old_index, new_index, line))
            old_index += 1
        elif op == "insert":
            records.append(("+", old_index, new_index, line))
            new_index += 1
        else:
            raise ValueError("unknown diff operation: {!r}".format(op))

    return records


def _group_records(records, context):
    context = int(context)
    if context < 0:
        context = 0

    changes = []
    for index, record in enumerate(records):
        if record[0] != " ":
            changes.append(index)

    if not changes:
        return []

    groups = []
    start = max(changes[0] - context, 0)
    last_change = changes[0]

    for change in changes[1:]:
        if change - last_change <= (2 * context) + 1:
            last_change = change
        else:
            end = min(last_change + context + 1, len(records))
            groups.append((start, end))
            start = max(change - context, 0)
            last_change = change

    end = min(last_change + context + 1, len(records))
    groups.append((start, end))
    return groups


def _format_range_unified(beginning, length):
    if length == 1:
        return str(beginning + 1)

    beginning += 1
    if length == 0:
        beginning -= 1

    return "{},{}".format(beginning, length)


def unified_diff(
    a_lines: list,
    b_lines: list,
    fromfile: str = "",
    tofile: str = "",
    fromfiledate: str = "",
    tofiledate: str = "",
    n: int = 3,
    lineterm: str = "\n",
) -> list:
    """Return a unified diff as a list of strings.

    The content lines are expected to be strings, typically preserving their
    original line terminators.
    """
    ops = diff(a_lines, b_lines)
    records = _line_records(ops)
    groups = _group_records(records, n)

    if not groups:
        return []

    output = []

    from_date = "\t" + fromfiledate if fromfiledate else ""
    to_date = "\t" + tofiledate if tofiledate else ""

    output.append("--- {}{}{}".format(fromfile, from_date, lineterm))
    output.append("+++ {}{}{}".format(tofile, to_date, lineterm))

    for start, end in groups:
        group = records[start:end]

        old_count = 0
        new_count = 0
        for tag, _old_index, _new_index, _line in group:
            if tag == " " or tag == "-":
                old_count += 1
            if tag == " " or tag == "+":
                new_count += 1

        old_begin = group[0][1]
        new_begin = group[0][2]

        old_range = _format_range_unified(old_begin, old_count)
        new_range = _format_range_unified(new_begin, new_count)

        output.append("@@ -{} +{} @@{}".format(old_range, new_range, lineterm))

        for tag, _old_index, _new_index, line in group:
            output.append(tag + line)

    return output