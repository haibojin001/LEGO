from functools import wraps


def bin2float(min_, max_, nbits):
    """Build a decorator that decodes fixed-width bit fields to floats."""
    def decorator(function):
        @wraps(function)
        def evaluate(individual, *args, **kargs):
            count = len(individual) // nbits
            values = []
            scale = (2 ** nbits) - 1
            for offset in range(count):
                start = offset * nbits
                bits = individual[start:start + nbits]
                integer = int("".join(str(bit) for bit in bits), 2)
                values.append(min_ + (integer / scale) * (max_ - min_))
            return function(values, *args, **kargs)
        return evaluate
    return decorator


def trap(individual):
    ones = sum(individual)
    length = len(individual)
    if ones == length:
        return length
    return length - ones - 1


def inv_trap(individual):
    ones = sum(individual)
    length = len(individual)
    if ones == 0:
        return length
    return ones - 1


def chuang_f1(individual):
    """Evaluate the first Chuang-Hsu binary deceptive landscape."""
    score = 0
    stop = len(individual) - 1

    if individual[-1] == 0:
        for start in range(0, stop, 4):
            score += inv_trap(individual[start:start + 4])
    else:
        for start in range(0, stop, 4):
            score += trap(individual[start:start + 4])

    return score,


def chuang_f2(individual):
    """Evaluate the second Chuang-Hsu binary deceptive landscape."""
    score = 0
    selector = individual[-2], individual[-1]
    stop = len(individual) - 2

    if selector == (0, 0):
        first, second = inv_trap, inv_trap
    elif selector == (0, 1):
        first, second = inv_trap, trap
    elif selector == (1, 0):
        first, second = trap, inv_trap
    else:
        first, second = trap, trap

    for start in range(0, stop, 8):
        score += first(individual[start:start + 4])
        score += second(individual[start + 4:start + 8])

    return score,


def chuang_f3(individual):
    """Evaluate the third Chuang-Hsu binary deceptive landscape."""
    score = 0

    if individual[-1] == 0:
        for start in range(0, len(individual) - 1, 4):
            score += inv_trap(individual[start:start + 4])
    else:
        for start in range(2, len(individual) - 3, 4):
            score += inv_trap(individual[start:start + 4])
        score += trap(individual[-2:] + individual[:2])

    return score,


def royal_road1(individual, order):
    """Compute the first Royal Road fitness value."""
    blocks = len(individual) // order
    target = int(2 ** order - 1)
    score = 0

    for block in range(blocks):
        start = block * order
        segment = individual[start:start + order]
        value = int("".join(str(bit) for bit in segment), 2)
        score += int(order) * int(value / target)

    return score,


def royal_road2(individual, order):
    """Compute the hierarchical second Royal Road fitness value."""
    score = 0
    current_order = order

    while current_order < order ** 2:
        score += royal_road1(individual, current_order)[0]
        current_order *= 2

    return score,