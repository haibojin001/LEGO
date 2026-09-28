_exp = [0] * 512
_log = [0] * 256
_value = 1

for _index in range(255):
    _exp[_index] = _value
    _log[_value] = _index
    _value <<= 1
    if _value & 0x100:
        _value ^= 0x11D

for _index in range(255, 512):
    _exp[_index] = _exp[_index - 255]


def _multiply(left, right):
    if not left or not right:
        return 0
    return _exp[_log[left] + _log[right]]


def _generator(degree):
    polynomial = [1]
    power = 1

    for _ in range(degree):
        result = [0] * (len(polynomial) + 1)

        for position, coefficient in enumerate(polynomial):
            result[position] ^= coefficient
            result[position + 1] ^= _multiply(coefficient, power)

        polynomial = result
        power = _multiply(power, 2)

    return polynomial


rsPoly_LUT = {
    degree: _generator(degree)
    for degree in (7, 10, 13, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30)
}