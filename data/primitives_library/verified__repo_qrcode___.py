def _gf_multiply(a, b):
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= 0x11D
    return result


rsPoly_LUT = {}

for _degree in (7, 10, 13, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30):
    _polynomial = [1]
    _power = 1
    for _ in range(_degree):
        _next = [0] * (len(_polynomial) + 1)
        for _index, _coefficient in enumerate(_polynomial):
            _next[_index] ^= _coefficient
            _next[_index + 1] ^= _gf_multiply(_coefficient, _power)
        _polynomial = _next
        _power = _gf_multiply(_power, 2)
    rsPoly_LUT[_degree] = _polynomial

del _gf_multiply
del _degree
del _polynomial
del _power
del _
del _next
del _index
del _coefficient