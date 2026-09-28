from math import ceil as _ceil, floor as _floor
import bisect
import binascii


def clamp(x, lower=float('-inf'), upper=float('inf')):
    """Constrain a value to an inclusive lower and upper bound."""
    if upper < lower:
        raise ValueError(
            'expected upper bound (%r) >= lower bound (%r)' % (upper, lower)
        )
    return min(max(x, lower), upper)


def ceil(x, options=None):
    """Return the mathematical ceiling, or the next value from options."""
    if options is None:
        return _ceil(x)

    options = sorted(options)
    position = bisect.bisect_left(options, x)
    if position == len(options):
        raise ValueError(
            'no ceil options greater than or equal to: %r' % x
        )
    return options[position]


def floor(x, options=None):
    """Return the mathematical floor, or the prior value from options."""
    if options is None:
        return _floor(x)

    options = sorted(options)
    position = bisect.bisect_right(options, x)
    if not position:
        raise ValueError(
            'no floor options less than or equal to: %r' % x
        )
    return options[position - 1]


class Bits:
    """An immutable, fixed-width representation of a nonnegative integer."""

    __slots__ = ('val', 'len')

    def __init__(self, val=0, len_=None):
        if type(val) is not int:
            if type(val) is list:
                val = ''.join('1' if bit else '0' for bit in val)

            if type(val) is bytes:
                val = val.decode('ascii')

            if type(val) is str:
                if len_ is None:
                    len_ = len(val)
                    if val.startswith('0x'):
                        len_ = (len_ - 2) * 4

                if val.startswith('0x'):
                    val = int(val, 16)
                elif val:
                    val = int(val, 2)
                else:
                    val = 0

            if type(val) is not int:
                raise TypeError(
                    f'initialized with bad type: {type(val).__name__}'
                )

        if val < 0:
            raise ValueError('Bits cannot represent negative values')

        if len_ is None:
            len_ = len(f'{val:b}')

        if val >= 2 ** len_:
            raise ValueError(
                f'value {val} cannot be represented with {len_} bits'
            )

        self.val = val
        self.len = len_

    def __getitem__(self, k):
        if type(k) is slice:
            return Bits(self.as_bin()[k])

        if type(k) is int:
            if k >= self.len:
                raise IndexError(k)
            return bool(self.val & (1 << (self.len - k - 1)))

        raise TypeError(type(k))

    def __len__(self):
        return self.len

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.val == other.val and self.len == other.len

    def __or__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return Bits(self.val | other.val, max(self.len, other.len))

    def __and__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return Bits(self.val & other.val, max(self.len, other.len))

    def __lshift__(self, other):
        return Bits(self.val << other, self.len + other)

    def __rshift__(self, other):
        return Bits(self.val >> other, self.len - other)

    def __hash__(self):
        return hash(self.val)

    def as_list(self):
        return [char == '1' for char in self.as_bin()]

    def as_bin(self):
        if not self.len:
            return ''
        return f'{{0:0{self.len}b}}'.format(self.val)

    def as_hex(self):
        if not self.len:
            return ''
        width = 2 * (self.len // 8 + ((self.len % 8) != 0))
        return ('%0' + str(width) + 'X') % self.val

    def as_int(self):
        return self.val

    def as_bytes(self):
        return binascii.unhexlify(self.as_hex())

    @classmethod
    def from_list(cls, list_):
        return cls(list_)

    @classmethod
    def from_bin(cls, bin):
        return cls(bin)

    @classmethod
    def from_hex(cls, hex):
        if isinstance(hex, bytes):
            hex = hex.decode('ascii')
        if hex == '':
            return cls('')
        if not hex.startswith('0x'):
            hex = '0x' + hex
        return cls(hex)

    @classmethod
    def from_int(cls, int_, len_=None):
        return cls(int_, len_)

    @classmethod
    def from_bytes(cls, bytes_):
        return cls.from_hex(binascii.hexlify(bytes_))

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.as_bin()}')"