BASE = 1 << 32
MASK = BASE - 1


class BigInt:
    def __init__(self, value=0):
        self._limbs = []

        if isinstance(value, BigInt):
            self._limbs = value._limbs[:]
        elif isinstance(value, int):
            if value < 0:
                raise ValueError("BigInt only supports unsigned integers")
            while value:
                self._limbs.append(value & MASK)
                value >>= 32
        elif isinstance(value, str):
            self._parse_decimal_string(value)
        else:
            raise TypeError("BigInt value must be an int or decimal str")

        self._normalize()

    @classmethod
    def _from_limbs(cls, limbs):
        obj = cls.__new__(cls)
        obj._limbs = list(limbs)
        obj._normalize()
        return obj

    @staticmethod
    def _coerce(value):
        if isinstance(value, BigInt):
            return value
        if isinstance(value, (int, str)):
            return BigInt(value)
        return NotImplemented

    def _normalize(self):
        limbs = self._limbs
        while limbs and limbs[-1] == 0:
            limbs.pop()
        return self

    def _parse_decimal_string(self, value):
        if not value:
            raise ValueError("invalid decimal string")

        if value[0] == "+":
            value = value[1:]
            if not value:
                raise ValueError("invalid decimal string")
        elif value[0] == "-":
            raise ValueError("BigInt only supports unsigned integers")

        for ch in value:
            if ch < "0" or ch > "9":
                raise ValueError("invalid decimal string")
            self._mul_small_inplace(10)
            self._add_small_inplace(ord(ch) - 48)

    def _mul_small_inplace(self, multiplier):
        if not self._limbs or multiplier == 1:
            return
        if multiplier == 0:
            self._limbs = []
            return

        carry = 0
        for i in range(len(self._limbs)):
            total = self._limbs[i] * multiplier + carry
            self._limbs[i] = total & MASK
            carry = total >> 32

        while carry:
            self._limbs.append(carry & MASK)
            carry >>= 32

    def _add_small_inplace(self, addend):
        if addend == 0:
            return

        carry = addend
        i = 0
        while carry:
            if i == len(self._limbs):
                self._limbs.append(0)
            total = self._limbs[i] + carry
            self._limbs[i] = total & MASK
            carry = total >> 32
            i += 1

    @staticmethod
    def _compare_limbs(left, right):
        if len(left) != len(right):
            return -1 if len(left) < len(right) else 1

        for i in range(len(left) - 1, -1, -1):
            if left[i] != right[i]:
                return -1 if left[i] < right[i] else 1

        return 0

    @staticmethod
    def _divmod_small_limbs(limbs, divisor):
        if divisor <= 0 or divisor >= BASE:
            raise ValueError("divisor must be in range 1..2^32-1")

        quotient = [0] * len(limbs)
        remainder = 0

        for i in range(len(limbs) - 1, -1, -1):
            current = (remainder << 32) + limbs[i]
            quotient[i] = current // divisor
            remainder = current % divisor

        while quotient and quotient[-1] == 0:
            quotient.pop()

        return quotient, remainder

    def __add__(self, other) -> "BigInt":
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        a = self._limbs
        b = other._limbs
        length = max(len(a), len(b))
        result = []
        carry = 0

        for i in range(length):
            av = a[i] if i < len(a) else 0
            bv = b[i] if i < len(b) else 0
            total = av + bv + carry
            result.append(total & MASK)
            carry = total >> 32

        if carry:
            result.append(carry)

        return BigInt._from_limbs(result)

    def __sub__(self, other) -> "BigInt":
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        if self < other:
            raise ValueError("BigInt subtraction would produce a negative result")

        a = self._limbs
        b = other._limbs
        result = []
        borrow = 0

        for i in range(len(a)):
            av = a[i]
            bv = b[i] if i < len(b) else 0
            total = av - bv - borrow

            if total < 0:
                total += BASE
                borrow = 1
            else:
                borrow = 0

            result.append(total)

        return BigInt._from_limbs(result)

    def __mul__(self, other) -> "BigInt":
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented

        a = self._limbs
        b = other._limbs

        if not a or not b:
            return BigInt._from_limbs([])

        result = [0] * (len(a) + len(b))

        for i, av in enumerate(a):
            if av == 0:
                continue

            carry = 0

            for j, bv in enumerate(b):
                total = result[i + j] + av * bv + carry
                result[i + j] = total & MASK
                carry = total >> 32

            k = i + len(b)
            while carry:
                if k == len(result):
                    result.append(0)
                total = result[k] + carry
                result[k] = total & MASK
                carry = total >> 32
                k += 1

        return BigInt._from_limbs(result)

    def __eq__(self, other) -> bool:
        other = self._coerce(other)
        if other is NotImplemented:
            return False
        return self._limbs == other._limbs

    def __lt__(self, other) -> bool:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self._compare_limbs(self._limbs, other._limbs) < 0

    def __str__(self) -> str:
        if not self._limbs:
            return "0"

        decimal_base = 1000000000
        temp = self._limbs[:]
        chunks = []

        while temp:
            temp, remainder = self._divmod_small_limbs(temp, decimal_base)
            chunks.append(remainder)

        result = str(chunks[-1])
        for i in range(len(chunks) - 2, -1, -1):
            result += f"{chunks[i]:09d}"

        return result