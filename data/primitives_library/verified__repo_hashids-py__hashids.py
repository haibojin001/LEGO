import warnings
from functools import wraps
from math import ceil

__version__ = '1.3.1'

RATIO_SEPARATORS = 3.5
RATIO_GUARDS = 12

try:
    StrType = basestring
except NameError:
    StrType = str


def _is_str(candidate):
    return isinstance(candidate, StrType)


def _is_uint(number):
    try:
        return number == int(number) and number >= 0
    except ValueError:
        return False


def _split(string, splitters):
    current = ''
    for char in string:
        if char in splitters:
            yield current
            current = ''
        else:
            current += char
    yield current


def _hash(number, alphabet):
    base = len(alphabet)
    output = ''

    while True:
        output = alphabet[number % base] + output
        number //= base
        if not number:
            return output


def _unhash(hashed, alphabet):
    base = len(alphabet)
    result = 0

    for char in hashed:
        result = result * base + alphabet.index(char)

    return result


def _reorder(string, salt):
    if not salt:
        return string

    chars = list(string)
    salt_length = len(salt)
    salt_index = 0
    total = 0

    for index in range(len(chars) - 1, 0, -1):
        value = ord(salt[salt_index])
        total += value
        other_index = (value + salt_index + total) % index
        chars[index], chars[other_index] = chars[other_index], chars[index]
        salt_index = (salt_index + 1) % salt_length

    return ''.join(chars)


def _index_from_ratio(dividend, divisor):
    return int(ceil(float(dividend) / divisor))


def _ensure_length(encoded, min_length, alphabet, guards, values_hash):
    guard_count = len(guards)
    first_guard = (values_hash + ord(encoded[0])) % guard_count
    encoded = guards[first_guard] + encoded

    if len(encoded) < min_length:
        second_guard = (values_hash + ord(encoded[2])) % guard_count
        encoded += guards[second_guard]

    middle = len(alphabet) // 2

    while len(encoded) < min_length:
        alphabet = _reorder(alphabet, alphabet)
        encoded = alphabet[middle:] + encoded + alphabet[:middle]

        overflow = len(encoded) - min_length
        if overflow > 0:
            start = overflow // 2
            encoded = encoded[start:start + min_length]

    return encoded


def _encode(values, salt, min_length, alphabet, separators, guards):
    alphabet_length = len(alphabet)
    separator_length = len(separators)
    values_hash = sum(value % (position + 100)
                      for position, value in enumerate(values))

    lottery = alphabet[values_hash % alphabet_length]
    encoded = lottery

    for position, value in enumerate(values):
        shuffle_salt = (lottery + salt + alphabet)[:alphabet_length]
        alphabet = _reorder(alphabet, shuffle_salt)

        portion = _hash(value, alphabet)
        encoded += portion

        value %= ord(portion[0]) + position
        encoded += separators[value % separator_length]

    encoded = encoded[:-1]

    if len(encoded) >= min_length:
        return encoded

    return _ensure_length(encoded, min_length, alphabet, guards, values_hash)


def _decode(hashid, salt, alphabet, separators, guards):
    guarded_parts = tuple(_split(hashid, guards))

    if 2 <= len(guarded_parts) <= 3:
        hashid = guarded_parts[1]
    else:
        hashid = guarded_parts[0]

    if not hashid:
        return

    lottery = hashid[0]
    hashid = hashid[1:]

    for portion in _split(hashid, separators):
        shuffle_salt = (lottery + salt + alphabet)[:len(alphabet)]
        alphabet = _reorder(alphabet, shuffle_salt)
        yield _unhash(portion, alphabet)


def _deprecated(func, name):
    @wraps(func)
    def warned(*args, **kwargs):
        warnings.warn(
            'The %s method is deprecated and will be removed in v2.*.*' % name,
            DeprecationWarning
        )
        return func(*args, **kwargs)

    return warned


class Hashids(object):
    ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'

    def __init__(self, salt='', min_length=0, alphabet=ALPHABET):
        self._min_length = max(int(min_length), 0)
        self._salt = salt

        separators = ''.join(
            char for char in 'cfhistuCFHISTU' if char in alphabet
        )

        alphabet = ''.join(
            char for index, char in enumerate(alphabet)
            if alphabet.index(char) == index and char not in separators
        )

        alphabet_length = len(alphabet)
        separator_length = len(separators)

        if alphabet_length + separator_length < 16:
            raise ValueError('Alphabet must contain at least 16 unique characters.')

        separators = _reorder(separators, salt)

        required_separators = _index_from_ratio(
            alphabet_length,
            RATIO_SEPARATORS
        )
        missing_separators = required_separators - separator_length

        if missing_separators > 0:
            separators += alphabet[:missing_separators]
            alphabet = alphabet[missing_separators:]
            alphabet_length = len(alphabet)

        alphabet = _reorder(alphabet, salt)
        guard_count = _index_from_ratio(alphabet_length, RATIO_GUARDS)

        if alphabet_length < 3:
            guards = separators[:guard_count]
            separators = separators[guard_count:]
        else:
            guards = alphabet[:guard_count]
            alphabet = alphabet[guard_count:]

        self._alphabet = alphabet
        self._guards = guards
        self._separators = separators

        self.decrypt = _deprecated(self.decode, 'decrypt')
        self.encrypt = _deprecated(self.encode, 'encrypt')

    def encode(self, *values):
        if not (values and all(_is_uint(value) for value in values)):
            return ''

        return _encode(
            values,
            self._salt,
            self._min_length,
            self._alphabet,
            self._separators,
            self._guards
        )

    def decode(self, hashid):
        if not hashid or not _is_str(hashid):
            return ()

        try:
            values = tuple(
                _decode(
                    hashid,
                    self._salt,
                    self._alphabet,
                    self._separators,
                    self._guards
                )
            )
            return values if hashid == self.encode(*values) else ()
        except ValueError:
            return ()

    def encode_hex(self, hex_str):
        values = (
            int('1' + hex_str[index:index + 12], 16)
            for index in range(0, len(hex_str), 12)
        )

        try:
            return self.encode(*values)
        except ValueError:
            return ''

    def decode_hex(self, hashid):
        return ''.join(('%x' % value)[1:] for value in self.decode(hashid))