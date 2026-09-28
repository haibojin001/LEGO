import math
import re

from qrcode import LUT, base, exceptions
from qrcode.base import RSBlock

MODE_NUMBER = 1 << 0
MODE_ALPHA_NUM = 1 << 1
MODE_8BIT_BYTE = 1 << 2
MODE_KANJI = 1 << 3

MODE_SIZE_SMALL = {
    MODE_NUMBER: 10,
    MODE_ALPHA_NUM: 9,
    MODE_8BIT_BYTE: 8,
    MODE_KANJI: 8,
}
MODE_SIZE_MEDIUM = {
    MODE_NUMBER: 12,
    MODE_ALPHA_NUM: 11,
    MODE_8BIT_BYTE: 16,
    MODE_KANJI: 10,
}
MODE_SIZE_LARGE = {
    MODE_NUMBER: 14,
    MODE_ALPHA_NUM: 13,
    MODE_8BIT_BYTE: 16,
    MODE_KANJI: 12,
}

ALPHA_NUM = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
RE_ALPHA_NUM = re.compile(b"^[" + re.escape(ALPHA_NUM) + rb"]*\Z")
RE_NUMBER = re.compile(b"^\d*\Z")

NUMBER_LENGTH = {3: 10, 2: 7, 1: 4}

PATTERN_POSITION_TABLE = [
    [],
    [6, 18],
    [6, 22],
    [6, 26],
    [6, 30],
    [6, 34],
    [6, 22, 38],
    [6, 24, 42],
    [6, 26, 46],
    [6, 28, 50],
    [6, 30, 54],
    [6, 32, 58],
    [6, 34, 62],
    [6, 26, 46, 66],
    [6, 26, 48, 70],
    [6, 26, 50, 74],
    [6, 30, 54, 78],
    [6, 30, 56, 82],
    [6, 30, 58, 86],
    [6, 34, 62, 90],
    [6, 28, 50, 72, 94],
    [6, 26, 50, 74, 98],
    [6, 30, 54, 78, 102],
    [6, 28, 54, 80, 106],
    [6, 32, 58, 84, 110],
    [6, 30, 58, 86, 114],
    [6, 34, 62, 90, 118],
    [6, 26, 50, 74, 98, 122],
    [6, 30, 54, 78, 102, 126],
    [6, 26, 52, 78, 104, 130],
    [6, 30, 56, 82, 108, 134],
    [6, 34, 60, 86, 112, 138],
    [6, 30, 58, 86, 114, 142],
    [6, 34, 62, 90, 118, 146],
    [6, 30, 54, 78, 102, 126, 150],
    [6, 24, 50, 76, 102, 128, 154],
    [6, 28, 54, 80, 106, 132, 158],
    [6, 32, 58, 84, 110, 136, 162],
    [6, 26, 54, 82, 110, 138, 166],
    [6, 30, 58, 86, 114, 142, 170],
]

G15 = (1 << 10) | (1 << 8) | (1 << 5) | (1 << 4) | (1 << 2) | (1 << 1) | 1
G18 = (
    (1 << 12)
    | (1 << 11)
    | (1 << 10)
    | (1 << 9)
    | (1 << 8)
    | (1 << 5)
    | (1 << 2)
    | 1
)
G15_MASK = (1 << 14) | (1 << 12) | (1 << 10) | (1 << 4) | (1 << 1)

PAD0 = 0xEC
PAD1 = 0x11


def _data_count(block):
    return block.data_count


BIT_LIMIT_TABLE = [
    [0]
    + [
        8 * sum(map(_data_count, base.rs_blocks(version, error_correction)))
        for version in range(1, 41)
    ]
    for error_correction in range(4)
]


def BCH_type_info(data):
    remainder = data << 10
    generator_bits = BCH_digit(G15)
    while BCH_digit(remainder) >= generator_bits:
        remainder ^= G15 << (BCH_digit(remainder) - generator_bits)
    return ((data << 10) | remainder) ^ G15_MASK


def BCH_type_number(data):
    remainder = data << 12
    generator_bits = BCH_digit(G18)
    while BCH_digit(remainder) >= generator_bits:
        remainder ^= G18 << (BCH_digit(remainder) - generator_bits)
    return (data << 12) | remainder


def BCH_digit(data):
    result = 0
    while data:
        result += 1
        data >>= 1
    return result


def pattern_position(version):
    return PATTERN_POSITION_TABLE[version - 1]


def mask_func(pattern):
    if pattern == 0:
        return lambda i, j: (i + j) % 2 == 0
    if pattern == 1:
        return lambda i, j: i % 2 == 0
    if pattern == 2:
        return lambda i, j: j % 3 == 0
    if pattern == 3:
        return lambda i, j: (i + j) % 3 == 0
    if pattern == 4:
        return lambda i, j: (math.floor(i / 2) + math.floor(j / 3)) % 2 == 0
    if pattern == 5:
        return lambda i, j: (i * j) % 2 + (i * j) % 3 == 0
    if pattern == 6:
        return lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0
    if pattern == 7:
        return lambda i, j: ((i * j) % 3 + (i + j) % 2) % 2 == 0
    raise TypeError("Bad mask pattern: " + pattern)


def mode_sizes_for_version(version):
    if version < 10:
        return MODE_SIZE_SMALL
    if version < 27:
        return MODE_SIZE_MEDIUM
    return MODE_SIZE_LARGE


def length_in_bits(mode, version):
    if mode not in (MODE_NUMBER, MODE_ALPHA_NUM, MODE_8BIT_BYTE, MODE_KANJI):
        raise TypeError(f"Invalid mode ({mode})")
    check_version(version)
    return mode_sizes_for_version(version)[mode]


def check_version(version):
    if version < 1 or version > 40:
        raise ValueError(f"Invalid version (was {version}, expected 1 to 40)")


def lost_point(modules):
    count = len(modules)
    return (
        _lost_point_level1(modules, count)
        + _lost_point_level2(modules, count)
        + _lost_point_level3(modules, count)
        + _lost_point_level4(modules, count)
    )


def _lost_point_level1(modules, modules_count):
    run_counts = [0] * (modules_count + 1)
    indexes = range(modules_count)

    for row in indexes:
        line = modules[row]
        color = line[0]
        run = 0
        for col in indexes:
            if line[col] == color:
                run += 1
            else:
                if run >= 5:
                    run_counts[run] += 1
                color = line[col]
                run = 1
        if run >= 5:
            run_counts[run] += 1

    for col in indexes:
        color = modules[0][col]
        run = 0
        for row in indexes:
            if modules[row][col] == color:
                run += 1
            else:
                if run >= 5:
                    run_counts[run] += 1
                color = modules[row][col]
                run = 1
        if run >= 5:
            run_counts[run] += 1

    return sum(run_counts[length] * (length - 2) for length in range(5, modules_count + 1))


def _lost_point_level2(modules, modules_count):
    score = 0
    positions = range(modules_count - 1)
    for row in positions:
        upper = modules[row]
        lower = modules[row + 1]
        iterator = iter(positions)
        for col in iterator:
            value = upper[col + 1]
            if value != lower[col + 1]:
                next(iterator, None)
            elif value != upper[col] or value != lower[col]:
                continue
            else:
                score += 3
    return score


def _lost_point_level3(modules, modules_count):
    score = 0
    all_indexes = range(modules_count)
    starts = range(modules_count - 10)

    for row in all_indexes:
        line = modules[row]
        iterator = iter(starts)
        for col in iterator:
            if (
                not line[col + 1]
                and line[col + 4]
                and not line[col + 5]
                and line[col + 6]
                and not line[col + 9]
                and (
                    (
                        line[col]
                        and line[col + 2]
                        and line[col + 3]
                        and not line[col + 7]
                        and not line[col + 8]
                        and not line[col + 10]
                    )
                    or (
                        not line[col]
                        and not line[col + 2]
                        and not line[col + 3]
                        and line[col + 7]
                        and line[col + 8]
                        and line[col + 10]
                    )
                )
            ):
                score += 40
            if line[col + 10]:
                next(iterator, None)

    for col in all_indexes:
        iterator = iter(starts)
        for row in iterator:
            if (
                not modules[row + 1][col]
                and modules[row + 4][col]
                and not modules[row + 5][col]
                and modules[row + 6][col]
                and not modules[row + 9][col]
                and (
                    (
                        modules[row][col]
                        and modules[row + 2][col]
                        and modules[row + 3][col]
                        and not modules[row + 7][col]
                        and not modules[row + 8][col]
                        and not modules[row + 10][col]
                    )
                    or (
                        not modules[row][col]
                        and not modules[row + 2][col]
                        and not modules[row + 3][col]
                        and modules[row + 7][col]
                        and modules[row + 8][col]
                        and modules[row + 10][col]
                    )
                )
            ):
                score += 40
            if modules[row + 10][col]:
                next(iterator, None)

    return score


def _lost_point_level4(modules, modules_count):
    dark = sum(sum(row) for row in modules)
    percentage = dark / (modules_count * modules_count)
    return int(abs(percentage * 100 - 50) // 5) * 10


def to_bytes(data):
    if isinstance(data, str):
        return data.encode("iso-8859-1")
    return data


def _optimal_split(data, pattern, minimum):
    while data:
        match = pattern.search(data)
        if match is None:
            break
        start, end = match.span()
        if start:
            yield False, data[:start]
        if end - start >= minimum:
            yield True, data[start:end]
        else:
            yield False, data[start:end]
        data = data[end:]
    if data:
        yield False, data


def optimal_data_chunks(data, minimum=4):
    data = to_bytes(data)
    if len(data) <= minimum:
        yield QRData(data)
        return

    number_pattern = re.compile(b"\d+")
    alpha_pattern = re.compile(b"[" + re.escape(ALPHA_NUM) + b"]+")

    for numeric, piece in _optimal_split(data, number_pattern, minimum):
        if numeric:
            yield QRData(piece, MODE_NUMBER)
            continue
        for alphanumeric, subpiece in _optimal_split(piece, alpha_pattern, minimum):
            if alphanumeric:
                yield QRData(subpiece, MODE_ALPHA_NUM)
            else:
                yield QRData(subpiece, MODE_8BIT_BYTE)


class QRData:
    def __init__(self, data, mode=None, check_data=True):
        if check_data:
            data = to_bytes(data)

        if mode is None:
            if RE_NUMBER.match(data):
                mode = MODE_NUMBER
            elif RE_ALPHA_NUM.match(data):
                mode = MODE_ALPHA_NUM
            else:
                mode = MODE_8BIT_BYTE
        elif mode not in (MODE_NUMBER, MODE_ALPHA_NUM, MODE_8BIT_BYTE, MODE_KANJI):
            raise TypeError(f"Invalid mode ({mode})")
        elif check_data and mode == MODE_NUMBER and not RE_NUMBER.match(data):
            raise ValueError(f"Provided data can not be represented in mode {mode}")
        elif check_data and mode == MODE_ALPHA_NUM and not RE_ALPHA_NUM.match(data):
            raise ValueError(f"Provided data can not be represented in mode {mode}")

        self.mode = mode
        self.data = data

    def __len__(self):
        return len(self.data)

    def write(self, buffer):
        if self.mode == MODE_NUMBER:
            for start in range(0, len(self.data), 3):
                part = self.data[start:start + 3]
                buffer.put(int(part), NUMBER_LENGTH[len(part)])
        elif self.mode == MODE_ALPHA_NUM:
            for start in range(0, len(self.data), 2):
                part = self.data[start:start + 2]
                if len(part) == 2:
                    buffer.put(ALPHA_NUM.find(part[0]) * 45 + ALPHA_NUM.find(part[1]), 11)
                else:
                    buffer.put(ALPHA_NUM.find(part[0]), 6)
        elif self.mode == MODE_8BIT_BYTE:
            for value in self.data:
                buffer.put(value, 8)
        else:
            raise NotImplementedError("Kanji mode not yet implemented")


class BitBuffer:
    def __init__(self):
        self.buffer = []
        self.length = 0

    def get(self, index):
        return ((self.buffer[index // 8] >> (7 - index % 8)) & 1) == 1

    def put(self, num, length):
        for shift in range(length - 1, -1, -1):
            self.put_bit(((num >> shift) & 1) == 1)

    def put_bit(self, bit):
        index = self.length // 8
        if index == len(self.buffer):
            self.buffer.append(0)
        if bit:
            self.buffer[index] |= 0x80 >> (self.length % 8)
        self.length += 1


def create_bytes(buffer, rs_blocks):
    offset = 0
    data_blocks = []
    error_blocks = []
    max_data = 0
    max_error = 0

    for block in rs_blocks:
        data_count = block.data_count
        error_count = block.total_count - data_count

        data = [0] * data_count
        for index in range(data_count):
            data[index] = 0xFF & buffer.buffer[index + offset]
        offset += data_count

        generator = LUT.rs_poly_LUT[error_count]
        if not generator:
            generator = base.Polynomial([1], 0)
            for index in range(error_count):
                generator *= base.Polynomial([1, base.gexp(index)], 0)

        remainder = base.Polynomial(data, len(generator) - 1) % generator
        error = [0] * (len(generator) - 1)
        padding = len(error) - len(remainder)
        for index in range(len(remainder)):
            error[index + padding] = remainder[index]

        data_blocks.append(data)
        error_blocks.append(error)
        max_data = max(max_data, len(data))
        max_error = max(max_error, len(error))

    result = []
    for index in range(max_data):
        for block in data_blocks:
            if index < len(block):
                result.append(block[index])

    for index in range(max_error):
        for block in error_blocks:
            if index < len(block):
                result.append(block[index])

    return result


def create_data(version, error_correction, data_list):
    blocks = base.rs_blocks(version, error_correction)
    buffer = BitBuffer()

    for data in data_list:
        buffer.put(data.mode, 4)
        buffer.put(len(data), length_in_bits(data.mode, version))
        data.write(buffer)

    limit = sum(block.data_count * 8 for block in blocks)
    if buffer.length > limit:
        raise exceptions.DataOverflowError(
            f"Code length overflow. Data size ({buffer.length}) > size available ({limit})"
        )

    for _ in range(min(limit - buffer.length, 4)):
        buffer.put_bit(False)

    remainder = buffer.length % 8
    if remainder:
        for _ in range(8 - remainder):
            buffer.put_bit(False)

    for index in range((limit - buffer.length) // 8):
        buffer.put(PAD0 if index % 2 == 0 else PAD1, 8)

    return create_bytes(buffer, blocks)