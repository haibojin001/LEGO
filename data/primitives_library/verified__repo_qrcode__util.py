import math
import re

from qrcode import LUT, base, exceptions
from qrcode.base import RSBlock

MODE_NUMBER = 1
MODE_ALPHA_NUM = 2
MODE_8BIT_BYTE = 4
MODE_KANJI = 8

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
        8 * sum(_data_count(block) for block in base.rs_blocks(version, level))
        for version in range(1, 41)
    ]
    for level in range(4)
]


def BCH_digit(data):
    result = 0
    while data:
        result += 1
        data >>= 1
    return result


def BCH_type_info(data):
    value = data << 10
    divisor_width = BCH_digit(G15)
    while BCH_digit(value) >= divisor_width:
        value ^= G15 << (BCH_digit(value) - divisor_width)
    return ((data << 10) | value) ^ G15_MASK


def BCH_type_number(data):
    value = data << 12
    divisor_width = BCH_digit(G18)
    while BCH_digit(value) >= divisor_width:
        value ^= G18 << (BCH_digit(value) - divisor_width)
    return (data << 12) | value


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


def check_version(version):
    if version < 1 or version > 40:
        raise ValueError(f"Invalid version (was {version}, expected 1 to 40)")


def length_in_bits(mode, version):
    if mode not in (MODE_NUMBER, MODE_ALPHA_NUM, MODE_8BIT_BYTE, MODE_KANJI):
        raise TypeError(f"Invalid mode ({mode})")
    check_version(version)
    return mode_sizes_for_version(version)[mode]


def lost_point(modules):
    size = len(modules)
    return (
        _lost_point_level1(modules, size)
        + _lost_point_level2(modules, size)
        + _lost_point_level3(modules, size)
        + _lost_point_level4(modules, size)
    )


def _lost_point_level1(modules, modules_count):
    counts = [0] * (modules_count + 1)
    indexes = range(modules_count)

    for row in indexes:
        line = modules[row]
        previous = line[0]
        run = 0
        for col in indexes:
            if line[col] == previous:
                run += 1
            else:
                if run >= 5:
                    counts[run] += 1
                previous = line[col]
                run = 1
        if run >= 5:
            counts[run] += 1

    for col in indexes:
        previous = modules[0][col]
        run = 0
        for row in indexes:
            if modules[row][col] == previous:
                run += 1
            else:
                if run >= 5:
                    counts[run] += 1
                previous = modules[row][col]
                run = 1
        if run >= 5:
            counts[run] += 1

    return sum(counts[length] * (length - 2) for length in range(5, modules_count + 1))


def _lost_point_level2(modules, modules_count):
    score = 0
    positions = range(modules_count - 1)

    for row in positions:
        upper = modules[row]
        lower = modules[row + 1]
        iterator = iter(positions)
        for col in iterator:
            color = upper[col + 1]
            if color != lower[col + 1]:
                next(iterator, None)
            elif color != upper[col] or color != lower[col]:
                continue
            else:
                score += 3
    return score


def _lost_point_level3(modules, modules_count):
    score = 0
    all_positions = range(modules_count)
    starts = range(modules_count - 10)

    for row in all_positions:
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

    for col in all_positions:
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
    percent = dark / (modules_count**2)
    return int(abs(percent * 100 - 50) // 5) * 10


def optimal_mode(data):
    if data.isdigit():
        return MODE_NUMBER
    if RE_ALPHA_NUM.match(data):
        return MODE_ALPHA_NUM
    return MODE_8BIT_BYTE


def to_bytestring(data):
    if not isinstance(data, bytes):
        data = str(data).encode("utf-8")
    return data


def _optimal_split(data, pattern):
    while data:
        match = pattern.search(data)
        if match is None:
            break
        start, end = match.span()
        if start:
            yield False, data[:start]
        yield True, data[start:end]
        data = data[end:]
    if data:
        yield False, data


def optimal_data_chunks(data, minimum=4):
    data = to_bytestring(data)
    if len(data) <= minimum:
        if data:
            yield QRData(data)
        return

    minimum_bytes = str(minimum).encode("ascii")
    numeric = re.compile(b"\\d{" + minimum_bytes + b",}")
    alphanumeric = re.compile(
        b"[" + re.escape(ALPHA_NUM) + b"]{" + minimum_bytes + b",}"
    )

    for is_numeric, chunk in _optimal_split(data, numeric):
        if is_numeric:
            yield QRData(chunk, mode=MODE_NUMBER, check_data=False)
        else:
            for is_alphanumeric, part in _optimal_split(chunk, alphanumeric):
                if is_alphanumeric:
                    yield QRData(part, mode=MODE_ALPHA_NUM, check_data=False)
                else:
                    yield QRData(part, mode=MODE_8BIT_BYTE, check_data=False)


class QRData:
    def __init__(self, data, mode=None, check_data=True):
        if check_data:
            data = to_bytestring(data)

        if mode is None:
            mode = optimal_mode(data)
        elif mode not in (MODE_NUMBER, MODE_ALPHA_NUM, MODE_8BIT_BYTE):
            raise TypeError(f"Invalid mode ({mode})")
        elif check_data and mode == MODE_NUMBER and not data.isdigit():
            raise ValueError("Provided data can not be represented in mode 1")
        elif check_data and mode == MODE_ALPHA_NUM and not RE_ALPHA_NUM.match(data):
            raise ValueError("Provided data can not be represented in mode 2")

        self.mode = mode
        self.data = data

    def __len__(self):
        return len(self.data)

    def write(self, buffer):
        if self.mode == MODE_NUMBER:
            for index in range(0, len(self.data), 3):
                piece = self.data[index : index + 3]
                buffer.put(int(piece), NUMBER_LENGTH[len(piece)])
        elif self.mode == MODE_ALPHA_NUM:
            for index in range(0, len(self.data), 2):
                piece = self.data[index : index + 2]
                if len(piece) == 2:
                    value = ALPHA_NUM.index(piece[0]) * 45 + ALPHA_NUM.index(piece[1])
                    buffer.put(value, 11)
                else:
                    buffer.put(ALPHA_NUM.index(piece[0]), 6)
        else:
            for byte in self.data:
                buffer.put(byte, 8)


class BitBuffer:
    def __init__(self):
        self.buffer = []
        self.length = 0

    def __repr__(self):
        return ".".join(f"{value:08b}" for value in self.buffer)

    def get(self, index):
        return bool((self.buffer[index // 8] >> (7 - index % 8)) & 1)

    def put(self, num, length):
        for shift in range(length - 1, -1, -1):
            self.put_bit(((num >> shift) & 1) == 1)

    def put_bit(self, bit):
        slot = self.length // 8
        if slot == len(self.buffer):
            self.buffer.append(0)
        if bit:
            self.buffer[slot] |= 0x80 >> (self.length % 8)
        self.length += 1


def create_bytes(buffer, rs_blocks):
    offset = 0
    data_blocks = []
    ecc_blocks = []
    max_data = 0
    max_ecc = 0

    for rs_block in rs_blocks:
        data_count = rs_block.data_count
        ecc_count = rs_block.total_count - data_count
        max_data = max(max_data, data_count)
        max_ecc = max(max_ecc, ecc_count)

        data = [buffer.buffer[offset + index] & 0xFF for index in range(data_count)]
        offset += data_count
        data_blocks.append(data)

        if ecc_count in LUT.rs_poly_LUT:
            generator = base.Polynomial(LUT.rs_poly_LUT[ecc_count], 0)
        else:
            generator = base.Polynomial([1], 0)
            for index in range(ecc_count):
                generator = generator * base.Polynomial([1, base.gexp(index)], 0)

        remainder = base.Polynomial(data, len(generator) - 1) % generator
        ecc = [0] * (len(generator) - 1)
        for index in range(len(ecc)):
            remainder_index = index + len(remainder) - len(ecc)
            if remainder_index >= 0:
                ecc[index] = remainder[remainder_index]
        ecc_blocks.append(ecc)

    output = []
    for index in range(max_data):
        for block in data_blocks:
            if index < len(block):
                output.append(block[index])

    for index in range(max_ecc):
        for block in ecc_blocks:
            if index < len(block):
                output.append(block[index])

    return output


def create_data(version, error_correction, data_list):
    rs_blocks = base.rs_blocks(version, error_correction)
    capacity = sum(block.data_count for block in rs_blocks) * 8
    buffer = BitBuffer()

    for data in data_list:
        buffer.put(data.mode, 4)
        buffer.put(len(data), length_in_bits(data.mode, version))
        data.write(buffer)

    if buffer.length > capacity:
        raise exceptions.DataOverflowError(
            "Code length overflow. Data size (%s) > size available (%s)"
            % (buffer.length, capacity)
        )

    for _ in range(min(4, capacity - buffer.length)):
        buffer.put_bit(False)

    remainder = buffer.length % 8
    if remainder:
        for _ in range(8 - remainder):
            buffer.put_bit(False)

    pad = PAD0
    while buffer.length < capacity:
        buffer.put(pad, 8)
        pad = PAD1 if pad == PAD0 else PAD0

    return create_bytes(buffer, rs_blocks)