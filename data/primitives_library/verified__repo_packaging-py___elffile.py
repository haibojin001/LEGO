from __future__ import annotations

import enum
import os
import struct
from typing import IO


class ELFInvalid(ValueError):
    pass


class EIClass(enum.IntEnum):
    C32 = 1
    C64 = 2


class EIData(enum.IntEnum):
    Lsb = 1
    Msb = 2


class EMachine(enum.IntEnum):
    I386 = 3
    S390 = 22
    Arm = 40
    X8664 = 62
    AArch64 = 183


class ELFFile:
    """
    Representation of an ELF executable.
    """

    def __init__(self, f: IO[bytes]) -> None:
        self._f = f

        try:
            identification = self._read("16B")
        except struct.error as exc:
            raise ELFInvalid("unable to parse identification") from exc

        magic = bytes(identification[:4])
        if magic != b"\x7fELF":
            raise ELFInvalid(f"invalid magic: {magic!r}")

        self.capacity = identification[4]
        self.encoding = identification[5]

        layouts = {
            (EIClass.C32, EIData.Lsb): (
                "<HHIIIIIHHH",
                "<IIIIIIII",
                (0, 1, 4),
            ),
            (EIClass.C32, EIData.Msb): (
                ">HHIIIIIHHH",
                ">IIIIIIII",
                (0, 1, 4),
            ),
            (EIClass.C64, EIData.Lsb): (
                "<HHIQQQIHHH",
                "<IIQQQQQQ",
                (0, 2, 5),
            ),
            (EIClass.C64, EIData.Msb): (
                ">HHIQQQIHHH",
                ">IIQQQQQQ",
                (0, 2, 5),
            ),
        }

        try:
            header_format, self._p_fmt, self._p_idx = layouts[
                (self.capacity, self.encoding)
            ]
        except KeyError as exc:
            raise ELFInvalid(
                f"unrecognized capacity ({self.capacity}) or encoding ({self.encoding})"
            ) from exc

        try:
            (
                _,
                self.machine,
                _,
                _,
                self._e_phoff,
                _,
                self.flags,
                _,
                self._e_phentsize,
                self._e_phnum,
            ) = self._read(header_format)
        except struct.error as exc:
            raise ELFInvalid(
                "unable to parse machine and section information"
            ) from exc

    def _read(self, fmt: str) -> tuple[int, ...]:
        return struct.unpack(fmt, self._f.read(struct.calcsize(fmt)))

    @property
    def interpreter(self) -> str | None:
        """
        The path recorded in the ``PT_INTERP`` section header.
        """
        for number in range(self._e_phnum):
            self._f.seek(self._e_phoff + self._e_phentsize * number)

            try:
                header = self._read(self._p_fmt)
            except struct.error:
                continue

            if header[self._p_idx[0]] != 3:
                continue

            self._f.seek(header[self._p_idx[1]])
            return os.fsdecode(self._f.read(header[self._p_idx[2]])).strip("\0")

        return None