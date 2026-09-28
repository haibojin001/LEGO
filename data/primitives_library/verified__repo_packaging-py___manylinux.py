from __future__ import annotations

import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import TYPE_CHECKING, NamedTuple

from ._elffile import EIClass, EIData, ELFFile, EMachine

if TYPE_CHECKING:
    import types
    from collections.abc import Generator, Iterator, Sequence


EF_ARM_ABIMASK = 0xFF000000
EF_ARM_ABI_VER5 = 0x05000000
EF_ARM_ABI_FLOAT_HARD = 0x00000400

_ALLOWED_ARCHS = {
    "x86_64",
    "aarch64",
    "ppc64",
    "ppc64le",
    "s390x",
    "loongarch64",
    "riscv64",
}


@contextlib.contextmanager
def _parse_elf(path: str) -> Generator[ELFFile | None]:
    try:
        with open(path, "rb") as stream:
            yield ELFFile(stream)
    except (OSError, TypeError, ValueError):
        yield None


def _is_linux_armhf(executable: str) -> bool:
    with _parse_elf(executable) as elf:
        if elf is None:
            return False
        return (
            elf.capacity == EIClass.C32
            and elf.encoding == EIData.Lsb
            and elf.machine == EMachine.Arm
            and elf.flags & EF_ARM_ABIMASK == EF_ARM_ABI_VER5
            and elf.flags & EF_ARM_ABI_FLOAT_HARD == EF_ARM_ABI_FLOAT_HARD
        )


def _is_linux_i686(executable: str) -> bool:
    with _parse_elf(executable) as elf:
        if elf is None:
            return False
        return (
            elf.capacity == EIClass.C32
            and elf.encoding == EIData.Lsb
            and elf.machine == EMachine.I386
        )


def _have_compatible_abi(executable: str, archs: Sequence[str]) -> bool:
    if "armv7l" in archs:
        return _is_linux_armhf(executable)
    if "i686" in archs:
        return _is_linux_i686(executable)
    return any(architecture in _ALLOWED_ARCHS for architecture in archs)


_LAST_GLIBC_MINOR: dict[int, int] = collections.defaultdict(lambda: 50)


class _GLibCVersion(NamedTuple):
    major: int
    minor: int


def _glibc_version_string_confstr() -> str | None:
    try:
        value: str | None = os.confstr("CS_GNU_LIBC_VERSION")
        assert value is not None
        _, version = value.rsplit()
    except (AssertionError, AttributeError, OSError, ValueError):
        return None
    return version


def _glibc_version_string_ctypes() -> str | None:
    try:
        import ctypes
    except ImportError:
        return None

    try:
        namespace = ctypes.CDLL(None)
    except OSError:
        return None

    try:
        get_version = namespace.gnu_get_libc_version
    except AttributeError:
        return None

    get_version.restype = ctypes.c_char_p
    result: str | bytes = get_version()
    if isinstance(result, bytes):
        result = result.decode("ascii")
    return result


def _glibc_version_string() -> str | None:
    return _glibc_version_string_confstr() or _glibc_version_string_ctypes()


def _parse_glibc_version(version_str: str) -> _GLibCVersion:
    match = re.match(r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)", version_str)
    if match is None:
        warnings.warn(
            f"Expected glibc version with 2 components major.minor, got: {version_str}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _GLibCVersion(-1, -1)
    return _GLibCVersion(
        int(match.group("major")),
        int(match.group("minor")),
    )


@functools.lru_cache
def _get_glibc_version() -> _GLibCVersion:
    version = _glibc_version_string()
    if version is None:
        return _GLibCVersion(-1, -1)
    return _parse_glibc_version(version)


@functools.lru_cache(maxsize=1)
def _get_manylinux_module() -> types.ModuleType | None:
    try:
        return __import__("_manylinux")
    except ImportError:
        return None


def _is_compatible(arch: str, version: _GLibCVersion) -> bool:
    if _get_glibc_version() < version:
        return False

    module = _get_manylinux_module()
    if module is None:
        return True

    if hasattr(module, "manylinux_compatible"):
        compatible = module.manylinux_compatible(version[0], version[1], arch)
        if compatible is not None:
            return bool(compatible)
        return True

    if version == _GLibCVersion(2, 5) and hasattr(
        module, "manylinux1_compatible"
    ):
        return bool(module.manylinux1_compatible)

    if version == _GLibCVersion(2, 12) and hasattr(
        module, "manylinux2010_compatible"
    ):
        return bool(module.manylinux2010_compatible)

    if version == _GLibCVersion(2, 17) and hasattr(
        module, "manylinux2014_compatible"
    ):
        return bool(module.manylinux2014_compatible)

    return True


_LEGACY_MANYLINUX_MAP: dict[_GLibCVersion, str] = {
    _GLibCVersion(2, 17): "manylinux2014",
    _GLibCVersion(2, 12): "manylinux2010",
    _GLibCVersion(2, 5): "manylinux1",
}


def platform_tags(archs: Sequence[str]) -> Iterator[str]:
    if not _have_compatible_abi(sys.executable, archs):
        return

    minimum_glibc2 = _GLibCVersion(2, 16)
    if set(archs).intersection({"x86_64", "i686"}):
        minimum_glibc2 = _GLibCVersion(2, 4)

    detected = _GLibCVersion(*_get_glibc_version())
    maxima = [detected]

    for major in range(detected.major - 1, 1, -1):
        maxima.append(_GLibCVersion(major, _LAST_GLIBC_MINOR[major]))

    for arch in archs:
        for maximum in maxima:
            lowest_minor = minimum_glibc2.minor if maximum.major == 2 else 0

            for minor in range(maximum.minor, lowest_minor - 1, -1):
                version = _GLibCVersion(maximum.major, minor)
                if not _is_compatible(arch, version):
                    continue

                yield f"manylinux_{version.major}_{version.minor}_{arch}"

                legacy = _LEGACY_MANYLINUX_MAP.get(version)
                if legacy is not None:
                    yield f"{legacy}_{arch}"