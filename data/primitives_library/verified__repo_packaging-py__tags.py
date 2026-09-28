from __future__ import annotations

import logging
import platform as _platform
import re
import struct
import subprocess
import sys
import sysconfig
from collections.abc import Iterator, Sequence
from importlib.machinery import EXTENSION_SUFFIXES
from typing import TypeVar

from . import _manylinux, _musllinux

__all__ = [
    "INTERPRETER_SHORT_NAMES",
    "AppleVersion",
    "InvalidTag",
    "PythonVersion",
    "Tag",
    "TooManyTagsError",
    "UnsortedTagsError",
    "android_platforms",
    "compatible_tags",
    "cpython_tags",
    "create_compatible_tags_selector",
    "generic_tags",
    "interpreter_abi",
    "interpreter_name",
    "interpreter_version",
    "ios_platforms",
    "mac_platforms",
    "parse_tag",
    "platform_tags",
    "pure_python_tags",
    "sys_tags",
]

logger = logging.getLogger(__name__)

PythonVersion = Sequence[int]
AppleVersion = tuple[int, int]
_T = TypeVar("_T")

INTERPRETER_SHORT_NAMES: dict[str, str] = {
    "python": "py",
    "cpython": "cp",
    "pypy": "pp",
    "ironpython": "ip",
    "jython": "jy",
}


def __dir__() -> list[str]:
    return __all__


def _compute_32_bit_interpreter() -> bool:
    return struct.calcsize("P") == 4


_32_BIT_INTERPRETER = _compute_32_bit_interpreter()


class UnsortedTagsError(ValueError):
    pass


class InvalidTag(ValueError):
    pass


class TooManyTagsError(ValueError):
    pass


class Tag:
    __slots__ = ["_abi", "_hash", "_interpreter", "_platform"]

    def __init__(self, interpreter: str, abi: str, platform: str) -> None:
        self._interpreter = interpreter.lower()
        self._abi = abi.lower()
        self._platform = platform.lower()
        self._hash = hash((self._interpreter, self._abi, self._platform))

    @property
    def interpreter(self) -> str:
        return self._interpreter

    @property
    def abi(self) -> str:
        return self._abi

    @property
    def platform(self) -> str:
        return self._platform

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tag):
            return NotImplemented
        return (
            self._hash == other._hash
            and self._platform == other._platform
            and self._abi == other._abi
            and self._interpreter == other._interpreter
        )

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f"{self._interpreter}-{self._abi}-{self._platform}"

    def __repr__(self) -> str:
        return f"<{self} @ {id(self)}>"

    def __getstate__(self) -> tuple[str, str, str]:
        return self._interpreter, self._abi, self._platform

    def __setstate__(self, state: object) -> None:
        if isinstance(state, tuple):
            if len(state) == 3 and all(isinstance(item, str) for item in state):
                self._interpreter, self._abi, self._platform = state
                self._hash = hash((self._interpreter, self._abi, self._platform))
                return
            if len(state) == 2 and isinstance(state[1], dict):
                slots = state[1]
                try:
                    interpreter = slots["_interpreter"]
                    abi = slots["_abi"]
                    platform = slots["_platform"]
                except KeyError:
                    raise TypeError(f"Cannot restore Tag from {state!r}") from None
                if all(isinstance(value, str) for value in (interpreter, abi, platform)):
                    self._interpreter = interpreter.lower()
                    self._abi = abi.lower()
                    self._platform = platform.lower()
                    self._hash = hash(
                        (self._interpreter, self._abi, self._platform)
                    )
                    return
        raise TypeError(f"Cannot restore Tag from {state!r}")


def parse_tag(
    tag: str, *, validate_order: bool = False, limit: int | None = None
) -> frozenset[Tag]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    fields = tag.split("-")
    if len(fields) != 3:
        raise InvalidTag(f"Invalid tag: {tag!r}")

    portions = [field.split(".") for field in fields]
    for index, items in enumerate(portions):
        if not items or any(item == "" for item in items):
            raise InvalidTag(f"Invalid tag: {tag!r}")
        if index == 0 and any(not item.isidentifier() for item in items):
            raise InvalidTag(f"Invalid tag: {tag!r}")
        if validate_order and items != sorted(items):
            raise UnsortedTagsError(f"Tag components are not sorted: {tag!r}")

    count = len(portions[0]) * len(portions[1]) * len(portions[2])
    if limit is not None and count > limit:
        raise TooManyTagsError(
            f"Compressed tag set contains {count} tags, exceeding limit of {limit}"
        )

    return frozenset(
        Tag(interpreter, abi, platform)
        for interpreter in portions[0]
        for abi in portions[1]
        for platform in portions[2]
    )


def _normalize_string(value: str) -> str:
    return re.sub(r"[-.\s]+", "_", value).lower()


def _get_config_var(name: str, warn: bool = False) -> int | str | None:
    value = sysconfig.get_config_var(name)
    if value is None and warn:
        logger.warning("Config variable '%s' is unset, Python ABI tag may be incorrect", name)
    return value


def interpreter_name() -> str:
    implementation = getattr(sys, "implementation", None)
    name = getattr(implementation, "name", None)
    if name is None:
        name = _platform.python_implementation().lower()
    return INTERPRETER_SHORT_NAMES.get(name, name)


def interpreter_version(*, warn: bool = False) -> str:
    version = _get_config_var("py_version_nodot", warn=warn)
    if version:
        return str(version)
    return "".join(str(value) for value in sys.version_info[:2])


def _abi3_applies(python_version: PythonVersion) -> bool:
    return tuple(python_version) >= (3, 2)


def _cpython_abis(
    python_version: PythonVersion, warn: bool = False
) -> list[str]:
    version = tuple(python_version)
    version_string = "".join(str(value) for value in version[:2])
    abis: list[str] = []

    if version >= (3, 13):
        gil_disabled = _get_config_var("Py_GIL_DISABLED", warn=warn)
        if gil_disabled:
            abis.append(f"cp{version_string}t")

    if version < (3, 8):
        debug = _get_config_var("Py_DEBUG", warn=warn)
        pymalloc = _get_config_var("WITH_PYMALLOC", warn=warn)
        unicode_size = _get_config_var("Py_UNICODE_SIZE", warn=warn)

        flags = ""
        if debug:
            flags += "d"
        if pymalloc:
            flags += "m"
        if unicode_size == 4:
            flags += "u"

        if flags:
            abis.append(f"cp{version_string}{flags}")
    else:
        debug = _get_config_var("Py_DEBUG", warn=warn)
        if debug:
            abis.append(f"cp{version_string}d")

    abis.append(f"cp{version_string}")
    return abis


def _is_threaded_cpython(python_version: PythonVersion) -> bool:
    return tuple(python_version) >= (3, 13) and bool(
        _get_config_var("Py_GIL_DISABLED")
    )


def cpython_tags(
    python_version: PythonVersion | None = None,
    abis: Sequence[str] | None = None,
    platforms: Sequence[str] | None = None,
    *,
    warn: bool = False,
) -> Iterator[Tag]:
    if python_version is None:
        python_version = sys.version_info[:2]
    version = tuple(python_version)
    interpreter = "cp" + "".join(str(value) for value in version[:2])

    if abis is None:
        abis = _cpython_abis(version, warn)
    if platforms is None:
        platforms = list(platform_tags())

    for abi in abis:
        for platform in platforms:
            yield Tag(interpreter, abi, platform)

    threaded = _is_threaded_cpython(version)
    if not threaded:
        for platform in platforms:
            yield Tag(interpreter, "abi3", platform)

    for platform in platforms:
        yield Tag(interpreter, "none", platform)

    if _abi3_applies(version):
        major = version[0]
        for minor in range(version[1] - 1, 1, -1):
            old_interpreter = f"cp{major}{minor}"
            for platform in platforms:
                yield Tag(old_interpreter, "abi3", platform)


def interpreter_abi() -> str:
    soabi = _get_config_var("SOABI")
    if soabi:
        soabi = str(soabi)
        if soabi.startswith("cpython-"):
            parts = soabi.split("-")
            if len(parts) >= 2:
                return "cp" + parts[1]
        return _normalize_string(soabi)

    for suffix in EXTENSION_SUFFIXES:
        value = suffix
        if value.startswith("."):
            value = value[1:]
        for ending in (".so", ".pyd", ".dll", ".dylib"):
            if value.endswith(ending):
                value = value[: -len(ending)]
                break
        if value.startswith("cpython-"):
            pieces = value.split("-")
            if len(pieces) >= 2:
                return "cp" + pieces[1]
        if value:
            return _normalize_string(value)

    return "none"


def generic_tags(
    interpreter: str | None = None,
    abis: Sequence[str] | None = None,
    platforms: Sequence[str] | None = None,
) -> Iterator[Tag]:
    if interpreter is None:
        interpreter = interpreter_name() + interpreter_version()
    if abis is None:
        abis = [interpreter_abi()]
    if platforms is None:
        platforms = list(platform_tags())

    for abi in abis:
        for platform in platforms:
            yield Tag(interpreter, abi, platform)

    for platform in platforms:
        yield Tag(interpreter, "none", platform)


def _py_interpreter_range(python_version: PythonVersion) -> Iterator[str]:
    major, minor = python_version[:2]
    yield f"py{major}{minor}"
    yield f"py{major}"
    for older_minor in range(minor - 1, -1, -1):
        yield f"py{major}{older_minor}"


def compatible_tags(
    python_version: PythonVersion | None = None,
    interpreter: str | None = None,
    platforms: Sequence[str] | None = None,
) -> Iterator[Tag]:
    if python_version is None:
        python_version = sys.version_info[:2]
    if platforms is None:
        platforms = list(platform_tags())
    if interpreter is None:
        interpreter = "py" + "".join(str(value) for value in python_version[:2])

    versions = list(_py_interpreter_range(python_version))
    for version in versions:
        for platform in platforms:
            yield Tag(version, "none", platform)

    yield Tag(interpreter, "none", "any")

    for version in versions:
        yield Tag(version, "none", "any")


def pure_python_tags(
    python_version: PythonVersion | None = None,
) -> Iterator[Tag]:
    if python_version is None:
        python_version = sys.version_info[:2]
    yield from compatible_tags(python_version, platforms=[])


def create_compatible_tags_selector(
    python_version: PythonVersion | None = None,
    interpreter: str | None = None,
    platforms: Sequence[str] | None = None,
):
    def selector() -> Iterator[Tag]:
        return compatible_tags(python_version, interpreter, platforms)

    return selector


def _mac_arch(arch: str) -> str:
    if not _32_BIT_INTERPRETER:
        return arch
    if arch == "x86_64":
        return "i386"
    if arch == "ppc64":
        return "ppc"
    return arch


def _mac_binary_formats(version: AppleVersion, cpu_arch: str) -> list[str]:
    major, minor = version
    if cpu_arch == "x86_64":
        formats = ["x86_64", "intel", "fat64", "fat32"]
    elif cpu_arch == "i386":
        formats = ["i386", "intel", "fat32"]
    elif cpu_arch == "ppc64":
        formats = ["ppc64", "fat64", "fat32"]
    elif cpu_arch == "ppc":
        formats = ["ppc", "fat32"]
    elif cpu_arch == "intel":
        formats = ["intel", "fat32"]
    else:
        formats = [cpu_arch]

    if major > 10 or (major == 10 and minor >= 4):
        return formats
    return [cpu_arch]


def mac_platforms(
    version: AppleVersion | None = None, arch: str | None = None
) -> Iterator[str]:
    if version is None:
        version_text, _, detected_arch = _platform.mac_ver()
        if not version_text:
            return
        version_parts = version_text.split(".")
        version = (int(version_parts[0]), int(version_parts[1]))
        if version == (10, 16):
            try:
                output = subprocess.check_output(
                    ["/usr/bin/sw_vers", "-productVersion"], stderr=subprocess.DEVNULL
                )
                actual = output.decode().strip().split(".")
                version = (int(actual[0]), int(actual[1]))
            except Exception:
                pass
        if arch is None:
            arch = detected_arch

    if arch is None:
        arch = _platform.machine()

    arch = _mac_arch(arch)
    major, minor = version

    if major >= 11:
        for current_minor in range(minor, -1, -1):
            yield f"macosx_{major}_{current_minor}_{arch}"
        for current_major in range(major - 1, 10, -1):
            yield f"macosx_{current_major}_0_{arch}"
        compat_version = (10, 16)
    else:
        for current_minor in range(minor, -1, -1):
            yield f"macosx_{major}_{current_minor}_{arch}"
        compat_version = version

    if compat_version[0] == 10:
        for current_minor in range(compat_version[1], 3, -1):
            for binary_format in _mac_binary_formats(
                (10, current_minor), arch
            ):
                yield f"macosx_10_{current_minor}_{binary_format}"


def ios_platforms(
    version: AppleVersion | None = None, multiarch: str | None = None
) -> Iterator[str]:
    if version is None:
        ios_ver = getattr(_platform, "ios_ver", None)
        if ios_ver is None:
            return
        release = ios_ver().release
        if not release:
            return
        parts = release.split(".")
        version = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)

    if multiarch is None:
        value = _normalize_string(sysconfig.get_platform())
        if value.startswith("ios_"):
            pieces = value.split("_")
            if len(pieces) >= 4:
                multiarch = "_".join(pieces[3:])
        if not multiarch:
            multiarch = _normalize_string(_platform.machine())

    major, minor = version
    for current_minor in range(minor, -1, -1):
        yield f"ios_{major}_{current_minor}_{multiarch}"
    for current_major in range(major - 1, 11, -1):
        yield f"ios_{current_major}_0_{multiarch}"


def android_platforms(
    api_level: int | None = None, abi: str | None = None
) -> Iterator[str]:
    if api_level is None:
        android_ver = getattr(_platform, "android_ver", None)
        if android_ver is None:
            return
        value = android_ver()
        api_level = getattr(value, "api_level", None)
        if api_level is None:
            return
        api_level = int(api_level)

    if abi is None:
        machine = _normalize_string(_platform.machine())
        abi = {
            "aarch64": "arm64_v8a",
            "armv7l": "armeabi_v7a",
            "x86_64": "x86_64",
            "i686": "x86",
            "x86": "x86",
        }.get(machine, machine)

    for level in range(api_level, 15, -1):
        yield f"android_{level}_{abi}"


def _linux_platforms() -> Iterator[str]:
    linux = _normalize_string(sysconfig.get_platform())
    if _32_BIT_INTERPRETER:
        if linux == "linux_x86_64":
            linux = "linux_i686"
        elif linux == "linux_aarch64":
            linux = "linux_armv7l"

    if not linux.startswith("linux_"):
        yield linux
        return

    arch = linux[6:]
    archs = [arch]
    yield from _manylinux.platform_tags(archs)
    yield from _musllinux.platform_tags(archs)
    yield linux


def platform_tags() -> Iterator[str]:
    system = _platform.system()
    if system == "Darwin":
        yield from mac_platforms()
    elif system == "iOS":
        yield from ios_platforms()
    elif system == "Android":
        yield from android_platforms()
    elif system == "Linux":
        yield from _linux_platforms()
    else:
        yield _normalize_string(sysconfig.get_platform())


def sys_tags(*, warn: bool = False) -> Iterator[Tag]:
    name = interpreter_name()
    if name == "cp":
        yield from cpython_tags(warn=warn)
    else:
        yield from generic_tags()

    yield from compatible_tags()