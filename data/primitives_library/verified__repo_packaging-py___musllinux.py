from __future__ import annotations

import functools
import re
import subprocess
import sys
from typing import TYPE_CHECKING, NamedTuple

from ._elffile import ELFFile

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class _MuslVersion(NamedTuple):
    major: int
    minor: int


def _parse_musl_version(output: str) -> _MuslVersion | None:
    records = []
    for line in output.splitlines():
        line = line.strip()
        if line:
            records.append(line)

    if len(records) < 2 or not records[0].startswith("musl"):
        return None

    match = re.match(r"Version (\d+)\.(\d+)", records[1])
    if match is None:
        return None

    return _MuslVersion(int(match.group(1)), int(match.group(2)))


@functools.lru_cache
def _get_musl_version(executable: str) -> _MuslVersion | None:
    try:
        with open(executable, "rb") as executable_file:
            interpreter = ELFFile(executable_file).interpreter
    except (OSError, TypeError, ValueError):
        return None

    if interpreter is None or "musl" not in interpreter:
        return None

    result = subprocess.run(
        [interpreter],
        check=False,
        stderr=subprocess.PIPE,
        text=True,
    )
    return _parse_musl_version(result.stderr)


def platform_tags(archs: Sequence[str]) -> Iterator[str]:
    musl_version = _get_musl_version(sys.executable)
    if musl_version is None:
        return

    for architecture in archs:
        for minor_version in range(musl_version.minor, -1, -1):
            yield (
                f"musllinux_{musl_version.major}_{minor_version}_{architecture}"
            )


if __name__ == "__main__":  # pragma: no cover
    import sysconfig

    platform = sysconfig.get_platform()
    assert platform.startswith("linux-"), "not linux"

    architecture = re.sub(r"[.-]", "_", platform.split("-", 1)[1])
    print("plat:", platform)
    print("musl:", _get_musl_version(sys.executable))
    print("tags:", end=" ")
    for tag in platform_tags([architecture]):
        print(tag, end="\n      ")