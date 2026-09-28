"""SBV format parser."""

import re
import typing

from . import utils
from .errors import MalformedFileError
from .models import Caption


class SBVCueBlock:
    """A parsed SBV timing entry and its text."""

    CUE_TIMINGS_PATTERN = re.compile(
        r'\s*(\d{1,2}:\d{1,2}:\d{1,2}.\d{3}),(\d{1,2}:\d{1,2}:\d{1,2}.\d{3})'
    )

    def __init__(
            self,
            start: str,
            end: str,
            payload: typing.Sequence[str]
            ):
        self.start = start
        self.end = end
        self.payload = payload

    @classmethod
    def is_valid(
            cls,
            lines: typing.Sequence[str]
            ) -> bool:
        if len(lines) < 2:
            return False

        return bool(
            re.match(cls.CUE_TIMINGS_PATTERN, lines[0])
            and lines[1].strip()
        )

    @classmethod
    def from_lines(
            cls,
            lines: typing.Sequence[str]
            ) -> 'SBVCueBlock':
        timing_match = re.match(cls.CUE_TIMINGS_PATTERN, lines[0])
        assert timing_match is not None

        return cls(
            timing_match.group(1),
            timing_match.group(2),
            lines[1:]
        )


def parse(lines: typing.Sequence[str]) -> typing.List[Caption]:
    if not _is_valid_content(lines):
        raise MalformedFileError('Invalid format')

    return _parse_captions(lines)


def _is_valid_content(lines: typing.Sequence[str]) -> bool:
    if len(lines) < 2:
        return False

    initial_block = next(utils.iter_blocks_of_lines(lines))
    return bool(initial_block and SBVCueBlock.is_valid(initial_block))


def _parse_captions(lines: typing.Sequence[str]) -> typing.List[Caption]:
    parsed = []

    for line_block in utils.iter_blocks_of_lines(lines):
        if not SBVCueBlock.is_valid(line_block):
            continue

        cue = SBVCueBlock.from_lines(line_block)
        parsed.append(Caption(cue.start, cue.end, cue.payload))

    return parsed