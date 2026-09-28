import re
import typing

from . import utils
from .errors import MalformedFileError
from .models import Caption


class SRTCueBlock:
    """A parsed SRT cue block containing its number, timing, and text."""

    CUE_TIMINGS_PATTERN = re.compile(
        r'\s*(\d+:\d{2}:\d{2},\d{3})\s*-->\s*(\d+:\d{2}:\d{2},\d{3})'
    )

    def __init__(
            self,
            index: str,
            start: str,
            end: str,
            payload: typing.Sequence[str]
            ):
        self.index = index
        self.start = start
        self.end = end
        self.payload = payload

    @classmethod
    def is_valid(
            cls,
            lines: typing.Sequence[str]
            ) -> bool:
        if len(lines) < 3:
            return False
        if not lines[0].isdigit():
            return False
        return bool(cls.CUE_TIMINGS_PATTERN.match(lines[1]))

    @classmethod
    def from_lines(
            cls,
            lines: typing.Sequence[str]
            ) -> 'SRTCueBlock':
        timing_match = cls.CUE_TIMINGS_PATTERN.match(lines[1])
        assert timing_match is not None
        return cls(
            lines[0],
            timing_match.group(1),
            timing_match.group(2),
            lines[2:],
        )


def parse(lines: typing.Sequence[str]) -> typing.List[Caption]:
    """Convert SRT input lines into caption objects."""
    if not is_valid_content(lines):
        raise MalformedFileError('Invalid format')
    return parse_captions(lines)


def is_valid_content(lines: typing.Sequence[str]) -> bool:
    """Return whether the input has the minimum shape of SRT content."""
    return bool(
        len(lines) >= 3
        and lines[0].isdigit()
        and '-->' in lines[1]
        and lines[2].strip()
    )


def parse_captions(lines: typing.Sequence[str]) -> typing.List[Caption]:
    """Extract valid caption blocks from SRT input lines."""
    result: typing.List[Caption] = []

    for block in utils.iter_blocks_of_lines(lines):
        if not SRTCueBlock.is_valid(block):
            continue

        cue = SRTCueBlock.from_lines(block)
        start = cue.start.replace(',', '.')
        end = cue.end.replace(',', '.')
        result.append(Caption(start, end, cue.payload))

    return result


def write(
        f: typing.IO[str],
        captions: typing.Iterable[Caption]
        ):
    """Serialize captions to SRT text and write them to *f*."""
    lines = []

    for number, caption in enumerate(captions, 1):
        lines.append(str(number))
        lines.append(
            '{} --> {}'.format(
                caption.start.replace('.', ','),
                caption.end.replace('.', ','),
            )
        )
        lines.extend(caption.text.splitlines())
        lines.append('')

    f.write('\n'.join(lines).rstrip())