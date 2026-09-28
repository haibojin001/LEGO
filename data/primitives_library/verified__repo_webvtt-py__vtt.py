"""VTT format module."""

import re
import typing
from dataclasses import dataclass

from . import utils
from .errors import MalformedFileError
from .models import Caption, Style


@dataclass
class ParserOutput:
    """Parsed WebVTT data."""

    styles: typing.List[Style]
    captions: typing.List[Caption]
    header_comments: typing.List[str]
    footer_comments: typing.List[str]

    @classmethod
    def from_data(
        cls,
        data: typing.Mapping[str, typing.Any]
    ) -> 'ParserOutput':
        entries = data.get('items', [])
        return cls(
            styles=[entry for entry in entries if isinstance(entry, Style)],
            captions=[entry for entry in entries if isinstance(entry, Caption)],
            header_comments=data.get('header_comments', []),
            footer_comments=data.get('footer_comments', [])
        )


class WebVTTCueBlock:
    """A block containing one WebVTT cue."""

    CUE_TIMINGS_PATTERN = re.compile(
        r'\s*((?:\d+:)?\d{2}:\d{2}.\d{3})\s*-->\s*((?:\d+:)?\d{2}:\d{2}.\d{3})'
    )

    def __init__(self, identifier, start, end, payload):
        self.identifier = identifier
        self.start = start
        self.end = end
        self.payload = payload

    @classmethod
    def is_valid(cls, lines: typing.Sequence[str]) -> bool:
        return bool(
            (
                len(lines) >= 2
                and re.match(cls.CUE_TIMINGS_PATTERN, lines[0])
                and '-->' not in lines[1]
            )
            or (
                len(lines) >= 3
                and '-->' not in lines[0]
                and re.match(cls.CUE_TIMINGS_PATTERN, lines[1])
                and '-->' not in lines[2]
            )
        )

    @classmethod
    def from_lines(
        cls,
        lines: typing.Iterable[str]
    ) -> 'WebVTTCueBlock':
        identifier = None
        start = None
        end = None
        payload = []

        for line in lines:
            match = re.match(cls.CUE_TIMINGS_PATTERN, line)
            if match:
                start = match.group(1)
                end = match.group(2)
            elif not start:
                identifier = line
            else:
                payload.append(line)

        return cls(identifier, start, end, payload)

    @staticmethod
    def format_lines(caption: Caption) -> typing.List[str]:
        return [
            '',
            *(identifier for identifier in {caption.identifier} if identifier),
            '{} --> {}'.format(caption.start, caption.end),
            *caption.lines
        ]


class WebVTTCommentBlock:
    """A block containing a WebVTT NOTE."""

    COMMENT_PATTERN = re.compile(r'NOTE\s(.*?)\Z', re.DOTALL)

    def __init__(self, text: str):
        self.text = text

    @classmethod
    def is_valid(cls, lines: typing.Sequence[str]) -> bool:
        return bool(lines and lines[0].startswith('NOTE'))

    @classmethod
    def from_lines(
        cls,
        lines: typing.Iterable[str]
    ) -> 'WebVTTCommentBlock':
        match = cls.COMMENT_PATTERN.match('\n'.join(lines))
        return cls(match.group(1).strip() if match else '')

    @staticmethod
    def format_lines(lines: str) -> typing.List[str]:
        parts = lines.split('\n')
        if len(parts) == 1:
            return ['NOTE {}'.format(lines)]
        return ['NOTE', *parts]


class WebVTTStyleBlock:
    """A block containing WebVTT CSS styles."""

    STYLE_PATTERN = re.compile(r'STYLE\s(.*?)\Z', re.DOTALL)

    def __init__(self, text: str):
        self.text = text

    @classmethod
    def is_valid(cls, lines: typing.Sequence[str]) -> bool:
        return (
            len(lines) >= 2
            and lines[0] == 'STYLE'
            and not any(line.strip() == '' or '-->' in line for line in lines)
        )

    @classmethod
    def from_lines(
        cls,
        lines: typing.Iterable[str]
    ) -> 'WebVTTStyleBlock':
        match = cls.STYLE_PATTERN.match('\n'.join(lines))
        return cls(match.group(1).strip() if match else '')

    @staticmethod
    def format_lines(lines: typing.List[str]) -> typing.List[str]:
        return ['STYLE', *lines]


def parse(lines: typing.Sequence[str]) -> ParserOutput:
    """Parse WebVTT content."""
    if not is_valid_content(lines):
        raise MalformedFileError('Invalid format')
    return parse_items(lines)


def is_valid_content(lines: typing.Sequence[str]) -> bool:
    """Return whether lines begin with a WebVTT header."""
    return bool(lines and lines[0].startswith('WEBVTT'))


def parse_items(lines: typing.Sequence[str]) -> ParserOutput:
    """Parse captions, styles, and comments from WebVTT lines."""
    header_comments: typing.List[str] = []
    items: typing.List[typing.Union[Caption, Style]] = []
    pending_comments: typing.List[WebVTTCommentBlock] = []

    for block in utils.iter_blocks_of_lines(lines):
        item = parse_item(block)
        if item:
            item.comments = [comment.text for comment in pending_comments]
            pending_comments = []
            items.append(item)
        elif WebVTTCommentBlock.is_valid(block):
            pending_comments.append(WebVTTCommentBlock.from_lines(block))

    if items:
        header_comments = items[0].comments
        items[0].comments = []

    return ParserOutput.from_data({
        'items': items,
        'header_comments': header_comments,
        'footer_comments': [comment.text for comment in pending_comments]
    })


def parse_item(
    lines: typing.Sequence[str]
) -> typing.Union[Caption, Style, None]:
    """Convert one block of lines into a caption or style object."""
    if WebVTTCueBlock.is_valid(lines):
        cue = WebVTTCueBlock.from_lines(lines)
        return Caption(cue.start, cue.end, cue.payload, cue.identifier)

    if WebVTTStyleBlock.is_valid(lines):
        return Style(WebVTTStyleBlock.from_lines(lines).text)

    return None


def write(
    f: typing.IO[str],
    captions: typing.Iterable[Caption],
    styles: typing.Iterable[Style],
    header_comments: typing.Iterable[str],
    footer_comments: typing.Iterable[str]
):
    """Write WebVTT data to a text stream."""
    f.write('WEBVTT\n')

    for comment in header_comments:
        f.write('\n')
        f.write('\n'.join(WebVTTCommentBlock.format_lines(comment)))
        f.write('\n')

    for style in styles:
        for comment in style.comments:
            f.write('\n')
            f.write('\n'.join(WebVTTCommentBlock.format_lines(comment)))
            f.write('\n')

        f.write('\n')
        f.write('\n'.join(WebVTTStyleBlock.format_lines(style.lines)))
        f.write('\n')

    for caption in captions:
        for comment in caption.comments:
            f.write('\n')
            f.write('\n'.join(WebVTTCommentBlock.format_lines(comment)))
            f.write('\n')

        f.write('\n'.join(WebVTTCueBlock.format_lines(caption)))
        f.write('\n')

    for comment in footer_comments:
        f.write('\n')
        f.write('\n'.join(WebVTTCommentBlock.format_lines(comment)))
        f.write('\n')