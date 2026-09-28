"""Models used to represent WebVTT timestamps, captions, and styles."""

import re
import typing

from .errors import MalformedCaptionError


class Timestamp:
    """A WebVTT timestamp."""

    PATTERN = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{1,2})\.(\d{3})")

    def __init__(
        self,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        milliseconds: int = 0,
    ):
        """Create a timestamp."""
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.milliseconds = milliseconds

    def __str__(self):
        """Format this timestamp as a WebVTT time string."""
        return (
            f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}"
            f".{self.milliseconds:03d}"
        )

    def to_tuple(self) -> typing.Tuple[int, int, int, int]:
        """Return timestamp components as a tuple."""
        return (self.hours, self.minutes, self.seconds, self.milliseconds)

    def __repr__(self):
        """Return a debug representation."""
        return (
            f"<{self.__class__.__name__} "
            f"hours={self.hours} "
            f"minutes={self.minutes} "
            f"seconds={self.seconds} "
            f"milliseconds={self.milliseconds}>"
        )

    def __eq__(self, other):
        """Compare timestamps for equality."""
        return self.to_tuple() == other.to_tuple()

    def __ne__(self, other):
        """Compare timestamps for inequality."""
        return self.to_tuple() != other.to_tuple()

    def __gt__(self, other):
        """Return whether this timestamp is later than another."""
        return self.to_tuple() > other.to_tuple()

    def __lt__(self, other):
        """Return whether this timestamp is earlier than another."""
        return self.to_tuple() < other.to_tuple()

    def __ge__(self, other):
        """Return whether this timestamp is no earlier than another."""
        return self.to_tuple() >= other.to_tuple()

    def __le__(self, other):
        """Return whether this timestamp is no later than another."""
        return self.to_tuple() <= other.to_tuple()

    @classmethod
    def from_string(cls, value: str) -> "Timestamp":
        """Parse a timestamp string."""
        if type(value) is not str:
            raise MalformedCaptionError(f"Invalid timestamp {value!r}")

        matched = re.match(cls.PATTERN, value)
        if not matched:
            raise MalformedCaptionError(f"Invalid timestamp {value!r}")

        hours = int(matched.group(1) or 0)
        minutes = int(matched.group(2))
        seconds = int(matched.group(3))
        milliseconds = int(matched.group(4))

        if minutes > 59 or seconds > 59:
            raise MalformedCaptionError(f"Invalid timestamp {value!r}")

        return cls(hours, minutes, seconds, milliseconds)

    def in_seconds(self) -> int:
        """Return the whole-second value of this timestamp."""
        return self.hours * 3600 + self.minutes * 60 + self.seconds


class Caption:
    """A WebVTT caption cue."""

    CUE_TEXT_TAGS = re.compile("<.*?>")
    VOICE_SPAN_PATTERN = re.compile(r"<v(?:\.\w+)*\s+([^>]+)>")

    def __init__(
        self,
        start: typing.Optional[str] = None,
        end: typing.Optional[str] = None,
        text: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
        identifier: typing.Optional[str] = None,
    ):
        """Create a caption."""
        text = text or []
        self.start = start or "00:00:00.000"
        self.end = end or "00:00:00.000"
        self.identifier = identifier
        self.lines = text.splitlines() if isinstance(text, str) else list(text)
        self.comments: typing.List[str] = []

    def __repr__(self):
        """Return a debug representation."""
        visible_text = self.text.replace("\n", "\\n")
        return (
            f"<{self.__class__.__name__} "
            f"start={self.start!r} "
            f"end={self.end!r} "
            f"text={visible_text!r} "
            f"identifier={self.identifier!r}>"
        )

    def __str__(self):
        """Return a concise human-readable representation."""
        visible_text = self.text.replace("\n", "\\n")
        return f"{self.start} {self.end} {visible_text}"

    def __eq__(self, other):
        """Compare caption contents."""
        if not isinstance(other, type(self)):
            return False

        return (
            self.start == other.start
            and self.end == other.end
            and self.raw_text == other.raw_text
            and self.identifier == other.identifier
        )

    @property
    def start(self):
        """Return the cue start time."""
        return str(self.start_time)

    @start.setter
    def start(self, value: str):
        """Set the cue start time."""
        self.start_time = Timestamp.from_string(value)

    @property
    def end(self):
        """Return the cue end time."""
        return str(self.end_time)

    @end.setter
    def end(self, value: str):
        """Set the cue end time."""
        self.end_time = Timestamp.from_string(value)

    @property
    def start_in_seconds(self) -> int:
        """Return the cue start time in seconds."""
        return self.start_time.in_seconds()

    @property
    def end_in_seconds(self):
        """Return the cue end time in seconds."""
        return self.end_time.in_seconds()

    @property
    def raw_text(self) -> str:
        """Return cue text without removing markup."""
        return "\n".join(self.lines)

    @property
    def text(self) -> str:
        """Return cue text with cue markup removed."""
        return re.sub(self.CUE_TEXT_TAGS, "", self.raw_text)

    @text.setter
    def text(self, value: str):
        """Replace cue text."""
        if not isinstance(value, str):
            raise AttributeError(f"String value expected but received {value}.")

        self.lines = value.splitlines()

    @property
    def voice(self) -> typing.Optional[str]:
        """Return the voice name declared on the first text line, if any."""
        if self.lines and self.lines[0].startswith("<v"):
            matched = re.match(self.VOICE_SPAN_PATTERN, self.lines[0])
            if matched:
                return matched.group(1)
        return None


class Style:
    """A WebVTT STYLE block."""

    def __init__(self, text: typing.Union[str, typing.List[str]]):
        """Create a style block."""
        self.lines = text.splitlines() if isinstance(text, str) else text
        self.comments: typing.List[str] = []

    @property
    def text(self):
        """Return style contents as a single string."""
        return "\n".join(self.lines)