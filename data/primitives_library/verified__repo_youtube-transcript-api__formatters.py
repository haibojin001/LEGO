import json
import pprint
from typing import Iterable, List

from ._transcripts import FetchedTranscript, FetchedTranscriptSnippet


class Formatter:
    """Formatter should be used as an abstract base class."""

    def format_transcript(self, transcript: FetchedTranscript, **kwargs) -> str:
        raise NotImplementedError(
            "A subclass of Formatter must implement "
            "their own .format_transcript() method."
        )

    def format_transcripts(self, transcripts: List[FetchedTranscript], **kwargs):
        raise NotImplementedError(
            "A subclass of Formatter must implement "
            "their own .format_transcripts() method."
        )


class PrettyPrintFormatter(Formatter):
    def format_transcript(self, transcript: FetchedTranscript, **kwargs) -> str:
        return pprint.pformat(transcript.to_raw_data(), **kwargs)

    def format_transcripts(self, transcripts: List[FetchedTranscript], **kwargs) -> str:
        return pprint.pformat(
            [transcript.to_raw_data() for transcript in transcripts], **kwargs
        )


class JSONFormatter(Formatter):
    def format_transcript(self, transcript: FetchedTranscript, **kwargs) -> str:
        return json.dumps(transcript.to_raw_data(), **kwargs)

    def format_transcripts(self, transcripts: List[FetchedTranscript], **kwargs) -> str:
        return json.dumps(
            [transcript.to_raw_data() for transcript in transcripts], **kwargs
        )


class TextFormatter(Formatter):
    def format_transcript(self, transcript: FetchedTranscript, **kwargs) -> str:
        return "\n".join(line.text for line in transcript)

    def format_transcripts(self, transcripts: List[FetchedTranscript], **kwargs) -> str:
        return "\n\n\n".join(
            [self.format_transcript(transcript, **kwargs) for transcript in transcripts]
        )


class _TextBasedFormatter(TextFormatter):
    def _format_timestamp(self, hours: int, mins: int, secs: int, ms: int) -> str:
        raise NotImplementedError(
            "A subclass of _TextBasedFormatter must implement "
            "their own .format_timestamp() method."
        )

    def _format_transcript_header(self, lines: Iterable[str]) -> str:
        raise NotImplementedError(
            "A subclass of _TextBasedFormatter must implement "
            "their own _format_transcript_header method."
        )

    def _format_transcript_helper(
        self, i: int, time_text: str, snippet: FetchedTranscriptSnippet
    ) -> str:
        raise NotImplementedError(
            "A subclass of _TextBasedFormatter must implement "
            "their own _format_transcript_helper method."
        )

    def _seconds_to_timestamp(self, time: float) -> str:
        time = float(time)
        hours_float, remainder = divmod(time, 3600)
        mins_float, secs_float = divmod(remainder, 60)
        hours, mins, secs = int(hours_float), int(mins_float), int(secs_float)
        ms = int(round((time - int(time)) * 1000, 2))
        return self._format_timestamp(hours, mins, secs, ms)

    def format_transcript(self, transcript: FetchedTranscript, **kwargs) -> str:
        lines = []
        for i, line in enumerate(transcript):
            end = line.start + line.duration
            time_text = "{} --> {}".format(
                self._seconds_to_timestamp(line.start),
                self._seconds_to_timestamp(
                    transcript[i + 1].start
                    if i < len(transcript) - 1 and transcript[i + 1].start < end
                    else end
                ),
            )
            lines.append(self._format_transcript_helper(i, time_text, line))
        return self._format_transcript_header(lines)


class SRTFormatter(_TextBasedFormatter):
    def _format_timestamp(self, hours: int, mins: int, secs: int, ms: int) -> str:
        return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, mins, secs, ms)

    def _format_transcript_header(self, lines: Iterable[str]) -> str:
        return "\n\n".join(lines) + "\n"

    def _format_transcript_helper(
        self, i: int, time_text: str, snippet: FetchedTranscriptSnippet
    ) -> str:
        return "{}\n{}\n{}".format(i + 1, time_text, snippet.text)


class WebVTTFormatter(_TextBasedFormatter):
    def _format_timestamp(self, hours: int, mins: int, secs: int, ms: int) -> str:
        return "{:02d}:{:02d}:{:02d}.{:03d}".format(hours, mins, secs, ms)

    def _format_transcript_header(self, lines: Iterable[str]) -> str:
        return "WEBVTT\n\n" + "\n\n".join(lines) + "\n"

    def _format_transcript_helper(
        self, i: int, time_text: str, snippet: FetchedTranscriptSnippet
    ) -> str:
        return "{}\n{}".format(time_text, snippet.text)


class FormatterLoader:
    TYPES = {
        "json": JSONFormatter,
        "pretty": PrettyPrintFormatter,
        "text": TextFormatter,
        "webvtt": WebVTTFormatter,
        "srt": SRTFormatter,
    }

    class UnknownFormatterType(Exception):
        def __init__(self, formatter_type: str):
            super().__init__(
                "The format '{formatter_type}' is not supported. "
                "Choose one of the following formats: {supported_formatter_types}".format(
                    formatter_type=formatter_type,
                    supported_formatter_types=", ".join(FormatterLoader.TYPES.keys()),
                )
            )

    def load(self, formatter_type: str = "pretty") -> Formatter:
        if formatter_type not in FormatterLoader.TYPES.keys():
            raise FormatterLoader.UnknownFormatterType(formatter_type)
        return FormatterLoader.TYPES[formatter_type]()