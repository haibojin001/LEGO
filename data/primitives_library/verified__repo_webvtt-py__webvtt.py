import io
import os
import typing
import warnings
from functools import partial

from . import sbv, srt, utils, vtt
from .errors import MissingFilenameError
from .models import Caption, Style, Timestamp

DEFAULT_ENCODING = "utf-8"


class WebVTT:
    def __init__(
        self,
        file: typing.Optional[str] = None,
        captions: typing.Optional[typing.List[Caption]] = None,
        styles: typing.Optional[typing.List[Style]] = None,
        header_comments: typing.Optional[typing.List[str]] = None,
        footer_comments: typing.Optional[typing.List[str]] = None,
    ):
        self.file = file
        self.captions = captions or []
        self.styles = styles or []
        self.header_comments = header_comments or []
        self.footer_comments = footer_comments or []
        self._has_bom = False
        self.encoding = DEFAULT_ENCODING

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return self.captions[index]

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} file={self.file!r} "
            f"encoding={self.encoding!r}>"
        )

    def __str__(self):
        return "\n".join(str(caption) for caption in self.captions)

    @classmethod
    def read(
        cls,
        file: str,
        encoding: typing.Optional[str] = None,
    ) -> "WebVTT":
        with utils.FileWrapper.open(file, encoding=encoding) as wrapped_file:
            result = cls.from_buffer(wrapped_file.file)
            if wrapped_file.bom_encoding:
                result.encoding = wrapped_file.bom_encoding
                result._has_bom = True
            return result

    @classmethod
    def read_buffer(
        cls,
        buffer: typing.Iterator[str],
    ) -> "WebVTT":
        warnings.warn(
            "Deprecated: use from_buffer instead.",
            DeprecationWarning,
        )
        return cls.from_buffer(buffer)

    @classmethod
    def from_buffer(
        cls,
        buffer: typing.Union[typing.Iterable[str], io.BytesIO],
        format: str = "vtt",
    ) -> "WebVTT":
        if isinstance(buffer, io.BytesIO):
            buffer = (item.decode("utf-8") for item in buffer)

        construct = partial(cls, file=getattr(buffer, "name", None))
        lines = cls._get_lines(buffer)

        if format == "vtt":
            parsed = vtt.parse(lines)
            return construct(
                captions=parsed.captions,
                styles=parsed.styles,
                header_comments=parsed.header_comments,
                footer_comments=parsed.footer_comments,
            )

        if format == "srt":
            return construct(captions=srt.parse(lines))

        if format == "sbv":
            return construct(captions=sbv.parse(lines))

        raise ValueError(f"Format {format} is not supported.")

    @classmethod
    def from_srt(
        cls,
        file: str,
        encoding: typing.Optional[str] = None,
    ) -> "WebVTT":
        with utils.FileWrapper.open(file, encoding=encoding) as wrapped_file:
            return cls(
                file=wrapped_file.file.name,
                captions=srt.parse(cls._get_lines(wrapped_file.file)),
            )

    @classmethod
    def from_sbv(
        cls,
        file: str,
        encoding: typing.Optional[str] = None,
    ) -> "WebVTT":
        with utils.FileWrapper.open(file, encoding=encoding) as wrapped_file:
            return cls(
                file=wrapped_file.file.name,
                captions=sbv.parse(cls._get_lines(wrapped_file.file)),
            )

    @classmethod
    def from_string(cls, string: str) -> "WebVTT":
        parsed = vtt.parse(cls._get_lines(string.splitlines()))
        return cls(
            captions=parsed.captions,
            styles=parsed.styles,
            header_comments=parsed.header_comments,
            footer_comments=parsed.footer_comments,
        )

    @staticmethod
    def _get_lines(lines: typing.Iterable[str]) -> typing.List[str]:
        return [line.rstrip("\n\r") for line in lines]

    def _get_destination_file(
        self,
        destination_path: typing.Optional[str] = None,
        extension: str = "vtt",
    ) -> str:
        if not destination_path and not self.file:
            raise MissingFilenameError

        if not destination_path and self.file:
            destination_path = f"{os.path.splitext(self.file)[0]}.{extension}"

        assert destination_path is not None

        destination = os.path.join(os.getcwd(), destination_path)

        if os.path.isdir(destination):
            if not self.file:
                raise MissingFilenameError

            filename = os.path.splitext(os.path.basename(self.file))[0]
            return os.path.join(destination, f"{filename}.{extension}")

        if destination[-4:].lower() != f".{extension}":
            destination = f"{destination}.{extension}"

        return destination

    def save(
        self,
        output: typing.Optional[str] = None,
        encoding: typing.Optional[str] = None,
        add_bom: typing.Optional[bool] = None,
    ):
        self.file = self._get_destination_file(output)
        selected_encoding = encoding or self.encoding

        if add_bom is None and self._has_bom:
            add_bom = True

        with open(self.file, "w", encoding=selected_encoding) as file:
            if add_bom and selected_encoding in utils.CODEC_BOMS:
                file.write(utils.CODEC_BOMS[selected_encoding].decode(selected_encoding))

            vtt.write(
                file,
                self.captions,
                self.styles,
                self.header_comments,
                self.footer_comments,
            )

    def save_as_srt(
        self,
        output: typing.Optional[str] = None,
        encoding: typing.Optional[str] = None,
    ):
        self.file = self._get_destination_file(output, extension="srt")
        selected_encoding = encoding or self.encoding

        with open(self.file, "w", encoding=selected_encoding) as file:
            srt.write(file, self.captions)

    def save_as_sbv(
        self,
        output: typing.Optional[str] = None,
        encoding: typing.Optional[str] = None,
    ):
        self.file = self._get_destination_file(output, extension="sbv")
        selected_encoding = encoding or self.encoding

        with open(self.file, "w", encoding=selected_encoding) as file:
            sbv.write(file, self.captions)