import codecs
import typing

CODEC_BOMS = {
    'utf-8': codecs.BOM_UTF8,
    'utf-32-le': codecs.BOM_UTF32_LE,
    'utf-32-be': codecs.BOM_UTF32_BE,
    'utf-16-le': codecs.BOM_UTF16_LE,
    'utf-16-be': codecs.BOM_UTF16_BE,
}


class FileWrapper:
    """File handling functionality with built-in support for Byte OrderMark."""

    def __init__(
        self,
        file_path: str,
        mode: typing.Optional[str] = None,
        encoding: typing.Optional[str] = None
    ):
        self.file_path = file_path
        self.mode = mode if mode is not None else 'r'
        self.bom_encoding = self.detect_bom_encoding(file_path)
        self.encoding = self.bom_encoding or encoding or 'utf-8'

    @classmethod
    def open(
        cls,
        file_path: str,
        mode: typing.Optional[str] = None,
        encoding: typing.Optional[str] = None
    ) -> 'FileWrapper':
        return cls(file_path, mode, encoding)

    def __enter__(self):
        self.file = open(
            file=self.file_path,
            mode=self.mode,
            encoding=self.encoding
        )
        if self.bom_encoding:
            self.file.seek(len(CODEC_BOMS[self.bom_encoding]))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    @staticmethod
    def detect_bom_encoding(file_path: str) -> typing.Optional[str]:
        with open(file_path, mode='rb') as file_object:
            prefix = file_object.read(4)
            for encoding, marker in CODEC_BOMS.items():
                if prefix.startswith(marker):
                    return encoding
        return None


def iter_blocks_of_lines(
    lines: typing.Iterable[str]
) -> typing.Generator[typing.List[str], None, None]:
    block = []
    for line in lines:
        if line.strip():
            block.append(line)
        elif block:
            yield block
            block = []
    if block:
        yield block