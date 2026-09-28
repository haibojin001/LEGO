from itertools import chain
from pathlib import Path

from qrcode.compat.png import PngWriter
from qrcode.image.base import BaseImage


class PyPNGImage(BaseImage):
    kind = "PNG"
    allowed_kinds = ("PNG",)
    needs_drawrect = False

    def new_image(self, **kwargs):
        if not PngWriter:
            raise ImportError("PyPNG library not installed.")
        return PngWriter(
            self.pixel_size,
            self.pixel_size,
            greyscale=True,
            bitdepth=1,
        )

    def drawrect(self, row, col):
        pass

    def save(self, stream, kind=None):
        if isinstance(stream, str):
            stream = Path(stream).open("wb")
        self._img.write(stream, self.rows_iter())

    def rows_iter(self):
        yield from self.border_rows_iter()

        side_border = [1] * (self.box_size * self.border)
        for modules in self.modules:
            pixels = (
                side_border
                + list(
                    chain.from_iterable(
                        [not value] * self.box_size for value in modules
                    )
                )
                + side_border
            )
            for _ in range(self.box_size):
                yield pixels

        yield from self.border_rows_iter()

    def border_rows_iter(self):
        pixels = [1] * (self.box_size * (self.width + 2 * self.border))
        for _ in range(self.box_size * self.border):
            yield pixels


PymagingImage = PyPNGImage