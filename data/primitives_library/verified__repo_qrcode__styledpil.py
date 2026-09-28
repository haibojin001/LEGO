from __future__ import annotations

import warnings
from typing import overload

import deprecation
from PIL import Image

import qrcode.image.base
from qrcode.image.styles.colormasks import QRColorMask, SolidFillColorMask
from qrcode.image.styles.moduledrawers.pil import SquareModuleDrawer


class StyledPilImage(qrcode.image.base.BaseImageWithDrawer):
    """
    PIL-based styled QR image builder.

    Supports module drawers, color masks, and an optional centered embedded
    image. Images are saved as PNG by default.
    """

    kind = "PNG"
    needs_processing = True
    color_mask: QRColorMask
    default_drawer_class = SquareModuleDrawer

    def __init__(self, *args, **kwargs):
        self.color_mask = kwargs.get("color_mask", SolidFillColorMask())

        if kwargs.get("embeded_image_path") or kwargs.get("embeded_image"):
            warnings.warn(
                "The 'embeded_*' parameters are deprecated. Use "
                "'embedded_image_path' or 'embedded_image' instead. The "
                "'embeded_*' parameters will be removed in v9.0.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        image_path = kwargs.get(
            "embedded_image_path",
            kwargs.get("embeded_image_path"),
        )
        self.embedded_image = kwargs.get(
            "embedded_image",
            kwargs.get("embeded_image"),
        )
        self.embedded_image_ratio = kwargs.get(
            "embedded_image_ratio",
            kwargs.get("embeded_image_ratio", 0.25),
        )
        self.embedded_image_resample = kwargs.get(
            "embedded_image_resample",
            kwargs.get("embeded_image_resample", Image.Resampling.LANCZOS),
        )

        if not self.embedded_image and image_path:
            self.embedded_image = Image.open(image_path)

        self.paint_color = tuple(0 for _ in self.color_mask.back_color)
        if self.color_mask.has_transparency:
            self.paint_color = (*self.color_mask.back_color[:3], 255)

        super().__init__(*args, **kwargs)

    @overload
    def drawrect(self, row, col):
        """Not used."""

    def new_image(self, **kwargs):
        mode = (
            "RGBA"
            if self.color_mask.has_transparency
            or (
                self.embedded_image
                and "A" in self.embedded_image.getbands()
            )
            else "RGB"
        )
        return Image.new(
            mode,
            (self.pixel_size, self.pixel_size),
            self.color_mask.back_color,
        )

    def init_new_image(self):
        self.color_mask.initialize(self, self._img)
        super().init_new_image()

    def process(self):
        self.color_mask.apply_mask(self._img)
        if self.embedded_image:
            self.draw_embedded_image()

    @deprecation.deprecated(
        deprecated_in="9.0",
        removed_in="8.3",
        current_version="8.2",
        details="Use draw_embedded_image() instead",
    )
    def draw_embeded_image(self):
        return self.draw_embedded_image()

    def draw_embedded_image(self):
        if not self.embedded_image:
            return

        image_width, _ = self._img.size
        image_width = int(image_width)
        approximate_logo_width = int(image_width * self.embedded_image_ratio)
        offset = (
            int(
                (
                    int(image_width / 2)
                    - int(approximate_logo_width / 2)
                )
                / self.box_size
            )
            * self.box_size
        )
        position = (offset, offset)
        logo_width = image_width - offset * 2

        logo = self.embedded_image.resize(
            (logo_width, logo_width),
            self.embedded_image_resample,
        )

        if "A" in logo.getbands():
            self._img.alpha_composite(logo, position)
        else:
            self._img.paste(logo, position)

    def save(self, stream, format=None, **kwargs):
        if format is None:
            format = kwargs.get("kind", self.kind)
        kwargs.pop("kind", None)
        self._img.save(stream, format=format, **kwargs)

    def __getattr__(self, name):
        return getattr(self._img, name)