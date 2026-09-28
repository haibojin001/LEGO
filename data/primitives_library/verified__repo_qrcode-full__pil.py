from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from qrcode.image.styles.moduledrawers.base import QRModuleDrawer

if TYPE_CHECKING:
    from qrcode.image.styledpil import StyledPilImage
    from qrcode.main import ActiveWithNeighbors


ANTIALIASING_FACTOR = 4


class StyledPilQRModuleDrawer(QRModuleDrawer):
    """
    Base drawer used by StyledPilImage module drawer implementations.
    """

    img: "StyledPilImage"


class SquareModuleDrawer(StyledPilQRModuleDrawer):
    """
    Renders active modules as filled square cells.
    """

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.imgDraw = ImageDraw.Draw(self.img._img)

    def drawrect(self, box, is_active: bool):
        if is_active:
            self.imgDraw.rectangle(box, fill=self.img.paint_color)


class GappedSquareModuleDrawer(StyledPilQRModuleDrawer):
    """
    Renders active modules as separated square cells.
    """

    def __init__(self, size_ratio=0.8):
        self.size_ratio = size_ratio

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.imgDraw = ImageDraw.Draw(self.img._img)
        self.delta = (1 - self.size_ratio) * self.img.box_size / 2

    def drawrect(self, box, is_active: bool):
        if is_active:
            x1, y1 = box[0]
            x2, y2 = box[1]
            inset = self.delta
            self.imgDraw.rectangle(
                (x1 + inset, y1 + inset, x2 - inset, y2 - inset),
                fill=self.img.paint_color,
            )


class CircleModuleDrawer(StyledPilQRModuleDrawer):
    """
    Renders active modules as antialiased circles.
    """

    circle = None

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        module_size = self.img.box_size
        enlarged_size = module_size * ANTIALIASING_FACTOR
        circle = Image.new(
            self.img.mode,
            (enlarged_size, enlarged_size),
            self.img.color_mask.back_color,
        )
        ImageDraw.Draw(circle).ellipse(
            (0, 0, enlarged_size, enlarged_size),
            fill=self.img.paint_color,
        )
        self.circle = circle.resize(
            (module_size, module_size),
            Image.Resampling.LANCZOS,
        )

    def drawrect(self, box, is_active: bool):
        if is_active:
            self.img._img.paste(self.circle, box[0])


class GappedCircleModuleDrawer(StyledPilQRModuleDrawer):
    """
    Renders active modules as separated antialiased circles.
    """

    circle = None

    def __init__(self, size_ratio=0.9):
        self.size_ratio = size_ratio

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        module_size = self.img.box_size
        enlarged_size = module_size * ANTIALIASING_FACTOR
        circle = Image.new(
            self.img.mode,
            (enlarged_size, enlarged_size),
            self.img.color_mask.back_color,
        )
        ImageDraw.Draw(circle).ellipse(
            (0, 0, enlarged_size, enlarged_size),
            fill=self.img.paint_color,
        )
        reduced_size = int(self.size_ratio * module_size)
        self.circle = circle.resize(
            (reduced_size, reduced_size),
            Image.Resampling.LANCZOS,
        )

    def drawrect(self, box, is_active: bool):
        if is_active:
            self.img._img.paste(self.circle, box[0])


class RoundedModuleDrawer(StyledPilQRModuleDrawer):
    """
    Renders modules with corners rounded where they border empty cells.
    """

    needs_neighbors = True

    def __init__(self, radius_ratio=1):
        self.radius_ratio = radius_ratio

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.corner_width = int(self.img.box_size / 2)
        self.setup_corners()

    def setup_corners(self):
        color_mode = self.img.mode
        foreground = self.img.paint_color
        background = self.img.color_mask.back_color
        corner_size = self.corner_width

        self.SQUARE = Image.new(
            color_mode,
            (corner_size, corner_size),
            foreground,
        )

        scaled_size = corner_size * ANTIALIASING_FACTOR
        radius = self.radius_ratio * scaled_size
        diameter = radius * 2
        corner = Image.new(
            color_mode,
            (scaled_size, scaled_size),
            background,
        )
        painter = ImageDraw.Draw(corner)
        painter.ellipse((0, 0, diameter, diameter), fill=foreground)
        painter.rectangle((radius, 0, scaled_size, scaled_size), fill=foreground)
        painter.rectangle((0, radius, scaled_size, scaled_size), fill=foreground)

        self.NW_ROUND = corner.resize(
            (corner_size, corner_size),
            Image.Resampling.LANCZOS,
        )
        self.SW_ROUND = self.NW_ROUND.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        self.SE_ROUND = self.NW_ROUND.transpose(Image.Transpose.ROTATE_180)
        self.NE_ROUND = self.NW_ROUND.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    def drawrect(self, box: list[list[int]], is_active: "ActiveWithNeighbors"):
        if not is_active:
            return

        north_west = self.NW_ROUND if not is_active.W and not is_active.N else self.SQUARE
        north_east = self.NE_ROUND if not is_active.N and not is_active.E else self.SQUARE
        south_east = self.SE_ROUND if not is_active.E and not is_active.S else self.SQUARE
        south_west = self.SW_ROUND if not is_active.S and not is_active.W else self.SQUARE

        x, y = box[0]
        half = self.corner_width
        self.img._img.paste(north_west, (x, y))
        self.img._img.paste(north_east, (x + half, y))
        self.img._img.paste(south_east, (x + half, y + half))
        self.img._img.paste(south_west, (x, y + half))


class VerticalBarsDrawer(StyledPilQRModuleDrawer):
    """
    Renders vertical runs as rounded bars.
    """

    needs_neighbors = True

    def __init__(self, horizontal_shrink=0.8):
        self.horizontal_shrink = horizontal_shrink

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.half_height = int(self.img.box_size / 2)
        self.delta = int((1 - self.horizontal_shrink) * self.half_height)
        self.setup_edges()

    def setup_edges(self):
        color_mode = self.img.mode
        foreground = self.img.paint_color
        background = self.img.color_mask.back_color

        half_height = self.half_height
        full_width = half_height * 2
        shrunk_width = int(full_width * self.horizontal_shrink)

        self.SQUARE = Image.new(
            color_mode,
            (shrunk_width, half_height),
            foreground,
        )

        scaled_width = full_width * ANTIALIASING_FACTOR
        scaled_height = half_height * ANTIALIASING_FACTOR
        edge = Image.new(
            color_mode,
            (scaled_width, scaled_height),
            background,
        )
        ImageDraw.Draw(edge).ellipse(
            (0, 0, scaled_width, scaled_height * 2),
            fill=foreground,
        )

        self.ROUND_TOP = edge.resize(
            (shrunk_width, half_height),
            Image.Resampling.LANCZOS,
        )
        self.ROUND_BOTTOM = self.ROUND_TOP.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def drawrect(self, box, is_active: "ActiveWithNeighbors"):
        if is_active:
            top = self.ROUND_TOP if not is_active.N else self.SQUARE
            bottom = self.ROUND_BOTTOM if not is_active.S else self.SQUARE
            x, y = box[0]
            self.img._img.paste(top, (x + self.delta, y))
            self.img._img.paste(
                bottom,
                (x + self.delta, y + self.half_height),
            )


class HorizontalBarsDrawer(StyledPilQRModuleDrawer):
    """
    Renders horizontal runs as rounded bars.
    """

    needs_neighbors = True

    def __init__(self, vertical_shrink=0.8):
        self.vertical_shrink = vertical_shrink

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.half_width = int(self.img.box_size / 2)
        self.delta = int((1 - self.vertical_shrink) * self.half_width)
        self.setup_edges()

    def setup_edges(self):
        color_mode = self.img.mode
        foreground = self.img.paint_color
        background = self.img.color_mask.back_color

        half_width = self.half_width
        full_height = half_width * 2
        shrunk_height = int(full_height * self.vertical_shrink)

        self.SQUARE = Image.new(
            color_mode,
            (half_width, shrunk_height),
            foreground,
        )

        scaled_width = half_width * ANTIALIASING_FACTOR
        scaled_height = full_height * ANTIALIASING_FACTOR
        edge = Image.new(
            color_mode,
            (scaled_width, scaled_height),
            background,
        )
        ImageDraw.Draw(edge).ellipse(
            (0, 0, scaled_width * 2, scaled_height),
            fill=foreground,
        )

        self.ROUND_LEFT = edge.resize(
            (half_width, shrunk_height),
            Image.Resampling.LANCZOS,
        )
        self.ROUND_RIGHT = self.ROUND_LEFT.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    def drawrect(self, box, is_active: "ActiveWithNeighbors"):
        if is_active:
            left = self.ROUND_LEFT if not is_active.W else self.SQUARE
            right = self.ROUND_RIGHT if not is_active.E else self.SQUARE
            x, y = box[0]
            self.img._img.paste(left, (x, y + self.delta))
            self.img._img.paste(
                right,
                (x + self.half_width, y + self.delta),
            )