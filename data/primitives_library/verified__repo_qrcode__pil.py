from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from qrcode.image.styles.moduledrawers.base import QRModuleDrawer

if TYPE_CHECKING:
    from qrcode.image.styledpil import StyledPilImage
    from qrcode.main import ActiveWithNeighbors


ANTIALIASING_FACTOR = 4


class StyledPilQRModuleDrawer(QRModuleDrawer):
    """
    Base drawer implementation for QR modules rendered by StyledPilImage.

    Drawers use the image's paint color; color masks subsequently transform
    those values into their final colors.
    """

    img: "StyledPilImage"


class SquareModuleDrawer(StyledPilQRModuleDrawer):
    """
    Draw QR modules as solid square cells.
    """

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.imgDraw = ImageDraw.Draw(self.img._img)

    def drawrect(self, box, is_active: bool):
        if is_active:
            self.imgDraw.rectangle(box, fill=self.img.paint_color)


class GappedSquareModuleDrawer(StyledPilQRModuleDrawer):
    """
    Draw square modules separated by space.

    size_ratio is the fraction of each module cell occupied by the square.
    """

    def __init__(self, size_ratio=0.8):
        self.size_ratio = size_ratio

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.imgDraw = ImageDraw.Draw(self.img._img)
        self.delta = (1 - self.size_ratio) * self.img.box_size / 2

    def drawrect(self, box, is_active: bool):
        if is_active:
            inset_box = (
                box[0][0] + self.delta,
                box[0][1] + self.delta,
                box[1][0] - self.delta,
                box[1][1] - self.delta,
            )
            self.imgDraw.rectangle(inset_box, fill=self.img.paint_color)


class CircleModuleDrawer(StyledPilQRModuleDrawer):
    """
    Draw QR modules as circles.
    """

    circle = None

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        size = self.img.box_size
        enlarged = size * ANTIALIASING_FACTOR
        circle = Image.new(
            self.img.mode,
            (enlarged, enlarged),
            self.img.color_mask.back_color,
        )
        ImageDraw.Draw(circle).ellipse(
            (0, 0, enlarged, enlarged),
            fill=self.img.paint_color,
        )
        self.circle = circle.resize((size, size), Image.Resampling.LANCZOS)

    def drawrect(self, box, is_active: bool):
        if is_active:
            self.img._img.paste(self.circle, (box[0][0], box[0][1]))


class GappedCircleModuleDrawer(StyledPilQRModuleDrawer):
    """
    Draw circular modules separated by space.

    size_ratio is the fraction of each module cell occupied by the circle.
    """

    circle = None

    def __init__(self, size_ratio=0.9):
        self.size_ratio = size_ratio

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        size = self.img.box_size
        enlarged = size * ANTIALIASING_FACTOR
        circle = Image.new(
            self.img.mode,
            (enlarged, enlarged),
            self.img.color_mask.back_color,
        )
        ImageDraw.Draw(circle).ellipse(
            (0, 0, enlarged, enlarged),
            fill=self.img.paint_color,
        )
        reduced_size = int(self.size_ratio * size)
        self.circle = circle.resize(
            (reduced_size, reduced_size),
            Image.Resampling.LANCZOS,
        )

    def drawrect(self, box, is_active: bool):
        if is_active:
            self.img._img.paste(self.circle, (box[0][0], box[0][1]))


class RoundedModuleDrawer(StyledPilQRModuleDrawer):
    """
    Draw modules with independently rounded outer corners.

    radius_ratio controls the corner roundness. A value of one makes isolated
    cells circular, while zero leaves square corners.
    """

    needs_neighbors = True

    def __init__(self, radius_ratio=1):
        self.radius_ratio = radius_ratio

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.corner_width = int(self.img.box_size / 2)
        self.setup_corners()

    def setup_corners(self):
        mode = self.img.mode
        background = self.img.color_mask.back_color
        foreground = self.img.paint_color

        self.SQUARE = Image.new(
            mode,
            (self.corner_width, self.corner_width),
            foreground,
        )

        enlarged_size = self.corner_width * ANTIALIASING_FACTOR
        radius = self.radius_ratio * enlarged_size
        diameter = radius * 2

        corner = Image.new(
            mode,
            (enlarged_size, enlarged_size),
            background,
        )
        drawer = ImageDraw.Draw(corner)
        drawer.ellipse((0, 0, diameter, diameter), fill=foreground)
        drawer.rectangle(
            (radius, 0, enlarged_size, enlarged_size),
            fill=foreground,
        )
        drawer.rectangle(
            (0, radius, enlarged_size, enlarged_size),
            fill=foreground,
        )

        self.NW_ROUND = corner.resize(
            (self.corner_width, self.corner_width),
            Image.Resampling.LANCZOS,
        )
        self.SW_ROUND = self.NW_ROUND.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        self.SE_ROUND = self.NW_ROUND.transpose(Image.Transpose.ROTATE_180)
        self.NE_ROUND = self.NW_ROUND.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    def drawrect(self, box: list[list[int]], is_active: "ActiveWithNeighbors"):
        if not is_active:
            return

        northwest = not is_active.W and not is_active.N
        northeast = not is_active.N and not is_active.E
        southeast = not is_active.E and not is_active.S
        southwest = not is_active.S and not is_active.W

        nw = self.NW_ROUND if northwest else self.SQUARE
        ne = self.NE_ROUND if northeast else self.SQUARE
        se = self.SE_ROUND if southeast else self.SQUARE
        sw = self.SW_ROUND if southwest else self.SQUARE

        x, y = box[0]
        self.img._img.paste(nw, (x, y))
        self.img._img.paste(ne, (x + self.corner_width, y))
        self.img._img.paste(se, (x + self.corner_width, y + self.corner_width))
        self.img._img.paste(sw, (x, y + self.corner_width))


class VerticalBarsDrawer(StyledPilQRModuleDrawer):
    """
    Draw connected vertical modules as rounded bars.

    horizontal_shrink controls the gap between adjacent vertical bands.
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
        mode = self.img.mode
        background = self.img.color_mask.back_color
        foreground = self.img.paint_color

        height = self.half_height
        width = height * 2
        shrunken_width = int(width * self.horizontal_shrink)

        self.SQUARE = Image.new(
            mode,
            (shrunken_width, height),
            foreground,
        )

        enlarged_width = width * ANTIALIASING_FACTOR
        enlarged_height = height * ANTIALIASING_FACTOR
        edge = Image.new(
            mode,
            (enlarged_width, enlarged_height),
            background,
        )
        ImageDraw.Draw(edge).ellipse(
            (0, 0, enlarged_width, enlarged_height * 2),
            fill=foreground,
        )

        self.ROUND_TOP = edge.resize(
            (shrunken_width, height),
            Image.Resampling.LANCZOS,
        )
        self.ROUND_BOTTOM = self.ROUND_TOP.transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )

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
    Draw connected horizontal modules as rounded bars.

    vertical_shrink controls the gap between adjacent horizontal bands.
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
        mode = self.img.mode
        background = self.img.color_mask.back_color
        foreground = self.img.paint_color

        width = self.half_width
        height = width * 2
        shrunken_height = int(height * self.vertical_shrink)

        self.SQUARE = Image.new(
            mode,
            (width, shrunken_height),
            foreground,
        )

        enlarged_width = width * ANTIALIASING_FACTOR
        enlarged_height = height * ANTIALIASING_FACTOR
        edge = Image.new(
            mode,
            (enlarged_width, enlarged_height),
            background,
        )
        ImageDraw.Draw(edge).ellipse(
            (0, 0, enlarged_width * 2, enlarged_height),
            fill=foreground,
        )

        self.ROUND_LEFT = edge.resize(
            (width, shrunken_height),
            Image.Resampling.LANCZOS,
        )
        self.ROUND_RIGHT = self.ROUND_LEFT.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

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