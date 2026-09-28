import math

from PIL import Image


class QRColorMask:
    back_color = (255, 255, 255)
    has_transparency = False
    paint_color = back_color

    def initialize(self, styledPilImage, image):
        self.paint_color = styledPilImage.paint_color

    def apply_mask(self, image, use_cache=False):
        width, height = image.size
        pixels = image.load()
        cached_colors = {} if use_cache else None

        for x in range(width):
            for y in range(height):
                source = pixels[x, y]

                if source == self.back_color:
                    continue

                if use_cache and source in cached_colors:
                    pixels[x, y] = cached_colors[source]
                    continue

                amount = self.extrap_color(
                    self.back_color,
                    self.paint_color,
                    source,
                )

                if amount is None:
                    pixels[x, y] = self.get_bg_pixel(image, x, y)
                    continue

                result = self.interp_color(
                    self.get_bg_pixel(image, x, y),
                    self.get_fg_pixel(image, x, y),
                    amount,
                )
                pixels[x, y] = result

                if use_cache:
                    cached_colors[source] = result

    def get_fg_pixel(self, image, x, y):
        raise NotImplementedError("QRModuleDrawer.paint_fg_pixel")

    def get_bg_pixel(self, image, x, y):
        return self.back_color

    def interp_num(self, n1, n2, norm):
        return int(n2 * norm + n1 * (1 - norm))

    def interp_color(self, col1, col2, norm):
        return tuple(
            self.interp_num(col1[index], col2[index], norm)
            for index in range(len(col1))
        )

    def extrap_num(self, n1, n2, interped_num):
        if n1 == n2:
            return None
        return (interped_num - n1) / (n2 - n1)

    def extrap_color(self, col1, col2, interped_color):
        values = []

        for first, second, value in zip(col1, col2, interped_color, strict=False):
            amount = self.extrap_num(first, second, value)
            if amount is not None:
                values.append(amount)

        if not values:
            return None

        return sum(values) / len(values)


class SolidFillColorMask(QRColorMask):
    def __init__(self, back_color=(255, 255, 255), front_color=(0, 0, 0)):
        self.back_color = back_color
        self.front_color = front_color
        self.has_transparency = len(back_color) == 4

    def apply_mask(self, image):
        if (
            self.back_color == (255, 255, 255)
            and self.front_color == (0, 0, 0)
        ):
            return

        selection = image.convert("L").point(
            lambda value: 255 if value < 128 else 0
        )
        foreground = Image.new("RGB", image.size, self.front_color)
        background = Image.new("RGB", image.size, self.back_color)
        image.paste(Image.composite(foreground, background, selection))

    def get_fg_pixel(self, image, x, y):
        return self.front_color


class RadialGradiantColorMask(QRColorMask):
    def __init__(
        self,
        back_color=(255, 255, 255),
        center_color=(0, 0, 0),
        edge_color=(0, 0, 255),
    ):
        self.back_color = back_color
        self.center_color = center_color
        self.edge_color = edge_color
        self.has_transparency = len(back_color) == 4

    def get_fg_pixel(self, image, x, y):
        width, _ = image.size
        distance = math.sqrt(
            (x - width / 2) ** 2 + (y - width / 2) ** 2
        )
        amount = distance / (math.sqrt(2) * width / 2)
        return self.interp_color(self.center_color, self.edge_color, amount)


class SquareGradiantColorMask(QRColorMask):
    def __init__(
        self,
        back_color=(255, 255, 255),
        center_color=(0, 0, 0),
        edge_color=(0, 0, 255),
    ):
        self.back_color = back_color
        self.center_color = center_color
        self.edge_color = edge_color
        self.has_transparency = len(back_color) == 4

    def get_fg_pixel(self, image, x, y):
        width, _ = image.size
        distance = max(abs(x - width / 2), abs(y - width / 2))
        amount = distance / (width / 2)
        return self.interp_color(self.center_color, self.edge_color, amount)


class HorizontalGradiantColorMask(QRColorMask):
    def __init__(
        self,
        back_color=(255, 255, 255),
        left_color=(0, 0, 0),
        right_color=(0, 0, 255),
    ):
        self.back_color = back_color
        self.left_color = left_color
        self.right_color = right_color
        self.has_transparency = len(back_color) == 4

    def get_fg_pixel(self, image, x, y):
        width, _ = image.size
        return self.interp_color(self.left_color, self.right_color, x / width)


class VerticalGradiantColorMask(QRColorMask):
    def __init__(
        self,
        back_color=(255, 255, 255),
        top_color=(0, 0, 0),
        bottom_color=(0, 0, 255),
    ):
        self.back_color = back_color
        self.top_color = top_color
        self.bottom_color = bottom_color
        self.has_transparency = len(back_color) == 4

    def get_fg_pixel(self, image, x, y):
        width, _ = image.size
        return self.interp_color(self.top_color, self.bottom_color, y / width)


class ImageColorMask(QRColorMask):
    def __init__(
        self,
        back_color=(255, 255, 255),
        color_mask_path=None,
        color_mask_image=None,
    ):
        self.back_color = back_color

        if color_mask_image:
            self.color_img = color_mask_image
        else:
            self.color_img = Image.open(color_mask_path)

        self.has_transparency = len(back_color) == 4

    def initialize(self, styledPilImage, image):
        self.paint_color = styledPilImage.paint_color
        self.color_img = self.color_img.resize(image.size)

    def get_fg_pixel(self, image, x, y):
        return self.color_img.getpixel((x, y))