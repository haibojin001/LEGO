import math

from PIL import Image


class QRColorMask:
    back_color = (255, 255, 255)
    has_transparency = False
    paint_color = back_color

    def initialize(self, styledPilImage, image):
        self.paint_color = styledPilImage.paint_color

    def apply_mask(self, image, use_cache=False):
        image_width, image_height = image.size
        image_pixels = image.load()
        cached_colors = {} if use_cache else None

        for x_pos in range(image_width):
            for y_pos in range(image_height):
                existing = image_pixels[x_pos, y_pos]

                if existing == self.back_color:
                    continue

                if use_cache and existing in cached_colors:
                    image_pixels[x_pos, y_pos] = cached_colors[existing]
                    continue

                amount = self.extrap_color(
                    self.back_color,
                    self.paint_color,
                    existing,
                )

                if amount is None:
                    image_pixels[x_pos, y_pos] = self.get_bg_pixel(
                        image, x_pos, y_pos
                    )
                    continue

                replacement = self.interp_color(
                    self.get_bg_pixel(image, x_pos, y_pos),
                    self.get_fg_pixel(image, x_pos, y_pos),
                    amount,
                )
                image_pixels[x_pos, y_pos] = replacement

                if use_cache:
                    cached_colors[existing] = replacement

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
        coefficients = []

        for first, second, value in zip(col1, col2, interped_color, strict=False):
            coefficient = self.extrap_num(first, second, value)
            if coefficient is not None:
                coefficients.append(coefficient)

        if not coefficients:
            return None

        return sum(coefficients) / len(coefficients)


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

        foreground = Image.new("RGB", image.size, self.front_color)
        background = Image.new("RGB", image.size, self.back_color)
        selector = image.convert("L").point(
            lambda value: 255 if value < 128 else 0
        )
        image.paste(Image.composite(foreground, background, selector))

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
        image_width, _ = image.size
        distance = math.sqrt(
            (x - image_width / 2) ** 2 + (y - image_width / 2) ** 2
        )
        ratio = distance / (math.sqrt(2) * image_width / 2)
        return self.interp_color(self.center_color, self.edge_color, ratio)


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
        image_width, _ = image.size
        distance = max(
            abs(x - image_width / 2),
            abs(y - image_width / 2),
        )
        ratio = distance / (image_width / 2)
        return self.interp_color(self.center_color, self.edge_color, ratio)


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
        image_width, _ = image.size
        return self.interp_color(self.left_color, self.right_color, x / image_width)


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
        image_width, _ = image.size
        return self.interp_color(self.top_color, self.bottom_color, y / image_width)


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