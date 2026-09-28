import json

from . import utils


def parse_colors(path):
    """Load color definitions from a supported color file."""
    if path.endswith(".txt"):
        return parse_rgb_txt_file(path)
    if path.endswith(".json"):
        return parse_json_color_file(path)
    raise TypeError("colorful only supports .txt and .json files for colors")


def parse_rgb_txt_file(path):
    """Read an X11-style rgb.txt file and return its colors."""
    colors = {}

    with open(path) as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith("!"):
                continue

            fields = line.split()
            colors[" ".join(fields[3:])] = (
                int(fields[0]),
                int(fields[1]),
                int(fields[2]),
            )

    return colors


def parse_json_color_file(path):
    """Read a JSON color list and map names to hexadecimal values."""
    with open(path) as source:
        entries = json.load(source)

    return {entry["name"]: entry["hex"] for entry in entries}


def sanitize_color_palette(colorpalette):
    """Convert palette names and hexadecimal values into usable forms."""
    sanitized = {}

    def normalize_name(parts):
        if len(parts) == 1:
            value = parts[0]
            return value[:1].lower() + value[1:]

        return parts[0].lower() + "".join(part.capitalize() for part in parts[1:])

    for name, color in colorpalette.items():
        if isinstance(color, str):
            color = utils.hex_to_rgb(color)
        sanitized[normalize_name(name.split())] = color

    return sanitized