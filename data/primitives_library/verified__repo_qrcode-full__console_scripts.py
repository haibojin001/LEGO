#!/usr/bin/env python
"""
qr - Convert stdin (or the first argument) to a QR Code.

When stdout is a tty the QR Code is printed to the terminal and when stdout is
a pipe to a file an image is written. The default image format is PNG.
"""

from __future__ import annotations

import optparse
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import qrcode

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qrcode.image.base import BaseImage, DrawerAliases

if sys.platform.startswith(("win", "cygwin")):  # pragma: no cover
    import colorama

    colorama.init()

default_factories = {
    "pil": "qrcode.image.pil.PilImage",
    "png": "qrcode.image.pure.PyPNGImage",
    "svg": "qrcode.image.svg.SvgImage",
    "svg-fragment": "qrcode.image.svg.SvgFragmentImage",
    "svg-path": "qrcode.image.svg.SvgPathImage",
    "pymaging": "qrcode.image.pure.PymagingImage",
}

error_correction = {
    "L": qrcode.ERROR_CORRECT_L,
    "M": qrcode.ERROR_CORRECT_M,
    "Q": qrcode.ERROR_CORRECT_Q,
    "H": qrcode.ERROR_CORRECT_H,
}


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = optparse.OptionParser(
        usage=(__doc__ or "").strip(),
        version=metadata.version("qrcode"),
    )

    def raise_error(message: str) -> NoReturn:
        parser.error(message)
        raise

    parser.add_option(
        "--factory",
        help=(
            "Full python path to the image factory class to create the image "
            "with. You can use the following shortcuts to the built-in image "
            f"factory classes: {commas(default_factories)}."
        ),
    )
    parser.add_option(
        "--factory-drawer",
        help=f"Use an alternate drawer. {get_drawer_help()}.",
    )
    parser.add_option(
        "--optimize",
        type=int,
        help=(
            "Optimize the data by looking for chunks of at least this many "
            "characters that could use a more efficient encoding method. Use "
            "0 to turn off chunk optimization."
        ),
    )
    parser.add_option(
        "--error-correction",
        type="choice",
        choices=sorted(error_correction),
        default="M",
        help=(
            "The error correction level to use. Choices are L (7%), "
            "M (15%, default), Q (25%), and H (30%)."
        ),
    )
    parser.add_option(
        "--ascii",
        action="store_true",
        help="Print as ascii even if stdout is piped.",
    )
    parser.add_option(
        "--output",
        help=(
            "The output file. If not specified, the image is sent to the "
            "standard output."
        ),
    )

    options, arguments = parser.parse_args(args)

    if options.factory:
        factory_path = default_factories.get(options.factory, options.factory)
        try:
            factory = get_factory(factory_path)
        except ValueError as exc:
            raise_error(str(exc))
    else:
        factory = None

    code = qrcode.QRCode(
        error_correction=error_correction[options.error_correction],
        image_factory=factory,
    )

    if arguments:
        payload = arguments[0].encode(errors="surrogateescape")
    else:
        payload = sys.stdin.buffer.read()

    if options.optimize is None:
        code.add_data(payload)
    else:
        code.add_data(payload, optimize=options.optimize)

    if options.output:
        image = code.make_image()
        with Path(options.output).open("wb") as output:
            image.save(output)
        return

    if factory is None and (os.isatty(sys.stdout.fileno()) or options.ascii):
        code.print_ascii(tty=not options.ascii)
        return

    image_options = {}
    aliases: DrawerAliases | None = getattr(
        code.image_factory, "drawer_aliases", None
    )

    if options.factory_drawer:
        if not aliases:
            raise_error("The selected factory has no drawer aliases.")
        if options.factory_drawer not in aliases:
            raise_error(
                f"{options.factory_drawer} factory drawer not found."
                f" Expected {commas(aliases)}"
            )
        drawer_type, drawer_options = aliases[options.factory_drawer]
        image_options["module_drawer"] = drawer_type(**drawer_options)

    image = code.make_image(**image_options)
    sys.stdout.flush()
    image.save(sys.stdout.buffer)


def get_factory(module: str) -> type[BaseImage]:
    if "." not in module:
        raise ValueError("The image factory is not a full python path")
    module_name, class_name = module.rsplit(".", 1)
    imported_module = __import__(module_name, {}, {}, [class_name])
    return getattr(imported_module, class_name)


def get_drawer_help() -> str:
    groups: dict[str, set] = {}

    for factory_name, factory_path in default_factories.items():
        try:
            factory = get_factory(factory_path)
        except ImportError:  # pragma: no cover
            continue

        aliases: DrawerAliases | None = getattr(factory, "drawer_aliases", None)
        if not aliases:
            continue

        groups.setdefault(commas(aliases), set()).add(factory_name)

    return ". ".join(
        f"For {commas(factory_names, 'and')}, use: {alias_names}"
        for alias_names, factory_names in groups.items()
    )


def commas(items: Iterable[str], joiner="or") -> str:
    values = tuple(items)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])} {joiner} {values[-1]}"


if __name__ == "__main__":  # pragma: no cover
    main()