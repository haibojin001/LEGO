"""Backward-compatible lazy access to PIL module drawer classes."""

import warnings

from qrcode.constants import PIL_AVAILABLE


def __getattr__(name):
    drawer_names = {
        "CircleModuleDrawer",
        "GappedCircleModuleDrawer",
        "GappedSquareModuleDrawer",
        "HorizontalBarsDrawer",
        "RoundedModuleDrawer",
        "SquareModuleDrawer",
        "VerticalBarsDrawer",
    }

    if name not in drawer_names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if PIL_AVAILABLE:
        warnings.warn(
            f"Importing '{name}' directly from this module is deprecated."
            f"Please use 'from qrcode.image.styles.moduledrawers.pil import {name}' "
            f"instead. This backwards compatibility import will be removed in v9.0.",
            DeprecationWarning,
            stacklevel=2,
        )

    from . import pil

    return getattr(pil, name)