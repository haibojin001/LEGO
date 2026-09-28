"""Compatibility access to PIL-based QR module drawers."""

import warnings

from qrcode.constants import PIL_AVAILABLE


_PIL_DRAWER_NAMES = frozenset(
    (
        "CircleModuleDrawer",
        "GappedCircleModuleDrawer",
        "GappedSquareModuleDrawer",
        "HorizontalBarsDrawer",
        "RoundedModuleDrawer",
        "SquareModuleDrawer",
        "VerticalBarsDrawer",
    )
)


def __getattr__(name):
    """Provide deprecated lazy access to PIL module drawer classes."""
    if name not in _PIL_DRAWER_NAMES:
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