from __future__ import annotations

import gettext as gettext_module
from threading import local

TYPE_CHECKING = False

if TYPE_CHECKING:
    import os
    import pathlib

__all__ = ["activate", "deactivate", "decimal_separator", "thousands_separator"]

_TRANSLATIONS: dict[str | None, gettext_module.NullTranslations] = {
    None: gettext_module.NullTranslations()
}
_CURRENT = local()

_THOUSANDS_SEPARATOR: dict[str | None, str] = {
    "de_DE": ".",
    "fr_FR": " ",
    "it_IT": ".",
    "pt_BR": ".",
    "hu_HU": " ",
    "lv": " ",
}

_DECIMAL_SEPARATOR: dict[str | None, str] = {
    "de_DE": ",",
    "fr_FR": ".",
    "it_IT": ",",
    "pt_BR": ",",
    "hu_HU": ",",
    "lv": ",",
}


def _get_default_locale_path() -> pathlib.Path | None:
    package_name = __spec__ and __spec__.parent
    if package_name is None:
        return None

    import importlib.resources

    package_files = importlib.resources.files(package_name)
    with importlib.resources.as_file(package_files) as package_path:
        return package_path / "locale"


def get_translation() -> gettext_module.NullTranslations:
    locale = getattr(_CURRENT, "locale", None)
    return _TRANSLATIONS.get(locale, _TRANSLATIONS[None])


def activate(
    locale: str | None, path: str | os.PathLike[str] | None = None
) -> gettext_module.NullTranslations:
    if locale is None or locale.startswith("en"):
        _CURRENT.locale = None
        return _TRANSLATIONS[None]

    locale_path = path if path is not None else _get_default_locale_path()
    if locale_path is None:
        raise FileNotFoundError(
            "Humanize cannot determinate the default location of the 'locale' folder. "
            "You need to pass the path explicitly."
        )

    translation = _TRANSLATIONS.get(locale)
    if translation is None:
        translation = gettext_module.translation("humanize", locale_path, [locale])
        _TRANSLATIONS[locale] = translation

    _CURRENT.locale = locale
    return translation


def deactivate() -> None:
    _CURRENT.locale = None


def _gettext(message: str) -> str:
    return get_translation().gettext(message)


def _pgettext(msgctxt: str, message: str) -> str:
    return get_translation().pgettext(msgctxt, message)


def _ngettext(message: str, plural: str, num: int) -> str:
    return get_translation().ngettext(message, plural, num)


def _gettext_noop(message: str) -> str:
    return message


def _ngettext_noop(singular: str, plural: str) -> tuple[str, str]:
    return singular, plural


def thousands_separator() -> str:
    locale = getattr(_CURRENT, "locale", None)
    return _THOUSANDS_SEPARATOR.get(locale, ",")


def decimal_separator() -> str:
    locale = getattr(_CURRENT, "locale", None)
    return _DECIMAL_SEPARATOR.get(locale, ".")