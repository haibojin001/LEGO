import json
import os
from functools import lru_cache

from flask_babel import gettext as _
from flask_babel import lazy_gettext as _lazy
from markupsafe import Markup, escape


@lru_cache(maxsize=None)
def get_available_locales(only_reviewed=True, sort_by_name=False):
    locales_root = os.path.join(os.path.dirname(__file__), "locales")
    locale_dirs = [
        os.path.join(locales_root, directory)
        for directory in os.listdir(locales_root)
    ]

    locales = [{"code": "en", "name": "English", "reviewed": True}]

    for locale_dir in locale_dirs:
        if locale_dir == "en":
            continue

        metadata_file = os.path.join(locale_dir, "meta.json")
        messages_dir = os.path.join(locale_dir, "LC_MESSAGES")

        if os.path.isdir(messages_dir) and os.path.isfile(metadata_file):
            try:
                with open(metadata_file) as file_handle:
                    metadata = json.loads(file_handle.read())
            except Exception as error:
                print(error)
                continue

            if metadata.get("reviewed") or not only_reviewed:
                locales.append(
                    {
                        "code": os.path.basename(locale_dir),
                        "name": metadata.get("name", ""),
                        "reviewed": metadata.get("reviewed", False),
                    }
                )

    if sort_by_name:
        locales.sort(key=lambda locale: locale["name"])

    return locales


@lru_cache(maxsize=None)
def get_available_locale_codes(only_reviewed=True):
    return [
        locale["code"]
        for locale in get_available_locales(only_reviewed=only_reviewed)
    ]


@lru_cache(maxsize=None)
def get_alternate_locale_links():
    template = os.environ.get("LT_LOCALE_LINK_TEMPLATE")
    if template is None:
        return []

    result = []
    for locale in get_available_locale_codes():
        link = template.replace("{LANG}", locale)
        if locale == "en":
            link = link.replace("en.", "")
        result.append({"link": link, "lang": locale})

    return result


def gettext_escaped(text, **variables):
    return json.dumps(_(text, **variables))


def gettext_html(text, **variables):
    translated = str(escape(_(text)))

    escaped_variables = {}
    if variables:
        for name, value in variables.items():
            if hasattr(value, "unescape"):
                escaped_variables[name] = value.unescape()
            else:
                escaped_variables[name] = Markup(value)

    return Markup(
        translated if not escaped_variables else translated % escaped_variables
    )


def swag_eval(swag, func):
    for key in swag:
        value = swag[key]

        if key in ["summary", "description"] and isinstance(value, str) and value != "":
            swag[key] = func(value)
        elif key == "tags" and isinstance(value, list):
            swag[key] = [func(item) for item in value]
        elif isinstance(value, dict):
            swag_eval(value, func)
        elif isinstance(value, list) and key != "consumes":
            for item in value:
                if isinstance(item, str):
                    func(item)
                elif isinstance(item, dict):
                    swag_eval(item, func)

    return swag


def lazy_swag(swag):
    return swag_eval(swag, _lazy)