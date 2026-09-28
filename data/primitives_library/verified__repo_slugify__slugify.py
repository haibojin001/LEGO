from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from html.entities import name2codepoint

try:
    import unidecode
except ImportError:
    import text_unidecode as unidecode  # type: ignore[import-untyped, no-redef]

__all__ = ["slugify", "smart_truncate"]

CHAR_ENTITY_PATTERN = re.compile(r"&(%s);" % "|".join(name2codepoint))
DECIMAL_PATTERN = re.compile(r"&#(\d+);")
HEX_PATTERN = re.compile(r"&#x([\da-fA-F]+);")
QUOTE_PATTERN = re.compile(r"[\']+")
DISALLOWED_CHARS_PATTERN = re.compile(r"[^-a-zA-Z0-9]+")
DISALLOWED_UNICODE_CHARS_PATTERN = re.compile(r"[\W_]+")
DUPLICATE_DASH_PATTERN = re.compile(r"-{2,}")
NUMBERS_PATTERN = re.compile(r"(?<=\d),(?=\d)")
DEFAULT_SEPARATOR = "-"


def smart_truncate(
    string: str,
    max_length: int = 0,
    word_boundary: bool = False,
    separator: str = " ",
    save_order: bool = False,
) -> str:
    value = string.strip(separator)

    if not max_length:
        return value

    if len(value) < max_length:
        return value

    if not word_boundary:
        return value[:max_length].strip(separator)

    if separator not in value:
        return value[:max_length]

    result = ""
    for part in value.split(separator):
        if part:
            length = len(result) + len(part)
            if length < max_length:
                result += "{}{}".format(part, separator)
            elif length == max_length:
                result += "{}".format(part)
                break
            elif save_order:
                break

    if not result:
        result = value[:max_length]

    return result.strip(separator)


def slugify(
    text: str,
    entities: bool = True,
    decimal: bool = True,
    hexadecimal: bool = True,
    max_length: int = 0,
    word_boundary: bool = False,
    separator: str = DEFAULT_SEPARATOR,
    save_order: bool = False,
    stopwords: Iterable[str] = (),
    regex_pattern: re.Pattern[str] | str | None = None,
    lowercase: bool = True,
    replacements: Iterable[Iterable[str]] = (),
    allow_unicode: bool = False,
) -> str:
    if replacements:
        for old, new in replacements:
            text = text.replace(old, new)

    if not isinstance(text, str):
        text = str(text, "utf-8", "ignore")

    text = QUOTE_PATTERN.sub(DEFAULT_SEPARATOR, text)

    if allow_unicode:
        text = unicodedata.normalize("NFKC", text)
    else:
        text = unicodedata.normalize("NFKD", text)
        text = unidecode.unidecode(text)

    if not isinstance(text, str):
        text = str(text, "utf-8", "ignore")

    if entities:
        text = CHAR_ENTITY_PATTERN.sub(
            lambda match: chr(name2codepoint[match.group(1)]),
            text,
        )

    if decimal:
        try:
            text = DECIMAL_PATTERN.sub(
                lambda match: chr(int(match.group(1))),
                text,
            )
        except Exception:
            pass

    if hexadecimal:
        try:
            text = HEX_PATTERN.sub(
                lambda match: chr(int(match.group(1), 16)),
                text,
            )
        except Exception:
            pass

    if allow_unicode:
        text = unicodedata.normalize("NFKC", text)
    else:
        text = unicodedata.normalize("NFKD", text)

    if lowercase:
        text = text.lower()

    text = QUOTE_PATTERN.sub("", text)
    text = NUMBERS_PATTERN.sub("", text)

    if allow_unicode:
        disallowed = regex_pattern or DISALLOWED_UNICODE_CHARS_PATTERN
    else:
        disallowed = regex_pattern or DISALLOWED_CHARS_PATTERN

    text = re.sub(disallowed, DEFAULT_SEPARATOR, text)
    text = DUPLICATE_DASH_PATTERN.sub(DEFAULT_SEPARATOR, text).strip(
        DEFAULT_SEPARATOR
    )

    if stopwords:
        if lowercase:
            excluded = [word.lower() for word in stopwords]
            words = [
                word
                for word in text.split(DEFAULT_SEPARATOR)
                if word not in excluded
            ]
        else:
            words = [
                word
                for word in text.split(DEFAULT_SEPARATOR)
                if word not in stopwords
            ]
        text = DEFAULT_SEPARATOR.join(words)

    if replacements:
        for old, new in replacements:
            text = text.replace(old, new)

    if max_length > 0:
        text = smart_truncate(
            text,
            max_length,
            word_boundary,
            DEFAULT_SEPARATOR,
            save_order,
        )

    if separator != DEFAULT_SEPARATOR:
        text = text.replace(DEFAULT_SEPARATOR, separator)

    return text