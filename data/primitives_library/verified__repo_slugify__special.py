from __future__ import annotations


def add_uppercase_char(char_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Given a replacement char list, this adds uppercase chars to the list."""
    for character, replacement in char_list:
        uppercase_entry = (character.upper(), replacement.capitalize())
        if uppercase_entry not in char_list and character != uppercase_entry[0]:
            char_list.insert(0, uppercase_entry)
    return char_list


_CYRILLIC = [
    ("ё", "e"),
    ("я", "ya"),
    ("х", "h"),
    ("у", "y"),
    ("щ", "sch"),
    ("ю", "u"),
]
CYRILLIC = add_uppercase_char(_CYRILLIC)

_GERMAN = [
    ("ä", "ae"),
    ("ö", "oe"),
    ("ü", "ue"),
]
GERMAN = add_uppercase_char(_GERMAN)

_GREEK = [
    ("χ", "ch"),
    ("Ξ", "X"),
    ("ϒ", "Y"),
    ("υ", "y"),
    ("ύ", "y"),
    ("ϋ", "y"),
    ("ΰ", "y"),
]
GREEK = add_uppercase_char(_GREEK)

PRE_TRANSLATIONS = CYRILLIC + GERMAN + GREEK