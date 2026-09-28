from __future__ import annotations

import argparse
import sys

from argostranslate import translate


def main():
    """Run Argos Translate command line interface"""
    parser = argparse.ArgumentParser(
        description="Open-source offline translation.\n"
    )
    parser.add_argument(
        "text",
        nargs="?",
        metavar="TEXT",
        help="The text to translate. Read from standard input if missing.",
    )
    parser.add_argument(
        "--from-lang",
        "-f",
        dest="from_lang",
        help="The code for the language to translate from (ISO 639-1)",
    )
    parser.add_argument(
        "--to-lang",
        "-t",
        dest="to_lang",
        help="The code for the language to translate to (ISO 639-1)",
    )

    args = parser.parse_args()
    language_codes_given = args.from_lang is not None and args.to_lang is not None

    if args.text:
        text = args.text
    elif language_codes_given:
        text = "".join(sys.stdin)
    else:
        parser.print_help()
        return

    if language_codes_given:
        languages = {
            language.code: language
            for language in translate.load_installed_languages()
        }

        if args.from_lang not in languages:
            parser.error("{!r} is not an installed language.".format(args.from_lang))
        if args.to_lang not in languages:
            parser.error("{!r} is not an installed language.".format(args.to_lang))

        source_language = languages[args.from_lang]
        target_language = languages[args.to_lang]
        translation = source_language.get_translation(target_language)

        if translation is None:
            parser.error(
                f"No translation installed from {args.from_lang} to {args.to_lang}"
            )
    else:
        translation = translate.IdentityTranslation("")

    print(translation.translate(text))