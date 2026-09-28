from __future__ import annotations

import re
from typing import NewType, cast

from ._spdx import EXCEPTIONS, LICENSES

__all__ = [
    "InvalidLicenseExpression",
    "NormalizedLicenseExpression",
    "canonicalize_license_expression",
]

license_ref_allowed = re.compile(r"^[A-Za-z0-9.-]+$")

NormalizedLicenseExpression = NewType("NormalizedLicenseExpression", str)


def __dir__() -> list[str]:
    return __all__


class InvalidLicenseExpression(ValueError):
    """Raised when a license-expression string is invalid."""


def canonicalize_license_expression(
    raw_license_expression: str,
) -> NormalizedLicenseExpression:
    if not raw_license_expression:
        raise InvalidLicenseExpression(
            f"Invalid license expression: {raw_license_expression!r}"
        )

    expanded = raw_license_expression.replace("(", " ( ").replace(")", " ) ")
    reference_prefix = "LicenseRef-"
    references = {}

    for value in expanded.split():
        if value.lower().startswith(reference_prefix.lower()):
            references[value.lower()] = (
                reference_prefix + value[len(reference_prefix) :]
            )

    tokens = expanded.lower().split()
    grammar_words = {"or", "and", "with", "(", ")"}
    python_parts: list[str] = []

    for token in tokens:
        if token not in grammar_words:
            python_parts.append("False")
            continue

        if token == "with":
            python_parts.append("or")
            continue

        invalid_opening = (
            token == "("
            and bool(python_parts)
            and python_parts[-1] not in {"or", "and", "("}
        )
        invalid_closing = (
            token == ")"
            and bool(python_parts)
            and python_parts[-1] == "("
        )

        if invalid_opening or invalid_closing:
            raise InvalidLicenseExpression(
                f"Invalid license expression: {raw_license_expression!r}"
            )

        python_parts.append(token)

    try:
        compile(" ".join(python_parts), "", "eval")
    except SyntaxError:
        raise InvalidLicenseExpression(
            f"Invalid license expression: {raw_license_expression!r}"
        ) from None

    output: list[str] = []
    license_seen = False

    for position, token in enumerate(tokens):
        if token in grammar_words:
            if token == "with":
                next_is_invalid = (
                    position + 1 == len(tokens)
                    or tokens[position + 1] in grammar_words
                )
                if not license_seen or next_is_invalid:
                    raise InvalidLicenseExpression(
                        f"Invalid license expression: {raw_license_expression!r}"
                    )

            output.append(token.upper())
            license_seen = False
            continue

        if output and output[-1] == "WITH":
            if token not in EXCEPTIONS:
                raise InvalidLicenseExpression(
                    f"Unknown license exception: {token!r}"
                )

            output.append(EXCEPTIONS[token]["id"])
            license_seen = False
            continue

        suffix = ""
        identifier = token
        if token.endswith("+"):
            identifier = token[:-1]
            suffix = "+"

        if identifier.startswith("licenseref-"):
            reference_id = identifier[len("licenseref-") :]
            if suffix or not license_ref_allowed.match(reference_id):
                raise InvalidLicenseExpression(f"Invalid licenseref: {token!r}")

            output.append(references[identifier])
        else:
            if identifier not in LICENSES:
                raise InvalidLicenseExpression(f"Unknown license: {identifier!r}")

            output.append(LICENSES[identifier]["id"] + suffix)

        license_seen = True

    normalized = " ".join(output).replace("( ", "(").replace(" )", ")")
    return cast("NormalizedLicenseExpression", normalized)