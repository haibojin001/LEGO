import unicodedata
from hmac import compare_digest
from typing import Dict, Optional, Union
from urllib.parse import quote, urlencode, urlparse


def build_uri(
    secret: str,
    name: str,
    initial_count: Optional[int] = None,
    issuer: Optional[str] = None,
    algorithm: Optional[str] = None,
    digits: Optional[int] = None,
    period: Optional[int] = None,
    **kwargs,
) -> str:
    """Construct an otpauth provisioning URI for an HOTP or TOTP token."""
    counter_is_supplied = initial_count is not None
    token_kind = "hotp" if counter_is_supplied else "totp"

    query_values: Dict[str, Union[None, int, str]] = {"secret": secret}

    account_label = quote(name)
    if issuer is not None:
        account_label = "{}:{}".format(quote(issuer), account_label)
        query_values["issuer"] = issuer

    if counter_is_supplied:
        query_values["counter"] = initial_count

    if algorithm is not None and algorithm != "sha1":
        query_values["algorithm"] = algorithm.upper()

    if digits is not None and digits != 6:
        query_values["digits"] = digits

    if period is not None and period != 30:
        query_values["period"] = period

    for parameter, value in kwargs.items():
        if not isinstance(value, str):
            raise ValueError("All otpauth uri parameters must be strings")

        if parameter == "image":
            parsed_image = urlparse(value)
            if (
                parsed_image.scheme != "https"
                or not parsed_image.netloc
                or not parsed_image.path
            ):
                raise ValueError("{} is not a valid url".format(parsed_image))

        query_values[parameter] = value

    encoded_query = urlencode(query_values).replace("+", "%20")
    return "otpauth://{}/{}?{}".format(token_kind, account_label, encoded_query)


def strings_equal(s1: str, s2: str) -> bool:
    """Compare Unicode strings with normalization and constant-time bytes comparison."""
    normalized_first = unicodedata.normalize("NFKC", s1)
    normalized_second = unicodedata.normalize("NFKC", s2)
    return compare_digest(
        normalized_first.encode("utf-8"),
        normalized_second.encode("utf-8"),
    )