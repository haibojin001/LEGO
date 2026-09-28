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
    counter_given = initial_count is not None
    kind = "hotp" if counter_given else "totp"

    values: Dict[str, Union[None, int, str]] = {"secret": secret}
    encoded_label = quote(name)

    if issuer is not None:
        encoded_label = quote(issuer) + ":" + encoded_label
        values["issuer"] = issuer

    if counter_given:
        values["counter"] = initial_count

    if algorithm is not None and algorithm != "sha1":
        values["algorithm"] = algorithm.upper()

    if digits is not None and digits != 6:
        values["digits"] = digits

    if period is not None and period != 30:
        values["period"] = period

    for parameter_name, parameter_value in kwargs.items():
        if not isinstance(parameter_value, str):
            raise ValueError("All otpauth uri parameters must be strings")

        if parameter_name == "image":
            image = urlparse(parameter_value)
            if image.scheme != "https" or not image.netloc or not image.path:
                raise ValueError("{} is not a valid url".format(image))

        values[parameter_name] = parameter_value

    query_string = urlencode(values).replace("+", "%20")
    return "otpauth://{}/{}/?{}".format(kind, encoded_label, query_string).replace(
        "{}/?", "{}?"
    )


def strings_equal(s1: str, s2: str) -> bool:
    normalized_s1 = unicodedata.normalize("NFKC", s1)
    normalized_s2 = unicodedata.normalize("NFKC", s2)
    return compare_digest(normalized_s1.encode("utf-8"), normalized_s2.encode("utf-8"))