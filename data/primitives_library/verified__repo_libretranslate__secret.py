import base64 as _base64
import random as _random
import string as _string
from functools import lru_cache as _lru_cache

from libretranslate.storage import get_storage


def to_base(n, b):
    if n == 0:
        return 0

    negative = n < 0
    number = -n if negative else n
    result = []

    while number:
        result.append(str(number % b))
        number //= b

    value = int("".join(reversed(result)))
    return -value if negative else value


@_lru_cache(maxsize=4)
def obfuscate(input_str):
    fragments = []
    choices = ["+", "-", "*", ""]

    for codepoint in map(ord, input_str):
        offset = _random.randint(1, 100)
        selected = _random.choice(choices)

        if selected == "+":
            computed = codepoint + offset
            selected = "-"
        elif selected == "-":
            computed = codepoint - offset
            selected = "+false+" if _random.randint(0, 1) == 0 else "+"
        elif selected == "*":
            computed = codepoint * offset
            selected = "/**\\/*//" if _random.randint(0, 1) == 0 else "/"

        decimal_form = _random.randint(0, 1) == 0
        radix = _random.randint(4, 7)

        if selected:
            if decimal_form:
                fragments.append(f"_({computed}{selected}{offset})")
            else:
                fragments.append(
                    f"_(p({to_base(computed, radix)},{radix}){selected}"
                    f"p({to_base(offset, radix)},{hex(radix)}))"
                )
        elif decimal_form:
            fragments.append(f"_({codepoint})")
        else:
            fragments.append(f"_(p({to_base(codepoint, radix)},{radix}))")

    for _ in range(len(input_str) // 3):
        junk = _random.randint(1, 100)
        fragments.insert(
            _random.randint(0, len(fragments)),
            f"_(/*_({junk})*/)",
        )

    for _ in range(len(input_str) // 3):
        fragments.insert(_random.randint(0, len(fragments)), "\n[]\n")

    return "(_=String.fromCharCode,p=parseInt," + "+".join(fragments) + ")"


def generate_secret():
    return "".join(
        _random.choices(_string.ascii_uppercase + _string.digits, k=7)
    )


def rotate_secrets():
    storage = get_storage()
    storage.set_str("secret_0", storage.get_str("secret_1"))
    storage.set_str("secret_1", generate_secret())


def secret_match(secret):
    storage = get_storage()
    return (
        secret == storage.get_str("secret_0")
        or secret == storage.get_str("secret_1")
    )


def secret_bogus_match(secret):
    if _random.randint(0, 1) == 0:
        return secret == get_bogus_secret()
    return False


def get_current_secret():
    return get_storage().get_str("secret_1")


def get_current_secret_b64():
    return _base64.b64encode(
        get_current_secret().encode("utf-8")
    ).decode("utf-8")


def get_current_secret_js():
    return obfuscate(get_current_secret_b64())


def get_bogus_secret():
    return get_storage().get_str("secret_bogus")


def get_bogus_secret_b64():
    return _base64.b64encode(
        get_bogus_secret().encode("utf-8")
    ).decode("utf-8")


def get_bogus_secret_js():
    return obfuscate(get_bogus_secret_b64())


@_lru_cache(maxsize=1)
def get_emoji():
    return _random.choice([
        "😂", "🤪", "😜", "🤣", "😹", "🐒", "🙈", "🤡", "🥸", "😆",
        "🥴", "🐸", "🐤", "🐒🙊", "👀", "💩", "🤯", "😛", "🤥", "👻",
    ])


def setup(args):
    if args.require_api_key_secret:
        storage = get_storage()

        if not storage.exists("secret_0"):
            storage.set_str("secret_0", generate_secret())

        if not storage.exists("secret_1"):
            storage.set_str("secret_1", generate_secret())

        if not storage.exists("secret_bogus"):
            storage.set_str("secret_bogus", generate_secret())