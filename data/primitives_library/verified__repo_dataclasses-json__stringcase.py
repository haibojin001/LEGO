import re


def uplowcase(string, case):
    value = str(string)
    if case == "up":
        return value.upper()
    if case == "low":
        return value.lower()


def capitalcase(string):
    value = str(string)
    if not value:
        return value
    return uplowcase(value[0], "up") + value[1:]


def camelcase(string):
    value = re.sub(r"^[-_.]", "", str(string))
    if not value:
        return value

    return uplowcase(value[0], "low") + re.sub(
        r"[-_.\s]([a-z0-9])",
        lambda match: uplowcase(match.group(1), "up"),
        value[1:],
    )


def snakecase(string):
    value = re.sub(r"[-.\s]", "_", str(string))
    if not value:
        return value

    return uplowcase(value[0], "low") + re.sub(
        r"[A-Z0-9]",
        lambda match: "_" + uplowcase(match.group(0), "low"),
        value[1:],
    )


def spinalcase(string):
    return re.sub(r"_", "-", snakecase(string))


def pascalcase(string):
    return capitalcase(camelcase(string))