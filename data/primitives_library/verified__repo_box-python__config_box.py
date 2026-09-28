from __future__ import annotations

from box.box import Box


class ConfigBox(Box):
    _protected_keys = dir(Box) + [
        "bool",
        "int",
        "float",
        "list",
        "getboolean",
        "getfloat",
        "getint",
    ]

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return super().__getattr__(item.lower())

    def __dir__(self) -> list[str]:
        return super().__dir__() + [
            "bool",
            "int",
            "float",
            "list",
            "getboolean",
            "getfloat",
            "getint",
        ]

    def bool(self, item, default=None):
        try:
            value = self.__getattr__(item)
        except AttributeError as error:
            if default is not None:
                return default
            raise error

        if isinstance(value, (bool, int)):
            return bool(value)
        if isinstance(value, str) and value.lower() in ("n", "no", "false", "f", "0"):
            return False
        return True if value else False

    def int(self, item, default=None):
        try:
            value = self.__getattr__(item)
        except AttributeError as error:
            if default is not None:
                return default
            raise error
        return int(value)

    def float(self, item, default=None):
        try:
            value = self.__getattr__(item)
        except AttributeError as error:
            if default is not None:
                return default
            raise error
        return float(value)

    def list(self, item, default=None, spliter: str = ",", strip=True, mod=None):
        try:
            value = self.__getattr__(item)
        except AttributeError as error:
            if default is not None:
                return default
            raise error

        if strip:
            value = value.lstrip("[").rstrip("]")

        values = [part.strip() if strip else part for part in value.split(spliter)]
        if mod:
            return list(map(mod, values))
        return values

    def getboolean(self, item, default=None):
        return self.bool(item, default)

    def getint(self, item, default=None):
        return self.int(item, default)

    def getfloat(self, item, default=None):
        return self.float(item, default)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.to_dict())})"

    def copy(self):
        return ConfigBox(super().copy())

    def __copy__(self):
        return ConfigBox(super().copy())