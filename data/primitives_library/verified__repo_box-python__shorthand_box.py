from __future__ import annotations

from box.box import Box

__all__ = ["SBox", "DDBox"]


class SBox(Box):
    _protected_keys = dir({}) + [
        "to_dict",
        "to_json",
        "to_yaml",
        "json",
        "yaml",
        "from_yaml",
        "from_json",
        "dict",
        "toml",
        "from_toml",
        "to_toml",
    ]

    @property
    def dict(self) -> dict:
        return self.to_dict()

    @property
    def json(self) -> str:
        return self.to_json()

    @property
    def yaml(self) -> str:
        return self.to_yaml()

    @property
    def toml(self) -> str:
        return self.to_toml()

    def __repr__(self):
        return f"{type(self).__name__}({self})"

    def copy(self) -> SBox:
        return SBox(super().copy())

    def __copy__(self) -> SBox:
        return SBox(super().copy())


class DDBox(SBox):
    def __init__(self, *args, **kwargs):
        kwargs["box_dots"] = True
        kwargs["default_box"] = True
        super().__init__(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance._box_config["box_dots"] = True
        instance._box_config["default_box"] = True
        return instance

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"