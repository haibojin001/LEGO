from __future__ import annotations

from pendulum.formatting import Formatter


_formatter = Formatter()


class FormattableMixin:
    _formatter: Formatter = _formatter

    def format(self, fmt: str, locale: str | None = None) -> str:
        return self._formatter.format(self, fmt, locale)

    def for_json(self) -> str:
        return self.isoformat()

    def __format__(self, format_spec: str) -> str:
        if format_spec:
            if "%" in format_spec:
                return self.strftime(format_spec)

            return self.format(format_spec)

        return str(self)

    def __str__(self) -> str:
        return self.isoformat()