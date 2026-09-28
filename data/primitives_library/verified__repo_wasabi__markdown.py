from typing import Iterable, List, Optional, Sequence

from .compat import Literal


class MarkdownRenderer:
    """Utility for constructing Markdown fragments and documents."""

    def __init__(self, no_emoji: bool = False):
        self.data: List = []
        self.no_emoji = no_emoji

    @property
    def text(self) -> str:
        return "\n\n".join(self.data)

    def add(self, content: str):
        self.data.append(content)

    def table(
        self,
        data: Iterable[Iterable[str]],
        header: Sequence[str],
        aligns: Optional[Sequence[Literal["r", "c", "l"]]] = None,
    ) -> str:
        if aligns is None:
            aligns = ["l"] * len(header)
        if len(aligns) != len(header):
            raise ValueError(
                "Invalid aligns: {} (header length: {})".format(aligns, len(header))
            )

        def divider(alignment):
            if alignment == "c":
                return ":---:"
            if alignment == "r":
                return "---:"
            return "---"

        heading = "| {} |".format(" | ".join(header))
        separators = "| {} |".format(
            " | ".join(divider(aligns[index]) for index in range(len(header)))
        )
        rows = "\n".join("| {} |".format(" | ".join(row)) for row in data)
        return "{}\n{}\n{}".format(heading, separators, rows)

    def title(self, level: int, text: str, emoji: Optional[str] = None) -> str:
        decoration = "{} ".format(emoji) if emoji and not self.no_emoji else ""
        return "{} {}{}".format("#" * level, decoration, text)

    def list(self, items: Iterable[str], numbered: bool = False) -> str:
        result = []
        for index, item in enumerate(items):
            if numbered:
                result.append("{}. {}".format(index + 1, item))
            else:
                result.append("- {}".format(item))
        return "\n".join(result)

    def link(self, text: str, url: str) -> str:
        return "[{}]({})".format(text, url)

    def code_block(self, text: str, lang: str = "") -> str:
        return "```{}\n{}\n```".format(lang, text)

    def code(self, text: str) -> str:
        return self._wrap(text, "`")

    def bold(self, text: str) -> str:
        return self._wrap(text, "**")

    def italic(self, text: str):
        return self._wrap(text, "_")

    def _wrap(self, text, marker):
        return "{}{}{}".format(marker, text, marker)