from __future__ import annotations

from enum import Enum, IntEnum

try:
    from typing import Dict, List, NotRequired, TypedDict, Union
except ImportError:
    from typing_extensions import Dict, List, NotRequired, TypedDict, Union

URI = str
DocumentUri = str
Uint = int
RegExp = str


class Position(TypedDict):
    """A zero-based position within a text document."""

    line: Uint
    character: Uint


class Range(TypedDict):
    """A span between two positions in a text document."""

    start: Position
    end: Position


class Location(TypedDict):
    """A source location together with local path information."""

    uri: DocumentUri
    range: Range
    absolutePath: str
    relativePath: str


class CompletionItemKind(IntEnum):
    """The classification assigned to a completion item."""

    Text = 1
    Method = 2
    Function = 3
    Constructor = 4
    Field = 5
    Variable = 6
    Class = 7
    Interface = 8
    Module = 9
    Property = 10
    Unit = 11
    Value = 12
    Enum = 13
    Keyword = 14
    Snippet = 15
    Color = 16
    File = 17
    Reference = 18
    Folder = 19
    EnumMember = 20
    Constant = 21
    Struct = 22
    Event = 23
    Operator = 24
    TypeParameter = 25


class CompletionItem(TypedDict):
    """A completion candidate returned by a language server."""

    completionText: str
    kind: CompletionItemKind
    detail: NotRequired[str]


class SymbolKind(IntEnum):
    """The category of a program symbol."""

    File = 1
    Module = 2
    Namespace = 3
    Package = 4
    Class = 5
    Method = 6
    Property = 7
    Field = 8
    Constructor = 9
    Enum = 10
    Interface = 11
    Function = 12
    Variable = 13
    Constant = 14
    String = 15
    Number = 16
    Boolean = 17
    Array = 18
    Object = 19
    Key = 20
    Null = 21
    EnumMember = 22
    Struct = 23
    Event = 24
    Operator = 25
    TypeParameter = 26


class SymbolTag(IntEnum):
    """Additional rendering metadata for a symbol."""

    Deprecated = 1


class UnifiedSymbolInformation(TypedDict):
    """A normalized representation of document or workspace symbol data."""

    deprecated: NotRequired[bool]
    location: NotRequired[Location]
    name: str
    kind: SymbolKind
    tags: NotRequired[List[SymbolTag]]
    containerName: NotRequired[str]
    detail: NotRequired[str]
    range: NotRequired[Range]
    selectionRange: NotRequired[Range]


TreeRepr = Dict[int, List["TreeRepr"]]


class MarkupKind(Enum):
    """The format used for markup content."""

    PlainText = "plaintext"
    Markdown = "markdown"


class __MarkedString_Type_1(TypedDict):
    language: str
    value: str


MarkedString = Union[str, "__MarkedString_Type_1"]


class MarkupContent(TypedDict):
    """Text whose interpretation is determined by its markup kind."""

    kind: MarkupKind
    value: str


class Hover(TypedDict):
    """Information shown when hovering over a document position."""

    contents: Union[MarkedString, List[MarkedString], MarkupContent]
    range: NotRequired[Range]