from __future__ import annotations

import json
import re
import typing
import warnings
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex, TypeAlias, TypedDict, TypeVar

import jmespath
from lxml import etree, html

from .csstranslator import GenericTranslator, HTMLTranslator
from .utils import extract_regex, flatten, iflatten, shorten

if TYPE_CHECKING:
    from collections.abc import Mapping
    from re import Pattern

    from typing_extensions import Self


_SelectorType = TypeVar("_SelectorType", bound="Selector")
_ParserType: TypeAlias = etree.XMLParser | etree.HTMLParser
_TostringMethodType = Literal["html", "xml"]


class CannotRemoveElementWithoutRoot(Exception):
    pass


class CannotRemoveElementWithoutParent(Exception):
    pass


class CannotDropElementWithoutParent(CannotRemoveElementWithoutParent):
    pass


class SafeXMLParser(etree.XMLParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("resolve_entities", False)
        super().__init__(*args, **kwargs)


class CTGroupValue(TypedDict):
    _parser: type[etree.XMLParser | html.HTMLParser]
    _csstranslator: GenericTranslator | HTMLTranslator
    _tostring_method: _TostringMethodType


_ctgroup: dict[str, CTGroupValue] = {
    "html": {
        "_parser": html.HTMLParser,
        "_csstranslator": HTMLTranslator(),
        "_tostring_method": "html",
    },
    "xml": {
        "_parser": SafeXMLParser,
        "_csstranslator": GenericTranslator(),
        "_tostring_method": "xml",
    },
}


def _xml_or_html(type_: str | None) -> str:
    return "xml" if type_ == "xml" else "html"


_XML_DECLARATION = re.compile(r"[\s\ufeff]*<\?xml\s")


def _detect_xml_or_html(text: str) -> str:
    return "xml" if _XML_DECLARATION.match(text) else "html"


def create_root_node(
    text: str,
    parser_cls: type[_ParserType],
    base_url: str | None = None,
    huge_tree: bool = True,
    body: bytes = b"",
    encoding: str = "utf-8",
) -> etree._Element:
    if not text:
        body = body.replace(b"\x00", b"").strip()
    else:
        body = text.strip().replace("\x00", "").encode(encoding) or b"<html/>"

    parser = parser_cls(recover=True, encoding=encoding, huge_tree=huge_tree)
    root = etree.fromstring(body, parser=parser, base_url=base_url)

    if not huge_tree:
        for error in parser.error_log:
            if "use XML_PARSE_HUGE option" in error.message:
                warnings.warn(
                    "Input data is too big. Set huge_tree=True for huge_tree support.",
                    stacklevel=2,
                )

    if root is None:
        root = etree.fromstring(b"<html/>", parser=parser, base_url=base_url)

    return root


def _get_root_from_text(
    text: str, *, type_: str, **lxml_kwargs: Any
) -> etree._Element:
    return create_root_node(text, _ctgroup[type_]["_parser"], **lxml_kwargs)


def _get_root_and_type_from_bytes(
    body: bytes,
    *,
    encoding: str,
    type_: str | None,
    **lxml_kwargs: Any,
) -> tuple[etree._Element, str]:
    if type_ is None:
        decoded = body.decode(encoding, errors="replace")
        type_ = _detect_xml_or_html(decoded)
    else:
        type_ = _xml_or_html(type_)

    root = create_root_node(
        "",
        _ctgroup[type_]["_parser"],
        body=body,
        encoding=encoding,
        **lxml_kwargs,
    )
    return root, type_


class SelectorList(list[_SelectorType]):
    @typing.overload
    def __getitem__(self, pos: SupportsIndex) -> _SelectorType:
        ...

    @typing.overload
    def __getitem__(self, pos: slice) -> SelectorList[_SelectorType]:
        ...

    def __getitem__(
        self, pos: SupportsIndex | slice
    ) -> _SelectorType | SelectorList[_SelectorType]:
        item = super().__getitem__(pos)
        if isinstance(pos, slice):
            return self.__class__(typing.cast("SelectorList[_SelectorType]", item))
        return typing.cast("_SelectorType", item)

    def __getstate__(self) -> None:
        raise TypeError("can't pickle SelectorList objects")

    def jmespath(self, query: str, **kwargs: Any) -> SelectorList[_SelectorType]:
        return self.__class__(flatten(x.jmespath(query, **kwargs) for x in self))

    def xpath(
        self,
        xpath: str,
        namespaces: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> SelectorList[_SelectorType]:
        return self.__class__(
            flatten(x.xpath(xpath, namespaces=namespaces, **kwargs) for x in self)
        )

    def css(self, query: str) -> SelectorList[_SelectorType]:
        return self.__class__(flatten(x.css(query) for x in self))

    def re(self, regex: str | Pattern[str], replace_entities: bool = True) -> list[str]:
        return flatten(x.re(regex, replace_entities=replace_entities) for x in self)

    @typing.overload
    def re_first(
        self,
        regex: str | Pattern[str],
        default: None = None,
        replace_entities: bool = True,
    ) -> str | None:
        ...

    @typing.overload
    def re_first(
        self,
        regex: str | Pattern[str],
        default: str,
        replace_entities: bool = True,
    ) -> str:
        ...

    def re_first(
        self,
        regex: str | Pattern[str],
        default: str | None = None,
        replace_entities: bool = True,
    ) -> str | None:
        for value in iflatten(
            x.re(regex, replace_entities=replace_entities) for x in self
        ):
            return typing.cast(str, value)
        return default

    def getall(self) -> list[str]:
        return [x.get() for x in self]

    extract = getall

    @typing.overload
    def get(self, default: None = None) -> str | None:
        ...

    @typing.overload
    def get(self, default: str) -> str:
        ...

    def get(self, default: str | None = None) -> str | None:
        for selector in self:
            return selector.get()
        return default

    extract_first = get

    @property
    def attrib(self) -> Mapping[str, str]:
        for selector in self:
            return selector.attrib
        return {}

    def drop(self) -> None:
        for selector in self:
            selector.drop()


_NOT_SET = object()


class Selector:
    _default_namespaces = {
        "re": "http://exslt.org/regular-expressions",
        "set": "http://exslt.org/sets",
    }
    _lxml_smart_strings = False

    def __init__(
        self,
        text: str | None = None,
        type: str | None = None,
        body: bytes = b"",
        encoding: str = "utf-8",
        namespaces: Mapping[str, str] | None = None,
        root: Any = _NOT_SET,
        base_url: str | None = None,
        _expr: str | None = None,
        huge_tree: bool = True,
    ) -> None:
        if type not in (None, "html", "xml", "json"):
            raise ValueError(f"Unsupported selector type: {type}")

        if text is not None and not isinstance(text, str):
            raise TypeError(
                f"text argument should be of type str, got {type(text).__name__}"
            )
        if body and not isinstance(body, bytes):
            raise TypeError(
                f"body argument should be of type bytes, got {type(body).__name__}"
            )
        if text is not None and root is not _NOT_SET:
            raise ValueError("Cannot use both text and root arguments")
        if body and root is not _NOT_SET:
            raise ValueError("Cannot use both body and root arguments")
        if text is not None and body:
            raise ValueError("Cannot use both text and body arguments")

        self._expr = _expr
        self._huge_tree = huge_tree
        self.namespaces = dict(self._default_namespaces)
        if namespaces:
            self.namespaces.update(namespaces)

        if text is not None:
            if type is None:
                type = _detect_xml_or_html(text)
            if type == "json":
                root = json.loads(text)
            else:
                root = _get_root_from_text(
                    text,
                    type_=type,
                    base_url=base_url,
                    huge_tree=huge_tree,
                    encoding=encoding,
                )
        elif body:
            if type == "json":
                root = json.loads(body.decode(encoding))
            else:
                root, type = _get_root_and_type_from_bytes(
                    body,
                    encoding=encoding,
                    type_=type,
                    base_url=base_url,
                    huge_tree=huge_tree,
                )
        elif root is _NOT_SET:
            if type == "json":
                root = None
            else:
                type = _xml_or_html(type)
                root = _get_root_from_text(
                    "",
                    type_=type,
                    base_url=base_url,
                    huge_tree=huge_tree,
                    encoding=encoding,
                )
        elif type is None:
            if isinstance(root, (dict, list, tuple, int, float, bool)) or root is None:
                type = "json"
            else:
                type = "html"

        self.type = typing.cast(str, type)
        self.root = root

        if self.type in _ctgroup:
            group = _ctgroup[self.type]
            self._csstranslator = group["_csstranslator"]
            self._tostring_method = group["_tostring_method"]
        else:
            self._csstranslator = None
            self._tostring_method = "html"

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} query={self._expr!r} "
            f"data={shorten(self.get())!r}>"
        )

    def __getstate__(self) -> None:
        raise TypeError("can't pickle Selector objects")

    def xpath(
        self: Self,
        query: str,
        namespaces: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> SelectorList[Self]:
        if self.type == "json":
            raise ValueError("Cannot use xpath on a Selector of type 'json'")

        ns = dict(self.namespaces)
        if namespaces:
            ns.update(namespaces)

        try:
            result = self.root.xpath(
                query,
                namespaces=ns,
                smart_strings=self._lxml_smart_strings,
                **kwargs,
            )
        except etree.XPathError as exc:
            raise ValueError(f"XPath error: {exc} in {query}") from exc

        if not isinstance(result, list):
            result = [result]

        return SelectorList(
            self.__class__(
                root=value,
                type=self.type,
                namespaces=self.namespaces,
                _expr=query,
                huge_tree=self._huge_tree,
            )
            for value in result
        )

    def css(self: Self, query: str) -> SelectorList[Self]:
        if self.type == "json":
            raise ValueError("Cannot use css on a Selector of type 'json'")
        xpath_query = self._csstranslator.css_to_xpath(query)
        return self.xpath(xpath_query)

    def jmespath(self: Self, query: str, **kwargs: Any) -> SelectorList[Self]:
        if self.type != "json":
            raise ValueError("Cannot use jmespath on a Selector of type '%s'" % self.type)

        result = jmespath.search(query, self.root, **kwargs)
        if result is None:
            return SelectorList()
        if not isinstance(result, list):
            result = [result]

        return SelectorList(
            self.__class__(
                root=value,
                type="json",
                namespaces=self.namespaces,
                _expr=query,
                huge_tree=self._huge_tree,
            )
            for value in result
        )

    def re(
        self, regex: str | Pattern[str], replace_entities: bool = True
    ) -> list[str]:
        return extract_regex(regex, self.get(), replace_entities=replace_entities)

    @typing.overload
    def re_first(
        self,
        regex: str | Pattern[str],
        default: None = None,
        replace_entities: bool = True,
    ) -> str | None:
        ...

    @typing.overload
    def re_first(
        self,
        regex: str | Pattern[str],
        default: str,
        replace_entities: bool = True,
    ) -> str:
        ...

    def re_first(
        self,
        regex: str | Pattern[str],
        default: str | None = None,
        replace_entities: bool = True,
    ) -> str | None:
        matches = self.re(regex, replace_entities=replace_entities)
        return matches[0] if matches else default

    def get(self) -> str:
        if self.type == "json":
            return json.dumps(self.root, ensure_ascii=False)

        if self.root is None:
            return ""

        if isinstance(self.root, etree._ElementUnicodeResult):
            return str(self.root)

        if isinstance(self.root, (str, bytes)):
            if isinstance(self.root, bytes):
                return self.root.decode("utf-8", errors="replace")
            return self.root

        if not isinstance(self.root, etree._Element):
            return str(self.root)

        return etree.tostring(
            self.root,
            method=self._tostring_method,
            encoding="unicode",
            with_tail=False,
        )

    extract = get

    def getall(self) -> list[str]:
        return [self.get()]

    @property
    def attrib(self) -> Mapping[str, str]:
        if isinstance(self.root, etree._Element):
            return self.root.attrib
        return {}

    def register_namespace(self, prefix: str, uri: str) -> None:
        self.namespaces[prefix] = uri

    def remove(self) -> None:
        if not isinstance(self.root, etree._Element):
            raise CannotRemoveElementWithoutRoot(
                "The selector does not contain an element that can be removed"
            )

        tree_root = self.root.getroottree().getroot()
        if tree_root is self.root:
            raise CannotRemoveElementWithoutRoot(
                "The root element cannot be removed"
            )

        parent = self.root.getparent()
        if parent is None:
            raise CannotRemoveElementWithoutParent(
                "The element has no parent and cannot be removed"
            )

        parent.remove(self.root)

    def drop(self) -> None:
        if self.type != "html":
            raise ValueError("Cannot use drop on a Selector of type '%s'" % self.type)

        if not isinstance(self.root, html.HtmlElement):
            raise CannotDropElementWithoutParent(
                "The selector does not contain an HTML element that can be dropped"
            )

        if self.root.getparent() is None:
            raise CannotDropElementWithoutParent(
                "The element has no parent and cannot be dropped"
            )

        self.root.drop_tree()