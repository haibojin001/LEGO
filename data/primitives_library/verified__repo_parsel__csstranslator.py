from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from cssselect import GenericTranslator as OriginalGenericTranslator
from cssselect import HTMLTranslator as OriginalHTMLTranslator
from cssselect.parser import Element, FunctionalPseudoElement, PseudoElement
from cssselect.xpath import ExpressionError, XPathExpr as OriginalXPathExpr, is_safe_name

if TYPE_CHECKING:
    from typing_extensions import Self


class XPathExpr(OriginalXPathExpr):
    textnode: bool = False
    attribute: str | None = None

    @classmethod
    def from_xpath(
        cls,
        xpath: OriginalXPathExpr,
        textnode: bool = False,
        attribute: str | None = None,
    ) -> Self:
        expression = cls(
            path=xpath.path,
            element=xpath.element,
            condition=xpath.condition,
        )
        expression.textnode = textnode
        expression.attribute = attribute
        return expression

    def __str__(self) -> str:
        result = super().__str__()

        if self.textnode:
            if result == "*":
                result = "text()"
            elif result.endswith("::*/*"):
                result = result[:-3] + "text()"
            else:
                result += "/text()"

        if self.attribute is not None:
            if result.endswith("::*/*"):
                result = result[:-2]

            if is_safe_name(self.attribute):
                result += f"/@{self.attribute}"
            else:
                attribute = OriginalGenericTranslator.xpath_literal(self.attribute)
                result += f"/attribute::*[name() = {attribute}]"

        return result

    def join(
        self: Self,
        combiner: str,
        other: OriginalXPathExpr,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        if not isinstance(other, XPathExpr):
            raise ValueError(
                f"Expressions of type {__name__}.XPathExpr can ony join expressions"
                f" of the same type (or its descendants), got {type(other)}"
            )
        super().join(combiner, other, *args, **kwargs)
        self.textnode = other.textnode
        self.attribute = other.attribute
        return self


class TranslatorProtocol(Protocol):
    def xpath_element(self, selector: Element) -> OriginalXPathExpr:
        pass

    def css_to_xpath(self, css: str, prefix: str = ...) -> str:
        pass


class TranslatorMixin:
    """Mixin adding support for CSS pseudo-elements."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._build_cache()

    def _build_cache(self) -> None:
        self._cache = lru_cache(maxsize=256)(self._translate)

    def _translate(self, css: str, prefix: str) -> str:
        return super().css_to_xpath(css, prefix)  # type: ignore[misc,no-any-return]

    def css_to_xpath(self, css: str, prefix: str = "descendant-or-self::") -> str:
        return self._cache(css, prefix)

    def __getstate__(self) -> dict[str, Any]:
        return {name: value for name, value in self.__dict__.items() if name != "_cache"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._build_cache()

    def xpath_element(self: TranslatorProtocol, selector: Element) -> XPathExpr:
        xpath = super().xpath_element(selector)  # type: ignore[safe-super]
        return XPathExpr.from_xpath(xpath)

    def xpath_pseudo_element(
        self,
        xpath: OriginalXPathExpr,
        pseudo_element: PseudoElement,
    ) -> OriginalXPathExpr:
        if isinstance(pseudo_element, FunctionalPseudoElement):
            method_name = (
                f"xpath_{pseudo_element.name.replace('-', '_')}"
                "_functional_pseudo_element"
            )
            method = getattr(self, method_name, None)
            if not method:
                raise ExpressionError(
                    f"The functional pseudo-element ::{pseudo_element.name}() is unknown"
                )
            return method(xpath, pseudo_element)

        method_name = (
            f"xpath_{pseudo_element.replace('-', '_')}_simple_pseudo_element"
        )
        method = getattr(self, method_name, None)
        if not method:
            raise ExpressionError(f"The pseudo-element ::{pseudo_element} is unknown")
        return method(xpath)

    def xpath_attr_functional_pseudo_element(
        self,
        xpath: OriginalXPathExpr,
        function: FunctionalPseudoElement,
    ) -> XPathExpr:
        if function.argument_types() not in (["STRING"], ["IDENT"]):
            raise ExpressionError(
                f"Expected a single string or ident for ::attr(), got {function.arguments!r}"
            )
        return XPathExpr.from_xpath(xpath, attribute=function.arguments[0].value)

    def xpath_text_simple_pseudo_element(
        self,
        xpath: OriginalXPathExpr,
    ) -> XPathExpr:
        return XPathExpr.from_xpath(xpath, textnode=True)


class GenericTranslator(TranslatorMixin, OriginalGenericTranslator):
    pass


class HTMLTranslator(TranslatorMixin, OriginalHTMLTranslator):
    pass


_translator = HTMLTranslator()


def css2xpath(query: str) -> str:
    """Return translated XPath version of a given CSS query."""
    return _translator.css_to_xpath(query)