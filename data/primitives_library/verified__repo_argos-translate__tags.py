from __future__ import annotations

from argostranslate.translate import ITranslation
from argostranslate.utils import info


class ITag:
    """Represents a tag tree."""

    translateable: bool

    def text(self) -> str:
        """The combined text of all children."""
        raise NotImplementedError()

    def __str__(self) -> str:
        return f'{str(type(self))} "{str(self.children)}"'


class Tag(ITag):
    def __init__(self, children: ITag | str, translateable: bool = True):
        self.children = children
        self.translateable = translateable

    def text(self) -> str:
        return "".join(
            [(child.text() if type(child) != str else child) for child in self.children]
        )


def depth(tag: ITag | str) -> int:
    """Return the depth of an ITag or string."""
    if type(tag) is str:
        return 0
    if len(tag.children) == 0:
        return 0
    return max([depth(child) for child in tag.children])


def translate_preserve_formatting(
    underlying_translation: ITranslation, input_text: str
) -> str:
    """Translate text while preserving leading and trailing spaces."""
    translated_text = underlying_translation.translate(input_text)

    if len(input_text) > 0:
        if input_text[0] == " " and not (
            len(translated_text) > 0 and translated_text[0] == " "
        ):
            translated_text = " " + translated_text
        if input_text[-1] == " " and not (
            len(translated_text) > 0 and translated_text[-1] == " "
        ):
            translated_text = translated_text + " "

    return translated_text


def inject_tags_inference(
    underlying_translation: ITranslation, tag: ITag
) -> ITag | None:
    """Return a translated tag tree using inferred tag placement, if possible."""
    MAX_SEQUENCE_LENGTH = 200

    text = tag.text()
    if len(text) > MAX_SEQUENCE_LENGTH:
        return None

    translated_text = translate_preserve_formatting(underlying_translation, text)

    class InjectionTag:
        def __init__(self, text: str, tag: ITag):
            self.text = text
            self.tag = tag
            self.injection_index = None

    injection_tags = []

    for child in tag.children:
        if depth(child) == 1:
            translated = translate_preserve_formatting(
                underlying_translation, child.text()
            )
            injection_tags.append(InjectionTag(translated, child))
        elif type(child) is not str:
            info("inject_tags_inference", "can't inject depth 0 ITag")
            return None

    for injection_tag in injection_tags:
        injection_index = translated_text.find(injection_tag.text)
        if injection_index != -1:
            injection_tag.injection_index = injection_index
        else:
            info(
                "inject_tags_inference",
                "injection text not found in translated text",
                translated_text,
                injection_tag.text,
            )
            return None

    injection_tags.sort(key=lambda item: item.injection_index)

    for i in range(len(injection_tags) - 1):
        injection_tag = injection_tags[i]
        next_injection_tag = injection_tags[i + 1]
        if (
            injection_tag.injection_index + len(injection_tag.text)
            >= next_injection_tag.injection_index
        ):
            info(
                "inject_tags_inference",
                "injection tags overlap",
                injection_tag,
                next_injection_tag,
            )
            return None

    children = []
    i = 0

    for injection_tag in injection_tags:
        if i < injection_tag.injection_index:
            children.append(translated_text[i : injection_tag.injection_index])
        children.append(injection_tag.tag)
        i = injection_tag.injection_index + len(injection_tag.text)

    if i < len(translated_text):
        children.append(translated_text[i:])

    tag.children = children
    return tag


def translate_tags(
    underlying_translation: ITranslation, tag: ITag | str
) -> ITag | str:
    """Translate a tag tree or string."""
    if type(tag) is str:
        return translate_preserve_formatting(underlying_translation, tag)
    elif tag.translateable is False:
        return tag
    elif depth(tag) == 2:
        tag_injection = inject_tags_inference(underlying_translation, tag)
        if tag_injection is not None:
            info("translate_tags", "tag injection successful")
            return tag_injection
    else:
        tag.children = [
            translate_tags(underlying_translation, child) for child in tag.children
        ]

    return tag