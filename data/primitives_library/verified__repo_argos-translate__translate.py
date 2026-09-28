from __future__ import annotations

import functools
from typing import List

import ctranslate2
from ctranslate2 import Translator

from argostranslate import apis, fewshot, package, sbd, settings
from argostranslate.models import ILanguageModel
from argostranslate.package import Package
from argostranslate.sbd import (
    MiniSBDSentencizer,
    SpacySentencizerSmall,
    StanzaSentencizer,
)
from argostranslate.utils import info


class Hypothesis:
    """Represents a translation hypothesis."""

    value: str
    score: float

    def __init__(self, value: str, score: float):
        self.value = value
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f"({repr(self.value)}, {self.score})"

    def __str__(self):
        return repr(self)


class ITranslation:
    """Base interface for translations between languages."""

    from_lang: Language
    to_lang: Language

    def translate(self, input_text: str) -> str:
        return self.hypotheses(input_text, num_hypotheses=1)[0].value

    def hypotheses(self, input_text: str, num_hypotheses: int = 4) -> list[Hypothesis]:
        raise NotImplementedError()

    @staticmethod
    def split_into_paragraphs(input_text: str) -> list[str]:
        return input_text.split("\n")

    @staticmethod
    def combine_paragraphs(paragraphs: list[str]) -> str:
        return "\n".join(paragraphs)

    def __repr__(self):
        return str(self.from_lang) + " -> " + str(self.to_lang)

    def __str__(self):
        return repr(self).replace("->", "→")


class Language:
    """A language for which translations may be installed."""

    translations_from: list[ITranslation] = []
    translations_to: list[ITranslation] = []

    def __init__(self, code: str, name: str):
        self.code = code
        self.name = name
        self.translations_from = []
        self.translations_to = []

    def __str__(self):
        return self.name

    def get_translation(self, to: Language) -> ITranslation | None:
        for translation in self.translations_from:
            if translation.to_lang.code == to.code:
                return translation
        return None


class PackageTranslation(ITranslation):
    """Translation implemented by an installed Argos package."""

    def __init__(self, from_lang: Language, to_lang: Language, pkg: Package):
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.pkg = pkg
        self.translator = None

        sentencizer_class = None
        stanza_not_installed = sbd.stanza is None

        if settings.chunk_type in (
            settings.ChunkType.ARGOSTRANSLATE,
            settings.ChunkType.DEFAULT,
        ):
            if "stanza" in str(pkg.packaged_sbd_path) and not stanza_not_installed:
                sentencizer_class = StanzaSentencizer
            elif "minisbd" in str(pkg.packaged_sbd_path):
                sentencizer_class = MiniSBDSentencizer
            else:
                sentencizer_class = MiniSBDSentencizer
        elif settings.chunk_type == settings.ChunkType.STANZA:
            sentencizer_class = StanzaSentencizer
        elif settings.chunk_type == settings.ChunkType.MINISBD:
            sentencizer_class = MiniSBDSentencizer
        elif settings.chunk_type == settings.ChunkType.SPACY:
            sentencizer_class = SpacySentencizerSmall

        if sentencizer_class is None:
            raise NotImplementedError()

        self.sentencizer = sentencizer_class(pkg)

    def hypotheses(self, input_text: str, num_hypotheses: int = 4) -> list[Hypothesis]:
        if self.translator is None:
            params = {
                "model_path": str(self.pkg.package_path / "model"),
                "device": settings.device,
                "inter_threads": settings.inter_threads,
                "intra_threads": settings.intra_threads,
            }
            if settings.compute_type != "auto":
                params["compute_type"] = settings.compute_type
            self.translator = ctranslate2.Translator(**params)

        paragraphs = ITranslation.split_into_paragraphs(input_text)
        info("paragraphs:", paragraphs)

        translated_paragraphs = [
            apply_packaged_translation(
                self.pkg,
                paragraph,
                self.translator,
                self.sentencizer,
                num_hypotheses,
            )
            for paragraph in paragraphs
        ]

        info("translated_paragraphs:", translated_paragraphs)

        results = [Hypothesis("", 0) for _ in range(num_hypotheses)]
        for index in range(num_hypotheses):
            for paragraph_hypotheses in translated_paragraphs:
                previous = results[index]
                current = paragraph_hypotheses[index]
                results[index] = Hypothesis(
                    ITranslation.combine_paragraphs([previous.value, current.value]),
                    previous.score + current.score,
                )
            results[index].value = results[index].value.lstrip("\n")

        info("hypotheses_to_return:", results)
        return results


class IdentityTranslation(ITranslation):
    """Translation that returns the original input unchanged."""

    def __init__(self, lang: Language):
        self.from_lang = lang
        self.to_lang = lang

    def hypotheses(self, input_text: str, num_hypotheses: int = 4):
        return [Hypothesis(input_text, 0) for _ in range(num_hypotheses)]


class CompositeTranslation(ITranslation):
    """Translation formed by applying two translations in sequence."""

    t1: ITranslation
    t2: ITranslation
    from_lang: Language
    to_lang: Language

    def __init__(self, t1: ITranslation, t2: ITranslation):
        self.t1 = t1
        self.t2 = t2
        self.from_lang = t1.from_lang
        self.to_lang = t2.to_lang

    def hypotheses(self, input_text: str, num_hypotheses: int = 4) -> list[Hypothesis]:
        first_hypotheses = self.t1.hypotheses(input_text, num_hypotheses)
        combined = []

        for first in first_hypotheses:
            for second in self.t2.hypotheses(first.value, num_hypotheses):
                combined.append(Hypothesis(second.value, first.score + second.score))

        combined.sort(reverse=True)
        return combined[:num_hypotheses]


class FewShotTranslation(ITranslation):
    """Translation performed by a language model using few-shot prompting."""

    def __init__(
        self,
        from_lang: Language,
        to_lang: Language,
        language_model: ILanguageModel,
    ):
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.language_model = language_model

    def hypotheses(self, input_text: str, num_hypotheses: int = 4) -> list[Hypothesis]:
        try:
            translated = fewshot.translate(
                input_text,
                self.from_lang,
                self.to_lang,
                self.language_model,
            )
        except TypeError:
            try:
                translated = fewshot.translate(
                    input_text,
                    self.from_lang.code,
                    self.to_lang.code,
                    self.language_model,
                )
            except TypeError:
                translated = fewshot.translate(
                    self.language_model,
                    input_text,
                    self.from_lang,
                    self.to_lang,
                )
        return [Hypothesis(translated, 0) for _ in range(num_hypotheses)]


class APIsTranslation(ITranslation):
    """Translation delegated to an Argos Translate API."""

    def __init__(self, from_lang: Language, to_lang: Language, api):
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.api = api

    def hypotheses(self, input_text: str, num_hypotheses: int = 4) -> list[Hypothesis]:
        if hasattr(self.api, "translate"):
            try:
                translated = self.api.translate(
                    input_text, self.from_lang.code, self.to_lang.code
                )
            except TypeError:
                translated = self.api.translate(
                    input_text,
                    from_code=self.from_lang.code,
                    to_code=self.to_lang.code,
                )
        else:
            translated = apis.translate(
                input_text,
                self.from_lang.code,
                self.to_lang.code,
            )
        return [Hypothesis(translated, 0) for _ in range(num_hypotheses)]


def apply_packaged_translation(
    pkg: Package,
    input_text: str,
    translator: Translator,
    sentencizer,
    num_hypotheses: int = 4,
) -> List[Hypothesis]:
    if len(input_text) == 0:
        return [Hypothesis("", 0) for _ in range(num_hypotheses)]

    sentences = sentencizer.split_sentences(input_text)
    info("sentences:", sentences)

    if len(sentences) == 0:
        return [Hypothesis("", 0) for _ in range(num_hypotheses)]

    tokenized_sentences = [pkg.tokenizer.encode(sentence) for sentence in sentences]
    info("tokenized_sentences:", tokenized_sentences)

    translated_batches = translator.translate_batch(
        tokenized_sentences,
        beam_size=num_hypotheses,
        num_hypotheses=num_hypotheses,
        length_penalty=0.2,
        return_scores=True,
    )

    hypotheses = [Hypothesis("", 0) for _ in range(num_hypotheses)]

    for translated_batch in translated_batches:
        for index in range(num_hypotheses):
            translated_text = pkg.tokenizer.decode(translated_batch.hypotheses[index])
            hypotheses[index].value += translated_text
            hypotheses[index].score += translated_batch.scores[index]

    info("translated_sentences:", hypotheses)
    return hypotheses


@functools.lru_cache()
def get_installed_languages() -> List[Language]:
    languages = {}

    for pkg in package.get_installed_packages():
        if getattr(pkg, "type", None) != "translate":
            continue

        if pkg.from_code not in languages:
            languages[pkg.from_code] = Language(pkg.from_code, pkg.from_name)

        if pkg.to_code not in languages:
            languages[pkg.to_code] = Language(pkg.to_code, pkg.to_name)

        from_lang = languages[pkg.from_code]
        to_lang = languages[pkg.to_code]
        translation = PackageTranslation(from_lang, to_lang, pkg)

        from_lang.translations_from.append(translation)
        to_lang.translations_to.append(translation)

    for language in languages.values():
        identity = IdentityTranslation(language)
        language.translations_from.append(identity)
        language.translations_to.append(identity)

    language_list = list(languages.values())

    for from_lang in language_list:
        direct_translations = list(from_lang.translations_from)

        for first_translation in direct_translations:
            intermediate_lang = first_translation.to_lang

            if intermediate_lang.code == from_lang.code:
                continue

            for second_translation in list(intermediate_lang.translations_from):
                if second_translation.to_lang.code == from_lang.code:
                    continue

                target_lang = second_translation.to_lang

                if from_lang.get_translation(target_lang) is not None:
                    continue

                composite = CompositeTranslation(first_translation, second_translation)
                from_lang.translations_from.append(composite)
                target_lang.translations_to.append(composite)

    return language_list


def get_language_from_code(code: str) -> Language:
    return next(
        language
        for language in get_installed_languages()
        if language.code == code
    )


def get_translation_from_codes(from_code: str, to_code: str) -> ITranslation | None:
    from_lang = get_language_from_code(from_code)
    to_lang = get_language_from_code(to_code)
    return from_lang.get_translation(to_lang)


def translate(input_text: str, from_code: str, to_code: str) -> str:
    translation = get_translation_from_codes(from_code, to_code)
    if translation is None:
        raise ValueError(
            "No translation installed for "
            + from_code
            + " -> "
            + to_code
        )
    return translation.translate(input_text)