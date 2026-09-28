from __future__ import annotations

import os
from difflib import SequenceMatcher
from typing import List

try:
    import spacy
except ImportError:
    spacy = None

try:
    import stanza
except ImportError:
    stanza = None

from minisbd import SBDetect, models as minisbd_models

from argostranslate import package, settings
from argostranslate.networking import cache_spacy
from argostranslate.package import Package
from argostranslate.utils import info, warning


minisbd_models.cache_dir = str(settings.data_dir / "minisbd")


def get_stanza_processors(lang_code: str, resources: dict) -> str:
    try:
        has_mwt = resources[lang_code].get("mwt")
    except (KeyError, TypeError):
        has_mwt = False
    if has_mwt:
        return "tokenize,mwt"
    return "tokenize"


_cached_spacy_path = None
if settings.chunk_type == settings.ChunkType.SPACY and spacy is not None:
    _cached_spacy_path = cache_spacy()


class ISentenceBoundaryDetectionModel:
    pkg: Package

    def split_sentences(self, text: str) -> List[str]:
        raise NotImplementedError


class SpacySentencizerSmall(ISentenceBoundaryDetectionModel):
    def __init__(self, pkg: Package):
        self.pkg = pkg

        if spacy is None:
            raise RuntimeError(
                "SpaCy is not installed. Install spacy or change ChunkType settings"
            )

        bundled_path = pkg.packaged_sbd_path
        if bundled_path is not None and "spacy" in str(bundled_path):
            model_source = bundled_path
        else:
            if _cached_spacy_path is None:
                raise RuntimeError("SpaCy cache not initialized")
            model_source = _cached_spacy_path

        self.nlp = spacy.load(model_source, exclude=["parser"])
        self.nlp.add_pipe("sentencizer")

    def split_sentences(self, text: str) -> List[str]:
        info(f"Splitting sentences using SBD Model: ({self.pkg.from_code}) {self}")
        parsed = self.nlp(text)
        return [sentence.text for sentence in parsed.sents]

    def __str__(self):
        return "Using Spacy model."


class MiniSBDSentencizer(ISentenceBoundaryDetectionModel):
    LANGUAGE_CODE_MAPPING = {
        "zt": "zh-hant",
        "zh": "zh-hans",
        "pb": "pt",
        "az": "tr",
        "bn": "hi",
        "eo": "en",
        "ms": "en",
        "tl": "en",
    }

    def __init__(self, pkg: Package):
        self.pkg = pkg

        model_dir = pkg.package_path / "minisbd"
        local_model = None

        if model_dir.exists():
            candidates = [
                filename
                for filename in os.listdir(model_dir)
                if filename.endswith(".onnx")
            ]
            if candidates:
                local_model = str(model_dir / candidates[0])

        if local_model is not None:
            selected_language = local_model
        else:
            selected_language = self.LANGUAGE_CODE_MAPPING.get(
                pkg.from_code, pkg.from_code
            )
            if selected_language not in minisbd_models.list_models():
                warning(
                    f"{self.pkg.from_code} is not available in MiniSBD, falling back to en"
                )
                selected_language = "en"

        self.lang = selected_language
        self.detector = None

    def lazy_detector(self):
        if self.detector is None:
            self.detector = SBDetect(
                self.lang,
                use_gpu=settings.device == "cuda",
            )
        return self.detector

    def split_sentences(self, text: str) -> List[str]:
        info(f"Splitting sentences using SBD Model: ({self.pkg.from_code}) {self}")
        return self.lazy_detector().sentences(text)

    def __str__(self):
        return "MiniSBDSentencizer"


class StanzaSentencizer(ISentenceBoundaryDetectionModel):
    LANGUAGE_CODE_MAPPING = {
        "zt": "zh-hant",
        "pb": "pt",
    }

    def __init__(self, pkg: Package):
        self.pkg = pkg
        self.stanza_lang_code = self.LANGUAGE_CODE_MAPPING.get(
            pkg.from_code, pkg.from_code
        )
        self.stanza_pipeline = None

    def lazy_pipeline(self):
        if self.stanza_pipeline is None:
            self.stanza_pipeline = stanza.Pipeline(
                lang=self.stanza_lang_code,
                dir=str(self.pkg.package_path / "stanza"),
                processors="tokenize",
                use_gpu=settings.device == "cuda",
                logging_level="WARNING",
            )
        return self.stanza_pipeline

    def split_sentences(self, text: str) -> List[str]:
        info(f"Splitting sentences using SBD Model: ({self.pkg.from_code}) {self}")
        document = self.lazy_pipeline()(text)
        return [sentence.text for sentence in document.sentences]

    def __str__(self):
        return "StanzaSentencizer"


fewshot_prompt = """<detect-sentence-boundaries> I walked down to the river. Then I went to the
I walked down to the river. <sentence-boundary>
----------
<detect-sentence-boundaries> Argos Translate is machine translation software. It is also
Argos Translate is machine translation software. <sentence-boundary>
----------
<detect-sentence-boundaries> Argos Translate is written in Python and uses OpenAI. It also supports
Argos Translate is written in Python and uses OpenAI. <sentence-boundary>
----------
"""

DETECT_SENTENCE_BOUNDARIES_TOKEN = "<detect-sentence-boundaries>"
SENTENCE_BOUNDARY_TOKEN = "<sentence-boundary>"
FEWSHOT_BOUNDARY_TOKEN = "-" * 10


def get_sbd_package() -> Package | None:
    for installed_package in package.get_installed_packages():
        if installed_package.type == "sbd":
            return installed_package
    return None


def generate_fewshot_sbd_prompt(
    input_text: str, sentence_guess_length: int = 150
) -> str:
    guessed_sentence = input_text[:sentence_guess_length]
    prompt = fewshot_prompt + "<detect-sentence-boundaries> " + guessed_sentence
    info("generate_fewshot_sbd_prompt", prompt)
    return prompt


def parse_fewshot_response(response_text: str) -> str | None:
    sections = response_text.split(FEWSHOT_BOUNDARY_TOKEN)
    info("parse_fewshot_response", sections)
    if len(sections) < 2:
        return None

    lines = sections[-2].split("\n")
    if len(lines) < 2:
        return None
    return lines[-1]


def process_seq2seq_sbd(input_text: str, sbd_translated_guess: str) -> int:
    boundary_position = sbd_translated_guess.find(SENTENCE_BOUNDARY_TOKEN)
    if boundary_position == -1:
        return -1

    detected_text = sbd_translated_guess[:boundary_position]
    info("sbd_translated_guess:", detected_text)

    highest_score = 0.0
    best_position = 0

    for position in range(len(input_text)):
        matcher = SequenceMatcher()
        matcher.set_seqs(input_text[:position], detected_text)
        score = matcher.ratio()

        if position == 0 or score > highest_score:
            best_position = position
            highest_score = score

    return best_position


def detect_sentence(
    input_text: str, sbd_translation, sentence_guess_length: int = 150
) -> int:
    """Given input text, return the index after the end of the first sentence.

    Args:
        input_text: The text to detect the first sentence of.
        sbd_translation: An ITranslation for detecting sentences.
        sentence_guess_length: Estimated number of chars > than most sentences.

    Returns:
        The index of the character after the end of the sentence.
                -1 if not found.
    """
    sentence_guess = input_text[:sentence_guess_length]
    info("sentence_guess:", sentence_guess)

    translated_guess = sbd_translation.translate(
        DETECT_SENTENCE_BOUNDARIES_TOKEN + sentence_guess
    )
    return process_seq2seq_sbd(input_text, translated_guess)