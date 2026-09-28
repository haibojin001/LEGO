from functools import lru_cache

from argostranslate import translate

from libretranslate.detect import Detector


__languages = None

aliases = {
    "pb": "pt-BR",
    "zh": "zh-Hans",
    "zt": "zh-Hant",
}

rev_aliases = {value.lower(): key for key, value in aliases.items()}


def iso2model(lang):
    if isinstance(lang, list):
        return [iso2model(item) for item in lang]

    if not isinstance(lang, str):
        return lang

    lang = lang.lower()
    return rev_aliases.get(lang, lang)


def model2iso(lang):
    if isinstance(lang, dict) and "language" in lang:
        result = dict(lang)
        result["language"] = model2iso(result["language"])
        return result
    elif isinstance(lang, list):
        return [model2iso(item) for item in lang]

    lang = lang.lower()
    return aliases.get(lang, lang)


def load_languages():
    global __languages

    if __languages is None or len(__languages) == 0:
        __languages = translate.get_installed_languages()

    return __languages


@lru_cache(maxsize=None)
def load_lang_codes():
    languages = load_languages()
    return tuple(language.code for language in languages)


def get_language_with_fallback(lang_code, languages):
    language = next(
        (language for language in languages if language.code == lang_code),
        None,
    )
    if language is not None:
        return language

    language_variants = {
        "pt": ["pb"],
        "pb": ["pt"],
        "zh": ["zt"],
        "zt": ["zh"],
    }

    fallbacks = language_variants.get(lang_code, [])
    for fallback_code in fallbacks:
        fallback_language = next(
            (language for language in languages if language.code == fallback_code),
            None,
        )
        if fallback_language is not None:
            return fallback_language

    return None


def detect_languages(text):
    if isinstance(text, list):
        is_batch = True
    else:
        is_batch = False
        text = [text]

    lang_codes = load_lang_codes()
    candidates = []

    for item in text:
        try:
            detections = Detector(lang_codes).detect(item)
            for detection in detections:
                detection.text_length = len(item)
            candidates.extend(detections)
        except Exception as error:
            print(str(error))

    text_length_total = sum(candidate.text_length for candidate in candidates)

    if not candidates:
        return [{"confidence": 0.0, "language": "en"}]

    if is_batch:
        temp_average_list = []
        for lang_code in lang_codes:
            matching = list(
                filter(lambda language: language.code == lang_code, candidates)
            )
            if len(matching) > 1:
                language = matching[0]
                language.confidence = sum(
                    candidate.confidence for candidate in matching
                ) / len(matching)
                language.text_length = sum(
                    candidate.text_length for candidate in matching
                )
                temp_average_list.append(language)
            elif matching:
                temp_average_list.append(matching[0])

        if temp_average_list:
            candidates = temp_average_list

    candidates.sort(
        key=lambda language: 0
        if text_length_total == 0
        else (language.confidence * language.text_length) / text_length_total,
        reverse=True,
    )

    return [
        {"confidence": language.confidence, "language": language.code}
        for language in candidates
    ]


def improve_translation_formatting(
    source,
    translation,
    improve_punctuation=True,
    remove_single_word_duplicates=True,
):
    source = source.strip()

    if not len(source):
        return ""

    if not len(translation):
        return source

    if improve_punctuation:
        source_last_char = source[len(source) - 1]
        translation_last_char = translation[len(translation) - 1]

        punctuation_chars = ["!", "?", ".", ",", ";", "。"]
        if source_last_char in punctuation_chars:
            if translation_last_char != source_last_char:
                if translation_last_char in punctuation_chars:
                    translation = translation[:-1]

                translation += source_last_char
        elif translation_last_char in punctuation_chars:
            translation = translation[:-1]

    if remove_single_word_duplicates:
        if len(source) < 20 and source.count(" ") == 0 and translation.count(" ") > 0:
            bag_of_words = translation.split()
            count = {}
            for word in bag_of_words:
                count[word] = count.get(word, 0) + 1

            for word in count:
                if count[word] / len(count) >= 2:
                    translation = bag_of_words[0]
                    break

    if source.islower():
        return translation.lower()

    if source.isupper():
        return translation.upper()

    if len(translation) == 0:
        return source

    if source[0].islower():
        return translation[0].lower() + translation[1:]

    if source[0].isupper():
        return translation[0].upper() + translation[1:]

    return translation