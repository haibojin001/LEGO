from langdetect import DetectorFactory, LangDetectException, detect_langs
from langdetect.lang_detect_exception import ErrorCode
from lexilang.detector import detect as lldetect


DetectorFactory.seed = 0


class Language:
    def __init__(self, code, confidence):
        self.code = code
        self.confidence = float(confidence)

    def __str__(self):
        return f"code: {self.code:<9} confidence: {self.confidence:>5.1f} "


def normalized_lang_code(lang):
    code = lang.lang
    if code == "zh-cn":
        code = "zh"
    elif code == "zh-tw":
        code = "zt"
    return code


def check_lang(langcodes, lang):
    return normalized_lang_code(lang) in langcodes


class Detector:
    def __init__(self, langcodes=()):
        self.langcodes = langcodes

    def detect(self, text):
        if len(text) < 20:
            code, confidence = lldetect(text, self.langcodes)
            if confidence > 0:
                return [Language(code, round(confidence * 100))]

        try:
            choices = [
                language
                for language in detect_langs(text)
                if check_lang(self.langcodes, language)
            ][:3]

            if not choices:
                return [Language("en", 0)]

            if choices[0].prob == 0:
                return [Language("en", 0)]
        except LangDetectException as error:
            if error.code == ErrorCode.CantDetectError:
                return [Language("en", 0)]
            raise error

        return [
            Language(normalized_lang_code(language), round(language.prob * 100))
            for language in choices
        ]