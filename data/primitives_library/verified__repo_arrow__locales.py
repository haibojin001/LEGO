from __future__ import annotations

from math import trunc
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Type, Union, cast

TimeFrameLiteral = Literal[
    "now", "second", "seconds", "minute", "minutes", "hour", "hours",
    "day", "days", "week", "weeks", "month", "months", "quarter",
    "quarters", "year", "years",
]

_TimeFrameElements = Union[str, Sequence[str], Mapping[str, str], Mapping[str, Sequence[str]]]

_locale_map: Dict[str, Type["Locale"]] = {}


def get_locale(name: str) -> "Locale":
    normalized = name.lower().replace("_", "-")
    cls = _locale_map.get(normalized)
    if cls is None:
        raise ValueError(f"Unsupported locale {normalized!r}.")
    return cls()


def get_locale_by_class_name(name: str) -> "Locale":
    cls = globals().get(name)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, Locale):
        raise ValueError(f"Unsupported locale {name!r}.")
    return cls()


class Locale:
    names: ClassVar[List[str]] = []
    timeframes: ClassVar[Mapping[TimeFrameLiteral, _TimeFrameElements]] = {
        "now": "", "second": "", "seconds": "", "minute": "", "minutes": "",
        "hour": "", "hours": "", "day": "", "days": "", "week": "",
        "weeks": "", "month": "", "months": "", "quarter": "", "quarters": "",
        "year": "", "years": "",
    }
    meridians: ClassVar[Dict[str, str]] = {"am": "", "pm": "", "AM": "", "PM": ""}
    past: ClassVar[str] = "{0}"
    future: ClassVar[str] = "{0}"
    and_word: ClassVar[Optional[str]] = None
    month_names: ClassVar[List[str]] = []
    month_abbreviations: ClassVar[List[str]] = []
    day_names: ClassVar[List[str]] = []
    day_abbreviations: ClassVar[List[str]] = []
    ordinal_day_re: ClassVar[str] = r"(\d+)"
    _month_name_to_ordinal: Optional[Dict[str, int]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in cls.names:
            normalized = name.lower().replace("_", "-")
            if normalized in _locale_map:
                raise LookupError(f"Duplicated locale name: {name}")
            _locale_map[normalized] = cls

    def __init__(self) -> None:
        self._month_name_to_ordinal = None

    def describe(self, timeframe: TimeFrameLiteral, delta: Union[float, int] = 0,
                 only_distance: bool = False) -> str:
        result = self._format_timeframe(timeframe, trunc(delta))
        return result if only_distance else self._format_relative(result, timeframe, delta)

    def describe_multi(self, timeframes: Sequence[Tuple[TimeFrameLiteral, Union[int, float]]],
                       only_distance: bool = False) -> str:
        parts = [self._format_timeframe(frame, trunc(value)) for frame, value in timeframes]
        if self.and_word:
            parts.insert(-1, self.and_word)
        result = " ".join(parts)
        if not only_distance:
            direction = 0
            for _, value in timeframes:
                if trunc(value):
                    direction = trunc(value)
                    break
            result = self._format_relative(result, "seconds", direction)
        return result

    def day_name(self, day: int) -> str:
        return self.day_names[day]

    def day_abbreviation(self, day: int) -> str:
        return self.day_abbreviations[day]

    def month_name(self, month: int) -> str:
        return self.month_names[month]

    def month_abbreviation(self, month: int) -> str:
        return self.month_abbreviations[month]

    def month_number(self, name: str) -> Optional[int]:
        if self._month_name_to_ordinal is None:
            self._month_name_to_ordinal = self._name_to_ordinal(self.month_names)
            self._month_name_to_ordinal.update(self._name_to_ordinal(self.month_abbreviations))
        return self._month_name_to_ordinal.get(name.lower())

    def year_full(self, year: int) -> str:
        return f"{year:04d}"

    def year_abbreviation(self, year: int) -> str:
        return f"{year:04d}"[2:]

    def meridian(self, hour: int, token: Any) -> Optional[str]:
        if token == "a":
            return self.meridians["am"] if hour < 12 else self.meridians["pm"]
        if token == "A":
            return self.meridians["AM"] if hour < 12 else self.meridians["PM"]
        return None

    def ordinal_number(self, n: int) -> str:
        return self._ordinal_number(n)

    def _ordinal_number(self, n: int) -> str:
        return str(n)

    def _name_to_ordinal(self, values: Sequence[str]) -> Dict[str, int]:
        return {value.lower(): index for index, value in enumerate(values[1:], 1)}

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        value = self.timeframes[timeframe]
        amount = trunc(abs(delta))
        if isinstance(value, str):
            return value.format(amount)
        if isinstance(value, Mapping):
            selected = value.get("one" if amount == 1 else "other", next(iter(value.values())))
            if isinstance(selected, str):
                return selected.format(amount)
            value = selected
        if isinstance(value, Sequence):
            if not value:
                return ""
            selected = value[0] if amount == 1 else value[min(1, len(value) - 1)]
            return str(selected).format(amount)
        return cast(str, value).format(amount)

    def _format_relative(self, humanized: str, timeframe: TimeFrameLiteral,
                         delta: Union[float, int]) -> str:
        if timeframe == "now":
            return humanized
        return (self.past if delta < 0 else self.future).format(humanized)


class EnglishLocale(Locale):
    names = ["en", "en-us", "en-gb", "en-au", "en-be", "en-jp", "en-za", "en-ca", "en-ph"]
    past = "{0} ago"
    future = "in {0}"
    and_word = "and"
    timeframes = {
        "now": "just now", "second": "a second", "seconds": "{0} seconds",
        "minute": "a minute", "minutes": "{0} minutes", "hour": "an hour",
        "hours": "{0} hours", "day": "a day", "days": "{0} days",
        "week": "a week", "weeks": "{0} weeks", "month": "a month",
        "months": "{0} months", "quarter": "a quarter", "quarters": "{0} quarters",
        "year": "a year", "years": "{0} years",
    }
    meridians = {"am": "am", "pm": "pm", "AM": "AM", "PM": "PM"}
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month_abbreviations = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    day_names = ["", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_abbreviations = ["", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def _ordinal_number(self, n: int) -> str:
        if 10 < n % 100 < 14:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"


def _frames(now: str, singular: Sequence[str], plural: Sequence[str]) -> Dict[str, str]:
    keys1 = ("second", "minute", "hour", "day", "week", "month", "quarter", "year")
    keys2 = ("seconds", "minutes", "hours", "days", "weeks", "months", "quarters", "years")
    result: Dict[str, str] = {"now": now}
    result.update(dict(zip(keys1, singular)))
    result.update(dict(zip(keys2, plural)))
    return result


class FrenchLocale(EnglishLocale):
    names = ["fr", "fr-fr", "fr-be", "fr-ca", "fr-ch", "fr-lu"]
    past = "il y a {0}"
    future = "dans {0}"
    and_word = "et"
    timeframes = _frames(
        "à l'instant",
        ["une seconde", "une minute", "une heure", "un jour", "une semaine", "un mois", "un trimestre", "un an"],
        ["{0} secondes", "{0} minutes", "{0} heures", "{0} jours", "{0} semaines", "{0} mois", "{0} trimestres", "{0} ans"],
    )


class SpanishLocale(EnglishLocale):
    names = ["es", "es-es", "es-us", "es-mx", "es-do"]
    past = "hace {0}"
    future = "dentro de {0}"
    and_word = "y"
    timeframes = _frames(
        "ahora",
        ["un segundo", "un minuto", "una hora", "un día", "una semana", "un mes", "un trimestre", "un año"],
        ["{0} segundos", "{0} minutos", "{0} horas", "{0} días", "{0} semanas", "{0} meses", "{0} trimestres", "{0} años"],
    )


class ItalianLocale(EnglishLocale):
    names = ["it", "it-it"]
    past = "{0} fa"
    future = "tra {0}"
    and_word = "e"
    timeframes = _frames(
        "proprio ora",
        ["un secondo", "un minuto", "un'ora", "un giorno", "una settimana", "un mese", "un trimestre", "un anno"],
        ["{0} secondi", "{0} minuti", "{0} ore", "{0} giorni", "{0} settimane", "{0} mesi", "{0} trimestri", "{0} anni"],
    )


class GermanLocale(EnglishLocale):
    names = ["de", "de-de"]
    past = "vor {0}"
    future = "in {0}"
    and_word = "und"
    timeframes = _frames(
        "gerade eben",
        ["einer Sekunde", "einer Minute", "einer Stunde", "einem Tag", "einer Woche", "einem Monat", "einem Quartal", "einem Jahr"],
        ["{0} Sekunden", "{0} Minuten", "{0} Stunden", "{0} Tagen", "{0} Wochen", "{0} Monaten", "{0} Quartalen", "{0} Jahren"],
    )


class AustrianGermanLocale(GermanLocale):
    names = ["de-at"]


class SwissGermanLocale(GermanLocale):
    names = ["de-ch"]


class PortugueseLocale(EnglishLocale):
    names = ["pt", "pt-pt"]
    past = "há {0}"
    future = "em {0}"
    and_word = "e"
    timeframes = _frames(
        "agora mesmo",
        ["um segundo", "um minuto", "uma hora", "um dia", "uma semana", "um mês", "um trimestre", "um ano"],
        ["{0} segundos", "{0} minutos", "{0} horas", "{0} dias", "{0} semanas", "{0} meses", "{0} trimestres", "{0} anos"],
    )


class BrazilianPortugueseLocale(PortugueseLocale):
    names = ["pt-br"]
    past = "há {0}"
    future = "em {0}"


class DutchLocale(EnglishLocale):
    names = ["nl", "nl-nl", "nl-be"]
    past = "{0} geleden"
    future = "over {0}"
    and_word = "en"


class SwedishLocale(EnglishLocale):
    names = ["sv", "sv-se"]
    past = "för {0} sedan"
    future = "om {0}"
    and_word = "och"


class DanishLocale(EnglishLocale):
    names = ["da", "da-dk"]
    past = "for {0} siden"
    future = "om {0}"
    and_word = "og"


class NorwegianLocale(EnglishLocale):
    names = ["nb", "nb-no", "nn-no", "no"]
    past = "for {0} siden"
    future = "om {0}"
    and_word = "og"


class FinnishLocale(EnglishLocale):
    names = ["fi", "fi-fi"]
    past = "{0} sitten"
    future = "{0} kuluttua"
    and_word = "ja"


class RussianLocale(EnglishLocale):
    names = ["ru", "ru-ru"]
    past = "{0} назад"
    future = "через {0}"
    and_word = "и"

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        forms = {
            "second": ("секунду", "секунды", "секунд"), "seconds": ("секунду", "секунды", "секунд"),
            "minute": ("минуту", "минуты", "минут"), "minutes": ("минуту", "минуты", "минут"),
            "hour": ("час", "часа", "часов"), "hours": ("час", "часа", "часов"),
            "day": ("день", "дня", "дней"), "days": ("день", "дня", "дней"),
            "week": ("неделю", "недели", "недель"), "weeks": ("неделю", "недели", "недель"),
            "month": ("месяц", "месяца", "месяцев"), "months": ("месяц", "месяца", "месяцев"),
            "quarter": ("квартал", "квартала", "кварталов"), "quarters": ("квартал", "квартала", "кварталов"),
            "year": ("год", "года", "лет"), "years": ("год", "года", "лет"),
        }
        if timeframe == "now":
            return "только что"
        n = abs(delta)
        forms_for_unit = forms[timeframe]
        if n % 10 == 1 and n % 100 != 11:
            word = forms_for_unit[0]
        elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
            word = forms_for_unit[1]
        else:
            word = forms_for_unit[2]
        return f"{n} {word}"


class UkrainianLocale(RussianLocale):
    names = ["uk", "uk-ua"]
    past = "{0} тому"
    future = "через {0}"


class PolishLocale(EnglishLocale):
    names = ["pl", "pl-pl"]
    past = "{0} temu"
    future = "za {0}"
    and_word = "i"


class CzechLocale(EnglishLocale):
    names = ["cs", "cs-cz"]
    past = "před {0}"
    future = "za {0}"
    and_word = "a"


class SlovakLocale(CzechLocale):
    names = ["sk", "sk-sk"]


class HungarianLocale(EnglishLocale):
    names = ["hu", "hu-hu"]
    past = "{0} ezelőtt"
    future = "{0} múlva"
    and_word = "és"


class RomanianLocale(EnglishLocale):
    names = ["ro", "ro-ro"]
    past = "acum {0}"
    future = "peste {0}"
    and_word = "și"


class TurkishLocale(EnglishLocale):
    names = ["tr", "tr-tr"]
    past = "{0} önce"
    future = "{0} sonra"
    and_word = "ve"


class GreekLocale(EnglishLocale):
    names = ["el", "el-gr"]
    past = "πριν από {0}"
    future = "σε {0}"
    and_word = "και"


class HebrewLocale(EnglishLocale):
    names = ["he", "he-il"]
    past = "לפני {0}"
    future = "בעוד {0}"
    and_word = "ו"


class ArabicLocale(EnglishLocale):
    names = ["ar", "ar-sa", "ar-eg", "ar-ae", "ar-kw", "ar-qa"]
    past = "منذ {0}"
    future = "خلال {0}"
    and_word = "و"


class JapaneseLocale(EnglishLocale):
    names = ["ja", "ja-jp"]
    past = "{0}前"
    future = "{0}後"
    timeframes = _frames(
        "たった今",
        ["1秒", "1分", "1時間", "1日", "1週間", "1か月", "1四半期", "1年"],
        ["{0}秒", "{0}分", "{0}時間", "{0}日", "{0}週間", "{0}か月", "{0}四半期", "{0}年"],
    )


class KoreanLocale(JapaneseLocale):
    names = ["ko", "ko-kr"]
    past = "{0} 전"
    future = "{0} 후"


class ChineseCNLocale(JapaneseLocale):
    names = ["zh", "zh-cn", "zh-hans"]
    past = "{0}前"
    future = "{0}后"


class ChineseTWLocale(JapaneseLocale):
    names = ["zh-tw", "zh-hk", "zh-hant"]
    past = "{0}前"
    future = "{0}後"


class ThaiLocale(JapaneseLocale):
    names = ["th", "th-th"]
    past = "{0}ที่แล้ว"
    future = "ในอีก {0}"


class HindiLocale(EnglishLocale):
    names = ["hi", "hi-in"]
    past = "{0} पहले"
    future = "{0} में"
    and_word = "और"


class BengaliLocale(HindiLocale):
    names = ["bn", "bn-bd", "bn-in"]


class MarathiLocale(HindiLocale):
    names = ["mr", "mr-in"]


class TamilLocale(HindiLocale):
    names = ["ta", "ta-in"]


class MalayalamLocale(HindiLocale):
    names = ["ml", "ml-in"]


class PersianLocale(EnglishLocale):
    names = ["fa", "fa-ir"]
    past = "{0} پیش"
    future = "{0} دیگر"
    and_word = "و"


class VietnameseLocale(EnglishLocale):
    names = ["vi", "vi-vn"]
    past = "{0} trước"
    future = "{0} nữa"
    and_word = "và"


class IndonesianLocale(EnglishLocale):
    names = ["id", "id-id"]
    past = "{0} yang lalu"
    future = "dalam {0}"
    and_word = "dan"


class EsperantoLocale(EnglishLocale):
    names = ["eo", "eo-xx"]
    past = "antaŭ {0}"
    future = "post {0}"
    and_word = "kaj"


class AfrikaansLocale(EnglishLocale):
    names = ["af", "af-za"]
    past = "{0} gelede"
    future = "oor {0}"
    and_word = "en"


class AlbanianLocale(EnglishLocale):
    names = ["sq", "sq-al"]


class ArmenianLocale(EnglishLocale):
    names = ["hy", "hy-am"]


class AzerbaijaniLocale(EnglishLocale):
    names = ["az", "az-az"]


class BasqueLocale(EnglishLocale):
    names = ["eu", "eu-es"]


class BelarusianLocale(EnglishLocale):
    names = ["be", "be-by"]


class BosnianLocale(EnglishLocale):
    names = ["bs", "bs-ba"]


class BulgarianLocale(EnglishLocale):
    names = ["bg", "bg-bg"]


class CatalanLocale(EnglishLocale):
    names = ["ca", "ca-es"]


class CroatianLocale(EnglishLocale):
    names = ["hr", "hr-hr"]


class EstonianLocale(EnglishLocale):
    names = ["et", "et-ee"]


class FilipinoLocale(EnglishLocale):
    names = ["fil", "fil-ph", "tl", "tl-ph"]


class GalicianLocale(EnglishLocale):
    names = ["gl", "gl-es"]


class GeorgianLocale(EnglishLocale):
    names = ["ka", "ka-ge"]


class IcelandicLocale(EnglishLocale):
    names = ["is", "is-is"]


class LatvianLocale(EnglishLocale):
    names = ["lv", "lv-lv"]


class LithuanianLocale(EnglishLocale):
    names = ["lt", "lt-lt"]


class LuxembourgishLocale(EnglishLocale):
    names = ["lb", "lb-lu"]


class MaoriLocale(EnglishLocale):
    names = ["mi", "mi-nz"]


class SerbianLocale(EnglishLocale):
    names = ["sr", "sr-rs", "sr-latn-rs"]


class SlovenianLocale(EnglishLocale):
    names = ["sl", "sl-si"]


class SwissGermanLocale(GermanLocale):
    names = ["gsw", "gsw-ch"]


__all__ = [
    "TimeFrameLiteral", "_TimeFrameElements", "_locale_map", "get_locale",
    "get_locale_by_class_name", "Locale",
] + [name for name, value in list(globals().items())
     if isinstance(value, type) and issubclass(value, Locale) and value is not Locale]