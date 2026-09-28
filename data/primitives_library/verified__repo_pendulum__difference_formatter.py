from __future__ import annotations

import typing as t

from pendulum.locales.locale import Locale

if t.TYPE_CHECKING:
    from pendulum import Duration


DAYS_THRESHOLD_FOR_HALF_WEEK = 3
DAYS_THRESHOLD_FOR_HALF_MONTH = 15
MONTHS_THRESHOLD_FOR_HALF_YEAR = 6

HOURS_IN_NEARLY_A_DAY = 22
DAYS_IN_NEARLY_A_MONTH = 27
MONTHS_IN_NEARLY_A_YEAR = 11

DAYS_OF_WEEK = 7
SECONDS_OF_MINUTE = 60
FEW_SECONDS_MAX = 10

KEY_FUTURE = ".future"
KEY_PAST = ".past"
KEY_AFTER = ".after"
KEY_BEFORE = ".before"


class DifferenceFormatter:
    """
    Produces localized human-readable representations of durations.
    """

    def __init__(self, locale: str = "en") -> None:
        self._locale = Locale.load(locale)

    def format(
        self,
        diff: Duration,
        is_now: bool = True,
        absolute: bool = False,
        locale: str | Locale | None = None,
    ) -> str:
        """
        Format a duration as a localized textual difference.
        """
        active_locale = self._locale if locale is None else Locale.load(locale)

        unit: str
        count: int | float

        if diff.years > 0:
            unit = "year"
            count = diff.years
            if diff.months > MONTHS_THRESHOLD_FOR_HALF_YEAR:
                count += 1
        elif (
            diff.months == MONTHS_IN_NEARLY_A_YEAR
            and diff.weeks * DAYS_OF_WEEK + diff.remaining_days
            > DAYS_THRESHOLD_FOR_HALF_MONTH
        ):
            unit = "year"
            count = 1
        elif diff.months > 0:
            unit = "month"
            count = diff.months
            if (
                diff.weeks * DAYS_OF_WEEK + diff.remaining_days
                >= DAYS_IN_NEARLY_A_MONTH
            ):
                count += 1
        elif diff.weeks > 0:
            unit = "week"
            count = diff.weeks
            if diff.remaining_days > DAYS_THRESHOLD_FOR_HALF_WEEK:
                count += 1
        elif diff.remaining_days > 0:
            unit = "day"
            count = diff.remaining_days
            if diff.hours >= HOURS_IN_NEARLY_A_DAY:
                count += 1
        elif diff.hours > 0:
            unit = "hour"
            count = diff.hours
        elif diff.minutes > 0:
            unit = "minute"
            count = diff.minutes
        elif FEW_SECONDS_MAX < diff.remaining_seconds < SECONDS_OF_MINUTE:
            unit = "second"
            count = diff.remaining_seconds
        else:
            few_seconds = active_locale.get("custom.units.few_second")
            if few_seconds is not None:
                if absolute:
                    return t.cast(str, few_seconds)

                future = diff.invert
                wrapper = "custom"
                if is_now:
                    wrapper += ".from_now" if future else ".ago"
                else:
                    wrapper += KEY_AFTER if future else KEY_BEFORE

                return t.cast(str, active_locale.get(wrapper).format(few_seconds))

            unit = "second"
            count = diff.remaining_seconds

        if count == 0:
            count = 1

        if absolute:
            translation_key = f"translations.units.{unit}"
        else:
            future = diff.invert

            if is_now:
                translation_key = f"translations.relative.{unit}"
                translation_key += KEY_FUTURE if future else KEY_PAST
            else:
                relative_key = "custom.units_relative"
                relative_key += (
                    f".{unit}{KEY_FUTURE}" if future else f".{unit}{KEY_PAST}"
                )

                relative_translation = active_locale.get(relative_key)
                plural_form = active_locale.plural(count)

                if relative_translation:
                    rendered_time = relative_translation[plural_form].format(count)
                else:
                    unit_key = f"translations.units.{unit}.{plural_form}"
                    rendered_time = active_locale.get(unit_key).format(count)

                wrapper = "custom"
                wrapper += KEY_AFTER if future else KEY_BEFORE
                return t.cast(str, active_locale.get(wrapper).format(rendered_time))

        plural_key = active_locale.plural(count)
        return t.cast(
            str,
            active_locale.get(f"{translation_key}.{plural_key}").format(count),
        )