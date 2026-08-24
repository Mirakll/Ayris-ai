"""Weather, from Open-Meteo.

No key, no registration, and a licence that permits non-commercial use without
one — which is why this is the forecast Ayris ships with. Two calls: the
geocoder resolves the city (:mod:`ayris.actions.system.providers.geocode`), then
the forecast endpoint answers with the current conditions and today's range.

**The parsing and the phrasing are separate on purpose.** :func:`parse_forecast`
turns a JSON body into a :class:`Forecast`, and :func:`forecast_ru` turns a
:class:`Forecast` into the one sentence Ayris says. That split is what lets the
recorded fixtures check both halves on any platform, and it is also what makes
the sentence editable without touching anything that could break the numbers.

The units are chosen for the ear, not for the API: metres per second because
that is what a Russian forecast says, and Celsius rounded to whole degrees
because «плюс двадцать один и три десятых» is not something a person wants to
hear when they asked whether to take a jacket.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.actions.system.providers.base import (
    InstantAnswer,
    InstantProvider,
    InstantProviderError,
)
from ayris.actions.system.providers.geocode import Place, geocode
from ayris.nlu.numbers import plural_form

if TYPE_CHECKING:
    from ayris.actions.system.providers.base import HttpFetcher

__all__ = [
    "FORECAST_URL",
    "WEATHER_CODES_RU",
    "Forecast",
    "WeatherProvider",
    "degrees_ru",
    "forecast_ru",
    "parse_forecast",
]

#: Open-Meteo's forecast endpoint.
FORECAST_URL: Final = "https://api.open-meteo.com/v1/forecast"

#: The fields asked for in ``current``. Kept as one string because that is the
#: shape the API wants and splitting it into a list would only be rejoined.
_CURRENT_FIELDS: Final = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m"
)

#: The fields asked for in ``daily``: today's range, and nothing else.
_DAILY_FIELDS: Final = "temperature_2m_max,temperature_2m_min"

#: WMO weather codes in Russian, as a forecast would read them out.
#:
#: The full table is 28 codes and Open-Meteo returns any of them. Grouping them
#: («слабый дождь» for 61, «дождь» for 63, «сильный дождь» for 65) is how a
#: forecast is actually spoken, and a code with no entry falls back to the
#: temperature alone rather than to a number the user cannot interpret.
WEATHER_CODES_RU: Final[Mapping[int, str]] = {
    0: "ясно",
    1: "почти ясно",
    2: "переменная облачность",
    3: "облачно",
    45: "туман",
    48: "туман с изморозью",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "морось с гололёдом",
    57: "сильная морось с гололёдом",
    61: "слабый дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "дождь с гололёдом",
    67: "сильный дождь с гололёдом",
    71: "слабый снег",
    73: "снег",
    75: "сильный снег",
    77: "снежная крупа",
    80: "небольшой ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с градом",
    99: "сильная гроза с градом",
}


def degrees_ru(value: int) -> str:
    """A temperature as a forecast says it: «плюс 21 градус», «ноль градусов».

    The sign is a word rather than a minus because a synthesiser reads «-3» as
    «дефис три» often enough to matter, and because that is how it is said aloud
    anyway.
    """
    word = plural_form(abs(value), "градус", "градуса", "градусов")
    if value == 0:
        return f"ноль {word}"
    sign = "плюс" if value > 0 else "минус"
    return f"{sign} {abs(value)} {word}"


@dataclass(frozen=True, slots=True)
class Forecast:
    """The current weather at one place, as much as an answer needs."""

    place: Place
    temperature_c: float
    feels_like_c: float | None = None
    humidity_pct: int | None = None
    wind_ms: float | None = None
    code: int | None = None
    high_c: float | None = None
    low_c: float | None = None
    observed_at: str = ""

    @property
    def condition_ru(self) -> str:
        """The sky in words, or ``""`` for a code with no Russian name."""
        return WEATHER_CODES_RU.get(self.code, "") if self.code is not None else ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe mapping, for the cache and the interface."""
        return {
            "place": self.place.as_dict(),
            "temperature_c": round(self.temperature_c, 1),
            "feels_like_c": None if self.feels_like_c is None else round(self.feels_like_c, 1),
            "humidity_pct": self.humidity_pct,
            "wind_ms": None if self.wind_ms is None else round(self.wind_ms, 1),
            "code": self.code,
            "condition_ru": self.condition_ru,
            "high_c": None if self.high_c is None else round(self.high_c, 1),
            "low_c": None if self.low_c is None else round(self.low_c, 1),
            "observed_at": self.observed_at,
        }


def forecast_ru(forecast: Forecast) -> str:
    """One sentence about the weather, for the synthesiser.

    Built out of clauses that each drop out when their field is missing, because
    Open-Meteo answers with what the nearest station has: a place with no wind
    reading should produce a shorter sentence, not «ветер None».

    «Ощущается как» is included only when it differs from the reading by more
    than two degrees — otherwise it is noise in a sentence that has to be short.
    """
    temperature = round(forecast.temperature_c)
    parts = [f"В {_prepositional(forecast.place.spoken)} {degrees_ru(temperature)}"]
    if forecast.condition_ru:
        parts.append(forecast.condition_ru)
    feels_like = forecast.feels_like_c
    if feels_like is not None and abs(feels_like - forecast.temperature_c) > 2:
        parts.append(f"ощущается как {degrees_ru(round(feels_like))}")
    if forecast.wind_ms is not None and forecast.wind_ms >= 4:
        speed = round(forecast.wind_ms)
        word = plural_form(speed, "метр", "метра", "метров")
        parts.append(f"ветер {speed} {word} в секунду")
    head = ", ".join(parts) + "."
    if forecast.high_c is None or forecast.low_c is None:
        return head
    low = degrees_ru(round(forecast.low_c))
    high = degrees_ru(round(forecast.high_c))
    return f"{head} Днём от {low} до {high}."


def _prepositional(city: str) -> str:
    """A city name after «в», declined for the common endings.

    Deliberately shallow. Russian city names decline in ways no table of six
    suffixes will cover, and a full morphology engine is not worth carrying for
    one preposition — but «в Москва» is jarring enough that the four endings
    which cover most of the map earn their place. A name this cannot handle is
    left alone, which is what every navigation system does too.
    """
    if " " in city or "," in city:
        return city
    lowered = city.casefold()
    for ending, replacement in _CASE_ENDINGS:
        if lowered.endswith(ending) and len(city) > len(ending) + 1:
            return city[: -len(ending)] + replacement
    return city


#: Nominative ending to prepositional ending, longest first.
_CASE_ENDINGS: Final[tuple[tuple[str, str], ...]] = (
    ("ква", "кве"),
    ("бург", "бурге"),
    ("град", "граде"),
    ("ия", "ии"),
    ("ая", "ой"),
    ("ка", "ке"),
    ("на", "не"),
    ("ра", "ре"),
    ("да", "де"),
    ("та", "те"),
)


def parse_forecast(payload: Any, place: Place) -> Forecast:
    """Read a forecast out of an Open-Meteo response.

    Raises:
        InstantProviderError: No temperature in the body. Everything else is
            optional — a missing wind reading shortens the sentence, a missing
            temperature means there is no answer to give.
    """
    if not isinstance(payload, Mapping):
        raise InstantProviderError(f"forecast is {type(payload).__name__}, not an object")
    current = payload.get("current")
    current = current if isinstance(current, Mapping) else {}
    temperature = _number(current.get("temperature_2m"))
    if temperature is None:
        raise InstantProviderError(
            "forecast has no temperature",
            user_message="Сервис погоды ответил непонятно.",
        )
    daily = payload.get("daily")
    daily = daily if isinstance(daily, Mapping) else {}
    return Forecast(
        place=place,
        temperature_c=temperature,
        feels_like_c=_number(current.get("apparent_temperature")),
        humidity_pct=_integer(current.get("relative_humidity_2m")),
        wind_ms=_number(current.get("wind_speed_10m")),
        code=_integer(current.get("weather_code")),
        high_c=_first_number(daily.get("temperature_2m_max")),
        low_c=_first_number(daily.get("temperature_2m_min")),
        observed_at=_text(current.get("time")),
    )


def _number(value: Any) -> float | None:
    """A float from a JSON value, or ``None`` when it is not a number."""
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _integer(value: Any) -> int | None:
    """An int from a JSON value, rounding a float, or ``None``."""
    number = _number(value)
    return None if number is None else round(number)


def _first_number(value: Any) -> float | None:
    """The first element of a daily series, or ``None``."""
    if isinstance(value, list) and value:
        return _number(value[0])
    return None


def _text(value: Any) -> str:
    """A stripped string from a JSON value, or ``""``."""
    return value.strip() if isinstance(value, str) else ""


class WeatherProvider(InstantProvider):
    """«какая погода», «погода в Питере» — the current conditions."""

    kind: ClassVar = "weather"
    title_ru: ClassVar = "погоду"
    triggers: ClassVar = (
        "погода",
        "погоду",
        "погоде",
        "погоды",
        "прогноз",
        "температура",
        "жарко",
        "холодно",
    )

    def __init__(self, fetcher: HttpFetcher, *, language: str = "ru") -> None:
        super().__init__(fetcher)
        self._language = language

    def fetch(self, query: str) -> InstantAnswer:
        """The weather in ``query``, which is a city name.

        Raises:
            InstantNotFound: No such city.
            InstantOffline: Neither service could be reached.
            InstantProviderError: A response that cannot be read.
        """
        place = geocode(self.fetcher, query, language=self._language)
        payload = self.fetcher.get_json(
            FORECAST_URL,
            params={
                "latitude": place.latitude,
                "longitude": place.longitude,
                "current": _CURRENT_FIELDS,
                "daily": _DAILY_FIELDS,
                "wind_speed_unit": "ms",
                "timezone": "auto",
                "forecast_days": 1,
            },
        )
        forecast = parse_forecast(payload, place)
        return InstantAnswer(
            kind=self.kind,
            message_ru=forecast_ru(forecast),
            data=forecast.as_dict(),
            source="open-meteo.com",
        )
