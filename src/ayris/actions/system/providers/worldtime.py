"""What time it is somewhere else.

One network call, not two. Open-Meteo's geocoder returns the IANA time zone of
every place it knows, and :mod:`zoneinfo` turns ``Asia/Tokyo`` into a reading
that is correct about daylight saving without asking anybody — so «сколько
времени в Токио» costs one lookup, and a repeat within the cache window costs
none at all, because the coordinates of a city do not expire the way a forecast
does.

That depends on the IANA database being present, and on Windows it is not:
``zoneinfo`` there raises ``ZoneInfoNotFoundError`` on the first city with an
empty ``TZPATH``. Hence the pinned ``tzdata`` dependency — it is what makes this
module work on the platform Ayris actually runs on.

The answer says the difference from local time as well as the reading, because
that is the half a person is usually after: «в Токио 3 часа ночи, это на 6 часов
больше» answers «можно ли звонить» and a bare clock reading does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import TYPE_CHECKING, Any, ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ayris.actions.system.providers.base import (
    InstantAnswer,
    InstantProvider,
    InstantProviderError,
)
from ayris.actions.system.providers.geocode import Place, geocode
from ayris.nlu.numbers import plural_form

if TYPE_CHECKING:
    from ayris.actions.system.providers.base import HttpFetcher

__all__ = ["CityTime", "WorldTimeProvider", "city_time_ru", "resolve_zone"]


@dataclass(frozen=True, slots=True)
class CityTime:
    """The clock at one place, and how far it is from here."""

    place: Place
    moment: datetime
    local: datetime

    @property
    def offset_hours(self) -> float:
        """Hours ahead of the local clock; negative when behind."""
        there = self.moment.utcoffset()
        here = self.local.utcoffset()
        if there is None or here is None:
            return 0.0
        return (there - here).total_seconds() / 3600.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe mapping, for the cache and the interface."""
        return {
            "place": self.place.as_dict(),
            "timezone": self.place.timezone,
            "iso": self.moment.isoformat(),
            "hour": self.moment.hour,
            "minute": self.moment.minute,
            "offset_hours": self.offset_hours,
        }


def resolve_zone(name: str) -> tzinfo:
    """A time zone by IANA name.

    Raises:
        InstantProviderError: No such zone, or no time-zone database at all —
            which on Windows means the ``tzdata`` package is missing, and saying
            so beats a ``ZoneInfoNotFoundError`` reaching the user as a traceback.
    """
    if not name:
        raise InstantProviderError(
            "place has no timezone",
            user_message="Не знаю, в какой часовой зоне этот город.",
        )
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise InstantProviderError(
            f"unknown timezone {name!r}: {exc}",
            user_message="Не смогла определить часовой пояс.",
        ) from exc


def city_time_ru(reading: CityTime) -> str:
    """One sentence about the time somewhere, for the synthesiser.

    The minutes are read as digits because that is how a clock is read out loud,
    and the difference is spelled out in words because «+6» is a symbol a
    synthesiser has no reliable pronunciation for.
    """
    moment = reading.moment
    clock = f"{moment.hour}:{moment.minute:02d}"
    head = f"В {_prepositional(reading.place.spoken)} {clock}"
    offset = reading.offset_hours
    if abs(offset) < 0.5:
        return f"{head}, столько же, сколько здесь."
    whole = int(abs(offset))
    minutes = round((abs(offset) - whole) * 60)
    span = f"{whole} {plural_form(whole, 'час', 'часа', 'часов')}"
    if minutes:
        span = f"{span} {minutes} {plural_form(minutes, 'минуту', 'минуты', 'минут')}"
    direction = "больше" if offset > 0 else "меньше"
    return f"{head}, это на {span} {direction}, чем здесь."


def _prepositional(city: str) -> str:
    """A city name after «в», for the endings worth declining.

    The same shallow treatment as the forecast uses, and for the same reason: a
    full morphology engine is not worth carrying for one preposition, but «в
    Москва» is wrong often enough to fix the common endings. Kept here rather
    than shared because the two lists will diverge — this one gets city names
    only, that one may get regions.
    """
    if " " in city or "," in city:
        return city
    lowered = city.casefold()
    for ending, replacement in (
        ("ква", "кве"),
        ("бург", "бурге"),
        ("град", "граде"),
        ("ия", "ии"),
        ("ка", "ке"),
        ("на", "не"),
        ("ра", "ре"),
        ("да", "де"),
        ("та", "те"),
    ):
        if lowered.endswith(ending) and len(city) > len(ending) + 1:
            return city[: -len(ending)] + replacement
    return city


class WorldTimeProvider(InstantProvider):
    """«сколько времени в Токио», «который час в Лондоне»."""

    kind: ClassVar = "time"
    title_ru: ClassVar = "время"
    triggers: ClassVar = ("время", "времени", "час", "часов", "который час", "сколько сейчас")

    def __init__(self, fetcher: HttpFetcher, *, language: str = "ru") -> None:
        super().__init__(fetcher)
        self._language = language

    def fetch(self, query: str) -> InstantAnswer:
        """The time in ``query``, which is a city name.

        Raises:
            InstantNotFound: No such city.
            InstantOffline: The geocoder could not be reached.
            InstantProviderError: A city with no usable time zone.
        """
        place = geocode(self.fetcher, query, language=self._language)
        zone = resolve_zone(place.timezone)
        local = datetime.now().astimezone()
        reading = CityTime(place=place, moment=local.astimezone(zone), local=local)
        return InstantAnswer(
            kind=self.kind,
            message_ru=city_time_ru(reading),
            data=reading.as_dict(),
            source="open-meteo.com",
        )

    def refresh(self, answer: InstantAnswer) -> InstantAnswer | None:
        """Recompute the reading from the cached time zone, without the network.

        A time zone is a property of a place and does not go stale, so a cached
        answer holds everything needed to say what time it is now. This is what
        makes the clock work with no connection at all — and it has to be a fresh
        answer rather than a stale one, because reading out yesterday's clock with
        «данные за вчера» attached would be useless where the real answer is free.

        Returns ``None`` when the cached entry predates the time zone being
        stored, so the caller falls back to fetching.
        """
        zone_name = _cached_zone(answer)
        if not zone_name:
            return None
        try:
            zone = resolve_zone(zone_name)
        except InstantProviderError:
            return None
        place = _cached_place(answer, zone_name)
        local = datetime.now().astimezone()
        reading = CityTime(place=place, moment=local.astimezone(zone), local=local)
        return InstantAnswer(
            kind=self.kind,
            message_ru=city_time_ru(reading),
            data=reading.as_dict(),
            source=answer.source,
        )


def _cached_zone(answer: InstantAnswer) -> str:
    """The IANA name stored with a cached reading, or ``""``."""
    name = answer.data.get("timezone")
    return name if isinstance(name, str) else ""


def _cached_place(answer: InstantAnswer, zone_name: str) -> Place:
    """The place stored with a cached reading, as much of it as survived JSON."""
    raw = answer.data.get("place")
    stored = raw if isinstance(raw, dict) else {}
    return Place(
        name=str(stored.get("name") or zone_name.rsplit("/", 1)[-1].replace("_", " ")),
        latitude=float(stored.get("latitude") or 0.0),
        longitude=float(stored.get("longitude") or 0.0),
        timezone=zone_name,
        country=str(stored.get("country") or ""),
        region=str(stored.get("region") or ""),
    )
