"""Where a city is: coordinates and time zone from one lookup.

Open-Meteo's geocoder answers a name with a latitude, a longitude, a country and
— the part that matters beyond the weather — the IANA time zone the place keeps.
So «сколько времени в Токио» needs no separate service: the same lookup that
feeds the forecast gives ``Asia/Tokyo``, and :mod:`zoneinfo` turns that into a
reading that is right about daylight saving without another request.

The geocoder is separate from the weather provider because two providers use it
and because it is the half worth caching hardest: a city does not move. Its
answers go into the same :class:`~ayris.actions.system.providers.base.AnswerCache`
as everything else, under a long time to live.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from ayris.actions.system.providers.base import InstantNotFound, InstantProviderError

if TYPE_CHECKING:
    from ayris.actions.system.providers.base import HttpFetcher

__all__ = ["GEOCODE_URL", "Place", "geocode"]

#: Open-Meteo's geocoder. No key, no registration, and it answers Cyrillic names.
GEOCODE_URL: Final = "https://geocoding-api.open-meteo.com/v1/search"


@dataclass(frozen=True, slots=True)
class Place:
    """One place, as much of it as an answer needs."""

    name: str
    latitude: float
    longitude: float
    timezone: str = ""
    country: str = ""
    region: str = ""

    @property
    def spoken(self) -> str:
        """The place as it should be read out.

        The country is added only when it is not Russia: «В Москве» needs no
        qualifier, «В Кембридже, Великобритания» does, because there is one in
        Massachusetts too and the forecast would otherwise be a mystery.
        """
        if not self.country or self.country in _HOME_COUNTRIES:
            return self.name
        return f"{self.name}, {self.country}"

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe mapping, for the cache and for the interface."""
        return {
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
            "country": self.country,
            "region": self.region,
        }


#: Countries whose name is noise in a spoken answer.
_HOME_COUNTRIES: Final[frozenset[str]] = frozenset({"Россия", "Russia"})


def geocode(fetcher: HttpFetcher, city: str, *, language: str = "ru") -> Place:
    """Resolve a city name to coordinates and a time zone.

    Args:
        fetcher: The HTTP client to go through.
        city: The name as the user said it.
        language: Which language to return names in.

    Raises:
        InstantNotFound: No such place.
        InstantProviderError: The geocoder answered something unexpected.
        InstantOffline: The geocoder could not be reached.
    """
    name = " ".join(city.split()).strip()
    if not name:
        raise InstantNotFound("empty city name", user_message="Не поняла, о каком городе речь.")
    payload = fetcher.get_json(
        GEOCODE_URL,
        params={"name": name, "count": 1, "language": language, "format": "json"},
    )
    return parse_place(payload, name)


def parse_place(payload: Any, city: str) -> Place:
    """Read one place out of a geocoder response.

    Separate from :func:`geocode` so the recorded fixtures can be parsed without
    an HTTP client at all — the parsing is where the bugs are, and it is checked
    on every platform.

    Raises:
        InstantNotFound: The response has no results.
        InstantProviderError: A result without coordinates, which no caller can use.
    """
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, list) or not results:
        raise InstantNotFound(
            f"geocoder has no result for {city!r}",
            user_message=f"Не нашла город «{city}».",
        )
    first = results[0]
    if not isinstance(first, dict):
        raise InstantProviderError(f"geocoder returned {type(first).__name__} for {city!r}")
    latitude = _number(first.get("latitude"))
    longitude = _number(first.get("longitude"))
    if latitude is None or longitude is None:
        raise InstantProviderError(
            f"geocoder result for {city!r} has no coordinates",
            user_message="Сервис ответил непонятно.",
        )
    return Place(
        name=_text(first.get("name")) or city,
        latitude=latitude,
        longitude=longitude,
        timezone=_text(first.get("timezone")),
        country=_text(first.get("country")),
        region=_text(first.get("admin1")),
    )


def _number(value: Any) -> float | None:
    """A float from a JSON value, or ``None`` when it is not a number."""
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _text(value: Any) -> str:
    """A stripped string from a JSON value, or ``""``."""
    return value.strip() if isinstance(value, str) else ""
