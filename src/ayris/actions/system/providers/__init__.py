"""Instant-answer providers: one class per source, one interface between them.

The interface is :class:`~ayris.actions.system.providers.base.InstantProvider`
and its one method, :meth:`fetch`. Everything the calling action does — routing a
phrase, caching the answer, falling back to a stale one when the network is gone
— it does through that method and the ``kind`` a provider declares, so a new
source is a new module here and nothing else. :func:`providers` is the only place
that names them, and it is four lines long.

What ships:

- :class:`~ayris.actions.system.providers.weather.WeatherProvider` — Open-Meteo,
  the forecast for a city.
- :class:`~ayris.actions.system.providers.currency.CurrencyProvider` — the daily
  rates the Central Bank of Russia publishes.
- :class:`~ayris.actions.system.providers.worldtime.WorldTimeProvider` — the
  clock in another city, from the time zone the geocoder returns.
- :class:`~ayris.actions.system.providers.page.FactProvider` — one paragraph from
  Wikipedia's summary API.
- :class:`~ayris.actions.system.providers.page.PageProvider` — the summary any
  other site publishes about a page of its own.
- :class:`~ayris.actions.system.providers.lookup.LookupProvider` — the catch-all:
  a question that matched none of the above is searched for first and answered
  from whatever article it turns out to be about.

Re-exported here so a caller writes one import, and so the split between
:mod:`base` and the sources is an implementation detail rather than something
every user of the package has to know about.
"""

from __future__ import annotations

from ayris.actions.system.providers.base import (
    FALLBACK_KIND,
    OFFLINE_MESSAGE,
    AnswerCache,
    HttpFetcher,
    InstantAnswer,
    InstantNotFound,
    InstantOffline,
    InstantProvider,
    InstantProviderError,
    network_ready,
    provider_names,
    providers,
)
from ayris.actions.system.providers.currency import CurrencyProvider, RateTable
from ayris.actions.system.providers.geocode import Place, geocode
from ayris.actions.system.providers.lookup import LookupProvider
from ayris.actions.system.providers.page import (
    MAX_SENTENCES,
    FactProvider,
    PageProvider,
    PageSummary,
    split_sentences,
)
from ayris.actions.system.providers.weather import Forecast, WeatherProvider
from ayris.actions.system.providers.worldtime import CityTime, WorldTimeProvider

__all__ = [
    "FALLBACK_KIND",
    "MAX_SENTENCES",
    "OFFLINE_MESSAGE",
    "AnswerCache",
    "CityTime",
    "CurrencyProvider",
    "FactProvider",
    "Forecast",
    "HttpFetcher",
    "InstantAnswer",
    "InstantNotFound",
    "InstantOffline",
    "InstantProvider",
    "InstantProviderError",
    "LookupProvider",
    "PageProvider",
    "PageSummary",
    "Place",
    "RateTable",
    "WeatherProvider",
    "WorldTimeProvider",
    "geocode",
    "network_ready",
    "provider_names",
    "providers",
    "split_sentences",
]
