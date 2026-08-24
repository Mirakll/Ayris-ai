"""What every instant-answer provider has in common.

A provider is one class with one method: :meth:`InstantProvider.fetch` takes the
words a person said and returns an :class:`InstantAnswer` — a Russian sentence
short enough to read out loud, plus the same facts in a shape the interface can
show. Adding weather to Ayris meant adding :mod:`ayris.actions.system.providers.weather`
and nothing else; adding the next one is the same amount of work, because the
action that calls them looks up a name in :func:`providers` and never mentions
a provider by class.

**Three things are here rather than in each provider**, because writing them
four times is how they drift:

*The HTTP call.* :class:`HttpFetcher` is the same pattern the cloud STT engines
use — separate connect and read timeouts, an absolute deadline over all attempts,
exponential backoff with jitter, and a retry only on a timeout, a transport error
or a 5xx. A 429 is never retried: it means the opposite of «try again». The
transport is injectable, which is the whole reason the test suite never opens a
socket.

*The cache.* Public APIs without a key have rate limits, and a voice assistant
gets asked «какая погода» four times in an evening. Answers go to one JSON file
per provider under :attr:`~ayris.core.paths.AppPaths.cache_dir`, with the
time-to-live the caller chose. The same file is what makes the offline path
useful: an answer past its TTL is still readable, and :attr:`InstantAnswer.age`
is what lets the action say «данные за …» instead of pretending.

*The offline check.* Asking the network when there is no network wastes the
timeout budget and produces a traceback where a sentence would do. So
:class:`HttpFetcher` refuses before the first attempt when :func:`network_ready`
is false — either :func:`~ayris.core.connectivity.link_up` reports no link, or
the user has turned on offline mode — and the action then reaches for the stale
cache.

Nothing here starts a request on its own. Every fetch is one user command — no
background refresh, no prefetching, no telemetry.
"""

from __future__ import annotations

import json
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any, ClassVar, Final

import httpx

from ayris.core.config import get_settings
from ayris.core.connectivity import link_up
from ayris.core.errors import ActionError
from ayris.core.paths import get_paths
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

__all__ = [
    "BACKOFF_BASE_SEC",
    "BACKOFF_MAX_SEC",
    "CONNECT_TIMEOUT_SEC",
    "FALLBACK_KIND",
    "OFFLINE_MESSAGE",
    "USER_AGENT",
    "AnswerCache",
    "HttpFetcher",
    "InstantAnswer",
    "InstantNotFound",
    "InstantOffline",
    "InstantProvider",
    "InstantProviderError",
    "network_ready",
    "provider_names",
    "providers",
]

_log = get_logger(__name__)

#: How long to wait for a connection. Short: a service that has not answered the
#: handshake in three seconds is down, not slow, and the user is standing there.
CONNECT_TIMEOUT_SEC: Final = 3.0

#: First backoff step, doubled per attempt.
BACKOFF_BASE_SEC: Final = 0.4

#: Ceiling for one backoff step.
BACKOFF_MAX_SEC: Final = 2.0

#: What Ayris calls itself to a public API.
#:
#: Wikipedia's terms ask for a real identifier and answer 403 to the default
#: ``python-httpx/…``; Open-Meteo does not care. One honest string satisfies both,
#: and it deliberately carries no machine or user identifier — zero telemetry
#: applies to what Ayris says about itself as much as to what it sends.
USER_AGENT: Final = "Ayris/1.0 (voice assistant; +https://github.com/Mirakll/Ayris-ai)"

#: What Ayris says when it cannot ask anything at all.
#:
#: One sentence for two situations the user does not distinguish: the machine has
#: no link, and Ayris was told not to use the one it has. Both mean the same thing
#: out loud — «нет подключения к сети» — and both are answered before a socket is
#: opened, so neither costs the timeout budget.
OFFLINE_MESSAGE: Final = "Нет подключения к сети."


def network_ready() -> bool:
    """Whether Ayris is allowed to ask the network, and able to.

    Two conditions, one answer. :func:`~ayris.core.connectivity.link_up` is the
    machine's own view of whether there is a link at all — free, local, and no
    traffic. ``actions.instant.offline`` is the user's: a switch that says «do not
    go out» regardless of what the adapter thinks, for a metered connection, a
    flight, or simply not wanting the assistant to talk to anything.

    Checked in one place so every provider, and the action above them, refuse the
    same way and say the same sentence.
    """
    return not get_settings().actions.instant.offline and link_up()


class InstantProviderError(ActionError):
    """A provider could not answer. Base class for the two real cases."""

    default_user_message = "Не удалось получить ответ."


class InstantOffline(InstantProviderError):
    """There is no network, offline mode is on, or the service never answered."""

    default_user_message = OFFLINE_MESSAGE


class InstantNotFound(InstantProviderError):
    """The service answered, and the answer is «nothing matches»."""

    default_user_message = "Ничего не нашла по этому запросу."


@dataclass(frozen=True, slots=True)
class InstantAnswer:
    """One answer, ready to be read out and to be shown.

    ``message_ru`` is the sentence and the only thing the speech synthesiser
    sees, so it holds no numbers the ear cannot follow and no units spelled as
    symbols. ``data`` is the same answer in full, for the interface — and it is
    also what goes into the cache, which is why every provider keeps it JSON-safe.

    ``fetched_at`` is a Unix timestamp rather than a ``datetime`` because it
    round-trips through JSON without a format to agree on. ``stale`` and
    :attr:`age_sec` are what the action needs to decide between reading the
    answer plainly and reading it with «данные за …».
    """

    kind: str
    message_ru: str
    data: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""
    fetched_at: float = 0.0
    stale: bool = False

    def at(self, moment: float) -> InstantAnswer:
        """The same answer, stamped with when it was fetched."""
        return InstantAnswer(
            kind=self.kind,
            message_ru=self.message_ru,
            data=dict(self.data),
            source=self.source,
            fetched_at=moment,
            stale=self.stale,
        )

    def aged(self, *, now: float) -> InstantAnswer:
        """The same answer marked stale, for the offline path."""
        return InstantAnswer(
            kind=self.kind,
            message_ru=self.message_ru,
            data=dict(self.data),
            source=self.source,
            fetched_at=self.fetched_at or now,
            stale=True,
        )

    def age_sec(self, *, now: float) -> float:
        """How long ago this was fetched, in seconds, never negative."""
        return max(0.0, now - self.fetched_at) if self.fetched_at else 0.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe mapping, for the cache file and for the action's ``data``."""
        return {
            "kind": self.kind,
            "message_ru": self.message_ru,
            "data": dict(self.data),
            "source": self.source,
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InstantAnswer | None:
        """Read one back from the cache, or ``None`` if the entry is unusable.

        Returns ``None`` rather than raising: a cache file is not input the user
        typed, and a corrupted entry means one extra request, not an error worth
        a sentence.
        """
        kind = payload.get("kind")
        message = payload.get("message_ru")
        if not isinstance(kind, str) or not isinstance(message, str) or not message:
            return None
        raw_data = payload.get("data")
        raw_at = payload.get("fetched_at")
        source = payload.get("source")
        return cls(
            kind=kind,
            message_ru=message,
            data=dict(raw_data) if isinstance(raw_data, dict) else {},
            source=source if isinstance(source, str) else "",
            fetched_at=float(raw_at) if isinstance(raw_at, int | float) else 0.0,
        )


class HttpFetcher:
    """One HTTP client with timeouts, retries and a deadline.

    Args:
        timeout_s: Read timeout for a single attempt.
        retries: How many times to try again after a retriable failure.
        transport: Injected in tests, so the suite never reaches the network.

    A provider gets one of these and calls :meth:`get_json` — nothing in a
    provider knows how many attempts it took, and nothing in the action knows
    there was HTTP involved at all.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 6.0,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._timeout_s = max(0.5, timeout_s)
        self._retries = max(0, retries)
        self._transport = transport

    @property
    def deadline_sec(self) -> float:
        """Total budget across every attempt, including the waits between them."""
        return self._timeout_s * float(self._retries + 1) + BACKOFF_MAX_SEC

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, str | int | float] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """GET a URL and parse the body as JSON.

        Raises:
            InstantOffline: No link, no answer, or a 5xx after every attempt.
            InstantProviderError: The service refused, or answered with something
                that is not JSON.
        """
        response = self.get(url, params=params, headers=headers)
        try:
            return response.json()
        except ValueError as exc:
            raise InstantProviderError(
                f"{url}: response is not JSON: {exc}",
                user_message="Сервис ответил непонятно.",
            ) from exc

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str | int | float] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """GET a URL, retrying what is worth retrying.

        Raises:
            InstantOffline: No link, no answer, or a 5xx after every attempt.
            InstantProviderError: The service refused with a 4xx.
        """
        if not network_ready():
            raise InstantOffline(f"{url}: no network", user_message=OFFLINE_MESSAGE)
        started = perf_counter()
        attempt = 0
        last: Exception | None = None
        while True:
            attempt += 1
            try:
                with self._client() as client:
                    response = client.get(
                        url, params=dict(params or {}), headers=dict(headers or {})
                    )
                self._raise_for_status(response, url)
            except (httpx.TimeoutException, httpx.TransportError, InstantOffline) as exc:
                last = exc
                delay = self._backoff(attempt)
                spent = perf_counter() - started
                if attempt > self._retries or (spent + delay) >= self.deadline_sec:
                    break
                _log.warning(
                    "%s: %s, попытка %d через %.1f с",
                    url,
                    type(exc).__name__,
                    attempt,
                    delay,
                )
                sleep(delay)
                continue
            return response
        raise InstantOffline(
            f"{url}: {type(last).__name__} after {attempt} attempt(s): {last}",
            user_message="Сервис не отвечает. Проверьте подключение к интернету.",
        ) from last

    def _client(self) -> httpx.Client:
        """Open a client. Separate method so a subclass can decorate it."""
        return httpx.Client(
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_SEC,
                read=self._timeout_s,
                write=CONNECT_TIMEOUT_SEC,
                pool=CONNECT_TIMEOUT_SEC,
            ),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.5"},
            transport=self._transport,
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response, url: str) -> None:
        """Turn a status code into the right exception, or nothing.

        Raises:
            InstantOffline: 5xx — the service is broken right now, so retry.
            InstantNotFound: 404 — the query matched nothing, so do not.
            InstantProviderError: Any other 4xx, including 429, which must never
                be retried: it is the service saying there have been too many
                requests already.
        """
        code = response.status_code
        if code < 400:
            return
        if code >= 500:
            raise InstantOffline(f"{url}: HTTP {code}", user_message="Сервис временно не отвечает.")
        if code == 404:
            raise InstantNotFound(f"{url}: HTTP 404")
        if code == 429:
            raise InstantProviderError(
                f"{url}: HTTP 429 rate limited",
                user_message="Слишком много запросов к сервису, попробуйте позже.",
            )
        raise InstantProviderError(
            f"{url}: HTTP {code}",
            user_message="Сервис отклонил запрос.",
        )

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Delay before the next attempt: exponential, capped, jittered.

        The jitter is not decoration. Several machines lose the same Wi-Fi at the
        same moment, and a fixed backoff sends all of them back at a free public
        API in the same millisecond, which is how a rate limit is earned.
        """
        capped = min(BACKOFF_BASE_SEC * float(2 ** (attempt - 1)), BACKOFF_MAX_SEC)
        return capped * (0.5 + secrets.randbelow(1000) / 2000.0)


class AnswerCache:
    """Answers on disk, one JSON file, keyed by provider and query.

    A file and not the database because this is throwaway data with a time to
    live: losing it costs one request, and a table would have to be migrated
    forever. It lives in :attr:`~ayris.core.paths.AppPaths.cache_dir`, which the
    profile switcher already treats as expendable.

    Nothing here raises. A cache that cannot be read is a cache miss, and a cache
    that cannot be written is a warning in the log — neither is a reason to fail
    an answer the user is waiting for.
    """

    #: Entries kept per provider. Enough for a household's cities and currencies;
    #: past that the oldest go, so the file cannot grow without bound.
    limit: ClassVar[int] = 64

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else get_paths().cache_dir / "instant.json"

    @property
    def path(self) -> Path:
        """Where the entries are stored."""
        return self._path

    def get(self, key: str, *, ttl_sec: float, now: float | None = None) -> InstantAnswer | None:
        """A fresh answer for ``key``, or ``None``.

        Freshness is the caller's TTL, not the cache's: the same file holds a
        forecast good for ten minutes and an article good for a day.
        """
        moment = time.time() if now is None else now
        answer = self.peek(key)
        if answer is None:
            return None
        return answer if answer.age_sec(now=moment) < ttl_sec else None

    def peek(self, key: str) -> InstantAnswer | None:
        """Whatever is stored for ``key``, however old. The offline path."""
        return InstantAnswer.from_dict(self._read().get(key, {}))

    def put(self, key: str, answer: InstantAnswer) -> None:
        """Store one answer, dropping the oldest entries past :attr:`limit`."""
        entries = self._read()
        entries[key] = answer.as_dict()
        if len(entries) > self.limit:
            ordered = sorted(entries.items(), key=lambda item: _entry_time(item[1]))
            entries = dict(ordered[-self.limit :])
        self._write(entries)

    def clear(self) -> None:
        """Forget everything. Used by the settings «очистить кэш» button."""
        self._write({})

    def _read(self) -> dict[str, Any]:
        """Parse the file, or ``{}`` when it is absent, empty or damaged."""
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            parsed = json.loads(raw)
        except ValueError:
            _log.warning("кэш мгновенных ответов повреждён, начинаю заново: %s", self._path)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _write(self, entries: Mapping[str, Any]) -> None:
        """Replace the file. A failure is logged and otherwise ignored."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_name(f"{self._path.name}.tmp")
            temporary.write_text(
                json.dumps(entries, ensure_ascii=False, indent=1),
                encoding="utf-8",
                newline="\n",
            )
            temporary.replace(self._path)
        except OSError as exc:
            _log.warning("не смогла сохранить кэш мгновенных ответов: %s", exc)


def _entry_time(entry: Any) -> float:
    """Sort key for a cache entry: when it was fetched, ``0`` when unknown."""
    if isinstance(entry, dict):
        stamp = entry.get("fetched_at")
        if isinstance(stamp, int | float):
            return float(stamp)
    return 0.0


class InstantProvider(ABC):
    """One source of instant answers.

    Subclasses declare :attr:`kind` — the word the action matches a request
    against — and implement :meth:`fetch`. Everything a provider needs from the
    outside arrives through the constructor, which is what keeps them testable
    against a recorded fixture and a mock transport.
    """

    #: What this provider answers: ``weather``, ``rates``, ``time``, ``fact``.
    kind: ClassVar[str] = ""

    #: Russian name of the kind, for «не могу узнать <это>» and the settings.
    title_ru: ClassVar[str] = ""

    #: Words that route a request here. Matched against the phrase by the action.
    triggers: ClassVar[tuple[str, ...]] = ()

    #: Whether those words belong to the subject as well as to the routing.
    #:
    #: ``False`` for almost everything: «какая погода в Питере» arrives here
    #: because of «погода», and what the geocoder needs is «питере» alone. The
    #: rates are the exception — «доллар» is both the word that routes the
    #: request and the word that says which rate to quote, so stripping the
    #: triggers there would leave «сколько стоит» and nothing to look up.
    keeps_triggers: ClassVar[bool] = False

    #: Whether this provider wants the question as asked, not a subject cut out of it.
    #:
    #: ``False`` for the providers that look one thing up: «какая погода в питере»
    #: is a city, and the prepositions around it are noise.
    #: :class:`~ayris.actions.system.providers.lookup.LookupProvider` sets it,
    #: because it searches — and «сколько лет москве» with «сколько» removed is a
    #: different question from the one that was asked.
    wants_phrase: ClassVar[bool] = False

    def __init__(self, fetcher: HttpFetcher) -> None:
        self._fetcher = fetcher

    @abstractmethod
    def fetch(self, query: str) -> InstantAnswer:
        """Answer ``query``, going to the network.

        Args:
            query: The subject only — a city, a currency, a topic. The action has
                already stripped «какая погода в» and the like.

        Raises:
            InstantOffline: The service could not be reached.
            InstantNotFound: The service has no answer for this query.
            InstantProviderError: Anything else the service did.
        """

    def cache_key(self, query: str) -> str:
        """Where an answer to ``query`` is stored. Overridable, rarely needed."""
        return f"{self.kind}:{' '.join(query.split()).casefold()}"

    def refresh(self, answer: InstantAnswer) -> InstantAnswer | None:  # noqa: ARG002
        """Rebuild a cached answer locally, or ``None`` when it cannot be.

        The default is ``None``: a forecast from an hour ago cannot be brought up
        to date without asking, and pretending otherwise would be a lie about the
        weather. The clock is the exception —
        :class:`~ayris.actions.system.providers.worldtime.WorldTimeProvider`
        overrides this and recomputes the reading from the cached time zone, which
        is why «сколько времени в Токио» is right offline and never says «данные
        за вчера» about a number it just worked out.
        """
        return None

    @property
    def fetcher(self) -> HttpFetcher:
        """The HTTP client this provider was given."""
        return self._fetcher


def providers(fetcher: HttpFetcher) -> dict[str, InstantProvider]:
    """Every provider that ships with Ayris, keyed by :attr:`InstantProvider.kind`.

    Imported inside the function because the providers import this module: the
    dependency has to point one way at import time and the other way at call
    time, and a function body is where that is spelled out rather than worked
    around.
    """
    from ayris.actions.system.providers.currency import CurrencyProvider
    from ayris.actions.system.providers.lookup import LookupProvider
    from ayris.actions.system.providers.page import FactProvider
    from ayris.actions.system.providers.weather import WeatherProvider
    from ayris.actions.system.providers.worldtime import WorldTimeProvider

    built: Sequence[InstantProvider] = (
        WeatherProvider(fetcher),
        CurrencyProvider(fetcher),
        WorldTimeProvider(fetcher),
        FactProvider(fetcher),
        LookupProvider(fetcher),
    )
    return {provider.kind: provider for provider in built}


def provider_names() -> tuple[str, ...]:
    """The kinds a request may ask for, in the order they are tried."""
    return tuple(_KINDS)


#: Provider kinds in matching order. Weather before facts, because «погода в
#: москве» is a forecast and not an article about a city, and facts last because
#: an encyclopedia has an article about everything. ``lookup`` is not here: it
#: has no triggers and is never matched *by* a word — it is where a phrase goes
#: when it matched nothing else, chosen by the action, not by :func:`detect_kind`.
_KINDS: Final[tuple[str, ...]] = ("weather", "rates", "time", "fact")

#: The catch-all kind, answered by
#: :class:`~ayris.actions.system.providers.lookup.LookupProvider`. Kept as a name
#: so the action and its tests refer to it without spelling the string twice.
FALLBACK_KIND: Final = "lookup"


def iter_kinds() -> Iterator[str]:
    """The provider kinds, in matching order."""
    yield from _KINDS
